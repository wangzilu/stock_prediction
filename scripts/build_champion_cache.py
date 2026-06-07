"""Daily refresh of the production champion's feature cache stack.

cx review 2026-06-07 P1 #1 + #2: pre-fix the daily ``feature_cache_rebuild``
cron only refreshed the legacy 174-family cache. The xgb_209 production
champion and the xgb_209_llm shadow candidate read separate parquets
(``feature_cache_209_latest.parquet`` /
``feature_cache_209_llm_latest.parquet``) that had no automation —
shadow_paper_trade.py would have read a stale snapshot for the entire
5-day promotion window.

This script chains the three builders so a single cron call refreshes
the whole stack:

  1. build_feature_cache_242.py   --end <DATE> --out <242-latest>
  2. build_feature_cache_209.py   --input <242-latest> --output <209-latest>
  3. build_feature_cache_209_llm.py --base <209-latest> --out <209-llm-latest>

The output paths use the ``*_latest.parquet`` filenames that
shadow_paper_trade.py already reads, so no consumer-side changes.
The legacy ``*_production.parquet`` aliases stay on disk via the
``--keep-production-alias`` flag so any older script that still
points at them continues to work for one cycle.

Cron: weekday 17:30 after qlib_data_update (17:00) + LLM event
pipeline finishes (16:30 + ~30 min). enforce_deps gates on both.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_242_LATEST = DATA_DIR / "feature_cache_242_latest.parquet"
CACHE_209_LATEST = DATA_DIR / "feature_cache_209_latest.parquet"
CACHE_209_LLM_LATEST = DATA_DIR / "feature_cache_209_llm_latest.parquet"


def _run(cmd: list[str]) -> int:
    """Run subprocess, stream stdout/stderr to logger."""
    logger.info(">> %s", " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return proc.returncode


FRESH_MTIME_WINDOW_SECONDS = 60 * 60  # 60 min


def _write_health_row(
    *, success: bool, end_date: str, elapsed: float,
    step_results: dict[str, bool], error: str = "",
    mtime_check: dict[str, dict] | None = None,
) -> None:
    """Emit a single champion_cache_rebuild health row.

    cx batch D P1 #3 (2026-06-07): the row's ``success`` field is now
    the AND of every sub-step result + an mtime-window check on each
    output parquet. Pre-fix the row was hardcoded success=True and
    only fired on the success path, so a chain that raised SystemExit
    half-way through left NO health record and the SLA gate saw "no
    row written" rather than "chain failed". Worse, when the chain
    DID complete it could not distinguish "ran fresh today" from "ran
    last week and we are reading a stale parquet". Now: write a row
    on EVERY exit (success or failure), with explicit per-step status
    and per-output mtime so downstream gates can see exactly which
    sub-step / output is broken.
    """
    try:
        from scheduler.data_health import HealthStatus, write_health
        extra = {
            "elapsed_seconds": int(elapsed),
            "end_date": end_date,
            "outputs": [
                str(CACHE_242_LATEST),
                str(CACHE_209_LATEST),
                str(CACHE_209_LLM_LATEST),
            ],
            "step_results": step_results,
        }
        if mtime_check is not None:
            extra["mtime_check"] = mtime_check
        if error:
            extra["error"] = error
        write_health("champion_cache_rebuild", HealthStatus(
            success=success,
            n_items=sum(1 for ok in step_results.values() if ok),
            latest_date=end_date,
            error_type="" if success else "champion_cache_chain_failure",
            error_message=error,
            network_profile="none",
            extra=extra,
        ))
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("write_health failed: %s", e)


def _check_output_freshness(
    paths: dict[str, Path], window_seconds: int = FRESH_MTIME_WINDOW_SECONDS,
) -> tuple[bool, dict[str, dict]]:
    """Verify each output parquet exists AND was written within the
    last ``window_seconds``. Returns (all_fresh, per_path_report).
    """
    now = time.time()
    all_fresh = True
    report: dict[str, dict] = {}
    for name, path in paths.items():
        if not path.exists():
            report[name] = {"exists": False, "fresh": False}
            all_fresh = False
            continue
        mtime = path.stat().st_mtime
        age = now - mtime
        fresh = age <= window_seconds
        report[name] = {
            "exists": True,
            "fresh": fresh,
            "mtime_iso": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
            "age_seconds": int(age),
        }
        if not fresh:
            all_fresh = False
    return all_fresh, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--end-date", default=None,
        help="YYYY-MM-DD end date for the 242 cache (default: today).",
    )
    ap.add_argument(
        "--skip-242", action="store_true",
        help="Skip the 242 rebuild — use the existing latest 242 cache.",
    )
    args = ap.parse_args()

    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    py = sys.executable

    t0 = time.time()

    # cx batch D P1 #3: per-step success tracking. We do NOT short-
    # circuit raise on the first failure — that would skip the health
    # write and the SLA gate would see "no row" rather than "failed".
    # Instead capture each step's rc, write a single health row at
    # the end with the AND of step success + mtime freshness check,
    # then raise SystemExit if anything failed.
    step_results: dict[str, bool] = {
        "build_feature_cache_242": False,
        "build_feature_cache_209": False,
        "build_feature_cache_209_llm": False,
    }
    error_chunks: list[str] = []

    # Step 1 — refresh 242 cache to today.
    if not args.skip_242:
        rc = _run([
            py, str(PROJECT_ROOT / "scripts" / "build_feature_cache_242.py"),
            "--end", end_date,
            "--out", str(CACHE_242_LATEST),
        ])
        if rc == 0:
            step_results["build_feature_cache_242"] = True
        else:
            error_chunks.append(f"build_feature_cache_242 rc={rc}")
    else:
        if not CACHE_242_LATEST.exists():
            error_chunks.append(
                f"--skip-242 but {CACHE_242_LATEST} missing"
            )
        else:
            logger.info("Skipping 242 rebuild (using existing %s)", CACHE_242_LATEST)
            # treat skip as a success-equivalent for downstream chaining;
            # the mtime check below will catch a stale skipped 242.
            step_results["build_feature_cache_242"] = True

    # Step 2 — filter to 209 (drop Phase B Bucket A). Only attempt if
    # 242 succeeded; otherwise the input parquet is stale/missing.
    if step_results["build_feature_cache_242"]:
        rc = _run([
            py, str(PROJECT_ROOT / "scripts" / "build_feature_cache_209.py"),
            "--input", str(CACHE_242_LATEST),
            "--output", str(CACHE_209_LATEST),
        ])
        if rc == 0:
            step_results["build_feature_cache_209"] = True
        else:
            error_chunks.append(f"build_feature_cache_209 rc={rc}")
    else:
        error_chunks.append("skipped build_feature_cache_209 (242 failed)")

    # Step 3 — join LLM event factors. Same guard.
    if step_results["build_feature_cache_209"]:
        rc = _run([
            py, str(PROJECT_ROOT / "scripts" / "build_feature_cache_209_llm.py"),
            "--base", str(CACHE_209_LATEST),
            "--out", str(CACHE_209_LLM_LATEST),
        ])
        if rc == 0:
            step_results["build_feature_cache_209_llm"] = True
        else:
            error_chunks.append(f"build_feature_cache_209_llm rc={rc}")
    else:
        error_chunks.append("skipped build_feature_cache_209_llm (209 failed)")

    elapsed = time.time() - t0

    # cx batch D P1 #3: success requires (a) every sub-step succeeded
    # AND (b) every output parquet has mtime within the last 60 min.
    # A stale parquet from a prior successful run is treated as
    # failure because shadow_paper_trade / smoke would otherwise
    # consume yesterday's data thinking it is today's.
    all_steps_ok = all(step_results.values())
    mtime_ok, mtime_report = _check_output_freshness({
        "feature_cache_242_latest": CACHE_242_LATEST,
        "feature_cache_209_latest": CACHE_209_LATEST,
        "feature_cache_209_llm_latest": CACHE_209_LLM_LATEST,
    })
    overall_success = all_steps_ok and mtime_ok
    if not mtime_ok:
        stale = [
            name for name, r in mtime_report.items()
            if not r.get("fresh", False)
        ]
        error_chunks.append(
            f"output mtime stale/missing (>60 min): {stale}"
        )
    error_text = "; ".join(error_chunks)

    if overall_success:
        logger.info(
            "Champion cache chain refreshed in %.1fs. Outputs:\n  %s\n  %s\n  %s",
            elapsed, CACHE_242_LATEST, CACHE_209_LATEST, CACHE_209_LLM_LATEST,
        )
    else:
        logger.error(
            "Champion cache chain FAILED in %.1fs. step_results=%s mtime_report=%s error=%s",
            elapsed, step_results, mtime_report, error_text,
        )

    _write_health_row(
        success=overall_success,
        end_date=end_date,
        elapsed=elapsed,
        step_results=step_results,
        error=error_text,
        mtime_check=mtime_report,
    )

    if not overall_success:
        raise SystemExit(f"champion_cache_rebuild failed: {error_text}")


if __name__ == "__main__":
    main()
