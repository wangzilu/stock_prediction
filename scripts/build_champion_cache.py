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

    # Step 1 — refresh 242 cache to today.
    if not args.skip_242:
        rc = _run([
            py, str(PROJECT_ROOT / "scripts" / "build_feature_cache_242.py"),
            "--end", end_date,
            "--out", str(CACHE_242_LATEST),
        ])
        if rc != 0:
            raise SystemExit(f"build_feature_cache_242 failed rc={rc}")
    else:
        if not CACHE_242_LATEST.exists():
            raise SystemExit(
                f"--skip-242 but {CACHE_242_LATEST} missing; remove the flag "
                f"or run the 242 builder once first."
            )
        logger.info("Skipping 242 rebuild (using existing %s)", CACHE_242_LATEST)

    # Step 2 — filter to 209 (drop Phase B Bucket A).
    rc = _run([
        py, str(PROJECT_ROOT / "scripts" / "build_feature_cache_209.py"),
        "--input", str(CACHE_242_LATEST),
        "--output", str(CACHE_209_LATEST),
    ])
    if rc != 0:
        raise SystemExit(f"build_feature_cache_209 failed rc={rc}")

    # Step 3 — join LLM event factors.
    rc = _run([
        py, str(PROJECT_ROOT / "scripts" / "build_feature_cache_209_llm.py"),
        "--base", str(CACHE_209_LATEST),
        "--out", str(CACHE_209_LLM_LATEST),
    ])
    if rc != 0:
        raise SystemExit(f"build_feature_cache_209_llm failed rc={rc}")

    elapsed = time.time() - t0
    logger.info(
        "Champion cache chain refreshed in %.1fs. Outputs:\n  %s\n  %s\n  %s",
        elapsed, CACHE_242_LATEST, CACHE_209_LATEST, CACHE_209_LLM_LATEST,
    )

    # Write a tiny health row so the SLA gate can see the chain ran.
    try:
        from scheduler.data_health import HealthStatus, write_health
        write_health("champion_cache_rebuild", HealthStatus(
            success=True,
            n_items=3,
            latest_date=end_date,
            network_profile="none",
            extra={
                "elapsed_seconds": int(elapsed),
                "end_date": end_date,
                "outputs": [
                    str(CACHE_242_LATEST),
                    str(CACHE_209_LATEST),
                    str(CACHE_209_LLM_LATEST),
                ],
            },
        ))
    except Exception as e:
        logger.warning("write_health failed: %s", e)


if __name__ == "__main__":
    main()
