"""Job dependency DAG and file-based status checks for scheduled jobs.

Provides simple dependency tracking: each job writes a status file on
completion, and downstream jobs can check whether their upstreams finished.

Status files live in ``data/storage/job_status/{job_name}_{date}.json``.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

STATUS_DIR: Path = DATA_DIR / "job_status"

# ---------------------------------------------------------------------------
# Job dependency DAG
# ---------------------------------------------------------------------------

JOB_DEPS: dict[str, list[str]] = {
    # ---- Data ingestion — no upstream ------------------------------------
    "qlib_data_update": [],
    "spot_cache_warmup": [],
    "global_industry_news": [],
    "sentiment_daily": [],
    "guba_popularity": [],
    "llm_event_pipeline": [],
    # llm_event_retry removed 2026-05-31: 17:30 full-rerun was data-destructive
    # (deleted partial jsonl, re-extracted under same throttling). See commit
    # 8beeab2. llm_retry_queue_drain at 22:30 is the correct compensation.
    "llm_retry_queue_drain": ["llm_event_pipeline"],
    # cx review 2026-06-06 (P1): llm_factor_quality MUST gate on the
    # pipeline. Without this dep an 18:00 quality cron could fire while
    # the 16:30 pipeline was still running, write a "0 events: success"
    # row, and make every downstream freshness gate believe the
    # pipeline was fine.
    "llm_factor_quality": ["llm_event_pipeline"],
    # Phase E.1 PE-1 chain — 15:30 → 15:50 → 16:10 strict sequence.
    # extract gates on the collector (no text → no LLM call), build
    # gates on extract (no events → no factor build).
    "pbc_policy_texts": [],
    "pbc_policy_events": ["pbc_policy_texts"],
    "pbc_liquidity_factors": ["pbc_policy_events"],
    # Phase E.2 PE-2 chain — 15:35 → 15:55 → 16:15 strict sequence.
    # Same gating discipline as PE-1: no text → no LLM call; no events
    # → no factor build. State Council is sparse so a 0-row day is the
    # expected state; the SLA gate uses a 3-day budget to cope.
    "state_council_policy_texts": [],
    "state_council_policy_events": ["state_council_policy_texts"],
    "state_council_policy_factors": ["state_council_policy_events"],
    # Phase E.3 PE-3 chain — 15:40 → 16:00 → 16:20 strict sequence.
    # NBS publishes MONTHLY (CPI/PPI/PMI/社零) so a 0-row weekday is
    # the steady-state expectation; the SLA gate uses a 35-day budget
    # (one monthly release cycle). Same gating discipline as PE-1/PE-2.
    "nbs_policy_texts": [],
    "nbs_policy_events": ["nbs_policy_texts"],
    "nbs_macro_factors": ["nbs_policy_events"],
    # ---- Post-data-update processing -------------------------------------
    "fund_flow_update": ["qlib_data_update"],
    "st_daily_factors_update": ["qlib_data_update"],
    "st_holder_number_update": [],
    "fundamental_update": [],
    "valuation_update": ["qlib_data_update"],
    "quality_update": [],
    "shareholder_update": ["qlib_data_update"],
    "regime_daily_update": ["qlib_data_update"],
    "global_chain_extract": ["global_industry_news"],
    "global_chain_factors": ["global_chain_extract"],
    # Feature cache rebuild reads qlib + holder + flow + valuation +
    # regime into a single parquet.
    # 2026-06-04 cx round 6 P1-7: added valuation_update,
    # regime_daily_update, fetch_shareholder_data, and
    # cross_market_regime as explicit deps. Pre-fix the DAG only
    # required qlib_data_update + fund_flow_update, so a stale
    # valuation / regime / holder source could land in the cache
    # without the gate noticing — silently mixed-vintage features
    # entered training.
    "feature_cache_rebuild": [
        "qlib_data_update",
        "fund_flow_update",
        "st_daily_factors_update",
        "valuation_update",
        "shareholder_update",
        "regime_daily_update",
    ],
    # 2026-06-07 cx P1 #1+#2 fix: daily refresh of the 209-family
    # caches the production champion + shadow candidate read.
    # Gates on qlib + LLM event pipeline so a failed upstream
    # blocks downstream rather than producing a stale "fresh" cache.
    "champion_cache_rebuild": [
        "qlib_data_update",
        "llm_event_pipeline",
    ],
    # ---- Training and inference depend on fresh cache --------------------
    "midweek_train": ["feature_cache_rebuild"],
    "lgb_after_close_smoke": ["feature_cache_rebuild"],
    "weekly_full_retrain": ["feature_cache_rebuild"],
    "predict_crash_daily": ["feature_cache_rebuild"],
    # ---- Shadow optimizer + paper trading depend on smoke test -----------
    "shadow_optimizer": ["lgb_after_close_smoke", "predict_crash_daily"],
    "paper_trading": ["lgb_after_close_smoke"],
    # Shadow overlays (each needs the smoke + their own factor source)
    "shadow_chain_overlay": ["lgb_after_close_smoke", "global_chain_factors"],
    "shadow_klen_overlay": ["lgb_after_close_smoke"],
    "shadow_vol_compression": ["lgb_after_close_smoke"],
    "shadow_roc5_tsmin10": ["lgb_after_close_smoke"],
    # ---- Factor monitoring depends on data -------------------------------
    "factor_decay_monitor": ["lgb_after_close_smoke"],
    "brinson_attribution": ["lgb_after_close_smoke"],
    # ---- Push jobs ------------------------------------------------------
    # Morning needs yesterday's after-close training cache — gated by
    # lgb_after_close_smoke's success the previous evening, not today's.
    # Without per-date "yesterday" support in check_upstream, we leave the
    # morning push ungated and rely on the freshness gate in run_paper_trading
    # and CandidateSanitizer to refuse stale predictions.
    "morning_recommendation": [],
    "sell_check": [],
    "daily_summary": [],
    "evening_outlook": ["qlib_data_update"],
    # ---- Ancillary ------------------------------------------------------
    "risk_check": [],
    "daily_health_check": ["qlib_data_update"],
}


def _status_path(job_name: str, date: str) -> Path:
    return STATUS_DIR / f"{job_name}_{date}.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mark_complete(job_name: str, date: str, success: bool, details: str = "") -> None:
    """Write a status file for *job_name* on *date*."""
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "job": job_name,
        "date": date,
        "success": success,
        "details": details,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = _status_path(job_name, date)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)
    logger.debug("Marked %s on %s as %s", job_name, date, "success" if success else "failed")


def _read_status(job_name: str, date: str) -> dict | None:
    """Read status file; return parsed dict or None if missing / corrupt."""
    path = _status_path(job_name, date)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Corrupt status file %s: %s", path, exc)
        return None


def check_upstream(job_name: str, date: str) -> dict:
    """Check whether all upstream dependencies of *job_name* completed on *date*.

    Returns::

        {
            "ready": True/False,
            "missing": ["job_a", ...],
            "completed": ["job_b", ...],
        }
    """
    deps = JOB_DEPS.get(job_name, [])
    completed: list[str] = []
    missing: list[str] = []
    for dep in deps:
        status = _read_status(dep, date)
        if status is not None and status.get("success"):
            completed.append(dep)
        else:
            missing.append(dep)
    return {
        "ready": len(missing) == 0,
        "missing": missing,
        "completed": completed,
    }


def daily_status(date: str) -> dict:
    """Return a summary of every known job's status for *date*.

    Returns::

        {
            "date": "2026-05-24",
            "jobs": {
                "qlib_data_update": {"status": "success", "completed_at": "..."},
                "fund_flow_update": {"status": "not_run"},
                ...
            }
        }
    """
    jobs: dict[str, dict] = {}
    for job_name in JOB_DEPS:
        status = _read_status(job_name, date)
        if status is None:
            jobs[job_name] = {"status": "not_run"}
        else:
            jobs[job_name] = {
                "status": "success" if status.get("success") else "failed",
                "completed_at": status.get("completed_at", ""),
                "details": status.get("details", ""),
            }
    return {"date": date, "jobs": jobs}
