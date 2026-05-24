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
    # Data ingestion — no upstream
    "qlib_data_update": [],
    "spot_cache_warmup": [],
    # Post-data-update processing
    "fund_flow_update": ["qlib_data_update"],
    "valuation_update": ["qlib_data_update"],
    "regime_daily_update": ["qlib_data_update"],
    # Training and inference depend on fresh data
    "midweek_train": ["qlib_data_update"],
    "lgb_after_close_smoke": ["qlib_data_update"],
    "weekly_full_retrain": ["qlib_data_update"],
    # Shadow optimizer + paper trading depend on smoke test
    "shadow_optimizer": ["lgb_after_close_smoke"],
    "paper_trading": ["lgb_after_close_smoke"],
    # Factor monitoring depends on data
    "factor_decay_monitor": ["qlib_data_update"],
    "brinson_attribution": ["qlib_data_update"],
    # Push jobs — morning needs data (previous day), evening needs data
    "morning_recommendation": [],
    "sell_check": [],
    "daily_summary": [],
    "evening_outlook": ["qlib_data_update"],
    # Ancillary
    "risk_check": [],
    "llm_event_pipeline": [],
    "guba_popularity": [],
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
