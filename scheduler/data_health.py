"""Data health tracking — collector status + freshness gate.

Every data collector writes a status file after completion.
Downstream jobs check freshness before using data.

Usage (in collector):
    from scheduler.data_health import write_health, HealthStatus

    write_health("qlib_data_update", HealthStatus(
        success=True, n_items=5173, latest_date="2026-05-25",
    ))

Usage (in downstream):
    from scheduler.data_health import check_freshness, is_fresh

    if not is_fresh("qlib_data_update"):
        logger.warning("Stale data — using yesterday's predictions")
"""
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEALTH_DIR = PROJECT_ROOT / "data" / "storage" / "data_health"


@dataclass
class HealthStatus:
    """Status of a single data collection run."""
    success: bool = False
    n_items: int = 0
    latest_date: str = ""          # latest data date in the fetched result
    error_type: str = ""           # empty if success
    error_message: str = ""
    retry_count: int = 0
    network_profile: str = ""
    coverage: float = 0.0          # fraction of expected items received
    partial: bool = False          # True if only partial data collected


def write_health(
    source: str,
    status: HealthStatus,
    date: str = None,
):
    """Write health status file for a data source.

    Args:
        source: collector name (e.g. "qlib_data_update", "gdelt_ai_server")
        status: HealthStatus with collection results
        date: trading date (default: today)
    """
    date = date or datetime.now().strftime("%Y-%m-%d")
    day_dir = HEALTH_DIR / date
    day_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "source": source,
        "date": date,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        **asdict(status),
    }

    path = day_dir / f"{source}.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    tmp.replace(path)
    logger.info(f"Health written: {source} success={status.success} n={status.n_items}")


def read_health(source: str, date: str = None) -> dict:
    """Read health status for a source on a given date."""
    date = date or datetime.now().strftime("%Y-%m-%d")
    path = HEALTH_DIR / date / f"{source}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def is_fresh(source: str, date: str = None) -> bool:
    """Check if a data source has fresh data for today.

    Returns True if:
    - Health file exists for today
    - success == True
    - partial == False
    """
    h = read_health(source, date)
    if not h:
        return False
    return h.get("success", False) and not h.get("partial", False)


def check_freshness(
    required_sources: list[str],
    date: str = None,
) -> dict:
    """Check freshness of multiple sources.

    Returns:
        {
            "all_fresh": bool,
            "fresh": ["source1", ...],
            "stale": ["source2", ...],
            "missing": ["source3", ...],
        }
    """
    date = date or datetime.now().strftime("%Y-%m-%d")
    fresh = []
    stale = []
    missing = []

    for source in required_sources:
        h = read_health(source, date)
        if not h:
            missing.append(source)
        elif h.get("success") and not h.get("partial"):
            fresh.append(source)
        else:
            stale.append(source)

    return {
        "all_fresh": len(stale) == 0 and len(missing) == 0,
        "fresh": fresh,
        "stale": stale,
        "missing": missing,
        "date": date,
    }


def daily_summary(date: str = None) -> dict:
    """Summarize all health statuses for a date."""
    date = date or datetime.now().strftime("%Y-%m-%d")
    day_dir = HEALTH_DIR / date
    if not day_dir.exists():
        return {"date": date, "sources": {}, "n_success": 0, "n_failed": 0}

    sources = {}
    n_success = 0
    n_failed = 0

    for f in sorted(day_dir.glob("*.json")):
        with open(f) as fh:
            h = json.load(fh)
        source = h.get("source", f.stem)
        sources[source] = {
            "success": h.get("success", False),
            "n_items": h.get("n_items", 0),
            "latest_date": h.get("latest_date", ""),
            "error_type": h.get("error_type", ""),
        }
        if h.get("success"):
            n_success += 1
        else:
            n_failed += 1

    return {
        "date": date,
        "sources": sources,
        "n_success": n_success,
        "n_failed": n_failed,
    }


# --- Predefined source groups for freshness gates ---

# Critical for training/prediction — must be fresh
CRITICAL_SOURCES = [
    "qlib_data_update",
]

# Required for full pipeline — stale = degrade overlay
OVERLAY_SOURCES = [
    "llm_event_pipeline",
    "regime_daily_update",
]

# Nice to have — missing = skip overlay
OPTIONAL_SOURCES = [
    "guba_popularity",
    "fund_flow_update",
    "valuation_update",
]


def check_training_gate(date: str = None) -> dict:
    """Check if it's safe to train/predict today.

    Returns {"gate": "pass"|"fail"|"degrade", "details": ...}
    """
    result = check_freshness(CRITICAL_SOURCES, date)
    if not result["all_fresh"]:
        return {
            "gate": "fail",
            "reason": f"Critical sources stale/missing: {result['stale'] + result['missing']}",
            "details": result,
        }

    overlay_result = check_freshness(OVERLAY_SOURCES, date)
    if not overlay_result["all_fresh"]:
        return {
            "gate": "degrade",
            "reason": f"Overlay sources stale: {overlay_result['stale'] + overlay_result['missing']}",
            "degraded_overlays": overlay_result["stale"] + overlay_result["missing"],
            "details": {**result, "overlay": overlay_result},
        }

    return {"gate": "pass", "details": result}
