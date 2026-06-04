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


def _expected_latest_trading_date(date: str | None = None) -> str:
    """Return the expected most-recent trading date for ``date`` (or now).

    Uses pandas business-day approximation. cx round 9 P1-5 notes this
    does not cover CN holidays / 调休 correctly — a follow-up should
    consult the Qlib calendar / exchange holiday source. For now this
    matches the existing ``--check-today`` semantics in
    ``scripts/update_qlib_data.py``, so the freshness gate behaves
    consistently with the freshness check that produced the artifact.
    """
    import pandas as _pd
    today = date or datetime.now().strftime("%Y-%m-%d")
    recent_bdays = _pd.bdate_range(end=today, periods=3)
    return str(recent_bdays[-1].date())


def is_fresh(
    source: str,
    date: str | None = None,
    *,
    require_latest_date: str | bool | None = None,
) -> bool:
    """Check if a data source has fresh data for today.

    Returns True if:
    - Health file exists for today
    - success == True
    - partial == False
    - if require_latest_date is provided, the recorded
      ``latest_date`` is >= that target (string ``YYYY-MM-DD``).
      Passing ``True`` substitutes the expected most-recent trading
      date (today or last business day per
      ``_expected_latest_trading_date``).

    2026-06-04 cx round 9 P0-1: pre-fix this only checked
    success/partial. A source could publish ``success=True,
    latest_date=2026-06-03`` on 2026-06-04 and downstream gates would
    treat it as today's data. ``lgb_after_close_smoke`` health and
    paper trading's freshness gate were both affected — that is how
    "all-green status board but stale signals" reached the user.
    """
    h = read_health(source, date)
    if not h:
        return False
    if not (h.get("success", False) and not h.get("partial", False)):
        return False
    if require_latest_date is None:
        return True
    expected = (
        _expected_latest_trading_date(date)
        if require_latest_date is True
        else str(require_latest_date)
    )
    recorded = str(h.get("latest_date") or "")
    if not recorded:
        logger.warning(
            "is_fresh(%s): success=True but latest_date is empty — "
            "treating as stale (expected >= %s)",
            source, expected,
        )
        return False
    return recorded >= expected


def check_freshness(
    required_sources: list[str],
    date: str = None,
    *,
    require_latest_date: str | bool | None = None,
) -> dict:
    """Check freshness of multiple sources.

    Returns:
        {
            "all_fresh": bool,
            "fresh": ["source1", ...],
            "stale": ["source2", ...],
            "missing": ["source3", ...],
        }

    ``require_latest_date`` is forwarded to ``is_fresh`` for every
    source — pass ``True`` to demand each source's recorded
    ``latest_date`` matches the expected most-recent trading date.
    """
    date = date or datetime.now().strftime("%Y-%m-%d")
    expected_for_log = (
        _expected_latest_trading_date(date)
        if require_latest_date is True
        else require_latest_date
    )
    fresh = []
    stale = []
    missing = []

    for source in required_sources:
        h = read_health(source, date)
        if not h:
            missing.append(source)
            continue
        if not (h.get("success") and not h.get("partial")):
            stale.append(source)
            continue
        if require_latest_date is not None:
            expected = (
                _expected_latest_trading_date(date)
                if require_latest_date is True
                else str(require_latest_date)
            )
            recorded = str(h.get("latest_date") or "")
            if not recorded or recorded < expected:
                stale.append(source)
                continue
        fresh.append(source)

    return {
        "all_fresh": len(stale) == 0 and len(missing) == 0,
        "fresh": fresh,
        "stale": stale,
        "missing": missing,
        "date": date,
        "expected_latest_date": expected_for_log,
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

    cx round 9 P0-1: critical sources are now checked against the
    expected latest-trading-date too, not just success=True. A
    "qlib_data_update success=True latest_date=yesterday" record is
    no longer treated as fresh enough to train on.
    """
    result = check_freshness(CRITICAL_SOURCES, date, require_latest_date=True)
    if not result["all_fresh"]:
        return {
            "gate": "fail",
            "reason": (
                f"Critical sources stale/missing: "
                f"{result['stale'] + result['missing']} "
                f"(expected latest_date >= {result.get('expected_latest_date')})"
            ),
            "details": result,
        }

    overlay_result = check_freshness(OVERLAY_SOURCES, date, require_latest_date=True)
    if not overlay_result["all_fresh"]:
        return {
            "gate": "degrade",
            "reason": f"Overlay sources stale: {overlay_result['stale'] + overlay_result['missing']}",
            "degraded_overlays": overlay_result["stale"] + overlay_result["missing"],
            "details": {**result, "overlay": overlay_result},
        }

    return {"gate": "pass", "details": result}
