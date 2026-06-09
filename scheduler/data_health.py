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
    # 2026-06-06 Phase A.6 (A6-3): per-sub-source freshness / ok / n_rows
    # so an aggregate collector (e.g. regime_daily_update with its 5
    # sub-sources) can publish a single row whose sub-source state is
    # still inspectable by the gate without forcing every consumer to
    # crawl 5 separate health files.
    extra: dict = field(default_factory=dict)


def _normalize_iso_date(value) -> str:
    """Coerce a date-like input to canonical ``YYYY-MM-DD`` string.

    2026-06-04 cx round 13 P0-1: collectors have been writing latest_date
    in inconsistent formats — fund_flow_update writes "20260604",
    regime_daily_update writes "2026-06-04", and a few write "".
    Naive string comparison ``"20260603" < "2026-06-04"`` evaluates True
    because the literal characters compare position-by-position and
    ``'0'`` is greater than ``'-'`` at position 4, which would let the
    freshness gate ACCEPT a 3-day-old YYYYMMDD record as newer than
    today's YYYY-MM-DD. Normalize on the way in (write_health) and
    on the way out (is_fresh/check_freshness) so the gate never sees
    a mixed-format pair.
    Returns "" when input cannot be parsed — callers should treat
    "" as "no date recorded".
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    # Already ISO?
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        try:
            datetime.strptime(text, "%Y-%m-%d")
            return text
        except ValueError:
            pass
    # YYYYMMDD?
    if len(text) == 8 and text.isdigit():
        try:
            return datetime.strptime(text, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            return ""
    # Anything pandas can parse (covers YYYY/MM/DD, datetime ISO with time, etc.)
    try:
        import pandas as _pd
        return str(_pd.Timestamp(text).date())
    except Exception:
        return ""


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

    # 2026-06-04 cx round 13 P0-1: normalize at the write boundary so
    # the on-disk record always uses YYYY-MM-DD regardless of what
    # the collector handed us (some pass "20260604", some pass
    # "2026-06-04"; the freshness gate later string-compares these).
    normalized_latest = _normalize_iso_date(status.latest_date)
    status_record = asdict(status)
    status_record["latest_date"] = normalized_latest

    record = {
        "source": source,
        "date": date,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        **status_record,
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
    """Return the expected most-recent CN trading date for ``date`` (or now).

    2026-06-04 cx round 9 P1-5: prefer Qlib's CN calendar over
    pandas bdate_range so 调休 / 临时休市 / public holidays don't
    produce false freshness errors. Falls back to pandas bdate_range
    only when Qlib's calendar isn't available (research environment,
    qlib_data not initialised yet).
    """
    today = date or datetime.now().strftime("%Y-%m-%d")
    # Try Qlib calendar first.
    try:
        from qlib.data import D
        cal = D.calendar(end_time=today)
        if cal is not None and len(cal) > 0:
            last = cal[-1]
            # cal can be a list/ndarray of Timestamps. Coerce.
            import pandas as _pd
            return str(_pd.Timestamp(last).date())
    except Exception:
        pass
    # Fallback: pandas business-day approximation (still wrong on
    # CN-specific holidays, but better than nothing).
    import pandas as _pd
    recent_bdays = _pd.bdate_range(end=today, periods=3)
    return str(recent_bdays[-1].date())


def trading_day_age(
    older_date: str, reference_date: str | None = None,
) -> int | None:
    """Number of CN trading days between ``older_date`` and ``reference_date``.

    2026-06-04 cx round 9 P2-7: risk-control freshness checks
    (crash predictions, supply-chain factors) previously used
    calendar-day age — Mondays after a 3-day weekend got the same
    "stale" verdict as a true 3-trading-day gap. Use this helper to
    count actual trading sessions instead.
    Returns None if either date is unparseable.
    """
    import pandas as _pd
    ref = reference_date or datetime.now().strftime("%Y-%m-%d")
    try:
        older_dt = _pd.Timestamp(older_date[:10])
        ref_dt = _pd.Timestamp(ref[:10])
    except Exception:
        return None
    if older_dt > ref_dt:
        return 0
    # Try Qlib calendar first.
    try:
        from qlib.data import D
        cal = D.calendar(
            start_time=str(older_dt.date()),
            end_time=str(ref_dt.date()),
        )
        if cal is not None and len(cal) > 0:
            # cal includes both endpoints when they are trading days.
            return max(int(len(cal) - 1), 0)
    except Exception:
        pass
    # Fallback: pandas bdate_range (CN holidays not modelled).
    return int(len(_pd.bdate_range(start=older_dt, end=ref_dt)) - 1)


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
    expected_raw = (
        _expected_latest_trading_date(date)
        if require_latest_date is True
        else str(require_latest_date)
    )
    # cx round 13 P0-1: normalize both sides before comparison so
    # mixed "20260604" vs "2026-06-04" formats cannot fool the gate.
    expected = _normalize_iso_date(expected_raw)
    recorded = _normalize_iso_date(h.get("latest_date"))
    if not recorded:
        logger.warning(
            "is_fresh(%s): success=True but latest_date is empty/unparseable — "
            "treating as stale (expected >= %s)",
            source, expected,
        )
        return False
    return recorded >= expected


def is_fresh_sla(
    source: str,
    date: str | None = None,
    *,
    if_unregistered: str = "fail_closed",
) -> bool:
    """SLA-aware freshness check — Phase A.7 entry point.

    Reads ``config.data_sla.SLA_BY_SOURCE`` for ``source`` and applies
    the per-source ``max_age_trading_days`` budget against the recorded
    ``latest_date``. The point of A.7 is to stop modelling
    weekly / quarterly disclosure data with a strict-daily rule (which
    would either keep them permanently red or force the gate to ignore
    them entirely).

    Parameters
    ----------
    source:
        Health source name (e.g. ``fundamental_update``).
    date:
        Reference date for the freshness check (default: today).
    if_unregistered:
        Policy for sources NOT in ``SLA_BY_SOURCE``.
        - ``"fail_closed"`` (default): return ``False`` so an
          unregistered source is treated as stale and surfaces in audit.
        - ``"exempt"``: return ``True`` so legacy paths can opt in
          gradually without breaking.

    Returns ``True`` when:
    - health row exists for ``date``,
    - ``success`` is True and ``partial`` is False,
    - the SLA budget for ``source`` is satisfied: the recorded
      ``latest_date`` is no more than ``max_age_trading_days``
      CN-calendar trading days behind ``date``.
    """
    from config.data_sla import get_sla

    sla = get_sla(source)
    if sla is None:
        if if_unregistered == "exempt":
            logger.warning(
                "is_fresh_sla(%s): source unregistered in SLA_BY_SOURCE, "
                "returning True per if_unregistered=exempt. Add it to "
                "config.data_sla before relying on this gate.",
                source,
            )
            return True
        logger.warning(
            "is_fresh_sla(%s): source unregistered in SLA_BY_SOURCE, "
            "returning False per if_unregistered=fail_closed. Register "
            "the source in config.data_sla so this gate has an honest "
            "answer.",
            source,
        )
        return False

    h = read_health(source, date)
    if not h:
        return False
    if not h.get("success", False):
        return False
    if h.get("partial", False):
        return False

    recorded = _normalize_iso_date(h.get("latest_date"))
    if not recorded:
        logger.warning(
            "is_fresh_sla(%s): success=True but latest_date empty / "
            "unparseable — treating as stale (SLA frequency=%s budget=%d).",
            source, sla.frequency, sla.max_age_trading_days,
        )
        return False

    ref_date = date or datetime.now().strftime("%Y-%m-%d")
    age = trading_day_age(recorded, ref_date)
    if age is None:
        logger.warning(
            "is_fresh_sla(%s): could not compute trading_day_age "
            "(recorded=%s, ref=%s) — treating as stale.",
            source, recorded, ref_date,
        )
        return False
    return age <= sla.max_age_trading_days


def sla_verdict(
    sources: list[str],
    date: str | None = None,
) -> dict:
    """SLA-aware multi-source verdict — Phase A.7 entry point.

    For each source, return one of:
      - ``"fresh"``: registered AND within its SLA budget.
      - ``"stale"``: registered AND outside its SLA budget (or
        success=False / partial=True / latest_date empty).
      - ``"exempt"``: NOT registered in ``SLA_BY_SOURCE``. The caller
        decides whether to block on these or warn.

    Returns::

        {
          "all_fresh": bool,                      # ignores 'exempt'
          "fresh":   ["src1", ...],
          "stale":   ["src2", ...],
          "exempt":  ["src3", ...],
          "details": {"src1": {"age": 0, "budget": 1, "frequency": "daily"}, ...},
        }
    """
    from config.data_sla import get_sla

    ref_date = date or datetime.now().strftime("%Y-%m-%d")
    fresh: list[str] = []
    stale: list[str] = []
    exempt: list[str] = []
    details: dict[str, dict] = {}
    for source in sources:
        sla = get_sla(source)
        if sla is None:
            exempt.append(source)
            details[source] = {"status": "exempt", "reason": "unregistered"}
            continue
        h = read_health(source, date)
        if not h or not h.get("success", False) or h.get("partial", False):
            stale.append(source)
            details[source] = {
                "status": "stale",
                "reason": "missing_or_failed_health",
                "frequency": sla.frequency,
                "budget": sla.max_age_trading_days,
            }
            continue
        recorded = _normalize_iso_date(h.get("latest_date"))
        if not recorded:
            stale.append(source)
            details[source] = {
                "status": "stale",
                "reason": "empty_latest_date",
                "frequency": sla.frequency,
                "budget": sla.max_age_trading_days,
            }
            continue
        age = trading_day_age(recorded, ref_date)
        if age is None or age > sla.max_age_trading_days:
            stale.append(source)
            details[source] = {
                "status": "stale",
                "reason": "exceeds_budget",
                "age_trading_days": age,
                "budget": sla.max_age_trading_days,
                "frequency": sla.frequency,
                "latest_date": recorded,
            }
            continue
        fresh.append(source)
        details[source] = {
            "status": "fresh",
            "age_trading_days": age,
            "budget": sla.max_age_trading_days,
            "frequency": sla.frequency,
            "latest_date": recorded,
        }
    return {
        "all_fresh": len(stale) == 0,
        "fresh": fresh,
        "stale": stale,
        "exempt": exempt,
        "details": details,
    }


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
            expected = _normalize_iso_date(
                _expected_latest_trading_date(date)
                if require_latest_date is True
                else require_latest_date
            )
            recorded = _normalize_iso_date(h.get("latest_date"))
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
    """Summarize all health statuses for a date.

    2026-06-04 cx round 13 P1-4: also reports MISSING CRITICAL sources
    explicitly. Pre-fix this only counted ``n_success`` / ``n_failed``
    over files that existed, so a day where ``qlib_data_update`` never
    ran AT ALL looked the same as a day where it succeeded — the
    summary said "n_success=N n_failed=0" with no signal that a
    critical source was simply absent. The new ``missing_critical``
    and ``overall_status`` fields surface absence as RED.
    """
    date = date or datetime.now().strftime("%Y-%m-%d")
    day_dir = HEALTH_DIR / date

    sources: dict = {}
    n_success = 0
    n_failed = 0

    if day_dir.exists():
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

    # Missing-critical detection (cx round 13 P1-4)
    missing_critical = [s for s in CRITICAL_SOURCES if s not in sources]
    failed_critical = [
        s for s in CRITICAL_SOURCES
        if s in sources and not sources[s].get("success")
    ]
    if missing_critical or failed_critical:
        overall_status = "RED"
    elif n_failed > 0:
        overall_status = "YELLOW"
    else:
        overall_status = "GREEN"

    return {
        "date": date,
        "sources": sources,
        "n_success": n_success,
        "n_failed": n_failed,
        "missing_critical": missing_critical,
        "failed_critical": failed_critical,
        "overall_status": overall_status,
    }


# --- Predefined source groups for freshness gates ---

# 2026-06-04 cx round 9 P1-4: critical now includes the supplementary
# feature data sources the trained champion actually consumes. Pre-fix
# fund_flow_update / valuation_update / regime_daily_update sat in
# OVERLAY or OPTIONAL, so stale snapshots only produced a degrade
# banner — but the trained model has ALREADY ingested those columns
# and inference will use whatever stale value is on disk. The right
# response to stale supplementary inputs is to block training and
# degrade live prediction, not to print a warning.
CRITICAL_SOURCES = [
    "qlib_data_update",
    # cx round 9 P1-4: production supplementary data sources
    "fund_flow_update",
    "valuation_update",
    "regime_daily_update",
]


# 2026-06-04 cx round 13 P1-5: every group listed in
# PRODUCTION_SUPPLEMENTARY_GROUPS must have a corresponding entry
# here. Groups without a health source can produce stale supp
# features without the gate noticing. ``validate_production_feature_coverage``
# below cross-checks at startup.
PRODUCTION_GROUP_TO_HEALTH_SOURCE: dict[str, str] = {
    "fundamental": "fundamental_update",
    "capital_flow": "fund_flow_update",
    "macro_zero_baseline": "qlib_data_update",  # zero-baseline, no fresh source needed
    "shareholder": "shareholder_update",
    "valuation": "valuation_update",
    "northbound": "northbound_update",
    "quality": "quality_update",
    "st_daily_basic": "st_daily_basic_update",
    "st_moneyflow": "st_moneyflow_update",
    "st_holder_number": "st_holder_number_update",
    "cross_market_regime": "regime_daily_update",
    # 2026-06-10 (Phase B.9 promote): global_chain_llm became a
    # production feature group. Without this mapping, training-gate
    # would silently skip it and lgb_after_close_smoke could pass
    # even if global_chain_factors_llm.parquet went stale or missing.
    # Use global_chain_factors_llm as the SLA source (the factor
    # builder publishes that key; rule-based global_chain stays
    # separate so we can independently detect LLM-pipeline stalls
    # without rule-side noise). Marked as the dedicated LLM health
    # source — do NOT alias to global_chain_factors.
    "global_chain_llm": "global_chain_factors_llm",
    # Keep the rule-based mapping too in case xgb_209_chain ever
    # promotes; today it's still shadow but the cross-check below
    # warns on unmapped production groups, so this is harmless.
    "global_chain": "global_chain_factors",
}


def validate_production_feature_coverage() -> tuple[bool, list[str]]:
    """Check that every PRODUCTION_SUPPLEMENTARY_GROUPS entry maps to
    a known health source. Returns (ok, missing_mappings)."""
    from config.production_features import PRODUCTION_SUPPLEMENTARY_GROUPS
    missing = [
        g for g in PRODUCTION_SUPPLEMENTARY_GROUPS
        if g not in PRODUCTION_GROUP_TO_HEALTH_SOURCE
    ]
    return (not missing), missing

# Required for full pipeline — stale = degrade overlay
OVERLAY_SOURCES = [
    "llm_event_pipeline",
]

# Nice to have — missing = skip overlay
OPTIONAL_SOURCES = [
    "guba_popularity",
]


def _resolve_profile_critical_sources() -> list[str]:
    """Derive the critical-source list from the live PRODUCTION_MODEL_PROFILE.

    cx batch D P1 #2 (2026-06-07): pre-fix CRITICAL_SOURCES was a
    hand-maintained list that drifted from the production profile's
    supplementary groups. When the default profile flipped from
    xgb_242 → xgb_209 on 2026-06-06 the hardcoded list was not
    updated, so a stale ``st_daily_basic_update`` (now critical to
    xgb_209) would slip past the gate as "OPTIONAL/OVERLAY-only".

    The right list is: every loader-group the LIVE profile reads,
    mapped through ``PRODUCTION_GROUP_TO_HEALTH_SOURCE``, unioned
    with the hardcoded CRITICAL_SOURCES floor (qlib_data_update +
    legacy supp sources stay critical regardless of profile). Groups
    without a health source (e.g. macro_zero_baseline, which maps to
    qlib_data_update; or research-only groups not in the table) are
    skipped silently — the explicit map is the contract.
    """
    from config.production_features import (
        PRODUCTION_MODEL_PROFILE,
        SUPPLEMENTARY_GROUPS_BY_PROFILE,
    )
    # 2026-06-08 follow-up: the training gate must NOT block on a
    # weekly/quarterly source being "not today". Those cadences are
    # validated by the per-source SLA budget in config/data_sla.py
    # (#157 A7), not by an "expected_latest_date >= today" check that
    # only makes sense for daily sources. Without this filter, the
    # gate fails Mon-Fri because fundamental_update / quality_update /
    # st_holder_number_update only refresh on Saturday — bug found
    # when tonight's rescue smoke kept failing on
    # ['fundamental_update', 'quality_update', 'st_holder_number_update'].
    try:
        from config.data_sla import SLA_BY_SOURCE
    except Exception:  # noqa: BLE001
        SLA_BY_SOURCE = {}
    NON_DAILY_FREQUENCIES = {"weekly", "quarterly"}

    sources: list[str] = list(CRITICAL_SOURCES)  # hard floor
    supp_groups = SUPPLEMENTARY_GROUPS_BY_PROFILE.get(
        PRODUCTION_MODEL_PROFILE, ()
    )
    for group in supp_groups:
        health_source = PRODUCTION_GROUP_TO_HEALTH_SOURCE.get(group)
        if not health_source or health_source in sources:
            continue
        sla = SLA_BY_SOURCE.get(health_source)
        if sla is not None and sla.frequency in NON_DAILY_FREQUENCIES:
            # weekly/quarterly — daily gate skips; SLA gate enforces
            continue
        sources.append(health_source)
    return sources


def check_training_gate(date: str = None) -> dict:
    """Check if it's safe to train/predict today.

    Returns {"gate": "pass"|"fail"|"degrade", "details": ...}

    cx round 9 P0-1: critical sources are now checked against the
    expected latest-trading-date too, not just success=True. A
    "qlib_data_update success=True latest_date=yesterday" record is
    no longer treated as fresh enough to train on.

    cx batch D P1 #2 (2026-06-07): critical list is derived from
    the LIVE PRODUCTION_MODEL_PROFILE in addition to the hardcoded
    CRITICAL_SOURCES floor. See ``_resolve_profile_critical_sources``.
    """
    critical_sources = _resolve_profile_critical_sources()
    result = check_freshness(critical_sources, date, require_latest_date=True)
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
