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
    # Phase E.4 PE-4 chain — 15:45 → 16:05 → 16:25 strict sequence.
    # Same gating discipline as PE-1/PE-2/PE-3: no text → no LLM call;
    # no events → no factor build. XWLB airs DAILY incl. weekends; the
    # SLA budget is 2 trading days so a single failed weekday scrape
    # can be recovered on Monday without painting the gate red.
    "xinwen_lianbo_policy_texts": [],
    "xinwen_lianbo_policy_events": ["xinwen_lianbo_policy_texts"],
    "xinwen_lianbo_theme_factors": ["xinwen_lianbo_policy_events"],
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
    #
    # 2026-06-07 cx batch D P1 #3: dep set expanded to the union of
    # every upstream the build_feature_cache_242 builder reads through
    # FeatureMerger for the production / shadow profiles. Pre-fix the
    # gate only required qlib + LLM, so a stale fund_flow / valuation /
    # st_daily_factors / shareholder / regime_daily would silently land
    # in the 242 -> 209 -> 209_llm chain (the same class of silent
    # mixed-vintage bug the feature_cache_rebuild dep expansion fixed
    # in cx round 6 P1-7). Mirror the feature_cache_rebuild dep set +
    # add the LLM pipeline.
    "champion_cache_rebuild": [
        "qlib_data_update",
        "fund_flow_update",
        "st_daily_factors_update",
        "valuation_update",
        "shareholder_update",
        "regime_daily_update",
        "llm_event_pipeline",
    ],
    # ---- Training and inference depend on fresh cache --------------------
    # 2026-06-07 cx batch D P1 #4: lgb_after_close_smoke moved off
    # feature_cache_rebuild (legacy 174-family cache) onto
    # champion_cache_rebuild (xgb_209 / xgb_209_llm 209-family caches
    # that smoke + paper actually consume). Pre-fix the smoke gate
    # passed even when champion_cache_rebuild had failed — only the
    # legacy 174 cache needed to be fresh. The 174 cache is now
    # research-only; production smoke/paper read 209-family parquets.
    # midweek_train + weekly_full_retrain + predict_crash_daily stay
    # on feature_cache_rebuild for now: midweek/weekly_train re-train
    # is a research operation and predict_crash uses the smaller 174
    # cache for its head-only crash model. They will migrate when the
    # 174 vs 209 retrain decision (#112) lands.
    "midweek_train": ["feature_cache_rebuild"],
    # cx batch D P1 #4 (commit 24a0122): smoke now gates on
    # champion_cache_rebuild — the 209-family caches that smoke + paper
    # actually consume. Legacy feature_cache_rebuild is research-only.
    "lgb_after_close_smoke": ["champion_cache_rebuild"],
    # cx batch G P1 #3: weekly_full_retrain runs Sat 04:00 against
    # Friday's 18:25 cache — the gate lives in JOB_DEPS_PREV_BDAY so
    # check_upstream_full walks the previous business day's status
    # file. Same-day deps stay empty so daily_status etc still report
    # the job correctly.
    "weekly_full_retrain": [],
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
    # Morning needs yesterday's after-close training cache. SAME-day deps
    # stay empty (nothing to wait on inside the morning window); cross-day
    # deps live in JOB_DEPS_PREV_BDAY below so check_upstream_full can
    # gate on yesterday's after-close success. See cx batch G P1 #2.
    "morning_recommendation": [],
    "sell_check": [],
    "daily_summary": [],
    "evening_outlook": ["qlib_data_update"],
    # ---- Ancillary ------------------------------------------------------
    "risk_check": [],
    "daily_health_check": ["qlib_data_update"],
}


# ---------------------------------------------------------------------------
# Cross-trading-day dependencies (cx batch G P1 #2)
# ---------------------------------------------------------------------------
# Some jobs run before the same-day post-close pipeline has had a chance to
# run, so their real upstream is YESTERDAY's lgb_after_close_smoke /
# champion_cache_rebuild output. The DAG previously could not express this
# — JOB_DEPS only checked today's status file, so morning_recommendation /
# sell_check were ungated. Per check_upstream_full below, a job in this dict
# is gated on `prev_business_day(date)` rather than `date`.
JOB_DEPS_PREV_BDAY: dict[str, list[str]] = {
    "morning_recommendation": [
        "lgb_after_close_smoke",
        "champion_cache_rebuild",
    ],
    "sell_check": [
        "lgb_after_close_smoke",
    ],
    # Saturday 04:00 weekly_full_retrain wants Friday's 18:25 cache —
    # not Saturday's, which never ran.
    "weekly_full_retrain": ["feature_cache_rebuild"],
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


def _prev_business_day(date: str) -> str:
    """Return the previous business day (YYYY-MM-DD) for ``date``.

    Uses pandas BDay arithmetic when available (production case); falls
    back to a Mon→Fri calendar walk if pandas is unavailable or the
    parse fails.
    """
    try:
        import pandas as _pd
        from pandas.tseries.offsets import BDay as _BDay
        return (_pd.Timestamp(date) - _BDay(1)).strftime("%Y-%m-%d")
    except Exception:
        # Fallback: skip weekends by walking back day-by-day until we
        # land on Mon-Fri. Does NOT honour exchange holidays but that
        # is fine for a dep gate — a false-fresh holiday entry only
        # delays an alarm by 1 trading day.
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return date  # malformed input; do not crash callers
        from datetime import timedelta as _td
        for _ in range(7):  # at most 7 calendar days back
            dt = dt - _td(days=1)
            if dt.weekday() < 5:  # 0=Mon..4=Fri
                return dt.strftime("%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")


def check_upstream_full(job_name: str, date: str) -> dict:
    """Check both same-day AND previous-business-day upstreams.

    Returns::

        {
            "ready": True/False,
            "missing": ["same-day deps still missing", ...],
            "completed": ["same-day deps completed", ...],
            "prev_bday_missing": ["prev-bday deps still missing", ...],
            "prev_bday_completed": ["prev-bday deps completed", ...],
            "prev_bday_date": "YYYY-MM-DD",
        }

    ``ready`` is True only when BOTH same-day and previous-bday
    dependency sets are fully satisfied. Callers that need to
    distinguish "today's pipeline not yet done" from "yesterday's
    after-close failed" can inspect the separate fields.
    """
    same_day = check_upstream(job_name, date)
    prev_deps = JOB_DEPS_PREV_BDAY.get(job_name, [])
    prev_bday = _prev_business_day(date) if prev_deps else date
    prev_completed: list[str] = []
    prev_missing: list[str] = []
    for dep in prev_deps:
        status = _read_status(dep, prev_bday)
        if status is not None and status.get("success"):
            prev_completed.append(dep)
        else:
            prev_missing.append(dep)
    return {
        "ready": same_day["ready"] and not prev_missing,
        "missing": same_day["missing"],
        "completed": same_day["completed"],
        "prev_bday_missing": prev_missing,
        "prev_bday_completed": prev_completed,
        "prev_bday_date": prev_bday,
    }


def check_upstream(job_name: str, date: str) -> dict:
    """Check whether all SAME-DAY upstream dependencies of *job_name* completed on *date*.

    Returns::

        {
            "ready": True/False,
            "missing": ["job_a", ...],
            "completed": ["job_b", ...],
        }

    Note: this function only inspects JOB_DEPS (same-day). For jobs
    with cross-trading-day dependencies (e.g. morning_recommendation
    needing yesterday's lgb_after_close_smoke), use
    ``check_upstream_full`` which also walks JOB_DEPS_PREV_BDAY.
    Existing callers that don't care about prev-bday continue to work
    unchanged.
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
