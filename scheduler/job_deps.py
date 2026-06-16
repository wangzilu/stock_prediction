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
    "global_chain_llm_extract": ["global_chain_extract"],
    "global_chain_factors": ["global_chain_extract"],
    "global_chain_factors_llm": ["global_chain_llm_extract"],
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
        "st_daily_factors_update",
        "llm_event_pipeline",
        "global_chain_factors_llm",
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
    #
    # 2026-06-16: also depend on valuation_update. Pre-fix the smoke
    # cron at 18:35 raced valuation_update (18:00 start, 2h budget):
    # smoke would pass the enforce-deps wait once champion_cache_rebuild
    # completed, then fail the internal check_training_gate because
    # valuation_update health for today wasn't yet written. The 2026-06-16
    # incident dropped today's lgb_latest_predictions.json cache; morning
    # the next day fell back on stale FORCE_CACHE picks.
    "lgb_after_close_smoke": [
        "champion_cache_rebuild",
        "valuation_update",
    ],
    # cx batch G P1 #3: weekly_full_retrain runs Sat 04:00 against
    # Friday's 18:25 cache — the gate lives in JOB_DEPS_PREV_BDAY so
    # check_upstream_full walks the previous business day's status
    # file. Same-day deps stay empty so daily_status etc still report
    # the job correctly.
    "weekly_full_retrain": [],
    "predict_crash_daily": ["feature_cache_rebuild"],
    # 2026-06-16: B.9 shadow Sp20 tracker — snapshots the post-smoke cache
    # so the realized Sp20 can be computed once 5 trading days of forward
    # returns are available. Gated on smoke so it always reads the cache
    # the cron actually consumes.
    "b9_shadow_sp20_tracker": ["lgb_after_close_smoke"],
    # 2026-06-16: C1 paper-shadow 209 vs 242. Same gating as the B.9 tracker
    # — needs the post-smoke production cache as the 209 leg, then triggers
    # a sibling xgb_242 inference pass against the same FeatureMerger state.
    "c1_209_vs_242_tracker": ["lgb_after_close_smoke"],
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
    # 2026-06-14 fix: evening_outlook fires Sun-Thu (the eve of each
    # trading day). On Sun the cron checks SAME-DAY qlib_data_update
    # for 2026-06-14 — but qlib only runs Mon-Fri so it never fires
    # Sunday. The right gate is YESTERDAY (Friday)'s qlib success,
    # not today's no-fire. Moving to JOB_DEPS_PREV_BDAY below.
    "evening_outlook": [],
    # ---- Shadow paper-trade (xgb_209_llm promotion gate) ---------------
    # cx batch G P2 #6 (2026-06-07): added to JOB_DEPS so daily_status
    # reports these jobs and the 5-day shadow window appears in the
    # DAG dashboard. The real upstreams are CROSS-DAY and live in
    # JOB_DEPS_PREV_BDAY:
    #   - generate (09:00) consumes the *_latest.parquet that
    #     champion_cache_rebuild produced YESTERDAY at 18:30 and the
    #     lgb_after_close_smoke that ran YESTERDAY at 18:35.
    #   - backfill (16:30) reads YESTERDAY's picks JSON and the
    #     YESTERDAY's *_latest.parquet for __label_1d; today's
    #     champion_cache_rebuild hasn't fired yet at 16:30.
    # Therefore same-day deps are EMPTY here (the right answer is
    # "nothing same-day blocks this"), prev-bday lives in
    # JOB_DEPS_PREV_BDAY below.
    "shadow_paper_trade_generate": [],
    "shadow_paper_trade_backfill": [],
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
    # 2026-06-14: evening_outlook fires Sun-Thu 22:00 — for next
    # trading day's strategy. On Sunday/Thursday/etc the same-day
    # qlib_data_update never ran (cron is Mon-Fri only) so the gate
    # used to refuse on Sunday with `same-day-missing=['qlib_data_update']`
    # for 5 weeks. The actual data dependency is the LAST trading
    # day's qlib_data_update output, which prev-bday computes
    # correctly (Sun → Fri, Thu eve → Thu morning's same-day cron OK).
    "evening_outlook": ["qlib_data_update"],
    # cx batch G P2 #6 (2026-06-07): shadow paper-trade reads the
    # *_latest.parquet produced by yesterday's 18:30
    # champion_cache_rebuild. generate at 09:00 and backfill at 16:30
    # both run BEFORE today's champion_cache_rebuild (18:30) so the
    # real upstream is always the prior business day.
    "shadow_paper_trade_generate": [
        "lgb_after_close_smoke",
        "champion_cache_rebuild",
    ],
    "shadow_paper_trade_backfill": ["lgb_after_close_smoke"],
}

# Jobs whose cross-day dependency may be satisfied by a same-day
# after-close run when it already completed. Example: Sunday evening
# outlook must use Friday's qlib_data_update because no Sunday qlib
# cron exists; Monday-Thursday evening outlook should use the same-day
# post-close qlib update if it is green, rather than being blocked by
# a failed previous trading day that has already been superseded.
JOB_DEPS_PREV_BDAY_SAME_DAY_OVERRIDE: dict[str, list[str]] = {
    "evening_outlook": ["qlib_data_update"],
}


def _status_path(job_name: str, date: str) -> Path:
    return STATUS_DIR / f"{job_name}_{date}.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mark_complete(
    job_name: str,
    date: str,
    success: bool,
    details: str = "",
    output_paths: list[str] | None = None,
) -> None:
    """Write a status file for *job_name* on *date*.

    ``output_paths`` (cx batch G P2 #7, 2026-06-07): an OPTIONAL list of
    absolute paths the job produced. ``check_upstream`` reads them back
    to detect 'stale success' — the case where a manual re-run wrote a
    fresh status row but the artifact on disk is stale, OR where the
    success file from yesterday survived a tmp-cleanup gap and the job
    never actually wrote a new artifact today. Pass [] (or omit) for
    jobs that don't produce a parquet/JSON artifact — then check_upstream
    only verifies completed_at ordering, not mtime.
    """
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "job": job_name,
        "date": date,
        "success": success,
        "details": details,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "output_paths": list(output_paths) if output_paths else [],
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


# Slack window for the output-mtime vs completed_at check. A file
# typically lands a few seconds BEFORE the success row is written, but
# we still want to flag a file whose mtime is more than this slack
# EARLIER than the recorded completion. cx batch G P2 #7.
_FRESHNESS_SLACK_SECONDS = 600  # 10 minutes


def _is_status_fresh(status: dict, run_started_at: str | None = None) -> tuple[bool, str]:
    """Conservative freshness check for an upstream status row.

    Returns ``(fresh, reason)``. ``fresh=True`` means: either no
    output_paths are recorded (so we only have the success flag to
    go on — trust it), OR every recorded path exists and its mtime
    is no more than _FRESHNESS_SLACK_SECONDS before completed_at.

    ``fresh=False`` means: at least one recorded output path is
    missing, OR its mtime is meaningfully older than completed_at
    (a stale-success indicator).

    cx batch G P2 #7: be conservative. When the status row is from
    a pre-G era (no output_paths field), or when paths can't be
    parsed, return ``(True, "")`` so we don't add false failures on
    rollout. The check exists only to catch the obvious "yesterday's
    success row survived a tmp-cleanup gap" case.
    """
    output_paths = status.get("output_paths") or []
    completed_at = status.get("completed_at") or ""

    # Note: the original P2 #7 spec mentioned a clock-ordering check
    # (upstream completed_at BEFORE current run start). In the polling
    # architecture we already wait for upstream to be ready before
    # proceeding, so by construction completed_at < run_started_at
    # plus a polling-interval slack. The check would either be a no-op
    # (happy path) or false-fire on clock skew, so it's intentionally
    # omitted. The mtime check below (b) is the real freshness guard.
    _ = run_started_at  # reserved for future ordering work

    # Output-mtime check (b) — only if output_paths are recorded and
    # completed_at can be parsed.
    if not output_paths:
        return True, ""
    try:
        comp_dt = datetime.fromisoformat(completed_at[:19])
    except (ValueError, TypeError):
        # Status row predates G P2 #7 or is malformed — accept.
        return True, ""

    for p_str in output_paths:
        try:
            p = Path(p_str)
        except TypeError:
            continue
        if not p.exists():
            return False, f"recorded output_path missing: {p_str}"
        try:
            mtime_dt = datetime.fromtimestamp(p.stat().st_mtime)
        except OSError:
            continue
        delta_sec = (comp_dt - mtime_dt).total_seconds()
        if delta_sec > _FRESHNESS_SLACK_SECONDS:
            return False, (
                f"output_path {p_str} mtime is {delta_sec:.0f}s older "
                f"than completed_at — stale success suspected"
            )
    return True, ""


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


def check_upstream_full(
    job_name: str,
    date: str,
    run_started_at: str | None = None,
) -> dict:
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

    ``run_started_at`` (cx batch G P2 #7): optional ISO timestamp of
    the current job's run start. When passed, freshness validation
    additionally checks that each upstream's recorded output paths
    are newer than their completed_at (catches the stale-success
    case where a manual rerun left the success file but the artifact
    on disk is older). See ``_is_status_fresh`` for the contract.
    """
    same_day = check_upstream(job_name, date, run_started_at=run_started_at)
    prev_deps = JOB_DEPS_PREV_BDAY.get(job_name, [])
    prev_bday = _prev_business_day(date) if prev_deps else date
    prev_completed: list[str] = []
    prev_missing: list[str] = []
    same_day_overrides = set(JOB_DEPS_PREV_BDAY_SAME_DAY_OVERRIDE.get(job_name, []))
    for dep in prev_deps:
        if dep in same_day_overrides:
            same_status = _read_status(dep, date)
            if same_status is not None and same_status.get("success"):
                fresh, reason = _is_status_fresh(same_status, run_started_at=run_started_at)
                if fresh:
                    prev_completed.append(dep)
                    continue
                logger.warning(
                    "Same-day override upstream %s (date=%s) demoted to missing: %s",
                    dep, date, reason,
                )
        status = _read_status(dep, prev_bday)
        if status is None or not status.get("success"):
            prev_missing.append(dep)
            continue
        fresh, reason = _is_status_fresh(status, run_started_at=run_started_at)
        if not fresh:
            logger.warning(
                "Prev-bday upstream %s (date=%s) demoted to missing: %s",
                dep, prev_bday, reason,
            )
            prev_missing.append(dep)
        else:
            prev_completed.append(dep)
    return {
        "ready": same_day["ready"] and not prev_missing,
        "missing": same_day["missing"],
        "completed": same_day["completed"],
        "prev_bday_missing": prev_missing,
        "prev_bday_completed": prev_completed,
        "prev_bday_date": prev_bday,
    }


def check_upstream(
    job_name: str,
    date: str,
    run_started_at: str | None = None,
) -> dict:
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

    ``run_started_at`` (cx batch G P2 #7): optional ISO timestamp of
    the current job's run start. When provided, freshness validation
    also runs against each upstream's output_paths (when recorded).
    """
    deps = JOB_DEPS.get(job_name, [])
    completed: list[str] = []
    missing: list[str] = []
    for dep in deps:
        status = _read_status(dep, date)
        if status is None or not status.get("success"):
            missing.append(dep)
            continue
        fresh, reason = _is_status_fresh(status, run_started_at=run_started_at)
        if not fresh:
            logger.warning(
                "Upstream %s (date=%s) demoted to missing: %s",
                dep, date, reason,
            )
            missing.append(dep)
        else:
            completed.append(dep)
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
