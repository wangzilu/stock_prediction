"""Install the stock prediction production crontab.

The generated entries are idempotent and replace older project-specific
entries such as `main.py --run-now`.

Usage:
    python scripts/install_crontab.py --dry-run
    python scripts/install_crontab.py --apply
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = os.environ.get(
    "STOCK_PREDICTION_PYTHON",
    "/Users/wangzilu/miniconda3/envs/tianshou/bin/python",
)
BEGIN_MARKER = "# BEGIN STOCK_PREDICTION_CX"
END_MARKER = "# END STOCK_PREDICTION_CX"


@dataclass(frozen=True)
class CronJob:
    job_id: str
    schedule: str
    target: list[str]
    log_name: str
    network: str = "none"       # domestic/global/none/llm/push
    timeout_sec: int = 0        # 0 = no limit
    critical: bool = False      # True = downstream depends on this
    # When True, run_with_status is invoked with --enforce-deps so cron will
    # short-circuit (exit 75) if any upstream from scheduler.job_deps hasn't
    # successfully completed today. Opt-in per job so the first cycle after
    # rollout doesn't brick jobs whose upstream hasn't written its status
    # file yet.
    enforce_deps: bool = False
    # When enforce_deps is True, max wall-clock seconds the downstream waits
    # for upstreams to complete. Must cover the WORST-case upstream chain
    # completion time from this job's start. Default 1800s (30 min) only
    # works for jobs whose upstreams are guaranteed done by start time.
    dep_wait_seconds: int = 1800


def managed_jobs(python_bin: str = DEFAULT_PYTHON, project_root: Path = PROJECT_ROOT) -> list[CronJob]:
    py = python_bin
    root = str(project_root)
    main_py = str(project_root / "main.py")
    scripts = project_root / "scripts"
    jobs = [
        # --- Market hours: domestic ---
        # morning/sell/risk/evening need Qlib Alpha158 load (~300s) + predict + push
        # Historical: morning took 840s on 5/22. Bumped 900 → 1800 on
        # 2026-06-04 after the inject_supplementary_into_handler fix
        # added ~9 min to live LGB inference (asof_merge across 12
        # parquet sources × 5195 stocks). 1800s = 30 min budget keeps
        # an 18-min headroom over the observed 12-min cold-path total.
        CronJob("morning_recommendation", "20 9 * * 1-5", [py, main_py, "--morning"], "cron_morning.log",
                network="domestic", timeout_sec=1800),
        CronJob("sell_check", "30 14 * * 1-5", [py, main_py, "--sell-check"], "cron_sell_check.log",
                network="domestic", timeout_sec=1800),
        CronJob("daily_summary", "30 15 * * 1-5", [py, main_py, "--daily-summary"], "cron_daily_summary.log",
                network="domestic", timeout_sec=600),
        CronJob("risk_check", "35 9-15 * * 1-5", [py, main_py, "--risk-check"], "cron_risk_check.log",
                network="domestic", timeout_sec=900),
        # evening_outlook generates the NEXT trading day's strategy /
        # outlook from current evening's data. Therefore it must fire on
        # the evening BEFORE each trading day, not on trading-day evenings:
        #   Sun 22:00 → Mon market  ← was missing under old "1-5" cron
        #   Mon 22:00 → Tue market
        #   Tue 22:00 → Wed market
        #   Wed 22:00 → Thu market
        #   Thu 22:00 → Fri market
        #   Fri / Sat → none needed (Sat+Sun closed)
        # Old schedule "0 22 * * 1-5" (Mon-Fri) missed Sun→Mon, causing
        # Monday's 9:20 morning_recommendation to read a Friday-22:00
        # outlook stale by 60+ hours (user-reported bug 2026-05-31).
        # Fix: 0-4 in cron = Sun-Thu.
        # 2026-06-04 cx round 15 P1-2: enforce DAG deps so 22:00
        # cannot ship "下一交易日策略" against a failed 17:00
        # qlib_data_update or a failed 18:35 lgb_after_close_smoke.
        # Pre-fix the dependency was declared in job_deps.py but the
        # cron entry did not pass enforce_deps=True, so the DAG was
        # decorative.
        CronJob("evening_outlook", "0 22 * * 0-4", [py, main_py, "--evening-outlook"], "cron_evening_outlook.log",
                network="domestic", timeout_sec=900,
                enforce_deps=True, dep_wait_seconds=900),
        # --- Post-close: LLM / event collection ---
    ]
    if (scripts / "collect_global_industry_news.py").exists():
        # ShadowsocksX provides HTTP proxy on port 10818 via bridge.
        # network=global sets http_proxy env var for the subprocess.
        # 7 topics × (GDELT + RSS) can take 10+ min. GDELT is slow/rate-limited.
        jobs.append(CronJob("global_industry_news", "25 16 * * 1-5",
                [py, str(scripts / "collect_global_industry_news.py")], "global_industry_news.log",
                network="global", timeout_sec=1200))
    if (scripts / "extract_global_supply_chain_events.py").exists():
        jobs.append(CronJob("global_chain_extract", "50 16 * * 1-5",
                [py, str(scripts / "extract_global_supply_chain_events.py")], "global_chain_extract.log",
                network="none", timeout_sec=600))
    if (scripts / "build_global_chain_factors.py").exists():
        jobs.append(CronJob("global_chain_factors", "10 17 * * 1-5",
                [py, str(scripts / "build_global_chain_factors.py")], "global_chain_factors.log",
                network="none", timeout_sec=600))
    # Sentiment: xueqiu + 同花顺 + 东财股吧
    if (scripts / "collect_sentiment_daily.py").exists():
        jobs.append(CronJob("sentiment_daily", "40 16 * * 1-5",
                [py, str(scripts / "collect_sentiment_daily.py")], "sentiment_daily.log",
                network="domestic", timeout_sec=600))
    jobs += [
        CronJob("llm_event_pipeline", "30 16 * * 1-5",
                [py, str(scripts / "run_llm_event_pipeline.py")], "llm_event_pipeline.log",
                network="llm", timeout_sec=7200),
        CronJob("guba_popularity", "35 16 * * 1-5",
                [py, str(scripts / "collect_guba_sentiment.py")], "guba_popularity.log",
                network="domestic", timeout_sec=600),
        # NOTE: The 17:30 llm_event_retry full-rerun was REMOVED 2026-05-31.
        # Reason (cx code review): factors/llm_event_extractor_v2.py:332-335
        # deletes any existing jsonl with <500 lines before re-running. So a
        # successful-but-partial 16:30 pipeline (e.g. 152 events written)
        # would be DESTROYED by 17:30 retry, and a still-throttled second
        # run could leave the day worse than before (regression of already-
        # written events). The 22:30 llm_retry_queue_drain (below) is the
        # correct compensation: it appends queue-recovered events without
        # touching the existing jsonl, and idempotently re-syncs EventStore
        # + rebuilds factors. Do NOT re-add this full-rerun entry.
        CronJob("spot_cache_warmup", "5 17 * * 1-5",
                [py, main_py, "--warm-spot-cache"], "cron_spot_cache_warmup.log",
                network="domestic", timeout_sec=600),
        # --- Post-close: data update (domestic, critical) ---
        CronJob("qlib_data_update", "45 17 * * 1-5",
                [py, str(scripts / "update_qlib_data.py"),
                 # 2026-06-04 cx round 4 P1-7: ``--universe-source
                 # baostock`` is hard-coded here even when price
                 # provider auto-picks Tushare. Production used to
                 # advertise "all Tushare" but the universe still
                 # came from baostock. If your goal is full-Tushare,
                 # change this to "tushare" — and make sure the
                 # Tushare universe endpoint is whitelisted in your
                 # TS token. Tracked in cx round 4 P1-7.
                 "--universe", "all", "--universe-source", "baostock",
                 "--refresh-universe",
                 "--min-health-instruments", "4500",
                 "--min-lgb-data-instruments", "4500",
                 "--check-today"],
                "data_update.log",
                network="domestic", timeout_sec=3600, critical=True),
        CronJob("fund_flow_update", "55 17 * * 1-5",
                [py, str(scripts / "fetch_fund_flow_history.py"), "--incremental", "--workers", "1"],
                "fund_flow_update.log",
                network="domestic", timeout_sec=1800),
        CronJob("valuation_update", "0 18 * * 1-5",
                [py, str(scripts / "fetch_fundamental_valuation.py"), "--days", "10", "--incremental"],
                "valuation_update.log",
                network="domestic", timeout_sec=1200),
        CronJob("regime_daily_update", "5 18 * * 1-5",
                [py, str(scripts / "update_regime_daily.py")], "regime_daily.log",
                network="domestic", timeout_sec=1200),
        # --- Training (none) ---
        CronJob("midweek_train", "15 18 * * 3",
                [py, str(scripts / "train_lgb.py")], "lgb_after_close_train.log",
                network="none", timeout_sec=7200),
        # --- Feature cache rebuild (depends on qlib_data_update + fund_flow_update) ---
        # qlib_data_update 17:45 + fund_flow_update 17:55 (timeout 1800s).
        # In the worst case fund_flow runs until ~18:25, so the cache rebuild
        # must wait for it via the upstream gate or it'll build on stale flows.
        # enforce_deps=True makes run_with_status poll until both upstreams have
        # successfully completed (up to 30 min by default).
        CronJob("feature_cache_rebuild", "25 18 * * 1-5",
                [py, str(scripts / "build_feature_cache.py"), "--all"], "feature_cache_rebuild.log",
                network="domestic", timeout_sec=1800, critical=True, enforce_deps=True),
        # --- Prediction + Paper (none, critical) ---
        # Smoke depends on feature_cache_rebuild; downstream paper/shadow
        # opt into --enforce-deps so stale upstream blocks rather than
        # silently trades on yesterday's signal.
        #
        # Wait-budget reasoning:
        #   qlib_data_update 17:45 + timeout 3600s → worst-case done 18:45
        #   feature_cache_rebuild 18:25 + (waits up to 30min for qlib) +
        #     own 30min timeout → worst-case done 19:15
        #   lgb_after_close_smoke 18:35 must therefore wait up to 40 min
        #     to see cache_rebuild complete; 30 min default would give
        #     up at 19:05, 10 min short. 3600s = 60 min covers it.
        #   All later jobs inherit the same worst case → 3600s across.
        # 2026-06-04 bumped 900 → 1800 — see morning_recommendation
        # comment above. Same inject_supplementary_into_handler cost
        # applies here.
        CronJob("lgb_after_close_smoke", "35 18 * * 1-5",
                [py, str(scripts / "smoke_lgb_predict.py")], "lgb_after_close_smoke.log",
                network="none", timeout_sec=1800, critical=True,
                enforce_deps=True, dep_wait_seconds=3600),
        CronJob("predict_crash_daily", "37 18 * * 1-5",
                [py, str(scripts / "predict_crash_daily.py")], "crash_predict.log",
                network="none", timeout_sec=120,
                enforce_deps=True, dep_wait_seconds=3600),
        CronJob("shadow_optimizer", "40 18 * * 1-5",
                [py, str(scripts / "run_shadow_optimizer.py")], "shadow_optimizer.log",
                network="none", timeout_sec=600,
                enforce_deps=True, dep_wait_seconds=3600),
        CronJob("paper_trading", "42 18 * * 1-5",
                [py, str(scripts / "run_paper_trading.py")], "paper_trading.log",
                network="none", timeout_sec=600,
                enforce_deps=True, dep_wait_seconds=3600),
        CronJob("shadow_chain_overlay", "45 18 * * 1-5",
                [py, str(scripts / "shadow_supply_chain_overlay.py")], "shadow_chain_overlay.log",
                network="none", timeout_sec=120,
                enforce_deps=True, dep_wait_seconds=3600),
        CronJob("shadow_klen_overlay", "46 18 * * 1-5",
                [py, str(scripts / "shadow_klen_overlay.py")], "shadow_klen_overlay.log",
                network="none", timeout_sec=120,
                enforce_deps=True, dep_wait_seconds=3600),
        CronJob("shadow_vol_compression", "47 18 * * 1-5",
                [py, str(scripts / "shadow_vol_compression.py")], "shadow_vol_compression.log",
                network="none", timeout_sec=120,
                enforce_deps=True, dep_wait_seconds=3600),
        CronJob("shadow_roc5_tsmin10", "48 18 * * 1-5",
                [py, str(scripts / "shadow_roc5_tsmin10.py")], "shadow_roc5_tsmin10.log",
                network="none", timeout_sec=120,
                enforce_deps=True, dep_wait_seconds=3600),
        # --- Monitoring (none) ---
        CronJob("factor_decay_monitor", "49 18 * * 1-5",
                [py, str(scripts / "monitor_factor_decay.py")], "factor_decay.log",
                network="none", timeout_sec=600),
        # brinson_attribution: timeout bumped 600 → 1200 (Task #76).
        # The window itself is a fixed 29 days (run_brinson_attribution.py
        # line 53), but Alpha158 dataset preparation has grown past 600s on
        # the current universe — 06-02 took 417s, 06-03 exceeded 600s and
        # was killed by the wrapper. 1200s gives a 2x headroom and still
        # finishes long before the 22:00 evening_outlook gate.
        CronJob("brinson_attribution", "50 18 * * 1-5",
                [py, str(scripts / "run_brinson_attribution.py")], "brinson_attribution.log",
                network="none", timeout_sec=1200),
        # --- LLM 429 retry queue drain (after main pipeline + evening) ---
        # Deliberately NOT enforce_deps. This is a recovery job — gating it
        # on the main pipeline's success would defeat its purpose when the
        # pipeline itself partially failed (which is exactly when the queue
        # has items to retry). Drain self-checks: it no-ops cleanly when the
        # queue file is absent, and its EventStore sync + factor rebuild are
        # idempotent so re-running them is safe.
        CronJob("llm_retry_queue_drain", "30 22 * * 1-5",
                [py, str(scripts / "drain_llm_retry_queue.py")], "llm_retry_drain.log",
                network="llm", timeout_sec=3600),
        CronJob("daily_health_check", "55 18 * * 1-5",
                [py, str(scripts / "daily_health_check.py")], "health_check.log",
                network="none", timeout_sec=300),
        # --- Weekly (Saturday) ---
        CronJob("weekly_full_retrain", "0 4 * * 6",
                [py, str(scripts / "nightly_train.py")], "weekly_retrain.log",
                network="none", timeout_sec=14400),
        CronJob("weekly_st_refresh", "0 3 * * 6",
                [py, str(scripts / "fetch_st_list.py")], "st_refresh.log",
                network="domestic", timeout_sec=600),
        CronJob("weekly_mask_rebuild", "10 3 * * 6",
                [py, str(scripts / "build_tradable_mask.py")], "mask_rebuild.log",
                network="none", timeout_sec=600),
        CronJob("weekly_regime_data", "20 3 * * 6",
                [py, str(scripts / "fetch_fund_holdings.py"), "--macro", "--regime"], "regime_data.log",
                network="domestic", timeout_sec=3600),
    ]
    return jobs


def _quote_arg(arg: str) -> str:
    if all(ch.isalnum() or ch in "/._=-:" for ch in arg):
        return arg
    return "'" + arg.replace("'", "'\"'\"'") + "'"


def render_job(job: CronJob, python_bin: str = DEFAULT_PYTHON, project_root: Path = PROJECT_ROOT) -> str:
    status_wrapper = project_root / "scripts" / "run_with_status.py"
    network_wrapper = project_root / "scripts" / "run_network_job.py"
    log_path = project_root / "logs" / job.log_name

    # Build the innermost command (the actual job)
    inner_cmd = list(job.target)

    # Wrap with run_network_job.py (network profile + timeout)
    network_cmd = [
        python_bin,
        str(network_wrapper),
        "--network", job.network,
    ]
    if job.timeout_sec > 0:
        network_cmd += ["--timeout", str(job.timeout_sec)]
    network_cmd += ["--"] + inner_cmd

    # Wrap with run_with_status.py (job status tracking)
    status_args = [
        python_bin,
        str(status_wrapper),
        "--job-id", job.job_id,
        "--cwd", str(project_root),
    ]
    if job.enforce_deps:
        status_args.append("--enforce-deps")
        if job.dep_wait_seconds != 1800:
            status_args.extend(["--dep-wait-seconds", str(job.dep_wait_seconds)])
    command = status_args + ["--"] + network_cmd

    return (
        f"{job.schedule} "
        f"{' '.join(_quote_arg(str(part)) for part in command)} "
        f">> {_quote_arg(str(log_path))} 2>&1"
    )


def render_block(python_bin: str = DEFAULT_PYTHON, project_root: Path = PROJECT_ROOT) -> str:
    lines = [BEGIN_MARKER]
    lines.extend(render_job(job, python_bin, project_root) for job in managed_jobs(python_bin, project_root))
    lines.append(END_MARKER)
    return "\n".join(lines)


def strip_managed_block(crontab_text: str) -> str:
    lines = crontab_text.splitlines()
    output: list[str] = []
    inside = False
    for line in lines:
        if line.strip() == BEGIN_MARKER:
            inside = True
            continue
        if line.strip() == END_MARKER:
            inside = False
            continue
        if inside:
            continue
        output.append(line)
    return "\n".join(output).strip()


def is_legacy_project_line(line: str, project_root: Path = PROJECT_ROOT) -> bool:
    if str(project_root) not in line:
        return False
    legacy_markers = (
        "main.py --run-now",
        "main.py --morning",
        "main.py --sell-check",
        "main.py --daily-summary",
        "main.py --evening-outlook",
        "main.py --risk-check",
        "main.py --warm-spot-cache",
        "scripts/update_qlib_data.py",
        "scripts/fetch_fund_flow_history.py",
        "scripts/fetch_fundamental_valuation.py",
        "scripts/fetch_fundamental_quality.py",
        "scripts/monitor_factor_decay.py",
        "scripts/run_brinson_attribution.py",
        "scripts/nightly_train.py",
        "scripts/train_lgb.py",
        "scripts/smoke_lgb_predict.py",
        "scripts/run_llm_event_pipeline.py",
    )
    return any(marker in line for marker in legacy_markers)


def merge_crontab(existing: str, python_bin: str = DEFAULT_PYTHON, project_root: Path = PROJECT_ROOT) -> str:
    stripped = strip_managed_block(existing)
    preserved = [
        line for line in stripped.splitlines()
        if line.strip() and not is_legacy_project_line(line, project_root)
    ]
    merged = "\n".join([*preserved, render_block(python_bin, project_root)]).strip()
    return merged + "\n"


def current_crontab() -> str:
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode != 0:
        return ""
    return result.stdout


def install_crontab(content: str) -> None:
    subprocess.run(["crontab", "-"], input=content, text=True, check=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    new_crontab = merge_crontab(current_crontab(), args.python_bin, PROJECT_ROOT)
    print(new_crontab)
    if args.apply:
        install_crontab(new_crontab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
