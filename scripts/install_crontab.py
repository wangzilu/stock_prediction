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


def managed_jobs(python_bin: str = DEFAULT_PYTHON, project_root: Path = PROJECT_ROOT) -> list[CronJob]:
    py = python_bin
    root = str(project_root)
    main_py = str(project_root / "main.py")
    scripts = project_root / "scripts"
    return [
        CronJob("morning_recommendation", "20 9 * * 1-5", [py, main_py, "--morning"], "cron_morning.log"),
        CronJob("sell_check", "30 14 * * 1-5", [py, main_py, "--sell-check"], "cron_sell_check.log"),
        CronJob("daily_summary", "30 15 * * 1-5", [py, main_py, "--daily-summary"], "cron_daily_summary.log"),
        CronJob("evening_outlook", "0 22 * * 1-5", [py, main_py, "--evening-outlook"], "cron_evening_outlook.log"),
        CronJob("risk_check", "35 9-15 * * 1-5", [py, main_py, "--risk-check"], "cron_risk_check.log"),
        CronJob(
            "llm_event_pipeline",
            "30 16 * * 1-5",  # 16:30 — after market close, news published after 15:00
            [py, str(scripts / "run_llm_event_pipeline.py")],
            "llm_event_pipeline.log",
        ),
        CronJob("spot_cache_warmup", "5 17 * * 1-5", [py, main_py, "--warm-spot-cache"], "cron_spot_cache_warmup.log"),
        CronJob(
            "qlib_data_update",
            "45 17 * * 1-5",  # 17:45 — baostock当天数据通常17:30后可用
            [
                py,
                str(scripts / "update_qlib_data.py"),
                "--universe",
                "all",
                "--universe-source",
                "baostock",
                "--refresh-universe",
                "--min-health-instruments",
                "4500",
                "--min-lgb-data-instruments",
                "4500",
                "--check-today",  # 健康检查：验证最新日期是今天
            ],
            "data_update.log",
        ),
        CronJob(
            "fund_flow_update",
            "55 17 * * 1-5",  # 17:55 (after data update)
            [py, str(scripts / "fetch_fund_flow_history.py"), "--incremental", "--workers", "1"],
            "fund_flow_update.log",
        ),
        CronJob(
            "valuation_update",
            "0 18 * * 1-5",  # 18:00
            [py, str(scripts / "fetch_fundamental_valuation.py"), "--days", "10", "--incremental"],
            "valuation_update.log",
        ),
        CronJob("lgb_after_close_train", "15 18 * * 1-5", [py, str(scripts / "train_lgb.py")], "lgb_after_close_train.log"),
        CronJob("lgb_after_close_smoke", "35 18 * * 1-5", [py, str(scripts / "smoke_lgb_predict.py")], "lgb_after_close_smoke.log"),
        CronJob("shadow_optimizer", "40 18 * * 1-5", [py, str(scripts / "run_shadow_optimizer.py")], "shadow_optimizer.log"),
        CronJob("paper_trading", "42 18 * * 1-5", [py, str(scripts / "run_paper_trading.py")], "paper_trading.log"),
        CronJob("factor_decay_monitor", "45 18 * * 1-5", [py, str(scripts / "monitor_factor_decay.py")], "factor_decay.log"),
        CronJob("brinson_attribution", "50 18 * * 1-5", [py, str(scripts / "run_brinson_attribution.py")], "brinson_attribution.log"),
        # Weekly full retrain on Saturday (replaces daily 04:00 — 18:15 daily train is sufficient)
        CronJob("weekly_full_retrain", "0 4 * * 6", [py, str(scripts / "nightly_train.py")], "weekly_retrain.log"),
        # Weekly ST list refresh on Saturday before retrain
        CronJob("weekly_st_refresh", "0 3 * * 6", [py, str(scripts / "fetch_st_list.py")], "st_refresh.log"),
        # Weekly tradable mask rebuild after ST refresh
        CronJob("weekly_mask_rebuild", "10 3 * * 6", [py, str(scripts / "build_tradable_mask.py")], "mask_rebuild.log"),
    ]


def _quote_arg(arg: str) -> str:
    if all(ch.isalnum() or ch in "/._=-:" for ch in arg):
        return arg
    return "'" + arg.replace("'", "'\"'\"'") + "'"


def render_job(job: CronJob, python_bin: str = DEFAULT_PYTHON, project_root: Path = PROJECT_ROOT) -> str:
    wrapper = project_root / "scripts" / "run_with_status.py"
    log_path = project_root / "logs" / job.log_name
    command = [
        python_bin,
        str(wrapper),
        "--job-id",
        job.job_id,
        "--cwd",
        str(project_root),
        "--",
        *job.target,
    ]
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
