"""Stock Prediction System - Main Entry Point.

Usage:
    python main.py              # Start scheduler (4-slot schedule + risk checks)
    python main.py --run-now    # Run recommendation pipeline immediately
    python main.py --morning    # Run morning recommendation once
    python main.py --sell-check # Run 14:30 intraday decision once
    python main.py --daily-summary
    python main.py --evening-outlook
    python main.py --warm-spot-cache
    python main.py --verify     # Run verification check immediately
    python main.py --risk-check # Run risk check immediately
    python main.py --setup      # Download Qlib data (first-time setup)
"""
import sys
import os
import logging
import importlib.util

# 2026-06-08 morning-hang root cause: macOS spawns joblib workers via
# `loky` (spawn-based). Workers re-import this script, triggering
# scheduler.jobs / qlib imports that themselves call joblib.Parallel
# (qlib.data.dataset_processor at qlib/data/data.py:577). Nested
# joblib in a spawn-bootstrapping child raises RuntimeError(
# "An attempt has been made to start a new process before the
# current process has finished its bootstrapping phase"), the worker
# dies, and the parent joblib.Parallel hangs forever waiting for
# results — that's the 25-minute silent hang at 09:24.
#
# Forcing JOBLIB_MULTIPROCESSING=0 tells joblib to fall back to a
# sequential loop. No spawn, no recursion, no hang. Slightly slower
# Alpha158 setup (~30s vs ~10s) but it actually completes. Set BEFORE
# any joblib/qlib import. setdefault so an operator who explicitly
# wants parallel (e.g. for batch experimentation) can override.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

# Ensure working directory is project root (for crontab execution)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _log_runtime_environment(args):
    """Log the actual Python/Qlib runtime used by this process."""
    qlib_spec = importlib.util.find_spec("qlib")
    logger.info(
        "Runtime: python=%s version=%s conda_prefix=%s qlib=%s args=%s",
        sys.executable,
        sys.version.split()[0],
        os.environ.get("CONDA_PREFIX", ""),
        qlib_spec.origin if qlib_spec else "NOT_FOUND",
        " ".join(args) if args else "(scheduler)",
    )
    if qlib_spec is None:
        logger.warning(
            "Qlib is not importable in this Python. For production runs use "
            "/Users/wangzilu/miniconda3/envs/tianshou/bin/python, not a system python3."
        )


def _status_wrapped(job_id, func):
    """Wrap APScheduler jobs with persisted status."""
    from scheduler.job_status import run_with_status

    def _run():
        return run_with_status(job_id, func)

    return _run


def main():
    args = sys.argv[1:]
    _log_runtime_environment(args)

    if "--setup" in args:
        from factors.quant import prepare_qlib_data
        prepare_qlib_data()
        return

    from scheduler.jobs import DailyPipeline
    pipeline = DailyPipeline()

    if "--run-now" in args:
        logger.info("Running recommendation pipeline now...")
        pipeline.run_daily_recommendation()
        return

    if "--morning" in args:
        logger.info("Running morning recommendation now...")
        pipeline.run_morning_recommendation()
        return

    if "--sell-check" in args:
        logger.info("Running sell check now...")
        pipeline.run_sell_check()
        return

    if "--daily-summary" in args:
        logger.info("Running daily summary now...")
        pipeline.run_daily_summary()
        return

    if "--evening-outlook" in args:
        logger.info("Running evening outlook now...")
        pipeline.run_evening_outlook()
        return

    if "--warm-spot-cache" in args:
        logger.info("Running spot cache warmup now...")
        pipeline.run_spot_cache_warmup()
        return

    if "--verify" in args:
        logger.info("Running verification now...")
        pipeline.run_verification()
        return

    if "--risk-check" in args:
        logger.info("Running risk check now...")
        pipeline.run_risk_check()
        return

    # 2026-06-04 cx round 15 P1-1: the in-process APScheduler block is
    # DISABLED by default. The production schedule lives in
    # scripts/install_crontab.py — the single source of truth — and
    # diverged from this block (e.g. evening_outlook here is Mon-Fri
    # but cron is Sun-Thu to cover Sun→Mon markets, the old bug from
    # 2026-05-31). Re-enabling this block would resurrect that bug.
    # To intentionally run the in-process scheduler for offline /
    # disconnected use, set MAIN_PY_INPROC_SCHEDULER=acknowledge_unsafe.
    if os.environ.get("MAIN_PY_INPROC_SCHEDULER") != "acknowledge_unsafe":
        logger.error(
            "main.py default scheduler path is DISABLED — use the cron "
            "installed by scripts/install_crontab.py. Set "
            "MAIN_PY_INPROC_SCHEDULER=acknowledge_unsafe to override "
            "(NOT recommended; the in-process schedule has diverged "
            "from the production cron and brings back the Sun→Mon bug)."
        )
        sys.exit(2)

    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler()

    # NOTE: the schedule below DOES NOT match
    # scripts/install_crontab.py. It is retained for diagnostic /
    # disconnected use only; the production schedule comes from cron.

    # 9:20 Morning recommendation (pre-market)
    scheduler.add_job(
        _status_wrapped("morning_recommendation", pipeline.run_morning_recommendation),
        CronTrigger(day_of_week="mon-fri", hour=9, minute=20),
        id="morning_recommendation",
        name="Morning Recommendation",
    )

    # 14:30 Intraday decision: next-open indices + strong buys + mandatory sells
    scheduler.add_job(
        _status_wrapped("sell_check", pipeline.run_sell_check),
        CronTrigger(day_of_week="mon-fri", hour=14, minute=30),
        id="sell_check",
        name="Intraday Decision",
    )

    # 15:30 Daily summary (post-close + verification)
    scheduler.add_job(
        _status_wrapped("daily_summary", pipeline.run_daily_summary),
        CronTrigger(day_of_week="mon-fri", hour=15, minute=30),
        id="daily_summary",
        name="Daily Summary",
    )

    # 22:00 Evening outlook — note: cron uses sun-thu, not mon-fri
    scheduler.add_job(
        _status_wrapped("evening_outlook", pipeline.run_evening_outlook),
        CronTrigger(day_of_week="sun-thu", hour=22, minute=0),
        id="evening_outlook",
        name="Evening Outlook",
    )

    # 17:05 Warm full-market spot cache for evening/morning one-shot pushes
    scheduler.add_job(
        _status_wrapped("spot_cache_warmup", pipeline.run_spot_cache_warmup),
        CronTrigger(day_of_week="mon-fri", hour=17, minute=5),
        id="spot_cache_warmup",
        name="Spot Cache Warmup",
    )

    # Hourly risk check (every hour during trading hours 9-15, weekdays)
    scheduler.add_job(
        _status_wrapped("risk_check", pipeline.run_risk_check),
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute=30),
        id="risk_check",
        name="Hourly Risk Check",
    )

    logger.info("Scheduler started (MAIN_PY_INPROC_SCHEDULER override active). Jobs:")
    logger.info("  - Morning recommendation: Mon-Fri 09:20")
    logger.info("  - Intraday decision: Mon-Fri 14:30")
    logger.info("  - Daily summary: Mon-Fri 15:30")
    logger.info("  - Spot cache warmup: Mon-Fri 17:05")
    logger.info("  - Evening outlook: Sun-Thu 22:00 (matches cron Sun-Thu fix)")
    logger.info("  - Risk check: Mon-Fri 9:30-15:30 (hourly)")
    logger.info("Press Ctrl+C to exit.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
