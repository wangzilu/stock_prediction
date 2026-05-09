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

    # Default: start scheduler
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler()

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

    # 22:00 Evening outlook
    scheduler.add_job(
        _status_wrapped("evening_outlook", pipeline.run_evening_outlook),
        CronTrigger(day_of_week="mon-fri", hour=22, minute=0),
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

    logger.info("Scheduler started. Jobs:")
    logger.info("  - Morning recommendation: Mon-Fri 09:20")
    logger.info("  - Intraday decision: Mon-Fri 14:30")
    logger.info("  - Daily summary: Mon-Fri 15:30")
    logger.info("  - Spot cache warmup: Mon-Fri 17:05")
    logger.info("  - Evening outlook: Mon-Fri 22:00")
    logger.info("  - Risk check: Mon-Fri 9:30-15:30 (hourly)")
    logger.info("Press Ctrl+C to exit.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
