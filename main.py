"""Stock Prediction System - Main Entry Point.

Usage:
    python main.py              # Start scheduler (runs daily at 14:00)
    python main.py --run-now    # Run recommendation pipeline immediately
    python main.py --verify     # Run verification check immediately
    python main.py --setup      # Download Qlib data (first-time setup)
"""
import sys
import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    args = sys.argv[1:]

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

    if "--verify" in args:
        logger.info("Running verification now...")
        pipeline.run_verification()
        return

    # Default: start scheduler
    scheduler = BlockingScheduler()

    scheduler.add_job(
        pipeline.run_daily_recommendation,
        CronTrigger(day_of_week="mon-fri", hour=14, minute=0),
        id="daily_recommendation",
        name="Daily Stock Recommendation",
    )

    scheduler.add_job(
        pipeline.run_verification,
        CronTrigger(day_of_week="mon-fri", hour=14, minute=5),
        id="verification",
        name="5-Day Verification Check",
    )

    logger.info("Scheduler started. Jobs:")
    logger.info("  - Daily recommendation: Mon-Fri 14:00")
    logger.info("  - Verification check: Mon-Fri 14:05")
    logger.info("Press Ctrl+C to exit.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
