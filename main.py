"""Stock Prediction System - Main Entry Point.

Usage:
    python main.py              # Start scheduler (daily 14:00 + hourly risk check)
    python main.py --run-now    # Run recommendation pipeline immediately
    python main.py --verify     # Run verification check immediately
    python main.py --risk-check # Run risk check immediately
    python main.py --setup      # Download Qlib data (first-time setup)
"""
import sys
import os
import logging

# Ensure working directory is project root (for crontab execution)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

    if "--risk-check" in args:
        logger.info("Running risk check now...")
        pipeline.run_risk_check()
        return

    # Default: start scheduler
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler()

    # Daily recommendation at 14:00 on weekdays
    scheduler.add_job(
        pipeline.run_daily_recommendation,
        CronTrigger(day_of_week="mon-fri", hour=14, minute=0),
        id="daily_recommendation",
        name="Daily Stock Recommendation",
    )

    # Verification check at 14:05 on weekdays
    scheduler.add_job(
        pipeline.run_verification,
        CronTrigger(day_of_week="mon-fri", hour=14, minute=5),
        id="verification",
        name="5-Day Verification Check",
    )

    # Hourly risk check (every hour during trading hours 9-15, weekdays)
    scheduler.add_job(
        pipeline.run_risk_check,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute=30),
        id="risk_check",
        name="Hourly Risk Check",
    )

    logger.info("Scheduler started. Jobs:")
    logger.info("  - Daily recommendation: Mon-Fri 14:00")
    logger.info("  - Verification check: Mon-Fri 14:05")
    logger.info("  - Risk check: Mon-Fri 9:30-15:30 (hourly)")
    logger.info("Press Ctrl+C to exit.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
