"""Nightly training pipeline.

Runs at 2:00 AM daily:
1. Update Qlib data from baostock (latest prices)
2. Retrain LightGBM model
3. Train/update RL agent (TODO: tianshou)

Usage: python scripts/nightly_train.py
"""
import os
import sys
import subprocess
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run_step(name, script):
    """Run a training step as subprocess."""
    logger.info(f"=== {name} ===")
    start = datetime.now()
    try:
        result = subprocess.run(
            [PY, os.path.join(PROJECT_ROOT, "scripts", script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )
        duration = (datetime.now() - start).total_seconds()
        if result.returncode == 0:
            logger.info(f"{name} completed in {duration:.0f}s")
            # Print last few lines of output
            for line in result.stdout.strip().split("\n")[-5:]:
                logger.info(f"  {line}")
        else:
            logger.error(f"{name} failed (exit {result.returncode}) in {duration:.0f}s")
            for line in result.stderr.strip().split("\n")[-10:]:
                logger.error(f"  {line}")
    except subprocess.TimeoutExpired:
        logger.error(f"{name} timed out after 1 hour")
    except Exception as e:
        logger.error(f"{name} error: {e}")


def main():
    logger.info(f"Nightly training started at {datetime.now()}")

    # Step 1: Update data
    run_step("Data Update (baostock → Qlib)", "update_qlib_data.py")

    # Step 2: Retrain LightGBM
    run_step("LightGBM Training", "train_lgb.py")

    # Step 3: RL training (TODO)
    # run_step("RL Agent Training", "train_rl.py")

    logger.info(f"Nightly training complete at {datetime.now()}")


if __name__ == "__main__":
    main()
