"""After-close serial pipeline: data update → health → train → smoke → evaluate.

Replaces the 3 independent cron jobs (17:00/17:35/17:55) with one serial pipeline.
Any step failure stops the chain — no training on stale/broken data.

Usage:
    python scripts/after_close_pipeline.py
    python scripts/after_close_pipeline.py --skip-update  # skip data update (for testing)
"""
import os
import sys
import subprocess
import logging
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run_step(name, script, *args, timeout=3600):
    """Run a pipeline step. Returns True on success, False on failure."""
    logger.info(f">>> {name}")
    start = time.monotonic()
    cmd = [PY, os.path.join(PROJECT_ROOT, "scripts", script), *args]
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
        )
        duration = time.monotonic() - start
        if result.returncode == 0:
            # Print last 5 lines of output
            for line in result.stdout.strip().split("\n")[-5:]:
                logger.info(f"  {line}")
            logger.info(f"  {name} OK ({duration:.0f}s)")
            return True
        else:
            logger.error(f"  {name} FAILED (exit {result.returncode}, {duration:.0f}s)")
            for line in result.stderr.strip().split("\n")[-10:]:
                logger.error(f"  {line}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"  {name} TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"  {name} ERROR: {e}")
        return False


def main():
    skip_update = "--skip-update" in sys.argv

    logger.info(f"After-close pipeline started at {datetime.now()}")
    logger.info(f"Python: {PY}")

    # Step 1: Data update (incremental)
    if not skip_update:
        if not run_step(
            "Qlib Data Update",
            "update_qlib_data.py",
            "--provider", os.environ.get("QLIB_DATA_PROVIDER", "auto"),
            "--universe", os.environ.get("LGB_INFERENCE_UNIVERSE", "all"),
            "--universe-source", os.environ.get("QLIB_UNIVERSE_SOURCE", "baostock"),
            "--refresh-universe",
            "--min-health-instruments", os.environ.get("LGB_MIN_DATA_INSTRUMENTS", "4500"),
            "--min-lgb-data-instruments", os.environ.get("LGB_MIN_DATA_INSTRUMENTS", "4500"),
            timeout=7200,  # 2 hours max for data update
        ):
            logger.error("PIPELINE STOPPED: data update failed")
            return 1
    else:
        logger.info("Skipping data update (--skip-update)")

    # Step 2: Health check
    if not run_step(
        "Qlib Data Health",
        "check_qlib_data_health.py",
        "--universe", os.environ.get("LGB_INFERENCE_UNIVERSE", "all"),
        "--min-instruments", os.environ.get("LGB_MIN_DATA_INSTRUMENTS", "4500"),
        timeout=300,
    ):
        logger.error("PIPELINE STOPPED: data health check failed")
        return 1

    # Step 3: Train LGB
    if not run_step("LGB Training", "train_lgb.py", timeout=1800):
        logger.error("PIPELINE STOPPED: LGB training failed")
        return 1

    # Step 4: Smoke prediction
    if not run_step("LGB Smoke", "smoke_lgb_predict.py", timeout=600):
        logger.error("PIPELINE STOPPED: LGB smoke prediction failed")
        return 1

    # Step 5: Evaluate quality
    if not run_step("LGB Evaluate", "evaluate_lgb_test.py", timeout=600):
        logger.warning("Evaluation failed but not blocking pipeline (model already saved)")

    logger.info(f"After-close pipeline completed at {datetime.now()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
