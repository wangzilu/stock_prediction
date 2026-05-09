"""Nightly training pipeline.

Runs at 4:00 AM daily (crontab):
1. Incrementally update Qlib data
2. Validate Qlib data health
3. Retrain LightGBM model
4. Smoke-test production LGB inference
5. Train/update RL agent (Transformer+SAC, offline)

Usage: python scripts/nightly_train.py
"""
import os
import sys
import subprocess
import logging
import selectors
import time
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
QLIB_DATA_PROVIDER = os.environ.get("QLIB_DATA_PROVIDER", "auto")
QLIB_UNIVERSE_SOURCE = os.environ.get("QLIB_UNIVERSE_SOURCE", "baostock")
LGB_INFERENCE_UNIVERSE = os.environ.get("LGB_INFERENCE_UNIVERSE", "all")
LGB_MIN_DATA_INSTRUMENTS = os.environ.get("LGB_MIN_DATA_INSTRUMENTS", "4500")


def run_step(name, script, *args, timeout=3600):
    """Run a training step as subprocess."""
    logger.info(f"=== {name} ===")
    start = datetime.now()
    start_mono = time.monotonic()
    cmd = [PY, os.path.join(PROJECT_ROOT, "scripts", script), *args]
    tail: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None

        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)

        while proc.poll() is None:
            if time.monotonic() - start_mono > timeout:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                logger.error(f"{name} timed out after {timeout}s")
                return False

            for key, _ in selector.select(timeout=1):
                line = key.fileobj.readline()
                if not line:
                    continue
                line = line.rstrip()
                tail = [*tail[-19:], line]
                logger.info("  %s", line)

        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            tail = [*tail[-19:], line]
            logger.info("  %s", line)

        duration = (datetime.now() - start).total_seconds()
        if proc.returncode == 0:
            logger.info(f"{name} completed in {duration:.0f}s")
            return True

        logger.error(f"{name} failed (exit {proc.returncode}) in {duration:.0f}s")
        for line in tail[-10:]:
            logger.error("  %s", line)
        return False
    except Exception as e:
        logger.error(f"{name} error: {e}")
        return False


def main():
    logger.info(f"Nightly training started at {datetime.now()}")

    # Step 1: Incremental update with staging + health gate.
    if not run_step(
        "Data Update (incremental → Qlib)",
        "update_qlib_data.py",
        "--provider",
        QLIB_DATA_PROVIDER,
        "--universe",
        LGB_INFERENCE_UNIVERSE,
        "--universe-source",
        QLIB_UNIVERSE_SOURCE,
        "--refresh-universe",
        "--min-universe-size",
        LGB_MIN_DATA_INSTRUMENTS,
        "--min-health-instruments",
        LGB_MIN_DATA_INSTRUMENTS,
        "--min-lgb-data-instruments",
        LGB_MIN_DATA_INSTRUMENTS,
        timeout=21600,
    ):
        logger.error("Stopping nightly training because data update failed")
        return 1

    # Step 2: Explicit health check before model training.
    if not run_step(
        "Qlib Data Health Check",
        "check_qlib_data_health.py",
        "--universe",
        LGB_INFERENCE_UNIVERSE,
        "--min-instruments",
        LGB_MIN_DATA_INSTRUMENTS,
        timeout=900,
    ):
        logger.error("Stopping nightly training because Qlib data health failed")
        return 1

    # Step 3: Retrain LightGBM.
    if not run_step("LightGBM Training", "train_lgb.py"):
        logger.error("Stopping nightly training because LightGBM training failed")
        return 1

    # Step 4: Confirm the exact production inference path works.
    if not run_step("LGB Smoke Prediction", "smoke_lgb_predict.py", timeout=900):
        logger.error("Stopping nightly training because LGB smoke prediction failed")
        return 1

    # Step 5: RL training (offline, not deployed until metrics mature).
    if not run_step("RL Agent Training (Transformer+SAC)", "train_rl.py"):
        logger.error("RL training failed after data/LGB steps; leaving previous RL model in place")
        logger.info("Nightly data and LGB steps completed; treating RL as non-blocking research")
        return 0

    logger.info(f"Nightly training complete at {datetime.now()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
