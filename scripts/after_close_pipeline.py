"""After-close serial pipeline: data update → health → train → smoke → evaluate.

Replaces the 3 independent cron jobs (17:00/17:35/17:55) with one serial pipeline.
Data update failure is non-blocking if yesterday's data passes health check.

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


def _push_alert(title, content):
    """Best-effort WeChat alert on pipeline failure."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from push.wechat import WeChatPusher
        pusher = WeChatPusher()
        pusher.send(content, title=title)
        logger.info(f"  Alert pushed: {title}")
    except Exception as e:
        logger.warning(f"  Alert push failed: {e}")


def main():
    skip_update = "--skip-update" in sys.argv

    logger.info(f"After-close pipeline started at {datetime.now()}")
    logger.info(f"Python: {PY}")

    # Step 1: Data update (incremental) — failure is non-blocking
    data_updated = True
    if not skip_update:
        data_updated = run_step(
            "Qlib Data Update",
            "update_qlib_data.py",
            "--provider", os.environ.get("QLIB_DATA_PROVIDER", "auto"),
            "--universe", os.environ.get("LGB_INFERENCE_UNIVERSE", "all"),
            "--universe-source", os.environ.get("QLIB_UNIVERSE_SOURCE", "auto"),
            "--refresh-universe",
            "--min-health-instruments", os.environ.get("LGB_MIN_DATA_INSTRUMENTS", "4500"),
            "--min-lgb-data-instruments", os.environ.get("LGB_MIN_DATA_INSTRUMENTS", "4500"),
            timeout=7200,
        )
        if not data_updated:
            logger.warning("⚠ Data update failed — checking if existing data is still usable")
    else:
        logger.info("Skipping data update (--skip-update)")

    # Step 2: Health check — decides whether to continue
    health_ok = run_step(
        "Qlib Data Health",
        "check_qlib_data_health.py",
        "--universe", os.environ.get("LGB_INFERENCE_UNIVERSE", "all"),
        "--min-instruments", os.environ.get("LGB_MIN_DATA_INSTRUMENTS", "4500"),
        timeout=300,
    )

    if not health_ok:
        msg = "数据更新失败" if not data_updated else "数据更新成功但健康检查失败"
        logger.error(f"PIPELINE STOPPED: {msg}，现有数据不可用")
        _push_alert("盘后Pipeline失败", f"{msg}，训练跳过。请手动检查数据。")
        return 1

    if not data_updated and not skip_update:
        logger.warning("✓ Existing data passed health check — continuing with yesterday's data")
        _push_alert("盘后数据更新失败",
                     "数据更新失败但现有数据健康检查通过，使用昨日数据继续训练。")

    # Step 3: Train LGB
    if not run_step("LGB Training", "train_lgb.py", timeout=1800):
        logger.error("PIPELINE STOPPED: LGB training failed")
        _push_alert("盘后Pipeline失败", "LGB训练失败")
        return 1

    # Step 4: Smoke prediction
    if not run_step("LGB Smoke", "smoke_lgb_predict.py", timeout=600):
        logger.error("PIPELINE STOPPED: LGB smoke prediction failed")
        _push_alert("盘后Pipeline失败", "LGB Smoke预测失败")
        return 1

    # Step 5: Evaluate quality (non-blocking)
    if not run_step("LGB Evaluate", "evaluate_lgb_test.py", timeout=600):
        logger.warning("Evaluation failed (non-blocking, model already saved)")

    # Step 6: Brinson attribution (non-blocking)
    if not run_step("Brinson Attribution", "attribution.py", timeout=300):
        logger.warning("Attribution failed (non-blocking)")

    # Step 7: Factor decay monitoring (non-blocking)
    if not run_step("Factor Decay Monitor", "monitor_factor_decay.py", timeout=60):
        logger.warning("Factor decay check failed (non-blocking)")

    # Step 8: Model promotion gate (non-blocking)
    if not run_step("Model Promotion Check", "phase4_promote.py", "--check", timeout=60):
        logger.warning("Model promotion check failed (non-blocking)")

    status = "completed" if data_updated else "completed (with stale data warning)"
    logger.info(f"After-close pipeline {status} at {datetime.now()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
