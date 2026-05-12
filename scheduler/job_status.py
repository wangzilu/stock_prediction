"""Persist lightweight status for scheduled jobs and push alerts on failure."""
from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from config.settings import DATA_DIR


logger = logging.getLogger(__name__)

DEFAULT_STATUS_PATH = DATA_DIR / "job_status.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class JobStatusStore:
    """Small JSON store for last-run job status."""

    def __init__(self, path: Path | str = DEFAULT_STATUS_PATH):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "jobs": {}}
        try:
            payload = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return {"version": 1, "jobs": {}}
        payload.setdefault("version", 1)
        payload.setdefault("jobs", {})
        return payload

    def save(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        tmp_path.replace(self.path)

    def update_job(self, job_id: str, **fields: Any) -> None:
        payload = self.load()
        jobs = payload.setdefault("jobs", {})
        current = jobs.setdefault(job_id, {})
        current.update(fields)
        payload["updated_at"] = _now()
        self.save(payload)


def run_with_status(
    job_id: str,
    func: Callable[[], Any],
    *,
    status_path: Path | str = DEFAULT_STATUS_PATH,
) -> Any:
    """Run a callable and persist started/succeeded/failed status."""
    store = JobStatusStore(status_path)
    started = time.time()
    started_at = _now()
    payload = store.load()
    previous = payload.setdefault("jobs", {}).get(job_id, {})
    run_count = int(previous.get("run_count", 0)) + 1
    store.update_job(
        job_id,
        status="running",
        started_at=started_at,
        finished_at="",
        duration_seconds=None,
        error="",
        traceback="",
        run_count=run_count,
    )

    try:
        result = func()
    except Exception as exc:
        duration = round(time.time() - started, 2)
        error_msg = f"{type(exc).__name__}: {exc}"
        store.update_job(
            job_id,
            status="failed",
            finished_at=_now(),
            duration_seconds=duration,
            error=error_msg,
            traceback=traceback.format_exc(limit=20),
        )
        _push_failure_alert(job_id, error_msg, duration, started_at)
        raise

    store.update_job(
        job_id,
        status="success",
        finished_at=_now(),
        duration_seconds=round(time.time() - started, 2),
        error="",
        traceback="",
    )
    return result


# ---------- failure alert ----------

_JOB_DISPLAY_NAMES = {
    "morning_recommendation": "晨推",
    "sell_check": "盘中决策",
    "daily_summary": "收盘总结",
    "evening_outlook": "晚间展望",
    "risk_check": "风控检查",
    "spot_cache_warmup": "行情缓存",
    "qlib_data_update": "Qlib数据更新",
    "fund_flow_update": "资金流向抓取",
    "valuation_update": "估值因子更新",
    "lgb_after_close_train": "模型训练",
    "lgb_after_close_smoke": "模型冒烟测试",
    "nightly_train": "夜间训练",
}


def _push_failure_alert(job_id: str, error: str, duration: float, started_at: str):
    """Push a WeChat alert when a scheduled job fails."""
    try:
        from push.wechat import WeChatPusher
        pusher = WeChatPusher()
    except Exception:
        logger.warning("Cannot send failure alert: WeChatPusher init failed")
        return

    display = _JOB_DISPLAY_NAMES.get(job_id, job_id)
    # Truncate error to avoid overly long push
    short_error = error[:200] + "..." if len(error) > 200 else error
    msg = (
        f"任务【{display}】执行失败\n"
        f"Job ID: {job_id}\n"
        f"开始时间: {started_at}\n"
        f"耗时: {duration:.0f}s\n"
        f"错误: {short_error}"
    )
    try:
        pusher.send(msg, title=f"🚨 任务异常: {display}")
        logger.info(f"Failure alert pushed for job {job_id}")
    except Exception as e:
        logger.warning(f"Failed to push failure alert for {job_id}: {e}")
