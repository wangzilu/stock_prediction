"""Persist lightweight status for scheduled jobs."""
from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from config.settings import DATA_DIR


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
        store.update_job(
            job_id,
            status="failed",
            finished_at=_now(),
            duration_seconds=round(time.time() - started, 2),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=20),
        )
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
