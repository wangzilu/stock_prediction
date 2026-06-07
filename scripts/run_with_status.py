"""Run a scheduled command and persist status to data/storage/job_status.json.

Usage:
    python scripts/run_with_status.py --job-id morning -- python main.py --morning

Cron-time upstream gate: if --enforce-deps is set, looks up the job's upstream
dependencies in scheduler.job_deps.JOB_DEPS and POLLS them until ready or the
wait budget expires. Standard crontab does NOT retry on non-zero exit codes,
so a single short-circuit check would leave downstream jobs unrun for the
entire day if an upstream is merely slow. The polling loop sleeps in 60s
increments up to --dep-wait-seconds (default 1800 = 30 min).

After ready, runs the wrapped command and writes mark_complete on every
exit path so check_upstream sees a clean DAG state — previously the per-job
status files in data/storage/job_status/ were never written by this wrapper,
so check_upstream always returned ready when the daily files were missing.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scheduler.job_deps import check_upstream_full, mark_complete  # noqa: E402
from scheduler.job_status import run_with_status  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--cwd", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--timeout", type=int, default=0, help="Timeout in seconds; 0 disables timeout")
    parser.add_argument(
        "--enforce-deps", action="store_true",
        help="Refuse to run when upstream deps are missing for today (cron use)",
    )
    parser.add_argument(
        "--dep-wait-seconds", type=int, default=1800,
        help="When --enforce-deps is set, max wall-clock seconds to wait for "
             "upstream to complete before giving up (default 1800 = 30 min). "
             "Polls every 60s.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("command is required after --")
    return args


def _push_dep_block_alert(job_id: str, date: str, details: str) -> None:
    """Send a WeChat alert when the dep-gate refuses to launch a job.

    Mirrors scheduler.job_status._push_failure_alert but composes a
    DAG-block message rather than an exception message. Any failure
    of the push itself is logged and swallowed — alerting must never
    take down the wrapper.
    """
    try:
        from push.wechat import WeChatPusher
        from scheduler.job_status import _JOB_DISPLAY_NAMES
        pusher = WeChatPusher()
        display = _JOB_DISPLAY_NAMES.get(job_id, job_id)
        msg = (
            f"任务【{display}】被 DAG 拒绝启动\n"
            f"Job ID: {job_id}\n"
            f"日期: {date}\n"
            f"原因: {details}"
        )
        pusher.send(msg, title=f"⚠️ 上游未就绪: {display}")
        logger.info("Dep-block alert pushed for job %s", job_id)
    except Exception as e:
        logger.warning("Failed to push dep-block alert for %s: %s", job_id, e)


def _wait_for_upstream(job_id: str, today: str, max_wait_sec: int) -> dict:
    """Poll upstream readiness up to max_wait_sec.

    Returns the LAST check_upstream_full payload. ``payload["ready"]``
    is True iff both same-day AND previous-business-day deps are
    satisfied; callers should inspect ``missing`` and
    ``prev_bday_missing`` separately to compose a useful failure
    message.
    """
    deadline = time.time() + max_wait_sec
    poll_interval = 60
    last_logged: tuple = ()
    while True:
        deps = check_upstream_full(job_id, today)
        if deps["ready"]:
            return deps
        same_missing = tuple(deps["missing"])
        prev_missing = tuple(deps["prev_bday_missing"])
        log_key = (same_missing, prev_missing)
        if log_key != last_logged:
            logger.warning(
                "Upstream not ready for %s on %s: "
                "same-day-missing=%s same-day-completed=%s "
                "prev-bday=%s prev-bday-missing=%s prev-bday-completed=%s — waiting",
                job_id, today,
                list(same_missing), deps["completed"],
                deps["prev_bday_date"],
                list(prev_missing), deps["prev_bday_completed"],
            )
            last_logged = log_key
        if time.time() >= deadline:
            logger.error(
                "Upstream wait budget (%ds) exhausted for %s on %s, "
                "same-day-missing=%s prev-bday-missing=%s — refusing to run",
                max_wait_sec, job_id, today,
                list(same_missing), list(prev_missing),
            )
            return deps
        time.sleep(poll_interval)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    today = datetime.now().strftime("%Y-%m-%d")

    if args.enforce_deps:
        deps = _wait_for_upstream(args.job_id, today, args.dep_wait_seconds)
        if not deps["ready"]:
            # Compose a detailed reason so the daily health dashboard can
            # distinguish "today's upstream pending" from "yesterday's
            # after-close failed" — the cross-day block is a different
            # operator workflow (rerun yesterday vs wait for today).
            same_missing = deps.get("missing", [])
            prev_missing = deps.get("prev_bday_missing", [])
            prev_date = deps.get("prev_bday_date", "")
            parts: list[str] = []
            if same_missing:
                parts.append(f"same-day missing={same_missing}")
            if prev_missing:
                parts.append(f"prev-bday={prev_date} missing={prev_missing}")
            reason = "; ".join(parts) or "unknown"
            details = (
                f"blocked: upstream wait budget {args.dep_wait_seconds}s "
                f"exhausted ({reason})"
            )
            mark_complete(args.job_id, today, success=False, details=details)
            # cx batch G P2 #4 (2026-06-07): push a WeChat alert on the
            # dep-block path so a quietly-blocked downstream surfaces
            # the same way an exception would. Catch any push exception
            # so an alerting failure does not itself fail the wrapper.
            _push_dep_block_alert(args.job_id, today, details)
            return 75  # EX_TEMPFAIL — semantically "try again later"

    def _run() -> None:
        result = subprocess.run(
            args.command,
            cwd=args.cwd,
            timeout=args.timeout or None,
        )
        if result.returncode != 0:
            raise RuntimeError(f"command failed with exit code {result.returncode}")

    try:
        run_with_status(args.job_id, _run)
    except Exception as e:
        mark_complete(args.job_id, today, success=False, details=str(e)[:200])
        return 1
    mark_complete(args.job_id, today, success=True, details="ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
