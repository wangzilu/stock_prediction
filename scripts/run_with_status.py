"""Run a scheduled command and persist status to data/storage/job_status.json.

Usage:
    python scripts/run_with_status.py --job-id morning -- python main.py --morning

Cron-time upstream gate: if --enforce-deps is set, looks up the job's upstream
dependencies in scheduler.job_deps.JOB_DEPS and refuses to run when any
upstream hasn't successfully completed for today. Exits with code 75
(EX_TEMPFAIL) so cron retries later or the next-day pass picks it up cleanly.

mark_complete is called on every exit path so check_upstream sees a clean
DAG state — previously the per-job status files in data/storage/job_status/
were never written by this wrapper, so check_upstream always returned ready
when the daily files were missing.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scheduler.job_deps import check_upstream, mark_complete  # noqa: E402
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
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("command is required after --")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    today = datetime.now().strftime("%Y-%m-%d")

    if args.enforce_deps:
        deps = check_upstream(args.job_id, today)
        if not deps["ready"]:
            logger.error(
                "Upstream gate blocked %s on %s: missing=%s completed=%s",
                args.job_id, today, deps["missing"], deps["completed"],
            )
            return 75  # EX_TEMPFAIL — cron will retry next minute

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
