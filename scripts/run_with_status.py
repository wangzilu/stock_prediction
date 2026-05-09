"""Run a scheduled command and persist status to data/storage/job_status.json.

Usage:
    python scripts/run_with_status.py --job-id morning -- python main.py --morning
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scheduler.job_status import run_with_status  # noqa: E402


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--cwd", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--timeout", type=int, default=0, help="Timeout in seconds; 0 disables timeout")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("command is required after --")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

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
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
