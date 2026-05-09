"""Check whether the scheduler will use production LGB predictions.

Run this from a real script file instead of stdin so Qlib/joblib
multiprocessing can spawn safely on macOS.

Usage:
    python scripts/check_scheduler_lgb_status.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def check_status() -> dict:
    from scheduler.jobs import DailyPipeline

    pipeline = DailyPipeline()
    preds = pipeline._load_lgb_predictions()
    status = dict(getattr(pipeline, "_lgb_status", {}))
    finite_count = len(preds)
    status.setdefault("count", finite_count)
    status["used_by_scheduler"] = status.get("status") == "ok" and finite_count > 0
    status["top_examples"] = sorted(
        preds.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    return status


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    status = check_status()
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
    else:
        print(
            "Scheduler LGB: "
            f"status={status.get('status')} "
            f"count={status.get('count')} "
            f"min={status.get('min_required')} "
            f"used={status.get('used_by_scheduler')}"
        )
        if status.get("error"):
            print(f"Reason: {status['error']}")

    return 0 if status.get("used_by_scheduler") else 1


if __name__ == "__main__":
    raise SystemExit(main())
