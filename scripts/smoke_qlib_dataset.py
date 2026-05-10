"""Smoke-test Qlib Alpha158 dataset loading without training a model.

This must run from a real script file on macOS because Qlib/joblib workers
cannot safely spawn from `python -` or stdin.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qlib.utils import init_instance_by_config  # noqa: E402

from config.qlib_runtime import init_qlib  # noqa: E402
from config.settings import LGB_INFERENCE_UNIVERSE, QLIB_PROVIDER_URI  # noqa: E402


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", default=QLIB_PROVIDER_URI)
    parser.add_argument("--universe", default=LGB_INFERENCE_UNIVERSE)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--joblib-backend", default=None)
    parser.add_argument("--kernels", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    today = datetime.now()
    start_time = (today - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")
    end_time = today.strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=29)).strftime("%Y-%m-%d")

    print(f"Initializing Qlib: {args.provider_uri}")
    init_qlib(args.provider_uri, joblib_backend=args.joblib_backend, kernels=args.kernels)

    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": start_time,
                    "end_time": end_time,
                    "instruments": args.universe,
                },
            },
            "segments": {
                "test": (test_start, end_time),
            },
        },
    }

    print(f"Loading Alpha158 dataset: universe={args.universe} {start_time} ~ {end_time}")
    dataset = init_instance_by_config(dataset_config)
    test = dataset.prepare("test")
    print(f"Dataset load OK: test_shape={getattr(test, 'shape', None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
