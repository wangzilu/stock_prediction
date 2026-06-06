"""Fetch weekly fundamental_features.parquet and write source health.

This script wraps ``data.collectors.fundamental.FundamentalCollector`` so
the production ``fundamental`` feature group has its own health source
instead of piggybacking on qlib_data_update.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from data.collectors.fundamental import DATA_DIR, FUNDAMENTAL_PATH, FundamentalCollector
from scheduler.data_health import HealthStatus, write_health


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=500,
                        help="Number of large-cap stocks for slower quality fields")
    args = parser.parse_args()

    try:
        collector = FundamentalCollector()
        df = collector.fetch_all(top_n=args.top_n)
        if df.empty:
            write_health("fundamental_update", HealthStatus(
                success=False,
                error_type="NoData",
                error_message="FundamentalCollector returned empty data",
                network_profile="domestic",
            ))
            return 1

        collector.save(df, FUNDAMENTAL_PATH)
        latest_date = str(df["date"].max()) if "date" in df.columns else ""
        coverage = 0.0
        if "qlib_code" in df.columns:
            features_dir = DATA_DIR / "qlib_data" / "cn_data" / "features"
            expected = 0
            if features_dir.exists():
                expected = sum(1 for p in features_dir.iterdir() if p.is_dir())
            coverage = float(df["qlib_code"].nunique()) / max(expected or len(df), 1)
        write_health("fundamental_update", HealthStatus(
            success=True,
            n_items=len(df),
            latest_date=latest_date,
            coverage=coverage,
            network_profile="domestic",
        ))
        return 0
    except Exception as exc:
        write_health("fundamental_update", HealthStatus(
            success=False,
            error_type=type(exc).__name__,
            error_message=str(exc)[:200],
            network_profile="domestic",
        ))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
