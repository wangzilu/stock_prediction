"""Pre-compute full feature matrix and save as parquet cache.

Runs Alpha158 + flow + custom + holder + cross-market regime once,
saves to data/storage/feature_cache_174.parquet (or 174+regime).

Subsequent experiments just read_parquet + slice by date = seconds not hours.

Usage:
    python scripts/build_feature_cache.py
    python scripts/build_feature_cache.py --include-regime --include-holder
    python scripts/build_feature_cache.py --start 2021-01-01 --end 2026-05-18
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config.settings import PREDICTION_HORIZON_DAYS
from config.qlib_runtime import init_qlib
from models.feature_merger import FeatureMerger
from models.feature_pipeline import prepare_features_174

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"
DAILY_RET_EXPR = "Ref($close, -1) / $close - 1"

# MA expressions for timing backtest
MA_EXPRS = ["$close", "Mean($close, 5)", "Mean($close, 20)"]
MA_NAMES = ["_close", "_ma5", "_ma20"]


def main():
    from qlib.utils import init_instance_by_config
    from qlib.data import D

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--include-regime", action="store_true",
                        help="Include cross-market regime features (HSI/HSTECH/NASDAQ)")
    parser.add_argument("--include-holder", action="store_true",
                        help="Include holder_num")
    parser.add_argument("--include-ma", action="store_true",
                        help="Include MA5/MA20/close for timing backtest")
    parser.add_argument("--all", action="store_true",
                        help="Include everything")
    args = parser.parse_args()

    if args.all:
        args.include_regime = True
        args.include_holder = True
        args.include_ma = True

    init_qlib(QLIB_DATA)
    merger = FeatureMerger(DATA_DIR)

    logger.info(f"=== Building Feature Cache ===")
    logger.info(f"Period: {args.start} ~ {args.end}")
    logger.info(f"Include: regime={args.include_regime}, holder={args.include_holder}, ma={args.include_ma}")

    # Load full dataset (one big segment)
    t0 = time.time()
    dataset = init_instance_by_config({
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
                "kwargs": {"start_time": args.start, "end_time": args.end,
                           "instruments": "all",
                           "label": [LABEL_EXPR]},
            },
            "segments": {"full": (args.start, args.end)},
        },
    })
    logger.info(f"Alpha158 loaded: {time.time()-t0:.1f}s")

    # Prepare 174 features
    t1 = time.time()
    X, y = prepare_features_174(dataset, "full", merger, include_holder=args.include_holder)
    logger.info(f"174 features prepared: {X.shape}, {time.time()-t1:.1f}s")

    # Add cross-market regime
    if args.include_regime:
        t2 = time.time()
        cross_mkt = merger._load_cross_market_regime(X.index)
        if cross_mkt is not None and not cross_mkt.empty:
            X = X.join(cross_mkt, how="left")
            logger.info(f"Cross-market regime added: +{cross_mkt.shape[1]} features, {time.time()-t2:.1f}s")
        else:
            logger.warning("Cross-market regime not available")

    logger.info(f"Final features: {X.shape}")

    # Save features + label
    t3 = time.time()
    cache = X.copy()
    cache["__label_5d"] = y.values

    # Add daily return for backtest
    logger.info("Loading daily returns...")
    insts = sorted(set(str(c) for c in X.index.get_level_values(1)))
    dates = sorted(X.index.get_level_values(0).unique())
    ret = D.features(insts, [DAILY_RET_EXPR],
                     start_time=str(min(dates))[:10],
                     end_time=str(max(dates))[:10])
    if ret is not None and not ret.empty:
        ret.columns = ["__pnl_return_1d"]
        ret = ret.swaplevel().sort_index()
        ret = ret.replace([np.inf, -np.inf], np.nan)
        cache = cache.join(ret, how="left")
        logger.info(f"Daily returns added: {ret.notna().sum().iloc[0]} non-null")

    # Add MA data for timing backtest
    if args.include_ma:
        logger.info("Loading MA data...")
        ma = D.features(insts, MA_EXPRS,
                        start_time=str(min(dates))[:10],
                        end_time=str(max(dates))[:10])
        if ma is not None and not ma.empty:
            ma.columns = MA_NAMES
            ma = ma.swaplevel().sort_index()
            ma = ma.replace([np.inf, -np.inf], np.nan)
            cache = cache.join(ma, how="left")
            logger.info(f"MA data added: {len(MA_NAMES)} columns")

    # Save
    suffix_parts = ["174"]
    if args.include_holder:
        suffix_parts.append("holder")
    if args.include_regime:
        suffix_parts.append("regime")
    if args.include_ma:
        suffix_parts.append("ma")
    suffix = "_".join(suffix_parts)

    out_path = DATA_DIR / f"feature_cache_{suffix}.parquet"
    cache.to_parquet(str(out_path))
    size_mb = out_path.stat().st_size / 1024 / 1024

    logger.info(f"\nSaved: {out_path}")
    logger.info(f"  Shape: {cache.shape}")
    logger.info(f"  Size: {size_mb:.1f} MB")
    logger.info(f"  Index: {cache.index.names}")
    logger.info(f"  Columns: {len(cache.columns)} total")
    logger.info(f"    Features: {len([c for c in cache.columns if not c.startswith('__')])}")
    logger.info(f"    Labels: {[c for c in cache.columns if c.startswith('__')]}")
    logger.info(f"  Date range: {str(min(dates))[:10]} ~ {str(max(dates))[:10]}")
    logger.info(f"  Stocks: {len(insts)}")
    logger.info(f"  Total time: {time.time()-t0:.1f}s")
    logger.info("Done!")


if __name__ == "__main__":
    main()
