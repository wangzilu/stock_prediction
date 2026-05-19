"""Build Alpha360 feature cache for fast rolling experiments.

Alpha360: 60-day raw OHLCV sequence (360 dims) — different from Alpha158's
hand-crafted technical indicators.

Usage:
    python scripts/build_alpha360_cache.py
"""
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config.settings import PREDICTION_HORIZON_DAYS
from config.qlib_runtime import init_qlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"
DAILY_RET_EXPR = "Ref($close, -1) / $close - 1"


def main():
    from qlib.utils import init_instance_by_config
    from qlib.data import D

    start = "2021-01-01"
    end = datetime.now().strftime("%Y-%m-%d")
    fit_end = "2026-01-01"  # fit on bulk of data

    init_qlib(QLIB_DATA)

    logger.info(f"=== Building Alpha360 Cache ===")
    logger.info(f"Period: {start} ~ {end}")

    t0 = time.time()
    dataset = init_instance_by_config({
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha360", "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": start, "end_time": end,
                    "fit_start_time": start, "fit_end_time": fit_end,
                    "instruments": "all",
                    "label": [LABEL_EXPR],
                },
            },
            "segments": {"full": (start, end)},
        },
    })
    logger.info(f"Alpha360 loaded: {time.time()-t0:.1f}s")

    # Extract features and label
    X = dataset.prepare("full", col_set="feature")
    y = dataset.prepare("full", col_set="label")
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    logger.info(f"Features: {X.shape}")

    # Build cache
    cache = X.copy()
    cache["__label_5d"] = y.values

    # Add daily return
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
        logger.info(f"Daily returns added")

    # Save
    out_path = DATA_DIR / "feature_cache_alpha360.parquet"
    t1 = time.time()
    cache.to_parquet(str(out_path))
    size_mb = out_path.stat().st_size / 1024 / 1024

    logger.info(f"\nSaved: {out_path}")
    logger.info(f"  Shape: {cache.shape}")
    logger.info(f"  Size: {size_mb:.1f} MB")
    logger.info(f"  Features: {len([c for c in cache.columns if not c.startswith('__')])}")
    logger.info(f"  Total time: {time.time()-t0:.1f}s")
    logger.info("Done!")


if __name__ == "__main__":
    main()
