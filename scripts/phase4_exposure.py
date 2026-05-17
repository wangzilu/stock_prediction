"""Phase 4C: Run exposure analysis on XGB 174 predictions.

Usage:
    python scripts/phase4_exposure.py
"""
import json
import logging
import os
import sys
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
from models.feature_pipeline import prepare_features_174, train_xgb, load_daily_returns
from backtest.exposure_report import ExposureAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"


def main():
    import xgboost as xgb
    from qlib.utils import init_instance_by_config

    init_qlib(QLIB_DATA)
    merger = FeatureMerger(DATA_DIR)

    today = datetime.now()
    test_end = today.strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=180)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=181)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=241)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=242)).strftime("%Y-%m-%d")
    train_start = (today - timedelta(days=365 * 3 + 242)).strftime("%Y-%m-%d")

    logger.info(f"=== Phase 4C: Exposure Analysis ===")
    logger.info(f"Test: {test_start}~{test_end}")

    dataset = init_instance_by_config({
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
                "kwargs": {"start_time": train_start, "end_time": test_end,
                           "instruments": "all", "label": [LABEL_EXPR]},
            },
            "segments": {
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (test_start, test_end),
            },
        },
    })

    # Prepare features and train
    logger.info("Preparing features...")
    X_train_df, y_train_s = prepare_features_174(dataset, "train", merger)
    X_valid_df, y_valid_s = prepare_features_174(dataset, "valid", merger)
    X_test_df, y_test_s = prepare_features_174(dataset, "test", merger)

    y_train = y_train_s.values.astype(np.float32)
    mask_train = np.isfinite(y_train)
    y_valid = y_valid_s.values.astype(np.float32)
    mask_valid = np.isfinite(y_valid)

    logger.info("Training XGB...")
    model = train_xgb(
        X_train_df.values.astype(np.float32)[mask_train], y_train[mask_train],
        X_valid_df.values.astype(np.float32)[mask_valid], y_valid[mask_valid])

    # Predict
    pred_raw = model.predict(xgb.DMatrix(X_test_df.values.astype(np.float32)))
    predictions = pd.Series(pred_raw, index=X_test_df.index, name="score")
    predictions = predictions[np.isfinite(predictions)]

    # Load daily returns
    logger.info("Loading daily returns...")
    daily_returns = load_daily_returns(X_test_df.index)

    # Run exposure analysis
    logger.info("Running exposure analysis...")
    analyzer = ExposureAnalyzer()
    report = analyzer.analyze(predictions, daily_returns, top_k=20, buffer=5)
    analyzer.print_report(report)

    # Save
    out_path = DATA_DIR / "phase4" / "exposure_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert non-serializable types
    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(str(out_path), "w") as f:
        json.dump(report, f, indent=2, default=_serialize, ensure_ascii=False)
    logger.info(f"Saved: {out_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
