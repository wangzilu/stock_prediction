"""Train XGB with Alpha158 + PE/PB/Turn custom factors (~170 features).

Adds valuation and turnover factors from existing Qlib bins ($pe, $pb, $turn, $amount)
that Alpha158 doesn't use. Logs every step with timing.

Usage:
    python scripts/train_enhanced_xgb.py
"""
import logging
import os
import json
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"

CUSTOM_EXPRS = [
    "$pe", "$pb", "$turn", "$amount",
    "$pe / Ref($pe, 20) - 1",
    "$pb / Ref($pb, 20) - 1",
    "$turn / Mean($turn, 20)",
    "$turn / Mean($turn, 60)",
    "$amount / Mean($amount, 20)",
    "Std($turn, 20)",
    "1.0 / $pe",
    "1.0 / $pb",
    "($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20) + 1e-8)",
]
CUSTOM_NAMES = [
    "pe", "pb", "turn_raw", "amount_raw",
    "pe_mom20", "pb_mom20", "turn_anom20", "turn_anom60",
    "amount_anom20", "turn_vol20", "ep", "bp", "price_pos20",
]


def main():
    from qlib.utils import init_instance_by_config
    from qlib.contrib.eva.alpha import calc_ic
    from qlib.data import D

    logger.info("Step 0: Init Qlib")
    t0 = time.time()
    init_qlib(QLIB_DATA)
    logger.info(f"  Qlib init: {time.time()-t0:.1f}s")

    today = datetime.now()
    train_start = (today - timedelta(days=365 * 5)).strftime("%Y-%m-%d")  # 5 years full history
    train_end = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=89)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")
    logger.info(f"  Dates: train {train_start}~{train_end}, test {test_start}~{test_end}")

    # Step 1: Train standard Alpha158 XGB first, get predictions
    # Then fetch PE/PB/Turn as separate features and train a SECOND model
    # that combines Alpha158 score + fundamental features
    # This avoids the Qlib handler compatibility issue entirely
    logger.info("Step 1: Loading Alpha158 dataset (standard)...")
    t1 = time.time()

    handler_config = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": train_start,
            "end_time": test_end,
            "instruments": "all",
            "label": [LABEL_EXPR],
        },
    }
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (test_start, test_end),
            },
        },
    }
    dataset = init_instance_by_config(dataset_config)
    logger.info(f"  Alpha158 loaded: {time.time()-t1:.1f}s")

    # Step 2: Get Alpha158 features + custom features for each segment
    segments_data = {}
    for seg in ["train", "valid", "test"]:
        logger.info(f"Step 2: Preparing {seg}...")
        t2 = time.time()
        X = dataset.prepare(seg, col_set="feature")
        y = dataset.prepare(seg, col_set="label")
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        logger.info(f"  Alpha158 {seg}: {X.shape}, {time.time()-t2:.1f}s")

        # Fetch custom features using the SAME index as Alpha158
        logger.info(f"  Fetching custom features for {seg}...")
        t3 = time.time()
        instruments = list(set(str(c) for c in X.index.get_level_values(1)))
        dates = sorted(X.index.get_level_values(0).unique())
        start_d = str(min(dates))[:10]
        end_d = str(max(dates))[:10]

        custom = D.features(instruments, CUSTOM_EXPRS, start_time=start_d, end_time=end_d)
        logger.info(f"  D.features: {custom.shape if custom is not None else 'None'}, {time.time()-t3:.1f}s")

        if custom is not None and not custom.empty:
            custom.columns = CUSTOM_NAMES
            # D.features returns (instrument, datetime) but Alpha158 is (datetime, instrument)
            # Swap custom index to match Alpha158
            custom = custom.swaplevel()
            custom = custom.sort_index()

            C_vals = custom.reindex(X.index).values
            nan_ratio = np.isnan(C_vals).mean()
            logger.info(f"  Custom NaN after swaplevel+reindex: {nan_ratio:.2%}")

            X_merged = np.hstack([X.values, C_vals]).astype(np.float32)
        else:
            X_merged = X.values.astype(np.float32)

        y_np = y.values.astype(np.float32)
        label_mask = np.isfinite(y_np)
        X_merged = X_merged[label_mask]
        y_np = y_np[label_mask]

        segments_data[seg] = (X_merged, y_np)
        if seg == "test":
            test_idx = X.index[label_mask]

        logger.info(f"  Final {seg}: {X_merged.shape}")

    X_train, y_train = segments_data["train"]
    X_valid, y_valid = segments_data["valid"]
    X_test, y_test = segments_data["test"]
    n_feat = X_train.shape[1]
    logger.info(f"Total features: {n_feat}")

    # Step 4: Train XGB
    logger.info("Step 4: Training XGB...")
    t4 = time.time()
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "max_depth": 8, "learning_rate": 0.05,
        "subsample": 0.8789, "colsample_bytree": 0.8879,
        "reg_alpha": 205.6999, "reg_lambda": 580.9768,
        "objective": "reg:squarederror", "nthread": 4, "verbosity": 0,
    }

    model = xgb.train(
        params, dtrain, num_boost_round=500,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=50, verbose_eval=50,
    )
    logger.info(f"  XGB trained: {time.time()-t4:.1f}s, best iter: {model.best_iteration}")

    # Step 5: Evaluate
    logger.info("Step 5: Evaluating...")
    pred_raw = model.predict(dtest)
    pred_s = pd.Series(pred_raw, index=test_idx, name="score")
    label_s = pd.Series(y_test, index=test_idx, name="label")

    mask = np.isfinite(pred_raw) & np.isfinite(y_test)
    p, l = pred_s[mask], label_s[mask]

    ic, ric = calc_ic(p, l)

    df = pd.DataFrame({"pred": p, "label": l})
    spreads = []
    for d, g in df.groupby(level=0):
        if len(g) < 40:
            continue
        s = g.sort_values("pred", ascending=False)
        spreads.append(s.head(20)["label"].mean() - s.tail(20)["label"].mean())

    ic_val = float(ic.mean())
    ric_val = float(ric.mean())
    spread_val = float(np.mean(spreads)) if spreads else 0

    logger.info("=" * 60)
    logger.info(f"ENHANCED XGB: {n_feat} features")
    logger.info(f"  IC:       {ic_val:+.4f}")
    logger.info(f"  ICIR:     {ic_val / (float(ic.std()) + 1e-8):.3f}")
    logger.info(f"  RankIC:   {ric_val:+.4f}")
    logger.info(f"  Spread:   {spread_val * 100:+.3f}%")
    logger.info(f"  BASELINE: IC=+0.024, Spread=+7.1%")
    logger.info(f"  DELTA IC: {(ic_val - 0.024) * 100:+.2f} bps")
    logger.info("=" * 60)

    # Save
    model.save_model(str(DATA_DIR / "xgb_enhanced_model.json"))
    results = {
        "model": "xgb_enhanced",
        "features": n_feat,
        "ic_mean": round(ic_val, 6),
        "icir": round(ic_val / (float(ic.std()) + 1e-8), 4),
        "rank_ic_mean": round(ric_val, 6),
        "top20_spread": round(spread_val, 6),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(str(DATA_DIR / "xgb_enhanced_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {results}")


if __name__ == "__main__":
    main()
