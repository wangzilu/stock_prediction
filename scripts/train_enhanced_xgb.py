"""Train XGB with Alpha158 + FeatureMerger supplementary factors (~178+ features).

Combines Alpha158 (158 dims) with valuation, capital flow, macro, shareholder,
northbound, and quality factors from downloaded parquet files.

Usage:
    python scripts/train_enhanced_xgb.py
    python scripts/train_enhanced_xgb.py --no-qlib-custom   # skip Qlib bin factors
"""
import argparse
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
from models.feature_merger import FeatureMerger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"

# Qlib bin custom factors (PE/PB/Turn that Alpha158 doesn't use)
CUSTOM_EXPRS = [
    "$pe", "$pb", "$turn", "$amount",
    "$pe / Ref($pe, 20) - 1",
    "$pb / Ref($pb, 20) - 1",
    "$turn / Mean($turn, 20)",
    "$turn / Mean($turn, 60)",
    "$amount / Mean($amount, 20)",
    "Std($turn, 20)",
    "1.0 / If(Abs($pe) > 0.01, $pe, 1.0)",
    "1.0 / If(Abs($pb) > 0.01, $pb, 1.0)",
    "($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20) + 1e-8)",
]
CUSTOM_NAMES = [
    "pe", "pb", "turn_raw", "amount_raw",
    "pe_mom20", "pb_mom20", "turn_anom20", "turn_anom60",
    "amount_anom20", "turn_vol20", "ep", "bp", "price_pos20",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-qlib-custom", action="store_true",
                        help="Skip Qlib bin custom factors, use only FeatureMerger")
    args = parser.parse_args()

    from qlib.utils import init_instance_by_config
    from qlib.contrib.eva.alpha import calc_ic
    from qlib.data import D

    logger.info("Step 0: Init Qlib")
    t0 = time.time()
    init_qlib(QLIB_DATA)
    logger.info(f"  Qlib init: {time.time()-t0:.1f}s")

    today = datetime.now()
    train_start = (today - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=89)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")
    logger.info(f"  Dates: train {train_start}~{train_end}, test {test_start}~{test_end}")

    # Step 1: Load Alpha158 dataset
    logger.info("Step 1: Loading Alpha158 dataset...")
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

    # Step 2: Prepare features for each segment
    merger = FeatureMerger(DATA_DIR)
    segments_data = {}
    test_idx = None

    for seg in ["train", "valid", "test"]:
        logger.info(f"Step 2: Preparing {seg}...")
        t2 = time.time()

        X = dataset.prepare(seg, col_set="feature")
        y = dataset.prepare(seg, col_set="label")
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        logger.info(f"  Alpha158 {seg}: {X.shape}")

        # 2a. FeatureMerger: add parquet-based supplementary factors
        # research/enhanced-xgb script — explicit opt-in to "load every
        # loader" (P0-e: production gate via PRODUCTION_SUPPLEMENTARY_GROUPS)
        from config.production_features import RESEARCH_ALL_LOADERS
        supp = merger._load_supplementary(X.index,
                                          groups=RESEARCH_ALL_LOADERS)
        if supp is not None and not supp.empty:
            logger.info(f"  FeatureMerger: +{supp.shape[1]} supplementary columns")
            X = X.join(supp, how="left")

        # 2b. Qlib bin custom factors (PE/PB/Turn expressions)
        if not args.no_qlib_custom:
            instruments = list(set(str(c) for c in X.index.get_level_values(1)))
            dates = sorted(X.index.get_level_values(0).unique())
            start_d = str(min(dates))[:10]
            end_d = str(max(dates))[:10]

            custom = D.features(instruments, CUSTOM_EXPRS,
                                start_time=start_d, end_time=end_d)
            if custom is not None and not custom.empty:
                custom.columns = CUSTOM_NAMES
                custom = custom.swaplevel().sort_index().reindex(X.index)
                custom = custom.replace([np.inf, -np.inf], np.nan)
                # Remove columns that overlap with FeatureMerger (avoid duplicates)
                existing = set(X.columns)
                new_cols = [c for c in custom.columns if c not in existing]
                if new_cols:
                    X = X.join(custom[new_cols], how="left")
                    logger.info(f"  Qlib custom: +{len(new_cols)} columns")

        # Convert to numpy
        X_np = X.values.astype(np.float32)
        y_np = y.values.astype(np.float32)
        label_mask = np.isfinite(y_np)
        X_np = X_np[label_mask]
        y_np = y_np[label_mask]

        segments_data[seg] = (X_np, y_np)
        if seg == "test":
            test_idx = X.index[label_mask]
            feature_names = list(X.columns)

        logger.info(f"  Final {seg}: {X_np.shape}, {time.time()-t2:.1f}s")

    X_train, y_train = segments_data["train"]
    X_valid, y_valid = segments_data["valid"]
    X_test, y_test = segments_data["test"]
    n_feat = X_train.shape[1]
    logger.info(f"Total features: {n_feat}")
    logger.info(f"  Alpha158: 158, Supplementary: {n_feat - 158}")

    # Step 3: Train XGB
    logger.info("Step 3: Training XGB...")
    t3 = time.time()
    import xgboost as xgb

    fnames = feature_names[:n_feat] if len(feature_names) == n_feat else None
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=fnames)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=fnames)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=fnames)

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
    logger.info(f"  XGB trained: {time.time()-t3:.1f}s, best iter: {model.best_iteration}")

    # Step 4: Evaluate
    logger.info("Step 4: Evaluating...")
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
    logger.info(f"  Label:    {LABEL_EXPR}")
    logger.info(f"  Test:     {test_start} ~ {test_end}")
    logger.info("=" * 60)

    # Feature importance top 20
    imp = model.get_score(importance_type="gain")
    if imp:
        sorted_imp = sorted(imp.items(), key=lambda x: -x[1])[:20]
        logger.info("Top 20 feature importance:")
        for fname, score in sorted_imp:
            logger.info(f"  {fname}: {score:.1f}")

    # Save
    model.save_model(str(DATA_DIR / "xgb_enhanced_model.json"))
    results = {
        "model": "xgb_enhanced",
        "features": n_feat,
        "feature_breakdown": {"alpha158": 158, "supplementary": n_feat - 158},
        "ic_mean": round(ic_val, 6),
        "icir": round(ic_val / (float(ic.std()) + 1e-8), 4),
        "rank_ic_mean": round(ric_val, 6),
        "top20_spread": round(spread_val, 6),
        "label": LABEL_EXPR,
        "test_period": f"{test_start} ~ {test_end}",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(str(DATA_DIR / "xgb_enhanced_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {results}")


if __name__ == "__main__":
    main()
