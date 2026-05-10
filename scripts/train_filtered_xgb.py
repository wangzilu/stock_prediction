"""Train XGB with Alpha158 + only STRONG factors (IC > 0.02)."""
import os, sys, json, time, logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parents[1].name == "MyProjects" else Path('/Users/wangzilu/MyProjects/stockPrediction')
sys.path.insert(0, str(PROJECT_ROOT))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from config.settings import PREDICTION_HORIZON_DAYS
from config.qlib_runtime import init_qlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"

# ONLY factors that passed single-factor IC test (STRONG + top OK)
STRONG_FACTORS = [
    ("($close - Min($close, 60)) / (Max($close, 60) - Min($close, 60) + 1e-8)", "price_pos60"),
    ("1.0 / If(Abs($pe) > 0.01, $pe, 1.0)", "ep"),
    ("($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20) + 1e-8)", "price_pos20"),
    ("$pb / Ref($pb, 5) - 1", "pb_mom5"),
]

def main():
    from qlib.utils import init_instance_by_config
    from qlib.contrib.eva.alpha import calc_ic
    from qlib.data import D

    init_qlib(QLIB_DATA)

    today = datetime.now()
    train_start = (today - timedelta(days=365*5)).strftime('%Y-%m-%d')
    train_end = (today - timedelta(days=90)).strftime('%Y-%m-%d')
    valid_start = (today - timedelta(days=89)).strftime('%Y-%m-%d')
    valid_end = (today - timedelta(days=30)).strftime('%Y-%m-%d')
    test_start = (today - timedelta(days=29)).strftime('%Y-%m-%d')
    test_end = today.strftime('%Y-%m-%d')

    logger.info(f"Train: {train_start}~{train_end}, Test: {test_start}~{test_end}")
    logger.info(f"Adding {len(STRONG_FACTORS)} STRONG factors to Alpha158")

    # Load Alpha158
    handler_config = {
        "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
        "kwargs": {"start_time": train_start, "end_time": test_end, "instruments": "all", "label": [LABEL_EXPR]},
    }
    dataset_config = {
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {"handler": handler_config, "segments": {
            "train": (train_start, train_end), "valid": (valid_start, valid_end), "test": (test_start, test_end)}},
    }

    logger.info("Loading Alpha158...")
    t0 = time.time()
    dataset = init_instance_by_config(dataset_config)
    logger.info(f"  Loaded: {time.time()-t0:.0f}s")

    # Prepare each segment with STRONG custom factors merged
    custom_exprs = [expr for expr, _ in STRONG_FACTORS]
    custom_names = [name for _, name in STRONG_FACTORS]

    segments = {}
    for seg in ["train", "valid", "test"]:
        logger.info(f"Preparing {seg}...")
        X = dataset.prepare(seg, col_set="feature")
        y = dataset.prepare(seg, col_set="label")
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        instruments = list(set(str(c) for c in X.index.get_level_values(1)))
        dates = sorted(X.index.get_level_values(0).unique())
        start_d, end_d = str(min(dates))[:10], str(max(dates))[:10]

        t1 = time.time()
        custom = D.features(instruments, custom_exprs, start_time=start_d, end_time=end_d)
        custom.columns = custom_names
        custom = custom.swaplevel().sort_index()
        custom = custom.reindex(X.index)
        nan_pct = custom.isna().mean().mean()
        logger.info(f"  Custom: {custom.shape[1]} cols, NaN={nan_pct:.2%}, {time.time()-t1:.0f}s")

        X_merged = np.hstack([X.values, custom.values]).astype(np.float32)
        y_np = y.values.astype(np.float32)
        mask = np.isfinite(y_np)
        X_merged, y_np = X_merged[mask], y_np[mask]

        segments[seg] = (X_merged, y_np)
        if seg == "test":
            test_idx = X.index[mask]
        logger.info(f"  {seg}: {X_merged.shape}")

    X_train, y_train = segments["train"]
    X_valid, y_valid = segments["valid"]
    X_test, y_test = segments["test"]
    n_feat = X_train.shape[1]
    logger.info(f"Total features: {n_feat} (Alpha158=158 + STRONG={len(STRONG_FACTORS)})")

    # Train XGB
    import xgboost as xgb
    logger.info("Training XGB...")
    t0 = time.time()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(
        {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
         "colsample_bytree": 0.8879, "reg_alpha": 205.7, "reg_lambda": 580.98,
         "objective": "reg:squarederror", "nthread": 4, "verbosity": 0},
        dtrain, num_boost_round=500,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=50, verbose_eval=100,
    )
    logger.info(f"  Trained: {time.time()-t0:.0f}s, best iter: {model.best_iteration}")

    # Evaluate
    pred = model.predict(dtest)
    pred_s = pd.Series(pred, index=test_idx, name="score")
    label_s = pd.Series(y_test, index=test_idx, name="label")
    mask = np.isfinite(pred) & np.isfinite(y_test)
    p, l = pred_s[mask], label_s[mask]

    ic, ric = calc_ic(p, l)
    df = pd.DataFrame({"pred": p, "label": l})
    spreads = [g.sort_values("pred", ascending=False).head(20)["label"].mean() -
               g.sort_values("pred", ascending=False).tail(20)["label"].mean()
               for _, g in df.groupby(level=0) if len(g) >= 40]

    ic_val = float(ic.mean())
    ric_val = float(ric.mean())
    spread_val = float(np.mean(spreads)) if spreads else 0

    logger.info("=" * 60)
    logger.info(f"FILTERED XGB: {n_feat} features (158 + {len(STRONG_FACTORS)} STRONG)")
    logger.info(f"  IC:       {ic_val:+.4f}  (baseline: +0.024)")
    logger.info(f"  ICIR:     {ic_val/(float(ic.std())+1e-8):.3f}")
    logger.info(f"  RankIC:   {ric_val:+.4f}  (baseline: +0.015)")
    logger.info(f"  Spread:   {spread_val*100:+.3f}%  (baseline: +7.1%)")
    logger.info(f"  DELTA IC: {(ic_val-0.024)*100:+.2f} bps")
    logger.info("=" * 60)

    results = {"model": "xgb_filtered", "features": n_feat, "strong_factors": custom_names,
               "ic_mean": round(ic_val, 6), "icir": round(ic_val/(float(ic.std())+1e-8), 4),
               "rank_ic_mean": round(ric_val, 6), "top20_spread": round(spread_val, 6)}
    with open(str(DATA_DIR / "xgb_filtered_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {results}")

if __name__ == "__main__":
    main()
