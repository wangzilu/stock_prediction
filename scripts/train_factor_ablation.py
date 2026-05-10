"""Factor ablation experiment: compare base vs enhanced with proper controls.

Runs 6+ groups in one script, same split, same seed, same label:
1. base_raw (Alpha158 only)
2. base + custom_raw
3. base + custom_winsor_zscore
4. base + custom_rank
5. base + shuffled_custom (NEGATIVE CONTROL)
6. base + each_one_factor (individual ablation)

Usage:
    python scripts/train_factor_ablation.py
    python scripts/train_factor_ablation.py --universe csi300 --json
"""
import argparse
import json
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

from config.qlib_runtime import init_qlib
from config.settings import PREDICTION_HORIZON_DAYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
ABLATION_PATH = DATA_DIR / "factor_ablation_results.json"
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"
SEED = 42

CUSTOM_EXPRS = [
    "($close - Min($close, 60)) / (Max($close, 60) - Min($close, 60) + 1e-8)",
    "1.0 / If(Abs($pe) > 0.01, $pe, 1.0)",
    "($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20) + 1e-8)",
    "$pb / Ref($pb, 5) - 1",
]
CUSTOM_NAMES = ["price_pos60", "ep", "price_pos20", "pb_mom5"]


def winsorize_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional winsorize + zscore per day (proper preprocessing)."""
    result = df.copy()
    for col in result.columns:
        grouped = result[col].groupby(level=0)
        # Winsorize at 1st/99th percentile per day
        p01 = grouped.transform(lambda x: x.quantile(0.01))
        p99 = grouped.transform(lambda x: x.quantile(0.99))
        result[col] = result[col].clip(lower=p01, upper=p99)
        # Z-score per day
        mean = grouped.transform("mean")
        std = grouped.transform("std")
        result[col] = (result[col] - mean) / (std + 1e-8)
        result[col] = result[col].clip(-3, 3)
    return result


def rank_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank transform per day."""
    return df.groupby(level=0).rank(pct=True)


def shuffle_columns(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Shuffle each column independently (breaks true signal, keeps distribution)."""
    rng = np.random.RandomState(seed + 999)
    result = df.copy()
    for col in result.columns:
        vals = result[col].values.copy()
        rng.shuffle(vals)
        result[col] = vals
    return result


def evaluate_group(X_train, y_train, X_valid, y_valid, X_test, y_test, test_idx, name):
    """Train XGB and evaluate one experiment group."""
    import xgboost as xgb
    from qlib.contrib.eva.alpha import calc_ic

    # Clean label NaN
    train_mask = np.isfinite(y_train)
    valid_mask = np.isfinite(y_valid)
    test_mask = np.isfinite(y_test)

    dtrain = xgb.DMatrix(X_train[train_mask], label=y_train[train_mask])
    dvalid = xgb.DMatrix(X_valid[valid_mask], label=y_valid[valid_mask])
    dtest = xgb.DMatrix(X_test[test_mask], label=y_test[test_mask])

    params = {
        "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
        "colsample_bytree": 0.8879, "reg_alpha": 205.7, "reg_lambda": 580.98,
        "objective": "reg:squarederror", "nthread": 4, "verbosity": 0,
        "seed": SEED,
    }

    t0 = time.time()
    model = xgb.train(
        params, dtrain, num_boost_round=300,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=30, verbose_eval=0,
    )
    train_time = time.time() - t0

    pred = model.predict(dtest)
    tidx = test_idx[test_mask]
    pred_s = pd.Series(pred, index=tidx, name="score")
    label_s = pd.Series(y_test[test_mask], index=tidx, name="label")

    mask = np.isfinite(pred) & np.isfinite(y_test[test_mask])
    p, l = pred_s[mask], label_s[mask]

    if len(p) < 100:
        return {"name": name, "error": f"Too few samples: {len(p)}"}

    ic, ric = calc_ic(p, l)

    df = pd.DataFrame({"pred": p, "label": l})
    spreads = []
    for d, g in df.groupby(level=0):
        if len(g) < 40:
            continue
        s = g.sort_values("pred", ascending=False)
        spreads.append(s.head(20)["label"].mean() - s.tail(20)["label"].mean())

    return {
        "name": name,
        "n_features": int(X_train.shape[1]),
        "ic_mean": round(float(ic.mean()), 6),
        "icir": round(float(ic.mean() / (ic.std() + 1e-8)), 4),
        "rank_ic_mean": round(float(ric.mean()), 6),
        "rank_ic_pos": round(float((ric > 0).mean()), 4),
        "top20_spread": round(float(np.mean(spreads)) if spreads else 0, 6),
        "spread_pos": round(float(np.mean([s > 0 for s in spreads])) if spreads else 0, 4),
        "train_time_s": round(train_time, 1),
        "best_iter": model.best_iteration,
        "n_samples": len(p),
    }


def main():
    parser = argparse.ArgumentParser(description="Factor ablation experiment")
    parser.add_argument("--universe", default="csi300")
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    from qlib.utils import init_instance_by_config
    from qlib.data import D

    init_qlib(QLIB_DATA)

    today = datetime.now()
    train_start = (today - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=89)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=args.test_days - 1)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")

    logger.info(f"Universe: {args.universe}, Train: {train_start}~{train_end}, Test: {test_start}~{test_end}")

    # Load Alpha158 base dataset
    logger.info("Loading Alpha158 base...")
    dataset = init_instance_by_config({
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
                "kwargs": {"start_time": train_start, "end_time": test_end,
                           "instruments": args.universe, "label": [LABEL_EXPR]},
            },
            "segments": {
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (test_start, test_end),
            },
        },
    })

    # Prepare base features for each segment
    segments = {}
    for seg in ["train", "valid", "test"]:
        X = dataset.prepare(seg, col_set="feature")
        y = dataset.prepare(seg, col_set="label")
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        # Fetch custom factors
        instruments = list(set(str(c) for c in X.index.get_level_values(1)))
        dates = sorted(X.index.get_level_values(0).unique())
        custom = D.features(instruments, CUSTOM_EXPRS,
                           start_time=str(min(dates))[:10],
                           end_time=str(max(dates))[:10])
        custom.columns = CUSTOM_NAMES
        custom = custom.swaplevel().sort_index().reindex(X.index)

        segments[seg] = {"X_base": X, "y": y, "custom": custom, "index": X.index}
        logger.info(f"  {seg}: base={X.shape}, custom NaN={custom.isna().mean().mean():.2%}")

    # Define experiment groups
    groups = []

    # Group 1: base only
    groups.append(("base_raw", lambda s: s["X_base"].values))

    # Group 2: base + custom raw (no preprocessing)
    groups.append(("base+custom_raw", lambda s: np.hstack([s["X_base"].values, s["custom"].values])))

    # Group 3: base + custom winsorize+zscore
    groups.append(("base+custom_wz", lambda s: np.hstack([
        s["X_base"].values, winsorize_zscore(s["custom"]).values])))

    # Group 4: base + custom rank
    groups.append(("base+custom_rank", lambda s: np.hstack([
        s["X_base"].values, rank_transform(s["custom"]).values])))

    # Group 5: NEGATIVE CONTROL - base + shuffled custom
    groups.append(("base+shuffled(NEG)", lambda s: np.hstack([
        s["X_base"].values, shuffle_columns(winsorize_zscore(s["custom"])).values])))

    # Group 6+: base + each individual factor
    for i, name in enumerate(CUSTOM_NAMES):
        groups.append((f"base+{name}", lambda s, idx=i: np.hstack([
            s["X_base"].values, winsorize_zscore(s["custom"][[CUSTOM_NAMES[idx]]]).values])))

    # Run all groups
    results = []
    for group_name, feature_fn in groups:
        logger.info(f"Running: {group_name}...")
        try:
            X_train = feature_fn(segments["train"]).astype(np.float32)
            X_valid = feature_fn(segments["valid"]).astype(np.float32)
            X_test = feature_fn(segments["test"]).astype(np.float32)
            y_train = segments["train"]["y"].values.astype(np.float32)
            y_valid = segments["valid"]["y"].values.astype(np.float32)
            y_test = segments["test"]["y"].values.astype(np.float32)
            test_idx = segments["test"]["index"]

            result = evaluate_group(X_train, y_train, X_valid, y_valid,
                                   X_test, y_test, test_idx, group_name)
            results.append(result)

            if "error" not in result:
                logger.info(
                    f"  {group_name}: IC={result['ic_mean']:+.4f} RankIC={result['rank_ic_mean']:+.4f} "
                    f"Spread={result['top20_spread']*100:+.3f}% ({result['train_time_s']:.0f}s)"
                )
        except Exception as e:
            logger.error(f"  {group_name} FAILED: {e}")
            results.append({"name": group_name, "error": str(e)})

    # Save
    output = {
        "ablation_at": datetime.now().isoformat(timespec="seconds"),
        "universe": args.universe,
        "seed": SEED,
        "label": LABEL_EXPR,
        "custom_factors": CUSTOM_NAMES,
        "results": results,
    }
    ABLATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = ABLATION_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    os.replace(tmp, ABLATION_PATH)

    # Summary
    if not args.json:
        print(f"\n{'='*80}")
        print(f"{'Group':<25} {'Feat':>5} {'IC':>8} {'RankIC':>8} {'Spread%':>9} {'SprdPos':>8}")
        print(f"{'-'*80}")
        for r in results:
            if "error" in r:
                print(f"{r['name']:<25} ERROR: {r['error'][:45]}")
            else:
                print(
                    f"{r['name']:<25} {r['n_features']:>5} {r['ic_mean']:>+8.4f} "
                    f"{r['rank_ic_mean']:>+8.4f} {r['top20_spread']*100:>+8.3f}% "
                    f"{r['spread_pos']:>7.0%}"
                )
        print(f"{'='*80}")

        # Interpret
        base = next((r for r in results if r["name"] == "base_raw"), None)
        shuffled = next((r for r in results if "shuffled" in r["name"]), None)
        best_enhanced = max(
            (r for r in results if "custom" in r["name"] and "shuffled" not in r["name"] and "error" not in r),
            key=lambda r: r["ic_mean"], default=None
        )

        if base and best_enhanced and shuffled:
            print(f"\nInterpretation:")
            if best_enhanced["ic_mean"] > base["ic_mean"] and shuffled["ic_mean"] <= base["ic_mean"]:
                print(f"  ✅ {best_enhanced['name']} improves over base AND shuffled is not better → REAL SIGNAL")
            elif best_enhanced["ic_mean"] <= base["ic_mean"]:
                print(f"  ❌ No custom factor group improves over base → FACTORS NOT USEFUL IN CURRENT FORM")
            elif shuffled["ic_mean"] > base["ic_mean"]:
                print(f"  ⚠️ Shuffled also improves → POSSIBLE DATA LEAK OR RANDOM LUCK")
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
