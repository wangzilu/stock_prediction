"""Phase 4I.0: Baseline v2 — 4-way feature set comparison.

Tests whether the 16 non-Alpha158 features have engineering defects
by comparing 4 variants:

1. FS-174: current champion (baseline)
2. FS-169-drop: remove 5 redundant features (pe/pb/turn_raw/amount_raw/amount_anom20)
3. FS-174-rank: keep all 174 but rank-normalize the 16 non-Alpha158 features
4. FS-169-rank: drop 5 + rank the remaining 11 non-Alpha158 features

Uses fast_rolling_gate style with feature cache.

Usage:
    python scripts/phase4i_baseline_v2_compare.py
"""
import json
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
SEED = 42

# Features to remove (redundant per audit)
DROP_FEATURES = ["pe", "pb", "turn_raw", "amount_raw", "amount_anom20"]

# Non-Alpha158 custom features (need rank normalization)
CUSTOM_FEATURES = [
    "pe", "pb", "turn_raw", "amount_raw",
    "pe_mom20", "pb_mom20", "turn_anom20", "turn_anom60",
    "amount_anom20", "turn_vol20", "ep", "bp", "price_pos20",
]

# Flow features (need rank normalization)
FLOW_FEATURES = ["flow_net_mf_latest", "flow_net_mf_5d", "flow_net_mf_20d_avg"]

ALL_NON_ALPHA158 = CUSTOM_FEATURES + FLOW_FEATURES


def train_xgb(X_train, y_train, X_valid, y_valid):
    import xgboost as xgb
    dt = xgb.DMatrix(X_train, label=y_train)
    dv = xgb.DMatrix(X_valid, label=y_valid)
    params = {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
              "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
              "objective": "reg:squarederror", "nthread": 12, "verbosity": 0, "seed": SEED}
    return xgb.train(params, dt, num_boost_round=400,
                     evals=[(dv, "valid")], early_stopping_rounds=30, verbose_eval=0)


def evaluate(pred, label, index):
    from scipy.stats import spearmanr
    mask = np.isfinite(pred) & np.isfinite(label)
    ps = pd.Series(pred[mask], index=index[mask])
    ls = pd.Series(label[mask], index=index[mask])
    rics, sprs = [], []
    for date in ps.index.get_level_values(0).unique():
        p = ps.loc[date].values; l = ls.loc[date].values
        if len(p) < 40: continue
        c, _ = spearmanr(p, l)
        if np.isfinite(c): rics.append(c)
        k = min(20, len(p) // 2)
        top = np.argpartition(p, -k)[-k:]
        bot = np.argpartition(p, k)[:k]
        sprs.append(l[top].mean() - l[bot].mean())
    return {
        "rank_ic_mean": round(float(np.nanmean(rics)), 6) if rics else 0,
        "top20_spread": round(float(np.mean(sprs)), 6) if sprs else 0,
    }


def rank_normalize_columns(df, cols, date_level=0):
    """Apply per-date cross-sectional rank to specified columns."""
    result = df.copy()
    for col in cols:
        if col in result.columns:
            result[col] = result[col].groupby(level=date_level).rank(pct=True)
    return result


def main():
    import xgboost as xgb
    from config.qlib_runtime import init_qlib
    init_qlib(str(DATA_DIR / "qlib_data" / "cn_data"))

    n_splits = 12
    test_days = 20
    train_days = 750
    valid_days = 60

    # Load cache
    cache_path = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
    logger.info(f"Loading cache: {cache_path}")
    cache = pd.read_parquet(str(cache_path))
    logger.info(f"  Shape: {cache.shape}")

    # Identify column groups
    all_feature_cols = [c for c in cache.columns if not c.startswith("__") and not c.startswith("_")
                        and not c.startswith("hsi_") and not c.startswith("hstech_")
                        and not c.startswith("nasdaq_")]
    label_col = "__label_5d"

    # Columns present in cache
    custom_in_cache = [c for c in CUSTOM_FEATURES if c in all_feature_cols]
    flow_in_cache = [c for c in FLOW_FEATURES if c in all_feature_cols]
    non_a158_in_cache = custom_in_cache + flow_in_cache
    drop_in_cache = [c for c in DROP_FEATURES if c in all_feature_cols]

    logger.info(f"  All features: {len(all_feature_cols)}")
    logger.info(f"  Non-Alpha158: {len(non_a158_in_cache)} ({custom_in_cache} + {flow_in_cache})")
    logger.info(f"  To drop: {drop_in_cache}")

    # Build 4 feature set variants
    fs_174_cols = all_feature_cols
    fs_169_cols = [c for c in all_feature_cols if c not in drop_in_cache]

    # Pre-compute rank versions
    logger.info("Pre-computing rank-normalized versions...")
    t0 = time.time()
    cache_ranked = rank_normalize_columns(cache, non_a158_in_cache)
    logger.info(f"  Ranked in {time.time()-t0:.1f}s")

    # Also clip pe_mom20/pb_mom20 to [-2, 2] in ranked version
    for col in ["pe_mom20", "pb_mom20"]:
        if col in cache_ranked.columns:
            # Clip before ranking (on original, rank is already 0-1)
            pass  # rank already handles outliers naturally

    feature_sets = {
        "FS-174": (cache, fs_174_cols),
        "FS-169-drop": (cache, fs_169_cols),
        "FS-174-rank": (cache_ranked, fs_174_cols),
        "FS-169-rank": (cache_ranked, fs_169_cols),
    }

    logger.info(f"\nFeature sets:")
    for name, (_, cols) in feature_sets.items():
        logger.info(f"  {name}: {len(cols)} features")

    # Rolling comparison
    trade_dates = sorted(cache.index.get_level_values(0).unique())
    today_idx = len(trade_dates) - 1
    dates_level = cache.index.get_level_values(0)

    all_results = []
    t_total = time.time()

    for split_idx in range(n_splits):
        test_end_idx = today_idx - split_idx * test_days
        test_start_idx = test_end_idx - test_days
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - valid_days
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - train_days
        if train_start_idx < 0:
            break

        tm = (dates_level >= trade_dates[train_start_idx]) & (dates_level <= trade_dates[train_end_idx])
        vm = (dates_level >= trade_dates[valid_start_idx]) & (dates_level <= trade_dates[valid_end_idx])
        em = (dates_level >= trade_dates[test_start_idx]) & (dates_level <= trade_dates[test_end_idx])

        y_tr = cache.loc[tm, label_col].values.astype(np.float32)
        y_va = cache.loc[vm, label_col].values.astype(np.float32)
        y_te = cache.loc[em, label_col].values.astype(np.float32)
        mtr = np.isfinite(y_tr); mva = np.isfinite(y_va); mte = np.isfinite(y_te)
        test_idx = cache.index[em]

        split_result = {"split": split_idx + 1}

        logger.info(f"\nSplit {split_idx+1}/{n_splits}:")

        for fs_name, (data_source, cols) in feature_sets.items():
            t1 = time.time()
            X_tr = data_source.loc[tm, cols].values.astype(np.float32)
            X_va = data_source.loc[vm, cols].values.astype(np.float32)
            X_te = data_source.loc[em, cols].values.astype(np.float32)

            model = train_xgb(X_tr[mtr], y_tr[mtr], X_va[mva], y_va[mva])
            pred = model.predict(xgb.DMatrix(X_te[mte]))
            metrics = evaluate(pred, y_te[mte], test_idx[mte])
            elapsed = time.time() - t1

            split_result[fs_name] = {**metrics, "n_feat": len(cols), "time_s": round(elapsed, 1)}
            logger.info(f"  {fs_name}({len(cols)}): RankIC={metrics['rank_ic_mean']:+.4f} "
                        f"Spread={metrics['top20_spread']*100:+.3f}% [{elapsed:.0f}s]")

        all_results.append(split_result)

    # Summary
    total_time = time.time() - t_total
    n = len(all_results)

    logger.info(f"\n{'='*70}")
    logger.info(f"PHASE 4I.0: BASELINE V2 COMPARISON ({n} splits, {total_time:.0f}s)")
    logger.info(f"{'='*70}")

    for fs_name in ["FS-174", "FS-169-drop", "FS-174-rank", "FS-169-rank"]:
        rics = [r[fs_name]["rank_ic_mean"] for r in all_results]
        sprs = [r[fs_name]["top20_spread"] for r in all_results]
        logger.info(f"\n  {fs_name} ({all_results[0][fs_name]['n_feat']} features):")
        logger.info(f"    avg RankIC: {np.mean(rics):+.4f}")
        logger.info(f"    avg Spread: {np.mean(sprs)*100:+.3f}%")
        logger.info(f"    RankIC>0:   {sum(1 for r in rics if r > 0)}/{n}")
        logger.info(f"    Spread>0:   {sum(1 for s in sprs if s > 0)}/{n}")

    # Find best
    avg_rics = {fs: np.mean([r[fs]["rank_ic_mean"] for r in all_results])
                for fs in ["FS-174", "FS-169-drop", "FS-174-rank", "FS-169-rank"]}
    best = max(avg_rics, key=avg_rics.get)
    logger.info(f"\n  Best: {best} (avg RankIC {avg_rics[best]:+.4f})")

    # Save
    from utils.json_utils import json_default
    out_path = DATA_DIR / "phase4" / "phase4i_baseline_v2_compare.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump({"evaluated_at": datetime.now().isoformat(timespec="seconds"),
                    "n_splits": n, "total_time_s": round(total_time, 1),
                    "avg_rank_ic": {k: round(float(v), 6) for k, v in avg_rics.items()},
                    "best": best,
                    "splits": all_results}, f, indent=2, default=json_default)
    logger.info(f"\nSaved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
