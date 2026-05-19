"""Phase 4E: Feature set comparison — FS-174 vs FS-360 vs FS-534.

Uses pre-built caches for fast rolling comparison.

Prerequisite:
    python scripts/build_feature_cache.py --all
    python scripts/build_alpha360_cache.py

Usage:
    python scripts/phase4e_feature_set_compare.py
"""
import argparse
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


def train_xgb(X_train, y_train, X_valid, y_valid):
    import xgboost as xgb
    dt = xgb.DMatrix(X_train, label=y_train)
    dv = xgb.DMatrix(X_valid, label=y_valid)
    params = {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
              "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
              "objective": "reg:squarederror", "nthread": 4, "verbosity": 0, "seed": SEED}
    model = xgb.train(params, dt, num_boost_round=500,
                      evals=[(dv, "valid")], early_stopping_rounds=50, verbose_eval=0)
    return model


def evaluate(pred, label, index):
    from qlib.contrib.eva.alpha import calc_ic
    mask = np.isfinite(pred) & np.isfinite(label)
    ps = pd.Series(pred[mask], index=index[mask])
    ls = pd.Series(label[mask], index=index[mask])
    ic, ric = calc_ic(ps, ls)
    spreads = []
    for _, g in pd.DataFrame({"pred": ps, "label": ls}).groupby(level=0):
        if len(g) < 40:
            continue
        s = g.sort_values("pred", ascending=False)
        spreads.append(s.head(20)["label"].mean() - s.tail(20)["label"].mean())
    return {
        "rank_ic_mean": round(float(ric.mean()), 6),
        "rank_ic_pos": round(float((ric > 0).mean()), 4),
        "top20_spread": round(float(np.mean(spreads)) if spreads else 0, 6),
        "spread_pos": round(float(np.mean([s > 0 for s in spreads])) if spreads else 0, 4),
    }


def load_and_align_caches(cache_174_path, cache_360_path):
    """Load both caches, align on common (date, instrument) index."""
    logger.info(f"Loading FS-174 cache: {cache_174_path}")
    c174 = pd.read_parquet(str(cache_174_path))
    logger.info(f"  Shape: {c174.shape}")

    logger.info(f"Loading FS-360 cache: {cache_360_path}")
    c360 = pd.read_parquet(str(cache_360_path))
    logger.info(f"  Shape: {c360.shape}")

    # Align on common index
    common_idx = c174.index.intersection(c360.index)
    logger.info(f"  Common index: {len(common_idx)} rows")

    c174 = c174.loc[common_idx]
    c360 = c360.loc[common_idx]

    return c174, c360, common_idx


def main():
    import xgboost as xgb
    from config.qlib_runtime import init_qlib
    init_qlib(str(DATA_DIR / "qlib_data" / "cn_data"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-splits", type=int, default=12)
    parser.add_argument("--test-days", type=int, default=20)
    parser.add_argument("--train-days", type=int, default=750)
    parser.add_argument("--valid-days", type=int, default=60)
    args = parser.parse_args()

    cache_174_path = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
    cache_360_path = DATA_DIR / "feature_cache_alpha360.parquet"

    if not cache_174_path.exists():
        logger.error("FS-174 cache not found. Run: python scripts/build_feature_cache.py --all")
        sys.exit(1)
    if not cache_360_path.exists():
        logger.error("FS-360 cache not found. Run: python scripts/build_alpha360_cache.py")
        sys.exit(1)

    c174, c360, common_idx = load_and_align_caches(cache_174_path, cache_360_path)

    # Feature columns
    feat_174 = [c for c in c174.columns if not c.startswith("__") and not c.startswith("_")]
    feat_360 = [c for c in c360.columns if not c.startswith("__") and not c.startswith("_")]

    # FS-534: combine 174 + 360 features
    feat_534 = feat_174 + feat_360

    feature_sets = {
        "FS-174": (c174, feat_174),
        "FS-360": (c360, feat_360),
        "FS-534": (None, feat_534),  # combined, built per-split
    }

    logger.info(f"\nFeature sets:")
    logger.info(f"  FS-174: {len(feat_174)} features")
    logger.info(f"  FS-360: {len(feat_360)} features")
    logger.info(f"  FS-534: {len(feat_534)} features (174 + 360)")

    # Rolling
    trade_dates = sorted(common_idx.get_level_values(0).unique())
    today_idx = len(trade_dates) - 1

    all_results = []
    t_total = time.time()

    for split_idx in range(args.n_splits):
        test_end_idx = today_idx - split_idx * args.test_days
        test_start_idx = test_end_idx - args.test_days
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - args.valid_days
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - args.train_days

        if train_start_idx < 0:
            break

        test_end = trade_dates[test_end_idx]
        test_start = trade_dates[test_start_idx]
        valid_end = trade_dates[valid_end_idx]
        valid_start = trade_dates[valid_start_idx]
        train_end = trade_dates[train_end_idx]
        train_start = trade_dates[train_start_idx]

        logger.info(f"\nSplit {split_idx+1}/{args.n_splits}: "
                    f"test {str(test_start)[:10]}~{str(test_end)[:10]}")

        dates_level = common_idx.get_level_values(0)
        train_mask = (dates_level >= train_start) & (dates_level <= train_end)
        valid_mask = (dates_level >= valid_start) & (dates_level <= valid_end)
        test_mask = (dates_level >= test_start) & (dates_level <= test_end)

        # Label (use 174 cache's label — same for both)
        y_train = c174.loc[train_mask, "__label_5d"].values.astype(np.float32)
        y_valid = c174.loc[valid_mask, "__label_5d"].values.astype(np.float32)
        y_test = c174.loc[test_mask, "__label_5d"].values.astype(np.float32)
        test_idx = common_idx[test_mask]

        mask_tr = np.isfinite(y_train)
        mask_va = np.isfinite(y_valid)
        mask_te = np.isfinite(y_test)

        split_result = {"split": split_idx + 1,
                        "test": f"{str(test_start)[:10]}~{str(test_end)[:10]}"}

        for fs_name, (cache, cols) in feature_sets.items():
            t1 = time.time()

            if fs_name == "FS-534":
                # Combine both caches
                X_train = np.hstack([
                    c174.loc[train_mask, feat_174].values.astype(np.float32)[mask_tr],
                    c360.loc[train_mask, feat_360].values.astype(np.float32)[mask_tr],
                ])
                X_valid = np.hstack([
                    c174.loc[valid_mask, feat_174].values.astype(np.float32)[mask_va],
                    c360.loc[valid_mask, feat_360].values.astype(np.float32)[mask_va],
                ])
                X_test = np.hstack([
                    c174.loc[test_mask, feat_174].values.astype(np.float32)[mask_te],
                    c360.loc[test_mask, feat_360].values.astype(np.float32)[mask_te],
                ])
            else:
                X_train = cache.loc[train_mask, cols].values.astype(np.float32)[mask_tr]
                X_valid = cache.loc[valid_mask, cols].values.astype(np.float32)[mask_va]
                X_test = cache.loc[test_mask, cols].values.astype(np.float32)[mask_te]

            model = train_xgb(X_train, y_train[mask_tr], X_valid, y_valid[mask_va])
            pred = model.predict(xgb.DMatrix(X_test))
            metrics = evaluate(pred, y_test[mask_te], test_idx[mask_te])
            elapsed = time.time() - t1

            split_result[fs_name] = {"n_feat": len(cols), **metrics, "time_s": round(elapsed, 1)}
            logger.info(f"  {fs_name}({len(cols)}): RankIC={metrics['rank_ic_mean']:+.4f} "
                        f"Spread={metrics['top20_spread']*100:+.3f}% [{elapsed:.1f}s]")

        all_results.append(split_result)

    # Summary
    total_time = time.time() - t_total
    n = len(all_results)

    logger.info(f"\n{'='*70}")
    logger.info(f"PHASE 4E: FEATURE SET COMPARISON ({n} splits, {total_time:.0f}s)")
    logger.info(f"{'='*70}")

    for fs_name in ["FS-174", "FS-360", "FS-534"]:
        rics = [r[fs_name]["rank_ic_mean"] for r in all_results if fs_name in r]
        sprs = [r[fs_name]["top20_spread"] for r in all_results if fs_name in r]
        logger.info(f"\n  {fs_name}:")
        logger.info(f"    avg RankIC: {np.mean(rics):+.4f}")
        logger.info(f"    avg Spread: {np.mean(sprs)*100:+.3f}%")
        logger.info(f"    RankIC>0:   {sum(1 for r in rics if r > 0)}/{len(rics)}")
        logger.info(f"    Spread>0:   {sum(1 for s in sprs if s > 0)}/{len(sprs)}")

    # Decision
    avg_rics = {fs: np.mean([r[fs]["rank_ic_mean"] for r in all_results if fs in r])
                for fs in ["FS-174", "FS-360", "FS-534"]}
    best = max(avg_rics, key=avg_rics.get)
    logger.info(f"\n  Best by RankIC: {best} ({avg_rics[best]:+.4f})")

    if avg_rics["FS-534"] > max(avg_rics["FS-174"], avg_rics["FS-360"]):
        logger.info("  → FS-534 beats both subsets: combined features add value")
    elif avg_rics["FS-360"] > avg_rics["FS-174"]:
        logger.info("  → FS-360 > FS-174: Alpha360 has independent value")
    else:
        logger.info("  → FS-174 still best: Alpha360 doesn't help for tree models")

    # Save
    out_path = DATA_DIR / "phase4" / "feature_set_compare.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump({"evaluated_at": datetime.now().isoformat(timespec="seconds"),
                    "n_splits": n, "total_time_s": round(total_time, 1),
                    "splits": all_results}, f, indent=2)
    logger.info(f"\nSaved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
