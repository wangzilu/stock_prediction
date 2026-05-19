"""Phase 2: PIT Audit — check if baseline features have look-ahead bias.

Tests capital flow features with lag0 (current, potentially leaking) vs
lag1 (safe: trade_date + 1 day). If lag1 IC drops significantly,
the baseline has been using future information.

Usage:
    python scripts/phase2_pit_audit.py
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
    mask = np.isfinite(pred) & np.isfinite(label)
    ps = pd.Series(pred[mask], index=index[mask])
    ls = pd.Series(label[mask], index=index[mask])
    ric_vals = []
    spreads = []
    for date in ps.index.get_level_values(0).unique():
        p = ps.loc[date]; l = ls.loc[date]
        if len(p) < 40:
            continue
        ric_vals.append(float(p.corr(l, method="spearman")))
        s = pd.DataFrame({"p": p, "l": l}).sort_values("p", ascending=False)
        spreads.append(s.head(20)["l"].mean() - s.tail(20)["l"].mean())
    ric = np.array(ric_vals)
    return {
        "rank_ic_mean": round(float(np.nanmean(ric)), 6) if len(ric) > 0 else 0,
        "top20_spread": round(float(np.mean(spreads)), 6) if spreads else 0,
    }


def main():
    import xgboost as xgb
    from config.qlib_runtime import init_qlib
    init_qlib(str(DATA_DIR / "qlib_data" / "cn_data"))

    # Load cache
    cache_path = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
    logger.info(f"Loading cache: {cache_path}")
    cache = pd.read_parquet(str(cache_path))
    logger.info(f"  Shape: {cache.shape}")

    label_col = "__label_5d"
    trade_dates = sorted(cache.index.get_level_values(0).unique())
    today_idx = len(trade_dates) - 1

    # Identify flow columns (the ones we're auditing)
    flow_cols = [c for c in cache.columns if c.startswith("flow_")]
    all_feature_cols = [c for c in cache.columns if not c.startswith("__") and not c.startswith("_")]
    non_flow_cols = [c for c in all_feature_cols if c not in flow_cols]

    logger.info(f"  Flow cols to audit: {flow_cols}")
    logger.info(f"  Non-flow cols: {len(non_flow_cols)}")

    if not flow_cols:
        logger.error("No flow columns found in cache!")
        sys.exit(1)

    # Create lag1 version of flow features
    # For each (date, instrument), use the PREVIOUS trading day's flow value
    logger.info("Creating lag1 flow features...")
    t0 = time.time()

    flow_data = cache[flow_cols].copy()
    lag1_flow = flow_data.copy()
    lag1_flow[:] = np.nan

    # For each stock, shift flow values by 1 trading day
    for stock in cache.index.get_level_values(1).unique():
        stock_mask = cache.index.get_level_values(1) == stock
        stock_data = flow_data.loc[stock_mask]
        if len(stock_data) < 2:
            continue
        # Shift by 1 position (1 trading day)
        shifted = stock_data.shift(1)
        lag1_flow.loc[stock_mask] = shifted.values

    logger.info(f"  Lag1 flow created: {time.time()-t0:.1f}s")

    # Also create lag2 for comparison
    lag2_flow = flow_data.copy()
    lag2_flow[:] = np.nan
    for stock in cache.index.get_level_values(1).unique():
        stock_mask = cache.index.get_level_values(1) == stock
        stock_data = flow_data.loc[stock_mask]
        if len(stock_data) < 3:
            continue
        shifted = stock_data.shift(2)
        lag2_flow.loc[stock_mask] = shifted.values

    # Build feature sets
    feature_sets = {
        "no_flow": non_flow_cols,
        "flow_lag0 (current)": non_flow_cols + flow_cols,  # potentially leaking
        "flow_lag1 (safe)": non_flow_cols,     # will add lag1 columns
        "flow_lag2": non_flow_cols,            # will add lag2 columns
    }

    # Rolling comparison
    n_splits = 8
    test_days = 20
    train_days = 750
    valid_days = 60

    dates_level = cache.index.get_level_values(0)
    all_results = []

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

        logger.info(f"\nSplit {split_idx+1}/{n_splits}: "
                    f"test {str(trade_dates[test_start_idx])[:10]}~{str(trade_dates[test_end_idx])[:10]}")

        split_result = {"split": split_idx + 1}

        for fs_name in ["no_flow", "flow_lag0 (current)", "flow_lag1 (safe)", "flow_lag2"]:
            if fs_name == "no_flow":
                X_tr = cache.loc[tm, non_flow_cols].values.astype(np.float32)
                X_va = cache.loc[vm, non_flow_cols].values.astype(np.float32)
                X_te = cache.loc[em, non_flow_cols].values.astype(np.float32)
            elif fs_name == "flow_lag0 (current)":
                X_tr = cache.loc[tm, non_flow_cols + flow_cols].values.astype(np.float32)
                X_va = cache.loc[vm, non_flow_cols + flow_cols].values.astype(np.float32)
                X_te = cache.loc[em, non_flow_cols + flow_cols].values.astype(np.float32)
            elif fs_name == "flow_lag1 (safe)":
                X_tr = np.hstack([cache.loc[tm, non_flow_cols].values.astype(np.float32),
                                  lag1_flow.loc[tm].values.astype(np.float32)])
                X_va = np.hstack([cache.loc[vm, non_flow_cols].values.astype(np.float32),
                                  lag1_flow.loc[vm].values.astype(np.float32)])
                X_te = np.hstack([cache.loc[em, non_flow_cols].values.astype(np.float32),
                                  lag1_flow.loc[em].values.astype(np.float32)])
            elif fs_name == "flow_lag2":
                X_tr = np.hstack([cache.loc[tm, non_flow_cols].values.astype(np.float32),
                                  lag2_flow.loc[tm].values.astype(np.float32)])
                X_va = np.hstack([cache.loc[vm, non_flow_cols].values.astype(np.float32),
                                  lag2_flow.loc[vm].values.astype(np.float32)])
                X_te = np.hstack([cache.loc[em, non_flow_cols].values.astype(np.float32),
                                  lag2_flow.loc[em].values.astype(np.float32)])

            t1 = time.time()
            model = train_xgb(X_tr[mtr], y_tr[mtr], X_va[mva], y_va[mva])
            pred = model.predict(xgb.DMatrix(X_te[mte]))
            metrics = evaluate(pred, y_te[mte], test_idx[mte])
            elapsed = time.time() - t1

            split_result[fs_name] = metrics
            logger.info(f"  {fs_name:<25} RankIC={metrics['rank_ic_mean']:+.4f} "
                        f"Spread={metrics['top20_spread']*100:+.3f}% [{elapsed:.0f}s]")

        all_results.append(split_result)

    # Summary
    n = len(all_results)
    logger.info(f"\n{'='*70}")
    logger.info(f"PIT AUDIT: CAPITAL FLOW LAG COMPARISON ({n} splits)")
    logger.info(f"{'='*70}")

    for fs_name in ["no_flow", "flow_lag0 (current)", "flow_lag1 (safe)", "flow_lag2"]:
        rics = [r[fs_name]["rank_ic_mean"] for r in all_results]
        sprs = [r[fs_name]["top20_spread"] for r in all_results]
        logger.info(f"  {fs_name:<25} avg RankIC={np.mean(rics):+.4f}  avg Spread={np.mean(sprs)*100:+.3f}%")

    # Key diagnostic
    lag0_rics = [r["flow_lag0 (current)"]["rank_ic_mean"] for r in all_results]
    lag1_rics = [r["flow_lag1 (safe)"]["rank_ic_mean"] for r in all_results]
    noflow_rics = [r["no_flow"]["rank_ic_mean"] for r in all_results]

    drop_pct = (np.mean(lag0_rics) - np.mean(lag1_rics)) / (abs(np.mean(lag0_rics)) + 1e-8)

    logger.info(f"\n  DIAGNOSTIC:")
    logger.info(f"    lag0 → lag1 RankIC drop: {drop_pct*100:+.1f}%")
    if abs(drop_pct) > 0.15:
        logger.warning(f"    ⚠️  SIGNIFICANT DROP ({drop_pct*100:+.1f}%) — flow may have look-ahead bias!")
        logger.warning(f"    Recommend: switch to lag1 flow in production")
    else:
        logger.info(f"    ✅ Drop is small ({drop_pct*100:+.1f}%) — flow features are likely PIT-safe")

    if np.mean(lag1_rics) <= np.mean(noflow_rics):
        logger.warning(f"    ⚠️  lag1 flow ≤ no_flow — flow features may be useless after proper lagging!")
    else:
        logger.info(f"    ✅ lag1 flow > no_flow — flow features provide genuine alpha even with lag")

    # Save
    out_path = DATA_DIR / "phase4" / "phase2_pit_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump({"evaluated_at": datetime.now().isoformat(timespec="seconds"),
                    "n_splits": n, "results": all_results,
                    "diagnostic": {
                        "lag0_avg_ric": round(float(np.mean(lag0_rics)), 6),
                        "lag1_avg_ric": round(float(np.mean(lag1_rics)), 6),
                        "noflow_avg_ric": round(float(np.mean(noflow_rics)), 6),
                        "drop_pct": round(float(drop_pct), 4),
                    }}, f, indent=2)
    logger.info(f"\nSaved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
