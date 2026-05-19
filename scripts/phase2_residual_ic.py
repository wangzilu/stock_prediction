"""Phase 2: Residual IC — test if new factors explain baseline model residuals.

If a factor has IC with actual returns but zero residual IC, it's just
repeating information the baseline already captures. Only factors with
residual IC provide genuine incremental alpha.

residual = actual_return - baseline_prediction
residual_IC = corr(new_factor, residual)

Usage:
    python scripts/phase2_residual_ic.py
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
              "objective": "reg:squarederror", "nthread": 12, "verbosity": 0, "seed": SEED}
    return xgb.train(params, dt, num_boost_round=400,
                     evals=[(dv, "valid")], early_stopping_rounds=30, verbose_eval=0)


def main():
    import xgboost as xgb
    from config.qlib_runtime import init_qlib
    init_qlib(str(DATA_DIR / "qlib_data" / "cn_data"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-splits", type=int, default=6,
                        help="Fewer splits OK — residual IC is per-factor, not per-model")
    parser.add_argument("--test-days", type=int, default=40)
    parser.add_argument("--train-days", type=int, default=750)
    parser.add_argument("--valid-days", type=int, default=60)
    args = parser.parse_args()

    cache_path = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
    logger.info(f"Loading cache: {cache_path}")
    cache = pd.read_parquet(str(cache_path))
    logger.info(f"  Shape: {cache.shape}")

    # Identify factor groups in cache
    base_cols = [c for c in cache.columns
                 if not c.startswith("__") and not c.startswith("_")
                 and not c.startswith("hsi_") and not c.startswith("hstech_")
                 and not c.startswith("nasdaq_") and c != "holder_num"]
    label_col = "__label_5d"

    factor_groups = {
        "regime_hsi": [c for c in cache.columns if c.startswith("hsi_")],
        "regime_hstech": [c for c in cache.columns if c.startswith("hstech_")],
        "regime_nasdaq": [c for c in cache.columns if c.startswith("nasdaq_")],
        "holder_num": ["holder_num"] if "holder_num" in cache.columns else [],
    }
    # Remove empty groups
    factor_groups = {k: v for k, v in factor_groups.items() if v}

    logger.info(f"Base features: {len(base_cols)}")
    for fg, cols in factor_groups.items():
        logger.info(f"  {fg}: {len(cols)} cols")

    trade_dates = sorted(cache.index.get_level_values(0).unique())
    today_idx = len(trade_dates) - 1
    dates_level = cache.index.get_level_values(0)

    all_results = {}
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

        tm = (dates_level >= trade_dates[train_start_idx]) & (dates_level <= trade_dates[train_end_idx])
        vm = (dates_level >= trade_dates[valid_start_idx]) & (dates_level <= trade_dates[valid_end_idx])
        em = (dates_level >= trade_dates[test_start_idx]) & (dates_level <= trade_dates[test_end_idx])

        y_tr = cache.loc[tm, label_col].values.astype(np.float32)
        y_va = cache.loc[vm, label_col].values.astype(np.float32)
        y_te = cache.loc[em, label_col].values.astype(np.float32)
        mtr = np.isfinite(y_tr); mva = np.isfinite(y_va); mte = np.isfinite(y_te)

        X_base_tr = cache.loc[tm, base_cols].values.astype(np.float32)
        X_base_va = cache.loc[vm, base_cols].values.astype(np.float32)
        X_base_te = cache.loc[em, base_cols].values.astype(np.float32)

        test_idx = cache.index[em]

        logger.info(f"\nSplit {split_idx+1}/{args.n_splits}: "
                    f"test {str(trade_dates[test_start_idx])[:10]}~{str(trade_dates[test_end_idx])[:10]}")

        # Train baseline model
        t1 = time.time()
        model_base = train_xgb(X_base_tr[mtr], y_tr[mtr], X_base_va[mva], y_va[mva])
        pred_base = model_base.predict(xgb.DMatrix(X_base_te[mte]))
        logger.info(f"  Baseline trained: {time.time()-t1:.1f}s")

        # Compute residuals
        residual = y_te[mte] - pred_base
        residual_s = pd.Series(residual, index=test_idx[mte])

        # For each factor group, compute residual IC per date
        for fg, cols in factor_groups.items():
            fg_data = cache.loc[em, cols].values.astype(np.float32)[mte]

            # Per-date residual IC for each factor column
            ric_per_col = {col: [] for col in cols}
            for i, date in enumerate(residual_s.index.get_level_values(0).unique()):
                date_mask = residual_s.index.get_level_values(0) == date
                r_day = residual_s.loc[date_mask].values
                if len(r_day) < 40:
                    continue

                for j, col in enumerate(cols):
                    f_day = fg_data[date_mask.values if hasattr(date_mask, 'values') else date_mask, j]
                    finite = np.isfinite(f_day) & np.isfinite(r_day)
                    if finite.sum() < 30:
                        continue
                    from scipy.stats import spearmanr
                    corr, _ = spearmanr(f_day[finite], r_day[finite])
                    if np.isfinite(corr):
                        ric_per_col[col].append(corr)

            # Summary for this split
            for col in cols:
                vals = ric_per_col[col]
                if not vals:
                    continue
                key = f"{fg}:{col}"
                if key not in all_results:
                    all_results[key] = []
                all_results[key].extend(vals)

            # Log summary
            all_vals = [v for vs in ric_per_col.values() for v in vs]
            if all_vals:
                avg_ric = np.mean(all_vals)
                pos_pct = np.mean([v > 0 for v in all_vals])
                logger.info(f"  {fg}: avg residual IC={avg_ric:+.4f}, >0={pos_pct:.0%}")

    # Final summary
    total_time = time.time() - t_total
    logger.info(f"\n{'='*60}")
    logger.info(f"RESIDUAL IC SUMMARY ({total_time:.0f}s)")
    logger.info(f"{'='*60}")

    summary = {}
    for key, vals in sorted(all_results.items()):
        avg = np.mean(vals)
        pos = np.mean([v > 0 for v in vals])
        significant = abs(avg) > 0.01 and pos > 0.55
        summary[key] = {
            "avg_residual_ic": round(float(avg), 6),
            "pos_pct": round(float(pos), 4),
            "n_obs": len(vals),
            "significant": significant,
        }
        marker = "✅" if significant else "  "
        logger.info(f"  {marker} {key:<35} avg={avg:+.4f} >0={pos:.0%} (n={len(vals)})")

    # Save
    out_path = DATA_DIR / "phase4" / "phase2_residual_ic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump({"evaluated_at": datetime.now().isoformat(timespec="seconds"),
                    "total_time_s": round(total_time, 1),
                    "factors": summary}, f, indent=2)
    logger.info(f"\nSaved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
