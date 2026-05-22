"""Phase 4i: Residualized ablation — test if moneyflow_v2 factors unlock
hidden alpha after removing baseline-captured information.

Method (two-stage residual model):
1. Train baseline XGB on 174 base features -> predict on test
2. Compute residuals = actual_label - baseline_prediction
3. Compute residual IC: spearman(each_moneyflow_v2_factor, residual) per date
4. Train stage-2 XGB using only moneyflow_v2 features to predict residuals
5. Final prediction = baseline_pred + stage2_pred
6. Compare base-only vs two-stage model

Usage:
    python scripts/phase4i_residualized_ablation.py
    python scripts/phase4i_residualized_ablation.py --n-splits 8
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

MONEYFLOW_V2_FACTORS = [
    "main_flow_adv", "order_imbalance", "large_small_divergence",
    "flow_zscore_60d", "flow_persistence_10d", "flow_industry_rank",
]


def train_xgb(X_train, y_train, X_valid, y_valid, nthread=12, max_rounds=400,
               early_stop=30):
    """Train XGB with standard params."""
    import xgboost as xgb
    dt = xgb.DMatrix(X_train, label=y_train)
    dv = xgb.DMatrix(X_valid, label=y_valid)
    params = {
        "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
        "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
        "objective": "reg:squarederror", "nthread": nthread, "verbosity": 0, "seed": SEED,
    }
    model = xgb.train(params, dt, num_boost_round=max_rounds,
                      evals=[(dv, "valid")], early_stopping_rounds=early_stop,
                      verbose_eval=0)
    return model


def train_xgb_stage2(X_train, y_train, X_valid, y_valid, nthread=12):
    """Train a lighter stage-2 XGB for residual prediction."""
    import xgboost as xgb
    dt = xgb.DMatrix(X_train, label=y_train)
    dv = xgb.DMatrix(X_valid, label=y_valid)
    params = {
        "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8,
        "colsample_bytree": 1.0, "reg_alpha": 50.0, "reg_lambda": 200.0,
        "objective": "reg:squarederror", "nthread": nthread, "verbosity": 0, "seed": SEED,
    }
    model = xgb.train(params, dt, num_boost_round=400,
                      evals=[(dv, "valid")], early_stopping_rounds=30,
                      verbose_eval=0)
    return model


def evaluate_rankic(pred, label, index):
    """Compute per-date RankIC and top-bottom spread."""
    from scipy.stats import spearmanr
    mask = np.isfinite(pred) & np.isfinite(label)
    ps = pd.Series(pred[mask], index=index[mask])
    ls = pd.Series(label[mask], index=index[mask])

    ric_vals = []
    spreads = []
    for date in ps.index.get_level_values(0).unique():
        p = ps.loc[date].values
        l = ls.loc[date].values
        n = len(p)
        if n < 40:
            continue
        corr, _ = spearmanr(p, l)
        if np.isfinite(corr):
            ric_vals.append(corr)
        k = min(20, n // 2)
        top_idx = np.argpartition(p, -k)[-k:]
        bot_idx = np.argpartition(p, k)[:k]
        spreads.append(l[top_idx].mean() - l[bot_idx].mean())

    ric = np.array(ric_vals)
    return {
        "rank_ic_mean": round(float(np.nanmean(ric)), 6) if len(ric) > 0 else 0,
        "rank_ic_pos": round(float(np.nanmean(ric > 0)), 4) if len(ric) > 0 else 0,
        "top20_spread": round(float(np.mean(spreads)), 6) if spreads else 0,
        "n_days": len(ric_vals),
    }


def merge_moneyflow_v2(cache_index, mf_df):
    """PIT-safe asof merge of moneyflow_v2 onto cache index.

    Uses vectorized searchsorted per stock, same approach as
    FeatureMerger._asof_merge_timeseries().
    """
    date_level = 0
    inst_level = 1

    mf = mf_df.copy()
    mf["qlib_code"] = mf["qlib_code"].astype(str).str.upper()
    mf["date"] = pd.to_datetime(mf["date"], errors="coerce")
    mf = mf.dropna(subset=["date"])
    for col in MONEYFLOW_V2_FACTORS:
        mf[col] = pd.to_numeric(mf[col], errors="coerce")
    mf = mf.sort_values(["qlib_code", "date"]).drop_duplicates(
        ["qlib_code", "date"], keep="last")

    train_dates = pd.to_datetime(cache_index.get_level_values(date_level))
    train_insts = cache_index.get_level_values(inst_level).astype(str).str.upper()

    n = len(cache_index)
    result_arrays = {col: np.full(n, np.nan, dtype=np.float64) for col in MONEYFLOW_V2_FACTORS}

    train_inst_arr = np.asarray(train_insts)
    train_date_arr = train_dates.values

    stocks_in_mf = set(mf["qlib_code"].unique())

    for stock, stock_df in mf.groupby("qlib_code"):
        stock_mask = train_inst_arr == stock
        if not stock_mask.any():
            continue

        stock_dates = stock_df["date"].values.astype("datetime64[ns]")
        idx_in_train = np.where(stock_mask)[0]
        query_dates = train_date_arr[idx_in_train].astype("datetime64[ns]")
        positions = np.searchsorted(stock_dates, query_dates, side="right") - 1
        valid = positions >= 0

        for col in MONEYFLOW_V2_FACTORS:
            vals = stock_df[col].values
            result_arrays[col][idx_in_train[valid]] = vals[positions[valid]]

    result = pd.DataFrame(result_arrays, index=cache_index)
    return result


def main():
    import xgboost as xgb
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-splits", type=int, default=8)
    parser.add_argument("--test-days", type=int, default=40)
    parser.add_argument("--train-days", type=int, default=750)
    parser.add_argument("--valid-days", type=int, default=60)
    args = parser.parse_args()

    # Load feature cache
    cache_path = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
    logger.info(f"Loading feature cache: {cache_path}")
    cache = pd.read_parquet(str(cache_path))
    logger.info(f"  Cache shape: {cache.shape}")

    # Identify base columns (174)
    base_cols = [c for c in cache.columns
                 if not c.startswith("__") and not c.startswith("_")
                 and not c.startswith("hsi_") and not c.startswith("hstech_")
                 and not c.startswith("nasdaq_") and c != "holder_num"]
    label_col = "__label_5d"
    logger.info(f"  Base features: {len(base_cols)}")

    # Load moneyflow_v2
    mf_path = DATA_DIR / "moneyflow_v2.parquet"
    logger.info(f"Loading moneyflow_v2: {mf_path}")
    mf_raw = pd.read_parquet(str(mf_path))
    logger.info(f"  moneyflow_v2 shape: {mf_raw.shape}")

    # Merge moneyflow_v2 onto cache index (PIT-safe asof merge)
    logger.info("Merging moneyflow_v2 onto cache index...")
    t_merge = time.time()
    mf_merged = merge_moneyflow_v2(cache.index, mf_raw)
    logger.info(f"  Merge done: {time.time()-t_merge:.1f}s, "
                f"coverage={mf_merged.notna().any(axis=1).mean():.1%}")

    # Trading dates
    trade_dates = sorted(cache.index.get_level_values(0).unique())
    today_idx = len(trade_dates) - 1
    dates_level = cache.index.get_level_values(0)

    all_split_results = []
    all_residual_ics = {col: [] for col in MONEYFLOW_V2_FACTORS}
    t_total = time.time()

    for split_idx in range(args.n_splits):
        test_end_idx = today_idx - split_idx * args.test_days
        test_start_idx = test_end_idx - args.test_days
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - args.valid_days
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - args.train_days
        if train_start_idx < 0:
            logger.warning(f"Split {split_idx+1}: out of bounds, stopping")
            break

        # Date masks
        tm = (dates_level >= trade_dates[train_start_idx]) & (dates_level <= trade_dates[train_end_idx])
        vm = (dates_level >= trade_dates[valid_start_idx]) & (dates_level <= trade_dates[valid_end_idx])
        em = (dates_level >= trade_dates[test_start_idx]) & (dates_level <= trade_dates[test_end_idx])

        # Labels
        y_tr = cache.loc[tm, label_col].values.astype(np.float32)
        y_va = cache.loc[vm, label_col].values.astype(np.float32)
        y_te = cache.loc[em, label_col].values.astype(np.float32)
        mtr = np.isfinite(y_tr); mva = np.isfinite(y_va); mte = np.isfinite(y_te)

        # Base features
        X_base_tr = cache.loc[tm, base_cols].values.astype(np.float32)
        X_base_va = cache.loc[vm, base_cols].values.astype(np.float32)
        X_base_te = cache.loc[em, base_cols].values.astype(np.float32)

        # Moneyflow features
        X_mf_tr = mf_merged.loc[tm].values.astype(np.float32)
        X_mf_va = mf_merged.loc[vm].values.astype(np.float32)
        X_mf_te = mf_merged.loc[em].values.astype(np.float32)

        test_idx = cache.index[em]

        logger.info(f"\n{'='*60}")
        logger.info(f"Split {split_idx+1}/{args.n_splits}: "
                    f"test {str(trade_dates[test_start_idx])[:10]}"
                    f"~{str(trade_dates[test_end_idx])[:10]}")

        # --- Stage 1: Train baseline on base features ---
        t1 = time.time()
        model_base = train_xgb(X_base_tr[mtr], y_tr[mtr], X_base_va[mva], y_va[mva])

        pred_base_te = model_base.predict(xgb.DMatrix(X_base_te[mte]))
        pred_base_va = model_base.predict(xgb.DMatrix(X_base_va[mva]))
        base_metrics = evaluate_rankic(pred_base_te, y_te[mte], test_idx[mte])
        logger.info(f"  Baseline: RankIC={base_metrics['rank_ic_mean']:+.4f}, "
                    f"Spread={base_metrics['top20_spread']:.4f} ({time.time()-t1:.1f}s)")

        # --- Compute residuals ---
        residual_te = y_te[mte] - pred_base_te
        residual_va = y_va[mva] - pred_base_va
        residual_te_s = pd.Series(residual_te, index=test_idx[mte])

        # --- Residual IC per factor ---
        from scipy.stats import spearmanr
        mf_te_valid = X_mf_te[mte]
        for j, col in enumerate(MONEYFLOW_V2_FACTORS):
            ric_day = []
            for date in residual_te_s.index.get_level_values(0).unique():
                date_mask = residual_te_s.index.get_level_values(0) == date
                r_day = residual_te_s.loc[date_mask].values
                f_day = mf_te_valid[date_mask.values if hasattr(date_mask, 'values') else date_mask, j]
                finite = np.isfinite(f_day) & np.isfinite(r_day)
                if finite.sum() < 30:
                    continue
                corr, _ = spearmanr(f_day[finite], r_day[finite])
                if np.isfinite(corr):
                    ric_day.append(corr)
            all_residual_ics[col].extend(ric_day)
            avg_ric = np.mean(ric_day) if ric_day else 0
            pos_pct = np.mean([v > 0 for v in ric_day]) if ric_day else 0
            logger.info(f"    Residual IC [{col}]: {avg_ric:+.4f} (>0: {pos_pct:.0%}, n={len(ric_day)})")

        # --- Stage 2: Train on moneyflow_v2 to predict residuals ---
        t2 = time.time()
        model_stage2 = train_xgb_stage2(
            X_mf_tr[mtr], y_tr[mtr] - model_base.predict(xgb.DMatrix(X_base_tr[mtr])),
            X_mf_va[mva], residual_va,
        )
        stage2_pred_te = model_stage2.predict(xgb.DMatrix(mf_te_valid))

        # --- Combined prediction: base + stage2 ---
        combined_pred = pred_base_te + stage2_pred_te
        combined_metrics = evaluate_rankic(combined_pred, y_te[mte], test_idx[mte])
        logger.info(f"  Two-stage: RankIC={combined_metrics['rank_ic_mean']:+.4f}, "
                    f"Spread={combined_metrics['top20_spread']:.4f} ({time.time()-t2:.1f}s)")

        delta_ric = combined_metrics["rank_ic_mean"] - base_metrics["rank_ic_mean"]
        delta_spread = combined_metrics["top20_spread"] - base_metrics["top20_spread"]
        logger.info(f"  Delta: RankIC={delta_ric:+.4f}, Spread={delta_spread:+.4f}")

        split_result = {
            "split": split_idx + 1,
            "test_period": (str(trade_dates[test_start_idx])[:10],
                           str(trade_dates[test_end_idx])[:10]),
            "baseline": base_metrics,
            "two_stage": combined_metrics,
            "delta_rank_ic": round(delta_ric, 6),
            "delta_spread": round(delta_spread, 6),
        }
        all_split_results.append(split_result)

    total_time = time.time() - t_total

    # --- Summary ---
    logger.info(f"\n{'='*60}")
    logger.info(f"RESIDUALIZED ABLATION SUMMARY ({total_time:.0f}s)")
    logger.info(f"{'='*60}")

    # Residual IC summary
    logger.info("\nPer-Factor Residual IC:")
    residual_ic_summary = {}
    for col in MONEYFLOW_V2_FACTORS:
        vals = all_residual_ics[col]
        if vals:
            avg = np.mean(vals)
            pos = np.mean([v > 0 for v in vals])
            residual_ic_summary[col] = {
                "avg_residual_ic": round(float(avg), 6),
                "pos_pct": round(float(pos), 4),
                "n_obs": len(vals),
                "significant": abs(avg) > 0.01 and pos > 0.55,
            }
            marker = "***" if abs(avg) > 0.01 and pos > 0.55 else "   "
            logger.info(f"  {marker} {col:<30} avg={avg:+.4f} >0={pos:.0%} (n={len(vals)})")
        else:
            logger.info(f"       {col:<30} no data")

    # Two-stage model summary
    logger.info("\nTwo-Stage Model vs Baseline:")
    base_rics = [r["baseline"]["rank_ic_mean"] for r in all_split_results]
    ts_rics = [r["two_stage"]["rank_ic_mean"] for r in all_split_results]
    deltas = [r["delta_rank_ic"] for r in all_split_results]

    avg_base_ric = np.mean(base_rics)
    avg_ts_ric = np.mean(ts_rics)
    avg_delta = np.mean(deltas)
    win_pct = np.mean([d > 0 for d in deltas])

    logger.info(f"  Baseline avg RankIC: {avg_base_ric:+.4f}")
    logger.info(f"  Two-stage avg RankIC: {avg_ts_ric:+.4f}")
    logger.info(f"  Avg delta: {avg_delta:+.4f}")
    logger.info(f"  Win rate (two-stage > baseline): {win_pct:.0%} ({sum(1 for d in deltas if d > 0)}/{len(deltas)})")

    base_spreads = [r["baseline"]["top20_spread"] for r in all_split_results]
    ts_spreads = [r["two_stage"]["top20_spread"] for r in all_split_results]
    logger.info(f"  Baseline avg Spread: {np.mean(base_spreads):.4f}")
    logger.info(f"  Two-stage avg Spread: {np.mean(ts_spreads):.4f}")

    verdict = "PASS" if avg_delta > 0.002 and win_pct >= 0.625 else "FAIL"
    logger.info(f"\n  Verdict: {verdict}")

    # Save results
    out_path = DATA_DIR / "phase4" / "residualized_ablation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "total_time_s": round(total_time, 1),
        "config": {
            "n_splits": len(all_split_results),
            "test_days": args.test_days,
            "train_days": args.train_days,
            "valid_days": args.valid_days,
            "base_features": len(base_cols),
            "moneyflow_v2_factors": MONEYFLOW_V2_FACTORS,
        },
        "residual_ic": residual_ic_summary,
        "summary": {
            "avg_baseline_rank_ic": round(float(avg_base_ric), 6),
            "avg_twostage_rank_ic": round(float(avg_ts_ric), 6),
            "avg_delta_rank_ic": round(float(avg_delta), 6),
            "win_rate": round(float(win_pct), 4),
            "avg_baseline_spread": round(float(np.mean(base_spreads)), 6),
            "avg_twostage_spread": round(float(np.mean(ts_spreads)), 6),
            "verdict": verdict,
        },
        "splits": all_split_results,
    }

    from utils.json_utils import json_default
    with open(str(out_path), "w") as f:
        json.dump(report, f, indent=2, default=json_default)
    logger.info(f"\nSaved: {out_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
