"""Compare factor IC before and after cross-sectional rank preprocessing.

Tests whether preprocessing (rank normalization) rescues failed factors.

Logic:
  - For each factor, compute RankIC with raw values and rank-normalized values
  - RankIC is rank-invariant → rank preprocessing should NOT change single-factor RankIC
  - But Pearson IC CAN change (sensitive to outliers/distribution)
  - The real question is: does preprocessing help in a MODEL context (XGB ablation)?

This script answers both:
  1. Single-factor IC: raw vs rank (sanity check — RankIC should be identical)
  2. XGB ablation: base174 vs base174+failed_rank (the real test)

Usage:
    python scripts/factor_preprocess_compare.py
"""
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
OUTPUT_DIR = DATA_DIR / "phase4"


def load_data(test_days: int = 250):
    """Load feature cache + supplementary factors, filtered to test window."""
    logger.info("Loading feature cache...")
    cache = pd.read_parquet(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet")

    all_dates = cache.index.get_level_values(0)
    cutoff = pd.Timestamp(datetime.now() - timedelta(days=test_days))
    mask = all_dates >= cutoff
    cache = cache.loc[mask]
    logger.info(f"Cache: {cache.shape}, test window: {cutoff.date()} ~ {all_dates.max().date()}")

    label = cache["__label_5d"]

    # Base 174 features (what champion model uses)
    meta_cols = ["__label_5d", "__pnl_return_1d", "_close", "_ma5", "_ma20", "holder_num"]
    cross_market = [c for c in cache.columns if any(c.startswith(p) for p in ["hsi_", "hstech_", "nasdaq_"])]
    exclude = set(meta_cols + cross_market)
    base_cols = [c for c in cache.columns if c not in exclude]
    base_df = cache[base_cols]
    logger.info(f"Base features: {len(base_cols)}")

    # Load supplementary (failed) factors
    supp_frames = {}

    for name, path, factor_cols in [
        ("moneyflow_v2", DATA_DIR / "moneyflow_v2.parquet",
         ["main_flow_adv", "order_imbalance", "large_small_divergence",
          "flow_zscore_60d", "flow_persistence_10d", "flow_industry_rank"]),
        ("block_trade_v2", DATA_DIR / "block_trade_v2.parquet",
         ["bt_discount", "bt_discount_5d_avg", "bt_has_recent",
          "bt_recency_decay", "bt_volume_ratio"]),
        ("derived_mf_cyq", DATA_DIR / "derived_moneyflow_cyq.parquet",
         ["net_flow", "net_flow_5d_change", "net_flow_20d_change",
          "net_flow_vol_20d", "big_order_ratio",
          "winner_rate_change_5d", "winner_rate_change_20d",
          "cost_concentration", "cost_concentration_change_5d"]),
    ]:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "qlib_code" in df.columns and "date" in df.columns:
            available = [c for c in factor_cols if c in df.columns]
            df = df[["qlib_code", "date"] + available].copy()
            df["date"] = pd.to_datetime(df["date"])
            df["qlib_code"] = df["qlib_code"].str.lower()
            df = df.drop_duplicates(subset=["date", "qlib_code"], keep="last")
            df = df.set_index(["date", "qlib_code"])
            df.index.names = ["datetime", "instrument"]
            target = pd.DataFrame(index=cache.index)
            df = target.join(df, how="left")[available]
            supp_frames[name] = df

    # Event factors (already has datetime/instrument index)
    path = DATA_DIR / "event_factors_v2.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        evt_cols = [c for c in df.columns if c.startswith(("fc_", "ti_"))]
        if evt_cols:
            df = df[evt_cols].reindex(cache.index)
            supp_frames["events"] = df

    if supp_frames:
        supp_all = pd.concat(supp_frames.values(), axis=1)
    else:
        supp_all = pd.DataFrame(index=cache.index)

    supp_all = supp_all.replace([np.inf, -np.inf], np.nan)
    logger.info(f"Supplementary factors: {supp_all.shape[1]} columns")

    return base_df, supp_all, label


def cross_sectional_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Apply per-date rank percentile normalization."""
    result = df.copy()
    for col in result.columns:
        result[col] = result[col].groupby(level=0).rank(pct=True)
    return result


def compute_ic_compare(raw_series: pd.Series, rank_series: pd.Series,
                       label: pd.Series) -> dict:
    """Compare IC of raw vs rank-preprocessed factor."""
    results = {}
    for tag, factor in [("raw", raw_series), ("rank", rank_series)]:
        df = pd.DataFrame({"f": factor, "l": label}).dropna()
        if len(df) < 1000:
            continue
        ic_list, ric_list = [], []
        for dt, g in df.groupby(level=0):
            if len(g) < 30:
                continue
            f, l = g["f"].values, g["l"].values
            if np.std(f) < 1e-10:
                continue
            ic = np.corrcoef(f, l)[0, 1]
            fr = stats.rankdata(f)
            lr = stats.rankdata(l)
            ric = np.corrcoef(fr, lr)[0, 1]
            if np.isfinite(ic) and np.isfinite(ric):
                ic_list.append(ic)
                ric_list.append(ric)
        if ic_list:
            results[tag] = {
                "ic_mean": float(np.mean(ic_list)),
                "rank_ic_mean": float(np.mean(ric_list)),
                "icir": float(np.mean(ic_list) / (np.std(ic_list) + 1e-8)),
                "rank_icir": float(np.mean(ric_list) / (np.std(ric_list) + 1e-8)),
                "n_days": len(ic_list),
            }
    return results


def run_xgb_ablation(base_df: pd.DataFrame, supp_raw: pd.DataFrame,
                     supp_rank: pd.DataFrame, label: pd.Series,
                     n_splits: int = 6) -> dict:
    """Run XGB ablation: base174 vs base174+failed_raw vs base174+failed_rank.

    Uses rolling splits for robustness.
    """
    import xgboost as xgb

    XGB_PARAMS = {
        "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
        "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
        "objective": "reg:squarederror", "nthread": 12, "verbosity": 0, "seed": 42,
    }

    dates = sorted(label.index.get_level_values(0).unique())
    test_days = 20
    valid_days = 40
    train_days = 250

    configs = {
        "base_174": base_df,
        "base_174_plus_failed_raw": base_df.join(supp_raw.add_prefix("failed_raw_"), how="left"),
        "base_174_plus_failed_rank": base_df.join(supp_rank.add_prefix("failed_rk_"), how="left"),
    }

    results = {k: [] for k in configs}

    for split_idx in range(n_splits):
        test_end_idx = len(dates) - 1 - split_idx * test_days
        test_start_idx = test_end_idx - test_days + 1
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - valid_days + 1
        train_end_idx = valid_start_idx - 1
        train_start_idx = max(0, train_end_idx - train_days + 1)

        if train_start_idx >= train_end_idx or valid_start_idx >= valid_end_idx:
            break

        train_dates = set(dates[train_start_idx:train_end_idx + 1])
        valid_dates = set(dates[valid_start_idx:valid_end_idx + 1])
        test_dates = set(dates[test_start_idx:test_end_idx + 1])

        logger.info(f"Split {split_idx}: train {len(train_dates)}d, valid {len(valid_dates)}d, test {len(test_dates)}d")

        for config_name, X_df in configs.items():
            # Split data
            all_dt = X_df.index.get_level_values(0)
            train_mask = all_dt.isin(train_dates)
            valid_mask = all_dt.isin(valid_dates)
            test_mask = all_dt.isin(test_dates)

            X_train = X_df.loc[train_mask].values.astype(np.float32)
            y_train = label.loc[train_mask].values.astype(np.float32)
            X_valid = X_df.loc[valid_mask].values.astype(np.float32)
            y_valid = label.loc[valid_mask].values.astype(np.float32)
            X_test = X_df.loc[test_mask].values.astype(np.float32)
            y_test = label.loc[test_mask].values.astype(np.float32)
            test_index = X_df.loc[test_mask].index

            # Filter NaN labels
            m_tr = np.isfinite(y_train); X_train, y_train = X_train[m_tr], y_train[m_tr]
            m_va = np.isfinite(y_valid); X_valid, y_valid = X_valid[m_va], y_valid[m_va]
            m_te = np.isfinite(y_test); X_test, y_test = X_test[m_te], y_test[m_te]
            test_index = test_index[m_te]

            if len(X_train) < 100 or len(X_test) < 100:
                continue

            # Train
            dt = xgb.DMatrix(X_train, label=y_train)
            dv = xgb.DMatrix(X_valid, label=y_valid)
            model = xgb.train(XGB_PARAMS, dt, num_boost_round=400,
                              evals=[(dv, "valid")], early_stopping_rounds=50,
                              verbose_eval=0)

            # Predict
            pred = model.predict(xgb.DMatrix(X_test))

            # Evaluate: RankIC + Spread
            pred_s = pd.Series(pred, index=test_index)
            label_s = pd.Series(y_test, index=test_index)

            ric_list, spread_list = [], []
            for dt_val, g_idx in pred_s.groupby(level=0):
                g_pred = pred_s.loc[dt_val]
                g_label = label_s.loc[dt_val]
                common = g_pred.index.intersection(g_label.index)
                if len(common) < 40:
                    continue
                p = g_pred.loc[common].values
                l = g_label.loc[common].values
                m = np.isfinite(p) & np.isfinite(l)
                if m.sum() < 40:
                    continue
                ric = stats.spearmanr(p[m], l[m]).statistic
                if np.isfinite(ric):
                    ric_list.append(ric)
                # Spread
                tmp = pd.DataFrame({"p": p[m], "l": l[m]}).sort_values("p", ascending=False)
                spread_list.append(tmp.head(20)["l"].mean() - tmp.tail(20)["l"].mean())

            if ric_list:
                results[config_name].append({
                    "split": split_idx,
                    "rank_ic": float(np.mean(ric_list)),
                    "spread": float(np.mean(spread_list)) if spread_list else 0,
                    "n_features": X_train.shape[1],
                })

    return results


def main():
    t_start = time.time()

    base_df, supp_all, label = load_data(test_days=500)

    # Drop factors with very low coverage
    coverage = supp_all.notna().mean()
    valid_cols = coverage[coverage > 0.1].index.tolist()
    supp_raw = supp_all[valid_cols]
    logger.info(f"Valid supplementary factors (>10% coverage): {len(valid_cols)}")

    # Rank-normalize supplementary factors
    logger.info("Rank-normalizing supplementary factors...")
    supp_rank = cross_sectional_rank(supp_raw)

    # ---- Part 1: Single-factor IC comparison (raw vs rank) ----
    logger.info("\n=== Part 1: Single-factor IC comparison ===")
    ic_compare = {}
    for col in valid_cols:
        result = compute_ic_compare(supp_raw[col], supp_rank[col], label)
        if result:
            ic_compare[col] = result

    logger.info(f"\n{'Factor':<35} {'Raw IC':>8} {'Rank IC':>8} {'Raw RIC':>8} {'Rank RIC':>8} {'ΔIC':>8} {'ΔRIC':>8}")
    logger.info("-" * 95)
    for col, r in sorted(ic_compare.items(), key=lambda x: abs(x[1].get("raw", {}).get("rank_ic_mean", 0)), reverse=True):
        raw = r.get("raw", {})
        rank = r.get("rank", {})
        raw_ic = raw.get("ic_mean", 0)
        rank_ic = rank.get("ic_mean", 0)
        raw_ric = raw.get("rank_ic_mean", 0)
        rank_ric = rank.get("rank_ic_mean", 0)
        delta_ic = rank_ic - raw_ic
        delta_ric = rank_ric - raw_ric
        logger.info(
            f"{col:<35} {raw_ic:>+8.4f} {rank_ic:>+8.4f} {raw_ric:>+8.4f} {rank_ric:>+8.4f} "
            f"{delta_ic:>+8.4f} {delta_ric:>+8.4f}"
        )

    # ---- Part 2: XGB ablation (the real test) ----
    logger.info("\n=== Part 2: XGB Ablation (6 rolling splits) ===")
    ablation = run_xgb_ablation(base_df, supp_raw, supp_rank, label, n_splits=6)

    logger.info(f"\n{'Config':<35} {'Avg RankIC':>10} {'Med RankIC':>10} {'Avg Spread':>10} {'#Splits':>8} {'#Features':>10}")
    logger.info("-" * 90)
    for config_name, splits in ablation.items():
        if not splits:
            continue
        rics = [s["rank_ic"] for s in splits]
        spreads = [s["spread"] for s in splits]
        n_feat = splits[0]["n_features"] if splits else 0
        logger.info(
            f"{config_name:<35} {np.mean(rics):>+10.4f} {np.median(rics):>+10.4f} "
            f"{np.mean(spreads)*100:>+9.3f}% {len(splits):>8} {n_feat:>10}"
        )

    # ---- Part 3: Also test rank-preprocessing custom+flow in base ----
    logger.info("\n=== Part 3: Base174 with rank-preprocessed custom+flow ===")
    custom_cols = ["pe", "pb", "turn_raw", "amount_raw", "pe_mom20", "pb_mom20",
                   "turn_anom20", "turn_anom60", "amount_anom20", "turn_vol20",
                   "ep", "bp", "price_pos20"]
    flow_cols = ["flow_net_mf_latest", "flow_net_mf_5d", "flow_net_mf_20d_avg"]
    preprocess_cols = [c for c in custom_cols + flow_cols if c in base_df.columns]

    base_ranked = base_df.copy()
    for col in preprocess_cols:
        base_ranked[col] = base_ranked[col].groupby(level=0).rank(pct=True)

    configs_v2 = {
        "base_174_raw": base_df,
        "base_174_custom_flow_ranked": base_ranked,
    }

    results_v2 = {k: [] for k in configs_v2}
    dates = sorted(label.index.get_level_values(0).unique())
    test_days_per_split = 20
    valid_days = 40
    train_days = 250

    import xgboost as xgb
    XGB_PARAMS = {
        "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8789,
        "colsample_bytree": 0.8879, "reg_alpha": 205.6999, "reg_lambda": 580.9768,
        "objective": "reg:squarederror", "nthread": 12, "verbosity": 0, "seed": 42,
    }

    for split_idx in range(6):
        test_end_idx = len(dates) - 1 - split_idx * test_days_per_split
        test_start_idx = test_end_idx - test_days_per_split + 1
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - valid_days + 1
        train_end_idx = valid_start_idx - 1
        train_start_idx = max(0, train_end_idx - train_days + 1)
        if train_start_idx >= train_end_idx:
            break

        train_dates = set(dates[train_start_idx:train_end_idx + 1])
        valid_dates = set(dates[valid_start_idx:valid_end_idx + 1])
        test_dates = set(dates[test_start_idx:test_end_idx + 1])

        for config_name, X_df in configs_v2.items():
            all_dt = X_df.index.get_level_values(0)
            X_tr = X_df.loc[all_dt.isin(train_dates)].values.astype(np.float32)
            y_tr = label.loc[all_dt.isin(train_dates)].values.astype(np.float32)
            X_va = X_df.loc[all_dt.isin(valid_dates)].values.astype(np.float32)
            y_va = label.loc[all_dt.isin(valid_dates)].values.astype(np.float32)
            X_te = X_df.loc[all_dt.isin(test_dates)].values.astype(np.float32)
            y_te = label.loc[all_dt.isin(test_dates)].values.astype(np.float32)
            te_idx = X_df.loc[all_dt.isin(test_dates)].index

            m_tr = np.isfinite(y_tr); X_tr, y_tr = X_tr[m_tr], y_tr[m_tr]
            m_va = np.isfinite(y_va); X_va, y_va = X_va[m_va], y_va[m_va]
            m_te = np.isfinite(y_te); X_te, y_te = X_te[m_te], y_te[m_te]
            te_idx = te_idx[m_te]

            if len(X_tr) < 100 or len(X_te) < 100:
                continue

            dt_m = xgb.DMatrix(X_tr, label=y_tr)
            dv_m = xgb.DMatrix(X_va, label=y_va)
            model = xgb.train(XGB_PARAMS, dt_m, num_boost_round=400,
                              evals=[(dv_m, "valid")], early_stopping_rounds=50,
                              verbose_eval=0)
            pred = model.predict(xgb.DMatrix(X_te))

            pred_s = pd.Series(pred, index=te_idx)
            label_s = pd.Series(y_te, index=te_idx)
            ric_list, spread_list = [], []
            for _, g in pred_s.groupby(level=0):
                gl = label_s.reindex(g.index).dropna()
                common = g.index.intersection(gl.index)
                if len(common) < 40:
                    continue
                p, l = g.loc[common].values, gl.loc[common].values
                m = np.isfinite(p) & np.isfinite(l)
                if m.sum() < 40:
                    continue
                ric = stats.spearmanr(p[m], l[m]).statistic
                if np.isfinite(ric):
                    ric_list.append(ric)
                tmp = pd.DataFrame({"p": p[m], "l": l[m]}).sort_values("p", ascending=False)
                spread_list.append(tmp.head(20)["l"].mean() - tmp.tail(20)["l"].mean())

            if ric_list:
                results_v2[config_name].append({
                    "split": split_idx,
                    "rank_ic": float(np.mean(ric_list)),
                    "spread": float(np.mean(spread_list)) if spread_list else 0,
                })

    logger.info(f"\n{'Config':<35} {'Avg RankIC':>10} {'Med RankIC':>10} {'Avg Spread':>10} {'#Splits':>8}")
    logger.info("-" * 80)
    for config_name, splits in results_v2.items():
        if not splits:
            continue
        rics = [s["rank_ic"] for s in splits]
        spreads = [s["spread"] for s in splits]
        logger.info(
            f"{config_name:<35} {np.mean(rics):>+10.4f} {np.median(rics):>+10.4f} "
            f"{np.mean(spreads)*100:>+9.3f}% {len(splits):>8}"
        )

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.0f}s")

    # Save results
    output = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "ic_compare": {k: v for k, v in ic_compare.items()},
        "xgb_ablation": {k: v for k, v in ablation.items()},
        "base_preprocess_ablation": {k: v for k, v in results_v2.items()},
    }
    out_path = OUTPUT_DIR / "preprocess_compare.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                   default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else str(o)))
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
