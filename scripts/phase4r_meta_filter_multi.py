"""Phase 4R — Multi-feature meta-filter for XGB Top100.

Combines disagreement + volatility + momentum + liquidity into a composite
meta-score to filter XGB Top100 down to Top75.

Meta-score formula (simple linear):
    meta_score = -0.3*disagreement_z - 0.3*volatility_z + 0.2*momentum_z + 0.2*liquidity_z

Higher meta_score => keep; lower => remove.

Usage:
    python -m scripts.phase4r_meta_filter_multi
"""
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "storage"
SPLIT_DIR = DATA_DIR / "phase4e_24split"
FEATURE_CACHE = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"

TOP_K = 100
KEEP_K = 75  # after filtering
RETURN_COL = "__pnl_return_1d"
META_FEATURES_FROM_CACHE = ["STD20", "ROC20", "amount_raw"]

# Composite weights (sign-aware: negative weight means "penalise high values")
WEIGHTS = {
    "disagreement_z": -0.30,
    "volatility_z":   -0.30,
    "momentum_z":     +0.20,
    "liquidity_z":    +0.20,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_oof_predictions() -> dict[str, pd.Series]:
    """Load per-split OOF predictions for xgb/lgb/catboost."""
    models = {"xgb": [], "lgb": [], "catboost": []}
    for split_id in range(24):
        for model_name in models:
            path = SPLIT_DIR / f"split{split_id:02d}_{model_name}.pkl"
            if not path.exists():
                continue
            with open(path, "rb") as f:
                pred = pickle.load(f)
            if isinstance(pred, pd.Series):
                models[model_name].append(pred)
            elif isinstance(pred, pd.DataFrame):
                models[model_name].append(pred.iloc[:, 0])

    result = {}
    for model_name, parts in models.items():
        if not parts:
            logger.warning("No predictions found for %s", model_name)
            continue
        combined = pd.concat(parts)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        result[model_name] = combined
        logger.info("Loaded %s OOF: %d rows, %d dates",
                    model_name, len(combined),
                    combined.index.get_level_values("datetime").nunique())
    return result


def load_cache_features() -> pd.DataFrame:
    """Load STD20, ROC20, amount_raw and forward return from feature cache."""
    cols = META_FEATURES_FROM_CACHE + [RETURN_COL]
    logger.info("Loading feature cache columns: %s", cols)
    df = pd.read_parquet(FEATURE_CACHE, columns=cols)
    logger.info("Feature cache: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Meta-feature computation
# ---------------------------------------------------------------------------

def compute_rank_features(oof: dict[str, pd.Series]) -> pd.DataFrame:
    """Compute per-stock per-date rank-based meta features.

    Returns DataFrame with columns:
        - disagreement: std of rank-pct across models
        - xgb_rank_pct: XGB within-day percentile rank (1 = best)
        - ensemble_rank_pct: mean rank across models
        - rank_gap: |xgb_rank_pct - ensemble_rank_pct|
    """
    model_names = list(oof.keys())
    assert "xgb" in model_names

    # Common index
    common_idx = oof[model_names[0]].index
    for m in model_names[1:]:
        common_idx = common_idx.intersection(oof[m].index)
    logger.info("Common index across %d models: %d rows", len(model_names), len(common_idx))

    # Per-day percentile ranks
    rank_dict = {}
    for m in model_names:
        preds = oof[m].loc[common_idx]
        rank_dict[m] = preds.groupby(level="datetime").rank(pct=True)

    rank_df = pd.DataFrame(rank_dict)

    result = pd.DataFrame(index=common_idx)
    result["disagreement"] = rank_df.std(axis=1)
    result["xgb_rank_pct"] = rank_dict["xgb"]
    result["ensemble_rank_pct"] = rank_df.mean(axis=1)
    result["rank_gap"] = (result["xgb_rank_pct"] - result["ensemble_rank_pct"]).abs()

    return result


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_meta_filter(
    xgb_preds: pd.Series,
    rank_features: pd.DataFrame,
    cache_features: pd.DataFrame,
):
    """Run composite meta-filter and compare Top100 vs filtered Top75.

    Vectorized implementation — avoids per-day Python loop.
    """

    # Merge all meta features
    meta = rank_features.copy()
    meta = meta.join(cache_features[META_FEATURES_FROM_CACHE], how="inner")
    meta = meta.join(cache_features[[RETURN_COL]], how="inner")

    # Rename for clarity
    meta = meta.rename(columns={
        "STD20": "volatility",
        "ROC20": "momentum",
        "amount_raw": "liquidity_raw",
    })
    meta["liquidity"] = np.log1p(meta["liquidity_raw"].clip(lower=0))

    # Drop rows with NaN in any meta feature
    needed = ["disagreement", "volatility", "momentum", "liquidity"]
    before = len(meta)
    meta = meta.dropna(subset=needed + [RETURN_COL])
    logger.info("After dropna: %d rows (dropped %d)", len(meta), before - len(meta))

    # Add XGB predictions as column
    meta["xgb_pred"] = xgb_preds.reindex(meta.index)
    meta = meta.dropna(subset=["xgb_pred"])
    logger.info("After adding xgb_pred: %d rows", len(meta))

    # ---- Vectorized: compute per-day XGB rank (descending = high pred = low rank number) ----
    g = meta.groupby(level="datetime")

    # Count per day — filter days with enough stocks
    day_counts = g.size()
    valid_dates = day_counts[day_counts >= TOP_K * 2].index
    meta = meta.loc[meta.index.get_level_values("datetime").isin(valid_dates)]
    logger.info("Days with >= %d stocks: %d", TOP_K * 2, len(valid_dates))

    g = meta.groupby(level="datetime")

    # XGB rank within day (ascending rank: 1=lowest pred, N=highest pred)
    meta["xgb_rank"] = g["xgb_pred"].rank(ascending=True)
    meta["day_count"] = g["xgb_pred"].transform("size")

    # Top100 = rank > (day_count - TOP_K), Bottom100 = rank <= TOP_K
    meta["is_top100"] = meta["xgb_rank"] > (meta["day_count"] - TOP_K)
    meta["is_bottom100"] = meta["xgb_rank"] <= TOP_K

    # ---- Z-score the meta features WITHIN Top100 per day ----
    top100 = meta[meta["is_top100"]].copy()
    logger.info("Top100 pool: %d rows across %d days",
                len(top100), top100.index.get_level_values("datetime").nunique())

    g_top = top100.groupby(level="datetime")
    for feat, zcol in [
        ("disagreement", "disagreement_z"),
        ("volatility", "volatility_z"),
        ("momentum", "momentum_z"),
        ("liquidity", "liquidity_z"),
    ]:
        mu = g_top[feat].transform("mean")
        sigma = g_top[feat].transform("std")
        top100[zcol] = ((top100[feat] - mu) / sigma.replace(0, np.nan)).fillna(0)

    # Composite meta score
    top100["meta_score"] = sum(
        WEIGHTS[col] * top100[col] for col in WEIGHTS
    )

    # ---- Filter: keep top KEEP_K by meta_score per day ----
    # Rank meta_score within day (descending: highest = rank 1)
    top100["meta_rank"] = top100.groupby(level="datetime")["meta_score"].rank(
        ascending=False, method="first"
    )
    top100["is_kept"] = top100["meta_rank"] <= KEEP_K

    # ---- Aggregate per day ----
    fwd_col = RETURN_COL

    # Bottom100 mean return per day
    bottom100 = meta[meta["is_bottom100"]]
    bottom_ret = bottom100.groupby(level="datetime")[fwd_col].mean().rename("bottom100_ret")

    # Top100 mean return per day
    top100_ret = top100.groupby(level="datetime")[fwd_col].mean().rename("top100_ret")

    # Filtered (kept) mean return
    kept = top100[top100["is_kept"]]
    removed = top100[~top100["is_kept"]]

    kept_ret = kept.groupby(level="datetime")[fwd_col].mean().rename("filtered_ret")
    removed_ret = removed.groupby(level="datetime")[fwd_col].mean().rename("removed_ret")

    # Counts
    n_kept = kept.groupby(level="datetime").size().rename("n_filtered")
    n_removed = removed.groupby(level="datetime").size().rename("n_removed")
    n_top100 = top100.groupby(level="datetime").size().rename("n_top100")
    n_universe = meta.groupby(level="datetime").size().rename("n_universe")

    # Feature profiles
    kept_dis = kept.groupby(level="datetime")["disagreement"].mean().rename("kept_disagreement")
    removed_dis = removed.groupby(level="datetime")["disagreement"].mean().rename("removed_disagreement")
    kept_vol = kept.groupby(level="datetime")["volatility"].mean().rename("kept_volatility")
    removed_vol = removed.groupby(level="datetime")["volatility"].mean().rename("removed_volatility")
    kept_mom = kept.groupby(level="datetime")["momentum"].mean().rename("kept_momentum")
    removed_mom = removed.groupby(level="datetime")["momentum"].mean().rename("removed_momentum")

    # Combine
    df = pd.concat([
        n_universe, n_top100, n_kept, n_removed,
        top100_ret, kept_ret, removed_ret, bottom_ret,
        kept_dis, removed_dis, kept_vol, removed_vol, kept_mom, removed_mom,
    ], axis=1)
    df = df.dropna(subset=["top100_ret", "filtered_ret", "bottom100_ret"])
    df["spread_top100"] = df["top100_ret"] - df["bottom100_ret"]
    df["spread_filtered"] = df["filtered_ret"] - df["bottom100_ret"]
    df = df.reset_index().rename(columns={"datetime": "date"})

    logger.info("Produced results for %d dates", len(df))
    return df


def print_results(df: pd.DataFrame):
    """Print comprehensive results."""
    print("\n" + "=" * 78)
    print("Phase 4R — Multi-Feature Meta-Filter Results")
    print("=" * 78)

    n = len(df)
    print(f"\nPeriod: {df['date'].min()} → {df['date'].max()}  ({n} trading days)")
    print(f"Filter: XGB Top{TOP_K} → Top{KEEP_K}  (remove bottom {TOP_K - KEEP_K} by meta_score)")
    print(f"Weights: {WEIGHTS}")

    # --- Spread comparison ---
    sp_unf = df["spread_top100"]
    sp_fil = df["spread_filtered"]
    diff = sp_fil - sp_unf

    print(f"\n{'--- Spread (Long - Short) ---':^78}")
    print(f"  Unfiltered Top{TOP_K}:  mean={sp_unf.mean():+.5f}  "
          f"median={sp_unf.median():+.5f}  std={sp_unf.std():.5f}")
    print(f"  Filtered  Top{KEEP_K}:  mean={sp_fil.mean():+.5f}  "
          f"median={sp_fil.median():+.5f}  std={sp_fil.std():.5f}")
    print(f"  Improvement:      mean={diff.mean():+.5f}  "
          f"median={diff.median():+.5f}")

    # --- Return comparison (long side only) ---
    print(f"\n{'--- Long-side Return ---':^78}")
    print(f"  Top{TOP_K} (unfiltered):  {df['top100_ret'].mean():+.5f}")
    print(f"  Top{KEEP_K} (filtered):   {df['filtered_ret'].mean():+.5f}")
    print(f"  Removed stocks:       {df['removed_ret'].mean():+.5f}")
    long_improve = df['filtered_ret'].mean() - df['top100_ret'].mean()
    print(f"  Long-side improvement: {long_improve:+.5f}")

    # --- Win rate ---
    wins = (sp_fil > sp_unf).sum()
    ties = (sp_fil == sp_unf).sum()
    wr = wins / n
    print(f"\n{'--- Consistency ---':^78}")
    print(f"  Win days (filtered > unfiltered): {wins}/{n} ({wr:.1%})")
    print(f"  Tie days: {ties}")

    # Paired t-test
    t_stat, p_val = ttest_rel(sp_fil, sp_unf)
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")

    # --- Feature profile: kept vs removed ---
    print(f"\n{'--- Feature Profile (mean) ---':^78}")
    print(f"  {'Feature':>20s}  {'Kept':>10s}  {'Removed':>10s}  {'Delta':>10s}")
    for feat in ["disagreement", "volatility", "momentum"]:
        k = df[f"kept_{feat}"].mean()
        r = df[f"removed_{feat}"].mean()
        print(f"  {feat:>20s}  {k:10.5f}  {r:10.5f}  {k - r:+10.5f}")

    # --- Quarterly breakdown ---
    df_copy = df.copy()
    df_copy["quarter"] = pd.to_datetime(df_copy["date"]).dt.to_period("Q")
    qtr = df_copy.groupby("quarter").agg({
        "spread_top100": "mean",
        "spread_filtered": "mean",
        "date": "count",
    }).rename(columns={"date": "n_days"})
    qtr["improve"] = qtr["spread_filtered"] - qtr["spread_top100"]

    print(f"\n{'--- Quarterly Breakdown ---':^78}")
    print(f"  {'Quarter':>10s}  {'Days':>5s}  {'Unfiltered':>12s}  {'Filtered':>12s}  {'Improve':>12s}")
    for q, row in qtr.iterrows():
        tag = " *" if row["improve"] > 0 else ""
        print(f"  {str(q):>10s}  {int(row['n_days']):5d}  "
              f"{row['spread_top100']:+12.5f}  {row['spread_filtered']:+12.5f}  "
              f"{row['improve']:+12.5f}{tag}")

    # --- Verdict ---
    print("\n" + "-" * 78)
    mean_imp = diff.mean()
    if mean_imp > 0 and wr > 0.52 and p_val < 0.1:
        print("VERDICT: Multi-feature meta-filter is effective.")
        print(f"  Spread improvement: {mean_imp:+.5f} | Win rate: {wr:.1%} | p={p_val:.4f}")
    elif mean_imp > 0 and wr > 0.50:
        print("VERDICT: Directionally positive but weak significance.")
        print(f"  Spread improvement: {mean_imp:+.5f} | Win rate: {wr:.1%} | p={p_val:.4f}")
        print("  Consider: tuning weights, adding features, or LGB-based meta-model.")
    else:
        print("VERDICT: Multi-feature meta-filter does NOT improve spread.")
        print(f"  Spread change: {mean_imp:+.5f} | Win rate: {wr:.1%} | p={p_val:.4f}")
    print("-" * 78)

    return {
        "spread_unfiltered": sp_unf.mean(),
        "spread_filtered": sp_fil.mean(),
        "improvement": mean_imp,
        "win_rate": wr,
        "p_value": p_val,
        "n_days": n,
    }


def main():
    # 1. Load OOF predictions
    oof = load_oof_predictions()
    if "xgb" not in oof or len(oof) < 2:
        print("FAILED: Need XGB + at least one other model OOF.")
        sys.exit(1)

    # 2. Compute rank-based meta features (disagreement, rank_gap, etc.)
    rank_features = compute_rank_features(oof)

    # 3. Load cache features (STD20, ROC20, amount_raw, forward return)
    cache_features = load_cache_features()

    # 4-6. Run meta-filter
    results_df = run_meta_filter(oof["xgb"], rank_features, cache_features)
    if results_df is None:
        print("FAILED: No results.")
        sys.exit(1)

    # 7. Print and save
    summary = print_results(results_df)

    # Save
    out_path = DATA_DIR / "phase4r_meta_filter_multi_results.parquet"
    results_df.to_parquet(out_path)
    logger.info("Saved daily results to %s", out_path)

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
