"""Phase 4R — Meta-filter prototype.

Tests CX hypothesis: ensemble ranking for recall, meta-filter for head selection.
Uses existing 24-split ensemble OOF predictions to:
  1. Load XGB, LGB, CatBoost OOF predictions from phase4e_24split/
  2. Compute model_disagreement per stock per date (std of ranks across models)
  3. Load forward returns
  4. Test: among XGB Top100, do stocks with LOW disagreement outperform HIGH?
  5. Simple meta-filter: remove Top100 stocks where disagreement > 75th percentile
  6. Compare: filtered Top75 spread vs unfiltered Top100 spread
  7. Print results

Usage:
    python -m scripts.phase4r_meta_filter_proto
"""
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "storage"
SPLIT_DIR = DATA_DIR / "phase4e_24split"
FEATURE_CACHE = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
LABEL_COL = "__label_5d"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_oof_predictions() -> dict[str, pd.Series]:
    """Load all per-split OOF predictions and concatenate per model.

    Returns:
        {"xgb": Series, "lgb": Series, "catboost": Series}
        Each Series has MultiIndex (datetime, instrument).
    """
    models = {"xgb": [], "lgb": [], "catboost": []}

    # Find all split files
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
                # Take first column
                models[model_name].append(pred.iloc[:, 0])

    result = {}
    for model_name, parts in models.items():
        if not parts:
            logger.warning("No predictions found for %s", model_name)
            continue
        combined = pd.concat(parts)
        # Remove duplicates (overlapping splits) — keep last
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        result[model_name] = combined
        logger.info("Loaded %s OOF: %d predictions, %d dates",
                    model_name, len(combined),
                    combined.index.get_level_values("datetime").nunique())

    return result


def load_forward_returns() -> pd.Series:
    """Load forward 5-day returns from feature cache."""
    logger.info("Loading forward returns from feature cache...")
    label = pd.read_parquet(FEATURE_CACHE, columns=[LABEL_COL])
    s = label[LABEL_COL]
    logger.info("Forward returns: %d rows, %d dates",
                len(s), s.index.get_level_values("datetime").nunique())
    return s


def compute_disagreement(oof: dict[str, pd.Series]) -> pd.Series:
    """Compute model disagreement = std of per-day ranks across models.

    For each date:
      1. Rank each model's predictions (percentile rank within that day)
      2. Compute std of percentile ranks across models for each stock
    Higher std = models disagree more on this stock's relative ranking.
    """
    model_names = list(oof.keys())
    if len(model_names) < 2:
        raise ValueError(f"Need at least 2 models, got {model_names}")

    # Find common index
    common_idx = oof[model_names[0]].index
    for m in model_names[1:]:
        common_idx = common_idx.intersection(oof[m].index)
    logger.info("Common index across %d models: %d rows", len(model_names), len(common_idx))

    # Compute per-day percentile ranks for each model
    ranks_list = []
    for m in model_names:
        preds = oof[m].loc[common_idx]
        # Per-day percentile rank
        ranks = preds.groupby(level="datetime").rank(pct=True)
        ranks_list.append(ranks)

    # Stack and compute std across models
    rank_df = pd.concat(ranks_list, axis=1, keys=model_names)
    disagreement = rank_df.std(axis=1)
    disagreement.name = "disagreement"

    logger.info("Disagreement stats: mean=%.4f, median=%.4f, std=%.4f",
                disagreement.mean(), disagreement.median(), disagreement.std())

    return disagreement


def run_meta_filter_test(
    xgb_preds: pd.Series,
    disagreement: pd.Series,
    forward_returns: pd.Series,
    top_k: int = 100,
    filter_pct: float = 0.75,
):
    """Test meta-filter hypothesis.

    Among XGB Top100 per day:
    - LOW disagreement group: disagreement <= 75th percentile (Top75)
    - HIGH disagreement group: disagreement > 75th percentile (Bottom25)
    Compare average forward returns.
    """
    # Align all three series
    common_idx = xgb_preds.index.intersection(disagreement.index).intersection(forward_returns.index)
    xgb = xgb_preds.loc[common_idx]
    dis = disagreement.loc[common_idx]
    fwd = forward_returns.loc[common_idx]

    dates = xgb.index.get_level_values("datetime").unique().sort_values()
    logger.info("Test period: %s to %s (%d dates)", dates[0], dates[-1], len(dates))

    # Per-day analysis
    records = []
    for dt in dates:
        try:
            xgb_day = xgb.xs(dt, level="datetime")
            dis_day = dis.xs(dt, level="datetime")
            fwd_day = fwd.xs(dt, level="datetime")
        except KeyError:
            continue

        common_inst = xgb_day.index.intersection(dis_day.index).intersection(fwd_day.index)
        if len(common_inst) < top_k * 2:
            continue

        xgb_day = xgb_day.loc[common_inst]
        dis_day = dis_day.loc[common_inst]
        fwd_day = fwd_day.loc[common_inst]

        # XGB Top100
        top100_codes = xgb_day.nlargest(top_k).index
        top100_dis = dis_day.loc[top100_codes]
        top100_fwd = fwd_day.loc[top100_codes]

        # Disagreement threshold within Top100
        threshold = top100_dis.quantile(filter_pct)

        # Split
        low_dis_mask = top100_dis <= threshold
        high_dis_mask = top100_dis > threshold

        low_codes = top100_codes[low_dis_mask]
        high_codes = top100_codes[high_dis_mask]

        if len(low_codes) < 5 or len(high_codes) < 3:
            continue

        # Bottom100 for spread calculation
        bottom100_codes = xgb_day.nsmallest(top_k).index
        bottom100_fwd_mean = fwd_day.loc[bottom100_codes].mean()

        # Returns
        low_ret = fwd_day.loc[low_codes].mean()
        high_ret = fwd_day.loc[high_codes].mean()
        top100_ret = top100_fwd.mean()

        records.append({
            "date": dt,
            "n_top100": len(top100_codes),
            "n_low_dis": len(low_codes),
            "n_high_dis": len(high_codes),
            "low_dis_ret": low_ret,
            "high_dis_ret": high_ret,
            "top100_ret": top100_ret,
            "bottom100_ret": bottom100_fwd_mean,
            "spread_top100": top100_ret - bottom100_fwd_mean,
            "spread_filtered": low_ret - bottom100_fwd_mean,
            "threshold": threshold,
        })

    if not records:
        logger.error("No valid dates found for meta-filter test")
        return None

    df = pd.DataFrame(records)
    return df


def print_results(df: pd.DataFrame):
    """Print meta-filter test results."""
    print("\n" + "=" * 70)
    print("Phase 4R — Meta-filter Prototype Results")
    print("=" * 70)

    n_days = len(df)
    print(f"\nTest period: {df['date'].min()} to {df['date'].max()}")
    print(f"Valid trading days: {n_days}")

    # Averages
    avg_low_ret = df["low_dis_ret"].mean()
    avg_high_ret = df["high_dis_ret"].mean()
    avg_top100_ret = df["top100_ret"].mean()
    avg_bottom100_ret = df["bottom100_ret"].mean()
    avg_spread_unfiltered = df["spread_top100"].mean()
    avg_spread_filtered = df["spread_filtered"].mean()
    avg_n_low = df["n_low_dis"].mean()
    avg_n_high = df["n_high_dis"].mean()

    print(f"\n--- Return Comparison (daily mean of 5d fwd return) ---")
    print(f"  Top100 (unfiltered):        {avg_top100_ret:+.5f}  (n=100)")
    print(f"  Top100 LOW disagreement:    {avg_low_ret:+.5f}  (n~{avg_n_low:.0f})")
    print(f"  Top100 HIGH disagreement:   {avg_high_ret:+.5f}  (n~{avg_n_high:.0f})")
    print(f"  Bottom100:                  {avg_bottom100_ret:+.5f}")

    print(f"\n--- Spread (Long - Short) ---")
    print(f"  Unfiltered Top100 spread:   {avg_spread_unfiltered:+.5f}")
    print(f"  Filtered (low dis) spread:  {avg_spread_filtered:+.5f}")
    improvement = avg_spread_filtered - avg_spread_unfiltered
    print(f"  Improvement:                {improvement:+.5f} "
          f"({'BETTER' if improvement > 0 else 'WORSE'})")

    # Win rate: days where filtered > unfiltered
    win_days = (df["spread_filtered"] > df["spread_top100"]).sum()
    win_rate = win_days / n_days
    print(f"\n--- Consistency ---")
    print(f"  Days filtered > unfiltered: {win_days}/{n_days} ({win_rate:.1%})")

    # T-test
    from scipy.stats import ttest_rel
    diff = df["spread_filtered"] - df["spread_top100"]
    t_stat, p_val = ttest_rel(df["spread_filtered"], df["spread_top100"])
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")

    # Monthly breakdown
    df_copy = df.copy()
    df_copy["month"] = pd.to_datetime(df_copy["date"]).dt.to_period("M")
    monthly = df_copy.groupby("month").agg({
        "spread_top100": "mean",
        "spread_filtered": "mean",
        "low_dis_ret": "mean",
        "high_dis_ret": "mean",
    })
    print(f"\n--- Monthly Breakdown ---")
    print(f"  {'Month':>10s}  {'Unfiltered':>12s}  {'Filtered':>12s}  "
          f"{'Low-Dis Ret':>12s}  {'High-Dis Ret':>12s}")
    for month, row in monthly.iterrows():
        print(f"  {str(month):>10s}  {row['spread_top100']:+12.5f}  "
              f"{row['spread_filtered']:+12.5f}  "
              f"{row['low_dis_ret']:+12.5f}  {row['high_dis_ret']:+12.5f}")

    # Verdict
    print("\n" + "-" * 70)
    if improvement > 0 and win_rate > 0.5 and p_val < 0.1:
        print("VERDICT: Meta-filter shows promise. Low-disagreement stocks outperform.")
        print("         CX hypothesis supported: ensemble recall + meta-filter head selection.")
    elif improvement > 0:
        print("VERDICT: Directionally positive but not statistically significant.")
        print("         Consider: more data, different filter thresholds, or additional features.")
    else:
        print("VERDICT: Meta-filter does NOT improve spread in this test.")
        print("         Disagreement alone may not be a sufficient filter signal.")
    print("-" * 70)


def main():
    # Step 1: Load OOF predictions
    oof = load_oof_predictions()
    if len(oof) < 2:
        print("FAILED: Need at least 2 model OOF predictions.")
        print(f"Found: {list(oof.keys())}")
        sys.exit(1)

    if "xgb" not in oof:
        print("FAILED: XGB predictions not found in phase4e_24split/")
        sys.exit(1)

    # Step 2: Compute disagreement
    disagreement = compute_disagreement(oof)

    # Step 3: Load forward returns
    fwd_returns = load_forward_returns()

    # Step 4-6: Run meta-filter test
    results_df = run_meta_filter_test(
        xgb_preds=oof["xgb"],
        disagreement=disagreement,
        forward_returns=fwd_returns,
        top_k=100,
        filter_pct=0.75,
    )

    if results_df is None:
        print("FAILED: No valid test results produced.")
        sys.exit(1)

    # Step 7: Print results
    print_results(results_df)


if __name__ == "__main__":
    main()
