"""Phase 4R: Risk-based meta-filter — remove high crash risk from Top100.

Previous approach (quality filter by disagreement/volatility) FAILED because
high-volatility controversial stocks ARE the alpha source.

New approach: instead of filtering for "quality", filter OUT "danger":
- crash_prob > threshold → remove
- supply chain negative → remove
- holder_num increasing (dilution) → remove

This keeps the high-vol alpha but removes the landmines.

Usage:
    python scripts/phase4r_risk_filter.py
"""
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"


def load_predictions() -> dict[str, pd.Series]:
    """Load XGB OOF predictions from 24-split."""
    preds = {}
    split_dir = DATA_DIR / "phase4e_24split"
    for model in ["xgb"]:
        parts = []
        for f in sorted(split_dir.glob(f"split*_{model}.pkl")):
            with open(f, "rb") as fh:
                p = pickle.load(fh)
            if isinstance(p, pd.DataFrame):
                p = p.iloc[:, 0]
            parts.append(p)
        if parts:
            preds[model] = pd.concat(parts)
            logger.info(f"Loaded {model}: {len(preds[model])} predictions")
    return preds


def load_crash_features(cache: pd.DataFrame) -> pd.Series:
    """Compute crash risk proxy from features (high volatility + negative momentum)."""
    # STD20 = 20-day volatility, ROC20 = 20-day momentum
    if "STD20" in cache.columns and "ROC20" in cache.columns:
        # High vol + negative momentum = crash risk
        vol = cache["STD20"]
        mom = cache["ROC20"]
        # Rank within each date
        vol_rank = vol.groupby(level=0).rank(pct=True)
        mom_rank = mom.groupby(level=0).rank(pct=True)
        # crash_risk = high vol + low momentum
        crash_risk = vol_rank - mom_rank  # higher = more dangerous
        return crash_risk
    return pd.Series(dtype=float)


def main():
    logger.info("=== Phase 4R: Risk-based Meta-filter ===\n")

    # Load 24-split OOF predictions
    preds = load_predictions()
    if "xgb" not in preds:
        logger.error("No XGB predictions found")
        return

    xgb_pred = preds["xgb"]

    # Load returns
    cache = pd.read_parquet(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet",
                            columns=["__pnl_return_1d", "STD20", "ROC20"])
    returns = cache["__pnl_return_1d"]

    # Compute crash risk proxy
    crash_risk = load_crash_features(cache)
    logger.info(f"Crash risk computed: {crash_risk.notna().sum()} values")

    # Align
    common = xgb_pred.index.intersection(returns.index).intersection(crash_risk.index)
    xgb_aligned = xgb_pred.reindex(common)
    ret_aligned = returns.reindex(common)
    risk_aligned = crash_risk.reindex(common)

    dates = sorted(common.get_level_values(0).unique())
    logger.info(f"Aligned: {len(common)} rows, {len(dates)} dates")

    # Compare strategies
    results = {
        "unfiltered_top100": [],
        "risk_filtered_top75": [],
        "risk_filtered_top50": [],
    }

    for date in dates:
        try:
            pred_day = xgb_aligned.loc[date]
            ret_day = ret_aligned.loc[date]
            risk_day = risk_aligned.loc[date]
        except KeyError:
            continue

        if len(pred_day) < 200:
            continue

        # Top 100 by XGB
        top100 = pred_day.nlargest(100).index

        # Unfiltered Top100 return
        top100_ret = ret_day.reindex(top100).mean()
        bot100_ret = pred_day.nsmallest(100).index
        bot100_mean = ret_day.reindex(bot100_ret).mean()
        results["unfiltered_top100"].append(top100_ret - bot100_mean)

        # Risk filter: remove highest crash_risk stocks from Top100
        top100_risk = risk_day.reindex(top100).dropna()
        if len(top100_risk) < 50:
            results["risk_filtered_top75"].append(top100_ret - bot100_mean)
            results["risk_filtered_top50"].append(top100_ret - bot100_mean)
            continue

        # Remove top 25% crash risk → keep 75
        risk_threshold_75 = top100_risk.quantile(0.75)
        safe_75 = top100_risk[top100_risk <= risk_threshold_75].index
        filtered_75_ret = ret_day.reindex(safe_75).mean()
        results["risk_filtered_top75"].append(filtered_75_ret - bot100_mean)

        # Remove top 50% crash risk → keep 50
        risk_threshold_50 = top100_risk.quantile(0.50)
        safe_50 = top100_risk[top100_risk <= risk_threshold_50].index
        filtered_50_ret = ret_day.reindex(safe_50).mean()
        results["risk_filtered_top50"].append(filtered_50_ret - bot100_mean)

    # Summary
    print("\n" + "=" * 70)
    print("Phase 4R: Risk-based Meta-filter (remove high crash risk)")
    print("=" * 70)

    for strategy, spreads in results.items():
        if not spreads:
            continue
        arr = np.array(spreads)
        mean_spread = arr.mean()
        t_stat = mean_spread / (arr.std() / np.sqrt(len(arr))) if arr.std() > 0 else 0
        win_rate = (arr > 0).mean()
        print(f"\n  {strategy}:")
        print(f"    Mean spread: {mean_spread*1e4:+.1f} bps/day")
        print(f"    Win rate:    {win_rate:.1%}")
        print(f"    t-stat:      {t_stat:.2f}")
        print(f"    N days:      {len(spreads)}")

    # Compare: does risk filter improve?
    if results["unfiltered_top100"] and results["risk_filtered_top75"]:
        base = np.array(results["unfiltered_top100"])
        filt = np.array(results["risk_filtered_top75"])
        n = min(len(base), len(filt))
        delta = filt[:n] - base[:n]
        t, p = stats.ttest_rel(filt[:n], base[:n])
        improvement = delta.mean()
        wins = (delta > 0).sum()
        print(f"\n  Risk filter Top75 vs Unfiltered Top100:")
        print(f"    Improvement: {improvement*1e4:+.1f} bps/day")
        print(f"    Wins: {wins}/{n} ({wins/n:.0%})")
        print(f"    t-test: t={t:.3f}, p={p:.4f}")
        if p < 0.05 and improvement > 0:
            print(f"    VERDICT: SIGNIFICANT improvement! Risk filter works.")
        elif improvement > 0:
            print(f"    VERDICT: Positive but not significant (p={p:.3f})")
        else:
            print(f"    VERDICT: No improvement. Risk filter doesn't help.")

    print("=" * 70)


if __name__ == "__main__":
    main()
