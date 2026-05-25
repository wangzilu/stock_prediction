#!/usr/bin/env python3
"""Phase 4L: ext_holder_decrease full 24-split rolling gate validation.

Validates whether the holder_decrease factor (negative pct change in
shareholder count) provides marginal predictive power across all 24 rolling
test windows, beyond what the champion feature set already captures.

Gate criteria:
  - Mean residual RankIC > 0.005
  - Positive RankIC ratio > 50% across splits

Usage:
    python scripts/phase4l_holder_24split.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from config.rolling_splits import get_standard_splits
from tracker.artifact_contract import ExperimentArtifact

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
STORAGE = PROJECT_ROOT / "data" / "storage"
CACHE_PATH = STORAGE / "feature_cache_174_holder_regime_ma.parquet"
HOLDER_PATH = STORAGE / "st_holder_number.parquet"
RETURNS_COL = "__pnl_return_1d"


# ---------------------------------------------------------------------------
# 1. Load holder_decrease factor
# ---------------------------------------------------------------------------

def load_holder_decrease() -> pd.Series:
    """Load st_holder_number.parquet and compute ext_holder_decrease.

    Steps:
      - Parse ann_date, apply +1 BDay PIT lag
      - Compute pct_change of holder_num per stock
      - Negate (decrease = positive signal)
      - Return as (datetime, instrument) MultiIndex Series
    """
    logger.info(f"Loading holder number data from {HOLDER_PATH}")
    df = pd.read_parquet(HOLDER_PATH)
    logger.info(f"  Raw shape: {df.shape}")

    df["holder_num"] = pd.to_numeric(df.get("holder_num"), errors="coerce")
    df["ann_date"] = pd.to_datetime(df["ann_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["ann_date", "holder_num"])
    df["qlib_code"] = df["qlib_code"].str.lower()

    # Sort and deduplicate
    df = df.sort_values(["qlib_code", "ann_date"]).drop_duplicates(
        ["qlib_code", "ann_date"], keep="last"
    )

    # Compute pct change per stock
    df["holder_pct_chg"] = df.groupby("qlib_code")["holder_num"].pct_change()
    # Invert: decreasing holders = concentrated ownership = bullish
    df["ext_holder_decrease"] = -df["holder_pct_chg"]

    # PIT safety: ann_date + 1 BDay (same as feature_merger)
    df["pit_date"] = df["ann_date"] + pd.tseries.offsets.BDay(1)

    # Build MultiIndex series — deduplicate (keep last per date+stock)
    df = df.drop_duplicates(["pit_date", "qlib_code"], keep="last")
    mi = df.set_index(["pit_date", "qlib_code"])
    mi.index.names = ["datetime", "instrument"]
    factor = mi["ext_holder_decrease"].dropna().copy()
    factor.name = "ext_holder_decrease"

    logger.info(f"  Factor: {factor.notna().sum():,} non-null observations")
    logger.info(f"  Date range: {factor.index.get_level_values(0).min().date()} ~ "
                f"{factor.index.get_level_values(0).max().date()}")
    logger.info(f"  Stocks: {factor.index.get_level_values(1).nunique()}")

    return factor


# ---------------------------------------------------------------------------
# 2. Load feature cache (returns + champion proxy)
# ---------------------------------------------------------------------------

def load_returns_and_champion():
    """Load forward returns and build champion proxy from feature cache."""
    logger.info(f"Loading feature cache from {CACHE_PATH}")
    df = pd.read_parquet(CACHE_PATH)
    logger.info(f"  Cache shape: {df.shape}")
    logger.info(f"  Date range: {df.index.get_level_values(0).min().date()} ~ "
                f"{df.index.get_level_values(0).max().date()}")

    fwd_returns = df[RETURNS_COL].copy()
    fwd_returns.name = "fwd_ret"

    # Champion proxy: rank-average of core alpha158 features (same as phase4l_ext)
    exclude_prefixes = ("__", "_", "hsi_", "hstech_", "nasdaq_")
    exclude_names = {
        "holder_num", "bp", "ep", "pb", "pe",
        "pb_mom20", "pe_mom20", "price_pos20",
        "flow_net_mf_latest", "flow_net_mf_5d",
        "flow_net_mf_20d_avg", "amount_anom20",
    }
    alpha_cols = [
        c for c in df.columns
        if not any(c.startswith(p) for p in exclude_prefixes)
        and c not in exclude_names
    ]
    logger.info(f"  Champion proxy: {len(alpha_cols)} features")
    rank_df = df[alpha_cols].rank(pct=True, axis=1)
    champion_pred = rank_df.mean(axis=1)
    champion_pred.name = "champion_pred"

    return fwd_returns, champion_pred


# ---------------------------------------------------------------------------
# 3. Forward-fill factor onto daily trading dates
# ---------------------------------------------------------------------------

def forward_fill_factor(
    factor: pd.Series,
    returns: pd.Series,
) -> pd.Series:
    """Forward-fill the sparse quarterly factor onto the daily trading grid.

    holder_number is published quarterly via ann_date.  Between announcements
    the factor value stays constant (high autocorrelation by design).
    We asof-merge the factor onto every (date, stock) in the returns index.
    """
    logger.info("Forward-filling factor onto daily trading grid...")

    # Get unique trading dates and instruments from returns
    dates_level = returns.index.get_level_values(0)
    inst_level = returns.index.get_level_values(1)

    # Unstack factor to (date x instrument), then reindex to daily dates and ffill
    factor_unstacked = factor.unstack("instrument")

    # Get all trading dates
    all_dates = sorted(dates_level.unique())
    # Combine factor dates and trading dates, sort, and ffill
    combined_idx = factor_unstacked.index.union(pd.DatetimeIndex(all_dates))
    combined_idx = combined_idx.sort_values()

    factor_daily = factor_unstacked.reindex(combined_idx).ffill()
    # Now slice to only trading dates
    factor_daily = factor_daily.reindex(pd.DatetimeIndex(all_dates))

    # Re-stack
    try:
        factor_filled = factor_daily.stack(future_stack=True)
    except TypeError:
        factor_filled = factor_daily.stack(dropna=False)
    factor_filled.index.names = ["datetime", "instrument"]
    factor_filled.name = "ext_holder_decrease"

    n_filled = factor_filled.notna().sum()
    n_total = len(returns)
    logger.info(f"  Filled: {n_filled:,} non-null / {n_total:,} total grid points "
                f"({n_filled / n_total:.1%} coverage)")

    return factor_filled


# ---------------------------------------------------------------------------
# 4. Per-split RankIC computation
# ---------------------------------------------------------------------------

def compute_daily_rank_ic(factor: pd.Series, returns: pd.Series) -> pd.Series:
    """Compute daily cross-sectional Spearman RankIC.

    Returns a Series indexed by date with RankIC for each day.
    """
    # Align
    common = factor.dropna().index.intersection(returns.dropna().index)
    f = factor.loc[common]
    r = returns.loc[common]

    dates = f.index.get_level_values(0)
    unique_dates = sorted(dates.unique())

    ics = {}
    for dt in unique_dates:
        f_day = f.xs(dt, level=0, drop_level=True)
        r_day = r.xs(dt, level=0, drop_level=True)
        # Align instruments
        common_inst = f_day.dropna().index.intersection(r_day.dropna().index)
        if len(common_inst) < 10:
            continue
        corr, _ = spearmanr(f_day.loc[common_inst], r_day.loc[common_inst])
        if np.isfinite(corr):
            ics[dt] = corr

    return pd.Series(ics)


def compute_residual_rank_ic(
    factor: pd.Series,
    champion: pd.Series,
    returns: pd.Series,
) -> pd.Series:
    """Compute daily residual RankIC: IC of factor residualized against champion.

    For each date:
      1. Regress factor on champion (cross-sectional OLS)
      2. Take residuals
      3. Compute Spearman correlation of residuals vs returns
    """
    common = (
        factor.dropna().index
        .intersection(champion.dropna().index)
        .intersection(returns.dropna().index)
    )
    f = factor.loc[common]
    c = champion.loc[common]
    r = returns.loc[common]

    dates = f.index.get_level_values(0)
    unique_dates = sorted(dates.unique())

    ics = {}
    for dt in unique_dates:
        try:
            f_day = f.xs(dt, level=0, drop_level=True)
            c_day = c.xs(dt, level=0, drop_level=True)
            r_day = r.xs(dt, level=0, drop_level=True)
        except KeyError:
            continue

        common_inst = (
            f_day.dropna().index
            .intersection(c_day.dropna().index)
            .intersection(r_day.dropna().index)
        )
        if len(common_inst) < 10:
            continue

        fv = f_day.loc[common_inst].values
        cv = c_day.loc[common_inst].values
        rv = r_day.loc[common_inst].values

        # OLS residual: factor ~ champion
        X = np.column_stack([np.ones(len(cv)), cv])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, fv, rcond=None)
            resid = fv - X @ beta
        except np.linalg.LinAlgError:
            continue

        corr, _ = spearmanr(resid, rv)
        if np.isfinite(corr):
            ics[dt] = corr

    return pd.Series(ics)


# ---------------------------------------------------------------------------
# 5. Main: 24-split rolling gate
# ---------------------------------------------------------------------------

def main():
    # Load data
    factor_raw = load_holder_decrease()
    fwd_returns, champion_pred = load_returns_and_champion()

    # Forward-fill factor to daily grid
    factor_daily = forward_fill_factor(factor_raw, fwd_returns)

    # Get 24-split config
    splits = get_standard_splits(preset="24split")
    logger.info(f"\n{'='*80}")
    logger.info(f"Running ext_holder_decrease through {len(splits)}-split rolling gate")
    logger.info(f"{'='*80}\n")

    # Per-split results
    split_results = []

    for sp in splits:
        sid = sp["split_id"]
        test_start = pd.Timestamp(sp["test_start"])
        test_end = pd.Timestamp(sp["test_end"])

        # Slice to test period
        dates = fwd_returns.index.get_level_values(0)
        mask = (dates >= test_start) & (dates <= test_end)
        if mask.sum() == 0:
            logger.warning(f"  Split {sid}: no data in test window "
                           f"[{test_start.date()} ~ {test_end.date()}]")
            split_results.append({
                "split_id": sid, "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "rank_ic": np.nan, "residual_ic": np.nan,
                "n_days": 0, "coverage": 0.0,
            })
            continue

        ret_slice = fwd_returns.loc[mask]
        factor_slice = factor_daily.reindex(ret_slice.index)
        champ_slice = champion_pred.reindex(ret_slice.index)

        # Coverage
        coverage = factor_slice.notna().sum() / len(ret_slice) if len(ret_slice) > 0 else 0.0

        # RankIC
        daily_ic = compute_daily_rank_ic(factor_slice, ret_slice)
        mean_ic = daily_ic.mean() if len(daily_ic) > 0 else np.nan

        # Residual RankIC
        daily_resid_ic = compute_residual_rank_ic(factor_slice, champ_slice, ret_slice)
        mean_resid_ic = daily_resid_ic.mean() if len(daily_resid_ic) > 0 else np.nan

        n_days = len(daily_ic)

        split_results.append({
            "split_id": sid,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "rank_ic": mean_ic,
            "residual_ic": mean_resid_ic,
            "n_days": n_days,
            "coverage": coverage,
        })

        logger.info(f"  Split {sid:2d}  [{test_start.date()} ~ {test_end.date()}]  "
                     f"RankIC={mean_ic:+.4f}  ResidualIC={mean_resid_ic:+.4f}  "
                     f"days={n_days}  cov={coverage:.1%}")

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(split_results)
    valid = results_df.dropna(subset=["rank_ic"])

    if len(valid) == 0:
        logger.error("No valid splits. Aborting.")
        return

    mean_rank_ic = valid["rank_ic"].mean()
    std_rank_ic = valid["rank_ic"].std()
    icir = mean_rank_ic / std_rank_ic if std_rank_ic > 0 else 0.0
    pos_ratio = (valid["rank_ic"] > 0).mean()
    n_valid = len(valid)

    mean_resid_ic = valid["residual_ic"].mean()
    std_resid_ic = valid["residual_ic"].std()
    resid_icir = mean_resid_ic / std_resid_ic if std_resid_ic > 0 else 0.0
    resid_pos_ratio = (valid["residual_ic"] > 0).mean()

    mean_coverage = valid["coverage"].mean()
    total_days = valid["n_days"].sum()

    # ------------------------------------------------------------------
    # Gate check
    # ------------------------------------------------------------------
    gate_residual_ic = mean_resid_ic > 0.005
    gate_pos_ratio = pos_ratio > 0.50
    gate_pass = gate_residual_ic and gate_pos_ratio

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("ext_holder_decrease — 24-Split Rolling Gate Validation")
    print("=" * 100)

    print("\n--- Per-Split Results ---")
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(results_df.to_string(index=False))

    print(f"\n--- Aggregate Metrics ({n_valid} valid splits) ---")
    print(f"  Mean RankIC:           {mean_rank_ic:+.4f}")
    print(f"  RankIC Std:            {std_rank_ic:.4f}")
    print(f"  ICIR:                  {icir:+.3f}")
    print(f"  Positive Ratio:        {pos_ratio:.1%} ({int(pos_ratio * n_valid)}/{n_valid})")
    print(f"  Mean Residual RankIC:  {mean_resid_ic:+.4f}")
    print(f"  Residual ICIR:         {resid_icir:+.3f}")
    print(f"  Residual Pos Ratio:    {resid_pos_ratio:.1%}")
    print(f"  Mean Coverage:         {mean_coverage:.1%}")
    print(f"  Total IC Days:         {total_days}")

    print(f"\n--- Gate Check ---")
    print(f"  [{'PASS' if gate_residual_ic else 'FAIL'}] Residual RankIC > 0.005: "
          f"{mean_resid_ic:+.4f}")
    print(f"  [{'PASS' if gate_pos_ratio else 'FAIL'}] Positive Ratio > 50%: "
          f"{pos_ratio:.1%}")
    verdict = "PASS" if gate_pass else "FAIL"
    print(f"\n  >>> VERDICT: {verdict} <<<")

    if gate_pass:
        print("\n  CONCLUSION: ext_holder_decrease PASSES the 24-split gate.")
        print("  Recommendation: ADD to champion feature cache.")
    else:
        failures = []
        if not gate_residual_ic:
            failures.append(f"residual_ic={mean_resid_ic:+.4f} <= 0.005")
        if not gate_pos_ratio:
            failures.append(f"pos_ratio={pos_ratio:.1%} <= 50%")
        print(f"\n  CONCLUSION: ext_holder_decrease FAILS the 24-split gate.")
        print(f"  Failures: {'; '.join(failures)}")
        print("  Recommendation: DO NOT add to champion feature cache.")

    # ------------------------------------------------------------------
    # Save as ExperimentArtifact
    # ------------------------------------------------------------------
    exp_id = f"ext_holder_decrease_24split_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    art = ExperimentArtifact.create(
        experiment_id=exp_id,
        model_name="ext_holder_decrease",
        feature_set="single_factor",
        description="24-split rolling gate validation for ext_holder_decrease factor",
        factor_name="ext_holder_decrease",
        n_splits=len(splits),
        gate_verdict=verdict,
    )
    art.save_metrics({
        "rank_ic_mean": mean_rank_ic,
        "rank_ic_std": std_rank_ic,
        "rank_icir": icir,
        "rank_ic_pos_ratio": pos_ratio,
        "residual_rank_ic_mean": mean_resid_ic,
        "residual_rank_ic_std": std_resid_ic,
        "residual_icir": resid_icir,
        "residual_pos_ratio": resid_pos_ratio,
        "coverage": mean_coverage,
        "n_valid_splits": n_valid,
        "total_ic_days": int(total_days),
        "gate_residual_ic_pass": gate_residual_ic,
        "gate_pos_ratio_pass": gate_pos_ratio,
        "gate_verdict": verdict,
        "per_split": split_results,
    })

    validation = art.validate()
    logger.info(f"\nArtifact saved: {art.artifact_dir}")
    logger.info(f"  Complete: {validation['complete']}")

    print(f"\n  Artifact: {exp_id}")
    print(f"  Location: {art.artifact_dir}")


if __name__ == "__main__":
    main()
