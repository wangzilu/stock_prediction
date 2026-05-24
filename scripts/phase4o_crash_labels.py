"""Phase 4O — Build crash / downside labels for risk model training.

Labels produced (per stock x date):
  - crash_1d:       future 1-day return < -5%                      (binary)
  - crash_5d:       future 5-day cumulative return < -10%          (binary)
  - max_dd_5d:      future 5-day maximum drawdown                  (continuous, <= 0)
  - underperform_5d: future 5-day return < industry_median - 8%    (binary)

Reads:
  data/storage/feature_cache_174_holder_regime_ma.parquet  (column: __pnl_return_1d)
  data/storage/industry_mapping.parquet

Writes:
  data/storage/crash_labels.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "storage"
FEATURE_CACHE = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
INDUSTRY_MAP = DATA_DIR / "industry_mapping.parquet"
OUTPUT_PATH = DATA_DIR / "crash_labels.parquet"


def load_returns() -> pd.Series:
    """Load daily returns from feature cache."""
    print("[1/5] Loading __pnl_return_1d from feature cache ...")
    df = pd.read_parquet(FEATURE_CACHE, columns=["__pnl_return_1d"])
    ret = df["__pnl_return_1d"].dropna()
    print(f"       {len(ret):,} rows, date range "
          f"{ret.index.get_level_values(0).min().date()} — "
          f"{ret.index.get_level_values(0).max().date()}")
    return ret


def load_industry_map() -> pd.Series:
    """Load industry mapping: qlib_code -> industry."""
    im = pd.read_parquet(INDUSTRY_MAP)
    return im.set_index("qlib_code")["industry"]


def build_crash_labels(ret: pd.Series, industry_map: pd.Series) -> pd.DataFrame:
    """Compute forward-looking crash labels."""

    # Unstack to (date x instrument) matrix for easy shifting
    print("[2/5] Unstacking returns to date x stock matrix ...")
    ret_wide = ret.unstack(level="instrument")  # index=datetime, columns=instrument
    n_dates, n_stocks = ret_wide.shape
    print(f"       Matrix: {n_dates} dates x {n_stocks} stocks")

    # --- crash_1d: next day return < -5% ---
    print("[3/5] Computing forward labels ...")
    # Shift returns so row t contains the return realised at t+1
    fwd_1d = ret_wide.shift(-1)
    crash_1d = (fwd_1d < -0.05).astype(np.int8)

    # --- crash_5d: cumulative 5-day forward return < -10% ---
    # cum_5d[t] = ret[t+1] + ret[t+2] + ... + ret[t+5]
    fwd_cum_5d = sum(ret_wide.shift(-i) for i in range(1, 6))
    crash_5d = (fwd_cum_5d < -0.10).astype(np.int8)

    # --- max_dd_5d: maximum drawdown over next 5 days ---
    # Cumulative return path: c[k] = sum(ret[t+1..t+k])
    # Drawdown at step k = c[k] (since starting from 0)
    # max_dd = min(c[1], c[2], ..., c[5])
    cum_returns = []
    running = pd.DataFrame(0.0, index=ret_wide.index, columns=ret_wide.columns)
    for i in range(1, 6):
        running = running + ret_wide.shift(-i)
        cum_returns.append(running.copy())
    # Stack and take row-wise min
    max_dd_5d = pd.concat(cum_returns, axis=0).groupby(level=0).min()
    # Reindex to match original index order
    max_dd_5d = max_dd_5d.reindex(ret_wide.index)
    # Clip to <= 0 (drawdown is non-positive by definition; if all positive, dd=0)
    max_dd_5d = max_dd_5d.clip(upper=0.0)

    # --- underperform_5d: 5-day return < industry_median - 8% ---
    print("[4/5] Computing industry-relative underperformance ...")
    # Map instrument codes to industry
    ind_series = ret_wide.columns.to_series().map(industry_map)

    # For each date, compute industry median of fwd_cum_5d
    # Then check if stock's return < median - 0.08
    # We do this column-by-column via groupby on industry
    ind_median = fwd_cum_5d.T.groupby(ind_series.values).transform("median").T
    underperform_5d = (fwd_cum_5d < (ind_median - 0.08)).astype(np.int8)

    # --- Stack back to multi-index ---
    print("[5/5] Stacking and saving ...")
    labels = pd.DataFrame({
        "crash_1d": crash_1d.stack(),
        "crash_5d": crash_5d.stack(),
        "max_dd_5d": max_dd_5d.stack().astype(np.float32),
        "underperform_5d": underperform_5d.stack(),
    })
    labels.index.names = ["datetime", "instrument"]

    # Drop rows where all labels are NaN (tail dates without forward data)
    labels = labels.dropna(how="all")

    return labels


def print_stats(labels: pd.DataFrame):
    """Print summary statistics."""
    print("\n=== Crash Label Statistics ===")
    print(f"Total rows: {len(labels):,}")
    print(f"Date range: {labels.index.get_level_values(0).min().date()} — "
          f"{labels.index.get_level_values(0).max().date()}")

    for col in ["crash_1d", "crash_5d", "max_dd_5d", "underperform_5d"]:
        vals = labels[col].dropna()
        if col == "max_dd_5d":
            print(f"\n  {col} (continuous):")
            print(f"    count   = {len(vals):,}")
            print(f"    mean    = {vals.mean():.4f}")
            print(f"    median  = {vals.median():.4f}")
            print(f"    p5      = {vals.quantile(0.05):.4f}")
            print(f"    p1      = {vals.quantile(0.01):.4f}")
        else:
            rate = vals.mean()
            n_pos = int(vals.sum())
            print(f"\n  {col} (binary):")
            print(f"    base rate = {rate:.4f}  ({n_pos:,} events / {len(vals):,} obs)")

    # Events per year
    print("\n  Crash events per year:")
    for col in ["crash_1d", "crash_5d", "underperform_5d"]:
        vals = labels[col].dropna()
        yearly = vals.groupby(vals.index.get_level_values(0).year).sum()
        print(f"    {col}:")
        for yr, cnt in yearly.items():
            print(f"      {yr}: {int(cnt):,}")


def main():
    ret = load_returns()
    industry_map = load_industry_map()
    labels = build_crash_labels(ret, industry_map)

    # Save
    labels.to_parquet(OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}  ({labels.shape[0]:,} rows)")

    print_stats(labels)


if __name__ == "__main__":
    main()
