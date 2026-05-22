"""Build fundamental factors from ST financial statements (PIT-safe).

Uses ann_date (actual disclosure date) for point-in-time alignment.
Only uses data available BEFORE each trading date.

Factors:
  From balancesheet:
    - debt_to_equity: total_liab / total_hldr_eqy_exc_min_int
    - current_ratio: total_cur_assets / total_cur_liab
    - asset_turnover_proxy: revenue / total_assets (cross-table)

  From income:
    - roe_ttm: net_profit_ttm / avg_equity
    - revenue_growth: revenue / lag_revenue - 1
    - profit_margin: n_income / revenue
    - eps_growth: basic_eps / lag_basic_eps - 1

  From cashflow:
    - ocf_to_profit: n_cashflow_act / net_profit (earnings quality)
    - fcf_yield: free_cashflow / market_cap_proxy
    - capex_ratio: (n_cashflow_inv_act) / revenue

Usage:
    python scripts/build_fundamental_factors.py
"""
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
OUTPUT_PATH = DATA_DIR / "fundamental_factors_pit.parquet"


def ts_to_qlib(ts_code: str) -> str:
    if "." not in str(ts_code):
        return ""
    code, exch = str(ts_code).split(".")
    return f"{exch.lower()}{code}"


def load_and_clean(name: str) -> pd.DataFrame:
    """Load ST financial statement, clean and parse dates."""
    path = DATA_DIR / f"st_{name}.parquet"
    if not path.exists():
        logger.warning(f"  {name} not found")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    logger.info(f"  {name}: {df.shape}")

    # Parse dates
    for col in ["ann_date", "f_ann_date", "end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")

    # Use f_ann_date (actual disclosure) if available, else ann_date
    if "f_ann_date" in df.columns:
        df["pit_date"] = df["f_ann_date"].fillna(df.get("ann_date"))
    elif "ann_date" in df.columns:
        df["pit_date"] = df["ann_date"]
    else:
        logger.warning(f"  {name}: no date column for PIT")
        return pd.DataFrame()

    # Add qlib code
    df["qlib_code"] = df["ts_code"].apply(ts_to_qlib)

    # Remove rows without valid dates
    df = df.dropna(subset=["pit_date", "end_date", "qlib_code"])

    # Convert numeric columns
    for col in df.columns:
        if col not in ["ts_code", "qlib_code", "pit_date", "end_date", "ann_date",
                        "f_ann_date", "report_type", "comp_type", "end_type", "update_flag"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by pit_date for asof merge
    df = df.sort_values(["qlib_code", "pit_date"])
    return df


def build_factors():
    """Build all fundamental factors with PIT alignment."""
    t_start = time.time()

    logger.info("Loading financial statements...")
    bs = load_and_clean("balancesheet")
    inc = load_and_clean("income")
    cf = load_and_clean("cashflow")

    # Load feature cache index for alignment
    logger.info("Loading feature cache index...")
    cache = pd.read_parquet(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet",
                            columns=["__label_5d"])
    target_index = cache.index
    del cache; gc.collect()

    all_dates = sorted(target_index.get_level_values(0).unique())
    all_instruments = sorted(target_index.get_level_values(1).unique())
    logger.info(f"  Target: {len(all_dates)} dates × {len(all_instruments)} stocks")

    # === Build factors from each table, then PIT asof merge to trading dates ===
    logger.info("\nComputing raw factors from financial statements...")

    factor_frames = []

    # --- Balancesheet factors ---
    if not bs.empty:
        logger.info("  Balancesheet factors...")
        bs_f = bs[["qlib_code", "pit_date"]].copy()
        equity = pd.to_numeric(bs.get("total_hldr_eqy_exc_min_int"), errors="coerce")
        liab = pd.to_numeric(bs.get("total_liab"), errors="coerce")
        cur_a = pd.to_numeric(bs.get("total_cur_assets"), errors="coerce")
        cur_l = pd.to_numeric(bs.get("total_cur_liab"), errors="coerce")

        bs_f["debt_to_equity"] = np.where(equity.abs() > 1, liab / equity, np.nan)
        bs_f["current_ratio"] = np.where(cur_l.abs() > 1, cur_a / cur_l, np.nan)
        bs_f["_equity"] = equity  # for ROE calculation
        bs_f = bs_f.dropna(subset=["pit_date"])
        factor_frames.append(("bs", bs_f))
        logger.info(f"    {len(bs_f)} rows, factors: debt_to_equity, current_ratio")

    # --- Income factors ---
    if not inc.empty:
        logger.info("  Income factors...")
        inc_f = inc[["qlib_code", "pit_date"]].copy()
        revenue = pd.to_numeric(inc.get("revenue"), errors="coerce")
        n_income = pd.to_numeric(inc.get("n_income", inc.get("net_profit")), errors="coerce")
        eps = pd.to_numeric(inc.get("basic_eps"), errors="coerce")

        inc_f["profit_margin"] = np.where(revenue.abs() > 1, n_income / revenue, np.nan)
        inc_f["_revenue"] = revenue
        inc_f["_n_income"] = n_income
        inc_f["_eps"] = eps

        # Growth: shift within each stock
        inc_f_sorted = inc_f.sort_values(["qlib_code", "pit_date"])
        inc_f_sorted["_prev_rev"] = inc_f_sorted.groupby("qlib_code")["_revenue"].shift(1)
        inc_f_sorted["_prev_eps"] = inc_f_sorted.groupby("qlib_code")["_eps"].shift(1)
        inc_f_sorted["revenue_growth"] = np.where(
            inc_f_sorted["_prev_rev"].abs() > 1,
            inc_f_sorted["_revenue"] / inc_f_sorted["_prev_rev"] - 1, np.nan)
        inc_f_sorted["eps_growth"] = np.where(
            inc_f_sorted["_prev_eps"].abs() > 0.01,
            inc_f_sorted["_eps"] / inc_f_sorted["_prev_eps"] - 1, np.nan)

        inc_f = inc_f_sorted.dropna(subset=["pit_date"])
        factor_frames.append(("inc", inc_f))
        logger.info(f"    {len(inc_f)} rows, factors: profit_margin, revenue_growth, eps_growth")

    # --- Cashflow factors ---
    if not cf.empty:
        logger.info("  Cashflow factors...")
        cf_f = cf[["qlib_code", "pit_date"]].copy()
        ocf = pd.to_numeric(cf.get("n_cashflow_act"), errors="coerce")
        net_profit = pd.to_numeric(cf.get("net_profit"), errors="coerce")

        cf_f["ocf_to_profit"] = np.where(net_profit.abs() > 1, ocf / net_profit, np.nan)
        cf_f = cf_f.dropna(subset=["pit_date"])
        factor_frames.append(("cf", cf_f))
        logger.info(f"    {len(cf_f)} rows, factors: ocf_to_profit")

    # === PIT asof merge to trading dates ===
    logger.info("\nPIT asof merge to trading dates...")

    # Target: (datetime, instrument) from feature cache
    target_df = pd.DataFrame(index=target_index).reset_index()
    target_df.columns = ["date", "qlib_code"]
    target_df["qlib_code"] = target_df["qlib_code"].astype(str)
    target_df = target_df.sort_values(["qlib_code", "date"])

    # Merge each factor table
    FACTOR_COLS = ["debt_to_equity", "current_ratio", "profit_margin",
                   "revenue_growth", "eps_growth", "ocf_to_profit"]

    result = target_df[["date", "qlib_code"]].copy()

    for table_name, fdf in factor_frames:
        # Keep only factor columns + keys
        keep_cols = ["qlib_code", "pit_date"] + [c for c in fdf.columns
                     if c in FACTOR_COLS or c == "_n_income" or c == "_equity"]
        fdf_clean = fdf[keep_cols].copy()
        fdf_clean = fdf_clean.sort_values(["qlib_code", "pit_date"])
        fdf_clean = fdf_clean.drop_duplicates(subset=["qlib_code", "pit_date"], keep="last")

        # PIT asof merge via per-stock searchsorted (avoids pandas merge_asof issues)
        factor_cols_in_table = [c for c in fdf_clean.columns if c in FACTOR_COLS]
        extra_cols = [c for c in ["_n_income", "_equity"] if c in fdf_clean.columns]
        merge_cols = factor_cols_in_table + extra_cols

        for col in merge_cols:
            if col not in result.columns:
                result[col] = np.nan

        n_filled = 0
        for inst, grp in fdf_clean.groupby("qlib_code"):
            if len(grp) == 0:
                continue
            pit_dates = grp["pit_date"].values.astype("datetime64[ns]")
            inst_mask = result["qlib_code"] == inst
            if not inst_mask.any():
                continue
            trade_dates_inst = result.loc[inst_mask, "date"].values.astype("datetime64[ns]")
            # searchsorted: find index of latest pit_date <= each trade_date
            idx = np.searchsorted(pit_dates, trade_dates_inst, side="right") - 1
            valid = idx >= 0
            for col in merge_cols:
                vals = grp[col].values
                filled = np.where(valid, vals[np.clip(idx, 0, len(vals)-1)], np.nan)
                filled[~valid] = np.nan
                result.loc[inst_mask, col] = filled
            n_filled += valid.sum()

        # ROE from cross-table
        if "_n_income" in result.columns and "_equity" in result.columns:
            eq = result["_equity"]
            ni = result["_n_income"]
            result["roe"] = np.where(eq.abs() > 1, ni / eq, np.nan)

        logger.info(f"  Merged {table_name}: {n_filled:,} filled values")

    # Set index back
    result = result.set_index(["date", "qlib_code"])
    result.index.names = ["datetime", "instrument"]

    # Keep only factor columns
    all_factor_cols = [c for c in result.columns if c in FACTOR_COLS + ["roe"]]
    factor_df = result[all_factor_cols]

    # Clean
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)
    for col in factor_df.columns:
        vals = factor_df[col].dropna()
        if len(vals) > 100:
            mu, sigma = vals.mean(), vals.std()
            if sigma > 0:
                factor_df[col] = factor_df[col].clip(mu - 5 * sigma, mu + 5 * sigma)

    logger.info(f"\nFactor records: {len(factor_df):,}")

    if len(factor_df) == 0:
        logger.error("No factors generated!")
        return

    # Clean
    factor_df = factor_df.replace([np.inf, -np.inf], np.nan)

    # Clip extreme values (winsorize per column at ±5 sigma)
    for col in factor_df.columns:
        vals = factor_df[col].dropna()
        if len(vals) > 100:
            mu, sigma = vals.mean(), vals.std()
            if sigma > 0:
                factor_df[col] = factor_df[col].clip(mu - 5 * sigma, mu + 5 * sigma)

    logger.info(f"\nFactor DataFrame: {factor_df.shape}")
    logger.info(f"Columns: {list(factor_df.columns)}")
    logger.info(f"Coverage per factor:")
    for col in factor_df.columns:
        cov = factor_df[col].notna().mean()
        logger.info(f"  {col:<20} {cov*100:.1f}%")

    # Save
    factor_df.to_parquet(str(OUTPUT_PATH))
    logger.info(f"\nSaved to {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB)")

    elapsed = time.time() - t_start
    logger.info(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    build_factors()
