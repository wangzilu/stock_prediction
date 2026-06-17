"""SUE / PEAD factor builder from ST forecast_vip + income historical.

Builds two complementary surprise factors per (ts_code, end_date):

    SUE_FCST  = (actual_NI − forecast_NI_mid) / σ(past 8q forecast surprises)
                Coverage limited by forecast availability (~38k forecasts vs 72k
                income statements). Academic: Bernard-Thomas 1989 SUE.

    SUE_YoY   = (actual_NI − same_quarter_last_year_NI) / σ(past 4q YoY surprises)
                Higher coverage (all income statements). Hou-Xue-Zhang q-factor's
                expected-growth analog.

PIT discipline: `asof_date` is the ann_date of the income statement (when actual
becomes public). For SUE_FCST, the forecast must be published before the actual
ann_date, which forecast_vip's ann_date already encodes.

Output: data/storage/sue_factor_history.parquet
Columns: ts_code, qlib_code, end_date, asof_date, sue_yoy, sue_fcst,
         fcst_surprise_pct, n_hist_yoy, n_hist_fcst

Usage:
    python scripts/build_sue_factor.py                 # full rebuild
    python scripts/build_sue_factor.py --min-history 4 # custom rolling window
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
FORECAST_PATH = DATA_DIR / "research" / "st_forecast_vip_historical.parquet"
INCOME_PATH = DATA_DIR / "research" / "st_income_historical.parquet"
OUTPUT_PATH = DATA_DIR / "sue_factor_history.parquet"

# n_income in ST income parquet is in raw yuan units; net_profit_min/max from
# forecast_vip is in WAN (万元). Convert forecast to yuan before subtracting.
FCST_UNIT_MULTIPLIER = 10_000.0


def _ts_to_qlib(ts: str) -> str | None:
    if not isinstance(ts, str) or "." not in ts:
        return None
    num, suf = ts.split(".", 1)
    return f"{suf}{num}"


def _shift_quarter(period: str, n: int) -> str:
    """Shift YYYYMMDD quarter-end by ±n quarters."""
    if not isinstance(period, str) or len(period) != 8:
        return ""
    y, m = int(period[:4]), int(period[4:6])
    idx = y * 4 + (m - 1) // 3 + n
    new_y, new_q = idx // 4, idx % 4
    new_m = new_q * 3 + 3
    new_d = 31 if new_m in (3, 12) else 30
    return f"{new_y}{new_m:02d}{new_d:02d}"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading income + forecast parquets...")
    income = pd.read_parquet(INCOME_PATH)
    fcst = pd.read_parquet(FORECAST_PATH)
    logger.info(f"  income: {len(income):,} rows")
    logger.info(f"  forecast: {len(fcst):,} rows")
    return income, fcst


def compute_sue_yoy(income: pd.DataFrame, min_history: int = 4) -> pd.DataFrame:
    """SUE_YoY = (actual − same-q-LY) / σ(past min_history YoY surprises)."""
    df = income[["ts_code", "end_date", "ann_date", "n_income"]].copy()
    df = df.dropna(subset=["ts_code", "end_date", "n_income"])
    df["end_date"] = df["end_date"].astype(str)
    # Self-join with same quarter last year
    df["prev_end_date"] = df["end_date"].apply(lambda p: _shift_quarter(p, -4))
    prev = df[["ts_code", "end_date", "n_income"]].rename(
        columns={"end_date": "prev_end_date", "n_income": "ni_prev"}
    )
    df = df.merge(prev, on=["ts_code", "prev_end_date"], how="left")

    df["surprise_yoy"] = df["n_income"] - df["ni_prev"]

    # Per-stock rolling std of past `min_history` surprises (PIT — only past).
    df = df.sort_values(["ts_code", "end_date"]).reset_index(drop=True)
    df["sigma_yoy"] = (
        df.groupby("ts_code")["surprise_yoy"]
          .shift(1)
          .groupby(df["ts_code"])
          .rolling(min_history, min_periods=min_history)
          .std()
          .reset_index(level=0, drop=True)
    )
    df["n_hist_yoy"] = (
        df.groupby("ts_code")["surprise_yoy"]
          .shift(1)
          .groupby(df["ts_code"])
          .rolling(min_history, min_periods=1)
          .count()
          .reset_index(level=0, drop=True)
    )
    df["sue_yoy"] = (df["surprise_yoy"] / df["sigma_yoy"]).replace(
        [np.inf, -np.inf], np.nan
    )
    return df[["ts_code", "end_date", "ann_date", "sue_yoy", "n_hist_yoy"]]


def compute_sue_fcst(income: pd.DataFrame, fcst: pd.DataFrame,
                     min_history: int = 4) -> pd.DataFrame:
    """SUE_FCST = (actual − forecast_mid) / σ(past forecast surprises)."""
    # Per (ts_code, end_date), take the LATEST forecast issued BEFORE income
    # ann_date. forecast_vip can have multiple updates per quarter.
    f = fcst[["ts_code", "end_date", "ann_date",
              "net_profit_min", "net_profit_max"]].copy()
    f["end_date"] = f["end_date"].astype(str)
    f["fcst_ann_date"] = f["ann_date"].astype(str)
    f["forecast_mid"] = (
        pd.to_numeric(f["net_profit_min"], errors="coerce") +
        pd.to_numeric(f["net_profit_max"], errors="coerce")
    ) / 2.0 * FCST_UNIT_MULTIPLIER  # WAN → yuan

    # Sort + take last per (ts_code, end_date)
    f = f.sort_values(["ts_code", "end_date", "fcst_ann_date"])
    f_last = (
        f.dropna(subset=["forecast_mid"])
         .groupby(["ts_code", "end_date"], as_index=False)
         .tail(1)
    )

    i = income[["ts_code", "end_date", "ann_date", "n_income"]].copy()
    i["end_date"] = i["end_date"].astype(str)
    i["ann_date"] = i["ann_date"].astype(str)

    j = i.merge(
        f_last[["ts_code", "end_date", "forecast_mid", "fcst_ann_date"]],
        on=["ts_code", "end_date"], how="left",
    )
    # PIT — only count forecasts published BEFORE income ann_date
    pit_ok = j["fcst_ann_date"].notna() & (j["fcst_ann_date"] <= j["ann_date"])
    j["surprise_fcst"] = np.where(
        pit_ok, j["n_income"] - j["forecast_mid"], np.nan
    )
    # Per-share-ish growth surprise = (actual - fcst) / max(|fcst|, ε)
    j["fcst_surprise_pct"] = np.where(
        pit_ok,
        (j["n_income"] - j["forecast_mid"]) / j["forecast_mid"].abs().clip(lower=1e-6),
        np.nan,
    )

    j = j.sort_values(["ts_code", "end_date"]).reset_index(drop=True)
    j["sigma_fcst"] = (
        j.groupby("ts_code")["surprise_fcst"]
         .shift(1)
         .groupby(j["ts_code"])
         .rolling(min_history, min_periods=min_history)
         .std()
         .reset_index(level=0, drop=True)
    )
    j["n_hist_fcst"] = (
        j.groupby("ts_code")["surprise_fcst"]
         .shift(1)
         .groupby(j["ts_code"])
         .rolling(min_history * 2, min_periods=1)
         .count()
         .reset_index(level=0, drop=True)
    )
    j["sue_fcst"] = (j["surprise_fcst"] / j["sigma_fcst"]).replace(
        [np.inf, -np.inf], np.nan
    )
    return j[["ts_code", "end_date", "ann_date", "sue_fcst",
              "fcst_surprise_pct", "n_hist_fcst"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-history", type=int, default=4,
                        help="Min past quarters of surprise history (default 4)")
    args = parser.parse_args()

    income, fcst = load_inputs()

    yoy = compute_sue_yoy(income, args.min_history)
    logger.info(f"YoY SUE: {yoy['sue_yoy'].notna().sum():,} non-null rows")

    fct = compute_sue_fcst(income, fcst, args.min_history)
    logger.info(f"FCST SUE: {fct['sue_fcst'].notna().sum():,} non-null rows")

    # Outer merge to retain all (ts_code, end_date) where either exists
    out = yoy.merge(
        fct.drop(columns=["ann_date"]),
        on=["ts_code", "end_date"], how="outer",
    )
    out["qlib_code"] = out["ts_code"].map(_ts_to_qlib)
    out = out.rename(columns={"ann_date": "asof_date"})

    # Coverage stats
    cover_either = ((out["sue_yoy"].notna()) | (out["sue_fcst"].notna())).sum()
    logger.info(f"Total (ts_code, end_date) rows: {len(out):,}, with any SUE: {cover_either:,}")
    logger.info(f"  by stock: {out['ts_code'].nunique()}, by end_date: {out['end_date'].nunique()}")
    logger.info(
        f"  SUE_YoY mean: {out['sue_yoy'].mean():.3f}  std: {out['sue_yoy'].std():.3f}  "
        f"P95: {out['sue_yoy'].quantile(0.95):.3f}"
    )
    logger.info(
        f"  SUE_FCST mean: {out['sue_fcst'].mean():.3f}  std: {out['sue_fcst'].std():.3f}  "
        f"P95: {out['sue_fcst'].quantile(0.95):.3f}"
    )

    cols = ["ts_code", "qlib_code", "end_date", "asof_date",
            "sue_yoy", "sue_fcst", "fcst_surprise_pct",
            "n_hist_yoy", "n_hist_fcst"]
    out = out[[c for c in cols if c in out.columns]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved {OUTPUT_PATH} ({len(out):,} rows, {len(out.columns)} cols)")

    # Health
    try:
        from scheduler.data_health import HealthStatus, write_health
        latest_period = out["end_date"].max() if "end_date" in out.columns else ""
        latest_human = ""
        if isinstance(latest_period, str) and len(latest_period) == 8:
            latest_human = f"{latest_period[:4]}-{latest_period[4:6]}-{latest_period[6:]}"
        write_health("sue_factor_build", HealthStatus(
            success=True, n_items=len(out), latest_date=latest_human,
            network_profile="none",
        ))
    except Exception as e:
        logger.warning(f"write_health failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
