"""Build moneyflow v2 microstructure factors (CX spec).

Raw moneyflow (18 cols) failed ablation at 50%. First-order derivatives also
failed (38%). The issue: raw amounts correlate with market cap. This script
normalizes by ADV and computes microstructure signals.

Factors:
  1. main_flow_adv         — net main flow / 20d ADV (removes size effect)
  2. order_imbalance        — large order buy-sell imbalance, bounded [-1,1]
  3. large_small_divergence — cross-sectional rank divergence big vs small
  4. flow_zscore_60d        — anomaly detection: is today's flow unusual?
  5. flow_persistence_10d   — fraction of last 10 days with positive net flow
  6. flow_industry_rank     — net flow rank within industry peers per date

PIT safety: dates shifted +1 BDay (flow data published after close).
"""
import os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
OUT_PATH = DATA_DIR / "moneyflow_v2.parquet"


def main():
    # ---- Load moneyflow ----
    logger.info("Loading st_moneyflow.parquet...")
    mf = pd.read_parquet(str(DATA_DIR / "st_moneyflow.parquet"))
    mf["date"] = pd.to_datetime(mf["date"], errors="coerce")
    mf["qlib_code"] = mf["qlib_code"].astype(str).str.upper()
    mf = mf.sort_values(["qlib_code", "date"]).reset_index(drop=True)
    logger.info(f"  shape: {mf.shape}, stocks: {mf['qlib_code'].nunique()}")

    # Ensure numeric
    amount_cols = [c for c in mf.columns if c.startswith("st_") and "amount" in c]
    for c in amount_cols:
        mf[c] = pd.to_numeric(mf[c], errors="coerce")

    # ---- Precompute helper columns ----
    buy_lg = mf["st_buy_lg_amount"].values.astype(np.float64)
    buy_elg = mf["st_buy_elg_amount"].values.astype(np.float64)
    sell_lg = mf["st_sell_lg_amount"].values.astype(np.float64)
    sell_elg = mf["st_sell_elg_amount"].values.astype(np.float64)
    buy_sm = mf["st_buy_sm_amount"].values.astype(np.float64)
    net_mf = pd.to_numeric(mf["st_net_mf_amount"], errors="coerce").values.astype(np.float64)

    # Total daily amount per stock (all order sizes, buy+sell)
    total_amount = (
        mf["st_buy_sm_amount"].astype(np.float64)
        + mf["st_sell_sm_amount"].astype(np.float64)
        + mf["st_buy_md_amount"].astype(np.float64)
        + mf["st_sell_md_amount"].astype(np.float64)
        + mf["st_buy_lg_amount"].astype(np.float64)
        + mf["st_sell_lg_amount"].astype(np.float64)
        + mf["st_buy_elg_amount"].astype(np.float64)
        + mf["st_sell_elg_amount"].astype(np.float64)
    )
    mf["_total_amount"] = total_amount
    mf["_net_main"] = buy_lg + buy_elg - sell_lg - sell_elg
    mf["_net_mf"] = net_mf

    # ---- Factor 1: main_flow_adv ----
    logger.info("Computing main_flow_adv...")
    adv_20d = mf.groupby("qlib_code")["_total_amount"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    mf["main_flow_adv"] = mf["_net_main"] / (adv_20d + 1e-8)

    # ---- Factor 2: order_imbalance ----
    logger.info("Computing order_imbalance...")
    mf["order_imbalance"] = (buy_lg - sell_lg) / (buy_lg + sell_lg + 1e-8)

    # ---- Factor 3: large_small_divergence (cross-sectional rank per date) ----
    logger.info("Computing large_small_divergence...")
    mf["_big_buy"] = buy_elg + buy_lg
    mf["_sm_buy"] = buy_sm
    # Per-date cross-sectional rank
    mf["_big_rank"] = mf.groupby("date")["_big_buy"].rank(pct=True)
    mf["_sm_rank"] = mf.groupby("date")["_sm_buy"].rank(pct=True)
    mf["large_small_divergence"] = mf["_big_rank"] - mf["_sm_rank"]

    # ---- Factor 4: flow_zscore_60d ----
    logger.info("Computing flow_zscore_60d...")
    g_net = mf.groupby("qlib_code")["_net_mf"]
    roll_60_mean = g_net.transform(lambda x: x.rolling(60, min_periods=20).mean())
    roll_60_std = g_net.transform(lambda x: x.rolling(60, min_periods=20).std())
    mf["flow_zscore_60d"] = (mf["_net_mf"] - roll_60_mean) / (roll_60_std + 1e-8)

    # ---- Factor 5: flow_persistence_10d ----
    logger.info("Computing flow_persistence_10d...")
    mf["_net_pos"] = (mf["_net_mf"] > 0).astype(np.float64)
    mf["flow_persistence_10d"] = mf.groupby("qlib_code")["_net_pos"].transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    )

    # ---- Factor 6: flow_industry_rank ----
    logger.info("Computing flow_industry_rank...")
    ind = pd.read_parquet(str(DATA_DIR / "industry_mapping.parquet"))
    ind["qlib_code"] = ind["qlib_code"].astype(str).str.upper()
    ind_map = ind.drop_duplicates("qlib_code").set_index("qlib_code")["industry"]
    mf["_industry"] = mf["qlib_code"].map(ind_map)
    ind_coverage = mf["_industry"].notna().mean()
    logger.info(f"  industry coverage: {ind_coverage:.1%}")

    # Per-date, per-industry rank of net_mf_amount
    mf["flow_industry_rank"] = mf.groupby(["date", "_industry"])["_net_mf"].rank(pct=True)

    # ---- PIT safety: shift +1 BDay ----
    logger.info("Applying PIT shift (+1 BDay)...")
    mf["date"] = mf["date"] + pd.tseries.offsets.BDay(1)

    # ---- Output ----
    factor_cols = [
        "main_flow_adv",
        "order_imbalance",
        "large_small_divergence",
        "flow_zscore_60d",
        "flow_persistence_10d",
        "flow_industry_rank",
    ]
    out = mf[["qlib_code", "date"] + factor_cols].copy()

    # Replace inf with NaN
    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Clip extreme values
    for c in factor_cols:
        q_lo, q_hi = out[c].quantile([0.001, 0.999])
        out[c] = out[c].clip(q_lo, q_hi)

    logger.info(f"Output shape: {out.shape}")
    logger.info(f"Date range: {out['date'].min()} to {out['date'].max()}")
    logger.info(f"Stocks: {out['qlib_code'].nunique()}")
    logger.info("Non-null rates:")
    for c in factor_cols:
        logger.info(f"  {c}: {out[c].notna().mean():.1%}")

    out.to_parquet(str(OUT_PATH), index=False)
    logger.info(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
