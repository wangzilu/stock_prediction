"""Build derivative features from moneyflow and cyq_perf data.

Raw moneyflow/cyq values failed ablation (50% delta>0). This script computes
per-stock time-series derivatives: change rates, volatility, ratios.
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
OUT_PATH = DATA_DIR / "derived_moneyflow_cyq.parquet"


def build_moneyflow_derivatives():
    logger.info("Loading moneyflow...")
    mf = pd.read_parquet(str(DATA_DIR / "st_moneyflow.parquet"))
    mf = mf.sort_values(["qlib_code", "date"])

    # net_flow: 主力净流入 (large + extra-large buy minus sell)
    mf["net_flow"] = (
        mf["st_buy_lg_vol"] + mf["st_buy_elg_vol"]
        - mf["st_sell_lg_vol"] - mf["st_sell_elg_vol"]
    ).astype(np.float64)

    # big_order_ratio: fraction of volume from large + extra-large orders
    total_buy = (
        mf["st_buy_sm_vol"] + mf["st_buy_md_vol"]
        + mf["st_buy_lg_vol"] + mf["st_buy_elg_vol"]
    ).astype(np.float64)
    mf["big_order_ratio"] = (
        (mf["st_buy_lg_vol"] + mf["st_buy_elg_vol"]).astype(np.float64) / total_buy
    )

    # Per-stock rolling features
    g = mf.groupby("qlib_code")["net_flow"]
    mf["net_flow_5d_change"] = g.transform(lambda x: x.pct_change(5))
    mf["net_flow_20d_change"] = g.transform(lambda x: x.pct_change(20))

    roll20_std = g.transform(lambda x: x.rolling(20, min_periods=10).std())
    roll20_mean = g.transform(lambda x: x.rolling(20, min_periods=10).mean())
    mf["net_flow_vol_20d"] = roll20_std / roll20_mean.replace(0, np.nan)

    out_cols = [
        "qlib_code", "date",
        "net_flow", "net_flow_5d_change", "net_flow_20d_change",
        "net_flow_vol_20d", "big_order_ratio",
    ]
    result = mf[out_cols].copy()
    logger.info(f"  moneyflow derivatives: {result.shape}, non-null rates:")
    for c in out_cols[2:]:
        logger.info(f"    {c}: {result[c].notna().mean():.1%}")
    return result


def build_cyq_derivatives():
    logger.info("Loading cyq_perf...")
    cyq = pd.read_parquet(str(DATA_DIR / "st_cyq_perf.parquet"))
    cyq = cyq.sort_values(["qlib_code", "date"])

    # cost_concentration
    cyq["cost_concentration"] = (
        (cyq["cyq_cost_85pct"] - cyq["cyq_cost_15pct"])
        / cyq["cyq_cost_50pct"].replace(0, np.nan)
    )

    # Per-stock rolling
    gw = cyq.groupby("qlib_code")["cyq_winner_rate"]
    cyq["winner_rate_change_5d"] = gw.transform(lambda x: x.pct_change(5))
    cyq["winner_rate_change_20d"] = gw.transform(lambda x: x.pct_change(20))

    gc = cyq.groupby("qlib_code")["cost_concentration"]
    cyq["cost_concentration_change_5d"] = gc.transform(lambda x: x.pct_change(5))

    out_cols = [
        "qlib_code", "date",
        "winner_rate_change_5d", "winner_rate_change_20d",
        "cost_concentration", "cost_concentration_change_5d",
    ]
    result = cyq[out_cols].copy()
    logger.info(f"  cyq derivatives: {result.shape}, non-null rates:")
    for c in out_cols[2:]:
        logger.info(f"    {c}: {result[c].notna().mean():.1%}")
    return result


def main():
    mf_deriv = build_moneyflow_derivatives()
    cyq_deriv = build_cyq_derivatives()

    # Merge on (qlib_code, date)
    merged = pd.merge(mf_deriv, cyq_deriv, on=["qlib_code", "date"], how="outer")
    merged = merged.sort_values(["qlib_code", "date"]).reset_index(drop=True)

    # Replace inf with NaN
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)

    logger.info(f"Final derived factors: {merged.shape}")
    logger.info(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    logger.info(f"Stocks: {merged['qlib_code'].nunique()}")

    merged.to_parquet(str(OUT_PATH), index=False)
    logger.info(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
