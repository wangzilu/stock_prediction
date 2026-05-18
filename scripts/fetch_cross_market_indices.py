"""Fetch historical daily data for cross-market regime signals.

Targets: 恒生指数, 恒生科技, 纳斯达克
These are leading indicators for A-share: HK reacts faster (no limit, T+0),
US tech themes propagate to A-share with 1-3 day delay.

Saves to data/storage/cross_market_indices.parquet

Usage:
    python scripts/fetch_cross_market_indices.py
    python scripts/fetch_cross_market_indices.py --start 20200101
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"

# AKShare index codes
INDICES = {
    "hsi": {"name": "恒生指数", "ak_func": "stock_hk_index_daily_em", "symbol": "HSI"},
    "hstech": {"name": "恒生科技", "ak_func": "stock_hk_index_daily_em", "symbol": "HSTECH"},
    "nasdaq": {"name": "纳斯达克", "ak_func": "index_us_stock_sina", "symbol": "IXIC"},
}


def fetch_hk_index(symbol: str, start: str) -> pd.DataFrame:
    """Fetch HK index daily data via AKShare."""
    import akshare as ak
    try:
        df = ak.stock_hk_index_daily_em(symbol=symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["日期"])
        df = df.rename(columns={"开盘": "open", "最高": "high", "最低": "low",
                                 "收盘": "close", "成交量": "volume"})
        df = df[df["date"] >= pd.to_datetime(start)]
        return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")
    except Exception as e:
        logger.warning(f"HK index {symbol} fetch failed: {e}")
        return pd.DataFrame()


def fetch_us_index(symbol: str, start: str) -> pd.DataFrame:
    """Fetch US index daily data via AKShare."""
    import akshare as ak
    try:
        df = ak.index_us_stock_sina(symbol=f".{symbol}")
        if df is None or df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce")
        df = df[df["date"] >= pd.to_datetime(start)]
        return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date")
    except Exception as e:
        logger.warning(f"US index {symbol} fetch failed: {e}")
        return pd.DataFrame()


def compute_regime_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute regime features from daily OHLCV.

    Features:
    - 1d/5d/20d return
    - 5d/20d volatility
    - 5d/20d momentum (close vs MA)
    - RSI-like: up days ratio in last 10 days
    """
    if df.empty or "close" not in df.columns:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    out["date"] = df["date"]
    c = df["close"]

    # Returns
    out[f"{prefix}_ret1d"] = c.pct_change(1)
    out[f"{prefix}_ret5d"] = c.pct_change(5)
    out[f"{prefix}_ret20d"] = c.pct_change(20)

    # Volatility
    out[f"{prefix}_vol5d"] = c.pct_change().rolling(5).std()
    out[f"{prefix}_vol20d"] = c.pct_change().rolling(20).std()

    # Momentum: close vs MA
    out[f"{prefix}_mom5d"] = c / c.rolling(5).mean() - 1
    out[f"{prefix}_mom20d"] = c / c.rolling(20).mean() - 1

    # Up days ratio (RSI proxy)
    up = (c.pct_change() > 0).astype(float)
    out[f"{prefix}_up_ratio_10d"] = up.rolling(10).mean()

    # Drawdown from 20-day high
    rolling_high = c.rolling(20).max()
    out[f"{prefix}_dd20d"] = (c - rolling_high) / rolling_high

    return out.dropna(subset=[f"{prefix}_ret5d"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20200101")
    args = parser.parse_args()

    all_features = []

    # Fetch HK indices
    for key in ["hsi", "hstech"]:
        info = INDICES[key]
        logger.info(f"Fetching {info['name']} ({info['symbol']})...")
        df = fetch_hk_index(info["symbol"], args.start)
        if not df.empty:
            logger.info(f"  Got {len(df)} rows, {df['date'].min()} ~ {df['date'].max()}")
            features = compute_regime_features(df, key)
            all_features.append(features)
        else:
            logger.warning(f"  Empty!")

    # Fetch US index
    logger.info(f"Fetching 纳斯达克 (IXIC)...")
    df_us = fetch_us_index("IXIC", args.start)
    if not df_us.empty:
        logger.info(f"  Got {len(df_us)} rows, {df_us['date'].min()} ~ {df_us['date'].max()}")
        features = compute_regime_features(df_us, "nasdaq")
        all_features.append(features)
    else:
        logger.warning(f"  Empty!")

    if not all_features:
        logger.error("No data fetched!")
        return

    # Merge all on date
    merged = all_features[0]
    for f in all_features[1:]:
        merged = merged.merge(f, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)

    # Save
    out_path = DATA_DIR / "cross_market_indices.parquet"
    merged.to_parquet(str(out_path), index=False)
    logger.info(f"Saved: {out_path}")
    logger.info(f"  Shape: {merged.shape}")
    logger.info(f"  Date range: {merged['date'].min()} ~ {merged['date'].max()}")
    logger.info(f"  Columns: {list(merged.columns)}")

    # Summary stats
    for col in merged.columns:
        if col != "date":
            logger.info(f"  {col}: mean={merged[col].mean():.4f}, "
                        f"non-null={merged[col].notna().sum()}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
