"""Fetch the 3 missing regime data sources via AKShare.

1. IC/IM futures (期货基差 — 量化拥挤信号)
2. Fund portfolio holdings (基金持仓集中度)
3. USD/CNY exchange rate

Usage:
    python scripts/fetch_missing_regime.py
"""
import logging
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"


def fetch_ic_im_futures():
    """Fetch IC/IM/IF main contract daily data from Sina."""
    import akshare as ak

    for symbol, name in [("IC0", "IC中证500"), ("IM0", "IM中证1000"), ("IF0", "IF沪深300")]:
        out_path = DATA_DIR / f"ak_futures_{symbol.lower()}.parquet"
        if out_path.exists():
            logger.info(f"  {name}: already exists, skip")
            continue

        logger.info(f"  Fetching {name} ({symbol})...")
        try:
            df = ak.futures_main_sina(symbol=symbol)
            if df is not None and not df.empty:
                df.to_parquet(str(out_path), index=False)
                logger.info(f"    ✅ {len(df)} rows → {out_path.name}")
            else:
                logger.warning(f"    empty")
        except Exception as e:
            logger.error(f"    ❌ {e}")


def fetch_fund_holdings():
    """Fetch top ETF/fund portfolio holdings from Eastmoney."""
    import akshare as ak

    out_path = DATA_DIR / "ak_fund_holdings.parquet"
    if out_path.exists():
        logger.info(f"  Fund holdings: already exists, skip")
        return

    # Top ETFs + active funds
    funds = [
        ("510300", "沪深300ETF"),
        ("510500", "中证500ETF"),
        ("159915", "创业板ETF"),
        ("510050", "上证50ETF"),
        ("512880", "证券ETF"),
    ]

    all_holdings = []
    for code, name in funds:
        logger.info(f"  Fetching {name} ({code})...")
        try:
            df = ak.fund_portfolio_hold_em(symbol=code, date="2025")
            if df is not None and not df.empty:
                df["fund_code"] = code
                df["fund_name"] = name
                all_holdings.append(df)
                logger.info(f"    ✅ {len(df)} holdings")
        except Exception as e:
            logger.error(f"    ❌ {e}")

    if all_holdings:
        result = pd.concat(all_holdings, ignore_index=True)
        for col in result.columns:
            if result[col].dtype == object:
                result[col] = result[col].astype(str)
        result.to_parquet(str(out_path), index=False)
        logger.info(f"  ✅ Total: {len(result)} holdings → {out_path.name}")


def fetch_usd_cny():
    """Fetch USD/CNY exchange rate history."""
    import akshare as ak

    out_path = DATA_DIR / "ak_usdcny.parquet"
    if out_path.exists():
        logger.info(f"  USD/CNY: already exists, skip")
        return

    logger.info(f"  Fetching USD/CNY...")
    try:
        # Try currency_boc_sina (中国银行汇率)
        df = ak.currency_boc_sina(symbol="美元", start_date="20210101", end_date="20261231")
        if df is not None and not df.empty:
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str)
            df.to_parquet(str(out_path), index=False)
            logger.info(f"    ✅ {len(df)} rows → {out_path.name}")
            return
    except Exception as e:
        logger.warning(f"    boc_sina: {e}")

    # Fallback: fx_pair_quote
    try:
        df = ak.fx_pair_quote(symbol="美元/人民币")
        if df is not None and not df.empty:
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str)
            df.to_parquet(str(out_path), index=False)
            logger.info(f"    ✅ {len(df)} rows → {out_path.name}")
    except Exception as e:
        logger.error(f"    ❌ {e}")


def main():
    logger.info("=== Fetching missing regime data ===\n")

    logger.info("1. IC/IM/IF Futures")
    fetch_ic_im_futures()

    logger.info("\n2. Fund Holdings")
    fetch_fund_holdings()

    logger.info("\n3. USD/CNY Exchange Rate")
    fetch_usd_cny()

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
