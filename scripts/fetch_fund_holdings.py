"""Fetch fund/ETF portfolio holdings + macro regime data.

1. Top ETF/基金持仓 → 机构重仓度/抱团度
2. 宏观数据 (PMI/CPI/M2/Shibor) → regime controller

Usage:
    python scripts/fetch_fund_holdings.py --macro          # 宏观数据（快）
    python scripts/fetch_fund_holdings.py --fund-holdings   # 基金持仓（慢）
    python scripts/fetch_fund_holdings.py --all
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"

# Major A-share ETFs to track
TOP_ETFS = [
    "510300.SH",  # 沪深300ETF
    "510500.SH",  # 中证500ETF
    "512100.SH",  # 中证1000ETF
    "159915.SZ",  # 创业板ETF
    "159919.SZ",  # 沪深300ETF
    "510050.SH",  # 上证50ETF
    "512010.SH",  # 医药ETF
    "512880.SH",  # 证券ETF
    "515790.SH",  # 光伏ETF
    "562000.SH",  # 科创板ETF
    "159852.SZ",  # 芯片ETF
    "159790.SZ",  # 碳中和ETF
]

# Top active funds (by AUM)
TOP_FUNDS = [
    "110011.OF",  # 易方达中小盘
    "161725.OF",  # 招商中证白酒
    "005827.OF",  # 易方达蓝筹精选
    "163406.OF",  # 兴全合润
    "260108.OF",  # 景顺长城新兴成长
    "519772.OF",  # 交银定期支付双息平衡
    "001838.OF",  # 国泰互联网+
    "000961.OF",  # 天弘沪深300
]


def get_st_client():
    token = Path(PROJECT_ROOT / ".st_token").read_text().strip()
    from ST_CLIENT import StockToday
    return StockToday(token=token)


def fetch_macro(st):
    """Fetch macro regime data: PMI, CPI, M2, Shibor."""
    logger.info("=== Fetching macro data ===")

    datasets = {
        "cn_pmi": {"method": "cn_pmi", "kwargs": {"start_date": "20210101", "end_date": "20261231"}},
        "cn_cpi": {"method": "cn_cpi", "kwargs": {"start_date": "20210101", "end_date": "20261231"}},
        "cn_m": {"method": "cn_m", "kwargs": {"start_date": "20210101", "end_date": "20261231"}},
        "shibor": {"method": "shibor", "kwargs": {"start_date": "20210101", "end_date": "20261231"}},
    }

    for name, cfg in datasets.items():
        out_path = DATA_DIR / f"st_{name}.parquet"
        if out_path.exists():
            logger.info(f"  {name}: already exists, skip")
            continue

        logger.info(f"  Fetching {name}...")
        try:
            method = getattr(st, cfg["method"])
            result = method(**cfg["kwargs"])

            if isinstance(result, dict):
                code = result.get("code")
                msg = result.get("msg", "")[:50]
                data = result.get("data")
                if code == 1:
                    logger.warning(f"  {name}: 需要升级 — {msg}")
                    continue
                if data and isinstance(data, list):
                    df = pd.DataFrame(data)
                    for col in df.columns:
                        if df[col].dtype == object:
                            df[col] = df[col].astype(str)
                    df.to_parquet(str(out_path), index=False)
                    logger.info(f"  ✅ {name}: {len(df)} records")
                else:
                    logger.warning(f"  {name}: empty (code={code})")
            elif isinstance(result, list) and result:
                df = pd.DataFrame(result)
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str)
                df.to_parquet(str(out_path), index=False)
                logger.info(f"  ✅ {name}: {len(df)} records")
            else:
                logger.warning(f"  {name}: unexpected result type")
        except Exception as e:
            logger.error(f"  {name}: ERROR {e}")

        time.sleep(0.5)


def fetch_fund_holdings(st):
    """Fetch portfolio holdings for top ETFs and funds."""
    logger.info("=== Fetching fund holdings ===")

    all_holdings = []
    fund_codes = TOP_ETFS + TOP_FUNDS

    for fund_code in fund_codes:
        logger.info(f"  {fund_code}...")
        try:
            result = st.fund_portfolio(ts_code=fund_code)
            if isinstance(result, dict):
                data = result.get("data")
                if data and isinstance(data, list):
                    for item in data:
                        item["fund_code"] = fund_code
                    all_holdings.extend(data)
                    logger.info(f"    {len(data)} holdings")
                else:
                    logger.info(f"    empty (code={result.get('code')}, msg={result.get('msg','')[:30]})")
            elif isinstance(result, list) and result:
                for item in result:
                    item["fund_code"] = fund_code
                all_holdings.extend(result)
                logger.info(f"    {len(result)} holdings")
            else:
                logger.info(f"    no data")
        except Exception as e:
            logger.error(f"    ERROR: {e}")

        time.sleep(0.3)

    if all_holdings:
        df = pd.DataFrame(all_holdings)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str)
        out_path = DATA_DIR / "st_fund_portfolio.parquet"
        df.to_parquet(str(out_path), index=False)
        logger.info(f"  ✅ Fund holdings: {len(df)} records → {out_path}")
    else:
        logger.warning(f"  No holdings data")

    # Also fetch ETF share changes
    logger.info("\n=== Fetching ETF share changes ===")
    all_shares = []
    for etf_code in TOP_ETFS:
        logger.info(f"  {etf_code}...")
        try:
            result = st.fund_share(ts_code=etf_code, start_date="20240101", end_date="20261231")
            if isinstance(result, dict):
                data = result.get("data")
                if data and isinstance(data, list):
                    all_shares.extend(data)
                    logger.info(f"    {len(data)} records")
                else:
                    logger.info(f"    empty")
            elif isinstance(result, list) and result:
                all_shares.extend(result)
                logger.info(f"    {len(result)} records")
        except Exception as e:
            logger.error(f"    ERROR: {e}")
        time.sleep(0.3)

    if all_shares:
        df = pd.DataFrame(all_shares)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str)
        out_path = DATA_DIR / "st_etf_share_changes.parquet"
        df.to_parquet(str(out_path), index=False)
        logger.info(f"  ✅ ETF shares: {len(df)} records → {out_path}")


def fetch_regime_data(st):
    """Fetch additional regime controller inputs."""
    logger.info("=== Fetching regime data ===")

    datasets = {
        "cn_cpi": {"method": "cn_cpi", "kwargs": {"start_m": "202101", "end_m": "202612"}},
        "cn_ppi": {"method": "cn_ppi", "kwargs": {"start_m": "202101", "end_m": "202612"}},
        "fx_daily": {"method": "fx_daily", "kwargs": {"ts_code": "USDCNH.FXCM", "start_date": "20210101", "end_date": "20261231"}},
        "us_tycr": {"method": "us_tycr", "kwargs": {}},
    }

    for name, cfg in datasets.items():
        out_path = DATA_DIR / f"st_{name}.parquet"
        if out_path.exists():
            logger.info(f"  {name}: already exists, skip")
            continue

        logger.info(f"  Fetching {name}...")
        try:
            method = getattr(st, cfg["method"])
            result = method(**cfg["kwargs"])
            if isinstance(result, dict):
                code = result.get("code")
                msg = result.get("msg", "")[:50]
                data = result.get("data")
                if code == 1:
                    logger.warning(f"  {name}: 需要升级 — {msg}")
                    continue
                if data and isinstance(data, list):
                    df = pd.DataFrame(data)
                    for col in df.columns:
                        if df[col].dtype == object:
                            df[col] = df[col].astype(str)
                    df.to_parquet(str(out_path), index=False)
                    logger.info(f"  ✅ {name}: {len(df)} records")
                else:
                    logger.warning(f"  {name}: empty")
            elif isinstance(result, list) and result:
                df = pd.DataFrame(result)
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str)
                df.to_parquet(str(out_path), index=False)
                logger.info(f"  ✅ {name}: {len(df)} records")
        except Exception as e:
            logger.error(f"  {name}: ERROR {e}")
        time.sleep(0.5)

    # IC/IM futures — try current month contracts
    from datetime import datetime
    ym = datetime.now().strftime("%y%m")  # e.g. "2606"
    for fut_prefix, fut_name in [("IF", "IF沪深300期货"), ("IC", "IC中证500期货"), ("IM", "IM中证1000期货")]:
        fut_code = f"{fut_prefix}{ym}.CFX"
        out_name = f"fut_{fut_code.split('.')[0].lower()}"
        out_path = DATA_DIR / f"st_{out_name}.parquet"
        if out_path.exists():
            logger.info(f"  {out_name}: already exists, skip")
            continue
        logger.info(f"  Fetching {fut_name}...")
        try:
            result = st.fut_daily(ts_code=fut_code, start_date="20240101", end_date="20261231")
            if isinstance(result, dict):
                data = result.get("data")
                if data and isinstance(data, list):
                    df = pd.DataFrame(data)
                    for col in df.columns:
                        if df[col].dtype == object:
                            df[col] = df[col].astype(str)
                    df.to_parquet(str(out_path), index=False)
                    logger.info(f"  ✅ {out_name}: {len(df)} records")
                else:
                    logger.warning(f"  {out_name}: empty (code={result.get('code')}, msg={result.get('msg','')[:30]})")
        except Exception as e:
            logger.error(f"  {out_name}: ERROR {e}")
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="Fetch fund holdings + macro + regime data")
    parser.add_argument("--macro", action="store_true", help="Fetch macro data only")
    parser.add_argument("--fund-holdings", action="store_true", help="Fetch fund holdings only")
    parser.add_argument("--regime", action="store_true", help="Fetch regime controller data")
    parser.add_argument("--all", action="store_true", help="Fetch everything")
    args = parser.parse_args()

    if not (args.macro or args.fund_holdings or args.regime or args.all):
        parser.print_help()
        return

    st = get_st_client()

    if args.macro or args.all:
        fetch_macro(st)

    if args.regime or args.all:
        fetch_regime_data(st)

    if args.fund_holdings or args.all:
        fetch_fund_holdings(st)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
