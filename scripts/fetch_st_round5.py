"""Fetch ST_CLIENT Round 5: high-priority remaining data.

1. stk_holdertrade — 高管/股东增减持 (事件型)
2. moneyflow_ind_dc — 行业板块资金流 (日频)
3. block_trade — 大宗交易 (事件型)
4. broker_recommend — 券商评级 (事件型)

Usage:
    python scripts/fetch_st_round5.py --api all
    python scripts/fetch_st_round5.py --api holdertrade
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
TOKEN_FILE = PROJECT_ROOT / ".st_token"

def get_token():
    token = os.environ.get("ST_TOKEN", "")
    if not token and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
    if not token:
        raise RuntimeError("ST_TOKEN not set")
    return token


def get_trade_dates(start: str, end: str) -> list[str]:
    dates = pd.bdate_range(start=start, end=end)
    return [d.strftime("%Y%m%d") for d in dates]


def fetch_holdertrade(start: str, end: str):
    """高管/股东增减持 — query by ann_date."""
    from ST_CLIENT import StockToday
    st = StockToday(token=get_token())
    output_path = DATA_DIR / "st_holdertrade.parquet"

    existing = None
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        logger.info(f"Resuming holdertrade ({len(existing)} rows)")

    dates = get_trade_dates(start, end)
    # Sample every 3 days (events are sparse)
    sample_dates = dates[::3] if len(dates) > 50 else dates
    logger.info(f"Fetching holdertrade: {len(sample_dates)} dates")

    all_rows = []
    for i, d in enumerate(sample_dates):
        try:
            result = st.stk_holdertrade(ann_date=d)
            if isinstance(result, list) and result:
                all_rows.extend(result)
            elif isinstance(result, dict) and result.get("msg"):
                if i == 0:
                    logger.warning(f"  API msg: {result['msg']}")
                    if "TOKEN" in result["msg"] or "套餐" in result["msg"]:
                        logger.error("  Token/permission issue, stopping")
                        return
            if (i + 1) % 50 == 0:
                logger.info(f"  {i+1}/{len(sample_dates)}: {len(all_rows)} rows")
            time.sleep(0.15)
        except Exception as e:
            logger.warning(f"  {d} failed: {e}")
            time.sleep(1)

    if not all_rows:
        logger.info("No new holdertrade data")
        return

    df = pd.DataFrame(all_rows)
    if "ts_code" in df.columns:
        df["qlib_code"] = df["ts_code"].apply(
            lambda x: ('sz' + x[:6]) if x.endswith('.SZ') else ('sh' + x[:6]))

    if existing is not None:
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["qlib_code", "ann_date", "holder_name"] if "holder_name" in df.columns
                                else ["qlib_code", "ann_date"], keep="last")

    df.to_parquet(str(output_path), index=False)
    logger.info(f"Saved holdertrade: {df.shape}")


def fetch_industry_moneyflow(start: str, end: str):
    """行业板块资金流 — daily industry-level fund flow."""
    from ST_CLIENT import StockToday
    st = StockToday(token=get_token())
    output_path = DATA_DIR / "st_moneyflow_ind.parquet"

    existing = None
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        last_date = existing["trade_date"].max() if "trade_date" in existing.columns else None
        if last_date:
            start = (pd.to_datetime(str(last_date)) + timedelta(days=1)).strftime("%Y%m%d")
            logger.info(f"Resuming from {start} ({len(existing)} rows)")

    dates = get_trade_dates(start, end)
    logger.info(f"Fetching industry moneyflow: {len(dates)} dates")

    all_rows = []
    for i, d in enumerate(dates):
        try:
            result = st.moneyflow_ind_dc(trade_date=d)
            if isinstance(result, list) and result:
                for r in result:
                    r["trade_date"] = d
                all_rows.extend(result)
            elif isinstance(result, dict) and result.get("msg"):
                if i == 0:
                    logger.warning(f"  API msg: {result['msg']}")
                    if "TOKEN" in result["msg"] or "套餐" in result["msg"]:
                        logger.error("  Token/permission issue, stopping")
                        return
            if (i + 1) % 50 == 0:
                logger.info(f"  {i+1}/{len(dates)}: {len(all_rows)} rows")
            time.sleep(0.12)
        except Exception as e:
            logger.warning(f"  {d} failed: {e}")
            time.sleep(1)

    if not all_rows:
        logger.info("No new industry moneyflow data")
        return

    df = pd.DataFrame(all_rows)
    if existing is not None:
        df = pd.concat([existing, df], ignore_index=True)
        if "trade_date" in df.columns and "ts_code" in df.columns:
            df = df.drop_duplicates(["trade_date", "ts_code"], keep="last")

    df.to_parquet(str(output_path), index=False)
    logger.info(f"Saved industry moneyflow: {df.shape}")


def fetch_block_trade(start: str, end: str):
    """大宗交易 — query by trade_date."""
    from ST_CLIENT import StockToday
    st = StockToday(token=get_token())
    output_path = DATA_DIR / "st_block_trade.parquet"

    existing = None
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        logger.info(f"Resuming block_trade ({len(existing)} rows)")

    dates = get_trade_dates(start, end)
    logger.info(f"Fetching block_trade: {len(dates)} dates")

    all_rows = []
    for i, d in enumerate(dates):
        try:
            result = st.block_trade(trade_date=d)
            if isinstance(result, list) and result:
                all_rows.extend(result)
            elif isinstance(result, dict) and result.get("msg"):
                if i == 0:
                    logger.warning(f"  API msg: {result['msg']}")
                    if "TOKEN" in result["msg"] or "套餐" in result["msg"]:
                        return
            if (i + 1) % 50 == 0:
                logger.info(f"  {i+1}/{len(dates)}: {len(all_rows)} rows")
            time.sleep(0.12)
        except Exception as e:
            logger.warning(f"  {d} failed: {e}")

    if not all_rows:
        logger.info("No new block_trade data")
        return

    df = pd.DataFrame(all_rows)
    if "ts_code" in df.columns:
        df["qlib_code"] = df["ts_code"].apply(
            lambda x: ('sz' + x[:6]) if x.endswith('.SZ') else ('sh' + x[:6]))

    if existing is not None:
        df = pd.concat([existing, df], ignore_index=True)

    df.to_parquet(str(output_path), index=False)
    logger.info(f"Saved block_trade: {df.shape}")


def fetch_broker_recommend(start: str, end: str):
    """券商评级 — query by month."""
    from ST_CLIENT import StockToday
    st = StockToday(token=get_token())
    output_path = DATA_DIR / "st_broker_recommend.parquet"

    existing = None
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        logger.info(f"Resuming broker_recommend ({len(existing)} rows)")

    # Query by month
    months = pd.date_range(start=start, end=end, freq="MS").strftime("%Y%m").tolist()
    logger.info(f"Fetching broker_recommend: {len(months)} months")

    all_rows = []
    for i, m in enumerate(months):
        try:
            result = st.broker_recommend(month=m)
            if isinstance(result, list) and result:
                all_rows.extend(result)
            elif isinstance(result, dict) and result.get("msg"):
                if i == 0:
                    logger.warning(f"  API msg: {result['msg']}")
                    if "TOKEN" in result["msg"] or "套餐" in result["msg"]:
                        return
            if (i + 1) % 10 == 0:
                logger.info(f"  {i+1}/{len(months)}: {len(all_rows)} rows")
            time.sleep(0.2)
        except Exception as e:
            logger.warning(f"  {m} failed: {e}")

    if not all_rows:
        logger.info("No new broker_recommend data")
        return

    df = pd.DataFrame(all_rows)
    if "ts_code" in df.columns:
        df["qlib_code"] = df["ts_code"].apply(
            lambda x: ('sz' + x[:6]) if x.endswith('.SZ') else ('sh' + x[:6]))

    if existing is not None:
        df = pd.concat([existing, df], ignore_index=True)

    df.to_parquet(str(output_path), index=False)
    logger.info(f"Saved broker_recommend: {df.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="all",
                        choices=["all", "holdertrade", "industry_mf", "block_trade", "broker"])
    parser.add_argument("--start", default="20210101")
    parser.add_argument("--end", default=datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()

    apis = {
        "holdertrade": fetch_holdertrade,
        "industry_mf": fetch_industry_moneyflow,
        "block_trade": fetch_block_trade,
        "broker": fetch_broker_recommend,
    }

    if args.api == "all":
        for name, func in apis.items():
            logger.info(f"\n{'='*40} {name} {'='*40}")
            try:
                func(args.start, args.end)
            except Exception as e:
                logger.error(f"{name} failed: {e}")
    else:
        apis[args.api](args.start, args.end)

    logger.info("\nAll done!")


if __name__ == "__main__":
    main()
