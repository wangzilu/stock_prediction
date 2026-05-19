"""Fetch ST_CLIENT Round 6: remaining high-value data.

Priority:
1. fund_portfolio — 公募基金重仓股 (季度, CX建议的拥挤度因子)
2. moneyflow_ind_dc — 行业资金流 (日频, 板块轮动信号)
3. dividend — 分红送转 (事件, 股息率因子)
4. repurchase — 回购 (事件, 正面信号)
5. stk_holdertrade — 高管增减持 (事件, 内部人信号)
6. stk_factor_pro — 官方技术因子 (日频)
7. express/express_vip — 业绩快报 (事件)
8. suspend_d — 停牌日历 (用于回测过滤)
9. block_trade — 大宗交易 (事件, 折价率)
10. top_inst — 机构调研 (事件)

Usage:
    python scripts/fetch_st_round6.py --api all
    python scripts/fetch_st_round6.py --api fund_portfolio
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


def get_st():
    from ST_CLIENT import StockToday
    return StockToday(token=get_token())


def safe_fetch(func, **kwargs):
    """Call ST API, return list or None. Handle permission errors."""
    try:
        result = func(**kwargs)
        if isinstance(result, list):
            return result if result else None
        if isinstance(result, dict):
            msg = result.get("msg", "")
            if "TOKEN" in msg or "套餐" in msg or "权限" in msg:
                logger.warning(f"  Permission denied: {msg}")
                return "PERMISSION_DENIED"
            return None
        return None
    except Exception as e:
        logger.warning(f"  Error: {e}")
        return None


def get_trade_dates(start, end):
    return [d.strftime("%Y%m%d") for d in pd.bdate_range(start=start, end=end)]


def save_df(df, name, existing_path=None, dedup_cols=None):
    """Save with resume support."""
    path = DATA_DIR / f"st_{name}.parquet"
    if existing_path and Path(existing_path).exists():
        old = pd.read_parquet(str(existing_path))
        df = pd.concat([old, df], ignore_index=True)
        if dedup_cols:
            df = df.drop_duplicates(subset=dedup_cols, keep="last")
    if "ts_code" in df.columns and "qlib_code" not in df.columns:
        df["qlib_code"] = df["ts_code"].apply(
            lambda x: ('sz' + x[:6]) if x.endswith('.SZ') else ('sh' + x[:6]))
    df.to_parquet(str(path), index=False)
    logger.info(f"  Saved {name}: {df.shape}")
    return path


# ========== Fetch functions ==========

def fetch_fund_portfolio(start, end):
    """公募基金重仓股 (季度)."""
    st = get_st()
    output = DATA_DIR / "st_fund_portfolio.parquet"
    periods = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        for q in ["0331", "0630", "0930", "1231"]:
            periods.append(f"{year}{q}")

    logger.info(f"Fetching fund_portfolio: {len(periods)} periods")
    all_rows = []
    for i, period in enumerate(periods):
        result = safe_fetch(st.fund_portfolio, period=period)
        if result == "PERMISSION_DENIED":
            return
        if result:
            all_rows.extend(result)
        if (i + 1) % 10 == 0:
            logger.info(f"  {i+1}/{len(periods)}: {len(all_rows)} rows")
        time.sleep(0.2)

    if all_rows:
        save_df(pd.DataFrame(all_rows), "fund_portfolio", output,
                dedup_cols=["ts_code", "symbol", "end_date"] if "symbol" in pd.DataFrame(all_rows).columns else None)


def fetch_moneyflow_ind(start, end):
    """行业资金流 (日频)."""
    st = get_st()
    output = DATA_DIR / "st_moneyflow_ind.parquet"
    dates = get_trade_dates(start, end)

    # Resume
    if output.exists():
        old = pd.read_parquet(str(output))
        if "trade_date" in old.columns:
            last = old["trade_date"].max()
            start = (pd.to_datetime(str(last)) + timedelta(days=1)).strftime("%Y%m%d")
            dates = get_trade_dates(start, end)
            logger.info(f"  Resuming from {start}")

    logger.info(f"Fetching moneyflow_ind_dc: {len(dates)} dates")
    all_rows = []
    for i, d in enumerate(dates):
        result = safe_fetch(st.moneyflow_ind_dc, trade_date=d)
        if result == "PERMISSION_DENIED":
            return
        if result:
            for r in result:
                r["trade_date"] = d
            all_rows.extend(result)
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(dates)}: {len(all_rows)} rows")
        time.sleep(0.12)

    if all_rows:
        save_df(pd.DataFrame(all_rows), "moneyflow_ind", output, dedup_cols=["trade_date", "ts_code"])


def fetch_dividend(start, end):
    """分红送转."""
    st = get_st()
    output = DATA_DIR / "st_dividend.parquet"
    all_rows = []
    # Query by ex_date range
    dates = get_trade_dates(start, end)
    sample = dates[::10]  # every 10 days
    logger.info(f"Fetching dividend: {len(sample)} sample dates")
    for i, d in enumerate(sample):
        result = safe_fetch(st.dividend, ex_date=d)
        if result == "PERMISSION_DENIED":
            return
        if result:
            all_rows.extend(result)
        time.sleep(0.15)

    if all_rows:
        save_df(pd.DataFrame(all_rows), "dividend", output)


def fetch_repurchase(start, end):
    """回购."""
    st = get_st()
    output = DATA_DIR / "st_repurchase.parquet"
    all_rows = []
    dates = get_trade_dates(start, end)
    sample = dates[::5]
    logger.info(f"Fetching repurchase: {len(sample)} dates")
    for i, d in enumerate(sample):
        result = safe_fetch(st.repurchase, ann_date=d)
        if result == "PERMISSION_DENIED":
            return
        if result:
            all_rows.extend(result)
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(sample)}: {len(all_rows)} rows")
        time.sleep(0.15)

    if all_rows:
        save_df(pd.DataFrame(all_rows), "repurchase", output)


def fetch_holdertrade(start, end):
    """高管增减持."""
    st = get_st()
    output = DATA_DIR / "st_holdertrade.parquet"
    dates = get_trade_dates(start, end)
    sample = dates[::3]
    logger.info(f"Fetching stk_holdertrade: {len(sample)} dates")
    all_rows = []
    for i, d in enumerate(sample):
        result = safe_fetch(st.stk_holdertrade, ann_date=d)
        if result == "PERMISSION_DENIED":
            return
        if result:
            all_rows.extend(result)
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(sample)}: {len(all_rows)} rows")
        time.sleep(0.15)

    if all_rows:
        save_df(pd.DataFrame(all_rows), "holdertrade", output,
                dedup_cols=["ts_code", "ann_date", "holder_name"] if "holder_name" in pd.DataFrame(all_rows).columns else None)


def fetch_suspend(start, end):
    """停牌日历."""
    st = get_st()
    output = DATA_DIR / "st_suspend.parquet"
    dates = get_trade_dates(start, end)
    logger.info(f"Fetching suspend_d: {len(dates)} dates")
    all_rows = []
    for i, d in enumerate(dates):
        result = safe_fetch(st.suspend_d, trade_date=d)
        if result == "PERMISSION_DENIED":
            return
        if result:
            all_rows.extend(result)
        if (i + 1) % 100 == 0:
            logger.info(f"  {i+1}/{len(dates)}: {len(all_rows)} rows")
        time.sleep(0.1)

    if all_rows:
        save_df(pd.DataFrame(all_rows), "suspend", output)


def fetch_express(start, end):
    """业绩快报."""
    st = get_st()
    output = DATA_DIR / "st_express.parquet"
    periods = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        for q in ["0331", "0630", "0930", "1231"]:
            periods.append(f"{year}{q}")
    logger.info(f"Fetching express: {len(periods)} periods")
    all_rows = []
    for i, p in enumerate(periods):
        result = safe_fetch(st.express, period=p)
        if result == "PERMISSION_DENIED":
            return
        if result:
            all_rows.extend(result)
        time.sleep(0.2)

    if all_rows:
        save_df(pd.DataFrame(all_rows), "express", output)


def fetch_block_trade(start, end):
    """大宗交易."""
    st = get_st()
    output = DATA_DIR / "st_block_trade.parquet"
    dates = get_trade_dates(start, end)
    logger.info(f"Fetching block_trade: {len(dates)} dates")
    all_rows = []
    for i, d in enumerate(dates):
        result = safe_fetch(st.block_trade, trade_date=d)
        if result == "PERMISSION_DENIED":
            return
        if result:
            all_rows.extend(result)
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(dates)}: {len(all_rows)} rows")
        time.sleep(0.12)

    if all_rows:
        save_df(pd.DataFrame(all_rows), "block_trade", output)


def fetch_top_inst(start, end):
    """机构调研."""
    st = get_st()
    output = DATA_DIR / "st_top_inst.parquet"
    dates = get_trade_dates(start, end)
    sample = dates[::5]
    logger.info(f"Fetching top_inst: {len(sample)} dates")
    all_rows = []
    for i, d in enumerate(sample):
        result = safe_fetch(st.top_inst, trade_date=d)
        if result == "PERMISSION_DENIED":
            return
        if result:
            all_rows.extend(result)
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(sample)}: {len(all_rows)} rows")
        time.sleep(0.15)

    if all_rows:
        save_df(pd.DataFrame(all_rows), "top_inst", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="all",
                        choices=["all", "fund_portfolio", "moneyflow_ind", "dividend",
                                 "repurchase", "holdertrade", "suspend", "express",
                                 "block_trade", "top_inst"])
    parser.add_argument("--start", default="20210101")
    parser.add_argument("--end", default=datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()

    apis = {
        "fund_portfolio": fetch_fund_portfolio,
        "moneyflow_ind": fetch_moneyflow_ind,
        "dividend": fetch_dividend,
        "repurchase": fetch_repurchase,
        "holdertrade": fetch_holdertrade,
        "suspend": fetch_suspend,
        "express": fetch_express,
        "block_trade": fetch_block_trade,
        "top_inst": fetch_top_inst,
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
