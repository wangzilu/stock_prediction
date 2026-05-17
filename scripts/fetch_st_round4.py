"""Fetch ST_CLIENT Round 4: high-value daily factors.

1. moneyflow       — 个股资金流 (主力/超大/大/中/小单净流入, 日频每只股)
2. stk_holdertrade — 高管/股东增减持 (事件型, 按公告日)
3. cyq_perf        — 筹码分布绩效 (获利比例/平均成本, 日频)
4. stk_factor_pro  — 官方技术因子 (MACD/KDJ/RSI/BOLL等, 日频)
5. forecast        — 业绩预告 (预计净利润增幅, 按公告日)

Usage:
    python scripts/fetch_st_round4.py
    python scripts/fetch_st_round4.py --api moneyflow
    python scripts/fetch_st_round4.py --api holdertrade
    python scripts/fetch_st_round4.py --start 20210101
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

from ST_CLIENT import StockToday

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
TOKEN = os.environ.get("ST_TOKEN", "")
if not TOKEN:
    # Try loading from project config
    _token_file = PROJECT_ROOT / ".st_token"
    if _token_file.exists():
        TOKEN = _token_file.read_text().strip()
    else:
        raise RuntimeError(
            "ST_TOKEN not set. Either export ST_TOKEN=<token> or "
            "create .st_token file in project root."
        )


def get_trade_dates(start: str, end: str) -> list[str]:
    """Generate business dates between start and end (YYYYMMDD format)."""
    dates = pd.bdate_range(start=start, end=end)
    return [d.strftime("%Y%m%d") for d in dates]


def fetch_moneyflow(start: str, end: str):
    """个股资金流: buy_sm/sell_sm/buy_md/sell_md/buy_lg/sell_lg/buy_elg/sell_elg/net_mf.

    Daily per-stock. ~5000 stocks/day. Rate limit: ~200 req/min.
    """
    st = StockToday(token=TOKEN)
    output_path = DATA_DIR / "st_moneyflow.parquet"

    # Resume from existing data
    existing = None
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        last_date = existing["date"].max()
        logger.info(f"Resuming moneyflow from {last_date} ({len(existing)} rows)")
        start = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y%m%d")

    dates = get_trade_dates(start, end)
    logger.info(f"Fetching moneyflow: {len(dates)} dates ({start}~{end})")

    all_rows = []
    for i, d in enumerate(dates):
        try:
            result = st.moneyflow(trade_date=d)
            if isinstance(result, list) and result:
                for r in result:
                    r["trade_date"] = d
                all_rows.extend(result)
                if (i + 1) % 50 == 0:
                    logger.info(f"  {i+1}/{len(dates)}: {d}, total rows={len(all_rows)}")
            time.sleep(0.12)  # Rate limit
        except Exception as e:
            logger.warning(f"  {d} failed: {e}")
            time.sleep(1)

    if not all_rows:
        logger.info("No new moneyflow data")
        return

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")

    # Convert ts_code to qlib_code
    if "ts_code" in df.columns:
        df["qlib_code"] = df["ts_code"].apply(
            lambda x: ('sz' + x[:6]) if x.endswith('.SZ') else ('sh' + x[:6]))

    # Select and prefix factor columns
    factor_cols = ["buy_sm_vol", "sell_sm_vol", "buy_md_vol", "sell_md_vol",
                   "buy_lg_vol", "sell_lg_vol", "buy_elg_vol", "sell_elg_vol",
                   "net_mf_vol", "net_mf_amount",
                   "buy_sm_amount", "sell_sm_amount", "buy_md_amount", "sell_md_amount",
                   "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount"]
    available = [c for c in factor_cols if c in df.columns]
    out_cols = {c: f"st_{c}" for c in available}
    df = df.rename(columns=out_cols)
    st_cols = list(out_cols.values())

    for c in st_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = df[["qlib_code", "date"] + st_cols].dropna(subset=["qlib_code", "date"])

    # Merge with existing
    if existing is not None:
        out = pd.concat([existing, out], ignore_index=True)
        out = out.drop_duplicates(["qlib_code", "date"], keep="last")

    out = out.sort_values(["qlib_code", "date"]).reset_index(drop=True)
    out.to_parquet(str(output_path), index=False)
    logger.info(f"Saved moneyflow: {out.shape}, {out['qlib_code'].nunique()} stocks, "
                f"{out['date'].nunique()} dates")


def fetch_holdertrade(start: str, end: str):
    """高管/股东增减持: ann_date, holder_name, change_vol, after_share, after_ratio.

    Query by ann_date. Event-type data.
    """
    st = StockToday(token=TOKEN)
    output_path = DATA_DIR / "st_holdertrade.parquet"

    existing = None
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        last_date = existing["ann_date"].max()
        logger.info(f"Resuming holdertrade from {last_date} ({len(existing)} rows)")
        start = (pd.to_datetime(str(last_date)) + timedelta(days=1)).strftime("%Y%m%d")

    dates = get_trade_dates(start, end)
    # Sample every 5 days to reduce API calls (events are sparse)
    sample_dates = dates[::5] if len(dates) > 100 else dates
    logger.info(f"Fetching holdertrade: {len(sample_dates)} sample dates")

    all_rows = []
    for i, d in enumerate(sample_dates):
        try:
            result = st.stk_holdertrade(ann_date=d)
            if isinstance(result, list) and result:
                all_rows.extend(result)
                if (i + 1) % 50 == 0:
                    logger.info(f"  {i+1}/{len(sample_dates)}: {d}, rows={len(all_rows)}")
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

    # Keep key columns
    keep = ["qlib_code", "ann_date", "holder_name", "holder_type",
            "in_de", "change_vol", "change_ratio", "after_share", "after_ratio"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    if existing is not None:
        out = pd.concat([existing, out], ignore_index=True)
        out = out.drop_duplicates(subset=["qlib_code", "ann_date", "holder_name"], keep="last")

    out.to_parquet(str(output_path), index=False)
    logger.info(f"Saved holdertrade: {out.shape}")


def fetch_cyq_perf(start: str, end: str):
    """筹码分布绩效: 获利比例/平均成本/集中度.

    Daily per-stock. Query by trade_date.
    """
    st = StockToday(token=TOKEN)
    output_path = DATA_DIR / "st_cyq_perf.parquet"

    existing = None
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        last_date = existing["date"].max()
        logger.info(f"Resuming cyq_perf from {last_date} ({len(existing)} rows)")
        start = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y%m%d")

    dates = get_trade_dates(start, end)
    logger.info(f"Fetching cyq_perf: {len(dates)} dates ({start}~{end})")

    all_rows = []
    for i, d in enumerate(dates):
        try:
            result = st.cyq_perf(trade_date=d)
            if isinstance(result, list) and result:
                for r in result:
                    r["trade_date"] = d
                all_rows.extend(result)
                if (i + 1) % 50 == 0:
                    logger.info(f"  {i+1}/{len(dates)}: {d}, total rows={len(all_rows)}")
            time.sleep(0.12)
        except Exception as e:
            logger.warning(f"  {d} failed: {e}")
            time.sleep(1)

    if not all_rows:
        logger.info("No new cyq_perf data")
        return

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")

    if "ts_code" in df.columns:
        df["qlib_code"] = df["ts_code"].apply(
            lambda x: ('sz' + x[:6]) if x.endswith('.SZ') else ('sh' + x[:6]))

    # Prefix columns
    factor_cols = ["his_low", "his_high", "his_low_alloc", "his_high_alloc",
                   "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", "cost_95pct",
                   "weight_avg", "winner_rate"]
    available = [c for c in factor_cols if c in df.columns]
    out_cols = {c: f"cyq_{c}" for c in available}
    df = df.rename(columns=out_cols)
    st_cols = list(out_cols.values())

    for c in st_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = df[["qlib_code", "date"] + st_cols].dropna(subset=["qlib_code", "date"])

    if existing is not None:
        out = pd.concat([existing, out], ignore_index=True)
        out = out.drop_duplicates(["qlib_code", "date"], keep="last")

    out = out.sort_values(["qlib_code", "date"]).reset_index(drop=True)
    out.to_parquet(str(output_path), index=False)
    logger.info(f"Saved cyq_perf: {out.shape}, {out['qlib_code'].nunique()} stocks")


def fetch_factor_pro(start: str, end: str):
    """官方技术因子: MACD/KDJ/RSI/BOLL/ATR等.

    Daily per-stock. Query by trade_date.
    """
    st = StockToday(token=TOKEN)
    output_path = DATA_DIR / "st_factor_pro.parquet"

    existing = None
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        last_date = existing["date"].max()
        logger.info(f"Resuming factor_pro from {last_date} ({len(existing)} rows)")
        start = (pd.to_datetime(last_date) + timedelta(days=1)).strftime("%Y%m%d")

    dates = get_trade_dates(start, end)
    logger.info(f"Fetching stk_factor_pro: {len(dates)} dates ({start}~{end})")

    all_rows = []
    for i, d in enumerate(dates):
        try:
            result = st.stk_factor_pro(trade_date=d)
            if isinstance(result, list) and result:
                for r in result:
                    r["trade_date"] = d
                all_rows.extend(result)
                if (i + 1) % 50 == 0:
                    logger.info(f"  {i+1}/{len(dates)}: {d}, total rows={len(all_rows)}")
            time.sleep(0.12)
        except Exception as e:
            logger.warning(f"  {d} failed: {e}")
            time.sleep(1)

    if not all_rows:
        logger.info("No new factor_pro data")
        return

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")

    if "ts_code" in df.columns:
        df["qlib_code"] = df["ts_code"].apply(
            lambda x: ('sz' + x[:6]) if x.endswith('.SZ') else ('sh' + x[:6]))

    # Keep all numeric columns as factors
    skip = {"ts_code", "trade_date", "date", "qlib_code"}
    factor_cols = [c for c in df.columns if c not in skip]
    for c in factor_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Only keep columns with >10% non-null
    keep_cols = [c for c in factor_cols if df[c].notna().mean() > 0.1]
    out = df[["qlib_code", "date"] + keep_cols].dropna(subset=["qlib_code", "date"])

    if existing is not None:
        out = pd.concat([existing, out], ignore_index=True)
        out = out.drop_duplicates(["qlib_code", "date"], keep="last")

    out = out.sort_values(["qlib_code", "date"]).reset_index(drop=True)
    out.to_parquet(str(output_path), index=False)
    logger.info(f"Saved factor_pro: {out.shape}, {len(keep_cols)} factors")


def fetch_forecast(start: str, end: str):
    """业绩预告: 预计净利润/增幅, 按公告日.

    Query by ann_date period.
    """
    st = StockToday(token=TOKEN)
    output_path = DATA_DIR / "st_forecast.parquet"

    existing = None
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        logger.info(f"Resuming forecast ({len(existing)} rows)")

    # Fetch by quarter periods
    periods = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        for q in ["0331", "0630", "0930", "1231"]:
            periods.append(f"{year}{q}")

    logger.info(f"Fetching forecast: {len(periods)} periods")

    all_rows = []
    for period in periods:
        try:
            result = st.forecast(period=period)
            if isinstance(result, list) and result:
                all_rows.extend(result)
            time.sleep(0.2)
        except Exception as e:
            logger.warning(f"  {period} failed: {e}")
            time.sleep(1)

    if not all_rows:
        logger.info("No new forecast data")
        return

    df = pd.DataFrame(all_rows)
    if "ts_code" in df.columns:
        df["qlib_code"] = df["ts_code"].apply(
            lambda x: ('sz' + x[:6]) if x.endswith('.SZ') else ('sh' + x[:6]))

    keep = ["qlib_code", "ann_date", "end_date", "type", "p_change_min",
            "p_change_max", "net_profit_min", "net_profit_max", "last_parent_net",
            "change_reason"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    if existing is not None:
        out = pd.concat([existing, out], ignore_index=True)
        out = out.drop_duplicates(subset=["qlib_code", "ann_date", "end_date"], keep="last")

    out.to_parquet(str(output_path), index=False)
    logger.info(f"Saved forecast: {out.shape}, {out['qlib_code'].nunique()} stocks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", type=str, default="all",
                        choices=["all", "moneyflow", "holdertrade", "cyq", "factor_pro", "forecast"],
                        help="Which API to fetch")
    parser.add_argument("--start", type=str, default="20210101",
                        help="Start date YYYYMMDD")
    parser.add_argument("--end", type=str, default=None,
                        help="End date YYYYMMDD (default: today)")
    args = parser.parse_args()

    end = args.end or datetime.now().strftime("%Y%m%d")
    start = args.start

    apis = {
        "moneyflow": fetch_moneyflow,
        "holdertrade": fetch_holdertrade,
        "cyq": fetch_cyq_perf,
        "factor_pro": fetch_factor_pro,
        "forecast": fetch_forecast,
    }

    if args.api == "all":
        for name, func in apis.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Fetching: {name}")
            logger.info(f"{'='*60}")
            try:
                func(start, end)
            except Exception as e:
                logger.error(f"{name} failed: {e}")
    else:
        apis[args.api](start, end)

    logger.info("\nAll done!")


if __name__ == "__main__":
    main()
