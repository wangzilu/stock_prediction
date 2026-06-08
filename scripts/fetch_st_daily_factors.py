"""Fetch daily factors via ST_CLIENT (Tushare-compatible API).

Batch-fetches by trade_date (全市场一次请求), naturally PIT-safe.
Sources: daily_basic (PE/PB/PS/turnover/mv) + moneyflow (资金流).

Saves to: data/storage/st_daily_basic.parquet
          data/storage/st_moneyflow.parquet

Usage:
    python scripts/fetch_st_daily_factors.py --days 60
    python scripts/fetch_st_daily_factors.py --start 20260101 --end 20260512
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"


def get_st_client():
    from config.settings import ST_TOKEN
    from ST_CLIENT import StockToday
    return StockToday(token=ST_TOKEN)


def parse_response(resp) -> pd.DataFrame:
    """Parse ST_CLIENT API response into DataFrame."""
    if resp is None:
        return pd.DataFrame()
    if isinstance(resp, list):
        return pd.DataFrame(resp)
    if not isinstance(resp, dict):
        return pd.DataFrame()
    if "error" in resp:
        logger.warning(f"API error: {resp['error']}")
        return pd.DataFrame()

    code = resp.get("code")
    if code not in (None, 0, "0"):
        logger.warning(f"API error: code={code}, msg={resp.get('msg', '')}")
        return pd.DataFrame()

    data = resp.get("data")
    if not data:
        return pd.DataFrame()
    if isinstance(data, dict) and "items" in data:
        items = data.get("items")
        columns = data.get("fields") or data.get("columns")
        if items and columns:
            return pd.DataFrame(items, columns=columns)
        if items:
            return pd.DataFrame(items)
    if isinstance(data, list):
        return pd.DataFrame(data)
    return pd.DataFrame()


def ts_to_qlib_code(ts_code: str) -> str:
    """Convert 600519.SH -> SH600519."""
    if not isinstance(ts_code, str) or "." not in ts_code:
        return ts_code
    num, ex = ts_code.split(".", 1)
    return f"{ex}{num}"


def get_trade_dates(st, start_date: str, end_date: str) -> list:
    """Get trading dates from trade_cal."""
    resp = st.trade_cal(exchange="SSE", start_date=start_date,
                        end_date=end_date, is_open="1")
    df = parse_response(resp)
    if df.empty:
        # Fallback: generate weekdays
        logger.warning("trade_cal failed, using weekday fallback")
        dates = pd.bdate_range(start_date, end_date)
        return [d.strftime("%Y%m%d") for d in dates]
    if "cal_date" in df.columns:
        return sorted(df["cal_date"].tolist())
    return []


def fetch_daily_basic(st, trade_dates: list) -> pd.DataFrame:
    """Fetch daily_basic for each trade_date (全市场一次请求)."""
    all_dfs = []
    for i, td in enumerate(trade_dates):
        resp = st.daily_basic(trade_date=td)
        df = parse_response(resp)
        if not df.empty:
            all_dfs.append(df)
            logger.info(f"  daily_basic {td}: {len(df)} stocks")
        else:
            logger.warning(f"  daily_basic {td}: empty")

        # Rate limit
        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)

    # Normalize
    if "ts_code" in result.columns:
        result["qlib_code"] = result["ts_code"].apply(ts_to_qlib_code)
    if "trade_date" in result.columns:
        result["date"] = result["trade_date"].astype(str)

    # Select and rename key columns
    col_map = {
        "pe_ttm": "st_pe_ttm",
        "pb": "st_pb",
        "ps_ttm": "st_ps_ttm",
        "turnover_rate_f": "st_turnover",
        "total_mv": "st_total_mv",
        "circ_mv": "st_circ_mv",
        "dv_ratio": "st_div_yield",
    }
    keep = ["qlib_code", "date"] + [c for c in col_map if c in result.columns]
    result = result[keep].copy()
    result = result.rename(columns=col_map)

    # Convert numeric
    num_cols = [c for c in result.columns if c.startswith("st_")]
    for c in num_cols:
        result[c] = pd.to_numeric(result[c], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)

    logger.info(f"daily_basic total: {len(result)} rows, "
                f"{result['qlib_code'].nunique()} stocks, {len(trade_dates)} dates")
    return result


def fetch_moneyflow(st, trade_dates: list) -> pd.DataFrame:
    """Fetch moneyflow for each trade_date."""
    all_dfs = []
    for i, td in enumerate(trade_dates):
        resp = st.moneyflow(trade_date=td)
        df = parse_response(resp)
        if not df.empty:
            all_dfs.append(df)
            logger.info(f"  moneyflow {td}: {len(df)} stocks")
        else:
            logger.warning(f"  moneyflow {td}: empty")

        if (i + 1) % 10 == 0:
            time.sleep(0.5)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)

    if "ts_code" in result.columns:
        result["qlib_code"] = result["ts_code"].apply(ts_to_qlib_code)
    if "trade_date" in result.columns:
        result["date"] = result["trade_date"].astype(str)

    # Key money flow columns
    col_map = {
        "buy_sm_amount": "st_buy_sm",       # 小单买入
        "sell_sm_amount": "st_sell_sm",      # 小单卖出
        "buy_md_amount": "st_buy_md",       # 中单买入
        "sell_md_amount": "st_sell_md",      # 中单卖出
        "buy_lg_amount": "st_buy_lg",       # 大单买入
        "sell_lg_amount": "st_sell_lg",      # 大单卖出
        "buy_elg_amount": "st_buy_elg",     # 超大单买入
        "sell_elg_amount": "st_sell_elg",   # 超大单卖出
        "net_mf_amount": "st_net_mf",       # 净流入
    }
    keep = ["qlib_code", "date"] + [c for c in col_map if c in result.columns]
    result = result[keep].copy()
    result = result.rename(columns=col_map)

    # Compute derived features
    if "st_buy_elg" in result.columns and "st_sell_elg" in result.columns:
        result["st_net_elg"] = result["st_buy_elg"] - result["st_sell_elg"]
    if "st_buy_lg" in result.columns and "st_sell_lg" in result.columns:
        result["st_net_lg"] = result["st_buy_lg"] - result["st_sell_lg"]

    num_cols = [c for c in result.columns if c.startswith("st_")]
    for c in num_cols:
        result[c] = pd.to_numeric(result[c], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)

    logger.info(f"moneyflow total: {len(result)} rows, "
                f"{result['qlib_code'].nunique()} stocks, {len(trade_dates)} dates")
    return result


def save_with_merge(new_df: pd.DataFrame, out: Path, dedup_cols: list[str], label: str) -> None:
    """Append new rows to an existing parquet and de-duplicate by key."""
    if new_df.empty:
        return

    merged = new_df.copy()
    if out.exists():
        old = pd.read_parquet(out)
        merged = pd.concat([old, merged], ignore_index=True, sort=False)

    if all(col in merged.columns for col in dedup_cols):
        merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
        merged = merged.sort_values(dedup_cols)
    else:
        missing = [col for col in dedup_cols if col not in merged.columns]
        logger.warning(f"{label}: skip dedup, missing columns: {missing}")

    out.parent.mkdir(parents=True, exist_ok=True)
    # 2026-06-08: ST_CLIENT's `date` column can come back as mixed
    # str/int (e.g. '2026-06-08' from new rows, 20260605 int from old
    # rows after a schema change). pyarrow.Table.from_pandas then
    # raises ArrowTypeError. Same coerce pattern as fund_flow ggt_ss.
    for _col in merged.select_dtypes(include="object").columns:
        merged[_col] = merged[_col].astype(str)
    merged.to_parquet(out, index=False)
    logger.info(f"Saved: {out} ({len(merged)} rows)")


def main():
    from scheduler.data_health import HealthStatus, write_health

    parser = argparse.ArgumentParser(description="Fetch daily factors via ST_CLIENT")
    parser.add_argument("--days", type=int, default=60, help="Number of recent trading days")
    parser.add_argument("--start", type=str, default="", help="Start date YYYYMMDD")
    parser.add_argument("--end", type=str, default="", help="End date YYYYMMDD")
    parser.add_argument("--skip-moneyflow", action="store_true")
    args = parser.parse_args()

    st = get_st_client()

    # Determine date range
    if args.start and args.end:
        start_date, end_date = args.start, args.end
    else:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=int(args.days * 1.5))).strftime("%Y%m%d")

    logger.info(f"Date range: {start_date} ~ {end_date}")

    # Get trade dates
    trade_dates = get_trade_dates(st, start_date, end_date)
    if args.days and len(trade_dates) > args.days:
        trade_dates = trade_dates[-args.days:]
    logger.info(f"Trade dates: {len(trade_dates)}")

    try:
        # Fetch daily_basic
        logger.info("Fetching daily_basic...")
        db = fetch_daily_basic(st, trade_dates)
        if not db.empty:
            out = DATA_DIR / "st_daily_basic.parquet"
            save_with_merge(db, out, ["qlib_code", "date"], "daily_basic")
            write_health("st_daily_basic_update", HealthStatus(
                success=True,
                n_items=len(db),
                latest_date=str(db["date"].max()) if "date" in db.columns else "",
                network_profile="domestic",
            ))
        else:
            write_health("st_daily_basic_update", HealthStatus(
                success=False,
                error_type="NoData",
                error_message="daily_basic returned empty",
                network_profile="domestic",
            ))

        # Fetch moneyflow
        if not args.skip_moneyflow:
            logger.info("Fetching moneyflow...")
            mf = fetch_moneyflow(st, trade_dates)
            if not mf.empty:
                out = DATA_DIR / "st_moneyflow.parquet"
                save_with_merge(mf, out, ["qlib_code", "date"], "moneyflow")
                write_health("st_moneyflow_update", HealthStatus(
                    success=True,
                    n_items=len(mf),
                    latest_date=str(mf["date"].max()) if "date" in mf.columns else "",
                    network_profile="domestic",
                ))
            else:
                write_health("st_moneyflow_update", HealthStatus(
                    success=False,
                    error_type="NoData",
                    error_message="moneyflow returned empty",
                    network_profile="domestic",
                ))

        logger.info("Done!")
    except Exception as e:
        write_health("st_daily_basic_update", HealthStatus(
            success=False,
            error_type=type(e).__name__,
            error_message=str(e)[:200],
            network_profile="domestic",
        ))
        if not args.skip_moneyflow:
            write_health("st_moneyflow_update", HealthStatus(
                success=False,
                error_type=type(e).__name__,
                error_message=str(e)[:200],
                network_profile="domestic",
            ))
        raise


if __name__ == "__main__":
    main()
