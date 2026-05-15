"""Fetch factors via ST_CLIENT - Round 2.

Round 1 (done): margin_detail, top_list, limit_list_d, moneyflow_hsgt
Round 2 (this): fina_indicator, stk_holdernumber, moneyflow, daily_basic

Mode A (batch by trade_date): moneyflow (timeout=60s, partial ~193 stocks/day)
Mode B (by stock): fina_indicator, stk_holdernumber, daily_basic

Saves to:
  data/storage/st_fina_indicator.parquet
  data/storage/st_holder_number.parquet
  data/storage/st_moneyflow_detail.parquet
  data/storage/st_daily_basic.parquet

Usage:
    python scripts/fetch_top5_factors.py                          # all round 2
    python scripts/fetch_top5_factors.py --only fina_indicator    # just one
    python scripts/fetch_top5_factors.py --top 1000               # top 1000 stocks
    python scripts/fetch_top5_factors.py --days 60                # moneyflow recent 60 days
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
import requests as req

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
WORKING_URLS = ["http://111.229.164.2:8083/", "https://tushare.citydata.club/"]
NO_PROXY = {"http": None, "https": None}


def get_token():
    from config.settings import ST_TOKEN
    return ST_TOKEN


def post_api(endpoint, params, timeout=30):
    params["TOKEN"] = get_token()
    for url in WORKING_URLS:
        try:
            r = req.post(f"{url}{endpoint}", data=params, timeout=timeout, proxies=NO_PROXY)
            if r.status_code == 200 and r.text.strip():
                data = r.json()
                if isinstance(data, dict) and data.get("code") == 0:
                    d = data.get("data", {})
                    if isinstance(d, dict) and "items" in d:
                        cols = d.get("fields") or d.get("columns")
                        return pd.DataFrame(d["items"], columns=cols) if cols else pd.DataFrame(d["items"])
                elif isinstance(data, dict) and data.get("code") == 1:
                    msg = data.get("msg", "")
                    if "超限" in msg:
                        raise RuntimeError(f"API limit: {msg}")
                    return pd.DataFrame()
                elif isinstance(data, list) and data:
                    return pd.DataFrame(data)
                return pd.DataFrame()
        except RuntimeError:
            raise
        except Exception:
            continue
    return pd.DataFrame()


def ts_to_qlib(ts_code):
    if not isinstance(ts_code, str) or "." not in ts_code:
        return str(ts_code)
    num, ex = ts_code.split(".", 1)
    return f"{ex}{num}"


def qlib_to_ts(code):
    num = code.replace("SH", "").replace("SZ", "").replace("BJ", "")
    if code.startswith("SH"): return f"{num}.SH"
    elif code.startswith("SZ"): return f"{num}.SZ"
    elif code.startswith("BJ"): return f"{num}.BJ"
    return f"{num}.SZ"


def get_stock_codes(top_n=None):
    features_dir = DATA_DIR / "qlib_data" / "cn_data" / "features"
    codes = sorted([d.name.upper() for d in features_dir.iterdir() if d.is_dir()])
    if top_n and top_n < len(codes):
        codes = codes[:top_n]
    return codes


def get_trade_dates(days=60):
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=int(days * 1.6))).strftime("%Y%m%d")
    try:
        df = post_api("trade_cal", {"exchange": "SSE", "start_date": start,
                                     "end_date": end, "is_open": "1"}, timeout=10)
        if not df.empty and "cal_date" in df.columns:
            dates = sorted(df["cal_date"].astype(str).tolist())
            return dates[-days:]
    except Exception:
        pass
    return [d.strftime("%Y%m%d") for d in pd.bdate_range(start, end)][-days:]


def save_parquet(df, path, dedup_cols, label):
    if df.empty:
        return
    # Force numeric on non-string columns
    skip = {"ts_code", "qlib_code", "code", "date", "trade_date", "name",
            "ann_date", "end_date", "industry"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
        if all(c in df.columns for c in dedup_cols):
            df = df.drop_duplicates(subset=dedup_cols, keep="last")
            df = df.sort_values(dedup_cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"  Saved: {path.name} ({len(df)} rows)")


class APILimitReached(Exception):
    pass


# ========== By-stock fetchers ==========

def fetch_by_stock(codes, endpoint, output_path, dedup_cols, label, checkpoint=200):
    """Fetch one endpoint per stock. Auto-skip already fetched."""
    existing = set()
    if output_path.exists():
        try:
            old = pd.read_parquet(output_path, columns=["ts_code"])
            existing = set(old["ts_code"].unique())
        except Exception:
            pass

    todo_codes = [c for c in codes if qlib_to_ts(c) not in existing]
    if not todo_codes:
        logger.info(f"  {label}: all {len(codes)} stocks already fetched")
        return

    logger.info(f"  {label}: {len(todo_codes)} to fetch (skipped {len(codes)-len(todo_codes)})")

    all_dfs = []
    ok = 0
    fail = 0
    consecutive_fail = 0

    for i, code in enumerate(todo_codes):
        ts_code = qlib_to_ts(code)
        time.sleep(1.0)
        try:
            df = post_api(endpoint, {"ts_code": ts_code}, timeout=30)
            if not df.empty:
                df["qlib_code"] = code
                all_dfs.append(df)
                ok += 1
                consecutive_fail = 0
            else:
                fail += 1
                consecutive_fail += 1
        except RuntimeError as e:
            logger.error(f"  {label} stopped: {e}")
            break
        except Exception:
            fail += 1
            consecutive_fail += 1

        if consecutive_fail >= 10:
            logger.error(f"  {label}: 10 consecutive failures, stopping")
            break

        if (i + 1) % 100 == 0 or (i + 1) == len(todo_codes):
            logger.info(f"  {label}: {i+1}/{len(todo_codes)} ({ok} ok, {fail} fail)")

        if (i + 1) % checkpoint == 0 and all_dfs:
            batch = pd.concat(all_dfs, ignore_index=True)
            save_parquet(batch, output_path, dedup_cols, label)
            all_dfs = []

    if all_dfs:
        batch = pd.concat(all_dfs, ignore_index=True)
        save_parquet(batch, output_path, dedup_cols, label)

    if output_path.exists():
        final = pd.read_parquet(output_path)
        n_stocks = final["ts_code"].nunique() if "ts_code" in final.columns else "?"
        logger.info(f"  {label} total: {len(final)} rows, {n_stocks} stocks")


# ========== By-date fetcher (moneyflow) ==========

def fetch_moneyflow_by_date(trade_dates, output_path):
    """Fetch moneyflow by trade_date. Returns ~193 stocks per day (partial but PIT-safe)."""
    label = "moneyflow_detail"

    # Skip existing dates
    existing_dates = set()
    if output_path.exists():
        try:
            old = pd.read_parquet(output_path, columns=["date"])
            existing_dates = set(old["date"].unique())
        except Exception:
            pass

    todo = [d for d in trade_dates if d not in existing_dates]
    if not todo:
        logger.info(f"  {label}: all {len(trade_dates)} dates already fetched")
        return

    logger.info(f"  {label}: {len(todo)} dates to fetch (skipped {len(trade_dates)-len(todo)})")

    all_dfs = []
    consecutive_fail = 0

    for i, td in enumerate(todo):
        time.sleep(1.5)
        try:
            df = post_api("moneyflow", {"trade_date": td}, timeout=60)
            if not df.empty:
                if "ts_code" in df.columns:
                    df["qlib_code"] = df["ts_code"].apply(ts_to_qlib)
                df["date"] = td
                all_dfs.append(df)
                consecutive_fail = 0
                if (i + 1) % 20 == 0 or i == 0:
                    logger.info(f"  {label} {td}: {len(df)} stocks")
            else:
                consecutive_fail += 1
        except RuntimeError as e:
            logger.error(f"  {label} stopped: {e}")
            break
        except Exception:
            consecutive_fail += 1

        if consecutive_fail >= 5:
            logger.error(f"  {label}: 5 consecutive failures, stopping")
            break

        # Checkpoint every 50 days
        if (i + 1) % 50 == 0 and all_dfs:
            batch = pd.concat(all_dfs, ignore_index=True)
            save_parquet(batch, output_path, ["qlib_code", "date"], label)
            all_dfs = []

    if all_dfs:
        batch = pd.concat(all_dfs, ignore_index=True)
        save_parquet(batch, output_path, ["qlib_code", "date"], label)

    if output_path.exists():
        final = pd.read_parquet(output_path)
        logger.info(f"  {label} total: {len(final)} rows, {final['date'].nunique()} dates")


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description="Fetch factors Round 2")
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated: fina_indicator,stk_holdernumber,moneyflow,daily_basic")
    parser.add_argument("--top", type=int, default=None, help="Top N stocks")
    parser.add_argument("--days", type=int, default=250, help="Trading days for moneyflow (default 250)")
    parser.add_argument("--checkpoint", type=int, default=200)
    args = parser.parse_args()

    selected = set(s.strip() for s in args.only.split(",")) if args.only else None
    codes = get_stock_codes(top_n=args.top)
    logger.info(f"Stocks: {len(codes)}")

    # 1. fina_indicator (by stock, ~100 rows each, 100+ columns)
    if not selected or "fina_indicator" in selected:
        logger.info("=== fina_indicator (by stock) ===")
        fetch_by_stock(codes, "fina_indicator",
                       DATA_DIR / "st_fina_indicator.parquet",
                       ["ts_code", "end_date"], "fina_indicator",
                       checkpoint=args.checkpoint)

    # 2. stk_holdernumber (by stock, ~95 rows each)
    if not selected or "stk_holdernumber" in selected:
        logger.info("=== stk_holdernumber (by stock) ===")
        fetch_by_stock(codes, "stk_holdernumber",
                       DATA_DIR / "st_holder_number.parquet",
                       ["ts_code", "end_date"], "stk_holdernumber",
                       checkpoint=args.checkpoint)

    # 3. moneyflow (by trade_date, partial ~193 stocks/day)
    if not selected or "moneyflow" in selected:
        logger.info("=== moneyflow (by date, 60s timeout) ===")
        trade_dates = get_trade_dates(args.days)
        logger.info(f"  Trade dates: {len(trade_dates)}")
        fetch_moneyflow_by_date(trade_dates,
                                DATA_DIR / "st_moneyflow_detail.parquet")

    # 4. daily_basic (by stock - trade_date mode times out)
    if not selected or "daily_basic" in selected:
        logger.info("=== daily_basic (by stock) ===")
        fetch_by_stock(codes, "daily_basic",
                       DATA_DIR / "st_daily_basic.parquet",
                       ["ts_code", "trade_date"], "daily_basic",
                       checkpoint=args.checkpoint)

    logger.info("Done!")


if __name__ == "__main__":
    main()
