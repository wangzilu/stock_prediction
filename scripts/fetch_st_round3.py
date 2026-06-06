"""Fetch remaining ST_CLIENT data - Round 3.

Already done: margin_detail, top_list, limit_list_d, moneyflow_hsgt, fina_indicator
This round: stk_holdernumber, daily_basic, pledge_stat, index_classify+members

All by-stock endpoints, auto-checkpoint, auto-skip already fetched.

Usage:
    python scripts/fetch_st_round3.py                           # all
    python scripts/fetch_st_round3.py --only stk_holdernumber   # just one
    python scripts/fetch_st_round3.py --top 1000                # first 1000 stocks
"""
import argparse
import logging
import os
import sys
import time
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
URLS = ["http://111.229.164.2:8083/", "https://tushare.citydata.club/"]
NO_PROXY = {"http": None, "https": None}


def _write_health(source: str, *, success: bool, n_items: int = 0,
                  latest_date: str = "", coverage: float = 0.0,
                  error_type: str = "", error_message: str = "") -> None:
    """Write data-health for endpoints that are production feature sources."""
    try:
        from scheduler.data_health import HealthStatus, write_health
        write_health(source, HealthStatus(
            success=success,
            n_items=n_items,
            latest_date=latest_date,
            coverage=coverage,
            error_type=error_type,
            error_message=error_message[:200],
            network_profile="domestic",
            partial=not success,
        ))
    except Exception as exc:
        logger.warning("write_health(%s) failed: %s", source, exc)


def get_token():
    from config.settings import ST_TOKEN
    return ST_TOKEN


def post_api(endpoint, params, timeout=30):
    params["TOKEN"] = get_token()
    for url in URLS:
        try:
            r = req.post(f"{url}{endpoint}", data=params, timeout=timeout, proxies=NO_PROXY)
            if r.status_code == 200 and r.text.strip():
                data = r.json()
                if isinstance(data, dict) and data.get("code") == 0:
                    d = data.get("data", {})
                    if isinstance(d, dict) and "items" in d:
                        cols = d.get("fields") or d.get("columns")
                        return pd.DataFrame(d["items"], columns=cols) if cols else pd.DataFrame(d["items"])
                elif isinstance(data, list) and data:
                    return pd.DataFrame(data)
                elif isinstance(data, dict) and data.get("code") == 1:
                    msg = data.get("msg", "")
                    if "超限" in msg:
                        raise RuntimeError(f"API limit: {msg}")
                return pd.DataFrame()
        except RuntimeError:
            raise
        except Exception:
            continue
    return pd.DataFrame()


def qlib_to_ts(code):
    num = code.replace("SH", "").replace("SZ", "").replace("BJ", "")
    if code.startswith("SH"): return f"{num}.SH"
    elif code.startswith("SZ"): return f"{num}.SZ"
    elif code.startswith("BJ"): return f"{num}.BJ"
    return f"{num}.SZ"


def ts_to_qlib(ts_code):
    if not isinstance(ts_code, str) or "." not in ts_code:
        return str(ts_code)
    num, ex = ts_code.split(".", 1)
    return f"{ex}{num}"


def get_stock_codes(top_n=None):
    d = DATA_DIR / "qlib_data" / "cn_data" / "features"
    codes = sorted([x.name.upper() for x in d.iterdir() if x.is_dir()])
    if top_n and top_n < len(codes):
        codes = codes[:top_n]
    return codes


def save_parquet(df, path, dedup_cols):
    if df.empty:
        return
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
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"  Saved: {path.name} ({len(df)} rows)")


def fetch_by_stock(codes, endpoint, output_path, dedup_cols, label,
                   checkpoint=200, force=False):
    existing = set()
    if output_path.exists() and not force:
        try:
            old = pd.read_parquet(output_path, columns=["ts_code"])
            existing = set(old["ts_code"].unique())
        except Exception:
            pass

    todo = [c for c in codes if qlib_to_ts(c) not in existing]
    if not todo:
        logger.info(f"  {label}: all {len(codes)} stocks done, skipping")
        final = pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame()
        return final, 0, 0
    logger.info(f"  {label}: {len(todo)} to fetch (skipped {len(codes)-len(todo)})")

    all_dfs = []
    ok, fail, consecutive_fail = 0, 0, 0

    for i, code in enumerate(todo):
        time.sleep(1.0)
        try:
            df = post_api(endpoint, {"ts_code": qlib_to_ts(code)}, timeout=30)
            if not df.empty:
                df["qlib_code"] = code
                all_dfs.append(df)
                ok += 1
                consecutive_fail = 0
            else:
                fail += 1
                consecutive_fail += 1
        except RuntimeError as e:
            logger.error(f"  Stopped: {e}")
            break
        except Exception:
            fail += 1
            consecutive_fail += 1

        if consecutive_fail >= 10:
            logger.error(f"  {label}: 10 consecutive failures, stopping")
            break
        if (i + 1) % 100 == 0 or (i + 1) == len(todo):
            logger.info(f"  {label}: {i+1}/{len(todo)} ({ok} ok, {fail} fail)")
        if (i + 1) % checkpoint == 0 and all_dfs:
            save_parquet(pd.concat(all_dfs, ignore_index=True), output_path, dedup_cols)
            all_dfs = []

    if all_dfs:
        save_parquet(pd.concat(all_dfs, ignore_index=True), output_path, dedup_cols)

    if output_path.exists():
        final = pd.read_parquet(output_path)
        logger.info(f"  {label} total: {len(final)} rows, {final['ts_code'].nunique()} stocks")
        return final, ok, fail
    return pd.DataFrame(), ok, fail


def fetch_index_classify():
    """One-shot: 申万行业分类 + 成分股."""
    logger.info("=== index_classify (申万行业分类) ===")
    out = DATA_DIR / "st_index_classify.parquet"
    if out.exists():
        logger.info("  Already exists, skipping")
        return

    df = post_api("index_classify", {"level": "L1", "src": "SW2021"}, timeout=15)
    if not df.empty:
        save_parquet(df, out, ["index_code"])
        logger.info(f"  Got {len(df)} industry categories")

        # Fetch members for each industry
        logger.info("  Fetching industry members...")
        all_members = []
        for _, row in df.iterrows():
            idx_code = row.get("index_code", "")
            if not idx_code:
                continue
            time.sleep(1.0)
            mdf = post_api("index_member_all", {"l1_code": idx_code}, timeout=15)
            if not mdf.empty:
                mdf["industry_code"] = idx_code
                mdf["industry_name"] = row.get("industry_name", "")
                all_members.append(mdf)

        if all_members:
            members = pd.concat(all_members, ignore_index=True)
            if "ts_code" in members.columns:
                members["qlib_code"] = members["ts_code"].apply(ts_to_qlib)
            mout = DATA_DIR / "st_industry_members.parquet"
            save_parquet(members, mout, ["ts_code", "industry_code"])
            logger.info(f"  Industry members: {len(members)} rows")


def main():
    parser = argparse.ArgumentParser(description="Fetch ST data Round 3")
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated: stk_holdernumber,daily_basic,pledge_stat,index")
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Refetch stocks even if they already exist in the parquet")
    args = parser.parse_args()

    selected = set(s.strip() for s in args.only.split(",")) if args.only else None
    codes = get_stock_codes(top_n=args.top)
    logger.info(f"Stocks: {len(codes)}")

    if not selected or "stk_holdernumber" in selected:
        logger.info("=== stk_holdernumber (股东户数) ===")
        try:
            df, ok, fail = fetch_by_stock(
                codes, "stk_holdernumber",
                DATA_DIR / "st_holder_number.parquet",
                ["ts_code", "end_date"], "stk_holdernumber",
                force=args.force,
            )
            if df.empty:
                _write_health(
                    "st_holder_number_update",
                    success=False,
                    error_type="NoData",
                    error_message="stk_holdernumber produced no rows",
                )
                raise RuntimeError("stk_holdernumber produced no rows")
            latest_col = "ann_date" if "ann_date" in df.columns else "end_date"
            n_codes = int(df["ts_code"].nunique()) if "ts_code" in df.columns else 0
            _write_health(
                "st_holder_number_update",
                success=n_codes > 0,
                n_items=len(df),
                latest_date=str(df[latest_col].max()) if latest_col in df.columns else "",
                coverage=n_codes / max(len(codes), 1),
                error_type="" if n_codes > 0 else "NoCoverage",
                error_message="" if n_codes > 0 else f"0/{len(codes)} stocks covered",
            )
        except Exception as exc:
            _write_health(
                "st_holder_number_update",
                success=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise

    if not selected or "daily_basic" in selected:
        logger.info("=== daily_basic (PE/PB/PS/换手/市值) ===")
        fetch_by_stock(codes, "daily_basic",
                       DATA_DIR / "st_daily_basic.parquet",
                       ["ts_code", "trade_date"], "daily_basic",
                       force=args.force)

    if not selected or "pledge_stat" in selected:
        logger.info("=== pledge_stat (股权质押) ===")
        fetch_by_stock(codes, "pledge_stat",
                       DATA_DIR / "st_pledge_stat.parquet",
                       ["ts_code", "end_date"], "pledge_stat",
                       force=args.force)

    if not selected or "index" in selected:
        fetch_index_classify()

    logger.info("Done!")


if __name__ == "__main__":
    main()
