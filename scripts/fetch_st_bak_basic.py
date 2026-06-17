"""Fetch ST_CLIENT bak_basic — daily full-universe one-shot for PE/PB/total_share/
float_share/EPS/BVPS/holder_num/rev_yoy/profit_yoy/gpr/npr/industry/area.

Replaces (in steady state) baostock valuation_update + shareholder_update,
each of which currently takes 1.5-2 hr per-stock and blocks the downstream
chain. bak_basic returns the full A-share universe (~5,525 stocks) in a single
sub-second call.

Saves to: data/storage/st_bak_basic.parquet

Usage:
    python scripts/fetch_st_bak_basic.py                       # today only
    python scripts/fetch_st_bak_basic.py --start 2020-01-01    # backfill
    python scripts/fetch_st_bak_basic.py --start 2026-06-10 --end 2026-06-17
"""
from __future__ import annotations

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

# Bypass proxy
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
            "ALL_PROXY", "all_proxy"):
    os.environ.pop(key, None)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
OUTPUT_PATH = DATA_DIR / "st_bak_basic.parquet"


# Map ST_CLIENT bak_basic raw columns → our st_ prefixed schema, mirroring
# the convention in st_daily_basic.parquet so downstream FeatureMerger
# code can recognise them.
COLUMN_MAP = {
    "ts_code": "ts_code",
    "trade_date": "trade_date",
    "name": "st_name",
    "industry": "st_industry",
    "area": "st_area",
    "pe": "st_pe",
    "pb": "st_pb",
    "float_share": "st_float_share",
    "total_share": "st_total_share",
    "total_assets": "st_total_assets",
    "liquid_assets": "st_liquid_assets",
    "fixed_assets": "st_fixed_assets",
    "reserved": "st_reserved",
    "reserved_pershare": "st_reserved_pershare",
    "eps": "st_eps",
    "bvps": "st_bvps",
    "list_date": "st_list_date",
    "undp": "st_undp",
    "per_undp": "st_per_undp",
    "rev_yoy": "st_rev_yoy",
    "profit_yoy": "st_profit_yoy",
    "gpr": "st_gpr",
    "npr": "st_npr",
    "holder_num": "st_holder_num",
}


def get_st_client():
    token_file = PROJECT_ROOT / ".st_token"
    token = token_file.read_text().strip() if token_file.exists() else None
    if not token:
        try:
            from config.settings import ST_TOKEN
            token = ST_TOKEN
        except Exception:
            pass
    if not token:
        raise RuntimeError("ST_TOKEN unavailable (neither .st_token nor settings)")
    from ST_CLIENT import StockToday
    return StockToday(token=token)


def fetch_one_day(st, trade_date: str) -> pd.DataFrame:
    """Pull bak_basic for one trading date. trade_date format: YYYYMMDD."""
    try:
        r = st.bak_basic(trade_date=trade_date)
    except Exception as e:
        logger.warning(f"bak_basic({trade_date}) raise: {e}")
        return pd.DataFrame()
    if not isinstance(r, dict):
        logger.warning(f"bak_basic({trade_date}) returned non-dict {type(r)}")
        return pd.DataFrame()
    inner = r.get("data") or {}
    items = (inner.get("items") if isinstance(inner, dict) else inner) or []
    if not items:
        msg = r.get("msg") or ""
        logger.info(f"bak_basic({trade_date}) empty (msg={msg[:60]})")
        return pd.DataFrame()
    return pd.DataFrame(items)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to st_-prefixed schema; add qlib_code; coerce numerics."""
    if df.empty:
        return df
    # Subset + rename
    keep = {raw: new for raw, new in COLUMN_MAP.items() if raw in df.columns}
    df = df[list(keep.keys())].rename(columns=keep)

    # qlib_code from ts_code (000001.SZ → SZ000001)
    if "ts_code" in df.columns:
        def _qlib_from_ts(ts: str) -> str | None:
            if not isinstance(ts, str) or "." not in ts:
                return None
            num, suf = ts.split(".", 1)
            return f"{suf}{num}"
        df["qlib_code"] = df["ts_code"].map(_qlib_from_ts)

    # Coerce numerics — every st_ field except identifier ones
    NUMERIC_FIELDS = {
        c for c in df.columns
        if c.startswith("st_") and c not in {"st_name", "st_industry", "st_area", "st_list_date"}
    }
    for c in NUMERIC_FIELDS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    # derived: liquid_ratio = float / total (replaces baostock's field)
    if "st_float_share" in df.columns and "st_total_share" in df.columns:
        df["st_liquid_ratio"] = (
            df["st_float_share"] / df["st_total_share"]
        ).replace([np.inf, -np.inf], np.nan)

    return df


def merge_and_save(new_df: pd.DataFrame, path: Path) -> None:
    if new_df.empty:
        return
    if path.exists():
        try:
            old_df = pd.read_parquet(path)
            combined = pd.concat([old_df, new_df], ignore_index=True)
            # Dedup on (qlib_code, trade_date) keeping latest
            dedup_cols = [c for c in ["qlib_code", "trade_date"] if c in combined.columns]
            if dedup_cols:
                combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
            combined.to_parquet(path, index=False)
        except Exception as e:
            logger.error(f"merge into {path} failed: {e}; writing new only")
            new_df.to_parquet(path, index=False)
    else:
        new_df.to_parquet(path, index=False)


def _date_range(start: str, end: str) -> list[str]:
    """All weekdays between start and end (inclusive), format YYYYMMDD."""
    d = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    out = []
    while d <= e:
        if d.weekday() < 5:
            out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None,
                        help="Backfill start YYYY-MM-DD (default: today)")
    parser.add_argument("--end", default=None,
                        help="Backfill end YYYY-MM-DD (default: today)")
    parser.add_argument("--rate-sleep", type=float, default=1.0,
                        help="Seconds between calls (default 1s, ~60 RPM)")
    args = parser.parse_args()

    today_str = datetime.now().strftime("%Y-%m-%d")
    start = args.start or today_str
    end = args.end or today_str
    if start > end:
        logger.error("start > end")
        return 1

    dates = _date_range(start, end)
    if not dates:
        logger.info(f"no weekdays in [{start}, {end}] — exiting")
        return 0
    logger.info(f"Will fetch bak_basic for {len(dates)} weekday(s) {dates[0]}..{dates[-1]}")

    st = get_st_client()
    all_rows = []
    ok = 0
    fail = 0
    for i, td in enumerate(dates):
        df = fetch_one_day(st, td)
        if df.empty:
            fail += 1
            time.sleep(args.rate_sleep)
            continue
        df = normalize_columns(df)
        all_rows.append(df)
        ok += 1
        if (i + 1) % 50 == 0:
            logger.info(f"  progress {i+1}/{len(dates)}  ok={ok} fail={fail}")
        time.sleep(args.rate_sleep)

    if not all_rows:
        logger.error("no data fetched")
        from scheduler.data_health import HealthStatus, write_health
        write_health("st_bak_basic_update", HealthStatus(
            success=False,
            error_type="NoData",
            error_message=f"0 rows over {len(dates)} dates",
            network_profile="domestic",
        ))
        return 1

    combined = pd.concat(all_rows, ignore_index=True)
    logger.info(f"Fetched {len(combined):,} rows over {ok} dates ({fail} empty)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merge_and_save(combined, OUTPUT_PATH)
    saved = pd.read_parquet(OUTPUT_PATH)
    logger.info(
        f"Saved {OUTPUT_PATH.name}: {len(saved):,} rows total "
        f"(uniq qlib_code={saved['qlib_code'].nunique() if 'qlib_code' in saved.columns else '?'}, "
        f"uniq trade_date={saved['trade_date'].nunique() if 'trade_date' in saved.columns else '?'})"
    )

    # Health
    try:
        from scheduler.data_health import HealthStatus, write_health
        latest = saved["trade_date"].max() if "trade_date" in saved.columns else ""
        # latest -> YYYY-MM-DD form for SLA gate
        latest_human = ""
        if isinstance(latest, str) and len(latest) == 8 and latest.isdigit():
            latest_human = f"{latest[:4]}-{latest[4:6]}-{latest[6:]}"
        write_health("st_bak_basic_update", HealthStatus(
            success=True,
            n_items=len(combined),
            latest_date=latest_human,
            network_profile="domestic",
        ))
    except Exception as e:
        logger.warning(f"write_health failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
