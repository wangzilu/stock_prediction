"""Fetch daily PE/PB/PS valuation factors for all A-shares via baostock.

Baostock provides PE_TTM, PB_MRQ, PS_TTM as part of daily K-line data,
free with no rate limits. ~30 min for full A-share universe.

Saves to: data/storage/fundamental_valuation.parquet

Usage:
    python scripts/fetch_fundamental_valuation.py                # all stocks, last 120 days
    python scripts/fetch_fundamental_valuation.py --days 365     # last year
    python scripts/fetch_fundamental_valuation.py --top 500      # top 500 only
    python scripts/fetch_fundamental_valuation.py --incremental  # skip already-fetched
"""
import argparse
import logging
import os
import sys
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
OUTPUT_PATH = DATA_DIR / "fundamental_valuation.parquet"

FIELDS = "date,close,peTTM,pbMRQ,psTTM,pcfNcfTTM"
FIELD_NAMES = ["date", "close", "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ncf_ttm"]


def get_all_stock_codes(top_n: int = None) -> list:
    """Get stock codes from Qlib features directory."""
    features_dir = DATA_DIR / "qlib_data" / "cn_data" / "features"
    codes = sorted([d.name.upper() for d in features_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(codes)} stocks in Qlib features")
    if top_n and top_n < len(codes):
        codes = codes[:top_n]
        logger.info(f"Using top {top_n} stocks")
    return codes


def qlib_to_baostock(code: str) -> str:
    """Convert Qlib code (SH600519) to baostock code (sh.600519)."""
    num = code.replace("SH", "").replace("SZ", "").replace("BJ", "")
    if code.startswith("SH"):
        return f"sh.{num}"
    elif code.startswith("SZ"):
        return f"sz.{num}"
    elif code.startswith("BJ"):
        return f"bj.{num}"
    if num.startswith("6"):
        return f"sh.{num}"
    elif num.startswith(("0", "3")):
        return f"sz.{num}"
    return f"sz.{num}"


def load_existing_codes(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        df = pd.read_parquet(path, columns=["qlib_code"])
        codes = set(df["qlib_code"].unique())
        logger.info(f"Incremental: {len(codes)} stocks already in {path.name}")
        return codes
    except Exception:
        return set()


def fetch_valuation(codes: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch PE/PB/PS for all codes from baostock."""
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        logger.error(f"baostock login failed: {lg.error_msg}")
        return pd.DataFrame()

    all_rows = []
    success = 0
    fail = 0

    logger.info(f"Fetching valuation for {len(codes)} stocks "
                f"({start_date} ~ {end_date})...")

    for i, code in enumerate(codes):
        bs_code = qlib_to_baostock(code)
        try:
            rs = bs.query_history_k_data_plus(
                bs_code, FIELDS,
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="3",
            )
            if rs.error_code != "0":
                fail += 1
                continue

            rows = []
            while rs.next():
                rows.append(rs.get_row_data())

            if rows:
                for row in rows:
                    all_rows.append([code] + row)
                success += 1
            else:
                fail += 1

        except Exception:
            fail += 1

        if (i + 1) % 200 == 0 or (i + 1) == len(codes):
            logger.info(f"  Valuation: {i+1}/{len(codes)} "
                        f"({success} ok, {fail} fail)")

    bs.logout()

    if not all_rows:
        logger.error("No valuation data fetched!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["qlib_code"] + FIELD_NAMES)

    # Convert numeric columns
    for col in ["close", "pe_ttm", "pb_mrq", "ps_ttm", "pcf_ncf_ttm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute derived factors (guard against div-by-zero → NaN, not inf)
    df["ep"] = np.where(df["pe_ttm"].abs() > 0.01, 1.0 / df["pe_ttm"], np.nan)
    df["bp"] = np.where(df["pb_mrq"].abs() > 0.01, 1.0 / df["pb_mrq"], np.nan)
    df["sp"] = np.where(df["ps_ttm"].abs() > 0.01, 1.0 / df["ps_ttm"], np.nan)

    logger.info(f"Valuation: {len(df)} records for {success} stocks, {fail} failed")
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch fundamental valuation factors")
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--days", type=int, default=120, help="Days of history (default: 120)")
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()

    codes = get_all_stock_codes(top_n=args.top)

    if args.incremental:
        existing = load_existing_codes(OUTPUT_PATH)
        before = len(codes)
        codes = [c for c in codes if c not in existing]
        logger.info(f"Incremental: skipping {before - len(codes)}, {len(codes)} remaining")

    if not codes:
        logger.info("Nothing to fetch")
        return

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    df = fetch_valuation(codes, start_date, end_date)

    if not df.empty:
        if args.incremental and OUTPUT_PATH.exists():
            old = pd.read_parquet(OUTPUT_PATH)
            df = pd.concat([old, df], ignore_index=True)
            df.drop_duplicates(subset=["qlib_code", "date"], keep="last", inplace=True)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUTPUT_PATH, index=False)
        logger.info(f"Saved to {OUTPUT_PATH}")
        logger.info(f"  Date range: {df['date'].min()} ~ {df['date'].max()}")
        logger.info(f"  Stocks: {df['qlib_code'].nunique()}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
