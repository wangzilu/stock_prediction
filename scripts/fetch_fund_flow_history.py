"""Fetch fund flow + northbound history for all A-shares.

Saves to:
  data/storage/fund_flow_history.parquet    (~120 days per stock)
  data/storage/northbound_history.parquet   (~1600 days per stock)

Usage:
    python scripts/fetch_fund_flow_history.py                # all A-shares
    python scripts/fetch_fund_flow_history.py --top 500      # top 500 by market cap
    python scripts/fetch_fund_flow_history.py --flow-only    # skip northbound
    python scripts/fetch_fund_flow_history.py --nb-only      # skip fund flow
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
FLOW_PATH = DATA_DIR / "fund_flow_history.parquet"
NB_PATH = DATA_DIR / "northbound_history.parquet"


def get_all_stock_codes(top_n: int = None) -> list:
    """Get stock codes from Qlib features directory."""
    features_dir = DATA_DIR / "qlib_data" / "cn_data" / "features"
    codes = sorted([d.name.upper() for d in features_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(codes)} stocks in Qlib features")
    if top_n and top_n < len(codes):
        codes = codes[:top_n]
        logger.info(f"Using top {top_n} stocks")
    return codes


def fetch_fund_flow(codes: list) -> pd.DataFrame:
    """Fetch main force fund flow history for all stocks."""
    import akshare as ak

    all_flows = []
    success = 0
    fail = 0

    logger.info(f"Fetching fund flow for {len(codes)} stocks...")
    for i, code in enumerate(codes):
        try:
            num = code.replace("SH", "").replace("SZ", "").replace("BJ", "")
            market = "sh" if code.startswith("SH") or num.startswith("6") else "sz"

            df = ak.stock_individual_fund_flow(stock=num, market=market)
            if df is not None and not df.empty:
                df["qlib_code"] = code
                df["code"] = num
                all_flows.append(df)
                success += 1

            if (i + 1) % 50 == 0:
                logger.info(f"  Fund flow: {i+1}/{len(codes)} ({success} ok, {fail} fail)")

            time.sleep(0.3)
        except Exception as e:
            fail += 1
            if fail <= 5:
                logger.warning(f"  Failed {code}: {e}")

    if not all_flows:
        logger.error("No fund flow data fetched!")
        return pd.DataFrame()

    result = pd.concat(all_flows, ignore_index=True)
    logger.info(f"Fund flow: {len(result)} records for {success} stocks, {fail} failed")
    return result


def fetch_northbound(codes: list) -> pd.DataFrame:
    """Fetch northbound (陆股通) holding history for all stocks."""
    import akshare as ak

    all_nb = []
    success = 0
    fail = 0

    logger.info(f"Fetching northbound history for {len(codes)} stocks...")
    for i, code in enumerate(codes):
        try:
            num = code.replace("SH", "").replace("SZ", "").replace("BJ", "")
            df = ak.stock_hsgt_individual_em(symbol=num)
            if df is not None and not df.empty:
                df["qlib_code"] = code
                df["code"] = num
                all_nb.append(df)
                success += 1

            if (i + 1) % 50 == 0:
                logger.info(f"  Northbound: {i+1}/{len(codes)} ({success} ok, {fail} fail)")

            time.sleep(0.5)
        except Exception:
            fail += 1

    if not all_nb:
        logger.error("No northbound data fetched!")
        return pd.DataFrame()

    result = pd.concat(all_nb, ignore_index=True)
    logger.info(f"Northbound: {len(result)} records for {success} stocks, {fail} failed")
    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch fund flow + northbound history")
    parser.add_argument("--top", type=int, default=None, help="Only fetch top N stocks")
    parser.add_argument("--flow-only", action="store_true", help="Skip northbound")
    parser.add_argument("--nb-only", action="store_true", help="Skip fund flow")
    args = parser.parse_args()

    codes = get_all_stock_codes(top_n=args.top)

    if not args.nb_only:
        flow_df = fetch_fund_flow(codes)
        if not flow_df.empty:
            FLOW_PATH.parent.mkdir(parents=True, exist_ok=True)
            flow_df.to_parquet(FLOW_PATH, index=False)
            logger.info(f"Saved fund flow to {FLOW_PATH}")
            logger.info(f"  Date range: {flow_df['日期'].min()} ~ {flow_df['日期'].max()}")

    if not args.flow_only:
        nb_df = fetch_northbound(codes)
        if not nb_df.empty:
            NB_PATH.parent.mkdir(parents=True, exist_ok=True)
            nb_df.to_parquet(NB_PATH, index=False)
            logger.info(f"Saved northbound to {NB_PATH}")
            if "持股日期" in nb_df.columns:
                logger.info(f"  Date range: {nb_df['持股日期'].min()} ~ {nb_df['持股日期'].max()}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
