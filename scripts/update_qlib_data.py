"""Update Qlib data with latest A-share data from baostock.

Runs nightly to keep training data up-to-date.
Converts baostock daily data to Qlib binary format.

Usage: python scripts/update_qlib_data.py
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = Path(__file__).parent.parent / "data" / "storage"
QLIB_DIR = DATA_DIR / "qlib_data" / "cn_data"


def get_all_stock_codes():
    """Get all A-share stock codes from baostock."""
    import baostock as bs
    bs.login()
    rs = bs.query_stock_basic()
    codes = []
    while rs.error_code == "0" and rs.next():
        row = rs.get_row_data()
        code = row[0]  # e.g. "sh.600519"
        status = row[4] if len(row) > 4 else "1"
        if status == "1":  # Active stocks only
            codes.append(code)
    bs.logout()
    return codes


def fetch_stock_data(bs_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV for a single stock from baostock."""
    import baostock as bs

    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,open,high,low,close,volume,amount,turn,peTTM,pbMRQ",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",  # 前复权
    )

    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close",
                                      "volume", "amount", "turn", "peTTM", "pbMRQ"])
    for col in ["open", "high", "low", "close", "volume", "amount", "turn", "peTTM", "pbMRQ"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.dropna(subset=["close"])

    # Rename to Qlib convention
    df = df.rename(columns={
        "open": "$open",
        "high": "$high",
        "low": "$low",
        "close": "$close",
        "volume": "$volume",
        "amount": "$amount",
        "turn": "$turn",
        "peTTM": "$pe",
        "pbMRQ": "$pb",
    })
    return df


def save_to_qlib_format(instrument: str, df: pd.DataFrame, qlib_dir: Path):
    """Save stock data in Qlib binary format (.bin)."""
    if df.empty:
        return

    # Qlib uses SH600519 format
    bs_prefix, code = instrument.split(".")
    qlib_code = f"{bs_prefix.upper()}{code}"

    inst_dir = qlib_dir / "features" / qlib_code
    inst_dir.mkdir(parents=True, exist_ok=True)

    for col in df.columns:
        feature_name = col.replace("$", "")
        file_path = inst_dir / f"{feature_name}.day.bin"

        values = df[col].values.astype(np.float32)
        values.tofile(str(file_path))

    # Save calendar entry
    calendar_file = qlib_dir / "calendars" / "day.txt"
    calendar_file.parent.mkdir(parents=True, exist_ok=True)

    existing_dates = set()
    if calendar_file.exists():
        existing_dates = set(calendar_file.read_text().strip().split("\n"))

    new_dates = set(df.index.strftime("%Y-%m-%d").tolist())
    all_dates = sorted(existing_dates | new_dates)

    with open(calendar_file, "w") as f:
        f.write("\n".join(all_dates) + "\n")


def main():
    # Determine date range: last 5 years to today
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")

    logger.info(f"Updating Qlib data: {start_date} to {end_date}")

    import baostock as bs
    bs.login()

    # Get CSI300 + CSI500 constituent codes
    rs300 = bs.query_hs300_stocks()
    rs500 = bs.query_zz500_stocks()

    codes = set()
    for rs in [rs300, rs500]:
        while rs.error_code == "0" and rs.next():
            codes.add(rs.get_row_data()[1])  # code column

    logger.info(f"Found {len(codes)} stocks (CSI300 + CSI500)")

    success = 0
    fail = 0
    for i, code in enumerate(sorted(codes)):
        try:
            df = fetch_stock_data(code, start_date, end_date)
            if not df.empty:
                save_to_qlib_format(code, df, QLIB_DIR)
                success += 1
            else:
                fail += 1

            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i+1}/{len(codes)} ({success} ok, {fail} failed)")

        except Exception as e:
            fail += 1
            if (i + 1) % 50 == 0:
                logger.warning(f"Failed {code}: {e}")

    bs.logout()

    logger.info(f"Update complete: {success} stocks updated, {fail} failed")
    logger.info(f"Data stored in {QLIB_DIR}")


if __name__ == "__main__":
    main()
