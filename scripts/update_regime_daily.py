"""Daily regime data refresh — pull day-frequency risk indicators.

Pulls data that changes daily (not monthly like PMI/CPI):
  - Margin detail (融资融券余额) — leverage signal
  - Limit-down list (跌停统计) — microcap crash signal
  - HSGT moneyflow (北向资金) — foreign flow signal

Monthly data (PMI/CPI/M2/Shibor) stays on weekly Saturday refresh.

Usage:
    python scripts/update_regime_daily.py
"""
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


def get_st_client():
    token = Path(PROJECT_ROOT / ".st_token").read_text().strip()
    from ST_CLIENT import StockToday
    return StockToday(token=token)


def update_margin(st, date: str):
    """Append today's margin data to existing parquet."""
    logger.info("  Updating margin_detail...")
    try:
        result = st.margin_detail(trade_date=date.replace("-", ""))
        if isinstance(result, dict):
            data = result.get("data")
            if data and isinstance(data, list):
                new_df = pd.DataFrame(data)
                for col in new_df.columns:
                    if new_df[col].dtype == object:
                        new_df[col] = new_df[col].astype(str)

                path = DATA_DIR / "st_margin_detail.parquet"
                if path.exists():
                    old_df = pd.read_parquet(path)
                    # Dedup by trade_date + ts_code
                    combined = pd.concat([old_df, new_df], ignore_index=True)
                    if "trade_date" in combined.columns and "ts_code" in combined.columns:
                        combined = combined.drop_duplicates(
                            subset=["trade_date", "ts_code"], keep="last"
                        )
                    combined.to_parquet(str(path), index=False)
                else:
                    new_df.to_parquet(str(path), index=False)

                logger.info(f"    ✅ margin: +{len(new_df)} records for {date}")
                return
        logger.info(f"    margin: no data for {date}")
    except Exception as e:
        logger.warning(f"    margin: {e}")


def update_limit_list(st, date: str):
    """Append today's limit-down data."""
    logger.info("  Updating limit_list_d...")
    try:
        result = st.limit_list_d(trade_date=date.replace("-", ""))
        if isinstance(result, dict):
            data = result.get("data")
            if data and isinstance(data, list):
                new_df = pd.DataFrame(data)
                for col in new_df.columns:
                    if new_df[col].dtype == object:
                        new_df[col] = new_df[col].astype(str)

                path = DATA_DIR / "st_limit_list_d.parquet"
                if path.exists():
                    old_df = pd.read_parquet(path)
                    combined = pd.concat([old_df, new_df], ignore_index=True)
                    if "trade_date" in combined.columns and "ts_code" in combined.columns:
                        combined = combined.drop_duplicates(
                            subset=["trade_date", "ts_code"], keep="last"
                        )
                    combined.to_parquet(str(path), index=False)
                else:
                    new_df.to_parquet(str(path), index=False)

                logger.info(f"    ✅ limit_list: +{len(new_df)} records for {date}")
                return
        logger.info(f"    limit_list: no data for {date}")
    except Exception as e:
        logger.warning(f"    limit_list: {e}")


def update_hsgt(st, date: str):
    """Append today's northbound flow data."""
    logger.info("  Updating moneyflow_hsgt...")
    try:
        result = st.moneyflow_hsgt(trade_date=date.replace("-", ""))
        if isinstance(result, dict):
            data = result.get("data")
            if data and isinstance(data, list):
                new_df = pd.DataFrame(data)
                for col in new_df.columns:
                    if new_df[col].dtype == object:
                        new_df[col] = new_df[col].astype(str)

                path = DATA_DIR / "st_moneyflow_hsgt.parquet"
                if path.exists():
                    old_df = pd.read_parquet(path)
                    combined = pd.concat([old_df, new_df], ignore_index=True)
                    if "trade_date" in combined.columns:
                        combined = combined.drop_duplicates(
                            subset=["trade_date"], keep="last"
                        )
                    combined.to_parquet(str(path), index=False)
                else:
                    new_df.to_parquet(str(path), index=False)

                logger.info(f"    ✅ hsgt: +{len(new_df)} records for {date}")
                return
        logger.info(f"    hsgt: no data for {date}")
    except Exception as e:
        logger.warning(f"    hsgt: {e}")


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"=== Daily Regime Data Update: {today} ===")

    try:
        st = get_st_client()
    except Exception as e:
        logger.error(f"ST_CLIENT init failed: {e}")
        return

    update_margin(st, today)
    update_limit_list(st, today)
    update_hsgt(st, today)

    logger.info("Done!")


if __name__ == "__main__":
    main()
