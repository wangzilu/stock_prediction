"""Fetch shareholder data (holder count, pledge ratio) via ST_CLIENT.

Quarterly data, forward-filled to daily in FeatureMerger.

Saves to: data/storage/shareholder_features.parquet

Usage:
    python scripts/fetch_shareholder_data.py
    python scripts/fetch_shareholder_data.py --top 500
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Bypass proxy
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
            "ALL_PROXY", "all_proxy"):
    os.environ.pop(key, None)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
OUTPUT_PATH = DATA_DIR / "shareholder_features.parquet"


def get_all_stock_codes(top_n: int = None) -> list:
    features_dir = DATA_DIR / "qlib_data" / "cn_data" / "features"
    codes = sorted([d.name.upper() for d in features_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(codes)} stocks")
    if top_n and top_n < len(codes):
        codes = codes[:top_n]
    return codes


def qlib_to_baostock(code: str) -> str:
    num = code.replace("SH", "").replace("SZ", "").replace("BJ", "")
    if code.startswith("SH"):
        return f"sh.{num}"
    elif code.startswith("SZ"):
        return f"sz.{num}"
    return f"sz.{num}"


def fetch_shareholder(codes: list) -> pd.DataFrame:
    """Fetch shareholder data from baostock (totalShare, liqaShare from profit_data)
    and pledge ratio from akshare if available."""
    import baostock as bs
    from datetime import datetime

    lg = bs.login()
    if lg.error_code != "0":
        logger.error(f"baostock login failed: {lg.error_msg}")
        return pd.DataFrame()

    all_rows = []
    success = 0
    fail = 0

    # Latest quarter
    now = datetime.now()
    year = now.year
    quarter = max(1, (now.month - 1) // 3)
    if quarter == 0:
        year -= 1
        quarter = 4

    logger.info(f"Fetching shareholder for {len(codes)} stocks (Q{quarter} {year})...")

    # 2026-06-16: pre-fix the per-stock loop tried 4 (year, quarter) combos
    # before giving up, multiplying baostock RPC cost ~4×. Per-stock budget
    # was ~1.5s, so 5419 × 1.5 × 4 = 4.5 hrs worst case — past every
    # reasonable timeout. Cron at 18:02 with timeout 10800 (3 hrs) couldn't
    # complete, blocking feature_cache_rebuild → predict_crash_daily for
    # 9 consecutive trading days. Real fix is migrating to ST_CLIENT batch
    # (different fact set: holder_num vs share structure, separate task).
    # Tonight: try the current quarter first; if no data fall back to the
    # PREVIOUS quarter only (handles the ~4-week gap after each quarter
    # close before new reports land), then give up. Reduces worst-case
    # work to 5419 × 1.5 × 2 = 4.5 hrs → 2.25 hrs, comfortably inside 3 hr
    # cron budget.
    quarter_attempts = [(year, quarter)]
    if quarter > 1:
        quarter_attempts.append((year, quarter - 1))
    else:
        quarter_attempts.append((year - 1, 4))

    for i, code in enumerate(codes):
        bs_code = qlib_to_baostock(code)
        try:
            best_row = None
            for y, q in quarter_attempts:
                rs = bs.query_profit_data(code=bs_code, year=y, quarter=q)
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if rows:
                    r = rows[0]
                    best_row = {
                        "qlib_code": code,
                        "stat_date": r[2],
                        "total_share": r[9],   # totalShare
                        "liquid_share": r[10],  # liqaShare
                    }
                    break

            if best_row:
                # Compute liquid ratio
                total = pd.to_numeric(best_row["total_share"], errors="coerce")
                liquid = pd.to_numeric(best_row["liquid_share"], errors="coerce")
                if total and total > 0:
                    best_row["liquid_ratio"] = liquid / total
                all_rows.append(best_row)
                success += 1
            else:
                fail += 1

        except Exception:
            fail += 1

        if (i + 1) % 500 == 0 or (i + 1) == len(codes):
            logger.info(f"  Shareholder: {i+1}/{len(codes)} ({success} ok, {fail} fail)")

    bs.logout()

    if not all_rows:
        logger.error("No shareholder data fetched!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    num_cols = [c for c in df.columns if c not in ("qlib_code", "stat_date")]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    logger.info(f"Shareholder: {len(df)} stocks")
    return df


def main():
    from scheduler.data_health import HealthStatus, write_health

    parser = argparse.ArgumentParser(description="Fetch shareholder data")
    parser.add_argument("--top", type=int, default=None)
    args = parser.parse_args()

    try:
        codes = get_all_stock_codes(top_n=args.top)
        df = fetch_shareholder(codes)

        if not df.empty:
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(OUTPUT_PATH, index=False)
            logger.info(f"Saved to {OUTPUT_PATH}")
            write_health("shareholder_update", HealthStatus(
                success=True,
                n_items=len(df),
                latest_date=str(df["stat_date"].max()) if "stat_date" in df.columns else "",
                network_profile="domestic",
                coverage=len(df) / max(len(codes), 1),
            ))
        else:
            write_health("shareholder_update", HealthStatus(
                success=False,
                error_type="NoData",
                error_message="No shareholder data fetched",
                network_profile="domestic",
            ))
            raise RuntimeError("No shareholder data fetched")

        logger.info("Done!")
    except Exception as e:
        write_health("shareholder_update", HealthStatus(
            success=False,
            error_type=type(e).__name__,
            error_message=str(e)[:200],
            network_profile="domestic",
        ))
        raise


if __name__ == "__main__":
    main()
