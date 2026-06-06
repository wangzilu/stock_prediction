"""Fetch fundamental quality factors (ROE/margins/growth) via baostock.

Quarterly data, forward-filled to daily in FeatureMerger.

Saves to: data/storage/fundamental_quality.parquet

Usage:
    python scripts/fetch_fundamental_quality.py
    python scripts/fetch_fundamental_quality.py --top 500
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
OUTPUT_PATH = DATA_DIR / "fundamental_quality.parquet"


def get_all_stock_codes(top_n: int = None) -> list:
    features_dir = DATA_DIR / "qlib_data" / "cn_data" / "features"
    codes = sorted([d.name.upper() for d in features_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(codes)} stocks in Qlib features")
    if top_n and top_n < len(codes):
        codes = codes[:top_n]
    return codes


def qlib_to_baostock(code: str) -> str:
    num = code.replace("SH", "").replace("SZ", "").replace("BJ", "")
    if code.startswith("SH"):
        return f"sh.{num}"
    elif code.startswith("SZ"):
        return f"sz.{num}"
    elif code.startswith("BJ"):
        return f"bj.{num}"
    if num.startswith("6"):
        return f"sh.{num}"
    return f"sz.{num}"


def fetch_quality(codes: list) -> pd.DataFrame:
    """Fetch ROE/margins/growth for all codes from baostock."""
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        logger.error(f"baostock login failed: {lg.error_msg}")
        return pd.DataFrame()

    all_rows = []
    success = 0
    fail = 0
    # Fetch last 4 quarters
    from datetime import datetime
    now = datetime.now()
    quarters = []
    y, q = now.year, (now.month - 1) // 3
    for _ in range(4):
        if q == 0:
            y -= 1
            q = 4
        quarters.append((y, q))
        q -= 1

    logger.info(f"Fetching quality for {len(codes)} stocks, "
                f"quarters: {quarters}...")

    for i, code in enumerate(codes):
        bs_code = qlib_to_baostock(code)
        try:
            best_row = None
            best_date = ""
            for year, quarter in quarters:
                # Profit data: ROE, margins
                rs = bs.query_profit_data(code=bs_code, year=year, quarter=quarter)
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if not rows:
                    continue

                r = rows[0]
                stat_date = r[2]  # statDate
                if stat_date <= best_date:
                    continue
                best_date = stat_date

                best_row = {
                    "qlib_code": code,
                    "stat_date": stat_date,
                    "roe": r[3],          # roeAvg
                    "net_margin": r[4],    # npMargin
                    "gross_margin": r[5],  # gpMargin
                    "net_profit": r[6],    # netProfit
                    "eps_ttm": r[7],       # epsTTM
                }

                # DuPont data
                rs2 = bs.query_dupont_data(code=bs_code, year=year, quarter=quarter)
                d_rows = []
                while rs2.next():
                    d_rows.append(rs2.get_row_data())
                if d_rows:
                    d = d_rows[0]
                    best_row["dupont_roe"] = d[3]
                    best_row["asset_turnover"] = d[5]     # dupontAssetTurn
                    best_row["equity_multiplier"] = d[4]  # dupontAssetStoEquity
                    best_row["tax_burden"] = d[8]         # dupontTaxBurden

                # Growth data
                rs3 = bs.query_growth_data(code=bs_code, year=year, quarter=quarter)
                g_rows = []
                while rs3.next():
                    g_rows.append(rs3.get_row_data())
                if g_rows:
                    g = g_rows[0]
                    best_row["yoy_net_profit"] = g[3]   # YOYEquity
                    best_row["yoy_revenue"] = g[5]       # YOYNI (or similar)

                break  # Got latest quarter, no need to go further

            if best_row:
                all_rows.append(best_row)
                success += 1
            else:
                fail += 1

        except Exception:
            fail += 1

        if (i + 1) % 200 == 0 or (i + 1) == len(codes):
            logger.info(f"  Quality: {i+1}/{len(codes)} ({success} ok, {fail} fail)")

    bs.logout()

    if not all_rows:
        logger.error("No quality data fetched!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # Convert numeric
    num_cols = [c for c in df.columns if c not in ("qlib_code", "stat_date")]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace inf
    df = df.replace([np.inf, -np.inf], np.nan)

    logger.info(f"Quality: {len(df)} stocks, {len(num_cols)} factors")
    return df


def main():
    from scheduler.data_health import HealthStatus, write_health

    parser = argparse.ArgumentParser(description="Fetch fundamental quality factors")
    parser.add_argument("--top", type=int, default=None)
    args = parser.parse_args()

    try:
        codes = get_all_stock_codes(top_n=args.top)
        df = fetch_quality(codes)

        if not df.empty:
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(OUTPUT_PATH, index=False)
            n_codes = int(df["qlib_code"].nunique()) if "qlib_code" in df.columns else 0
            logger.info(f"Saved to {OUTPUT_PATH}")
            logger.info(f"  Stocks: {n_codes}")
            write_health("quality_update", HealthStatus(
                success=True,
                n_items=len(df),
                latest_date=str(df["stat_date"].max()) if "stat_date" in df.columns else "",
                coverage=n_codes / max(len(codes), 1),
                network_profile="domestic",
            ))
        else:
            write_health("quality_update", HealthStatus(
                success=False,
                error_type="NoData",
                error_message="No quality data fetched",
                network_profile="domestic",
            ))
            raise RuntimeError("No quality data fetched")

        logger.info("Done!")
    except Exception as exc:
        write_health("quality_update", HealthStatus(
            success=False,
            error_type=type(exc).__name__,
            error_message=str(exc)[:200],
            network_profile="domestic",
        ))
        raise


if __name__ == "__main__":
    main()
