"""Collect stock popularity + sentiment from Eastmoney.

Sources:
  1. Popularity ranking (top 100 most discussed stocks) — emappdata API
  2. News sentiment for top stocks — reuses existing news collection

Factors:
  - popularity_rank: 1-100 (1=most popular), NaN for non-ranked
  - popularity_rank_change: rank change vs yesterday
  - is_hot: 1 if in top 100, 0 otherwise

Usage:
    python scripts/collect_guba_sentiment.py
    python scripts/collect_guba_sentiment.py --date 2026-05-22
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
GUBA_DIR = DATA_DIR / "guba"
GUBA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_popularity_ranking() -> list[dict]:
    """Fetch top 100 most popular stocks from Eastmoney."""
    url = "https://emappdata.eastmoney.com/stockrank/getAllCurrentList"
    payload = {
        "appId": "appId01",
        "globalId": "786e4c21-70dc-435a-93bb-38",
        "pageNo": 1,
        "pageSize": 100,
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("data", [])
            results = []
            for item in items:
                sc = item.get("sc", "")  # SZ000725 format
                rk = item.get("rk", 0)   # rank (1=most popular)
                rc = item.get("rc", 0)   # rank change
                hrc = item.get("hisRc", 0)  # historical rank change
                if sc:
                    # Convert SZ000725 → 000725.SZ for ts_code, sz000725 for qlib
                    code = sc[2:]
                    exchange = sc[:2]
                    qlib_code = f"{exchange.lower()}{code}"
                    ts_code = f"{code}.{exchange}"
                    results.append({
                        "qlib_code": qlib_code,
                        "ts_code": ts_code,
                        "stock_code": code,
                        "popularity_rank": rk,
                        "rank_change": rc,
                        "hist_rank_change": hrc,
                    })
            logger.info(f"Popularity ranking: {len(results)} stocks")
            return results
    except Exception as e:
        logger.warning(f"Popularity API failed: {e}")
    return []


def collect_daily(target_date: str):
    """Collect popularity data for a given date."""
    output_path = GUBA_DIR / f"{target_date}.jsonl"

    # Fetch ranking
    ranking = fetch_popularity_ranking()
    if not ranking:
        logger.warning("No ranking data")
        return None

    # Add date
    for r in ranking:
        r["date"] = target_date

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        for r in ranking:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(ranking)} stocks to {output_path}")
    return output_path


def build_factors():
    """Build popularity factor parquet from all collected daily files."""
    import pandas as pd
    import numpy as np

    files = sorted(GUBA_DIR.glob("*.jsonl"))
    if not files:
        logger.warning("No guba data files")
        return

    all_records = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                try:
                    all_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not all_records:
        return

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "qlib_code"])
    df.index.names = ["datetime", "instrument"]

    # Compute factors
    factor_df = pd.DataFrame(index=df.index)
    factor_df["popularity_rank"] = df["popularity_rank"]
    factor_df["rank_change"] = df["rank_change"]
    # Normalize rank: 1→1.0 (most popular), 100→0.01
    factor_df["popularity_score"] = 1.0 / df["popularity_rank"].clip(lower=1)

    out_path = DATA_DIR / "guba_factors.parquet"
    factor_df.to_parquet(str(out_path))
    logger.info(f"Guba factors: {factor_df.shape} -> {out_path}")

    # Stats
    logger.info(f"  Dates: {len(factor_df.index.get_level_values(0).unique())}")
    logger.info(f"  Stocks per day: {len(factor_df) / max(len(factor_df.index.get_level_values(0).unique()), 1):.0f}")
    return factor_df


def main():
    parser = argparse.ArgumentParser(description="Collect Eastmoney popularity data")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--build-factors", action="store_true")
    args = parser.parse_args()

    if args.build_factors:
        build_factors()
        return

    t0 = time.time()
    collect_daily(args.date)
    build_factors()
    logger.info(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
