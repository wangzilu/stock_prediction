"""Fetch high-value data from JQData (聚宽) — 3-month trial, prioritize!

Data available: 2025-02-12 ~ 2026-02-19 (~1 year)
Query limit: 1,000,000 queries

Priority data to pull:
  1. Alpha101 (101 factors × all A-shares × daily)
  2. Alpha191 (191 factors × all A-shares × daily)
  3. Barra factor values (9 style factors × all A-shares × daily)
  4. Industry classification (申万三级 × all A-shares)
  5. Concept mapping (概念板块 × all A-shares)
  6. Valuation (PE/PB/市值 × all A-shares × daily)

Usage:
    python scripts/fetch_jqdata.py --list
    python scripts/fetch_jqdata.py --pull alpha101
    python scripts/fetch_jqdata.py --pull alpha191
    python scripts/fetch_jqdata.py --pull barra
    python scripts/fetch_jqdata.py --pull all
"""
import argparse
import gc
import json
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
JQ_DIR = DATA_DIR / "jqdata"
JQ_DIR.mkdir(parents=True, exist_ok=True)

# Account
JQ_USER = "18910565831"
JQ_PASS = "W-z-l19830218"

# Date range (account limit)
DATE_START = "2025-02-12"
DATE_END = "2026-02-19"

BARRA_FACTORS = [
    "size", "beta", "momentum", "residual_volatility", "liquidity",
    "book_to_price_ratio", "earnings_yield", "growth", "leverage",
]


def jq_login():
    import jqdatasdk as jq
    jq.auth(JQ_USER, JQ_PASS)
    return jq


def get_trade_days(jq):
    days = jq.get_trade_days(start_date=DATE_START, end_date=DATE_END)
    return [d.strftime("%Y-%m-%d") for d in days]


def pull_alpha101(jq):
    """Pull Alpha101 for all trading days."""
    output = JQ_DIR / "alpha101.parquet"
    if output.exists():
        logger.info(f"  alpha101 already exists ({output}), skipping")
        return

    days = get_trade_days(jq)
    logger.info(f"  Pulling Alpha101 for {len(days)} days...")

    frames = []
    for i, day in enumerate(days):
        try:
            df = jq.get_all_alpha_101(date=day)
            if not df.empty:
                df = df.copy(); df["date"] = day
                frames.append(df)
        except Exception as e:
            logger.warning(f"  Day {day}: {e}")

        if (i + 1) % 20 == 0:
            count = jq.get_query_count()
            logger.info(f"  Progress: {i+1}/{len(days)} days, {len(frames)} with data, "
                        f"queries remaining: {count['spare']}")

        # Save checkpoint every 50 days
        if (i + 1) % 50 == 0 and frames:
            tmp = pd.concat(frames)
            tmp.to_parquet(str(output.with_suffix(".tmp.parquet")))

    if frames:
        result = pd.concat(frames)
        result.to_parquet(str(output))
        logger.info(f"  ✅ Alpha101: {result.shape} -> {output}")
    else:
        logger.warning("  No Alpha101 data")


def pull_alpha191(jq):
    """Pull Alpha191 for all trading days."""
    output = JQ_DIR / "alpha191.parquet"
    if output.exists():
        logger.info(f"  alpha191 already exists, skipping")
        return

    days = get_trade_days(jq)
    logger.info(f"  Pulling Alpha191 for {len(days)} days...")

    frames = []
    for i, day in enumerate(days):
        try:
            df = jq.get_all_alpha_191(date=day)
            if not df.empty:
                df = df.copy(); df["date"] = day
                frames.append(df)
        except Exception as e:
            logger.warning(f"  Day {day}: {e}")

        if (i + 1) % 20 == 0:
            count = jq.get_query_count()
            logger.info(f"  Progress: {i+1}/{len(days)} days, "
                        f"queries remaining: {count['spare']}")

        if (i + 1) % 50 == 0 and frames:
            tmp = pd.concat(frames)
            tmp.to_parquet(str(output.with_suffix(".tmp.parquet")))

    if frames:
        result = pd.concat(frames)
        result.to_parquet(str(output))
        logger.info(f"  ✅ Alpha191: {result.shape} -> {output}")


def pull_barra(jq):
    """Pull Barra style factor values for all stocks × all days."""
    output = JQ_DIR / "barra_factors.parquet"
    if output.exists():
        logger.info(f"  barra already exists, skipping")
        return

    days = get_trade_days(jq)
    # Get all stock codes
    stocks = jq.get_all_securities(types=["stock"], date=days[-1])
    stock_list = list(stocks.index)
    logger.info(f"  Pulling Barra for {len(stock_list)} stocks × {len(days)} days...")

    # Pull in daily batches (all stocks per day)
    frames = []
    for i, day in enumerate(days):
        try:
            fv = jq.get_factor_values(
                securities=stock_list,
                factors=BARRA_FACTORS,
                start_date=day, end_date=day,
            )
            if fv:
                # fv is dict of {factor_name: DataFrame(date × stock)}
                day_df = pd.DataFrame(index=stock_list)
                for fname, fdf in fv.items():
                    if not fdf.empty:
                        day_df[fname] = fdf.iloc[0]  # single date row
                day_df["date"] = day
                day_df = day_df.dropna(how="all", subset=BARRA_FACTORS)
                if not day_df.empty:
                    frames.append(day_df)
        except Exception as e:
            logger.warning(f"  Day {day}: {e}")

        if (i + 1) % 20 == 0:
            count = jq.get_query_count()
            logger.info(f"  Progress: {i+1}/{len(days)} days, "
                        f"queries remaining: {count['spare']}")

    if frames:
        result = pd.concat(frames)
        result.to_parquet(str(output))
        logger.info(f"  ✅ Barra: {result.shape} -> {output}")


def pull_industry(jq):
    """Pull 申万三级行业分类 for all stocks."""
    output = JQ_DIR / "industry_sw.parquet"
    if output.exists():
        logger.info(f"  industry already exists, skipping")
        return

    stocks = jq.get_all_securities(types=["stock"])
    stock_list = list(stocks.index)
    logger.info(f"  Pulling industry for {len(stock_list)} stocks...")

    records = []
    for code in stock_list:
        try:
            ind = jq.get_industry(code, date=DATE_END)
            if ind and code in ind:
                info = ind[code]
                records.append({
                    "code": code,
                    "sw_l1_code": info.get("sw_l1", {}).get("industry_code", ""),
                    "sw_l1_name": info.get("sw_l1", {}).get("industry_name", ""),
                    "sw_l2_code": info.get("sw_l2", {}).get("industry_code", ""),
                    "sw_l2_name": info.get("sw_l2", {}).get("industry_name", ""),
                    "sw_l3_code": info.get("sw_l3", {}).get("industry_code", ""),
                    "sw_l3_name": info.get("sw_l3", {}).get("industry_name", ""),
                    "zjw_code": info.get("zjw", {}).get("industry_code", ""),
                    "zjw_name": info.get("zjw", {}).get("industry_name", ""),
                })
        except Exception:
            pass

    if records:
        df = pd.DataFrame(records)
        df.to_parquet(str(output), index=False)
        logger.info(f"  ✅ Industry: {df.shape} -> {output}")


def pull_concepts(jq):
    """Pull concept/theme mapping for all stocks."""
    output = JQ_DIR / "concepts.parquet"
    if output.exists():
        logger.info(f"  concepts already exists, skipping")
        return

    stocks = jq.get_all_securities(types=["stock"])
    stock_list = list(stocks.index)
    logger.info(f"  Pulling concepts for {len(stock_list)} stocks...")

    records = []
    for i, code in enumerate(stock_list):
        try:
            concept = jq.get_concept(code, date=DATE_END)
            if concept and code in concept:
                concepts = concept[code].get("jq_concept", [])
                for c in concepts:
                    records.append({
                        "code": code,
                        "concept_code": c.get("concept_code", ""),
                        "concept_name": c.get("concept_name", ""),
                    })
        except Exception:
            pass
        if (i + 1) % 500 == 0:
            logger.info(f"  Progress: {i+1}/{len(stock_list)}")

    if records:
        df = pd.DataFrame(records)
        df.to_parquet(str(output), index=False)
        logger.info(f"  ✅ Concepts: {df.shape} -> {output}")


def pull_valuation(jq):
    """Pull daily valuation for all stocks."""
    output = JQ_DIR / "valuation.parquet"
    if output.exists():
        logger.info(f"  valuation already exists, skipping")
        return

    days = get_trade_days(jq)
    logger.info(f"  Pulling valuation for {len(days)} days...")

    frames = []
    for i, day in enumerate(days):
        try:
            q = jq.query(jq.valuation)
            df = jq.get_fundamentals(q, date=day)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            logger.warning(f"  Day {day}: {e}")

        if (i + 1) % 20 == 0:
            count = jq.get_query_count()
            logger.info(f"  Progress: {i+1}/{len(days)} days, "
                        f"queries remaining: {count['spare']}")

    if frames:
        result = pd.concat(frames, ignore_index=True)
        result.to_parquet(str(output))
        logger.info(f"  ✅ Valuation: {result.shape} -> {output}")


def pull_baidu_factor(jq):
    """Pull Baidu search index factor — alternative attention data."""
    output = JQ_DIR / "baidu_factor.parquet"
    if output.exists():
        logger.info(f"  baidu_factor already exists, skipping")
        return

    days = get_trade_days(jq)
    # Sample stocks (top liquid)
    stocks = jq.get_all_securities(types=["stock"], date=days[-1])
    stock_list = list(stocks.index[:500])  # top 500 for speed
    logger.info(f"  Pulling Baidu factor for {len(stock_list)} stocks...")

    frames = []
    for i, code in enumerate(stock_list):
        try:
            df = jq.get_baidu_factor(stock=code, day=days[-1], duration="30")
            if df is not None and not df.empty:
                df["code"] = code
                frames.append(df)
        except Exception as e:
            pass

        if (i + 1) % 100 == 0:
            count = jq.get_query_count()
            logger.info(f"  Progress: {i+1}/{len(stock_list)}, "
                        f"queries remaining: {count['spare']}")

    if frames:
        result = pd.concat(frames, ignore_index=True)
        result.to_parquet(str(output))
        logger.info(f"  ✅ Baidu factor: {result.shape} -> {output}")
    else:
        logger.warning("  No Baidu factor data")


SOURCES = {
    "alpha101": ("Alpha101 因子 (101个)", pull_alpha101),
    "alpha191": ("Alpha191 因子 (191个)", pull_alpha191),
    "barra": ("Barra 风格因子 (9个)", pull_barra),
    "industry": ("申万行业分类", pull_industry),
    "concepts": ("概念板块映射", pull_concepts),
    "valuation": ("每日估值数据", pull_valuation),
    "baidu": ("百度搜索指数 (另类数据)", pull_baidu_factor),
}


def main():
    parser = argparse.ArgumentParser(description="Fetch JQData")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--pull", nargs="*", help="Sources to pull (or 'all')")
    args = parser.parse_args()

    if args.list:
        print(f"\nJQData sources (date range: {DATE_START} ~ {DATE_END}):\n")
        for name, (desc, _) in SOURCES.items():
            exists = (JQ_DIR / f"{name}.parquet").exists()
            status = "✅" if exists else "❌"
            print(f"  {status} {name:<15} {desc}")
        return

    targets = list(SOURCES.keys()) if args.pull and "all" in args.pull else (args.pull or [])
    if not targets:
        parser.print_help()
        return

    jq = jq_login()
    count = jq.get_query_count()
    logger.info(f"Logged in. Queries remaining: {count['spare']}/{count['total']}")

    for name in targets:
        if name not in SOURCES:
            logger.warning(f"Unknown source: {name}")
            continue
        desc, fn = SOURCES[name]
        logger.info(f"\n=== Pulling {name}: {desc} ===")
        t0 = time.time()
        fn(jq)
        logger.info(f"  Time: {time.time()-t0:.0f}s")

    count = jq.get_query_count()
    logger.info(f"\nDone. Queries remaining: {count['spare']}/{count['total']}")
    jq.logout()


if __name__ == "__main__":
    main()
