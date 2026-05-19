"""Collect daily stock news from ST_CLIENT (primary) or AKShare (fallback).

Usage:
    python scripts/collect_daily_news.py [--date 2024-01-15] [--portfolio]

By default collects news for the top 100 most liquid A-share stocks.
With --portfolio, collects only for stocks in the current Top20 portfolio.

Output: data/storage/daily_news/YYYY-MM-DD.jsonl
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

NEWS_DIR = DATA_DIR / "daily_news"
NEWS_DIR.mkdir(parents=True, exist_ok=True)


def get_liquid_stocks(top_n: int = 100) -> list[dict]:
    """Get top N most liquid A-share stocks.

    Tries ST_CLIENT (reliable) first, AKShare as fallback.
    """
    # Try ST_CLIENT first
    try:
        from ST_CLIENT import StockToday
        token_file = PROJECT_ROOT / ".st_token"
        token = token_file.read_text().strip() if token_file.exists() else ""
        if token:
            st = StockToday(token=token)
            date_str = datetime.now().strftime("%Y%m%d")
            result = st.bak_basic(trade_date=date_str)
            if isinstance(result, list) and result:
                df = pd.DataFrame(result)
                df = df[~df["name"].str.contains("ST|退", na=False)]
                df = df.head(top_n)
                results = []
                for _, row in df.iterrows():
                    ts = str(row["ts_code"])
                    code = ts[:6]
                    prefix = "SH" if ts.endswith(".SH") else "SZ"
                    results.append({
                        "code": code,
                        "name": row["name"],
                        "qlib_code": f"{prefix}{code}",
                        "ts_code": ts,
                    })
                logger.info(f"Got {len(results)} stocks from ST_CLIENT")
                return results
    except Exception as e:
        logger.warning(f"ST_CLIENT failed: {e}")

    # Fallback to AKShare
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        df = df[~df["名称"].str.contains("ST|退", na=False)]
        df = df.sort_values("成交额", ascending=False).head(top_n)
        results = []
        for _, row in df.iterrows():
            code = str(row["代码"]).zfill(6)
            prefix = "SH" if code.startswith(("6", "9")) else "SZ" if code.startswith(("0", "3")) else "BJ"
            results.append({"code": code, "name": row["名称"], "qlib_code": f"{prefix}{code}"})
        logger.info(f"Got {len(results)} liquid stocks from AKShare")
        return results
    except Exception as e:
        logger.error(f"Failed to get liquid stocks: {e}")
        return []


def get_portfolio_stocks() -> list[dict]:
    """Get stocks from the current overnight snapshot (Top20 portfolio).

    Returns:
        List of dicts with keys: code, name, qlib_code
    """
    snapshot_path = DATA_DIR / "overnight_stock_forecasts.json"
    if not snapshot_path.exists():
        logger.warning("No overnight snapshot found, falling back to liquid stocks")
        return get_liquid_stocks()

    try:
        with open(snapshot_path) as f:
            data = json.load(f)

        results = []
        for item in data:
            qlib_code = item.get("code", "")
            if len(qlib_code) < 8:
                continue
            code = qlib_code[2:]
            results.append({
                "code": code,
                "name": item.get("name", ""),
                "qlib_code": qlib_code,
            })
        logger.info(f"Got {len(results)} portfolio stocks from snapshot")
        return results
    except Exception as e:
        logger.error(f"Failed to load portfolio snapshot: {e}")
        return get_liquid_stocks()


def collect_news_for_stock(code: str, name: str, max_items: int = 10) -> list[dict]:
    """Collect recent news for a single stock via AKShare.

    Args:
        code: 6-digit stock code, e.g. '600519'
        name: stock name for logging
        max_items: max news items to return

    Returns:
        List of news dicts with standardized fields
    """
    # Try ST_CLIENT news first (anns_d for announcements)
    try:
        from ST_CLIENT import StockToday
        token_file = PROJECT_ROOT / ".st_token"
        token = token_file.read_text().strip() if token_file.exists() else ""
        if token:
            st = StockToday(token=token)
            ts_code = f"{code}.SH" if code.startswith("6") else f"{code}.SZ"
            result = st.anns_d(ts_code=ts_code)
            if isinstance(result, list) and result:
                records = []
                for item in result[:max_items]:
                    records.append({
                        "stock_code": code,
                        "stock_name": name,
                        "title": str(item.get("title", "")),
                        "content_snippet": str(item.get("content", ""))[:500],
                        "source": "交易所公告",
                        "publish_time": str(item.get("ann_date", "")),
                        "url": str(item.get("url", "")),
                    })
                if records:
                    return records
    except Exception:
        pass

    # Fallback to AKShare
    try:
        import akshare as ak
        df = ak.stock_news_em(symbol=code)
        if df is None or df.empty:
            return []

        records = []
        for _, row in df.head(max_items).iterrows():
            records.append({
                "stock_code": code,
                "stock_name": name,
                "title": str(row.get("新闻标题", "")),
                "content_snippet": str(row.get("新闻内容", ""))[:500],
                "source": str(row.get("文章来源", "")),
                "publish_time": str(row.get("发布时间", "")),
                "url": str(row.get("新闻链接", "")),
            })
        return records

    except Exception as e:
        logger.warning(f"Failed to collect news for {code} ({name}): {e}")
        return []


def collect_daily_news(
    target_date: str = None,
    use_portfolio: bool = False,
    top_n: int = 100,
) -> Path:
    """Collect news for all target stocks and save as JSONL.

    Args:
        target_date: YYYY-MM-DD, defaults to today
        use_portfolio: if True, use portfolio stocks instead of liquid stocks
        top_n: number of liquid stocks to use (ignored if use_portfolio=True)

    Returns:
        Path to the saved JSONL file
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    output_path = NEWS_DIR / f"{target_date}.jsonl"

    # Skip if already collected today
    if output_path.exists():
        n_existing = sum(1 for _ in open(output_path))
        logger.info(f"News already collected for {target_date} ({n_existing} items), skipping")
        return output_path

    # Get target stocks
    if use_portfolio:
        stocks = get_portfolio_stocks()
    else:
        stocks = get_liquid_stocks(top_n)

    if not stocks:
        logger.error("No stocks to collect news for")
        return output_path

    logger.info(f"Collecting news for {len(stocks)} stocks on {target_date}")

    total_news = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, stock in enumerate(stocks):
            news_items = collect_news_for_stock(stock["code"], stock["name"])

            for item in news_items:
                item["qlib_code"] = stock["qlib_code"]
                item["collect_date"] = target_date
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_news += 1

            # Rate limit: AKShare may throttle
            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i+1}/{len(stocks)} stocks, {total_news} news items")
                time.sleep(1.0)
            else:
                time.sleep(0.3)

    logger.info(f"Collected {total_news} news items for {len(stocks)} stocks -> {output_path}")
    return output_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Collect daily stock news from AKShare")
    parser.add_argument("--date", type=str, default=None, help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--portfolio", action="store_true", help="Use portfolio stocks instead of top liquid")
    parser.add_argument("--top-n", type=int, default=100, help="Number of liquid stocks (default: 100)")
    args = parser.parse_args()

    collect_daily_news(
        target_date=args.date,
        use_portfolio=args.portfolio,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
