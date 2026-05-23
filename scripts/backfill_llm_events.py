"""Backfill historical LLM events — collect + extract for past N days.

Reuses existing pipeline but runs for multiple dates.
Skips dates that already have sufficient data.

Usage:
    python scripts/backfill_llm_events.py --days 30
    python scripts/backfill_llm_events.py --days 30 --top-n 1000  # fewer stocks for speed
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"


def get_trade_days(n_days: int) -> list[str]:
    """Get last N calendar days (filter weekends)."""
    days = []
    d = datetime.now()
    while len(days) < n_days:
        d -= timedelta(days=1)
        if d.weekday() < 5:  # Mon-Fri
            days.append(d.strftime("%Y-%m-%d"))
    return list(reversed(days))


def main():
    parser = argparse.ArgumentParser(description="Backfill historical LLM events")
    parser.add_argument("--days", type=int, default=30, help="Number of trading days to backfill")
    parser.add_argument("--top-n", type=int, default=1000, help="Stocks per day")
    parser.add_argument("--max-news-per-stock", type=int, default=1)
    args = parser.parse_args()

    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from scripts.collect_daily_news import collect_news_for_stock, get_liquid_stocks
    from factors.llm_event_extractor import LLMEventExtractor
    from scripts.build_llm_event_factors import build_factors

    dates = get_trade_days(args.days)
    logger.info(f"Backfilling {len(dates)} trading days: {dates[0]} ~ {dates[-1]}")

    NEWS_DIR = DATA_DIR / "daily_news"
    EVENTS_DIR = DATA_DIR / "llm_events"
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect news for all stocks (one-time, max 10 per stock)
    # News contains publish_time — we'll sort into date buckets
    stocks = get_liquid_stocks(args.top_n)
    if not stocks:
        logger.error("No stocks")
        return
    logger.info(f"Step 1: Collecting news for {len(stocks)} stocks (10 per stock)...")

    all_news = []
    def _fetch(stock):
        items = collect_news_for_stock(stock["code"], stock["name"], max_items=10)
        for item in items:
            item["qlib_code"] = stock.get("qlib_code", "")
        time.sleep(0.1)
        return items

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch, s): s for s in stocks}
        done = 0
        for f in as_completed(futures):
            done += 1
            items = f.result()
            if items:
                all_news.extend(items)
            if done % 200 == 0:
                logger.info(f"  {done}/{len(stocks)} stocks, {len(all_news)} news")

    logger.info(f"  Total collected: {len(all_news)} news items")

    # Step 2: Sort news into date buckets by publish_time
    from collections import defaultdict
    date_buckets = defaultdict(list)
    date_set = set(dates)

    for item in all_news:
        pt = item.get("publish_time", "")
        if not pt:
            continue
        # Extract date from publish_time (various formats)
        news_date = pt[:10]  # "2026-05-20 12:34:56" → "2026-05-20"
        if news_date in date_set:
            date_buckets[news_date].append(item)

    logger.info(f"  Sorted into {len(date_buckets)} date buckets")
    for d in sorted(date_buckets.keys()):
        logger.info(f"    {d}: {len(date_buckets[d])} news")

    # Step 3: Write per-date news files
    for date, items in sorted(date_buckets.items()):
        path = NEWS_DIR / f"{date}.jsonl"
        if path.exists():
            existing = sum(1 for _ in open(path))
            if existing >= 100:
                continue  # already have enough
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                item["collect_date"] = date
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Step 4: LLM extraction for each date
    logger.info(f"\nStep 4: LLM extraction...")
    extractor = LLMEventExtractor()
    total_events = 0

    for date in sorted(date_buckets.keys()):
        news_path = NEWS_DIR / f"{date}.jsonl"
        events_path = EVENTS_DIR / f"{date}.jsonl"

        # Skip if already have enough events
        if events_path.exists():
            n = sum(1 for _ in open(events_path))
            if n >= 50:
                logger.info(f"  {date}: already {n} events, skip")
                total_events += n
                continue

        n_news = sum(1 for _ in open(news_path))
        if n_news < 5:
            continue

        logger.info(f"  {date}: extracting from {n_news} news...")
        try:
            extractor.extract_from_news_file(
                news_path=news_path,
                max_news_per_stock=args.max_news_per_stock,
                target_date=date,
            )
            n_events = sum(1 for _ in open(events_path)) if events_path.exists() else 0
            total_events += n_events
            logger.info(f"    → {n_events} events")
        except Exception as e:
            logger.error(f"    Failed: {e}")

    # Step 5: Build factors
    logger.info(f"\nStep 5: Building factors...")
    try:
        for date in sorted(date_buckets.keys())[-10:]:
            build_factors(date)
    except Exception as e:
        logger.error(f"Factor build failed: {e}")

    logger.info(f"\n=== BACKFILL COMPLETE ===")
    logger.info(f"  Days: {len(dates)}")
    logger.info(f"  Total news: {total_news:,}")
    logger.info(f"  Total events: {total_events:,}")

    # Show coverage
    news_dir = DATA_DIR / "daily_news"
    events_dir = DATA_DIR / "llm_events"
    n_news_files = len(list(news_dir.glob("*.jsonl")))
    n_event_files = len(list(events_dir.glob("*.jsonl")))
    logger.info(f"  News files: {n_news_files}")
    logger.info(f"  Event files: {n_event_files}")


if __name__ == "__main__":
    main()
