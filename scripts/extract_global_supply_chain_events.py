"""Extract structured events from global industry news using rule-based extractor.

Reads: data/storage/global_industry_news/YYYY-MM-DD.jsonl
Writes: data/storage/global_chain_events/YYYY-MM-DD.jsonl

Usage:
    python scripts/extract_global_supply_chain_events.py [--date YYYY-MM-DD]
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
NEWS_DIR = DATA_DIR / "global_industry_news"
EVENTS_DIR = DATA_DIR / "global_chain_events"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()
    date = args.date

    news_path = NEWS_DIR / f"{date}.jsonl"
    if not news_path.exists():
        logger.info(f"No global news for {date}, skipping")
        return

    # Load news
    news_items = []
    with open(news_path) as f:
        for line in f:
            line = line.strip()
            if line:
                news_items.append(json.loads(line))

    logger.info(f"Loaded {len(news_items)} news items for {date}")

    if not news_items:
        return

    # Extract events
    from factors.global_supply_chain_extractor import batch_extract
    events = batch_extract(news_items)
    logger.info(f"Extracted {len(events)} structured events")

    # Save
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVENTS_DIR / f"{date}.jsonl"
    with open(out_path, "w") as f:
        for e in events:
            e["date"] = date
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    logger.info(f"Saved to {out_path}")

    # Write health
    try:
        from scheduler.data_health import write_health, HealthStatus
        write_health("global_chain_extract", HealthStatus(
            success=True, n_items=len(events), latest_date=date,
            network_profile="none",
        ))
    except Exception:
        pass


if __name__ == "__main__":
    main()
