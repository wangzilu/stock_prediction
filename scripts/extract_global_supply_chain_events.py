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


def _write_extract_health(
    *,
    success: bool,
    date: str,
    n_items: int = 0,
    error_type: str = "",
    error_message: str = "",
) -> None:
    try:
        from scheduler.data_health import HealthStatus, write_health
        write_health("global_chain_extract", HealthStatus(
            success=success,
            n_items=n_items,
            latest_date=date if success else "",
            error_type=error_type,
            error_message=error_message[:200],
            network_profile="none",
        ))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()
    date = args.date

    news_path = NEWS_DIR / f"{date}.jsonl"
    if not news_path.exists():
        logger.info(f"No global news for {date}, skipping")
        _write_extract_health(
            success=False,
            date=date,
            error_type="MissingInput",
            error_message=f"No global news file: {news_path}",
        )
        sys.exit(1)

    # Load news
    news_items = []
    with open(news_path) as f:
        for line in f:
            line = line.strip()
            if line:
                news_items.append(json.loads(line))

    logger.info(f"Loaded {len(news_items)} news items for {date}")

    if not news_items:
        _write_extract_health(
            success=False,
            date=date,
            error_type="EmptyInput",
            error_message=f"Global news file has zero items: {news_path}",
        )
        sys.exit(1)

    # Extract events
    from factors.global_supply_chain_extractor import batch_extract
    events = batch_extract(news_items)
    logger.info(f"Extracted {len(events)} structured events")
    if not events:
        _write_extract_health(
            success=False,
            date=date,
            error_type="NoEvents",
            error_message="No structured supply-chain events extracted",
        )
        sys.exit(1)

    # Save
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVENTS_DIR / f"{date}.jsonl"
    # 2026-06-06 PIT fix follow-up (cx review P1 #2): the previous
    # ``e["date"] = date`` line clobbered the real publication date
    # ``batch_extract`` had just preserved (from published_at /
    # collect_date). Now keep what the extractor put on the event:
    # ``date`` is the canonical publish date when known, with
    # ``published_at`` and ``collect_date`` recorded alongside. Only
    # backfill ``date`` from the cron-run date when the event has no
    # publish info AT ALL (= upstream gave nothing parseable).
    with open(out_path, "w") as f:
        for e in events:
            if not e.get("date"):
                e["date"] = date  # fallback for truly unknown publish dates
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    logger.info(f"Saved to {out_path}")

    _write_extract_health(success=True, n_items=len(events), date=date)


if __name__ == "__main__":
    main()
