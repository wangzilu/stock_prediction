"""Collect stock announcements from Eastmoney 公告 API.

Higher coverage than news search — every listed company has periodic announcements.
Free, no auth needed.

Usage:
    python scripts/collect_announcements.py                  # today
    python scripts/collect_announcements.py --date 2026-05-22
    python scripts/collect_announcements.py --days 5          # backfill 5 days
"""
import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
ANN_DIR = DATA_DIR / "announcements"
ANN_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}

# Eastmoney announcement API
ANN_API = "https://np-anotice-stock.eastmoney.com/api/security/ann"


def fetch_announcements_for_date(date: str, page_size: int = 100, max_pages: int = 20) -> list[dict]:
    """Fetch all announcements published on a specific date."""
    all_items = []

    for page in range(1, max_pages + 1):
        try:
            params = {
                "sr": -1,
                "page_size": page_size,
                "page_index": page,
                "ann_type": "A",
                "stock_list": "",
                "f_node": "0",
                "s_node": "0",
                "begin_time": date.replace("-", ""),
                "end_time": date.replace("-", ""),
            }
            resp = requests.get(ANN_API, params=params, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                break

            data = resp.json()
            items = data.get("data", {}).get("list", [])
            if not items:
                break

            for item in items:
                codes = item.get("codes", [])
                title = item.get("title", "")
                notice_date = item.get("notice_date", "")
                ann_id = item.get("art_code", "")

                # Each announcement can relate to multiple stocks
                for code_info in codes:
                    stock_code = code_info.get("stock_code", "")
                    short_name = code_info.get("short_name", "")

                    if not stock_code or len(stock_code) != 6:
                        continue

                    all_items.append({
                        "stock_code": stock_code,
                        "stock_name": short_name,
                        "title": title,
                        "notice_date": notice_date,
                        "ann_id": ann_id,
                        "source": "eastmoney_announcement",
                        "publish_time": date,
                    })

            # Check if there are more pages
            total = data.get("data", {}).get("total_hits", 0)
            if page * page_size >= total:
                break

            time.sleep(0.3)  # rate limit

        except Exception as e:
            logger.warning(f"  Page {page} failed: {e}")
            break

    return all_items


def collect_for_date(date: str) -> Path:
    """Collect announcements for a date and save to JSONL."""
    output_path = ANN_DIR / f"{date}.jsonl"

    if output_path.exists():
        n = sum(1 for _ in open(output_path))
        if n >= 50:
            logger.info(f"  {date}: already {n} announcements, skip")
            return output_path

    items = fetch_announcements_for_date(date)

    # Convert stock codes to qlib format
    for item in items:
        code = item["stock_code"]
        if code.startswith("6"):
            item["qlib_code"] = f"sh{code}"
        elif code.startswith(("0", "3")):
            item["qlib_code"] = f"sz{code}"
        else:
            item["qlib_code"] = f"bj{code}"

    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"  {date}: {len(items)} announcements, {len(set(i['stock_code'] for i in items))} stocks")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Collect Eastmoney announcements")
    parser.add_argument("--date", default=None)
    parser.add_argument("--days", type=int, default=1, help="Backfill N trading days")
    args = parser.parse_args()

    if args.date:
        dates = [args.date]
    else:
        dates = []
        d = datetime.now()
        n = 0
        while n < args.days:
            if d.weekday() < 5:
                dates.append(d.strftime("%Y-%m-%d"))
                n += 1
            d -= timedelta(days=1)
        dates.reverse()

    logger.info(f"Collecting announcements for {len(dates)} dates...")
    total = 0
    for date in dates:
        path = collect_for_date(date)
        if path.exists():
            total += sum(1 for _ in open(path))

    logger.info(f"Total: {total} announcements across {len(dates)} dates")


if __name__ == "__main__":
    main()
