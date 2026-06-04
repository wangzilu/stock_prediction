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
    manifest_path = ANN_DIR / f"{date}.manifest.json"

    # 2026-06-04 cx round 16 P2-6: the "≥50 lines → skip" rule
    # accepted half-failed collections (full A-share day has thousands
    # of announcements). Use a sibling manifest with
    # ``finished=True`` + a sanity floor instead.
    if output_path.exists() and manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text())
            if m.get("finished") and int(m.get("n_items", 0)) >= 50:
                n = int(m["n_items"])
                logger.info(
                    f"  {date}: finished collection ({n} announcements), "
                    f"skip"
                )
                return output_path
        except Exception:
            pass
        # Stale/invalid manifest — fall through to re-collect.
        logger.warning(
            f"  {date}: manifest missing/invalid, re-collecting "
            f"(was len={sum(1 for _ in open(output_path))})"
        )

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

    # 2026-06-04 cx round 17 P1-1: write to .tmp then atomic-replace
    # so a half-failed write cannot overwrite a previously-good file.
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp_path, output_path)

    # cx round 17 P1-1: WRITE the manifest sidecar after a successful
    # save. Pre-fix only the SKIP path read the manifest; nothing wrote
    # it, so every cron re-collected and an Eastmoney blip could
    # overwrite a complete file with a half-empty one.
    manifest = {
        "target_date": date,
        "finished": True,
        "n_items": len(items),
        "n_unique_stocks": len(set(i.get("stock_code", "") for i in items)),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    os.replace(manifest_tmp, manifest_path)

    logger.info(f"  {date}: {len(items)} announcements, {len(set(i['stock_code'] for i in items))} stocks (manifest finished=True)")
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
