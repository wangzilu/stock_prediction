"""Drain LLM 429 retry queue.

The V2 extractor writes items that exhausted their inline 429 backoff to
`data/storage/llm_retry_queue/{date}.jsonl`. This script reads today's
queue (or a specified date), re-runs each item through the V2 extractor,
appends successful extractions to the same llm_events_v2/{date}.jsonl
the main pipeline writes, and rewrites the queue to keep only items that
STILL failed.

Designed to run as a cron at 22:30, after the main pipeline's 16:30 run
and the 17:30 retry job — by 22:30 the MiniMax RPM bucket should be
fully reset and account-level rate limits relaxed.

Usage:
    python scripts/drain_llm_retry_queue.py                # today
    python scripts/drain_llm_retry_queue.py --date 2026-05-29
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from factors.llm_event_extractor_v2 import (
    EVENTS_DIR,
    LLMEventExtractorV2,
    RETRY_QUEUE_DIR,
    SOURCE_TIERS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def drain(target_date: str) -> dict:
    """Process the retry queue for one date. Returns counters."""
    queue_path = RETRY_QUEUE_DIR / f"{target_date}.jsonl"
    if not queue_path.exists():
        logger.info("No retry queue for %s — nothing to do", target_date)
        return {"items": 0, "recovered": 0, "still_failed": 0}

    items: list[dict] = []
    with open(queue_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not items:
        logger.info("Queue %s is empty", queue_path)
        return {"items": 0, "recovered": 0, "still_failed": 0}

    logger.info("Draining %d items from %s", len(items), queue_path.name)
    extractor = LLMEventExtractorV2()

    events_path = EVENTS_DIR / f"{target_date}.jsonl"
    recovered: list[dict] = []
    still_failed: list[dict] = []

    for item in items:
        code = item.get("stock_code", "")
        name = item.get("stock_name", "")
        title = item.get("title", "")
        content = item.get("content", "")
        source = item.get("source", "unknown")
        # Re-attempt. Pass target_date="" so a second failure doesn't
        # double-enqueue — the still_failed list captures it for us.
        event = extractor.extract_single(
            code, name, title, content,
            source=source,
            publish_time=item.get("publish_time", ""),
            qlib_code=item.get("qlib_code", ""),
            target_date="",
        )
        if event:
            source_info = SOURCE_TIERS.get(source, {"tier": "media", "quality": 0.5})
            record = {
                "stock_code": code,
                "stock_name": name,
                "qlib_code": item.get("qlib_code", ""),
                "publish_time": item.get("publish_time", ""),
                "title": title,
                "source": source,
                "source_tier": source_info["tier"],
                "source_quality": source_info["quality"],
                "extract_date": target_date,
                "extractor_version": "v2_retry",
                **event,
            }
            recovered.append(record)
        else:
            still_failed.append(item)

    if recovered:
        with open(events_path, "a", encoding="utf-8") as f:
            for rec in recovered:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Appended %d recovered events to %s", len(recovered), events_path)

    # Rewrite queue with still-failed items only
    if still_failed:
        tmp = queue_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for item in still_failed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        tmp.replace(queue_path)
        logger.info("%d items still failed, queue rewritten", len(still_failed))
    else:
        queue_path.unlink(missing_ok=True)
        logger.info("Queue fully drained")

    stats = dict(extractor._stats)
    logger.info(
        "Drain stats: calls=%d http_fail=%d (rate_limited=%d) parse_fail=%d",
        stats.get("calls", 0),
        stats.get("http_fail", 0),
        stats.get("rate_limited", 0),
        stats.get("parse_fail", 0),
    )
    return {
        "items": len(items),
        "recovered": len(recovered),
        "still_failed": len(still_failed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Drain LLM 429 retry queue")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    result = drain(target_date)
    logger.info("Done: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
