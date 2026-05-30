"""Drain LLM 429 retry queue.

The V2 extractor writes items that exhausted their inline 429 backoff to
`data/storage/llm_retry_queue/{date}.jsonl`. This script reads today's
queue (or a specified date), re-runs each item, dedupes against the
already-written llm_events_v2/{date}.jsonl + against itself, appends the
unique recoveries, then mirrors the main pipeline's closeout: sync the
recoveries to the unified EventStore and rebuild the day's factor parquet
so downstream consumers actually see the recovered events.

Designed to run as a cron at 22:30. By then MiniMax RPM and account
limits have hours to relax — a better shot than the 17:30 inline retry.

Exit codes:
    0  queue empty OR fully drained
    1  fatal error during drain
    2  some items still failed (partial recovery) — surfaces as failed
       in the daily health dashboard so an operator can react

Usage:
    python scripts/drain_llm_retry_queue.py                # today
    python scripts/drain_llm_retry_queue.py --date 2026-05-29
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import traceback
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


def _dedup_key(item: dict) -> str:
    """Stable key for dedup: stock_code + title[:60] + publish_time[:16]."""
    code = str(item.get("stock_code") or item.get("qlib_code") or "")[-6:]
    title = (item.get("title") or "")[:60].strip()
    pub = (item.get("publish_time") or "")[:16]
    raw = f"{code}|{title}|{pub}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _existing_event_keys(events_path: Path) -> set[str]:
    """Build a key set from the already-written V2 jsonl for this date."""
    keys: set[str] = set()
    if not events_path.exists():
        return keys
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add(_dedup_key(rec))
    return keys


def _sync_to_eventstore(events_path: Path, target_date: str) -> None:
    """Mirror run_llm_event_pipeline._write_to_unified_store."""
    try:
        from factors.event_store import EventStore, _convert_legacy_event
    except Exception as e:
        logger.warning("EventStore unavailable (%s), skipping sync", e)
        return
    if not events_path.exists():
        return
    records: list[dict] = []
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return
    converted = [_convert_legacy_event(r, target_date) for r in records]
    store = EventStore()
    n_stored = store.add_events(converted)
    logger.info(
        "EventStore sync: %d/%d events written for %s",
        n_stored, len(converted), target_date,
    )


def _rebuild_factors(target_date: str) -> None:
    """Rebuild llm_event_factors.parquet for target_date (mirrors pipeline Step 3)."""
    try:
        from scripts.build_llm_event_factors import build_factors_range
    except Exception as e:
        logger.warning("Factor builder unavailable (%s), skipping rebuild", e)
        return
    df = build_factors_range(start_date=target_date, end_date=target_date, lookback_days=30)
    n_stocks = len(df) if df is not None and not df.empty else 0
    logger.info("Factors rebuilt for %s: %d stocks", target_date, n_stocks)


def drain(target_date: str) -> dict:
    """Process the retry queue for one date. Returns counters.

    Closeout (EventStore sync + factor rebuild) runs UNCONDITIONALLY when
    the queue file existed at entry, even if every item was a duplicate or
    every retry still failed. Rationale: a prior drain may have appended
    recoveries to V2 jsonl but crashed before sync — the items now show
    as duplicates and would be cleared without ever reaching EventStore /
    factor parquet. Both operations are idempotent (EventStore.add_events
    dedups by hash, build_factors rewrites the day's parquet), so the
    wasted-work cost is bounded and the safety gain is real.
    """
    queue_path = RETRY_QUEUE_DIR / f"{target_date}.jsonl"
    events_path = EVENTS_DIR / f"{target_date}.jsonl"

    raw_items: list[dict] = []
    if queue_path.exists():
        with open(queue_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not raw_items:
            queue_path.unlink(missing_ok=True)
            logger.info("Queue %s was empty, removed", queue_path)
    else:
        logger.info("No retry queue for %s — drain mode is closeout-only", target_date)

    n_dup = 0
    still_failed: list[dict] = []
    recovered: list[dict] = []

    if raw_items:
        # Dedup within the queue + against already-written events.
        existing_keys = _existing_event_keys(events_path)
        seen_in_queue: set[str] = set()
        items: list[dict] = []
        for it in raw_items:
            k = _dedup_key(it)
            if k in seen_in_queue or k in existing_keys:
                n_dup += 1
                continue
            seen_in_queue.add(k)
            items.append(it)
        if n_dup:
            logger.info("Dropped %d duplicate items (queue self-dup or already-extracted)", n_dup)

        if items:
            logger.info("Draining %d unique items from %s", len(items), queue_path.name)
            extractor = LLMEventExtractorV2()
            for item in items:
                code = item.get("stock_code", "")
                name = item.get("stock_name", "")
                title = item.get("title", "")
                content = item.get("content", "")
                source = item.get("source", "unknown")
                # Pass target_date="" so a 2nd failure doesn't double-enqueue.
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

            stats = dict(extractor._stats)
            logger.info(
                "Drain stats: calls=%d http_fail=%d (rate_limited=%d) parse_fail=%d duplicates=%d",
                stats.get("calls", 0), stats.get("http_fail", 0),
                stats.get("rate_limited", 0), stats.get("parse_fail", 0), n_dup,
            )

        # Rewrite queue with still-failed items only (or delete if empty)
        if still_failed:
            tmp = queue_path.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for item in still_failed:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            tmp.replace(queue_path)
            logger.warning("%d items still failed after retry, queue kept", len(still_failed))
        else:
            queue_path.unlink(missing_ok=True)
            logger.info("Queue fully drained")

    # Unconditional closeout: covers the half-completed scenario where a
    # prior run appended to V2 jsonl but crashed before EventStore/factor
    # sync. Both downstream ops are idempotent.
    closeout_ran = False
    if events_path.exists():
        _sync_to_eventstore(events_path, target_date)
        _rebuild_factors(target_date)
        closeout_ran = True
    else:
        logger.info("No V2 jsonl for %s — skipping closeout", target_date)

    return {
        "items": len(raw_items),
        "recovered": len(recovered),
        "still_failed": len(still_failed),
        "duplicates": n_dup,
        "closeout_ran": closeout_ran,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Drain LLM 429 retry queue")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    try:
        result = drain(target_date)
    except Exception:
        logger.error("Drain crashed:\n%s", traceback.format_exc())
        return 1
    logger.info("Done: %s", result)
    # Exit 2 (partial) when some items still failed; cron will mark the
    # job as failed, surfacing the partial recovery in daily health.
    if result.get("still_failed", 0) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
