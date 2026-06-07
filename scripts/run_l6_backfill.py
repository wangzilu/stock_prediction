"""L6 — historical LLM event backfill (task #136).

Re-extracts events from cached daily_news/ files for dates whose
``llm_events_v2/<date>.jsonl`` is anomalously thin or zero, and
writes them into the unified EventStore so the L6 ablation can
re-measure xgb_209_llm vs xgb_209 with denser coverage.

Why not just call ``run_llm_event_pipeline.py --date <past>``?
That script's Step 1 (collect_daily_news) only returns news within
``_NEWS_RECENCY_DAYS=7`` of today and would refuse to backfill
historical dates — the news source is not retroactively queryable.
For dates older than ~7 days we work with what's already cached in
``data/storage/daily_news/`` and just re-run the LLM extractor on it.

Scope decision (see ``docs/llm_l6_backfill_plan_20260607.md``):
the project lead's task spec asked for 60-90 trading days, but the
news cache only covers 2026-04-27 → 2026-06-05 (~30 trading days)
and the daily pipeline filter caps L1 candidates at 500/day. We
backfill what is recoverable, then honestly report the realistic
coverage uplift to the operator.

Usage:
    python scripts/run_l6_backfill.py [--dry-run]
    python scripts/run_l6_backfill.py --dates 2026-05-28,2026-05-29
    python scripts/run_l6_backfill.py --min-yield 0.4   # re-extract anywhere yield < 40%

The script is idempotent: dates whose v2 jsonl already passes the
``--min-events`` threshold are skipped. Dates passed via --dates are
always re-extracted (force) regardless of current count.

Sleep between dates: ``--sleep-secs`` (default 30s) keeps us off
the MiniMax RPM=1000 cap when re-running multiple dates back to back.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config.settings import DATA_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("l6_backfill")

NEWS_DIR = DATA_DIR / "daily_news"
NEWS_FILTERED_DIR = DATA_DIR / "daily_news_filtered"
EVENTS_V2_DIR = DATA_DIR / "llm_events_v2"
UNIFIED_DIR = DATA_DIR / "events"
COMPLETION_MARKER = DATA_DIR / "llm_l6_backfill_done.json"


def discover_thin_dates(min_events: int = 500) -> list[tuple[str, int, int]]:
    """Find dates in news cache whose v2 extraction yield is thin.

    Returns list of (date, n_news, n_events) sorted ascending by date,
    filtered to those with ``n_events < min_events`` AND ``n_news >= 50``
    (skip true-holiday days where news source itself had nothing).
    """
    if not NEWS_DIR.exists():
        return []
    thin = []
    for news_path in sorted(NEWS_DIR.glob("*.jsonl")):
        date = news_path.stem
        try:
            with open(news_path) as f:
                n_news = sum(1 for _ in f)
        except OSError:
            continue
        if n_news < 50:
            continue
        ev_path = EVENTS_V2_DIR / f"{date}.jsonl"
        if ev_path.exists():
            try:
                with open(ev_path) as f:
                    n_events = sum(1 for _ in f)
            except OSError:
                n_events = 0
        else:
            n_events = 0
        if n_events < min_events:
            thin.append((date, n_news, n_events))
    return thin


def write_to_unified_store(events_path: Path, target_date: str) -> int:
    """Mirror of run_llm_event_pipeline._write_to_unified_store.

    Reads the legacy v2 jsonl and pushes records through EventStore
    so signal_date/execution_date routing is consistent. Returns the
    number of events written (0 on any failure — non-fatal).
    """
    try:
        from factors.event_store import EventStore, _convert_legacy_event
    except Exception as e:  # noqa: BLE001
        logger.warning("EventStore import failed: %s", e)
        return 0
    if not events_path.exists():
        logger.warning("Events file missing for unified-store push: %s", events_path)
        return 0
    records: list[dict] = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return 0
    try:
        converted = [_convert_legacy_event(r, target_date) for r in records]
        store = EventStore()
        return store.add_events(converted)
    except Exception as e:  # noqa: BLE001
        logger.warning("Unified store write failed for %s: %s", target_date, e)
        return 0


def reextract_one_date(date: str, *, force: bool, min_events: int,
                         rpm: int = 30) -> dict:
    """Re-extract events for a single date from cached news.

    Returns a stats dict with keys: date, n_news, n_events_before,
    n_events_after, n_unified, action, elapsed_s.
    """
    t0 = time.time()
    news_path = NEWS_DIR / f"{date}.jsonl"
    ev_path = EVENTS_V2_DIR / f"{date}.jsonl"

    if not news_path.exists():
        return {"date": date, "action": "skip-no-news",
                "elapsed_s": 0.0, "n_events_after": 0}

    with open(news_path) as f:
        n_news = sum(1 for _ in f)

    n_before = 0
    if ev_path.exists():
        with open(ev_path) as f:
            n_before = sum(1 for _ in f)
        if not force and n_before >= min_events:
            return {"date": date, "n_news": n_news, "n_events_before": n_before,
                    "n_events_after": n_before, "n_unified": 0,
                    "action": "skip-already-sat", "elapsed_s": time.time() - t0}

    # The extractor's extract_from_news_file has its own "skip if ≥500" gate.
    # We must remove an existing thin file so it actually re-runs.
    if ev_path.exists():
        try:
            os.remove(str(ev_path))
            logger.info("  removed thin v2 file (%d events) to force re-extract", n_before)
        except OSError as e:
            logger.warning("  failed to remove %s: %s", ev_path, e)
            return {"date": date, "n_news": n_news, "n_events_before": n_before,
                    "n_events_after": n_before, "n_unified": 0,
                    "action": "skip-rm-failed", "elapsed_s": time.time() - t0}

    # Run the V2 extractor on the cached news file
    try:
        from factors.llm_event_extractor_v2 import LLMEventExtractorV2
    except Exception as e:  # noqa: BLE001
        logger.error("Cannot import LLMEventExtractorV2: %s", e)
        return {"date": date, "n_news": n_news, "n_events_before": n_before,
                "n_events_after": 0, "n_unified": 0,
                "action": "fail-import", "elapsed_s": time.time() - t0}

    extractor = LLMEventExtractorV2(max_calls_per_minute=rpm)
    logger.info("  [%s] extracting from %d news (cached, rpm=%d)...", date, n_news, rpm)
    try:
        extractor.extract_from_news_file(
            news_path=news_path,
            max_news_per_stock=1,
            target_date=date,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("  extraction crashed for %s: %s", date, e)
        return {"date": date, "n_news": n_news, "n_events_before": n_before,
                "n_events_after": 0, "n_unified": 0,
                "action": "fail-extract", "elapsed_s": time.time() - t0}

    n_after = 0
    if ev_path.exists():
        with open(ev_path) as f:
            n_after = sum(1 for _ in f)

    # Push to unified EventStore (signal_date / execution_date routing)
    n_unified = write_to_unified_store(ev_path, date)

    elapsed = time.time() - t0
    logger.info(
        "  [%s] done: %d events extracted, %d pushed to unified store (%.0fs)",
        date, n_after, n_unified, elapsed,
    )
    return {"date": date, "n_news": n_news, "n_events_before": n_before,
            "n_events_after": n_after, "n_unified": n_unified,
            "action": "extracted", "elapsed_s": elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", default="",
                        help="Comma-separated explicit dates to re-extract (force).")
    parser.add_argument("--min-events", type=int, default=500,
                        help="Dates with v2 events below this are considered thin.")
    parser.add_argument("--sleep-secs", type=int, default=30,
                        help="Sleep between dates so MiniMax RPM doesn't get hammered.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan without re-extracting.")
    parser.add_argument("--max-dates", type=int, default=0,
                        help="Cap the number of dates processed (0=no cap).")
    parser.add_argument("--rpm", type=int, default=30,
                        help="MiniMax RPM cap (lower if account-shared 1000 cap is hit).")
    args = parser.parse_args()

    if args.dates:
        explicit = [d.strip() for d in args.dates.split(",") if d.strip()]
        targets = [(d, 0, 0) for d in explicit]
        force = True
        logger.info("Explicit dates supplied (force): %s", ", ".join(explicit))
    else:
        targets = discover_thin_dates(min_events=args.min_events)
        force = False
        logger.info("Auto-discovered %d thin dates (threshold=%d events)",
                    len(targets), args.min_events)

    if args.max_dates > 0:
        targets = targets[:args.max_dates]

    if not targets:
        logger.info("No thin dates to re-extract — nothing to do.")
        print("[L6 backfill done]")
        return 0

    print(f"\n=== L6 backfill plan ({len(targets)} dates) ===")
    print(f"{'date':<12} {'n_news':>8} {'n_events_before':>17}")
    for d, nn, ne in targets:
        print(f"{d:<12} {nn:>8} {ne:>17}")
    print()

    if args.dry_run:
        logger.info("Dry-run requested — exiting before extraction.")
        return 0

    started_at = datetime.now()
    results = []
    for i, (date, _, _) in enumerate(targets):
        logger.info(
            "=== [%d/%d] %s ===", i + 1, len(targets), date,
        )
        stats = reextract_one_date(
            date, force=force, min_events=args.min_events, rpm=args.rpm,
        )
        results.append(stats)
        if i < len(targets) - 1 and stats.get("action") == "extracted":
            time.sleep(args.sleep_secs)

    finished_at = datetime.now()
    summary = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "elapsed_s": (finished_at - started_at).total_seconds(),
        "results": results,
        "totals": {
            "dates_processed": len(results),
            "events_extracted": sum(
                r.get("n_events_after", 0) - r.get("n_events_before", 0)
                for r in results if r.get("action") == "extracted"
            ),
        },
    }
    COMPLETION_MARKER.parent.mkdir(parents=True, exist_ok=True)
    COMPLETION_MARKER.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n=== L6 backfill summary ===")
    print(f"started:  {summary['started_at']}")
    print(f"finished: {summary['finished_at']}")
    print(f"elapsed:  {summary['elapsed_s']:.0f}s")
    print(f"dates:    {summary['totals']['dates_processed']}")
    print(f"events Δ: {summary['totals']['events_extracted']:+d}")
    print(f"marker:   {COMPLETION_MARKER}")
    print("[L6 backfill done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
