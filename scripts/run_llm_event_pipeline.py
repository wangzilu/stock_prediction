"""Daily LLM Event Factor pipeline runner.

Orchestrates the full pipeline: collect news -> extract events -> build factors.
Designed for crontab execution before the evening outlook.

Usage:
    python -m scripts.run_llm_event_pipeline [--date 2024-01-15] [--portfolio]

Crontab example (run at 16:30 after market close):
    30 16 * * 1-5 cd /path/to/stockPrediction && python -m scripts.run_llm_event_pipeline
"""
import argparse
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger(__name__)


def _write_to_unified_store(events_path: Path, target_date: str) -> None:
    """Write extracted events to the unified EventStore (Phase 4T).

    This is the PRIMARY write path.  The legacy llm_events/ JSONL file is
    still produced by the extractor for backward compatibility but is
    deprecated and will be removed in a future release.

    Reads the legacy JSONL file produced by the extractor, converts each
    record to the unified schema (including the 5 explicit time fields),
    and calls EventStore.add_events().
    Errors are logged but never raised so the main pipeline is not affected.
    """
    try:
        import json as _json
        from factors.event_store import EventStore, _convert_legacy_event

        if not events_path.exists():
            logger.warning("Unified store: events file not found at %s", events_path)
            return

        # Read legacy records
        records: list[dict] = []
        with open(events_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(_json.loads(line))
                except _json.JSONDecodeError:
                    continue

        if not records:
            logger.info("Unified store: no events to write for %s", target_date)
            return

        # Convert using the same logic as migrate_legacy_events
        # (now populates all 5 time fields: event_time, publish_time,
        #  available_time, signal_date, execution_date)
        converted = [_convert_legacy_event(r, target_date) for r in records]

        store = EventStore()
        n_stored = store.add_events(converted)
        logger.info(
            "Unified store (PRIMARY): wrote %d/%d events for %s (dir=%s)",
            n_stored, len(converted), target_date, store.store_dir,
        )
    except Exception as e:
        logger.warning("Unified store write failed (non-fatal): %s", e)
        logger.debug(traceback.format_exc())


def run_pipeline(target_date: str = None, use_portfolio: bool = False):
    """Execute the full LLM event pipeline.

    Steps:
        1. Collect daily news from AKShare
        2. Extract structured events via MiniMax LLM
        3. Build quantitative factors from extracted events

    Args:
        target_date: YYYY-MM-DD (default: today)
        use_portfolio: use portfolio stocks instead of top liquid
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"=== LLM Event Pipeline START for {target_date} ===")
    start_time = datetime.now()

    # Step 0: Collect announcements (higher coverage than news)
    logger.info("[Step 0/3] Collecting announcements...")
    try:
        from scripts.collect_announcements import collect_for_date
        ann_path = collect_for_date(target_date)
        n_ann = sum(1 for _ in open(ann_path)) if ann_path.exists() else 0
        logger.info(f"  Announcements: {n_ann}")

        # Merge announcements into news format for LLM extraction
        if ann_path.exists() and n_ann > 0:
            import json as _json
            ann_items = []
            with open(ann_path) as f:
                for line in f:
                    a = _json.loads(line)
                    ann_items.append({
                        "stock_code": a.get("stock_code", ""),
                        "stock_name": a.get("stock_name", ""),
                        "title": a.get("title", ""),
                        "content_snippet": a.get("title", ""),  # announcements: title IS the content
                        "source": "交易所公告",
                        "publish_time": a.get("notice_date", a.get("publish_time", "")),
                        "qlib_code": a.get("qlib_code", ""),
                        "collect_date": target_date,
                    })
            # Will be appended to news file after news collection
            logger.info(f"  {len(ann_items)} announcement items ready for merge")
    except Exception as e:
        logger.warning(f"  Announcement collection failed: {e}")
        ann_items = []

    # Step 1: Collect news
    logger.info("[Step 1/3] Collecting daily news...")
    try:
        from scripts.collect_daily_news import collect_daily_news

        news_path = collect_daily_news(
            target_date=target_date,
            use_portfolio=use_portfolio,
            top_n=5000,  # full A-share coverage
        )
        logger.info(f"  News collected -> {news_path}")

        # Merge announcements into news file (append, dedup by stock+title)
        if ann_items and news_path.exists():
            import json as _json2
            existing_keys = set()
            with open(news_path) as f:
                for line in f:
                    item = _json2.loads(line)
                    existing_keys.add(f"{item.get('stock_code','')}_{item.get('title','')[:30]}")

            n_merged = 0
            with open(news_path, "a") as f:
                for item in ann_items:
                    key = f"{item.get('stock_code','')}_{item.get('title','')[:30]}"
                    if key not in existing_keys:
                        f.write(_json2.dumps(item, ensure_ascii=False) + "\n")
                        existing_keys.add(key)
                        n_merged += 1
            logger.info(f"  Merged {n_merged} announcements into news file")

    except Exception as e:
        logger.error(f"  News collection failed: {e}")
        logger.debug(traceback.format_exc())
        return False

    # Step 2: Extract events via LLM (120-min timeout for full-A 5000 stocks)
    logger.info("[Step 2/3] Extracting events via MiniMax LLM...")
    try:
        import signal as _signal
        from factors.llm_event_extractor import LLMEventExtractor

        class _Timeout(Exception):
            pass

        def _handler(signum, frame):
            raise _Timeout("LLM extraction exceeded 120-minute timeout")

        old_handler = _signal.signal(_signal.SIGALRM, _handler)
        _signal.alarm(7200)  # 120 minutes (5000 stocks full A coverage)
        events_path = None
        try:
            extractor = LLMEventExtractor()
            events_path = extractor.extract_from_news_file(
                news_path=news_path,
                max_news_per_stock=1,  # 1 per stock for 5000 stocks (full A)
                target_date=target_date,
            )
            logger.info(f"  Events extracted -> {events_path}")
        except _Timeout:
            logger.warning("  LLM extraction timed out at 15 min — partial results saved")
            # Extractor streams to disk, so partial file may exist
            from factors.llm_event_extractor import EVENTS_DIR
            events_path = EVENTS_DIR / f"{target_date}.jsonl"
        finally:
            _signal.alarm(0)
            _signal.signal(_signal.SIGALRM, old_handler)

        # Write to unified EventStore (PRIMARY path, non-fatal on failure)
        if events_path is not None:
            logger.warning(
                "DEPRECATION: Legacy llm_events/ file at %s is kept for "
                "backward compatibility only. The unified EventStore is now "
                "the primary store. Remove llm_events/ writes in a future release.",
                events_path,
            )
            _write_to_unified_store(events_path, target_date)

    except Exception as e:
        logger.error(f"  Event extraction failed: {e}")
        logger.debug(traceback.format_exc())
        return False

    # Step 3: Build factors
    logger.info("[Step 3/3] Building quantitative factors...")
    try:
        from scripts.build_llm_event_factors import build_factors_range

        df = build_factors_range(
            start_date=target_date,
            end_date=target_date,
            lookback_days=30,
        )
        n_stocks = len(df) if not df.empty else 0
        logger.info(f"  Factors built for {n_stocks} stocks")
    except Exception as e:
        logger.error(f"  Factor building failed: {e}")
        logger.debug(traceback.format_exc())
        return False

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"=== LLM Event Pipeline DONE in {elapsed:.0f}s ===")
    return True


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run full LLM Event Factor pipeline")
    parser.add_argument("--date", type=str, default=None, help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--portfolio", action="store_true", help="Use portfolio stocks instead of top liquid")
    args = parser.parse_args()

    success = run_pipeline(target_date=args.date, use_portfolio=args.portfolio)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
