"""Daily LLM Event Factor pipeline runner.

Orchestrates the full pipeline: collect news -> extract events -> build factors.
Designed for crontab execution before the evening outlook.

Uses LLMEventExtractorV2 by default (fact extraction, no LLM impact prediction).
Pass --legacy to fall back to the deprecated V1 extractor.

Usage:
    python -m scripts.run_llm_event_pipeline [--date 2024-01-15] [--portfolio]
    python -m scripts.run_llm_event_pipeline --legacy   # deprecated V1

Crontab example (run at 16:30 after market close):
    30 16 * * 1-5 cd /path/to/stockPrediction && python -m scripts.run_llm_event_pipeline
"""
import argparse
import logging
import os
import sys
import traceback
import warnings
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


def run_pipeline(target_date: str = None, use_portfolio: bool = False,
                  use_legacy_v1: bool = False):
    """Execute the full LLM event pipeline.

    Steps:
        1. Collect daily news from AKShare
        2. Extract structured events via MiniMax LLM (V2 default, V1 legacy)
        3. Build quantitative factors from extracted events

    Args:
        target_date: YYYY-MM-DD (default: today)
        use_portfolio: use portfolio stocks instead of top liquid
        use_legacy_v1: use deprecated V1 extractor (LLM predicts impact)
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

    # Step 1.5: L0 classify + ranking filter, then write filtered output.
    # L0 classify_l0() partitions items into: direct (rule-emits structured
    # event, skips LLM), drop (noise), dup (already extracted from cache),
    # and l1 (default LLM path). The ranking filter then trims l1 to top-N.
    # Direct events get appended to the V2 jsonl ahead of Step 2 so the
    # extractor can resume / dedup against them.
    # Raw daily_news/ stays untouched; filtered output goes to a sister dir.
    logger.info("[Step 1.5/3] L0 classify + event_filter ranking...")
    filtered_path = news_path
    direct_events: list[dict] = []
    try:
        from factors.event_filter import classify_l0, filter_candidates, select_for_llm
        import json as _json_filter

        if news_path.exists():
            raw_items = []
            with open(news_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_items.append(_json_filter.loads(line))
                    except _json_filter.JSONDecodeError:
                        continue

            total_before = len(raw_items)
            if total_before > 0:
                # L0 classify with persistent dedup cache
                l0_cache = DATA_DIR / "llm_event_cache" / "seen.jsonl"
                routed = classify_l0(
                    raw_items, extract_date=target_date, cache_path=l0_cache,
                )
                stats = routed["stats"]
                logger.info(
                    "  L0 routing: total=%d direct=%d l1=%d drop=%d dup=%d rules=%s",
                    stats["total_in"], stats["direct"], stats["l1"],
                    stats["drop"], stats["dup"], stats["rule_hits"],
                )
                direct_events = routed["direct"]

                # Rank-and-cap the L1 stream
                scored = filter_candidates(routed["l1"])
                selected = select_for_llm(scored)
                total_after = len(selected)
                logger.info(
                    "  Rank+select: %d L1 candidates -> %d for LLM",
                    len(routed["l1"]), total_after,
                )

                filtered_dir = news_path.parent.parent / "daily_news_filtered"
                filtered_dir.mkdir(parents=True, exist_ok=True)
                filtered_path = filtered_dir / news_path.name
                with open(filtered_path, "w", encoding="utf-8") as f:
                    for item in selected:
                        item.pop("priority_score", None)
                        item.pop("must_send", None)
                        item.pop("_l2_hint", None)
                        f.write(_json_filter.dumps(item, ensure_ascii=False) + "\n")
            else:
                logger.info("  No items to filter")
        else:
            logger.info("  News file not found, skipping filter")

    except ImportError as ie:
        # 2026-06-04 cx round 4 P0-3: pre-fix this only warned and
        # let Step 2 fan 5113 raw articles into the LLM. That is
        # exactly the path that produced today's HTTP 429 RPM=1002
        # storm + 7200s wrapper timeout. ImportError of the filter
        # means a deploy is incomplete — fail closed.
        raise RuntimeError(
            f"event_filter not importable ({ie}); refusing to fan "
            f"unfiltered news into LLM extractor — that's the path "
            f"that produced today's RPM=1002 + 7200s wrapper kill."
        ) from ie
    except Exception as e:
        # Same logic for any other filter failure: applying a
        # minimum conservative fallback (date cutoff + title dedup +
        # absolute cap) so we never fan-out raw news. The fallback
        # cap mirrors what select_for_llm would have done at
        # the configured ceiling.
        logger.warning(
            "  event_filter failed (%s) — applying conservative "
            "fallback (title dedup + 500-item cap) instead of raw fan-out",
            e,
        )
        import traceback as _tb
        logger.debug(_tb.format_exc())
        try:
            import json as _json_fallback
            # cx round 16 P2-4: fallback also enforces the same 7-day
            # recency cutoff so stale news from older raw files cannot
            # leak through the bypass. Pre-fix the fallback did only
            # title dedup + 500 cap, leaving old news visible to the
            # extractor when the upstream collector was bypassed.
            from scripts.collect_daily_news import _is_recent_news
            if news_path.exists():
                seen_titles: set[str] = set()
                fallback_items: list[dict] = []
                with open(news_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = _json_fallback.loads(line)
                        except _json_fallback.JSONDecodeError:
                            continue
                        if not _is_recent_news(item.get("publish_time", "")):
                            continue
                        t = (item.get("title") or "").strip()
                        if not t or t in seen_titles:
                            continue
                        seen_titles.add(t)
                        fallback_items.append(item)
                        if len(fallback_items) >= 500:
                            break
                fallback_dir = news_path.parent.parent / "daily_news_filtered"
                fallback_dir.mkdir(parents=True, exist_ok=True)
                filtered_path = fallback_dir / news_path.name
                with open(filtered_path, "w", encoding="utf-8") as f:
                    for item in fallback_items:
                        f.write(_json_fallback.dumps(item, ensure_ascii=False) + "\n")
                logger.warning(
                    "  event_filter fallback wrote %d items (cap=500) to %s",
                    len(fallback_items), filtered_path,
                )
        except Exception as fallback_exc:
            raise RuntimeError(
                f"event_filter failed AND conservative fallback "
                f"failed ({fallback_exc}); refusing to feed LLM."
            ) from fallback_exc

    news_path = filtered_path

    # Step 2: Extract events via LLM (120-min timeout for full-A 5000 stocks)
    logger.info("[Step 2/3] Extracting events via MiniMax LLM...")
    try:
        import signal as _signal

        class _Timeout(Exception):
            pass

        def _handler(signum, frame):
            raise _Timeout("LLM extraction exceeded 120-minute timeout")

        # Select extractor version
        if use_legacy_v1:
            warnings.warn(
                "LLMEventExtractor V1 is deprecated and will be removed in a "
                "future release. Migrate to V2 (the default) which extracts "
                "structured facts instead of LLM-predicted impacts.",
                DeprecationWarning,
                stacklevel=2,
            )
            logger.warning("Using DEPRECATED V1 extractor (--legacy flag)")
            from factors.llm_event_extractor import LLMEventExtractor
            from factors.llm_event_extractor import EVENTS_DIR as _EVENTS_DIR
            extractor = LLMEventExtractor()
        else:
            logger.info("  Using V2 extractor (fact extraction, no LLM impact prediction)")
            from factors.llm_event_extractor_v2 import LLMEventExtractorV2
            from factors.llm_event_extractor_v2 import EVENTS_DIR as _EVENTS_DIR
            extractor = LLMEventExtractorV2()

        old_handler = _signal.signal(_signal.SIGALRM, _handler)
        _signal.alarm(7200)  # 120 minutes (5000 stocks full A coverage)
        events_path = None
        try:
            events_path = extractor.extract_from_news_file(
                news_path=news_path,
                max_news_per_stock=1,  # 1 per stock for 5000 stocks (full A)
                target_date=target_date,
            )
            logger.info(f"  Events extracted -> {events_path}")
        except _Timeout:
            logger.warning("  LLM extraction timed out at 120 min — partial results saved")
            # Extractor streams to disk, so partial file may exist
            events_path = _EVENTS_DIR / f"{target_date}.jsonl"
        finally:
            _signal.alarm(0)
            _signal.signal(_signal.SIGALRM, old_handler)

        # Append L0 direct events (rule-classified routines, no LLM call)
        # to the V2 jsonl AFTER the extractor writes so they aren't wiped
        # by extract_from_news_file's "fewer than 500 rows => rewrite" guard.
        if events_path is not None and direct_events:
            try:
                import json as _json_l0
                with open(events_path, "a", encoding="utf-8") as f:
                    for ev in direct_events:
                        f.write(_json_l0.dumps(ev, ensure_ascii=False) + "\n")
                logger.info(
                    "  L0 direct events appended: %d -> %s",
                    len(direct_events), events_path,
                )
            except Exception as e:
                logger.warning("L0 direct events append failed (non-fatal): %s", e)

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
    # 2026-06-04 cx round 17 P1-3 + round 19 P1-1: production
    # default is ``jsonl`` to match the builder default and the
    # CLI help block ("EventStore changes factor distribution ~100x,
    # not for live"). Flipping the production source is now an
    # opt-in via env LLM_EVENT_FACTOR_SOURCE=eventstore only.
    logger.info("[Step 3/3] Building quantitative factors...")
    try:
        from scripts.build_llm_event_factors import build_factors_range
        import os as _os
        factor_source = _os.environ.get("LLM_EVENT_FACTOR_SOURCE", "jsonl").strip().lower()
        df = build_factors_range(
            start_date=target_date,
            end_date=target_date,
            lookback_days=30,
            source=factor_source,
        )
        n_stocks = len(df) if not df.empty else 0
        logger.info(
            f"  Factors built for {n_stocks} stocks (source={factor_source})"
        )
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
    parser.add_argument("--legacy", action="store_true",
                        help="[DEPRECATED] Use V1 extractor (LLM-predicted impacts)")
    args = parser.parse_args()

    success = run_pipeline(target_date=args.date, use_portfolio=args.portfolio,
                           use_legacy_v1=args.legacy)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
