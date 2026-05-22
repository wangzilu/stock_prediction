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

    # Step 1: Collect news
    logger.info("[Step 1/3] Collecting daily news from AKShare...")
    try:
        from scripts.collect_daily_news import collect_daily_news

        news_path = collect_daily_news(
            target_date=target_date,
            use_portfolio=use_portfolio,
            top_n=3000,  # 3000 stocks — concurrent collection makes this feasible
        )
        logger.info(f"  News collected -> {news_path}")
    except Exception as e:
        logger.error(f"  News collection failed: {e}")
        logger.debug(traceback.format_exc())
        return False

    # Step 2: Extract events via LLM (with 15-min total timeout)
    logger.info("[Step 2/3] Extracting events via MiniMax LLM...")
    try:
        import signal as _signal
        from factors.llm_event_extractor import LLMEventExtractor

        class _Timeout(Exception):
            pass

        def _handler(signum, frame):
            raise _Timeout("LLM extraction exceeded 15-minute timeout")

        old_handler = _signal.signal(_signal.SIGALRM, _handler)
        _signal.alarm(1500)  # 25 minutes (1000 stocks × 16 concurrent)
        try:
            extractor = LLMEventExtractor()
            events_path = extractor.extract_from_news_file(
                news_path=news_path,
                max_news_per_stock=1,  # 1 per stock for 1000 stocks within timeout
                target_date=target_date,
            )
            logger.info(f"  Events extracted -> {events_path}")
        except _Timeout:
            logger.warning("  LLM extraction timed out at 15 min — partial results saved")
        finally:
            _signal.alarm(0)
            _signal.signal(_signal.SIGALRM, old_handler)
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
