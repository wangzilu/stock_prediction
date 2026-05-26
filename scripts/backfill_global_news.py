"""Backfill historical global industry news and rebuild factors.

Orchestrates the loop over a date range, calling the existing collector
and factor builder for each missing day.

Usage:
    # Dry run: just show what would be fetched
    python scripts/backfill_global_news.py --start 2024-06-01 --end 2024-06-05 --dry-run

    # Actual backfill (will take a long time)
    python scripts/backfill_global_news.py --start 2024-01-01 --end 2026-05-25

    # Resume after interruption (skips already-fetched days)
    python scripts/backfill_global_news.py --start 2024-01-01 --end 2026-05-25 --resume
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
NEWS_DIR = DATA_DIR / "global_industry_news"
PROGRESS_PATH = DATA_DIR / "backfill_progress.json"

SLEEP_BETWEEN_DAYS = 5  # seconds between API calls to avoid rate limiting


# ---------------------------------------------------------------------------
# Trading calendar (simple approximation)
# ---------------------------------------------------------------------------

def _is_weekday(dt: datetime) -> bool:
    """Return True if dt is a weekday (Mon-Fri)."""
    return dt.weekday() < 5


def generate_trading_days(start: str, end: str) -> list[str]:
    """Generate approximate list of trading days (weekdays) in [start, end]."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    days = []
    current = start_dt
    while current <= end_dt:
        if _is_weekday(current):
            days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# Progress tracking (resume support)
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    """Load backfill progress from disk."""
    if PROGRESS_PATH.exists():
        try:
            return json.loads(PROGRESS_PATH.read_text())
        except Exception:
            pass
    return {"fetched": [], "failed": [], "last_run": None}


def save_progress(progress: dict):
    """Save backfill progress to disk."""
    progress["last_run"] = datetime.now().isoformat(timespec="seconds")
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Scan existing data
# ---------------------------------------------------------------------------

def scan_existing(trading_days: list[str]) -> tuple[list[str], list[str]]:
    """Partition trading_days into already-done and to-fetch lists.

    A day is "done" if data/storage/global_industry_news/YYYY-MM-DD.jsonl exists.

    Returns:
        (to_fetch, already_done) — both are sorted lists of date strings.
    """
    to_fetch = []
    already_done = []
    for day in trading_days:
        news_path = NEWS_DIR / f"{day}.jsonl"
        if news_path.exists():
            already_done.append(day)
        else:
            to_fetch.append(day)
    return to_fetch, already_done


# ---------------------------------------------------------------------------
# Fetch + extract + build
# ---------------------------------------------------------------------------

def fetch_single_day(date_str: str) -> bool:
    """Collect global industry news for a single day.

    Returns True on success, False on failure.
    """
    try:
        from scripts.collect_global_industry_news import collect_global_industry_news
        result_path = collect_global_industry_news(target_date=date_str, retry=True)
        logger.info("Fetched news for %s -> %s", date_str, result_path)
        return True
    except Exception as e:
        logger.error("Failed to fetch news for %s: %s", date_str, e)
        return False


def rebuild_factors(dates: list[str]):
    """Run extract + build_factors for all backfilled dates."""
    try:
        from scripts.build_global_chain_factors import build_factors
    except ImportError as e:
        logger.error("Cannot import build_factors: %s", e)
        return

    for date_str in dates:
        try:
            logger.info("Building factors for %s ...", date_str)
            build_factors(target_date=date_str, demo=False)
        except Exception as e:
            logger.error("Failed to build factors for %s: %s", date_str, e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backfill historical global industry news and rebuild factors."
    )
    parser.add_argument(
        "--start", type=str, required=True,
        help="Start date YYYY-MM-DD",
    )
    parser.add_argument(
        "--end", type=str, required=True,
        help="End date YYYY-MM-DD",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only show summary, do not fetch anything",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last progress checkpoint",
    )
    parser.add_argument(
        "--skip-factors", action="store_true",
        help="Skip factor rebuild after fetching",
    )
    parser.add_argument(
        "--sleep", type=int, default=SLEEP_BETWEEN_DAYS,
        help=f"Seconds to sleep between days (default: {SLEEP_BETWEEN_DAYS})",
    )
    args = parser.parse_args()

    # Validate dates
    try:
        datetime.strptime(args.start, "%Y-%m-%d")
        datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError as e:
        logger.error("Invalid date format: %s", e)
        sys.exit(1)

    if args.start > args.end:
        logger.error("Start date must be <= end date")
        sys.exit(1)

    # Generate trading days
    trading_days = generate_trading_days(args.start, args.end)
    logger.info("Date range: %s to %s (%d trading days)", args.start, args.end, len(trading_days))

    # Scan existing data
    to_fetch, already_done = scan_existing(trading_days)

    # If resuming, also skip days recorded in progress as already fetched
    if args.resume:
        progress = load_progress()
        prev_fetched = set(progress.get("fetched", []))
        to_fetch = [d for d in to_fetch if d not in prev_fetched]
        logger.info("Resume mode: %d days previously fetched", len(prev_fetched))

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Backfill Summary: {args.start} to {args.end}")
    print(f"{'=' * 60}")
    print(f"  Total trading days:  {len(trading_days)}")
    print(f"  Already done:        {len(already_done)}")
    print(f"  To fetch:            {len(to_fetch)}")
    if to_fetch:
        print(f"  First to fetch:      {to_fetch[0]}")
        print(f"  Last to fetch:       {to_fetch[-1]}")
        est_minutes = len(to_fetch) * args.sleep / 60
        print(f"  Estimated time:      ~{est_minutes:.0f} minutes (at {args.sleep}s/day)")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        if to_fetch:
            print("Days to fetch:")
            for d in to_fetch:
                print(f"  {d}")
        print("\n[DRY RUN] No data fetched.")
        return

    if not to_fetch:
        logger.info("Nothing to fetch -- all days already done")
        if not args.skip_factors:
            logger.info("Rebuilding factors for all dates ...")
            rebuild_factors(trading_days)
        return

    # Fetch loop with progress saving
    progress = load_progress() if args.resume else {"fetched": [], "failed": [], "last_run": None}
    fetched_this_run = []

    for i, date_str in enumerate(to_fetch):
        logger.info("[%d/%d] Fetching %s ...", i + 1, len(to_fetch), date_str)

        success = fetch_single_day(date_str)
        if success:
            progress["fetched"].append(date_str)
            fetched_this_run.append(date_str)
        else:
            progress["failed"].append(date_str)

        # Save progress after each day (resume support)
        save_progress(progress)

        # Sleep between days (skip sleep after last day)
        if i < len(to_fetch) - 1:
            logger.info("Sleeping %ds before next day ...", args.sleep)
            time.sleep(args.sleep)

    # Summary
    print(f"\nFetch complete: {len(fetched_this_run)} fetched, "
          f"{len(progress['failed'])} failed")
    if progress["failed"]:
        print(f"Failed dates: {progress['failed']}")

    # Rebuild factors for all fetched dates
    if not args.skip_factors and fetched_this_run:
        logger.info("Rebuilding factors for %d newly fetched dates ...", len(fetched_this_run))
        rebuild_factors(fetched_this_run)

    logger.info("Backfill complete.")


if __name__ == "__main__":
    main()
