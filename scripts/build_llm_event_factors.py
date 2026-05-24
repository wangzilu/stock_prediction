"""Build quantitative factors from LLM-extracted events.

Reads LLM event JSONL files and computes per-stock, per-date factor values
with exponential time decay. PIT-safe: only uses events published on or before
the signal date.

Output: data/storage/llm_event_factors.parquet

Factors:
  - llm_impact_1d_decayed: 1-day impact with decay (half-life 2 days)
  - llm_impact_5d_decayed: 5-day impact with decay (half-life 5 days)
  - llm_event_count_5d: count of LLM events in last 5 days
  - llm_avg_confidence: mean confidence of recent events
  - llm_sentiment_score: weighted sum of recent event impacts

Usage:
    python -m scripts.build_llm_event_factors [--date 2024-01-15] [--lookback 30]
"""
import argparse
import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

EVENTS_DIR = DATA_DIR / "llm_events"
OUTPUT_PATH = DATA_DIR / "llm_event_factors.parquet"


def load_events(lookback_days: int = 30, as_of: str = None) -> pd.DataFrame:
    """Load LLM-extracted events from JSONL files within lookback window.

    Args:
        lookback_days: number of past days to include
        as_of: reference date YYYY-MM-DD (default: today)

    Returns:
        DataFrame with all events from the lookback window
    """
    if as_of is None:
        as_of = datetime.now().strftime("%Y-%m-%d")

    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
    start_dt = as_of_dt - timedelta(days=lookback_days)

    all_records = []
    for day_offset in range(lookback_days + 1):
        date = (start_dt + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        event_file = EVENTS_DIR / f"{date}.jsonl"
        if not event_file.exists():
            continue

        with open(event_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    record["file_date"] = date
                    all_records.append(record)
                except json.JSONDecodeError:
                    continue

    if not all_records:
        logger.warning(f"No LLM events found in last {lookback_days} days as of {as_of}")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    logger.info(f"Loaded {len(df)} LLM events from {lookback_days} days")
    return df


def parse_publish_date(publish_time_str: str) -> str | None:
    """Parse publish_time string to YYYY-MM-DD date."""
    if not publish_time_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
                "%Y/%m/%d %H:%M:%S", "%Y%m%d", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(publish_time_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Try to extract date portion
    try:
        return publish_time_str.strip()[:10]
    except Exception:
        return None


def build_factors(signal_date: str = None, lookback_days: int = 30) -> pd.DataFrame:
    """Build LLM event factors for a given signal date.

    PIT-safe: only uses events with publish_time <= signal_date.

    Args:
        signal_date: YYYY-MM-DD (default: today)
        lookback_days: days of history to consider

    Returns:
        DataFrame indexed by qlib_code with factor columns
    """
    if signal_date is None:
        signal_date = datetime.now().strftime("%Y-%m-%d")

    signal_dt = datetime.strptime(signal_date, "%Y-%m-%d")

    df = load_events(lookback_days=lookback_days, as_of=signal_date)
    if df.empty:
        return pd.DataFrame()

    # Parse publish date and enforce PIT constraint
    # CX: must distinguish pre-close (before 15:00) vs post-close news
    # Pre-close news: usable for same-day signal
    # Post-close news: only usable for next-day signal
    df["publish_date"] = df["publish_time"].apply(parse_publish_date)
    df["event_date"] = df["publish_date"].fillna(df["file_date"])

    # Parse publish hour for PIT boundary
    def _get_publish_hour(pt):
        if not pt or len(pt) < 13:
            return 20  # unknown → treat as post-close (conservative)
        try:
            return int(pt[11:13])
        except (ValueError, IndexError):
            return 20

    df["publish_hour"] = df["publish_time"].apply(_get_publish_hour)

    # PIT rule: post-close news (>= 15:00) shifts signal_date to next day
    # For signal_date computation: news published after 15:00 on date D
    # can only be used for signal on date D+1
    # Here we adjust event_date: if publish_hour >= 15, shift event_date +1 day
    mask_postclose = (df["publish_hour"] >= 15) & (df["event_date"] == signal_date)
    # Don't shift — just exclude same-day post-close news
    # They'll be included when signal_date is tomorrow
    df = df[~mask_postclose | (df["event_date"] < signal_date)].copy()
    df = df[df["event_date"] <= signal_date].copy()

    if df.empty:
        logger.warning(f"No PIT-safe events for {signal_date}")
        return pd.DataFrame()

    # Compute age in days
    df["age_days"] = df["event_date"].apply(
        lambda d: (signal_dt - datetime.strptime(d, "%Y-%m-%d")).days
    )

    # Filter to events within 5 days for count/avg, but use all for decayed impacts
    df["qlib_code"] = df["qlib_code"].fillna("")
    df = df[df["qlib_code"] != ""].copy()

    # Ensure numeric columns
    for col in ["impact_1d", "impact_5d", "confidence", "relevance", "novelty"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Source quality weighting (CX requirement: exchange > institutional > general > social)
    df["source_quality"] = pd.to_numeric(df.get("source_quality", 0.5), errors="coerce").fillna(0.5)

    # Compute per-event weights
    df["decay_1d"] = df["age_days"].apply(lambda a: math.exp(-a / 2.0))
    df["decay_5d"] = df["age_days"].apply(lambda a: math.exp(-a / 5.0))
    df["quality"] = df["confidence"] * df["relevance"] * df["novelty"] * df["source_quality"]

    # Factor 1: impact_1d * quality * source_quality * exp(-age/2)
    df["weighted_impact_1d"] = df["impact_1d"] * df["quality"] * df["decay_1d"]
    # Factor 2: impact_5d * quality * source_quality * exp(-age/5)
    df["weighted_impact_5d"] = df["impact_5d"] * df["quality"] * df["decay_5d"]
    # Factor 5: sentiment = impact * confidence * source_quality * decay_5d
    df["sentiment_contrib"] = df["impact_1d"] * df["confidence"] * df["source_quality"] * df["decay_5d"]

    # Aggregate per stock
    recent = df[df["age_days"] <= 5]

    agg_all = df.groupby("qlib_code").agg(
        llm_impact_1d_decayed=("weighted_impact_1d", "sum"),
        llm_impact_5d_decayed=("weighted_impact_5d", "sum"),
        llm_sentiment_score=("sentiment_contrib", "sum"),
    )

    agg_recent = recent.groupby("qlib_code").agg(
        llm_event_count_5d=("event_type", "count"),
        llm_avg_confidence=("confidence", "mean"),
    )

    result = agg_all.join(agg_recent, how="left")
    result["llm_event_count_5d"] = result["llm_event_count_5d"].fillna(0).astype(int)
    result["llm_avg_confidence"] = result["llm_avg_confidence"].fillna(0.0)

    result["signal_date"] = signal_date
    result = result.reset_index()

    logger.info(
        f"Built LLM event factors for {len(result)} stocks on {signal_date}, "
        f"from {len(df)} events"
    )
    return result


def build_factors_range(
    start_date: str = None,
    end_date: str = None,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """Build factors for a date range and save to parquet.

    Args:
        start_date: YYYY-MM-DD (default: 30 days ago)
        end_date: YYYY-MM-DD (default: today)
        lookback_days: days of event history per signal date

    Returns:
        Combined DataFrame for all dates
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    all_dfs = []
    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        df = build_factors(signal_date=date_str, lookback_days=lookback_days)
        if not df.empty:
            all_dfs.append(df)
        current += timedelta(days=1)

    if not all_dfs:
        logger.warning("No factors built for any date in range")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Append to existing file if present
    if OUTPUT_PATH.exists():
        try:
            existing = pd.read_parquet(OUTPUT_PATH)
            # Remove overlapping dates to avoid duplicates
            new_dates = set(combined["signal_date"].unique())
            existing = existing[~existing["signal_date"].isin(new_dates)]
            combined = pd.concat([existing, combined], ignore_index=True)
        except Exception as e:
            logger.warning(f"Failed to read existing parquet, overwriting: {e}")

    combined.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved {len(combined)} factor rows to {OUTPUT_PATH}")
    return combined


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build quantitative factors from LLM events")
    parser.add_argument("--date", type=str, default=None, help="Single signal date YYYY-MM-DD")
    parser.add_argument("--start", type=str, default=None, help="Range start date")
    parser.add_argument("--end", type=str, default=None, help="Range end date")
    parser.add_argument("--lookback", type=int, default=30, help="Event lookback days (default: 30)")
    args = parser.parse_args()

    if args.date:
        df = build_factors(signal_date=args.date, lookback_days=args.lookback)
        if not df.empty:
            # Single-date mode: save/append to parquet
            build_factors_range(
                start_date=args.date,
                end_date=args.date,
                lookback_days=args.lookback,
            )
    else:
        build_factors_range(
            start_date=args.start,
            end_date=args.end,
            lookback_days=args.lookback,
        )


if __name__ == "__main__":
    main()
