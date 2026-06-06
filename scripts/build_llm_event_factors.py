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

EVENTS_DIR_V1 = DATA_DIR / "llm_events"
EVENTS_DIR_V2 = DATA_DIR / "llm_events_v2"
OUTPUT_PATH = DATA_DIR / "llm_event_factors.parquet"


def _load_events_via_eventstore(start_dt: datetime, end_dt: datetime) -> pd.DataFrame | None:
    """Try to load events via the unified EventStore. Returns None if the
    store is unavailable / empty so the caller falls back to legacy jsonl.

    Tonight's audit (P1-12) flagged that this builder bypassed EventStore
    and read llm_events_v2/llm_events jsonl directly, which meant the 5-field
    time-semantics work in EventStore (signal_date, execution_date, etc.)
    didn't flow through to factor construction. Prefer EventStore; fall
    back keeps backfill scripts working.
    """
    try:
        from factors.event_store import EventStore
    except Exception as e:
        logger.debug("EventStore import failed (%s) — using legacy jsonl path", e)
        return None
    try:
        store = EventStore()
        df = store.query(
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
        )
        if df is None or df.empty:
            return None
        # Match the legacy schema the rest of build_llm_event_factors expects.
        # EventStore rows have: stock_code, event_type, direction, confidence,
        # publish_time, signal_date, source, summary, source_quality, etc.
        # Legacy code expects file_date (proxy for "extract day") + the V2
        # impact synthesis based on direction/is_price_sensitive.
        df = df.copy()
        df["file_date"] = df.get("signal_date", "")
        # Derive qlib_code from stock_code (legacy schema expects qlib_code,
        # EventStore only stores stock_code).
        if "qlib_code" not in df.columns and "stock_code" in df.columns:
            def _to_qlib(c):
                c = str(c).strip().upper()
                if not c:
                    return ""
                # Already a qlib code (SH/SZ/BJ + 6 digits)
                if len(c) == 8 and c[:2] in ("SH", "SZ", "BJ") and c[2:].isdigit():
                    return c
                # Bare numeric -> infer prefix
                if c.isdigit() and len(c) == 6:
                    if c.startswith(("60", "68", "9")):
                        return f"SH{c}"
                    if c.startswith(("00", "30", "20")):
                        return f"SZ{c}"
                    if c.startswith(("4", "8")):
                        return f"BJ{c}"
                return ""
            df["qlib_code"] = df["stock_code"].apply(_to_qlib)
        # Synthesize impact from direction + sensitivity (mirrors load_events V2 branch)
        direction = pd.to_numeric(df.get("direction", 0), errors="coerce").fillna(0)
        is_price_sensitive = df.get("is_price_sensitive", False)
        if not isinstance(is_price_sensitive, pd.Series):
            is_price_sensitive = pd.Series([False] * len(df), index=df.index)
        magnitude = is_price_sensitive.map(lambda x: 0.05 if x else 0.02)
        if "impact_1d" not in df.columns:
            df["impact_1d"] = direction * magnitude
        if "impact_5d" not in df.columns:
            df["impact_5d"] = direction * magnitude * 0.6
        if "relevance" not in df.columns:
            df["relevance"] = df.get("is_official_disclosure", False).map(lambda x: 1.0 if x else 0.7) if "is_official_disclosure" in df.columns else 0.7
        if "novelty" not in df.columns:
            df["novelty"] = df.get("is_new_information", True).map(lambda x: 0.9 if x else 0.2) if "is_new_information" in df.columns else 0.7
        logger.info("Loaded %d events via EventStore", len(df))
        return df
    except Exception as e:
        logger.warning("EventStore query failed (%s) — using legacy jsonl path", e)
        return None


def load_events(lookback_days: int = 30, as_of: str = None,
                source: str = "eventstore", allow_fallback: bool = False) -> pd.DataFrame:
    """Load LLM-extracted events from EventStore (default) or JSONL.

    2026-06-06: default flipped from "jsonl" to "eventstore" after the
    project lead's data-pipeline architecture critique. JSONL groups
    events by *file_date* (the day the extractor ran), so a re-extraction
    inflates counts and shifts the time bucket; EventStore groups by
    *signal_date* (the next business day after publish time = the first
    actionable trading day), which is the PIT-safe semantic the rest
    of the pipeline assumes.

    source: "eventstore" (default — PIT-safe, signal_date semantics) or
        "jsonl" (legacy, file_date semantics; kept for debug / backfill).
    allow_fallback: when source="eventstore" and the EventStore query is empty
        or fails, return jsonl results instead of raising. Default False so
        IC backtests can't be silently contaminated — if you ask for
        EventStore, you must get EventStore or a clear failure.

    Distribution caveat: on 2026-05-29 production data the two paths
    produced wildly different factor distributions (sentiment_score
    mean differed by 17000%, only ~24% stock overlap). After the flip,
    a one-time IC backtest comparing the two profiles is required
    before locking eventstore in for ablation runs.

    Args:
        lookback_days: number of past days to include
        as_of: reference date YYYY-MM-DD (default: today)
        source: "jsonl" or "eventstore"
        allow_fallback: when True, eventstore empty/error falls through to jsonl

    Returns:
        DataFrame with all events from the lookback window
    """
    if as_of is None:
        as_of = datetime.now().strftime("%Y-%m-%d")

    as_of_dt = datetime.strptime(as_of, "%Y-%m-%d")
    start_dt = as_of_dt - timedelta(days=lookback_days)

    if source == "eventstore":
        es_df = _load_events_via_eventstore(start_dt, as_of_dt)
        if es_df is not None and not es_df.empty:
            return es_df
        if not allow_fallback:
            raise RuntimeError(
                f"source='eventstore' explicitly requested but EventStore returned "
                f"empty/None for [{start_dt.date()}, {as_of_dt.date()}]. Pass "
                f"allow_fallback=True (or --allow-fallback on the CLI) to silently "
                f"fall back to jsonl. Default is fail-loud so IC backtests can't "
                f"unknowingly measure the wrong source."
            )
        logger.warning("EventStore source requested but empty/unavailable — falling back to jsonl (--allow-fallback)")

    # Default / fallback: legacy jsonl files
    all_records = []
    for day_offset in range(lookback_days + 1):
        date = (start_dt + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        # Prefer V2 events; fall back to V1 for historical data
        event_file = EVENTS_DIR_V2 / f"{date}.jsonl"
        if not event_file.exists():
            event_file = EVENTS_DIR_V1 / f"{date}.jsonl"
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
                    # Normalize V2 records to have impact_1d/impact_5d columns
                    # V2 has direction + is_price_sensitive instead of LLM-predicted impacts
                    if "extractor_version" in record and record.get("extractor_version") == "v2":
                        direction = float(record.get("direction", 0))
                        is_price_sensitive = record.get("is_price_sensitive", False)
                        # Synthesize impact from direction + sensitivity
                        # Price-sensitive events get larger magnitude
                        magnitude = 0.05 if is_price_sensitive else 0.02
                        record.setdefault("impact_1d", direction * magnitude)
                        record.setdefault("impact_5d", direction * magnitude * 0.6)
                        # V2 doesn't have relevance/novelty — use confidence + flags
                        record.setdefault("relevance", 1.0 if record.get("is_official_disclosure") else 0.7)
                        record.setdefault("novelty", 0.9 if record.get("is_new_information", True) else 0.2)
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


def build_factors(signal_date: str = None, lookback_days: int = 30,
                  source: str = "eventstore", allow_fallback: bool = False) -> pd.DataFrame:
    """Build LLM event factors for a given signal date.

    PIT-safe: only uses events with publish_time <= signal_date.

    Args:
        signal_date: YYYY-MM-DD (default: today)
        lookback_days: days of history to consider
        source: "eventstore" (default, 2026-06-06 flip) or "jsonl" (legacy)

    Returns:
        DataFrame indexed by qlib_code with factor columns
    """
    if signal_date is None:
        signal_date = datetime.now().strftime("%Y-%m-%d")

    signal_dt = datetime.strptime(signal_date, "%Y-%m-%d")

    df = load_events(lookback_days=lookback_days, as_of=signal_date, source=source,
                     allow_fallback=allow_fallback)
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
    # Phase C.3 (L1) fact-count window — project lead's instruction is
    # to count facts over a short window (3 days) instead of a longer
    # ranked-impact window. The 5-day count is kept for back-compat;
    # the 3-day counts are the new canonical inputs.
    very_recent = df[df["age_days"] <= 3]

    agg_all = df.groupby("qlib_code").agg(
        llm_impact_1d_decayed=("weighted_impact_1d", "sum"),
        llm_impact_5d_decayed=("weighted_impact_5d", "sum"),
        llm_sentiment_score=("sentiment_contrib", "sum"),
    )

    agg_recent = recent.groupby("qlib_code").agg(
        llm_event_count_5d=("event_type", "count"),
        llm_avg_confidence=("confidence", "mean"),
    )

    # Phase C.3 (L1): replace the ``direction * 0.05`` impact synthesis
    # with fact counts the project lead's 2026-06-06 LLM critique asked
    # for. These are emitted alongside (not instead of) the legacy
    # weighted_impact_* columns, so downstream factor consumers can
    # adopt the new columns at their own pace. The synthesized impact
    # path stays in place for one release as a deprecation window.
    fact_recent = very_recent.copy()
    fact_recent["_pos"] = (fact_recent["direction"] > 0).astype(int)
    fact_recent["_neg"] = (fact_recent["direction"] < 0).astype(int)
    fact_recent["_price_sens"] = fact_recent.get("is_price_sensitive", False).astype(int)
    fact_recent["_official"] = fact_recent.get("is_official_disclosure", False).astype(int)
    fact_recent["_repeated"] = fact_recent.get("is_repeated_news", False).astype(int)
    agg_facts = fact_recent.groupby("qlib_code").agg(
        llm_positive_event_count_3d=("_pos", "sum"),
        llm_negative_event_count_3d=("_neg", "sum"),
        llm_price_sensitive_count_3d=("_price_sens", "sum"),
        llm_official_event_count_3d=("_official", "sum"),
        llm_event_count_3d=("event_type", "count"),
        _repeated_sum=("_repeated", "sum"),
    )
    n3 = agg_facts["llm_event_count_3d"]
    agg_facts["llm_repeated_ratio_3d"] = (
        agg_facts["_repeated_sum"] / n3.where(n3 > 0, 1)
    ).fillna(0.0)
    agg_facts["llm_event_intensity"] = (n3 / 3.0).fillna(0.0)
    agg_facts = agg_facts.drop(columns=["_repeated_sum"])

    result = agg_all.join(agg_recent, how="left").join(agg_facts, how="left")
    result["llm_event_count_5d"] = result["llm_event_count_5d"].fillna(0).astype(int)
    result["llm_avg_confidence"] = result["llm_avg_confidence"].fillna(0.0)
    # L1 columns default to 0 (= no recent events) when a stock has
    # 5-day events but no 3-day events. The join above leaves NaN for
    # codes outside very_recent; cast cleanly.
    for col in (
        "llm_positive_event_count_3d", "llm_negative_event_count_3d",
        "llm_price_sensitive_count_3d", "llm_official_event_count_3d",
        "llm_event_count_3d",
    ):
        result[col] = result[col].fillna(0).astype(int)
    for col in ("llm_repeated_ratio_3d", "llm_event_intensity"):
        result[col] = result[col].fillna(0.0)

    result["signal_date"] = signal_date
    result = result.reset_index()

    logger.info(
        f"Built LLM event factors for {len(result)} stocks on {signal_date}, "
        f"from {len(df)} events"
    )
    return result


_ALLOWED_FACTOR_SOURCES = ("jsonl", "eventstore")


def resolve_llm_event_factor_source(
    explicit: str | None = None,
    *,
    default: str = "eventstore",
) -> str:
    """Single resolver for the LLM event factor source.

    2026-06-04 cx round 20 P1-1 + P1-2: pre-fix the pipeline read
    ``os.environ.get("LLM_EVENT_FACTOR_SOURCE", "jsonl")`` and any
    other call site (retry drain, ad-hoc backfills) used the bare
    ``source="jsonl"`` keyword argument. Typos like ``event_store``
    or ``eventsstore`` silently fell through to JSONL, AND the
    retry drain could pick a DIFFERENT source than the main pipeline
    on the same day — overwriting the morning's parquet with the
    evening's other-source output.

    Args:
        explicit: optional caller-provided source. If None, consults
            the env var ``LLM_EVENT_FACTOR_SOURCE``.
        default: value when neither caller nor env supplies one.

    Returns:
        Normalised source name (``"jsonl"`` or ``"eventstore"``).

    Raises:
        RuntimeError: when the resolved value is not one of the
            allowed options. Callers MUST treat invalid input as
            a hard fail rather than silently substituting a default.
    """
    import os as _os
    raw = (
        explicit
        if explicit is not None
        else _os.environ.get("LLM_EVENT_FACTOR_SOURCE", default)
    )
    norm = str(raw or "").strip().lower()
    if norm not in _ALLOWED_FACTOR_SOURCES:
        raise RuntimeError(
            f"Invalid LLM event factor source {raw!r}. "
            f"Allowed: {list(_ALLOWED_FACTOR_SOURCES)}."
        )
    return norm


def build_factors_range(
    start_date: str = None,
    end_date: str = None,
    lookback_days: int = 30,
    source: str = "eventstore",
    allow_fallback: bool = False,
) -> pd.DataFrame:
    # 2026-06-06: flipped default back to "eventstore" after the project
    # lead's data-pipeline architecture critique. JSONL groups by
    # file_date which is mis-PIT relative to the rest of the pipeline;
    # EventStore groups by signal_date (next business day after
    # publish_time) which matches what every other overlay assumes.
    # cx round 17 P1-3 reverted this on 2026-06-04 to "jsonl" because
    # the parquet contract didn't yet declare the source, so a silent
    # 100x distribution change could ship without ablation evidence.
    # The contract gate now records source under "extra", so the flip
    # is no longer silent — callers can still pass source="jsonl"
    # explicitly for back-compat backfills.
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
        df = build_factors(signal_date=date_str, lookback_days=lookback_days,
                           source=source, allow_fallback=allow_fallback)
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
    parser.add_argument(
        "--source", choices=["jsonl", "eventstore"], default="eventstore",
        help="Event source: 'jsonl' (default, current production) groups by file_date; "
             "'eventstore' groups by signal_date and changes factor distributions by ~100x. "
             "Use eventstore only for IC backtests, not live production.",
    )
    parser.add_argument(
        "--allow-fallback", action="store_true",
        help="If --source eventstore is requested but EventStore is empty/unavailable, "
             "fall back to jsonl. Default is fail-loud so IC backtests can't silently "
             "measure the wrong source.",
    )
    args = parser.parse_args()

    if args.date:
        df = build_factors(signal_date=args.date, lookback_days=args.lookback,
                           source=args.source, allow_fallback=args.allow_fallback)
        if not df.empty:
            build_factors_range(
                start_date=args.date,
                end_date=args.date,
                lookback_days=args.lookback,
                source=args.source,
                allow_fallback=args.allow_fallback,
            )
    else:
        build_factors_range(
            start_date=args.start,
            end_date=args.end,
            lookback_days=args.lookback,
            source=args.source,
            allow_fallback=args.allow_fallback,
        )


if __name__ == "__main__":
    main()
