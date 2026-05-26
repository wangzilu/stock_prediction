"""Event study analysis framework — CAR (Cumulative Abnormal Return) computation.

Computes event study returns for supply chain events and LLM events:
  1. Load events from EventStore and global_chain_events
  2. Load returns from feature cache (__pnl_return_1d)
  3. For each event, compute T+1, T+3, T+5, T+10 cumulative returns
  4. Compare with industry/market average (abnormal return)
  5. Group by: event_type, direction, topic, source
  6. Print summary table: mean CAR by group, t-stat, n_events
  7. Save results to data/storage/event_study_results.json

Usage:
    python scripts/run_event_study.py
    python scripts/run_event_study.py --start 2026-04-27 --end 2026-05-25
    python scripts/run_event_study.py --min-events 3
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
EVENTS_DIR = DATA_DIR / "events"
CHAIN_EVENTS_DIR = DATA_DIR / "global_chain_events"
OUTPUT_PATH = DATA_DIR / "event_study_results.json"

# CAR windows: trading days after event
CAR_WINDOWS = [1, 3, 5, 10]


# ---------------------------------------------------------------------------
# 1. Load events
# ---------------------------------------------------------------------------

def load_events(start_date: str, end_date: str) -> pd.DataFrame:
    """Load events from EventStore and global_chain_events."""
    all_events = []

    # a) EventStore (unified events/)
    if EVENTS_DIR.exists():
        for fp in sorted(EVENTS_DIR.glob("*.jsonl")):
            file_date = fp.stem
            if file_date < start_date or file_date > end_date:
                continue
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        obj.setdefault("date", file_date)
                        obj["_event_source"] = "event_store"
                        all_events.append(obj)
                    except json.JSONDecodeError:
                        continue

    # b) global_chain_events (supply chain extractor output)
    if CHAIN_EVENTS_DIR.exists():
        for fp in sorted(CHAIN_EVENTS_DIR.glob("*.jsonl")):
            file_date = fp.stem
            if file_date < start_date or file_date > end_date:
                continue
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        obj.setdefault("date", file_date)
                        obj["_event_source"] = "global_chain"
                        all_events.append(obj)
                    except json.JSONDecodeError:
                        continue

    if not all_events:
        return pd.DataFrame()

    df = pd.DataFrame(all_events)
    logger.info(f"Loaded {len(df)} events ({start_date} to {end_date})")
    return df


# ---------------------------------------------------------------------------
# 2. Load returns
# ---------------------------------------------------------------------------

def load_returns() -> pd.DataFrame:
    """Load daily returns from the champion feature cache.

    Returns DataFrame with columns: [datetime, instrument, __pnl_return_1d]
    """
    from config.feature_path import CHAMPION_PATH
    cache_path = DATA_DIR / CHAMPION_PATH["cache_file"]

    if not cache_path.exists():
        logger.warning(f"Feature cache not found at {cache_path}")
        return pd.DataFrame()

    df = pd.read_parquet(cache_path, columns=["datetime", "instrument", "__pnl_return_1d"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["instrument", "datetime"]).reset_index(drop=True)
    logger.info(f"Loaded returns: {len(df)} rows, {df['instrument'].nunique()} stocks, "
                f"{df['datetime'].min().date()} to {df['datetime'].max().date()}")
    return df


def build_cumulative_returns(returns_df: pd.DataFrame) -> dict:
    """Build a lookup: (instrument, date) -> {1: cum_ret_1d, 3: cum_ret_3d, ...}

    For each stock-date, compute forward cumulative return over each window.
    """
    if returns_df.empty:
        return {}

    lookup = {}
    for instrument, grp in returns_df.groupby("instrument"):
        grp = grp.sort_values("datetime").reset_index(drop=True)
        dates = grp["datetime"].dt.strftime("%Y-%m-%d").values
        rets = grp["__pnl_return_1d"].values

        for i, date_str in enumerate(dates):
            cum = {}
            for w in CAR_WINDOWS:
                if i + w <= len(rets):
                    # Forward cumulative return: product of (1+r) for T+1..T+w
                    window_rets = rets[i:i + w]
                    cum_ret = np.prod(1 + window_rets) - 1
                    cum[w] = float(cum_ret)
            if cum:
                lookup[(instrument, date_str)] = cum

    logger.info(f"Built cumulative return lookup: {len(lookup)} stock-date entries")
    return lookup


def compute_market_average(returns_df: pd.DataFrame) -> dict:
    """Compute market (cross-sectional) average cumulative returns per date.

    Returns: {date_str: {window: mean_cum_ret}}
    """
    if returns_df.empty:
        return {}

    market_avg = {}
    for dt, grp in returns_df.groupby("datetime"):
        date_str = dt.strftime("%Y-%m-%d")
        rets = grp["__pnl_return_1d"].values
        # For market average, we just use the daily mean as a simple proxy.
        # A more rigorous approach would use index returns.
        market_avg[date_str] = {"daily_mean": float(np.nanmean(rets))}

    # For each date, build forward cum market returns
    sorted_dates = sorted(market_avg.keys())
    daily_means = {d: market_avg[d]["daily_mean"] for d in sorted_dates}

    result = {}
    for i, d in enumerate(sorted_dates):
        cum = {}
        for w in CAR_WINDOWS:
            if i + w <= len(sorted_dates):
                window_means = [daily_means[sorted_dates[j]] for j in range(i, i + w)]
                cum[w] = float(np.prod([1 + r for r in window_means]) - 1)
        if cum:
            result[d] = cum

    return result


# ---------------------------------------------------------------------------
# 3. Normalize stock codes
# ---------------------------------------------------------------------------

def normalize_stock_code(code: str) -> str:
    """Normalize stock code to match feature cache instrument format.

    EventStore uses '600519', cache uses 'sh600519' or 'SH600519'.
    Supply chain edges use 'sh600519'.
    """
    if not code:
        return ""
    code = str(code).strip()
    # Already has prefix
    if code.startswith(("sh", "sz", "SH", "SZ")):
        return code.upper()
    # Pure digits: infer exchange
    digits = code.zfill(6)
    if digits.startswith(("6", "9")):
        return f"SH{digits}"
    else:
        return f"SZ{digits}"


# ---------------------------------------------------------------------------
# 4. Event study computation
# ---------------------------------------------------------------------------

def compute_event_study(
    events_df: pd.DataFrame,
    cum_lookup: dict,
    market_avg: dict,
) -> list[dict]:
    """For each event, compute CAR (Cumulative Abnormal Return) over each window.

    CAR = stock cumulative return - market average cumulative return
    """
    results = []

    for _, row in events_df.iterrows():
        # Get stock code and event date
        stock_code = row.get("stock_code", "")
        # For global_chain events, stock codes may be in dst_stock field
        if not stock_code:
            stock_code = row.get("dst_stock", "")
        if not stock_code:
            continue

        event_date = row.get("signal_date") or row.get("date", "")
        if not event_date:
            continue

        instrument = normalize_stock_code(stock_code)

        # Look up forward returns
        stock_cum = cum_lookup.get((instrument, event_date))
        if stock_cum is None:
            # Try lowercase
            instrument_lower = instrument.lower()
            stock_cum = cum_lookup.get((instrument_lower, event_date))
        if stock_cum is None:
            continue

        mkt_cum = market_avg.get(event_date, {})

        record = {
            "stock_code": stock_code,
            "instrument": instrument,
            "event_date": event_date,
            "event_type": row.get("event_type", "unknown"),
            "direction": int(row.get("direction", 0)),
            "confidence": float(row.get("confidence", 0.5)),
            "topic": row.get("topic", "unknown"),
            "source": row.get("source", row.get("_event_source", "unknown")),
            "summary": str(row.get("summary", ""))[:100],
        }

        for w in CAR_WINDOWS:
            raw_ret = stock_cum.get(w)
            mkt_ret = mkt_cum.get(w, 0.0)
            if raw_ret is not None:
                record[f"raw_ret_{w}d"] = round(raw_ret, 6)
                record[f"car_{w}d"] = round(raw_ret - mkt_ret, 6)
            else:
                record[f"raw_ret_{w}d"] = None
                record[f"car_{w}d"] = None

        results.append(record)

    logger.info(f"Computed event study for {len(results)} event-stock pairs "
                f"(dropped {len(events_df) - len(results)} with no return data)")
    return results


# ---------------------------------------------------------------------------
# 5. Group analysis
# ---------------------------------------------------------------------------

def _t_stat(values: np.ndarray) -> float:
    """Compute t-statistic for H0: mean = 0."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = np.nanmean(values)
    std = np.nanstd(values, ddof=1)
    if std == 0:
        return 0.0
    return float(mean / (std / np.sqrt(n)))


def group_analysis(
    results: list[dict],
    group_key: str,
    min_events: int = 3,
) -> list[dict]:
    """Group event study results by a key and compute summary stats.

    Returns list of dicts with: group, n_events, and for each window:
      mean_car, t_stat, mean_raw_ret, pct_positive
    """
    groups = defaultdict(list)
    for r in results:
        key_val = r.get(group_key, "unknown")
        if key_val is None or key_val == "":
            key_val = "unknown"
        groups[str(key_val)].append(r)

    summaries = []
    for key_val, records in sorted(groups.items()):
        if len(records) < min_events:
            continue

        summary = {
            "group_key": group_key,
            "group_value": key_val,
            "n_events": len(records),
        }

        for w in CAR_WINDOWS:
            cars = np.array([r[f"car_{w}d"] for r in records
                             if r.get(f"car_{w}d") is not None], dtype=float)
            raws = np.array([r[f"raw_ret_{w}d"] for r in records
                             if r.get(f"raw_ret_{w}d") is not None], dtype=float)

            if len(cars) >= min_events:
                summary[f"car_{w}d_mean"] = round(float(np.nanmean(cars)), 6)
                summary[f"car_{w}d_tstat"] = round(_t_stat(cars), 3)
                summary[f"raw_{w}d_mean"] = round(float(np.nanmean(raws)), 6)
                summary[f"car_{w}d_pct_pos"] = round(
                    float(np.nanmean(cars > 0)), 3) if len(cars) > 0 else None
                summary[f"car_{w}d_n"] = int(len(cars))
            else:
                summary[f"car_{w}d_mean"] = None
                summary[f"car_{w}d_tstat"] = None
                summary[f"raw_{w}d_mean"] = None
                summary[f"car_{w}d_pct_pos"] = None
                summary[f"car_{w}d_n"] = int(len(cars))

        summaries.append(summary)

    return summaries


def print_summary_table(summaries: list[dict], group_key: str):
    """Pretty-print a summary table."""
    if not summaries:
        print(f"  (no groups with enough events for {group_key})")
        return

    # Header
    header = f"{'Group':30s} {'N':>5s}"
    for w in CAR_WINDOWS:
        header += f"  CAR{w:>2d}d  t-stat"
    print(header)
    print("-" * len(header))

    for s in sorted(summaries, key=lambda x: -(x.get("car_5d_mean") or 0)):
        line = f"{str(s['group_value']):30s} {s['n_events']:5d}"
        for w in CAR_WINDOWS:
            car = s.get(f"car_{w}d_mean")
            tstat = s.get(f"car_{w}d_tstat")
            if car is not None:
                line += f"  {car:+.4f}  {tstat:+.2f} "
            else:
                line += f"  {'N/A':>7s}  {'N/A':>5s} "
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Event Study Analysis (CAR)")
    parser.add_argument("--start", default=None,
                        help="Start date (YYYY-MM-DD). Default: 30 days ago.")
    parser.add_argument("--end", default=None,
                        help="End date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--min-events", type=int, default=3,
                        help="Minimum events per group to show in summary.")
    parser.add_argument("--output", default=str(OUTPUT_PATH),
                        help="Output path for results JSON.")
    args = parser.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    start_date = args.start or (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    end_date = args.end or today

    print(f"=== Event Study Analysis ===")
    print(f"Period: {start_date} to {end_date}")
    print(f"Min events per group: {args.min_events}")
    print()

    # 1. Load events
    events_df = load_events(start_date, end_date)
    if events_df.empty:
        print("No events found. Exiting.")
        return

    print(f"Total events loaded: {len(events_df)}")
    if "_event_source" in events_df.columns:
        for src, cnt in events_df["_event_source"].value_counts().items():
            print(f"  {src}: {cnt}")
    print()

    # 2. Load returns
    returns_df = load_returns()
    if returns_df.empty:
        print("No returns data. Cannot compute event study. Exiting.")
        return

    # 3. Build lookups
    cum_lookup = build_cumulative_returns(returns_df)
    market_avg = compute_market_average(returns_df)

    # 4. Compute event study
    results = compute_event_study(events_df, cum_lookup, market_avg)

    if not results:
        print("No events matched to return data. Exiting.")
        return

    print(f"\nMatched {len(results)} events to return data.\n")

    # 5. Group analysis
    group_keys = ["event_type", "direction", "topic", "source"]
    all_summaries = {}

    for gk in group_keys:
        print(f"\n{'=' * 60}")
        print(f"Group by: {gk}")
        print(f"{'=' * 60}")
        summaries = group_analysis(results, gk, min_events=args.min_events)
        all_summaries[gk] = summaries
        print_summary_table(summaries, gk)

    # 6. Direction-aligned analysis (for directional events)
    # For events with direction != 0, check if direction-aligned returns are positive
    directional = [r for r in results if r.get("direction", 0) != 0]
    if directional:
        print(f"\n{'=' * 60}")
        print(f"Direction-aligned CAR (does predicted direction match actual?)")
        print(f"{'=' * 60}")
        aligned_cars = defaultdict(list)
        for r in directional:
            d = r["direction"]
            for w in CAR_WINDOWS:
                car = r.get(f"car_{w}d")
                if car is not None:
                    # Aligned = direction * car > 0
                    aligned_cars[w].append(d * car)

        header = f"{'Window':>10s}  {'Mean':>8s}  {'t-stat':>7s}  {'%correct':>8s}  {'N':>5s}"
        print(header)
        print("-" * len(header))
        for w in CAR_WINDOWS:
            vals = np.array(aligned_cars[w])
            if len(vals) >= 3:
                mean = np.nanmean(vals)
                t = _t_stat(vals)
                pct = np.nanmean(vals > 0)
                print(f"    T+{w:<4d}  {mean:+.4f}  {t:+.3f}  {pct:.1%}     {len(vals):5d}")

    # 7. Caveat for small sample
    n_unique_dates = len(set(r["event_date"] for r in results))
    print(f"\n--- Caveat ---")
    print(f"Analysis covers {n_unique_dates} unique event dates and {len(results)} event-stock pairs.")
    if n_unique_dates < 30:
        print(f"WARNING: Only {n_unique_dates} dates — results are noisy. "
              f"Framework is ready; re-run when more data accumulates.")

    # 8. Save results
    output = {
        "meta": {
            "start_date": start_date,
            "end_date": end_date,
            "n_events_loaded": len(events_df),
            "n_events_matched": len(results),
            "n_unique_dates": n_unique_dates,
            "min_events_per_group": args.min_events,
            "car_windows": CAR_WINDOWS,
            "generated_at": datetime.now().isoformat(),
        },
        "summaries": all_summaries,
        "raw_results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
