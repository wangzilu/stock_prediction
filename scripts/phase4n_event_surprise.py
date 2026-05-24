#!/usr/bin/env python3
"""Phase 4N — Event Surprise Features from LLM Event Store.

Builds 5 event-based alpha factors from the unified EventStore,
registers them with Alpha Factory, and runs tearsheet validation.

Factors:
    evt_direction_confidence     — direction * confidence (basic signal)
    evt_event_count_5d           — number of events in trailing 5 days
    evt_positive_ratio_5d        — fraction of positive events in trailing 5 days
    evt_novelty_weighted         — direction * confidence * novelty penalty
    evt_source_quality_weighted  — direction * confidence * source_quality

PIT safety: uses the `date` field (= available_date) for alignment.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config.settings import DATA_DIR
from factors.event_store import EventStore
from tracker.alpha_factory import AlphaFactory, run_tearsheet_from_series

# ── Config ──────────────────────────────────────────────────────────────────
FEATURE_CACHE_PATH = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
LABEL_COL = "__label_5d"
LOOKBACK_5D = 5   # rolling window for count / ratio features
# "New information" event types (first-time material disclosure)
NEW_INFO_TYPES = {
    "earnings_beat", "earnings_miss", "earnings_positive", "earnings_negative",
    "revenue_growth", "revenue_decline",
    "order_win", "major_contract", "product_launch", "tech_breakthrough",
    "regulatory_approval", "regulatory_penalty",
    "management_change", "restructuring",
    "insider_buy", "insider_sell",
    "analyst_upgrade", "analyst_downgrade",
}


def stock_code_to_instrument(code: str) -> str:
    """Convert '600519' -> 'sh600519', '000001' -> 'sz000001'."""
    code = str(code).strip()
    if code.startswith(("sh", "sz")):
        return code
    if code.startswith(("6", "9")):
        return f"sh{code}"
    return f"sz{code}"


def load_events() -> pd.DataFrame:
    """Load all events from EventStore, add instrument column."""
    store = EventStore()
    # Query broad range
    df = store.query("2020-01-01", "2026-12-31")
    if df.empty:
        raise RuntimeError("No events found in EventStore")

    df["instrument"] = df["stock_code"].apply(stock_code_to_instrument)
    df["date"] = pd.to_datetime(df["date"])
    df["direction"] = df["direction"].astype(int)
    df["confidence"] = df["confidence"].astype(float)
    if "source_quality" in df.columns:
        df["source_quality"] = df["source_quality"].astype(float).fillna(0.5)
    else:
        df["source_quality"] = 0.5
    if "magnitude" in df.columns:
        df["magnitude"] = df["magnitude"].astype(float).fillna(0.5)
    else:
        df["magnitude"] = 0.5
    df["is_new_info"] = df["event_type"].isin(NEW_INFO_TYPES)
    return df


def load_forward_returns() -> pd.Series:
    """Load forward returns from feature cache."""
    cache = pd.read_parquet(FEATURE_CACHE_PATH, columns=[LABEL_COL])
    ret = cache[LABEL_COL].dropna()
    ret.index.names = ["datetime", "instrument"]
    return ret


def build_daily_event_features(events: pd.DataFrame, all_dates: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Build all 5 event factors as (datetime, instrument) MultiIndex Series.

    For point-in-time factors (direction_confidence, novelty, source_quality):
        we take the *sum* across all events on that date for each stock.
    For rolling factors (event_count_5d, positive_ratio_5d):
        we use a trailing 5-business-day window.
    """
    # Pre-aggregate per (date, instrument)
    daily = events.groupby(["date", "instrument"]).agg(
        dir_conf_sum=("direction", lambda x: (x * events.loc[x.index, "confidence"]).sum()),
        novelty_sum=("direction", lambda x: (
            x * events.loc[x.index, "confidence"]
            * events.loc[x.index, "is_new_info"].map({True: 1.0, False: 0.3})
        ).sum()),
        sq_sum=("direction", lambda x: (
            x * events.loc[x.index, "confidence"]
            * events.loc[x.index, "source_quality"]
        ).sum()),
        n_events=("direction", "count"),
        n_positive=("direction", lambda x: (x > 0).sum()),
    ).sort_index()

    # Build (date, instrument) → value mappings
    all_instruments = daily.index.get_level_values("instrument").unique()
    print(f"  Unique instruments with events: {len(all_instruments)}")
    print(f"  Event dates: {daily.index.get_level_values('date').nunique()}")

    # Point-in-time factors (only where events exist, else NaN)
    factors = {}

    # 1. evt_direction_confidence — sum of direction*confidence on that date
    factors["evt_direction_confidence"] = daily["dir_conf_sum"]

    # 2. evt_novelty_weighted
    factors["evt_novelty_weighted"] = daily["novelty_sum"]

    # 3. evt_source_quality_weighted
    factors["evt_source_quality_weighted"] = daily["sq_sum"]

    # 4 & 5. Rolling 5d features — need to expand per instrument
    #   For each instrument, build a time series then roll
    print("  Computing rolling 5d features...")
    count_5d_records = []
    posratio_5d_records = []

    for instr in all_instruments:
        if instr not in daily.index.get_level_values("instrument"):
            continue
        try:
            instr_data = daily.xs(instr, level="instrument")
        except KeyError:
            continue
        if instr_data.empty:
            continue

        # Reindex to all event dates for this instrument (only dates where universe exists)
        # Use the dates present in the events universe
        instr_dates = instr_data.index
        min_d, max_d = instr_dates.min(), instr_dates.max()

        # Create business day range for this instrument's active period
        bdays = pd.bdate_range(min_d, max_d)
        ts = instr_data.reindex(bdays).fillna(0)

        # Rolling 5 business day count and positive ratio
        roll_count = ts["n_events"].rolling(LOOKBACK_5D, min_periods=1).sum()
        roll_pos = ts["n_positive"].rolling(LOOKBACK_5D, min_periods=1).sum()
        roll_total = ts["n_events"].rolling(LOOKBACK_5D, min_periods=1).sum()
        roll_ratio = roll_pos / roll_total.replace(0, np.nan)

        # Only keep dates where at least 1 event existed in the 5d window
        mask = roll_count > 0
        for dt in roll_count[mask].index:
            count_5d_records.append((dt, instr, roll_count[dt]))
        for dt in roll_ratio.dropna().index:
            if mask[dt]:
                posratio_5d_records.append((dt, instr, roll_ratio[dt]))

    # Build Series from records
    if count_5d_records:
        idx = pd.MultiIndex.from_tuples(
            [(r[0], r[1]) for r in count_5d_records],
            names=["datetime", "instrument"],
        )
        factors["evt_event_count_5d"] = pd.Series(
            [r[2] for r in count_5d_records], index=idx, dtype=float
        )
    if posratio_5d_records:
        idx = pd.MultiIndex.from_tuples(
            [(r[0], r[1]) for r in posratio_5d_records],
            names=["datetime", "instrument"],
        )
        factors["evt_positive_ratio_5d"] = pd.Series(
            [r[2] for r in posratio_5d_records], index=idx, dtype=float
        )

    # Rename index for point-in-time factors
    for name in ["evt_direction_confidence", "evt_novelty_weighted", "evt_source_quality_weighted"]:
        s = factors[name]
        s.index.names = ["datetime", "instrument"]
        factors[name] = s

    return factors


def align_to_universe(factor: pd.Series, ret: pd.Series) -> pd.Series:
    """Expand factor to the full universe index, filling NaN for missing stocks.

    This preserves the sparse event nature — stocks without events get NaN.
    Only keep dates present in both factor and returns.
    """
    # Get overlapping dates
    factor_dates = factor.index.get_level_values("datetime").unique()
    ret_dates = ret.index.get_level_values("datetime").unique()
    common_dates = factor_dates.intersection(ret_dates)

    if len(common_dates) == 0:
        return pd.Series(dtype=float)

    # For each common date, get the full instrument set from returns
    # and the factor values (sparse)
    aligned = factor.reindex(ret.index)
    # Keep only common dates
    dt_level = aligned.index.get_level_values("datetime")
    mask = dt_level.isin(common_dates)
    return aligned[mask]


def main():
    print("=" * 70)
    print("Phase 4N — Event Surprise Features")
    print("=" * 70)

    # 1. Load events
    print("\n[1] Loading events from EventStore...")
    events = load_events()
    print(f"  Total events: {len(events)}")
    print(f"  Date range: {events['date'].min().date()} ~ {events['date'].max().date()}")
    print(f"  Stocks: {events['instrument'].nunique()}")
    print(f"  Event types: {events['event_type'].nunique()}")
    print(f"  Direction distribution: {events['direction'].value_counts().to_dict()}")

    # 2. Load forward returns
    print("\n[2] Loading forward returns...")
    ret = load_forward_returns()
    ret_dates = ret.index.get_level_values("datetime").unique()
    print(f"  Returns shape: {ret.shape}")
    print(f"  Returns date range: {ret_dates.min().date()} ~ {ret_dates.max().date()}")

    # 3. Build factors
    print("\n[3] Building event surprise factors...")
    factors = build_daily_event_features(events, ret_dates)
    print(f"  Built {len(factors)} factors")

    for name, s in factors.items():
        non_nan = s.notna().sum()
        print(f"    {name}: {non_nan} non-NaN values, "
              f"dates={s.index.get_level_values('datetime').nunique()}")

    # 4. Align and run tearsheets
    print("\n[4] Registering with Alpha Factory and running tearsheets...")
    factory = AlphaFactory()

    results = {}
    for factor_name, factor_series in factors.items():
        print(f"\n  --- {factor_name} ---")

        # Align factor to returns universe
        aligned = align_to_universe(factor_series, ret)
        non_nan = aligned.notna().sum()
        total = len(aligned)
        coverage = non_nan / total if total > 0 else 0
        print(f"  Aligned: {total} slots, {non_nan} non-NaN ({coverage:.1%} coverage)")

        if non_nan < 100:
            print(f"  SKIP: too few observations ({non_nan})")
            results[factor_name] = {"error": f"Too few observations: {non_nan}"}
            continue

        # Capture aligned series in closure
        def make_build_func(s):
            return lambda: s
        build_func = make_build_func(aligned)

        factory.register(
            name=factor_name,
            description=f"Event surprise factor: {factor_name}",
            build_func=build_func,
        )

        metrics = factory.run_tearsheet(factor_name, returns=ret)
        gate = factory.check_gate(factor_name)
        results[factor_name] = {**metrics, "gate_pass": gate["pass"], "gate_failures": gate["failures"]}

        # Print key metrics
        if "error" not in metrics:
            print(f"  RankIC:    {metrics['rank_ic_mean']:+.4f} (std={metrics['rank_ic_std']:.4f})")
            print(f"  RankICIR:  {metrics['rank_icir']:+.3f}")
            print(f"  IC pos%:   {metrics['rank_ic_pos_ratio']:.1%}")
            print(f"  Spread:    {metrics.get('spread_q1_q5', 'N/A')}")
            print(f"  Coverage:  {metrics['coverage']:.1%}")
            print(f"  Neg ctrl:  {metrics['negative_control_ic']:.4f}")
            print(f"  N days:    {metrics['n_days']}")
            print(f"  Gate:      {'PASS' if gate['pass'] else 'FAIL'}")
            if gate["failures"]:
                for f in gate["failures"]:
                    print(f"    - {f}")
        else:
            print(f"  ERROR: {metrics['error']}")

    # 5. Summary table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    summary = factory.summary_table()
    # Filter to only our event factors
    evt_names = list(factors.keys())
    if not summary.empty:
        mask = summary["name"].isin(evt_names)
        if mask.any():
            evt_summary = summary[mask].copy()
            display_cols = [
                "name", "rank_ic_mean", "rank_icir", "rank_ic_pos_ratio",
                "spread_q1_q5", "coverage", "negative_control_ic",
                "autocorr_1d", "n_days", "verdict",
            ]
            display_cols = [c for c in display_cols if c in evt_summary.columns]
            pd.set_option("display.width", 200)
            pd.set_option("display.max_columns", 20)
            pd.set_option("display.float_format", lambda x: f"{x:.4f}" if abs(x) < 10 else f"{x:.1f}")
            print(evt_summary[display_cols].to_string(index=False))

    # 6. Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    valid_results = {k: v for k, v in results.items() if "error" not in v}
    if valid_results:
        best = max(valid_results, key=lambda k: abs(valid_results[k].get("rank_ic_mean", 0)))
        best_m = valid_results[best]
        print(f"\nBest factor by |RankIC|: {best}")
        print(f"  RankIC = {best_m['rank_ic_mean']:+.4f}, ICIR = {best_m['rank_icir']:+.3f}")

        passing = [k for k, v in valid_results.items() if v.get("gate_pass", False)]
        if passing:
            print(f"\nFactors passing gate: {', '.join(passing)}")
        else:
            print("\nNo factors passed the gate (expected with ~20 days of data).")
            print("Low coverage from sparse event data is the main constraint.")
            print("Recommendation: accumulate 60+ days of events before re-evaluation.")
    else:
        print("No valid tearsheet results. Check event data coverage.")

    print("\nDone.")


if __name__ == "__main__":
    main()
