"""Validate B'+C' event overlay via AlphaFactory gate pipeline.

This script wires the gated_event_score from build_event_overlay into the
standard AlphaFactory tearsheet + gate-check so the event overlay goes through
the same validation as any other candidate factor.

Design notes
------------
* Factor value = gated_event_score per (date, instrument).
  - For each date with events in the unified EventStore, load events and
    compute the gated impact per stock (noise filtered, unstable dampened).
  - Stocks *without* events on a given date receive NaN (not 0), so that
    coverage reflects actual event coverage, not artificially inflated to 100%.
* Forward returns = NEXT-DAY return (shifted by 1 trading day).
  - Events on day T are known at T close -> can only trade at T+1 open ->
    forward alpha is measured against T+1 close-to-close return.
  - This is the correct as-of / tradable evaluation.
* Coverage is expected to be low (~10-20%) because events only cover a
  subset of the stock universe on any given day. The gate uses event_coverage
  (fraction of non-NaN values) which correctly reflects sparse event data.

NOTE (2026-05-24): Legacy llm_events/ dual-track merge has been removed.
  All event data is now read exclusively from the unified EventStore.
  Run ``migrate_legacy_events()`` if legacy data has not been migrated yet.

Usage:
    python scripts/validate_event_overlay.py
"""
import json
import logging
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
FEATURE_CACHE = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
LLM_EVENTS_DIR = DATA_DIR / "llm_events"  # kept for deprecation check only
EVENTS_DIR = DATA_DIR / "events"

# Re-use gating rules from build_event_overlay
NOISE_TYPES = {"other", "routine_announcement", "reorganize"}
UNSTABLE_TYPES = {
    "earnings_negative", "industry_trend_positive",
    "product_launch", "analyst_upgrade",
}
UNSTABLE_WEIGHT = 0.2


# ---- helpers ---------------------------------------------------------------

def _load_gated_scores_llm(date_str: str) -> dict[str, float]:
    """DEPRECATED — legacy llm_events/ loader.

    Kept only for backward compatibility. Always returns empty dict now.
    Callers should use _load_gated_scores_unified() instead.
    """
    if LLM_EVENTS_DIR.exists() and any(LLM_EVENTS_DIR.glob("*.jsonl")):
        warnings.warn(
            "llm_events/ is deprecated. Run migrate_legacy_events() to move "
            "data into the unified EventStore, then delete llm_events/.",
            DeprecationWarning,
            stacklevel=2,
        )
    return {}


def _load_gated_scores_unified(date_str: str) -> dict[str, float]:
    """Load gated event scores from unified events/ store for one date.

    Uses the signal_date field when available to correctly attribute events
    to their actionable trading day.
    """
    from factors.event_store import EventStore

    store = EventStore()

    # Prefer query_by_signal_date if events have signal_date populated
    df = store.query_by_signal_date(date_str)
    if df.empty:
        # Fallback: plain date query for pre-migration data
        df = store.query(date_str, date_str)

    if df.empty:
        return {}

    stock_impacts: dict[str, list[float]] = defaultdict(list)
    for _, e in df.iterrows():
        etype = e.get("event_type", "other")
        code = e.get("stock_code", "")
        if not code:
            continue
        if etype in NOISE_TYPES:
            continue

        direction = int(e.get("direction", 0))
        confidence = float(e.get("confidence", 0.5))
        magnitude = float(e.get("magnitude", 0.5))
        impact = direction * confidence * magnitude

        if etype in UNSTABLE_TYPES:
            impact *= UNSTABLE_WEIGHT

        if impact != 0:
            stock_impacts[code].append(impact)

    return {code: float(np.mean(vals)) for code, vals in stock_impacts.items() if vals}


def _code6_to_qlib(code6: str) -> str:
    """Convert 6-digit stock code to qlib instrument format (sh/sz prefix)."""
    if code6.startswith(("6", "9")):
        return f"sh{code6}"
    return f"sz{code6}"


# ---- main build function ---------------------------------------------------

def build_event_overlay_factor() -> pd.Series:
    """Build the event overlay factor as a pd.Series.

    Returns a Series with (datetime, instrument) MultiIndex where:
      - datetime = pd.Timestamp of the trading date
      - instrument = qlib code like 'sh600519'
      - value = gated_event_score (NaN for stocks without events)

    Stocks WITHOUT events get NaN (not 0) so that coverage correctly
    reflects the fraction of stocks that actually have event data.
    """
    logger.info("Loading feature cache for universe...")
    cache = pd.read_parquet(FEATURE_CACHE, columns=["__pnl_return_1d"])

    # Collect available event dates from unified store only
    event_dates: set[str] = set()
    if LLM_EVENTS_DIR.exists() and any(LLM_EVENTS_DIR.glob("*.jsonl")):
        warnings.warn(
            "llm_events/ directory still exists. Run migrate_legacy_events() "
            "to move data into the unified EventStore, then delete llm_events/.",
            DeprecationWarning,
            stacklevel=2,
        )
    for d in EVENTS_DIR.glob("*.jsonl"):
        event_dates.add(d.stem)

    # Only keep dates that exist in the feature cache
    cache_dates = sorted(cache.index.get_level_values("datetime").unique())
    cache_date_strs = {d.strftime("%Y-%m-%d"): d for d in cache_dates}
    valid_dates = sorted(event_dates & set(cache_date_strs.keys()))
    logger.info(
        f"Event dates: {len(event_dates)}, "
        f"Cache dates: {len(cache_date_strs)}, "
        f"Overlap: {len(valid_dates)}"
    )

    if not valid_dates:
        raise RuntimeError("No overlapping dates between events and feature cache")

    # Build factor values — NaN for no-event stocks
    records: list[tuple] = []  # (datetime, instrument, score)

    for date_str in valid_dates:
        ts = cache_date_strs[date_str]

        # Load scores from unified store only (legacy llm_events/ deprecated)
        merged = _load_gated_scores_unified(date_str)

        # Get the full instrument universe for this date from the cache
        try:
            day_instruments = cache.loc[ts].index.tolist()
        except KeyError:
            continue

        n_events = 0
        for instr in day_instruments:
            code6 = instr[2:] if len(instr) > 2 else instr
            if code6 in merged:
                records.append((ts, instr, merged[code6]))
                n_events += 1
            else:
                records.append((ts, instr, np.nan))

        if n_events > 0:
            logger.info(f"  {date_str}: {n_events}/{len(day_instruments)} stocks with events")

    if not records:
        raise RuntimeError("No factor records built")

    idx = pd.MultiIndex.from_tuples(
        [(r[0], r[1]) for r in records],
        names=["datetime", "instrument"],
    )
    factor = pd.Series(
        [r[2] for r in records],
        index=idx,
        name="event_overlay_bpc",
        dtype=float,
    )

    n_with_events = factor.notna().sum()
    logger.info(
        f"Factor built: {len(factor)} obs, "
        f"{n_with_events} with events ({n_with_events / len(factor):.1%} coverage)"
    )
    return factor


# ---- main ------------------------------------------------------------------

def main():
    from tracker.alpha_factory import AlphaFactory

    # 1. Build the factor
    factor = build_event_overlay_factor()

    # 2. Load FORWARD returns (T+1) — events at T close, tradable at T+1
    logger.info("Loading forward returns (shifted by 1 trading day)...")
    cache = pd.read_parquet(FEATURE_CACHE, columns=["__pnl_return_1d"])
    returns_raw = cache["__pnl_return_1d"]

    # Shift returns: for each instrument, shift return back by 1 day
    # so that factor[T] is compared with return[T+1]
    factor_dates = sorted(factor.index.get_level_values("datetime").unique())

    # Build date mapping: T → T+1
    all_dates = sorted(returns_raw.index.get_level_values("datetime").unique())
    date_to_next = {}
    for i, d in enumerate(all_dates):
        if i + 1 < len(all_dates):
            date_to_next[d] = all_dates[i + 1]

    # Build forward return series aligned to factor dates
    forward_records = []
    for date in factor_dates:
        next_date = date_to_next.get(date)
        if next_date is None:
            continue
        try:
            next_day_returns = returns_raw.loc[next_date]
        except KeyError:
            continue
        for instr, ret in next_day_returns.items():
            forward_records.append((date, instr, ret))

    if not forward_records:
        logger.error("No forward returns built")
        return

    fwd_idx = pd.MultiIndex.from_tuples(
        [(r[0], r[1]) for r in forward_records],
        names=["datetime", "instrument"],
    )
    forward_returns = pd.Series(
        [r[2] for r in forward_records],
        index=fwd_idx,
        name="forward_return_1d",
        dtype=float,
    )

    logger.info(f"Forward returns: {len(forward_returns)} obs")

    # 3. Register with AlphaFactory
    #    Event factors are sparse — use event_coverage (non-NaN %) as true coverage
    factory = AlphaFactory()
    factory.register(
        name="event_overlay_bpc",
        description=(
            "B'+C' gated event overlay: LLM impact_1d with noise filter "
            "and unstable-bucket dampening (0.2x weight). "
            "Forward T+1 returns for tradable alpha evaluation. "
            "NaN for stocks without events (true sparse coverage)."
        ),
        build_func=build_event_overlay_factor,
    )

    # 4. Run tearsheet with forward returns
    logger.info("Running tearsheet with forward returns...")
    metrics = factory.run_tearsheet("event_overlay_bpc", returns=forward_returns)

    # Coverage from tearsheet is now the TRUE event coverage (non-NaN fraction)
    logger.info(f"  Event coverage (non-NaN): {metrics.get('coverage', 0):.1%}")

    # 5. Gate check
    gate_result = factory.check_gate("event_overlay_bpc")

    # 6. Print results
    print("\n" + "=" * 60)
    print("Event Overlay Factor Validation (FORWARD returns, true coverage)")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print(f"\n  Gate: {'PASS' if gate_result.get('pass') else 'FAIL'}")
    if gate_result.get("failures"):
        for f in gate_result["failures"]:
            print(f"    FAIL: {f}")
    if gate_result.get("warnings"):
        for w in gate_result["warnings"]:
            print(f"    WARN: {w}")
    print("=" * 60)


if __name__ == "__main__":
    main()
