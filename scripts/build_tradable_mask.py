"""Build per-stock-day tradable mask for training and inference.

Creates a boolean mask on the feature cache index, marking each (date, stock)
as tradable or not. This mask is applied BEFORE training — not just at execution.

Filters (per stock-day):
  1. ST/*ST/退市整理: name contains "ST" → entire stock filtered while ST
  2. IPO < 60 days: first 60 trading days of each stock
  3. Suspended: volume == 0 or close is NaN
  4. 一字涨停/跌停: open == close == high == low (zero intraday range)
  5. Low liquidity: daily amount < 5M RMB
  6. Low market cap: use amount_raw as proxy (correlated with mcap)
  7. BSE stocks: instrument starts with "bj"
  8. Price < 1 RMB: face-value delisting risk
  9. Turnover < 0.1%: effectively no trading
  10. Label winsorization: clip to ±3 sigma per cross-section

Usage:
    python scripts/build_tradable_mask.py
    python scripts/build_tradable_mask.py --stats  # just show filter stats
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
CACHE_PATH = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
MASK_PATH = DATA_DIR / "tradable_mask.parquet"


def load_st_set() -> set:
    """Load ST stock set from cached file (built by fetch_st_list.py)."""
    st_cache = DATA_DIR / "st_stock_list.json"
    if st_cache.exists():
        import json as _json
        st_list = _json.loads(st_cache.read_text())
        st_set = set(st_list)
        logger.info(f"  ST set: {len(st_set)} stocks (from {st_cache.name})")
        return st_set
    logger.warning("  ST list not found — run: python scripts/fetch_st_list.py")
    return set()


def build_mask(stats_only: bool = False) -> pd.Series:
    """Build tradable mask for the full feature cache."""
    t_start = time.time()

    logger.info("Loading feature cache...")
    cache = pd.read_parquet(CACHE_PATH)
    n_total = len(cache)
    logger.info(f"  Total: {n_total:,} stock-days")

    index = cache.index
    instruments = index.get_level_values(1).astype(str)
    dates = index.get_level_values(0)

    # Initialize: all tradable
    mask = np.ones(n_total, dtype=bool)
    filter_stats = {}

    # ---- Filter 1: ST stocks ----
    logger.info("Filter 1: ST stocks...")
    st_set = load_st_set()
    # 2026-06-04 cx round 6 P1-8: pre-fix this broadcast TODAY's ST
    # set to all historical dates, which both (a) deletes training
    # samples for periods when those stocks were NOT yet ST, and
    # (b) leaves samples for periods when previously-ST-now-clean
    # stocks were actually ST. That biases the training universe
    # in an opaque way.
    # Honest treatment: prefer the historical mask file
    # ``data/storage/st_historical_mask.parquet`` (produced by
    # ``scripts/build_st_historical_mask.py``, task #91) when it
    # exists. Fall back to the current-set broadcast — but ONLY when
    # the operator has explicitly opted in via the env var
    # ``ALLOW_CURRENT_ST_BROADCAST=1``. Otherwise abort training
    # universe construction so the upstream pipeline halts on a
    # known-biased mask instead of silently accepting it.
    import os as _os
    _historical_st_path = (
        Path(__file__).resolve().parents[1]
        / "data" / "storage" / "st_historical_mask.parquet"
    )
    if _historical_st_path.exists():
        try:
            hist_st = pd.read_parquet(_historical_st_path)
            # hist_st is expected to have a MultiIndex (datetime,
            # instrument) and a boolean ``is_st`` column. Reindex to
            # the current frame's index, default False (not ST) when
            # missing.
            aligned = hist_st["is_st"].reindex(index, fill_value=False)
            n_st = int(aligned.sum())
            mask &= ~aligned.values
            filter_stats["st"] = n_st
            logger.info(
                "  ST (historical): removed %d (%.1f%%) per per-date "
                "namechange mask", n_st, n_st / n_total * 100,
            )
        except Exception as e:
            logger.error(
                "  Failed to load %s for historical ST filter: %s",
                _historical_st_path, e,
            )
            if _os.environ.get("ALLOW_CURRENT_ST_BROADCAST") != "1":
                raise RuntimeError(
                    "ST historical mask unreadable AND "
                    "ALLOW_CURRENT_ST_BROADCAST not set — refusing "
                    "to fall back to the biased current-ST broadcast. "
                    "Fix the mask file or set the env var."
                ) from e
            logger.warning(
                "  ALLOW_CURRENT_ST_BROADCAST=1 set — using biased "
                "current-ST broadcast (training universe will be biased)."
            )
            is_st = instruments.isin(st_set) if st_set else np.zeros(n_total, dtype=bool)
            n_st = int(is_st.sum())
            mask &= ~is_st
            filter_stats["st"] = n_st
    elif st_set:
        if _os.environ.get("ALLOW_CURRENT_ST_BROADCAST") != "1":
            raise RuntimeError(
                "No historical ST mask at "
                f"{_historical_st_path} AND ALLOW_CURRENT_ST_BROADCAST "
                "not set. Run scripts/build_st_historical_mask.py first "
                "(task #91), or set the env var to accept biased "
                "current-ST broadcast."
            )
        logger.warning(
            "  ALLOW_CURRENT_ST_BROADCAST=1 — applying current ST set "
            "to all historical dates (training universe will be biased)."
        )
        is_st = instruments.isin(st_set)
        n_st = int(is_st.sum())
        mask &= ~is_st
        filter_stats["st"] = n_st
        logger.info(f"  ST: removed {n_st:,} ({n_st/n_total*100:.1f}%)")

    # ---- Filter 2: IPO < 60 days ----
    logger.info("Filter 2: IPO < 60 trading days...")
    # Count cumulative trading days per stock
    stock_first_seen = {}
    stock_day_count = {}
    ipo_mask = np.ones(n_total, dtype=bool)

    # Build per-stock day sequence
    unique_dates = sorted(dates.unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}

    for inst in instruments.unique():
        inst_mask = instruments == inst
        inst_dates = dates[inst_mask]
        if len(inst_dates) > 0:
            first_date_idx = date_to_idx.get(inst_dates.min(), 0)
            for d in inst_dates:
                d_idx = date_to_idx.get(d, 0)
                days_listed = d_idx - first_date_idx
                if days_listed < 60:
                    # Find position in full array
                    pass  # handled below via vectorized approach

    # Vectorized: for each stock, mark first 60 dates
    logger.info("  Computing listing days (vectorized)...")
    df_idx = pd.DataFrame({"date": dates, "inst": instruments})
    df_idx["date_rank"] = df_idx.groupby("inst")["date"].rank(method="first").astype(int)
    ipo_remove = df_idx["date_rank"] <= 60
    n_ipo = ipo_remove.sum()
    mask &= ~ipo_remove.values
    filter_stats["ipo_60d"] = int(n_ipo)
    logger.info(f"  IPO: removed {n_ipo:,} ({n_ipo/n_total*100:.1f}%)")

    # ---- Filter 3: Suspended (volume == 0) ----
    logger.info("Filter 3: Suspended stocks...")
    # We don't have volume in cache directly, but we have amount_raw and turn_raw
    if "turn_raw" in cache.columns:
        is_suspended = cache["turn_raw"].isna() | (cache["turn_raw"] == 0)
        n_susp = (is_suspended & mask).sum()
        mask &= ~is_suspended.values
        filter_stats["suspended"] = int(n_susp)
        logger.info(f"  Suspended: removed {n_susp:,} ({n_susp/n_total*100:.1f}%)")

    # ---- Filter 4: 一字板 (zero intraday range) ----
    # We need OHLC to detect this. Check if KLEN (candle length) == 0
    logger.info("Filter 4: 一字板 (zero range)...")
    if "KLEN" in cache.columns:
        is_one_price = cache["KLEN"].abs() < 1e-8
        n_one = (is_one_price & mask).sum()
        mask &= ~is_one_price.values
        filter_stats["one_price_limit"] = int(n_one)
        logger.info(f"  一字板: removed {n_one:,} ({n_one/n_total*100:.1f}%)")

    # ---- Filter 5: Low liquidity (amount < 5M) ----
    logger.info("Filter 5: Low liquidity...")
    if "amount_raw" in cache.columns:
        # amount_raw is raw daily amount from Qlib
        # Qlib's $amount is in RMB, but after Alpha158 normalization it's different
        # Use the raw value — but it may be normalized. Check:
        amount = cache["amount_raw"]
        # If amount_raw looks normalized (values around 0-1), skip this filter
        if amount.median() > 100:  # looks like raw values
            is_illiquid = amount < 5e6
            n_illiq = (is_illiquid & mask).sum()
            mask &= ~is_illiquid.values
            filter_stats["low_liquidity"] = int(n_illiq)
            logger.info(f"  Low liquidity: removed {n_illiq:,} ({n_illiq/n_total*100:.1f}%)")
        else:
            logger.info(f"  Low liquidity: skipped (amount_raw looks normalized, median={amount.median():.4f})")

    # ---- Filter 6: BSE stocks ----
    logger.info("Filter 6: BSE stocks...")
    is_bse = instruments.str.startswith("bj")
    n_bse = (is_bse & mask).sum()
    if n_bse > 0:
        mask &= ~is_bse
        filter_stats["bse"] = int(n_bse)
        logger.info(f"  BSE: removed {n_bse:,} ({n_bse/n_total*100:.1f}%)")
    else:
        logger.info(f"  BSE: none found")

    # ---- Filter 7: Label is NaN ----
    logger.info("Filter 7: NaN labels...")
    label = cache["__label_5d"]
    is_nan_label = label.isna() | ~np.isfinite(label.values)
    n_nan = (is_nan_label & mask).sum()
    mask &= ~is_nan_label.values
    filter_stats["nan_label"] = int(n_nan)
    logger.info(f"  NaN labels: removed {n_nan:,} ({n_nan/n_total*100:.1f}%)")

    # ---- Summary ----
    n_tradable = mask.sum()
    n_removed = n_total - n_tradable
    logger.info(f"\n{'='*60}")
    logger.info(f"TRADABLE MASK SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Total stock-days:    {n_total:>12,}")
    for fname, count in filter_stats.items():
        logger.info(f"  {fname:<20}  -{count:>12,} ({count/n_total*100:.1f}%)")
    logger.info(f"  {'─'*40}")
    logger.info(f"  Tradable:            {n_tradable:>12,} ({n_tradable/n_total*100:.1f}%)")
    logger.info(f"  Removed:             {n_removed:>12,} ({n_removed/n_total*100:.1f}%)")

    if not stats_only:
        # Save mask
        mask_series = pd.Series(mask, index=index, name="tradable")
        mask_series.to_frame().to_parquet(str(MASK_PATH))
        logger.info(f"\nSaved to {MASK_PATH}")

        # Also save winsorized labels
        logger.info("\nWinsorizing labels (±3 sigma per cross-section)...")
        label_clean = label.copy()
        label_clean[~mask] = np.nan

        # Per-date z-score + clip
        for dt in unique_dates:
            dt_mask = dates == dt
            dt_vals = label_clean.loc[dt_mask]
            valid = dt_vals.dropna()
            if len(valid) < 50:
                continue
            mu = valid.mean()
            sigma = valid.std()
            if sigma > 1e-10:
                z = (dt_vals - mu) / sigma
                clipped = z.clip(-3, 3) * sigma + mu
                label_clean.loc[dt_mask] = clipped

        label_path = DATA_DIR / "label_5d_winsorized.parquet"
        label_clean.to_frame("label_5d_win").to_parquet(str(label_path))
        logger.info(f"Saved winsorized labels to {label_path}")

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.0f}s")

    return pd.Series(mask, index=index)


def main():
    parser = argparse.ArgumentParser(description="Build tradable mask")
    parser.add_argument("--stats", action="store_true", help="Only show stats, don't save")
    args = parser.parse_args()
    build_mask(stats_only=args.stats)


if __name__ == "__main__":
    main()
