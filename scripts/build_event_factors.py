"""Build unified event kernel factors for forecast (业绩预告) and top_inst (龙虎榜).

Sparse events need: exponential decay, frequency signal, recency, abnormality.
PIT safe: forecast uses ann_date, top_inst uses trade_date + 1 BDay.

Output: data/storage/event_factors_v2.parquet
"""
import os, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"

# ── Type-score mapping for forecast ──────────────────────────────────
TYPE_SCORE = {
    '预增': 1.0, '略增': 1.0, '扭亏': 1.0, '续盈': 0.5,
    '预减': -1.0, '略减': -1.0, '首亏': -1.0, '续亏': -0.5,
    '不确定': 0.0,
}


def _compute_event_kernel_vectorized(
    cache_index: pd.MultiIndex,
    events_df: pd.DataFrame,
    event_date_col: str,
    signal_cols: dict,
    half_life: float,
    max_age: int,
    count_window: int,
    recent_window: int,
    prefix: str,
) -> pd.DataFrame:
    """Vectorized event kernel computation per (stock, date).

    For each stock, uses searchsorted to find relevant events for each query
    date, then computes decay/count/recent factors using numpy vectorization.

    Args:
        events_df: must have columns [qlib_code, event_date_col] + signal column(s)
        signal_cols: dict of {output_col_name: source_col_name} for decay signals
        half_life: exponential decay half-life in days
        max_age: maximum event age in days for decay
        count_window: window in days for frequency count
        recent_window: window in days for has_recent flag
        prefix: column name prefix
    """
    train_dates = cache_index.get_level_values(0)
    train_insts = cache_index.get_level_values(1).astype(str).str.upper()
    n = len(cache_index)

    # Pre-allocate result arrays
    result_arrays = {}
    for out_col in signal_cols:
        result_arrays[f"{prefix}_{out_col}_decayed"] = np.zeros(n, dtype=np.float64)
    result_arrays[f"{prefix}_has_recent_{recent_window}d"] = np.zeros(n, dtype=np.float64)
    result_arrays[f"{prefix}_frequency_{count_window}d"] = np.zeros(n, dtype=np.float64)

    inst_arr = train_insts.values if hasattr(train_insts, 'values') else np.array(train_insts)
    date_arr = train_dates.values.astype('datetime64[D]')  # day precision for age calc

    # Group training index by stock
    inst_series = pd.Series(np.arange(n), index=inst_arr)
    inst_groups = inst_series.groupby(inst_series.index)

    # Group events by stock
    events_df = events_df.copy()
    events_df['qlib_code'] = events_df['qlib_code'].str.upper()
    events_df[event_date_col] = pd.to_datetime(events_df[event_date_col], errors='coerce')
    events_df = events_df.dropna(subset=[event_date_col])
    events_df = events_df.sort_values(['qlib_code', event_date_col])
    stocks_in_ev = set(events_df['qlib_code'].unique())
    ev_by_stock = events_df.groupby('qlib_code')

    processed = 0
    t0 = time.time()

    for stock, pos_idx in inst_groups:
        if stock not in stocks_in_ev:
            continue

        positions = pos_idx.values
        query_dates_d = date_arr[positions]  # datetime64[D]

        stock_events = ev_by_stock.get_group(stock)
        ev_dates_d = stock_events[event_date_col].values.astype('datetime64[D]')

        # Get unique query dates and build mapping back to positions
        unique_qd, inverse = np.unique(query_dates_d, return_inverse=True)
        n_unique = len(unique_qd)

        # For each unique query date, compute factors
        # Use searchsorted to find the range of events within [qd - max_window, qd]
        max_window = max(max_age, count_window)

        # Pre-extract signal values
        sig_values = {}
        for out_col, src_col in signal_cols.items():
            sig_values[out_col] = stock_events[src_col].values

        for j in range(n_unique):
            qd = unique_qd[j]
            # Age of each event from this query date
            age_days = (qd - ev_dates_d).astype(float)  # timedelta64[D] -> float days

            # Events that occurred on or before query date
            occurred = age_days >= 0

            # Count within count_window
            in_count = occurred & (age_days <= count_window)
            count_val = in_count.sum()

            # Recent flag
            in_recent = occurred & (age_days <= recent_window)
            recent_val = 1.0 if in_recent.any() else 0.0

            # Decayed signals within max_age
            in_decay = occurred & (age_days <= max_age)

            # Map back to all positions with this unique query date
            mask = inverse == j
            target_positions = positions[mask]

            result_arrays[f"{prefix}_has_recent_{recent_window}d"][target_positions] = recent_val
            result_arrays[f"{prefix}_frequency_{count_window}d"][target_positions] = count_val

            if in_decay.any():
                decay = np.exp(-age_days[in_decay] / half_life)
                for out_col in signal_cols:
                    decayed_sum = np.sum(sig_values[out_col][in_decay] * decay)
                    result_arrays[f"{prefix}_{out_col}_decayed"][target_positions] = decayed_sum

        processed += 1
        if processed % 500 == 0:
            elapsed = time.time() - t0
            logger.info(f"  {prefix}: {processed} stocks ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    logger.info(f"{prefix} done: {processed} stocks in {elapsed:.1f}s")

    return pd.DataFrame(result_arrays, index=cache_index)


def build_forecast_events(cache_index: pd.MultiIndex) -> pd.DataFrame:
    """Build forecast event kernel factors.

    PIT: uses ann_date (announcement date).
    half_life=30d, max_age=120d.
    """
    path = DATA_DIR / "st_forecast.parquet"
    if not path.exists():
        logger.warning("st_forecast.parquet not found")
        return pd.DataFrame(index=cache_index)

    df = pd.read_parquet(str(path))
    df['ann_date'] = pd.to_datetime(df['ann_date'], format='%Y%m%d', errors='coerce')
    df['qlib_code'] = df['qlib_code'].str.upper()
    df['type_score'] = df['type'].map(TYPE_SCORE).fillna(0.0)
    df['p_change_min'] = pd.to_numeric(df['p_change_min'], errors='coerce')
    df['p_change_max'] = pd.to_numeric(df['p_change_max'], errors='coerce')
    df['magnitude'] = (df['p_change_min'].fillna(0) + df['p_change_max'].fillna(0)) / 2.0
    logger.info(f"Forecast events: {len(df)} rows, {df['qlib_code'].nunique()} stocks")
    logger.info(f"  Type distribution: {df['type'].value_counts().to_dict()}")

    return _compute_event_kernel_vectorized(
        cache_index=cache_index,
        events_df=df,
        event_date_col='ann_date',
        signal_cols={'signal': 'type_score', 'magnitude': 'magnitude'},
        half_life=30,
        max_age=120,
        count_window=180,
        recent_window=90,
        prefix='fc',
    )


def build_topinst_events(cache_index: pd.MultiIndex) -> pd.DataFrame:
    """Build top_inst (龙虎榜) event kernel factors.

    PIT: uses trade_date + 1 BDay.
    half_life=5d, max_age=20d.
    """
    path = DATA_DIR / "st_top_inst.parquet"
    if not path.exists():
        logger.warning("st_top_inst.parquet not found")
        return pd.DataFrame(index=cache_index)

    df = pd.read_parquet(str(path))
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d', errors='coerce')
    df['qlib_code'] = df['qlib_code'].str.upper()
    df['net_buy'] = pd.to_numeric(df['net_buy'], errors='coerce').fillna(0)

    # PIT safety: data available next business day
    df['avail_date'] = df['trade_date'] + pd.tseries.offsets.BDay(1)

    # Aggregate per (stock, avail_date): sum net_buy across seats, count seats
    agg = df.groupby(['qlib_code', 'avail_date']).agg(
        net_buy_sum=('net_buy', 'sum'),
        seat_count=('net_buy', 'count'),
    ).reset_index()

    # Direction: sign of aggregated net_buy (will be used as signal)
    agg['direction'] = np.sign(agg['net_buy_sum'])

    logger.info(f"Top_inst events: {len(agg)} stock-date pairs, "
                f"{agg['qlib_code'].nunique()} stocks")

    # Scale net_buy to reasonable range (it's in yuan, can be billions)
    # Use log-scale: sign(x) * log10(1 + |x|/1e6) to compress
    agg['net_buy_log'] = np.sign(agg['net_buy_sum']) * np.log10(
        1 + np.abs(agg['net_buy_sum']) / 1e6)

    return _compute_event_kernel_vectorized(
        cache_index=cache_index,
        events_df=agg,
        event_date_col='avail_date',
        signal_cols={'net_buy': 'net_buy_log', 'direction': 'direction'},
        half_life=5,
        max_age=20,
        count_window=30,
        recent_window=5,
        prefix='ti',
    )


def main():
    logger.info("Loading feature cache index...")
    cache = pd.read_parquet(str(DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"),
                            columns=['__label_5d'])  # only need index
    cache_index = cache.index
    logger.info(f"Cache index: {len(cache_index)} rows, "
                f"{cache_index.get_level_values(0).nunique()} dates")

    # Build both event types
    fc_factors = build_forecast_events(cache_index)
    ti_factors = build_topinst_events(cache_index)

    # Combine
    result = pd.concat([fc_factors, ti_factors], axis=1)
    logger.info(f"Combined event factors: {result.shape}")

    # Save
    out_path = DATA_DIR / "event_factors_v2.parquet"
    result.to_parquet(str(out_path))
    logger.info(f"Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Summary stats
    for col in result.columns:
        nz = (result[col] != 0).sum()
        logger.info(f"  {col}: non-zero={nz} ({nz/len(result):.2%}), "
                    f"mean={result[col].mean():.6f}, std={result[col].std():.6f}")


if __name__ == "__main__":
    main()
