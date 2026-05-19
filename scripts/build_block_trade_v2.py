"""Build block_trade v2 factors: discount rate, decay, volume ratio.

The classic alpha signal is the DISCOUNT RATE: (block_price / close - 1).
A large discount means institutional sellers are dumping at lower prices.

Factors:
  bt_discount         -- mean (block_price / close - 1) per (stock, date)
  bt_discount_5d_avg  -- rolling 5-day average of bt_discount
  bt_has_recent       -- 1 if any block trade in last 5 trading days
  bt_recency_decay    -- bt_discount * exp(-age_days / 5) summed over recent trades
  bt_volume_ratio     -- block_trade_volume / rolling_20d_avg_volume
"""
import os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"


def main():
    from config.qlib_runtime import init_qlib
    from qlib.data import D

    init_qlib(str(DATA_DIR / "qlib_data" / "cn_data"))

    # ---------- 1. Load block trade data ----------
    logger.info("Loading st_block_trade.parquet...")
    bt = pd.read_parquet(str(DATA_DIR / "st_block_trade.parquet"))
    bt['trade_date'] = pd.to_datetime(bt['trade_date'], format='%Y%m%d', errors='coerce')
    bt['qlib_code'] = bt['qlib_code'].str.upper()
    bt['price'] = pd.to_numeric(bt['price'], errors='coerce')
    bt['vol'] = pd.to_numeric(bt['vol'], errors='coerce')
    bt = bt.dropna(subset=['trade_date', 'price', 'qlib_code'])
    logger.info(f"  {len(bt)} rows after dropna")

    # ---------- 2. Load close prices from Qlib ----------
    logger.info("Loading close prices from Qlib...")
    insts = sorted(bt['qlib_code'].unique().tolist())
    start_time = str(bt['trade_date'].min().date())
    end_time = str(bt['trade_date'].max().date())
    close_df = D.features(insts, ["$close"], start_time=start_time, end_time=end_time)
    close_df.columns = ["close"]
    # D.features returns MultiIndex (instrument, datetime); flatten properly
    close_flat = close_df.reset_index()
    idx_cols = [c for c in close_flat.columns if c != "close"]
    logger.info(f"  close_df index names: {idx_cols}")
    # Map: 'instrument' -> instrument, 'datetime' -> date
    rename_map = {}
    for c in idx_cols:
        if 'date' in c.lower() or 'time' in c.lower():
            rename_map[c] = "date"
        else:
            rename_map[c] = "instrument"
    close_flat = close_flat.rename(columns=rename_map)
    close_flat["instrument"] = close_flat["instrument"].astype(str).str.upper()
    close_flat["date"] = pd.to_datetime(close_flat["date"])
    logger.info(f"  close prices: {len(close_flat)} rows, {close_flat['instrument'].nunique()} stocks")

    # ---------- 3. Merge close to block trades ----------
    bt_with_close = bt.merge(
        close_flat,
        left_on=["qlib_code", "trade_date"],
        right_on=["instrument", "date"],
        how="left"
    )
    logger.info(f"  close match rate: {bt_with_close['close'].notna().mean():.1%}")

    # Compute per-trade discount
    bt_with_close['discount'] = bt_with_close['price'] / bt_with_close['close'] - 1
    bt_with_close = bt_with_close.dropna(subset=['discount'])
    logger.info(f"  {len(bt_with_close)} trades with valid discount")

    # ---------- 4. Aggregate per (stock, date) ----------
    agg = bt_with_close.groupby(['qlib_code', 'trade_date']).agg(
        bt_discount=('discount', 'mean'),
        bt_total_vol=('vol', 'sum'),
        bt_count=('discount', 'count'),
    ).reset_index()
    agg = agg.sort_values(['qlib_code', 'trade_date'])
    logger.info(f"  {len(agg)} (stock, date) pairs")

    # ---------- 5. Rolling / decay features ----------
    records = []
    n_stocks = agg['qlib_code'].nunique()
    for i, (stock, grp) in enumerate(agg.groupby('qlib_code')):
        if (i + 1) % 500 == 0:
            logger.info(f"  processing stock {i+1}/{n_stocks}")
        grp = grp.set_index('trade_date').sort_index()

        # Get this stock's Qlib trading dates
        stock_dates = close_flat.loc[close_flat['instrument'] == stock, 'date'].sort_values().values
        if len(stock_dates) == 0:
            continue

        # Build daily series aligned to trading calendar
        daily = pd.DataFrame(index=pd.DatetimeIndex(stock_dates),
                             columns=['bt_discount', 'bt_total_vol'])
        # Use reindex to align grp index to daily index
        common_idx = daily.index.intersection(grp.index)
        if len(common_idx) > 0:
            daily.loc[common_idx, 'bt_discount'] = grp.loc[common_idx, 'bt_discount'].values
            daily.loc[common_idx, 'bt_total_vol'] = grp.loc[common_idx, 'bt_total_vol'].values
        daily['bt_discount'] = pd.to_numeric(daily['bt_discount'], errors='coerce')
        daily['bt_total_vol'] = pd.to_numeric(daily['bt_total_vol'], errors='coerce')

        # bt_discount_5d_avg: rolling 5-day mean of discount (only counting non-NaN)
        daily['bt_discount_5d_avg'] = daily['bt_discount'].rolling(5, min_periods=1).mean()

        # bt_has_recent: 1 if any block trade in last 5 trading days
        daily['bt_has_recent'] = daily['bt_discount'].rolling(5, min_periods=1).count().clip(upper=1)

        # bt_recency_decay: sum of discount * exp(-age/5) over last 5 days
        decay_vals = np.full(len(daily), np.nan)
        disc_arr = daily['bt_discount'].values.astype(float)
        for j in range(len(daily)):
            total = 0.0
            has_any = False
            for lag in range(min(5, j + 1)):
                d = disc_arr[j - lag]
                if np.isfinite(d):
                    total += d * np.exp(-lag / 5.0)
                    has_any = True
            if has_any:
                decay_vals[j] = total
        daily['bt_recency_decay'] = decay_vals

        # Only keep rows where at least one feature is non-NaN
        mask = daily[['bt_discount_5d_avg', 'bt_has_recent', 'bt_recency_decay']].notna().any(axis=1)
        sub = daily.loc[mask].copy()
        sub['qlib_code'] = stock
        sub['date'] = sub.index
        records.append(sub)

    logger.info("Concatenating daily records...")
    result = pd.concat(records, ignore_index=True)
    logger.info(f"  {len(result)} rows before volume ratio")

    # ---------- 6. Volume ratio from Qlib ----------
    logger.info("Loading volume from Qlib for volume ratio...")
    vol_df = D.features(insts, ["$volume"], start_time=start_time, end_time=end_time)
    vol_df.columns = ["volume"]
    vol_flat = vol_df.reset_index()
    vidx_cols = [c for c in vol_flat.columns if c != "volume"]
    vrename = {}
    for c in vidx_cols:
        if 'date' in c.lower() or 'time' in c.lower():
            vrename[c] = "date"
        else:
            vrename[c] = "instrument"
    vol_flat = vol_flat.rename(columns=vrename)
    vol_flat["instrument"] = vol_flat["instrument"].astype(str).str.upper()
    vol_flat["date"] = pd.to_datetime(vol_flat["date"])

    # Compute rolling 20d avg volume per stock
    vol_flat = vol_flat.sort_values(["instrument", "date"])
    vol_flat["vol_20d_avg"] = vol_flat.groupby("instrument")["volume"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )

    result = result.merge(
        vol_flat[["instrument", "date", "vol_20d_avg"]],
        left_on=["qlib_code", "date"],
        right_on=["instrument", "date"],
        how="left"
    )
    result["bt_volume_ratio"] = result["bt_total_vol"] / result["vol_20d_avg"]
    result["bt_volume_ratio"] = result["bt_volume_ratio"].clip(upper=50)

    # ---------- 7. PIT safe: shift dates +1 BDay ----------
    logger.info("Applying PIT shift (+1 BDay)...")
    result['date'] = result['date'] + BDay(1)

    # ---------- 8. Select final columns and save ----------
    keep_cols = ['qlib_code', 'date', 'bt_discount', 'bt_discount_5d_avg',
                 'bt_has_recent', 'bt_recency_decay', 'bt_volume_ratio']
    out = result[keep_cols].copy()

    out_path = DATA_DIR / "block_trade_v2.parquet"
    out.to_parquet(str(out_path), index=False)
    logger.info(f"Saved to {out_path}")
    logger.info(f"  shape: {out.shape}")
    logger.info(f"  columns: {out.columns.tolist()}")
    for c in keep_cols[2:]:
        nonnull = out[c].notna().sum()
        avg = out[c].mean() if nonnull > 0 else float('nan')
        logger.info(f"  {c}: non-null={nonnull}, mean={avg:.6f}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
