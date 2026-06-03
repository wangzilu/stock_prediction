"""Daily per-stock vol + ADV snapshot for the paper OMS cost model.

Purpose
=======

The paper OMS' sqrt_adv slippage path (backtest/cost_model.py) needs
per-fill daily_volatility (returns std) and adv (yuan-volume) to
compute realistic slippage. Without these inputs the CostModel falls
back to a fixed slippage_rate and the sqrt_adv wiring is dead code in
production paper. This module produces the per-stock dict that the
PaperOMS' fill loop looks up at each fill.

Output shape
============

    {qlib_code: {"vol": <float>, "adv": <float>}}

where:
  vol = trailing-`lookback`-day std of close-to-close returns
        (defaults to 20 days, matches `cost_vol_window` default in
        backtest/portfolio_backtest.py)
  adv = trailing-`lookback`-day mean of `$amount`
        (qlib's `amount` field is already yuan-volume for cn_data;
        no need to multiply by close)

Both values are point-in-time as of `asof_date`. Stocks without
enough lookback rows are EXCLUDED from the dict (callers see
"code missing" and the fill site falls back to bare slippage_rate).

Cache
=====

Snapshot is cached on disk under
`data/storage/paper_cost_inputs/{asof_date}.parquet` so the daily
paper run avoids rebuilding from raw qlib data on every restart.
"""
from __future__ import annotations

import logging
from datetime import date as _date_t
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "storage" / "paper_cost_inputs"


def build_vol_adv_snapshot(
    asof_date: str | _date_t,
    *,
    lookback_days: int = 20,
    universe: str | None = None,
    qlib_loader=None,
) -> dict:
    """Build the snapshot dict for `asof_date`.

    Args:
        asof_date: YYYY-MM-DD. Snapshot reflects state at end of that date.
        lookback_days: rolling window for vol + ADV.
        universe: optional qlib universe handle; None uses qlib's default.
        qlib_loader: optional callable
            `(start_date, end_date, universe) -> pd.DataFrame`
            returning a frame indexed by (date, code) with columns
            ['close', 'amount']. When None, uses
            qlib.data.D.features. The callable form exists so tests
            can inject a fixture frame without touching qlib state.

    Returns:
        dict {code: {"vol": float, "adv": float}}. Codes with fewer
        than `lookback_days` non-NaN return / amount samples are
        omitted (caller fallback path takes over).
    """
    if isinstance(asof_date, str):
        asof = pd.Timestamp(asof_date)
    else:
        asof = pd.Timestamp(asof_date)

    start = (asof - pd.Timedelta(days=lookback_days * 2 + 14)).date()
    end = asof.date()

    if qlib_loader is None:
        qlib_loader = _default_qlib_loader

    df = qlib_loader(start, end, universe)
    if df is None or df.empty:
        logger.warning("vol/adv snapshot: loader returned no data for %s", asof_date)
        return {}

    return _compute_snapshot_from_panel(df, lookback_days=lookback_days)


def _compute_snapshot_from_panel(
    panel: pd.DataFrame, *, lookback_days: int = 20,
) -> dict:
    """Pure-pandas snapshot computation.

    Expects `panel` indexed by (date, code) with columns at least
    {'close', 'amount'}. Splits into per-code series, computes
    trailing stats, returns the last row's value per code where
    enough samples exist.
    """
    if panel.empty:
        return {}
    # Defensive: ensure expected columns
    required = {"close", "amount"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(
            f"_compute_snapshot_from_panel: missing required columns {missing}"
        )

    # Make sure index is sorted for groupby + rolling
    panel = panel.sort_index()

    # Vol: per-code rolling std of pct_change of close
    closes = panel["close"].unstack(level=1)  # date × code
    rets = closes.pct_change()
    rolling_vol = rets.rolling(window=lookback_days, min_periods=lookback_days).std()

    # ADV: per-code rolling mean of amount
    amounts = panel["amount"].unstack(level=1)
    rolling_adv = amounts.rolling(window=lookback_days, min_periods=lookback_days).mean()

    # Take the last row that has at least one finite value per series
    if rolling_vol.empty or rolling_adv.empty:
        return {}
    last_vol = rolling_vol.iloc[-1]
    last_adv = rolling_adv.iloc[-1]

    out: dict = {}
    for code in last_vol.index:
        v = last_vol.get(code)
        a = last_adv.get(code)
        if pd.isna(v) or pd.isna(a):
            continue
        if not (a > 0):
            continue
        if not (v > 0):
            continue
        out[str(code)] = {"vol": float(v), "adv": float(a)}
    return out


def _default_qlib_loader(start_date, end_date, universe):
    """Default loader pulling close + amount from qlib's D.features API."""
    try:
        from qlib.data import D
    except ImportError:
        logger.warning("qlib not importable; vol/adv snapshot will be empty")
        return None

    # Use the qlib expression language for daily close + amount.
    instruments = universe if universe is not None else "all"
    fields = ["$close", "$volume * $close"]  # amount = volume * close as fallback
    try:
        # Prefer raw amount when available
        df = D.features(D.instruments(instruments), ["$close", "$amount"],
                          start_time=str(start_date), end_time=str(end_date),
                          freq="day")
        df.columns = ["close", "amount"]
    except Exception:
        df = D.features(D.instruments(instruments), fields,
                          start_time=str(start_date), end_time=str(end_date),
                          freq="day")
        df.columns = ["close", "amount"]
    # Qlib index is (instrument, datetime); we want (datetime, instrument).
    df = df.reorder_levels(["datetime", "instrument"]).sort_index()
    df.index = df.index.set_names(["date", "code"])
    return df


# -----------------------------------------------------------------------------
# Disk cache
# -----------------------------------------------------------------------------

def _cache_path(asof_date: str | _date_t) -> Path:
    if isinstance(asof_date, str):
        stem = asof_date
    else:
        stem = asof_date.strftime("%Y-%m-%d")
    return CACHE_DIR / f"{stem}.parquet"


def load_or_build_snapshot(
    asof_date: str | _date_t, *,
    lookback_days: int = 20,
    universe: str | None = None,
    qlib_loader=None,
    force_rebuild: bool = False,
) -> dict:
    """Cached wrapper for `build_vol_adv_snapshot`.

    Reads `data/storage/paper_cost_inputs/{asof_date}.parquet` if it
    exists; otherwise computes the snapshot and writes the cache.
    Returns the snapshot dict regardless of which path was taken.
    """
    path = _cache_path(asof_date)
    if path.exists() and not force_rebuild:
        try:
            df = pd.read_parquet(path)
            return _dict_from_cache_frame(df)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "vol/adv cache read failed at %s — rebuilding (%s)", path, e,
            )

    snapshot = build_vol_adv_snapshot(
        asof_date, lookback_days=lookback_days, universe=universe,
        qlib_loader=qlib_loader,
    )
    if snapshot:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _cache_frame_from_dict(snapshot).to_parquet(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("vol/adv cache write failed at %s: %s", path, e)
    return snapshot


def _cache_frame_from_dict(snapshot: dict) -> pd.DataFrame:
    rows = [{"code": k, "vol": v["vol"], "adv": v["adv"]}
            for k, v in snapshot.items()]
    return pd.DataFrame(rows)


def _dict_from_cache_frame(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    out: dict = {}
    for _, row in df.iterrows():
        try:
            out[str(row["code"])] = {
                "vol": float(row["vol"]),
                "adv": float(row["adv"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return out
