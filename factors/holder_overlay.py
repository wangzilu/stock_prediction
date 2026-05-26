"""Holder decrease overlay — annotates stock picks with shareholder change signal.

Positive holder_decrease_score means shareholders are decreasing (bullish).
Negative means shareholders are increasing (bearish).

Usage:
    from factors.holder_overlay import get_holder_decrease_scores
    scores = get_holder_decrease_scores(["SH600519", "SZ000858"])
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOLDER_PATH = PROJECT_ROOT / "data" / "storage" / "st_holder_number.parquet"


def get_holder_decrease_scores(
    stock_codes: list[str],
    date: str | None = None,
) -> dict[str, float]:
    """Return {stock_code: holder_decrease_score} for given stocks.

    Positive = shareholders decreasing (bullish signal).
    Returns empty dict if data unavailable.

    Parameters
    ----------
    stock_codes : list[str]
        Instrument codes, e.g. ["SH600519", "SZ000858"].
    date : str, optional
        Reference date (YYYY-MM-DD). Uses latest available if None.
    """
    if not HOLDER_PATH.exists():
        logger.warning("Holder number data not found: %s", HOLDER_PATH)
        return {}

    try:
        df = pd.read_parquet(HOLDER_PATH)
    except Exception as e:
        logger.error("Failed to load holder data: %s", e)
        return {}

    # Expect columns like 'holder_num' or similar; inspect what's available
    # Index should be (datetime, instrument) or have those as columns
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # Normalise column names
    date_col = None
    inst_col = None
    holder_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ("datetime", "date", "end_date"):
            date_col = c
        elif cl in ("instrument", "stock_code", "code", "qlib_code"):
            inst_col = c
        elif "holder" in cl and "num" in cl:
            holder_col = c

    if date_col is None or inst_col is None or holder_col is None:
        logger.warning(
            "Cannot identify columns in holder data. Columns: %s",
            df.columns.tolist(),
        )
        return {}

    # Filter to requested stocks
    df = df[df[inst_col].isin(stock_codes)].copy()
    if df.empty:
        return {}

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([inst_col, date_col])

    # If a reference date is given, filter to data up to that date
    if date is not None:
        ref = pd.Timestamp(date)
        df = df[df[date_col] <= ref]

    if df.empty:
        return {}

    # Compute pct_change per stock, take latest value
    # holder_decrease_score = -pct_change (positive when holders decrease)
    result = {}
    for code, grp in df.groupby(inst_col):
        if len(grp) < 2:
            continue
        grp = grp.sort_values(date_col)
        latest = grp[holder_col].iloc[-1]
        prev = grp[holder_col].iloc[-2]
        if prev == 0 or pd.isna(prev) or pd.isna(latest):
            continue
        pct_change = (latest - prev) / abs(prev)
        result[code] = round(-pct_change, 6)  # negative pct_change = bullish

    return result
