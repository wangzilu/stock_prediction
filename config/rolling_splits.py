"""Unified rolling split configurations for train/valid/test windows.

Centralises split definitions so that train_lgb, phase4_rolling_gate,
and any future scripts use the same date boundaries.
"""

from __future__ import annotations

import datetime as dt
from typing import List, Optional


def generate_splits(
    n_splits: int,
    train_days: int,
    valid_days: int,
    test_days: int,
    end_date: Optional[str] = None,
    calendar: Optional[List[dt.date]] = None,
) -> list[dict]:
    """Generate rolling train/valid/test splits walking backward from *end_date*.

    Parameters
    ----------
    n_splits : int
        Number of non-overlapping test windows.
    train_days, valid_days, test_days : int
        Lengths (in trading/business days) for each segment.
    end_date : str or None
        Last calendar date to consider (ISO format, e.g. "2026-05-23").
        Defaults to today.
    calendar : list[datetime.date] or None
        Ordered list of trading dates.  If provided, indexing uses this
        calendar; otherwise pandas ``BDay`` offsets are used.

    Returns
    -------
    list[dict]
        Each element: {"split_id", "train_start", "train_end",
        "valid_start", "valid_end", "test_start", "test_end"} with
        string dates in ISO format.
    """
    if end_date is None:
        end_dt = dt.date.today()
    else:
        end_dt = dt.date.fromisoformat(str(end_date)[:10])

    if calendar is not None:
        return _generate_from_calendar(
            n_splits, train_days, valid_days, test_days, end_dt, calendar
        )
    return _generate_from_bday(
        n_splits, train_days, valid_days, test_days, end_dt
    )


# ── internal helpers ─────────────────────────────────────────────────

def _generate_from_calendar(
    n_splits: int,
    train_days: int,
    valid_days: int,
    test_days: int,
    end_dt: dt.date,
    calendar: list,
) -> list[dict]:
    import pandas as pd

    cal = sorted(pd.Timestamp(d).date() for d in calendar)
    # Find the last calendar date <= end_dt
    last_idx = None
    for i in range(len(cal) - 1, -1, -1):
        if cal[i] <= end_dt:
            last_idx = i
            break
    if last_idx is None:
        raise ValueError(f"end_date {end_dt} is before calendar start {cal[0]}")

    splits = []
    for s in range(n_splits):
        test_end_idx = last_idx - s * test_days
        test_start_idx = test_end_idx - test_days + 1
        valid_end_idx = test_start_idx - 1
        valid_start_idx = valid_end_idx - valid_days + 1
        train_end_idx = valid_start_idx - 1
        train_start_idx = train_end_idx - train_days + 1

        if train_start_idx < 0:
            break

        splits.append({
            "split_id": s,
            "train_start": cal[train_start_idx].isoformat(),
            "train_end": cal[train_end_idx].isoformat(),
            "valid_start": cal[valid_start_idx].isoformat(),
            "valid_end": cal[valid_end_idx].isoformat(),
            "test_start": cal[test_start_idx].isoformat(),
            "test_end": cal[test_end_idx].isoformat(),
        })

    return splits


def _generate_from_bday(
    n_splits: int,
    train_days: int,
    valid_days: int,
    test_days: int,
    end_dt: dt.date,
) -> list[dict]:
    import pandas as pd

    end_ts = pd.Timestamp(end_dt)
    splits = []

    for s in range(n_splits):
        test_end = end_ts - pd.tseries.offsets.BDay(s * test_days)
        test_start = test_end - pd.tseries.offsets.BDay(test_days - 1)
        valid_end = test_start - pd.tseries.offsets.BDay(1)
        valid_start = valid_end - pd.tseries.offsets.BDay(valid_days - 1)
        train_end = valid_start - pd.tseries.offsets.BDay(1)
        train_start = train_end - pd.tseries.offsets.BDay(train_days - 1)

        splits.append({
            "split_id": s,
            "train_start": train_start.date().isoformat(),
            "train_end": train_end.date().isoformat(),
            "valid_start": valid_start.date().isoformat(),
            "valid_end": valid_end.date().isoformat(),
            "test_start": test_start.date().isoformat(),
            "test_end": test_end.date().isoformat(),
        })

    return splits


# ── standard presets ─────────────────────────────────────────────────

STANDARD_24SPLIT = dict(n_splits=24, train_days=480, valid_days=60, test_days=60)
STANDARD_12SPLIT = dict(n_splits=12, train_days=480, valid_days=60, test_days=120)
FAST_6SPLIT = dict(n_splits=6, train_days=480, valid_days=60, test_days=240)

_PRESETS = {
    "24split": STANDARD_24SPLIT,
    "12split": STANDARD_12SPLIT,
    "6split": FAST_6SPLIT,
}


def get_standard_splits(
    preset: str = "24split",
    end_date: Optional[str] = None,
    calendar: Optional[list] = None,
) -> list[dict]:
    """Return splits for a named preset.

    Parameters
    ----------
    preset : str
        One of "24split", "12split", "6split".
    end_date, calendar :
        Forwarded to :func:`generate_splits`.
    """
    if preset not in _PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from {list(_PRESETS)}")
    return generate_splits(**_PRESETS[preset], end_date=end_date, calendar=calendar)
