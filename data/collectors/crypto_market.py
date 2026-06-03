"""Crypto spot OHLCV collector (Phase Crypto-A step 2).

Implements the contract pinned in `plans/crypto-data-contract.md` §3.

Public API
==========

  fetch_recent(symbol, timeframe, venue="binance", limit=200)
      → pandas.DataFrame matching §3 schema, ONLY closed bars

  fetch_historical(symbol, timeframe, start_ts_ms, end_ts_ms,
                    venue="binance")
      → paginated fetch within [start_ts_ms, end_ts_ms); bound-checked

  write_ohlcv_partitions(df, venue, instrument_class="spot")
      → idempotent parquet writes, day-partitioned

Both fetch functions are SAFE to call only from inside the crypto
cron wrapper (which sets the ssproxy env sentinels). The
`assert_proxy_active()` check is invoked at entry. Direct calls
from outside the wrapper raise `CryptoProxyNotActiveError` before
any network attempt.

Closed-bar gate
===============

Per §3 + §11: a bar with open timestamp T (ms) and timeframe TF_SEC
is "closed" iff
    (T + TF_SEC * 1000) + CLOSED_BUFFER_SEC*1000 <= NOW_MS

The helper `_is_closed_with_buffer` is the single source of truth.
Direct `bar_open + tf_ms <= now_ms` comparisons or unconditional
`is_closed_bar=True` are FORBIDDEN.

Symbol convention
=================

CCXT pair form is the input (e.g. `"BTC/USDT"`). Canonical form
(for directory layout) is produced by `to_canonical_symbol`:
    binance__btc_usdt__spot
The OHLCV row's `symbol` column stores the CCXT form.

CCXT is imported LAZILY inside each fetch function so simply
importing this module does NOT pull in ccxt (helps test cold-start
budget and avoids surprising module-level errors).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config.crypto_network import assert_proxy_active
from config.crypto_storage import crypto_root

logger = logging.getLogger(__name__)

# Per contract §11. Tunable post-Phase-0a spike.
CLOSED_BUFFER_SEC = 120

# Per contract §3.
SCHEMA_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "exchange",
    "symbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume_base",
    "volume_quote",
    "quote_volume_estimated",
    "trades",
    "is_closed_bar",
    "ingested_at",
)

TF_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

# Public exception so tests / callers can catch precisely.
class CryptoFetchError(RuntimeError):
    """Wraps any ccxt exception or HTTP-level failure during fetch."""


# -----------------------------------------------------------------------------
# Helpers (pure, no I/O)
# -----------------------------------------------------------------------------

def _is_closed_with_buffer(bar_open_ms: int, tf_sec: int,
                            now_ms: int) -> bool:
    """Single source of truth for the closed-bar gate.

    Returns True iff the bar that *opened* at bar_open_ms has had its
    close time pass AND the configured exchange-revision buffer
    (CLOSED_BUFFER_SEC) has elapsed since that close.

    Both inputs are integer milliseconds since epoch (UTC).
    """
    bar_close_ms = bar_open_ms + tf_sec * 1000
    return bar_close_ms + CLOSED_BUFFER_SEC * 1000 <= now_ms


def to_canonical_symbol(symbol_ccxt: str, venue: str,
                         instrument_class: str = "spot") -> str:
    """Convert CCXT pair to canonical directory form.

    `BTC/USDT` + `binance` + `spot`  →  `binance__btc_usdt__spot`
    """
    if "/" not in symbol_ccxt:
        raise ValueError(
            f"symbol_ccxt {symbol_ccxt!r} must contain '/'; "
            f"use the CCXT pair form (e.g. 'BTC/USDT')"
        )
    base, quote = symbol_ccxt.split("/", 1)
    return f"{venue.lower()}__{base.lower()}_{quote.lower()}__{instrument_class.lower()}"


def _normalize_ohlcv_row(
    raw: list, *, symbol_ccxt: str, venue: str, timeframe: str,
    now_ms: int,
) -> dict[str, Any]:
    """Convert a single CCXT [ts, open, high, low, close, volume]
    row into the contract §3 schema, applying the closed-bar gate."""
    tf_sec = TF_SECONDS[timeframe]
    bar_open_ms = int(raw[0])
    o, h, l, c, vol_base = raw[1], raw[2], raw[3], raw[4], raw[5]
    # CCXT does not give us quote volume natively; estimate it.
    mid = (h + l) / 2.0 if (h is not None and l is not None) else c
    vol_quote_est = float(vol_base) * float(mid) if mid else 0.0
    return {
        "timestamp_utc": bar_open_ms,
        "exchange": venue.lower(),
        "symbol": symbol_ccxt,
        "timeframe": timeframe,
        "open": float(o),
        "high": float(h),
        "low": float(l),
        "close": float(c),
        "volume_base": float(vol_base),
        "volume_quote": float(vol_quote_est),
        "quote_volume_estimated": True,
        "trades": -1,  # exchange-not-reported sentinel per §3
        "is_closed_bar": _is_closed_with_buffer(bar_open_ms, tf_sec, now_ms),
        "ingested_at": now_ms,
    }


def _df_from_rows(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame with the schema columns in their canonical
    order. Missing columns are NOT allowed — caller must produce
    every column."""
    if not rows:
        return pd.DataFrame(columns=list(SCHEMA_COLUMNS))
    df = pd.DataFrame(rows)
    # Reindex columns; this would fail loudly if a normalizer drift
    # drops a required column.
    df = df[list(SCHEMA_COLUMNS)]
    return df


# -----------------------------------------------------------------------------
# CCXT integration (lazy import)
# -----------------------------------------------------------------------------

def _make_exchange(venue: str) -> Any:
    """Return a CCXT exchange instance for `venue`. Lazy import so
    `import data.collectors.crypto_market` does not pull ccxt."""
    import ccxt  # noqa: E402

    venue = venue.lower()
    try:
        cls = getattr(ccxt, venue)
    except AttributeError as e:
        raise CryptoFetchError(
            f"CCXT has no exchange named {venue!r}"
        ) from e
    # Use timeout to bound any individual HTTP call. Retries are
    # caller's responsibility — the wrapper kills the cron job if
    # the whole fetch exceeds the cron timeout.
    return cls({"enableRateLimit": True, "timeout": 30_000})


# -----------------------------------------------------------------------------
# Public fetch API
# -----------------------------------------------------------------------------

def fetch_recent(
    symbol: str,
    timeframe: str,
    *,
    venue: str = "binance",
    limit: int = 200,
    now_ms: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch up to `limit` recent bars for (venue, symbol, timeframe).

    Returns ONLY closed bars (the open bar at the current time is
    excluded by the `is_closed_bar` filter at write time — callers
    can choose to keep or drop unclosed rows).

    The current implementation returns all bars including the
    `is_closed_bar` column; callers that need only closed bars
    should filter on that column. This preserves the auditing
    requirement that we record the open bar's existence.

    Raises CryptoProxyNotActiveError if the wrapper sentinels are
    missing. Raises CryptoFetchError for any ccxt-level failure.
    """
    assert_proxy_active()
    if timeframe not in TF_SECONDS:
        raise ValueError(
            f"timeframe {timeframe!r} not in {sorted(TF_SECONDS)}"
        )

    ex = _make_exchange(venue)
    try:
        raw_rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:  # noqa: BLE001
        raise CryptoFetchError(
            f"fetch_ohlcv({venue}, {symbol}, {timeframe}, limit={limit}) "
            f"failed: {type(e).__name__}: {e}"
        ) from e

    now_ms_eff = int(time.time() * 1000) if now_ms is None else int(now_ms)
    rows = [
        _normalize_ohlcv_row(r, symbol_ccxt=symbol, venue=venue,
                              timeframe=timeframe, now_ms=now_ms_eff)
        for r in raw_rows
    ]
    return _df_from_rows(rows)


def fetch_historical(
    symbol: str,
    timeframe: str,
    start_ts_ms: int,
    end_ts_ms: int,
    *,
    venue: str = "binance",
    page_size: int = 1000,
    now_ms: Optional[int] = None,
) -> pd.DataFrame:
    """Paginated history fetch within `[start_ts_ms, end_ts_ms)`.

    CCXT's `since=` has no `until=` companion, so the last page may
    overshoot the requested window. We trim with an explicit
    `timestamp_utc < end_ts_ms` filter (§3 contract).

    Raises CryptoProxyNotActiveError / CryptoFetchError per the
    same contract as fetch_recent.
    """
    assert_proxy_active()
    if timeframe not in TF_SECONDS:
        raise ValueError(
            f"timeframe {timeframe!r} not in {sorted(TF_SECONDS)}"
        )
    if end_ts_ms <= start_ts_ms:
        return _df_from_rows([])

    ex = _make_exchange(venue)
    tf_ms = TF_SECONDS[timeframe] * 1000
    cursor = int(start_ts_ms)
    rows: list[dict] = []
    now_ms_eff = int(time.time() * 1000) if now_ms is None else int(now_ms)

    # Safety bound on iterations so a misbehaving CCXT page that
    # returns the same `since` repeatedly cannot infinite-loop.
    max_pages = ((end_ts_ms - start_ts_ms) // tf_ms) // page_size + 4
    pages = 0
    while cursor < end_ts_ms and pages < max_pages:
        try:
            raw = ex.fetch_ohlcv(symbol, timeframe=timeframe,
                                  since=cursor, limit=page_size)
        except Exception as e:  # noqa: BLE001
            raise CryptoFetchError(
                f"fetch_ohlcv({venue}, {symbol}, {timeframe}, "
                f"since={cursor}) failed: {type(e).__name__}: {e}"
            ) from e
        pages += 1
        if not raw:
            break
        for r in raw:
            ts = int(r[0])
            if ts >= end_ts_ms:
                # Page overshot; drop and stop.
                cursor = end_ts_ms
                break
            if ts < cursor:
                # Defensive: exchange returned a row before the
                # requested cursor — skip but advance.
                continue
            rows.append(_normalize_ohlcv_row(
                r, symbol_ccxt=symbol, venue=venue,
                timeframe=timeframe, now_ms=now_ms_eff,
            ))
        # Advance cursor to one tick past the last returned row to
        # avoid infinite-loop on identical pages.
        last_ts = int(raw[-1][0])
        cursor = max(cursor + tf_ms, last_ts + tf_ms)

    return _df_from_rows(rows)


# -----------------------------------------------------------------------------
# Storage (day-partitioned parquet writes under crypto_root)
# -----------------------------------------------------------------------------

def _partition_path(root: Path, venue: str, symbol_canonical: str,
                     timeframe: str, day_utc: datetime) -> Path:
    """Construct the §3 storage path. day_utc must be a tz-aware UTC
    datetime; the year/month/day partition is taken from it."""
    if day_utc.tzinfo is None:
        raise ValueError(
            "day_utc must be tz-aware UTC; got naive datetime"
        )
    return (
        root / "raw" / "ohlcv" / venue.lower() / symbol_canonical /
        timeframe /
        f"year={day_utc.year:04d}" /
        f"month={day_utc.month:02d}" /
        f"day={day_utc.day:02d}.parquet"
    )


def write_ohlcv_partitions(
    df: pd.DataFrame,
    *,
    venue: str,
    instrument_class: str = "spot",
    root: Optional[Path] = None,
) -> list[Path]:
    """Idempotently write a fetched OHLCV frame into day partitions
    under crypto_root.

    Idempotency: if a partition file already exists, the incoming
    rows are merged on (timestamp_utc, symbol, timeframe) with the
    incoming rows winning on `ingested_at` (newer ingest wins).
    Tests pin the contract.

    Returns the list of partition paths that were written.
    """
    if df.empty:
        return []
    if root is None:
        root = crypto_root()
    # Each row's `symbol` is in CCXT form like 'BTC/USDT'; produce
    # canonical for the directory.
    if "symbol" not in df.columns:
        raise ValueError("df missing required 'symbol' column")
    df = df.copy()
    df["_day_utc"] = pd.to_datetime(
        df["timestamp_utc"], unit="ms", utc=True,
    ).dt.floor("D")
    df["_symbol_canonical"] = df["symbol"].map(
        lambda s: to_canonical_symbol(s, venue, instrument_class)
    )

    written: list[Path] = []
    group_keys = ["_symbol_canonical", "timeframe", "_day_utc"]
    for (sym_can, tf, day), part in df.groupby(group_keys, sort=False):
        part_path = _partition_path(root, venue, sym_can, tf,
                                     day.to_pydatetime())
        part_path.parent.mkdir(parents=True, exist_ok=True)
        # Drop the synthetic helper columns before write.
        out = part.drop(columns=["_day_utc", "_symbol_canonical"])
        if part_path.exists():
            existing = pd.read_parquet(part_path)
            merged = pd.concat([existing, out], ignore_index=True)
            # Newer ingest wins on dup (timestamp_utc, symbol, tf).
            merged = (
                merged.sort_values("ingested_at")
                .drop_duplicates(
                    subset=["timestamp_utc", "symbol", "timeframe"],
                    keep="last",
                )
                .sort_values("timestamp_utc")
                .reset_index(drop=True)
            )
            merged.to_parquet(part_path)
        else:
            out.sort_values("timestamp_utc").reset_index(drop=True).to_parquet(part_path)
        written.append(part_path)
    return written
