"""Crypto perpetual derivatives collector (Phase Crypto-A step 3).

Implements two contract sections from `plans/crypto-data-contract.md`:

  §4 Funding Rate Schema
    - fetch_funding_recent / fetch_funding_history
    - write_funding_partitions  (month-partitioned parquet)

  §5 Open Interest Schema
    - fetch_open_interest_recent / fetch_open_interest_history
    - write_oi_partitions  (month-partitioned parquet)

Both fetch functions call `assert_proxy_active()` first, then a
helper that lazy-imports ccxt. Direct calls outside the cron
wrapper raise `CryptoProxyNotActiveError` before any network.

Symbol convention
=================

CCXT perp form: `BTC/USDT:USDT` (base/quote:settle).
Canonical directory form: `binance__btc_usdt__perp`.
The `symbol` column inside each row stores the CCXT perp form.

§4 pagination contract (verbatim)
=================================
> `fetch_funding_history(symbol, start, end)` MUST use cursor advance.
> Single-call `limit=1000` is insufficient for 1-year windows (1095 events).
> Defensive: break on `last_ts <= cursor` to prevent infinite loop on
> malformed responses.

Both helpers below honor this contract.

CCXT lazy import
================
Same pattern as crypto_market.py — ccxt is imported inside each
helper that needs it. Module-level import of this collector does
NOT pull in ccxt.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config.crypto_network import assert_proxy_active
from config.crypto_storage import crypto_root
from data.collectors.crypto_market import (
    CryptoFetchError,
    _make_exchange,
)

logger = logging.getLogger(__name__)

# Schemas (per data-contract §4 + §5)
FUNDING_SCHEMA_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "exchange",
    "symbol",
    "funding_rate",
    "next_funding_ts",
    "mark_price",
    "index_price",
    "ingested_at",
)

OI_SCHEMA_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "exchange",
    "symbol",
    "open_interest",
    "oi_quote",
    "long_short_ratio",
    "ingested_at",
)

# Open-interest sample grid — contract §5 says "aligned to 15-minute grid"
OI_SAMPLE_GRID_MS: int = 15 * 60 * 1000


# -----------------------------------------------------------------------------
# Symbol helpers
# -----------------------------------------------------------------------------

def parse_perp_symbol(symbol_perp_ccxt: str) -> tuple[str, str, str]:
    """Decompose a CCXT perp symbol into (base, quote, settle).

    `BTC/USDT:USDT` → ('BTC', 'USDT', 'USDT')
    `ETH/USD:ETH`   → ('ETH', 'USD', 'ETH')   # inverse perp form

    Raises ValueError on shapes we don't recognise so a typo upstream
    doesn't silently mis-bucket data.
    """
    if "/" not in symbol_perp_ccxt or ":" not in symbol_perp_ccxt:
        raise ValueError(
            f"perp symbol {symbol_perp_ccxt!r} not in CCXT perp form "
            "(expected 'BASE/QUOTE:SETTLE')"
        )
    pair, settle = symbol_perp_ccxt.split(":", 1)
    base, quote = pair.split("/", 1)
    if not base or not quote or not settle:
        raise ValueError(
            f"perp symbol {symbol_perp_ccxt!r} has empty component"
        )
    return base, quote, settle


def to_canonical_perp_symbol(symbol_perp_ccxt: str, venue: str) -> str:
    """Canonical directory form for a perp."""
    base, quote, _settle = parse_perp_symbol(symbol_perp_ccxt)
    return f"{venue.lower()}__{base.lower()}_{quote.lower()}__perp"


# -----------------------------------------------------------------------------
# Row normalizers
# -----------------------------------------------------------------------------

def _normalize_funding_row(
    raw: dict, *, symbol_perp_ccxt: str, venue: str, now_ms: int,
) -> dict[str, Any]:
    """Convert one CCXT fetch_funding_rate_history entry to §4 schema.

    Field name conventions (CCXT 4.x):
      raw['timestamp']    : int ms, funding event UTC
      raw['fundingRate']  : float, signed
      raw['nextFundingTime'] : optional int ms
      raw['markPrice']    : optional float
      raw['indexPrice']   : optional float
    """
    ts = int(raw.get("timestamp") or raw.get("fundingTimestamp") or 0)
    rate = raw.get("fundingRate")
    if rate is None:
        raise ValueError(
            f"funding row missing fundingRate: {raw!r}"
        )
    return {
        "timestamp_utc": ts,
        "exchange": venue.lower(),
        "symbol": symbol_perp_ccxt,
        "funding_rate": float(rate),
        "next_funding_ts": int(raw["nextFundingTime"])
            if raw.get("nextFundingTime") is not None else None,
        "mark_price": float(raw["markPrice"])
            if raw.get("markPrice") is not None else None,
        "index_price": float(raw["indexPrice"])
            if raw.get("indexPrice") is not None else None,
        "ingested_at": now_ms,
    }


def _normalize_oi_row(
    raw: dict, *, symbol_perp_ccxt: str, venue: str, now_ms: int,
) -> dict[str, Any]:
    """Convert one CCXT fetch_open_interest entry to §5 schema.

    CCXT 4.x convention:
      raw['timestamp']      : int ms
      raw['openInterest']   : float, base currency
      raw['openInterestValue'] / raw['openInterestAmount'] : optional quote
      info.longShortRatio   : sometimes available
    """
    ts = int(raw.get("timestamp") or 0)
    oi_base = raw.get("openInterest")
    if oi_base is None:
        raise ValueError(
            f"OI row missing openInterest: {raw!r}"
        )
    oi_quote = (
        raw.get("openInterestValue")
        or raw.get("openInterestAmount")
    )
    # long_short_ratio sometimes lives under 'info'
    lsr = None
    info = raw.get("info") or {}
    if isinstance(info, dict):
        for key in ("longShortRatio", "long_short_ratio", "longshortRatio"):
            if key in info:
                try:
                    lsr = float(info[key])
                except (TypeError, ValueError):
                    pass
                break
    return {
        "timestamp_utc": ts,
        "exchange": venue.lower(),
        "symbol": symbol_perp_ccxt,
        "open_interest": float(oi_base),
        "oi_quote": float(oi_quote) if oi_quote is not None else None,
        "long_short_ratio": lsr,
        "ingested_at": now_ms,
    }


def _frame(rows: list[dict], cols: tuple[str, ...]) -> pd.DataFrame:
    """Build a DataFrame in the canonical column order. Empty → empty
    frame with the right columns (so callers can rely on .columns)."""
    if not rows:
        return pd.DataFrame(columns=list(cols))
    df = pd.DataFrame(rows)
    return df[list(cols)]


# -----------------------------------------------------------------------------
# Public fetch API — funding
# -----------------------------------------------------------------------------

def fetch_funding_recent(
    symbol: str,
    *,
    venue: str = "binance",
    limit: int = 200,
    now_ms: Optional[int] = None,
) -> pd.DataFrame:
    """Recent funding events for one perp. Useful for daily cron polls."""
    assert_proxy_active()
    ex = _make_exchange(venue)
    try:
        raw = ex.fetch_funding_rate_history(symbol, limit=limit)
    except Exception as e:  # noqa: BLE001
        raise CryptoFetchError(
            f"fetch_funding_rate_history({venue}, {symbol}, limit={limit})"
            f" failed: {type(e).__name__}: {e}"
        ) from e
    now_ms_eff = int(time.time() * 1000) if now_ms is None else int(now_ms)
    rows = [
        _normalize_funding_row(r, symbol_perp_ccxt=symbol, venue=venue,
                                now_ms=now_ms_eff)
        for r in raw
    ]
    return _frame(rows, FUNDING_SCHEMA_COLUMNS)


def fetch_funding_history(
    symbol: str,
    start_ts_ms: int,
    end_ts_ms: int,
    *,
    venue: str = "binance",
    page_size: int = 1000,
    now_ms: Optional[int] = None,
) -> pd.DataFrame:
    """Paginated funding history in [start_ts_ms, end_ts_ms).

    Honours data-contract §4 pagination contract: cursor advance,
    defensive break on `last_ts <= cursor` to avoid infinite loops
    on malformed pages.
    """
    assert_proxy_active()
    if end_ts_ms <= start_ts_ms:
        return _frame([], FUNDING_SCHEMA_COLUMNS)

    ex = _make_exchange(venue)
    cursor = int(start_ts_ms)
    rows: list[dict] = []
    now_ms_eff = int(time.time() * 1000) if now_ms is None else int(now_ms)

    # Pages cap as a hard safety bound — funding is ~3 events/day so
    # a 5-year window is ~5475 events. With page_size 1000 that's 6
    # pages. 50 is generous.
    max_pages = 50
    pages = 0

    while cursor < end_ts_ms and pages < max_pages:
        try:
            raw = ex.fetch_funding_rate_history(
                symbol, since=cursor, limit=page_size,
            )
        except Exception as e:  # noqa: BLE001
            raise CryptoFetchError(
                f"fetch_funding_rate_history({venue}, {symbol}, "
                f"since={cursor}) failed: {type(e).__name__}: {e}"
            ) from e
        pages += 1
        if not raw:
            break
        last_ts = int(raw[-1].get("timestamp") or 0)
        for item in raw:
            ts = int(item.get("timestamp") or 0)
            if ts >= end_ts_ms:
                cursor = end_ts_ms
                break
            if ts < cursor:
                continue
            rows.append(_normalize_funding_row(
                item, symbol_perp_ccxt=symbol, venue=venue,
                now_ms=now_ms_eff,
            ))
        # Contract §4 defensive: if exchange didn't advance the cursor,
        # break to avoid infinite loop.
        if last_ts <= cursor:
            logger.warning(
                "funding history page did not advance cursor "
                "(last_ts=%s cursor=%s); breaking to avoid loop",
                last_ts, cursor,
            )
            break
        cursor = last_ts + 1  # advance just past the last seen ts

    return _frame(rows, FUNDING_SCHEMA_COLUMNS)


# -----------------------------------------------------------------------------
# Public fetch API — open interest
# -----------------------------------------------------------------------------

def fetch_open_interest_recent(
    symbol: str,
    *,
    venue: str = "binance",
    now_ms: Optional[int] = None,
) -> pd.DataFrame:
    """Current OI snapshot for one perp. Cron poll uses this every
    15 min per data-contract §5 ("aligned to 15-minute grid"). The
    timestamp on the returned row reflects the exchange's reported
    sample time, not the local clock."""
    assert_proxy_active()
    ex = _make_exchange(venue)
    try:
        raw = ex.fetch_open_interest(symbol)
    except Exception as e:  # noqa: BLE001
        raise CryptoFetchError(
            f"fetch_open_interest({venue}, {symbol}) failed: "
            f"{type(e).__name__}: {e}"
        ) from e
    now_ms_eff = int(time.time() * 1000) if now_ms is None else int(now_ms)
    row = _normalize_oi_row(raw, symbol_perp_ccxt=symbol, venue=venue,
                              now_ms=now_ms_eff)
    return _frame([row], OI_SCHEMA_COLUMNS)


def fetch_open_interest_history(
    symbol: str,
    start_ts_ms: int,
    end_ts_ms: int,
    *,
    venue: str = "binance",
    page_size: int = 500,
    now_ms: Optional[int] = None,
) -> pd.DataFrame:
    """Paginated OI history. Same cursor-advance + defensive-break
    contract as funding."""
    assert_proxy_active()
    if end_ts_ms <= start_ts_ms:
        return _frame([], OI_SCHEMA_COLUMNS)
    ex = _make_exchange(venue)
    cursor = int(start_ts_ms)
    rows: list[dict] = []
    now_ms_eff = int(time.time() * 1000) if now_ms is None else int(now_ms)

    max_pages = 200  # OI samples every 15min → 1 year ≈ 35k samples
    pages = 0
    while cursor < end_ts_ms and pages < max_pages:
        try:
            raw = ex.fetch_open_interest_history(
                symbol, since=cursor, limit=page_size,
            )
        except Exception as e:  # noqa: BLE001
            raise CryptoFetchError(
                f"fetch_open_interest_history({venue}, {symbol}, "
                f"since={cursor}) failed: {type(e).__name__}: {e}"
            ) from e
        pages += 1
        if not raw:
            break
        last_ts = int(raw[-1].get("timestamp") or 0)
        for item in raw:
            ts = int(item.get("timestamp") or 0)
            if ts >= end_ts_ms:
                cursor = end_ts_ms
                break
            if ts < cursor:
                continue
            rows.append(_normalize_oi_row(
                item, symbol_perp_ccxt=symbol, venue=venue,
                now_ms=now_ms_eff,
            ))
        if last_ts <= cursor:
            logger.warning(
                "OI history page did not advance cursor "
                "(last_ts=%s cursor=%s); breaking",
                last_ts, cursor,
            )
            break
        cursor = last_ts + 1
    return _frame(rows, OI_SCHEMA_COLUMNS)


# -----------------------------------------------------------------------------
# Storage (month-partitioned per §4 + §5)
# -----------------------------------------------------------------------------

def _month_path(root: Path, kind: str, venue: str,
                 symbol_canonical: str, month_utc: datetime) -> Path:
    """Construct the §4 / §5 month-partition path. `kind` is one of
    'funding' or 'open_interest'."""
    if month_utc.tzinfo is None:
        raise ValueError(
            "month_utc must be tz-aware UTC; got naive datetime"
        )
    if kind not in ("funding", "open_interest"):
        raise ValueError(f"kind {kind!r} not in ('funding', 'open_interest')")
    return (
        root / "raw" / kind / venue.lower() / symbol_canonical /
        f"year={month_utc.year:04d}" /
        f"month={month_utc.month:02d}.parquet"
    )


def _write_month_partitions(
    df: pd.DataFrame, *, kind: str, venue: str,
    root: Optional[Path] = None,
) -> list[Path]:
    """Shared month-partitioned writer for funding + OI. Idempotent on
    duplicate timestamps (newer ingested_at wins on conflict per
    symbol)."""
    if df.empty:
        return []
    if root is None:
        root = crypto_root()
    if "symbol" not in df.columns:
        raise ValueError("df missing required 'symbol' column")

    df = df.copy()
    df["_month_utc"] = pd.to_datetime(
        df["timestamp_utc"], unit="ms", utc=True,
    ).dt.to_period("M").dt.start_time.dt.tz_localize("UTC")
    df["_symbol_canonical"] = df["symbol"].map(
        lambda s: to_canonical_perp_symbol(s, venue)
    )

    written: list[Path] = []
    group_keys = ["_symbol_canonical", "_month_utc"]
    dedup_keys = ["timestamp_utc", "symbol"]
    for (sym_can, month), part in df.groupby(group_keys, sort=False):
        part_path = _month_path(root, kind, venue, sym_can,
                                  month.to_pydatetime())
        part_path.parent.mkdir(parents=True, exist_ok=True)
        out = part.drop(columns=["_month_utc", "_symbol_canonical"])
        if part_path.exists():
            existing = pd.read_parquet(part_path)
            merged = pd.concat([existing, out], ignore_index=True)
            merged = (
                merged.sort_values("ingested_at")
                .drop_duplicates(subset=dedup_keys, keep="last")
                .sort_values("timestamp_utc")
                .reset_index(drop=True)
            )
            merged.to_parquet(part_path)
        else:
            out.sort_values("timestamp_utc").reset_index(drop=True).to_parquet(part_path)
        written.append(part_path)
    return written


def write_funding_partitions(
    df: pd.DataFrame, *, venue: str, root: Optional[Path] = None,
) -> list[Path]:
    """Month-partitioned funding write per §4."""
    return _write_month_partitions(df, kind="funding", venue=venue, root=root)


def write_oi_partitions(
    df: pd.DataFrame, *, venue: str, root: Optional[Path] = None,
) -> list[Path]:
    """Month-partitioned OI write per §5."""
    return _write_month_partitions(df, kind="open_interest", venue=venue,
                                     root=root)
