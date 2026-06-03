"""Tests for data/collectors/crypto_derivatives.py — Phase Crypto-A step 3.

All offline: ccxt stubbed, env sentinels controlled, crypto_root via
CRYPTO_STORAGE_ROOT to tmp_path.

Coverage:
  - Perp symbol parsing + canonical form
  - Funding row normaliser (CCXT 4.x conventions)
  - OI row normaliser
  - fetch_funding_recent/_history: refuse without ssproxy, paginate,
    defensive break on stuck cursor
  - fetch_open_interest_recent/_history: same gates
  - write_funding_partitions / write_oi_partitions: month layout,
    idempotency, multi-month split
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest


def _activate_proxy_env(monkeypatch):
    from config import crypto_network as cn
    monkeypatch.setenv(cn.CRYPTO_NETWORK_ENV, "crypto")
    monkeypatch.setenv(cn.CRYPTO_SSPROXY_ENV, "verified")


def _ts_utc(year, month, day, hour=0) -> int:
    return int(datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp() * 1000)


# -----------------------------------------------------------------------------
# Symbol helpers
# -----------------------------------------------------------------------------

def test_parse_perp_symbol_basic():
    from data.collectors.crypto_derivatives import parse_perp_symbol
    assert parse_perp_symbol("BTC/USDT:USDT") == ("BTC", "USDT", "USDT")
    assert parse_perp_symbol("ETH/USD:ETH") == ("ETH", "USD", "ETH")


def test_parse_perp_symbol_rejects_spot_form():
    from data.collectors.crypto_derivatives import parse_perp_symbol
    with pytest.raises(ValueError, match="CCXT perp form"):
        parse_perp_symbol("BTC/USDT")  # spot, missing :SETTLE


def test_to_canonical_perp_symbol():
    from data.collectors.crypto_derivatives import to_canonical_perp_symbol
    assert to_canonical_perp_symbol("BTC/USDT:USDT", "Binance") == "binance__btc_usdt__perp"
    assert to_canonical_perp_symbol("SOL/USDT:USDT", "OKX") == "okx__sol_usdt__perp"


# -----------------------------------------------------------------------------
# Normalisers
# -----------------------------------------------------------------------------

def test_normalize_funding_row_full_payload():
    from data.collectors.crypto_derivatives import (
        _normalize_funding_row, FUNDING_SCHEMA_COLUMNS,
    )
    raw = {
        "timestamp": 1_700_000_000_000,
        "fundingRate": 0.0001,
        "nextFundingTime": 1_700_028_800_000,
        "markPrice": 50000.0,
        "indexPrice": 50001.0,
    }
    row = _normalize_funding_row(
        raw, symbol_perp_ccxt="BTC/USDT:USDT", venue="binance",
        now_ms=1_700_000_000_500,
    )
    assert set(row.keys()) == set(FUNDING_SCHEMA_COLUMNS)
    assert row["funding_rate"] == 0.0001
    assert row["next_funding_ts"] == 1_700_028_800_000
    assert row["mark_price"] == 50000.0


def test_normalize_funding_row_missing_rate_raises():
    from data.collectors.crypto_derivatives import _normalize_funding_row
    with pytest.raises(ValueError, match="fundingRate"):
        _normalize_funding_row(
            {"timestamp": 1, "markPrice": 1.0},  # no fundingRate
            symbol_perp_ccxt="BTC/USDT:USDT", venue="binance", now_ms=1,
        )


def test_normalize_oi_row_with_long_short_in_info():
    from data.collectors.crypto_derivatives import (
        _normalize_oi_row, OI_SCHEMA_COLUMNS,
    )
    raw = {
        "timestamp": 1_700_000_000_000,
        "openInterest": 12345.6,
        "openInterestValue": 6.17e8,
        "info": {"longShortRatio": 1.42},
    }
    row = _normalize_oi_row(
        raw, symbol_perp_ccxt="BTC/USDT:USDT", venue="binance", now_ms=1,
    )
    assert set(row.keys()) == set(OI_SCHEMA_COLUMNS)
    assert row["open_interest"] == 12345.6
    assert row["oi_quote"] == 6.17e8
    assert row["long_short_ratio"] == 1.42


def test_normalize_oi_row_missing_open_interest_raises():
    from data.collectors.crypto_derivatives import _normalize_oi_row
    with pytest.raises(ValueError, match="openInterest"):
        _normalize_oi_row(
            {"timestamp": 1, "info": {}},
            symbol_perp_ccxt="BTC/USDT:USDT", venue="binance", now_ms=1,
        )


# -----------------------------------------------------------------------------
# Fetch gating
# -----------------------------------------------------------------------------

def test_funding_recent_refuses_without_proxy(monkeypatch):
    from config import crypto_network as cn
    from data.collectors.crypto_derivatives import fetch_funding_recent
    monkeypatch.delenv(cn.CRYPTO_NETWORK_ENV, raising=False)
    monkeypatch.delenv(cn.CRYPTO_SSPROXY_ENV, raising=False)
    with pytest.raises(cn.CryptoProxyNotActiveError):
        fetch_funding_recent("BTC/USDT:USDT")


def test_oi_recent_refuses_without_proxy(monkeypatch):
    from config import crypto_network as cn
    from data.collectors.crypto_derivatives import fetch_open_interest_recent
    monkeypatch.delenv(cn.CRYPTO_NETWORK_ENV, raising=False)
    monkeypatch.delenv(cn.CRYPTO_SSPROXY_ENV, raising=False)
    with pytest.raises(cn.CryptoProxyNotActiveError):
        fetch_open_interest_recent("BTC/USDT:USDT")


# -----------------------------------------------------------------------------
# Pagination
# -----------------------------------------------------------------------------

class _StubExchange:
    """Stand-in for ccxt.<venue>; queues funding/oi responses."""
    def __init__(self, *a, **kw):
        self.funding_pages: list[list[dict]] = []
        self.oi_pages: list[list[dict]] = []
        self.calls: list[dict] = []

    def fetch_funding_rate_history(self, symbol, since=None, limit=None,
                                     params=None):
        self.calls.append({
            "fn": "funding", "symbol": symbol, "since": since,
            "limit": limit,
        })
        if not self.funding_pages:
            return []
        return self.funding_pages.pop(0)

    def fetch_open_interest(self, symbol):
        return {"timestamp": _ts_utc(2023, 11, 15, 12),
                "openInterest": 100.0, "info": {}}

    def fetch_open_interest_history(self, symbol, since=None, limit=None):
        self.calls.append({
            "fn": "oi", "symbol": symbol, "since": since, "limit": limit,
        })
        if not self.oi_pages:
            return []
        return self.oi_pages.pop(0)


def _inject_stub_ccxt(monkeypatch, stub):
    """Install a fake ccxt module exposing `binance` → returns stub."""
    fake = type("_ccxt", (), {"binance": lambda *a, **kw: stub})
    monkeypatch.setitem(__import__("sys").modules, "ccxt", fake)


def test_fetch_funding_history_paginates_and_trims(monkeypatch):
    from data.collectors.crypto_derivatives import fetch_funding_history
    _activate_proxy_env(monkeypatch)

    stub = _StubExchange()
    # Page 1: 3 events
    base = _ts_utc(2023, 11, 14)
    stub.funding_pages = [[
        {"timestamp": base + 0 * 28_800_000, "fundingRate": 0.0001},
        {"timestamp": base + 1 * 28_800_000, "fundingRate": 0.0002},
        {"timestamp": base + 2 * 28_800_000, "fundingRate": 0.0003},
    ], []]  # second page empty → loop terminates
    _inject_stub_ccxt(monkeypatch, stub)

    # Window covers all 3 events
    end = base + 3 * 28_800_000
    df = fetch_funding_history("BTC/USDT:USDT", base, end, now_ms=end + 1000)
    assert len(df) == 3
    assert df["funding_rate"].tolist() == [0.0001, 0.0002, 0.0003]


def test_fetch_funding_history_defensive_break_on_stuck_cursor(monkeypatch):
    """Exchange returns the SAME page over and over with last_ts <= cursor.
    Per contract §4 we MUST break instead of looping forever."""
    from data.collectors.crypto_derivatives import fetch_funding_history
    _activate_proxy_env(monkeypatch)

    base = _ts_utc(2023, 11, 14)
    stuck_page = [{"timestamp": base, "fundingRate": 0.0001}]
    stub = _StubExchange()
    # If we DIDN'T break, the stub would deplete after a handful of
    # pages anyway, but the test wants to assert call_count is bounded
    # and that one row was kept.
    stub.funding_pages = [stuck_page] * 100  # plenty if we DON'T break
    _inject_stub_ccxt(monkeypatch, stub)

    df = fetch_funding_history(
        "BTC/USDT:USDT", base, base + 365 * 86_400_000,
        now_ms=base + 1000,
    )
    # Defensive break means we made at most a couple of calls before
    # the cursor-stuck check kicked in.
    funding_calls = [c for c in stub.calls if c["fn"] == "funding"]
    assert len(funding_calls) <= 3, (
        f"defensive break failed: {len(funding_calls)} pages fetched on stuck cursor"
    )
    assert len(df) == 1


def test_fetch_funding_history_empty_window():
    from data.collectors.crypto_derivatives import (
        fetch_funding_history, FUNDING_SCHEMA_COLUMNS,
    )
    import os
    # Env sentinels required even for empty window — assert_proxy_active
    # runs first.
    saved = {
        k: os.environ.pop(k, None) for k in
        ("CRYPTO_NETWORK_ACTIVE", "CRYPTO_SSPROXY_VERIFIED")
    }
    os.environ["CRYPTO_NETWORK_ACTIVE"] = "crypto"
    os.environ["CRYPTO_SSPROXY_VERIFIED"] = "1"
    try:
        df = fetch_funding_history("BTC/USDT:USDT", 100, 100)
        assert df.empty
        assert list(df.columns) == list(FUNDING_SCHEMA_COLUMNS)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# -----------------------------------------------------------------------------
# Write paths
# -----------------------------------------------------------------------------

def _make_funding_df(symbol="BTC/USDT:USDT", venue="binance", n_rows=3,
                      base_ts=None):
    from data.collectors.crypto_derivatives import FUNDING_SCHEMA_COLUMNS
    if base_ts is None:
        base_ts = _ts_utc(2023, 11, 15, 0)
    rows = []
    for i in range(n_rows):
        rows.append({
            "timestamp_utc": base_ts + i * 28_800_000,
            "exchange": venue,
            "symbol": symbol,
            "funding_rate": 0.0001 * (i + 1),
            "next_funding_ts": base_ts + (i + 1) * 28_800_000,
            "mark_price": 50000.0 + i,
            "index_price": 50001.0 + i,
            "ingested_at": base_ts + i * 28_800_000 + 5000,
        })
    return pd.DataFrame(rows)[list(FUNDING_SCHEMA_COLUMNS)]


def test_write_funding_partitions_uses_month_path(monkeypatch, tmp_path):
    from data.collectors.crypto_derivatives import write_funding_partitions
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    df = _make_funding_df()
    paths = write_funding_partitions(df, venue="binance")
    assert len(paths) >= 1
    p = paths[0]
    rel = p.relative_to(tmp_path).as_posix()
    # §4 path: raw/funding/binance/binance__btc_usdt__perp/year=YYYY/month=MM.parquet
    assert rel.startswith("raw/funding/binance/binance__btc_usdt__perp/year=")
    assert "/month=" in rel
    assert rel.endswith(".parquet")


def test_write_funding_partitions_idempotent(monkeypatch, tmp_path):
    from data.collectors.crypto_derivatives import write_funding_partitions
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    df = _make_funding_df()
    write_funding_partitions(df, venue="binance")
    df2 = df.copy()
    df2["ingested_at"] = df2["ingested_at"] + 10_000
    paths = write_funding_partitions(df2, venue="binance")
    read = pd.read_parquet(paths[0])
    assert len(read) == len(df), (
        f"funding idempotency broken: expected {len(df)}, got {len(read)}"
    )
    # Newer ingested_at wins
    assert read["ingested_at"].max() == df2["ingested_at"].max()


def test_write_funding_partitions_splits_by_month(monkeypatch, tmp_path):
    from data.collectors.crypto_derivatives import write_funding_partitions
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    # 2 funding events: one Nov 30, one Dec 1 → 2 month partitions
    nov30 = _ts_utc(2023, 11, 30, 22)
    df = _make_funding_df(n_rows=2, base_ts=nov30)
    # The 2nd row is base + 28800000 ms = 8 hours later → Dec 1 06:00 UTC
    paths = write_funding_partitions(df, venue="binance")
    assert len(set(paths)) == 2, (
        f"expected 2 month partitions, got {len(set(paths))}: {paths}"
    )


def _make_oi_df(symbol="BTC/USDT:USDT", venue="binance", n_rows=3,
                  base_ts=None):
    from data.collectors.crypto_derivatives import OI_SCHEMA_COLUMNS
    if base_ts is None:
        base_ts = _ts_utc(2023, 11, 15, 0)
    rows = []
    for i in range(n_rows):
        rows.append({
            "timestamp_utc": base_ts + i * 15 * 60_000,  # 15-min grid
            "exchange": venue,
            "symbol": symbol,
            "open_interest": 100.0 + i,
            "oi_quote": 5_000_000.0 + i,
            "long_short_ratio": 1.5 + 0.01 * i,
            "ingested_at": base_ts + i * 15 * 60_000 + 1000,
        })
    return pd.DataFrame(rows)[list(OI_SCHEMA_COLUMNS)]


def test_write_oi_partitions_uses_open_interest_path(monkeypatch, tmp_path):
    from data.collectors.crypto_derivatives import write_oi_partitions
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    df = _make_oi_df()
    paths = write_oi_partitions(df, venue="binance")
    assert len(paths) >= 1
    p = paths[0]
    rel = p.relative_to(tmp_path).as_posix()
    # §5 path: raw/open_interest/binance/binance__btc_usdt__perp/year=YYYY/month=MM.parquet
    assert rel.startswith("raw/open_interest/binance/binance__btc_usdt__perp/year=")
    assert rel.endswith(".parquet")


def test_write_oi_partitions_idempotent(monkeypatch, tmp_path):
    from data.collectors.crypto_derivatives import write_oi_partitions
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    df = _make_oi_df()
    write_oi_partitions(df, venue="binance")
    df2 = df.copy()
    df2["ingested_at"] = df2["ingested_at"] + 10_000
    paths = write_oi_partitions(df2, venue="binance")
    read = pd.read_parquet(paths[0])
    assert len(read) == len(df)
    assert read["ingested_at"].max() == df2["ingested_at"].max()


# -----------------------------------------------------------------------------
# Import hygiene
# -----------------------------------------------------------------------------

def test_derivatives_module_does_not_import_ccxt_at_module_top():
    src = (
        Path(__file__).resolve().parents[1] / "data" / "collectors"
        / "crypto_derivatives.py"
    ).read_text()
    for line in src.splitlines():
        if line.startswith("import ccxt") or line.startswith("from ccxt"):
            pytest.fail(
                f"crypto_derivatives.py imports ccxt at module top: {line!r}"
            )
