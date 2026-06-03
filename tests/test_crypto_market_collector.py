"""Tests for data/collectors/crypto_market.py — Phase Crypto-A step 2.

All tests are offline:
  - ccxt is monkeypatched with a stub exchange that returns canned rows
  - crypto_root is monkeypatched / CRYPTO_STORAGE_ROOT env is set to tmp_path
  - assert_proxy_active env sentinels are set / unset as the contract demands

The pinning purpose:

  1. Closed-bar gate semantics (single helper, exact arithmetic)
  2. Canonical symbol format
  3. fetch_recent / fetch_historical refuse without ssproxy sentinels
  4. fetch_historical correctly pages AND trims overshoot at end_ts_ms
  5. write_ohlcv_partitions is idempotent and uses the §3 path layout
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _activate_proxy_env(monkeypatch):
    from config import crypto_network as cn
    monkeypatch.setenv(cn.CRYPTO_NETWORK_ENV, "crypto")
    monkeypatch.setenv(cn.CRYPTO_SSPROXY_ENV, "verified")


def _make_ccxt_stub(rows_by_call: list[list[list]]):
    """Build a stub class whose `fetch_ohlcv` returns canned rows from
    a per-call queue. Mirrors CCXT's calling convention. Used as a
    drop-in replacement for `ccxt.binance`."""
    class _Stub:
        def __init__(self, *a, **kw):
            self._queue = list(rows_by_call)
            self.calls: list[dict] = []

        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
            self.calls.append({
                "symbol": symbol, "timeframe": timeframe,
                "since": since, "limit": limit,
            })
            if not self._queue:
                return []
            return self._queue.pop(0)

    return _Stub


# -----------------------------------------------------------------------------
# Closed-bar gate
# -----------------------------------------------------------------------------

def test_is_closed_with_buffer_at_boundary():
    """Bar closes at T + tf; closed iff that close + buffer <= now."""
    from data.collectors.crypto_market import _is_closed_with_buffer, CLOSED_BUFFER_SEC

    bar_open_ms = 1_700_000_000_000
    tf_sec = 3600  # 1h
    bar_close_ms = bar_open_ms + tf_sec * 1000

    # Exactly at close → buffer NOT yet elapsed → not closed
    assert _is_closed_with_buffer(bar_open_ms, tf_sec, bar_close_ms) is False
    # close + buffer - 1ms → just shy → not closed
    assert _is_closed_with_buffer(
        bar_open_ms, tf_sec, bar_close_ms + CLOSED_BUFFER_SEC * 1000 - 1
    ) is False
    # close + buffer exactly → closed
    assert _is_closed_with_buffer(
        bar_open_ms, tf_sec, bar_close_ms + CLOSED_BUFFER_SEC * 1000
    ) is True
    # well after → closed
    assert _is_closed_with_buffer(
        bar_open_ms, tf_sec, bar_close_ms + CLOSED_BUFFER_SEC * 1000 + 60_000
    ) is True


# -----------------------------------------------------------------------------
# Canonical symbol
# -----------------------------------------------------------------------------

def test_to_canonical_symbol_basic():
    from data.collectors.crypto_market import to_canonical_symbol
    assert to_canonical_symbol("BTC/USDT", "Binance") == "binance__btc_usdt__spot"
    assert to_canonical_symbol("ETH/USDC", "okx", "perp") == "okx__eth_usdc__perp"


def test_to_canonical_symbol_rejects_missing_slash():
    from data.collectors.crypto_market import to_canonical_symbol
    with pytest.raises(ValueError, match="must contain '/'"):
        to_canonical_symbol("BTCUSDT", "binance")


# -----------------------------------------------------------------------------
# Schema integrity
# -----------------------------------------------------------------------------

def test_normalize_row_produces_all_schema_columns():
    from data.collectors.crypto_market import (
        _normalize_ohlcv_row, SCHEMA_COLUMNS,
    )
    row = _normalize_ohlcv_row(
        [1_700_000_000_000, 50000.0, 50500.0, 49800.0, 50300.0, 12.5],
        symbol_ccxt="BTC/USDT", venue="binance", timeframe="1h",
        now_ms=1_700_000_000_000 + 3600_000 + 121_000,
    )
    assert set(row.keys()) == set(SCHEMA_COLUMNS), (
        f"normalizer drift: missing or extra keys: {set(row.keys()) ^ set(SCHEMA_COLUMNS)}"
    )
    assert row["symbol"] == "BTC/USDT"
    assert row["exchange"] == "binance"
    assert row["is_closed_bar"] is True
    assert row["quote_volume_estimated"] is True


# -----------------------------------------------------------------------------
# fetch_recent — proxy gate + happy path
# -----------------------------------------------------------------------------

def test_fetch_recent_refuses_without_proxy_sentinels(monkeypatch):
    from config import crypto_network as cn
    from data.collectors.crypto_market import fetch_recent
    monkeypatch.delenv(cn.CRYPTO_NETWORK_ENV, raising=False)
    monkeypatch.delenv(cn.CRYPTO_SSPROXY_ENV, raising=False)
    with pytest.raises(cn.CryptoProxyNotActiveError):
        fetch_recent("BTC/USDT", "1h")


def test_fetch_recent_returns_schema_dataframe(monkeypatch):
    from data.collectors.crypto_market import fetch_recent, SCHEMA_COLUMNS
    _activate_proxy_env(monkeypatch)

    rows = [
        [1_700_000_000_000, 50000.0, 50500.0, 49800.0, 50300.0, 12.5],
        [1_700_003_600_000, 50300.0, 50600.0, 50100.0, 50400.0, 8.0],
    ]
    stub_cls = _make_ccxt_stub([rows])
    fake_ccxt = type("_ccxt", (), {"binance": stub_cls})
    monkeypatch.setitem(__import__("sys").modules, "ccxt", fake_ccxt)

    df = fetch_recent("BTC/USDT", "1h", venue="binance",
                       now_ms=1_700_007_500_000)
    assert list(df.columns) == list(SCHEMA_COLUMNS)
    assert len(df) == 2
    assert df.iloc[0]["symbol"] == "BTC/USDT"
    # Bar 0 closed at +3600s; now=+7500s; buffer 120s ⇒ closed.
    # pandas may unwrap to numpy.bool_, so compare via bool() instead
    # of `is True`.
    assert bool(df.iloc[0]["is_closed_bar"]) is True


def test_fetch_recent_rejects_unknown_timeframe(monkeypatch):
    from data.collectors.crypto_market import fetch_recent
    _activate_proxy_env(monkeypatch)
    with pytest.raises(ValueError, match="timeframe"):
        fetch_recent("BTC/USDT", "7m")


# -----------------------------------------------------------------------------
# fetch_historical — pagination + overshoot trim
# -----------------------------------------------------------------------------

def test_fetch_historical_trims_overshoot_at_end_ts_ms(monkeypatch):
    """CCXT's last page may return rows AT or AFTER end_ts_ms. The
    collector must drop them."""
    from data.collectors.crypto_market import fetch_historical
    _activate_proxy_env(monkeypatch)

    start = 1_700_000_000_000
    end = start + 3 * 3600_000   # 3 hours window → expect ≤ 3 rows kept
    # CCXT returns 4 rows (3 inside window + 1 past end_ts)
    page = [
        [start + 0 * 3600_000, 1, 2, 0.5, 1.5, 10],
        [start + 1 * 3600_000, 1, 2, 0.5, 1.5, 10],
        [start + 2 * 3600_000, 1, 2, 0.5, 1.5, 10],
        [start + 3 * 3600_000, 1, 2, 0.5, 1.5, 10],   # AT end → drop
        [start + 4 * 3600_000, 1, 2, 0.5, 1.5, 10],   # past → drop
    ]
    stub_cls = _make_ccxt_stub([page, []])
    fake_ccxt = type("_ccxt", (), {"binance": stub_cls})
    monkeypatch.setitem(__import__("sys").modules, "ccxt", fake_ccxt)

    df = fetch_historical("BTC/USDT", "1h", start, end,
                           now_ms=end + 600_000)
    assert len(df) == 3, (
        f"expected 3 rows after end-bound trim, got {len(df)}: "
        f"{df['timestamp_utc'].tolist()}"
    )
    assert df["timestamp_utc"].max() == start + 2 * 3600_000


def test_fetch_historical_empty_window_returns_empty():
    from data.collectors.crypto_market import fetch_historical, SCHEMA_COLUMNS
    # Even without env sentinels: empty window short-circuits before
    # the proxy check? Actually no — assert_proxy_active runs first.
    # So set them.
    import os, contextlib
    saved_env = {
        k: os.environ.pop(k, None) for k in
        ("CRYPTO_NETWORK_ACTIVE", "CRYPTO_SSPROXY_VERIFIED")
    }
    os.environ["CRYPTO_NETWORK_ACTIVE"] = "crypto"
    os.environ["CRYPTO_SSPROXY_VERIFIED"] = "1"
    try:
        df = fetch_historical("BTC/USDT", "1h",
                                start_ts_ms=1_700_000_000_000,
                                end_ts_ms=1_700_000_000_000)
        assert df.empty
        assert list(df.columns) == list(SCHEMA_COLUMNS)
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# -----------------------------------------------------------------------------
# write_ohlcv_partitions — idempotency + path layout
# -----------------------------------------------------------------------------

def _ts_utc(year, month, day, hour=0) -> int:
    """Deterministic UTC-anchored ms timestamp for test data."""
    from datetime import datetime, timezone
    return int(datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp() * 1000)


def _make_sample_df(symbol="BTC/USDT", venue="binance", timeframe="1h",
                     n_rows=3, base_ts=None):
    """Build a synthetic OHLCV frame. Default base_ts is 2023-11-15
    02:00 UTC so n_rows ≤ 22 stays within one UTC day (avoids
    accidentally exercising the day-boundary split in tests that
    just want a single partition)."""
    from data.collectors.crypto_market import SCHEMA_COLUMNS
    if base_ts is None:
        base_ts = _ts_utc(2023, 11, 15, 2)
    rows = []
    for i in range(n_rows):
        rows.append({
            "timestamp_utc": base_ts + i * 3600_000,
            "exchange": venue,
            "symbol": symbol,
            "timeframe": timeframe,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume_base": 10.0,
            "volume_quote": 1000.0,
            "quote_volume_estimated": True,
            "trades": -1,
            "is_closed_bar": True,
            "ingested_at": base_ts + i * 3600_000 + 5000,
        })
    df = pd.DataFrame(rows)
    return df[list(SCHEMA_COLUMNS)]


def test_write_partitions_creates_section_3_path_layout(monkeypatch, tmp_path):
    from data.collectors.crypto_market import write_ohlcv_partitions
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    df = _make_sample_df()
    written = write_ohlcv_partitions(df, venue="binance")
    assert len(written) >= 1
    # The §3 layout: raw/ohlcv/binance/binance__btc_usdt__spot/1h/year=YYYY/month=MM/day=DD.parquet
    p = written[0]
    rel = p.relative_to(tmp_path).as_posix()
    assert rel.startswith("raw/ohlcv/binance/binance__btc_usdt__spot/1h/year=")
    assert rel.endswith(".parquet")


def test_write_partitions_idempotent_on_dup_timestamps(monkeypatch, tmp_path):
    """Writing the same rows twice yields the same row count; the
    second write must NOT duplicate rows. Newer ingest_at wins on
    conflict."""
    from data.collectors.crypto_market import write_ohlcv_partitions
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    df = _make_sample_df()

    write_ohlcv_partitions(df, venue="binance")
    # Bump ingested_at to simulate re-fetch
    df2 = df.copy()
    df2["ingested_at"] = df2["ingested_at"] + 10_000
    paths = write_ohlcv_partitions(df2, venue="binance")
    # Read back and confirm no dup
    read = pd.read_parquet(paths[0])
    assert len(read) == len(df), (
        f"idempotency broken: expected {len(df)} rows, got {len(read)}"
    )
    # The second write's ingested_at wins
    assert read["ingested_at"].max() == df2["ingested_at"].max()


def test_write_partitions_groups_by_day(monkeypatch, tmp_path):
    """Rows spanning two UTC days produce two partition files."""
    from data.collectors.crypto_market import write_ohlcv_partitions
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    # Start 2 hours before UTC midnight, 4 hourly bars → 2 on day D,
    # 2 on day D+1.
    base = _ts_utc(2023, 11, 14, 22)
    df = _make_sample_df(n_rows=4, base_ts=base)
    paths = write_ohlcv_partitions(df, venue="binance")
    assert len(set(paths)) == 2, (
        f"expected 2 distinct day partitions, got {len(set(paths))}: {paths}"
    )


# -----------------------------------------------------------------------------
# Import hygiene
# -----------------------------------------------------------------------------

def test_module_does_not_import_ccxt_at_module_top():
    """ccxt is heavy and known to make HTTP calls at certain points;
    we want to defer that until actual fetch. Verify by inspecting
    sys.modules before / after a fresh import in a subprocess-style
    check (cheap approximation: scan source for top-level ccxt usage)."""
    src = (
        Path(__file__).resolve().parents[1] / "data" / "collectors"
        / "crypto_market.py"
    ).read_text()
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ccxt") or stripped.startswith("from ccxt"):
            # Allow inside function body (indented) but NOT top-level
            if not line.startswith((" ", "\t")):
                pytest.fail(
                    f"crypto_market.py imports ccxt at module top: {line!r}. "
                    "Use lazy import inside functions."
                )
