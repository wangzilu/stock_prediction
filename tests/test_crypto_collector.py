"""Legacy CryptoCollector integration tests.

Per code review P1 (2026-05-30) + quarantine §6.5: these tests load
`data.collectors.crypto` at module import and may issue real Binance /
AKShare network calls. They are inconsistent with the runtime
quarantine that keeps A-share daily cron clean of legacy crypto.

Default: SKIP. Opt in with `RUN_LEGACY_CRYPTO_TESTS=1` when explicitly
exercising the legacy collector (e.g. before retiring it).
"""

from __future__ import annotations

import os

import pytest

if os.environ.get("RUN_LEGACY_CRYPTO_TESTS", "").lower() not in ("1", "true", "yes"):
    pytest.skip(
        "Legacy crypto collector tests skipped by default (quarantine §6.5). "
        "Set RUN_LEGACY_CRYPTO_TESTS=1 to enable — these tests import "
        "data.collectors.crypto and may hit Binance / AKShare network.",
        allow_module_level=True,
    )

import pandas as pd  # noqa: E402
from data.collectors.crypto import CryptoCollector  # noqa: E402


def test_fetch_daily_returns_dataframe():
    """Fetching daily crypto data should return a DataFrame with OHLCV columns."""
    collector = CryptoCollector()
    df = collector.fetch_daily("BTC/USDT", days=5)
    assert isinstance(df, pd.DataFrame)
    # May fail if no network or ccxt not installed, but structure should be correct
    if not df.empty:
        required_cols = {"open", "high", "low", "close", "volume"}
        assert required_cols.issubset(set(df.columns))


def test_fetch_daily_eth():
    """Should also work for ETH."""
    collector = CryptoCollector()
    df = collector.fetch_daily("ETH/USDT", days=5)
    assert isinstance(df, pd.DataFrame)


def test_fetch_realtime_returns_dict():
    """Fetching realtime quote should return a dict."""
    collector = CryptoCollector()
    quote = collector.fetch_realtime("BTC/USDT")
    assert isinstance(quote, dict)
    if quote:
        assert "price" in quote
        assert "change_pct" in quote


def test_fetch_daily_invalid_symbol_returns_empty():
    """Invalid symbol should return empty DataFrame."""
    collector = CryptoCollector()
    df = collector.fetch_daily("INVALID/PAIR", days=5)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
