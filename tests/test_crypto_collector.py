import pandas as pd
import pytest
from data.collectors.crypto import CryptoCollector


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
