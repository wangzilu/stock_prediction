import pandas as pd
import pytest
from data.collectors.market import MarketCollector


def test_fetch_daily_returns_dataframe():
    """Fetching daily data for a stock should return a DataFrame with OHLCV columns."""
    collector = MarketCollector()
    df = collector.fetch_daily("sh600519", days=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    required_cols = {"open", "high", "low", "close", "volume"}
    assert required_cols.issubset(set(df.columns))
    assert df.index.name == "date"


def test_fetch_daily_invalid_code_returns_empty():
    """Invalid stock code should return empty DataFrame."""
    collector = MarketCollector()
    df = collector.fetch_daily("sh999999", days=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_fetch_realtime_returns_dict():
    """Fetching realtime quote should return a dict with price info."""
    collector = MarketCollector()
    quote = collector.fetch_realtime("sh600519")
    assert isinstance(quote, dict)
    assert "price" in quote
    assert "change_pct" in quote
