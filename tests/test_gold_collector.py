import pandas as pd
import pytest
from data.collectors.gold import GoldCollector


def test_fetch_daily_returns_dataframe():
    """Fetching daily gold data should return a DataFrame."""
    collector = GoldCollector()
    df = collector.fetch_daily(days=10)
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        required_cols = {"open", "high", "low", "close", "volume"}
        assert required_cols.issubset(set(df.columns))
        assert df.index.name == "date"


def test_fetch_realtime_returns_dict():
    """Fetching realtime gold quote should return a dict."""
    collector = GoldCollector()
    quote = collector.fetch_realtime()
    assert isinstance(quote, dict)
    if quote:
        assert "price" in quote
        assert "change_pct" in quote
