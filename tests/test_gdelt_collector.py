import pandas as pd
import pytest
from data.collectors.gdelt import GDELTCollector


def test_fetch_events_by_country_returns_dataframe():
    """Fetching GDELT events should return a DataFrame."""
    collector = GDELTCollector()
    df = collector.fetch_events_by_country("CH", days=3, max_records=10)
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert "title" in df.columns
        assert "tone" in df.columns


def test_fetch_china_us_relations():
    """Fetching China-US relation events should return a DataFrame."""
    collector = GDELTCollector()
    df = collector.fetch_china_us_relations(days=3)
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert "tone" in df.columns


def test_fetch_geopolitical_conflicts():
    """Fetching conflict events should return a DataFrame."""
    collector = GDELTCollector()
    df = collector.fetch_geopolitical_conflicts(days=3)
    assert isinstance(df, pd.DataFrame)


def test_fetch_tone_timeseries():
    """Fetching tone timeseries should return a DataFrame."""
    collector = GDELTCollector()
    df = collector.fetch_tone_timeseries("china economy", days=7)
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert "tone" in df.columns
