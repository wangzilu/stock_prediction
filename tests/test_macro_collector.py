import pytest
from data.collectors.macro import MacroCollector


def test_fetch_rss_returns_list():
    """Fetching an RSS feed should return a list."""
    collector = MacroCollector()
    # Use Google News RSS as a reliable test source
    items = collector.fetch_rss(
        "https://news.google.com/rss/search?q=economy&hl=en",
        max_items=5,
    )
    assert isinstance(items, list)
    if len(items) > 0:
        assert "title" in items[0]
        assert "published" in items[0]


def test_fetch_market_news():
    """Fetching market news should return a list."""
    collector = MacroCollector()
    items = collector.fetch_market_news(max_items=5)
    assert isinstance(items, list)


def test_fetch_all_adds_source():
    """fetch_all should add source key to each item."""
    collector = MacroCollector()
    items = collector.fetch_all(max_per_source=3)
    assert isinstance(items, list)
    if len(items) > 0:
        assert "source" in items[0]


def test_invalid_rss_returns_empty():
    """Invalid RSS URL should return empty list."""
    collector = MacroCollector()
    items = collector.fetch_rss("https://invalid.example.com/rss.xml", max_items=5)
    assert isinstance(items, list)
    assert len(items) == 0
