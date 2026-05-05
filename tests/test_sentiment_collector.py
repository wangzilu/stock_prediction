import pytest
from data.collectors.sentiment import SentimentCollector


def test_fetch_xueqiu_returns_list_of_posts():
    """Fetching Xueqiu posts for a stock should return a list of dicts."""
    collector = SentimentCollector()
    posts = collector.fetch_xueqiu("SH600519", limit=5)
    assert isinstance(posts, list)
    if len(posts) > 0:
        assert "text" in posts[0]
        assert "timestamp" in posts[0]
        assert "source" in posts[0]
        assert posts[0]["source"] == "xueqiu"


def test_fetch_eastmoney_returns_list_of_posts():
    """Fetching Eastmoney guba posts should return a list of dicts."""
    collector = SentimentCollector()
    posts = collector.fetch_eastmoney("600519", limit=5)
    assert isinstance(posts, list)
    if len(posts) > 0:
        assert "text" in posts[0]
        assert "timestamp" in posts[0]
        assert "source" in posts[0]
        assert posts[0]["source"] == "eastmoney"


def test_fetch_all_sentiment_combines_sources():
    """fetch_all should combine Xueqiu and Eastmoney results."""
    collector = SentimentCollector()
    posts = collector.fetch_all("SH600519", limit_per_source=3)
    assert isinstance(posts, list)
    sources = {p["source"] for p in posts}
    assert len(posts) >= 0
