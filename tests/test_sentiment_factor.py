import pytest
from factors.sentiment import SentimentScorer


def test_score_single_text_returns_float():
    """Scoring a single text should return a float between -1 and 1."""
    scorer = SentimentScorer()
    score = scorer.score_text("这只股票最近表现非常好，看涨！")
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_score_negative_text():
    """Negative text should get a negative or low score."""
    scorer = SentimentScorer()
    score = scorer.score_text("暴跌崩盘了，赶紧跑，要完蛋了")
    assert score < 0.3


def test_score_batch_returns_aggregate():
    """Scoring a batch should return aggregate metrics."""
    scorer = SentimentScorer()
    posts = [
        {"text": "看好这只票，业绩很棒", "timestamp": "2026-05-05T10:00:00", "source": "xueqiu"},
        {"text": "下跌趋势明显，不看好", "timestamp": "2026-05-05T11:00:00", "source": "eastmoney"},
        {"text": "持续关注，等待机会", "timestamp": "2026-05-05T12:00:00", "source": "xueqiu"},
    ]
    result = scorer.score_batch(posts)
    assert "sentiment_score" in result
    assert "heat" in result
    assert "post_count" in result
    assert -1.0 <= result["sentiment_score"] <= 1.0
    assert result["post_count"] == 3


def test_score_empty_batch():
    """Empty post list should return neutral score."""
    scorer = SentimentScorer()
    result = scorer.score_batch([])
    assert result["sentiment_score"] == 0.0
    assert result["heat"] == 0.0
    assert result["post_count"] == 0
