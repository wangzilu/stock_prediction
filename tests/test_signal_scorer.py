import pytest
from signals.scorer import SignalScorer, Recommendation


def test_score_stock_returns_recommendation():
    """Scoring a stock should return a Recommendation object."""
    scorer = SignalScorer()
    rec = scorer.score_stock(
        code="SH600519",
        name="贵州茅台",
        model_score=0.8,
        sentiment_score=0.5,
        sentiment_heat=0.6,
    )
    assert isinstance(rec, Recommendation)
    assert rec.code == "SH600519"
    assert rec.name == "贵州茅台"
    assert -1.0 <= rec.final_score <= 1.0
    assert rec.signal in ("强烈看多", "看多", "观望", "看空", "强烈看空")


def test_high_score_gives_bullish_signal():
    """High model + sentiment scores should give bullish signal."""
    scorer = SignalScorer()
    rec = scorer.score_stock(
        code="SH600519",
        name="贵州茅台",
        model_score=0.9,
        sentiment_score=0.7,
        sentiment_heat=0.8,
    )
    assert rec.signal in ("强烈看多", "看多")


def test_low_score_gives_bearish_signal():
    """Low model + negative sentiment should give bearish signal."""
    scorer = SignalScorer()
    rec = scorer.score_stock(
        code="SH600519",
        name="贵州茅台",
        model_score=-0.8,
        sentiment_score=-0.6,
        sentiment_heat=0.5,
    )
    assert rec.signal in ("强烈看空", "看空")


def test_generate_daily_report():
    """Generate daily report should return formatted recommendations."""
    scorer = SignalScorer()
    recs = [
        scorer.score_stock("SH600519", "贵州茅台", 0.8, 0.5, 0.6),
        scorer.score_stock("SZ300750", "宁德时代", 0.6, 0.3, 0.4),
    ]
    report = scorer.generate_report(recs)
    assert "今日推荐" in report
    assert "贵州茅台" in report
    assert "评分" in report
