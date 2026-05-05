"""Integration tests - require network access.
Run with: /usr/bin/python3 -m pytest tests/test_integration.py -v
"""
import pytest


def test_market_collector_real_data():
    """Verify AKShare can fetch real stock data."""
    from data.collectors.market import MarketCollector

    collector = MarketCollector()
    df = collector.fetch_daily("sh600519", days=5)
    assert not df.empty
    assert "close" in df.columns


def test_sentiment_scoring_real():
    """Verify sentiment scoring works end-to-end."""
    from factors.sentiment import SentimentScorer

    scorer = SentimentScorer()
    score = scorer.score_text("市场情绪高涨，看好后市")
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_signal_scorer_end_to_end():
    """Full signal scoring pipeline."""
    from signals.scorer import SignalScorer

    scorer = SignalScorer()
    rec = scorer.score_stock(
        code="SH600519",
        name="贵州茅台",
        model_score=0.5,
        sentiment_score=0.3,
        sentiment_heat=0.7,
    )
    assert rec.signal in ("强烈看多", "看多", "观望", "看空", "强烈看空")
    report = scorer.generate_report([rec])
    assert "贵州茅台" in report


def test_verifier_full_cycle(tmp_path):
    """Full verification cycle: record -> verify -> report."""
    from tracker.verifier import Verifier

    v = Verifier(db_path=str(tmp_path / "test.db"))
    v.record_recommendation("2026-04-28", "SH600519", "贵州茅台", "看多", 0.8, 1800.0)
    v.verify("2026-04-28", "SH600519", 1800.0, 1860.0, 1880.0, 1780.0)

    report = v.generate_verification_report("2026-04-28")
    assert "贵州茅台" in report
    assert "✅" in report

    stats = v.get_cumulative_stats()
    assert stats["win_rate"] == 100.0
