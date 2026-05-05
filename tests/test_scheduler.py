import pytest
import pandas as pd
from unittest.mock import MagicMock
from scheduler.jobs import DailyPipeline


def _make_pipeline():
    """Create a fully mocked DailyPipeline."""
    pipeline = DailyPipeline.__new__(DailyPipeline)
    pipeline.market_collector = MagicMock()
    pipeline.crypto_collector = MagicMock()
    pipeline.gold_collector = MagicMock()
    pipeline.sentiment_collector = MagicMock()
    pipeline.gdelt_collector = MagicMock()
    pipeline.macro_collector = MagicMock()
    pipeline.sentiment_scorer = MagicMock()
    pipeline.geo_scorer = MagicMock()
    pipeline.news_analyzer = MagicMock()
    pipeline.signal_scorer = MagicMock()
    pipeline.risk_monitor = MagicMock()
    pipeline.pusher = MagicMock()
    pipeline.verifier = MagicMock()
    pipeline.market_judge = MagicMock()
    pipeline._geo_factors = None
    pipeline._nlp_result = None
    return pipeline


def test_pipeline_runs_without_error():
    """Pipeline should handle mocked components without crashing."""
    pipeline = _make_pipeline()

    pipeline.market_collector.fetch_realtime.return_value = {"price": 1800.0, "change_pct": 1.5}
    pipeline.crypto_collector.fetch_realtime.return_value = {"price": 100000.0, "change_pct": 2.0}
    pipeline.gold_collector.fetch_realtime.return_value = {"price": 550.0, "change_pct": 0.5}
    pipeline.sentiment_collector.fetch_all.return_value = [
        {"text": "看好", "timestamp": "2026-05-05T10:00:00", "source": "xueqiu"}
    ]
    pipeline.sentiment_scorer.score_batch.return_value = {
        "sentiment_score": 0.5, "heat": 0.6, "post_count": 1
    }

    # Mock macro + FinBERT analysis
    pipeline.macro_collector.fetch_all.return_value = []
    pipeline.news_analyzer.analyze_geopolitical_news.return_value = {
        "overall_sentiment": -0.1,
        "conflict_sentiment": -0.2,
        "china_us_sentiment": 0.1,
        "policy_sentiment": -0.1,
        "market_sentiment": 0.0,
        "num_analyzed": 50,
        "num_conflict": 10,
        "num_china_us": 8,
    }

    mock_rec = MagicMock()
    mock_rec.code = "SH600519"
    mock_rec.name = "[A股] 贵州茅台"
    mock_rec.final_score = 0.7
    mock_rec.signal = "看多"
    mock_rec.reason = "量化模型看多"

    pipeline.signal_scorer.score_stock.return_value = mock_rec
    pipeline.signal_scorer.generate_report.return_value = "test report"
    pipeline.pusher.send_recommendation.return_value = True
    pipeline.pusher.send.return_value = True

    mock_judgment = {
        "direction": "中性",
        "score": 0.0,
        "reason": "市场平稳",
        "suggested_position": "5成",
        "index_change": 0.0,
    }
    pipeline.market_judge.judge.return_value = mock_judgment
    pipeline.market_judge.format_for_report.return_value = "大盘研判：中性（市场平稳）\n建议整体仓位：5成"

    pipeline.run_daily_recommendation()


def test_pipeline_verification():
    """Pipeline verification should check due items."""
    pipeline = _make_pipeline()

    pipeline.verifier.get_due_verifications.return_value = []

    pipeline.run_verification()
    pipeline.verifier.get_due_verifications.assert_called_once()
