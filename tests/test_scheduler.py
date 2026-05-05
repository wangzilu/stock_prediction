import pytest
from unittest.mock import patch, MagicMock
from scheduler.jobs import DailyPipeline


def test_pipeline_runs_without_error():
    """Pipeline should handle mocked components without crashing."""
    pipeline = DailyPipeline.__new__(DailyPipeline)
    pipeline.market_collector = MagicMock()
    pipeline.sentiment_collector = MagicMock()
    pipeline.sentiment_scorer = MagicMock()
    pipeline.signal_scorer = MagicMock()
    pipeline.pusher = MagicMock()
    pipeline.verifier = MagicMock()

    pipeline.market_collector.fetch_realtime.return_value = {"price": 1800.0, "change_pct": 1.5}
    pipeline.sentiment_collector.fetch_all.return_value = [
        {"text": "看好", "timestamp": "2026-05-05T10:00:00", "source": "xueqiu"}
    ]
    pipeline.sentiment_scorer.score_batch.return_value = {
        "sentiment_score": 0.5, "heat": 0.6, "post_count": 1
    }

    mock_rec = MagicMock()
    mock_rec.code = "SH600519"
    mock_rec.name = "贵州茅台"
    mock_rec.final_score = 0.7
    mock_rec.signal = "看多"
    mock_rec.reason = "量化模型看多"

    pipeline.signal_scorer.score_stock.return_value = mock_rec
    pipeline.signal_scorer.generate_report.return_value = "test report"
    pipeline.pusher.send_recommendation.return_value = True
    pipeline.pusher.send.return_value = True

    pipeline.run_daily_recommendation()


def test_pipeline_verification():
    """Pipeline verification should check due items."""
    pipeline = DailyPipeline.__new__(DailyPipeline)
    pipeline.market_collector = MagicMock()
    pipeline.verifier = MagicMock()
    pipeline.pusher = MagicMock()

    pipeline.verifier.get_due_verifications.return_value = []

    pipeline.run_verification()
    pipeline.verifier.get_due_verifications.assert_called_once()
