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
    pipeline.signal_scorer = MagicMock()
    pipeline.risk_monitor = MagicMock()
    pipeline.pusher = MagicMock()
    pipeline.verifier = MagicMock()
    pipeline._geo_factors = None
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

    # Mock GDELT and macro collectors
    pipeline.gdelt_collector.fetch_geopolitical_conflicts.return_value = pd.DataFrame()
    pipeline.gdelt_collector.fetch_china_us_relations.return_value = pd.DataFrame()
    pipeline.macro_collector.fetch_all.return_value = []
    pipeline.geo_scorer.compute_all_factors.return_value = {
        "geo_risk_index": -0.2,
        "china_us_temperature": 0.1,
        "policy_signal": 0.0,
        "safe_haven_signal": 0.3,
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

    pipeline.run_daily_recommendation()


def test_pipeline_verification():
    """Pipeline verification should check due items."""
    pipeline = _make_pipeline()

    pipeline.verifier.get_due_verifications.return_value = []

    pipeline.run_verification()
    pipeline.verifier.get_due_verifications.assert_called_once()
