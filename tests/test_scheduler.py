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
    pipeline.macro_collector = MagicMock()
    pipeline.sentiment_scorer = MagicMock()
    pipeline.signal_scorer = MagicMock()
    pipeline.risk_monitor = MagicMock()
    pipeline.pusher = MagicMock()
    pipeline.verifier = MagicMock()
    pipeline.market_judge = MagicMock()
    pipeline.llm_analyst = MagicMock()
    pipeline._geo_factors = None
    pipeline._headlines = None
    return pipeline


def test_pipeline_runs_without_error():
    """Pipeline should handle mocked components without crashing."""
    pipeline = _make_pipeline()

    pipeline.market_collector.fetch_realtime.return_value = {"price": 1800.0, "change_pct": 1.5}
    pipeline.crypto_collector.fetch_realtime.return_value = {"price": 100000.0, "change_pct": 2.0}
    pipeline.gold_collector.fetch_realtime.return_value = {"price": 550.0, "change_pct": 0.5}
    pipeline.sentiment_collector.fetch_all.return_value = [
        {"text": "看好", "timestamp": "2026-05-06T10:00:00", "source": "xueqiu"}
    ]
    pipeline.sentiment_scorer.score_batch.return_value = {
        "sentiment_score": 0.5, "heat": 0.6, "post_count": 1
    }

    # Mock LLM analyst
    pipeline.macro_collector.fetch_all.return_value = [{"title": "test headline"}]
    pipeline.llm_analyst.analyze_geopolitics.return_value = {
        "geo_risk_index": -0.2,
        "china_us_temperature": 0.1,
        "policy_signal": -0.1,
        "safe_haven_signal": 0.3,
        "market_direction": 0.1,
        "reasoning": {"geo_risk": "test"},
    }
    pipeline.llm_analyst.generate_report.return_value = "LLM generated report"

    pipeline.market_judge.judge.return_value = {
        "direction": "中性", "score": 0.0, "reason": "市场平稳",
        "suggested_position": "5成", "index_change": 0.0,
    }

    mock_rec = MagicMock()
    mock_rec.code = "SH600519"
    mock_rec.name = "[A股] 贵州茅台"
    mock_rec.final_score = 0.7
    mock_rec.signal = "看多"
    mock_rec.reason = "量化模型看多"

    pipeline.signal_scorer.score_stock.return_value = mock_rec
    pipeline.pusher.send_recommendation.return_value = True
    pipeline.pusher.send.return_value = True

    pipeline.run_daily_recommendation()


def test_pipeline_verification():
    """Pipeline verification should check due items."""
    pipeline = _make_pipeline()
    pipeline.verifier.get_due_verifications.return_value = []
    pipeline.run_verification()
    pipeline.verifier.get_due_verifications.assert_called_once()
