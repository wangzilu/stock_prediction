import sys
import types

import pytest
import pandas as pd
from unittest.mock import MagicMock
from config.settings import LGB_MIN_PREDICTIONS
from scheduler.jobs import DailyPipeline, _sanitize_push_text
from signals.index_predictor import OvernightIndexPredictor
from signals.scorer import SignalScorer


@pytest.fixture(autouse=True)
def _isolate_sanitizer_from_production_data(monkeypatch):
    """Quarantine tests in this file from production data files.

    DailyPipeline._make_sanitizer reads three live production state
    files at call time:
      - data/storage/crash_predictions_latest.json
      - data/storage/global_chain_factors.parquet
      - data/storage/paper_shadow/risk_guard_state.json

    Tests that exercise methods which internally build a sanitizer
    (e.g. _build_evening_stock_forecasts, _build_stock_candidates,
    _candidates_from_stock_snapshot) would otherwise inherit whatever
    crash probabilities / chain-alpha flags / cooldown set happens to
    be in production at test-run time. That made the suite
    non-deterministic — pinning the previous reviewer hours on
    "SZ300750 missing intermittently" — chain-alpha < -2.0 on the
    latest snapshot was the actual cause.

    Each of these loaders now returns the empty / None default during
    tests. Tests that need to exercise a specific reject rule should
    monkeypatch the specific loader they care about after this fixture
    runs.
    """
    monkeypatch.setattr(
        DailyPipeline, "_load_crash_probs_for_sanitizer",
        lambda self: {},
    )
    monkeypatch.setattr(
        DailyPipeline, "_load_chain_alpha_for_sanitizer",
        lambda self, today: None,
    )
    monkeypatch.setattr(
        DailyPipeline, "_load_cooldown_for_sanitizer",
        lambda self, today: set(),
    )


def _make_pipeline():
    """Create a fully mocked DailyPipeline.

    Mirrors scheduler/jobs.py:DailyPipeline.__init__() instance-attribute
    setup. Uses __new__ to skip the real __init__ (which would instantiate
    network-touching collectors), then manually sets every attribute that
    __init__ would set. If __init__ gains a new attribute, this helper
    must be updated in lockstep — otherwise tests silently break with
    AttributeError deep in production code paths (caught by helpers like
    `except Exception: continue`, which makes diagnosis hard).

    Crypto-quarantine specifics (§6.5): production now accesses the legacy
    crypto collector via `self._get_crypto_collector()` (lazy, flag-gated)
    and `self._crypto_collector` (cache slot), not `self.crypto_collector`.
    To keep legacy test setups like `pipeline.crypto_collector.X` meaningful
    (so the mock intersects the production read path), this helper wires
    all three references to the same MagicMock instance. The instance-level
    `_get_crypto_collector` override bypasses the
    LEGACY_MARKET_CONTEXT_ENABLED flag so tests exercise the legacy code
    branches uniformly.
    """
    pipeline = DailyPipeline.__new__(DailyPipeline)
    # Live collectors → MagicMock
    pipeline.market_collector = MagicMock()
    pipeline.market_collector._spot_cache = None
    pipeline.market_collector._spot_loaded = False
    pipeline.market_collector._akshare_down = False
    crypto_mock = MagicMock()
    pipeline._crypto_collector = crypto_mock
    pipeline._get_crypto_collector = lambda: crypto_mock
    pipeline.crypto_collector = crypto_mock  # alias for legacy test setup
    pipeline.gold_collector = MagicMock()
    pipeline.sentiment_collector = MagicMock()
    pipeline.macro_collector = MagicMock()
    pipeline.sentiment_scorer = MagicMock()
    pipeline.signal_scorer = MagicMock()
    pipeline.global_indices = MagicMock()
    pipeline.risk_monitor = MagicMock()
    pipeline.pusher = MagicMock()
    pipeline.verifier = MagicMock()
    pipeline.market_judge = MagicMock()
    pipeline.llm_analyst = MagicMock()
    pipeline.index_predictor = OvernightIndexPredictor()
    # Cached data (matches __init__ block at scheduler/jobs.py:85-88)
    pipeline._geo_factors = None
    pipeline._headlines = None
    pipeline._capital_flow_signals = None
    # Pre-trained models (matches __init__ block at scheduler/jobs.py:90-95)
    pipeline._lgb_predictions = None
    pipeline._lgb_status = {"status": "unknown", "count": 0, "error": ""}
    pipeline._rl_agent = None
    pipeline._mid_model = None
    pipeline._mid_model_checked = False
    return pipeline


def test_pipeline_runs_without_error():
    """Pipeline should handle mocked components without crashing."""
    pipeline = _make_pipeline()

    import pandas as pd
    # Mock spot cache with sample data
    pipeline.market_collector._spot_cache = pd.DataFrame([
        {"代码": "600519", "名称": "贵州茅台", "最新价": 1800.0, "涨跌幅": 1.5, "成交量": 50000, "最高": 1820, "最低": 1780},
    ])
    pipeline.market_collector._spot_loaded = True
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
    pipeline.global_indices.format_for_report.return_value = "道琼斯: 49000 (+0.5%)"
    pipeline.global_indices.fetch_all.return_value = {
        "上证指数": {"price": 3100, "change_pct": 0.1},
        "深证成指": {"price": 9900, "change_pct": 0.2},
        "北证50": {"price": 900, "change_pct": -0.1},
        "创业板指": {"price": 1900, "change_pct": 0.3},
        "沪深300": {"price": 4000, "change_pct": 0.2},
    }

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
    assert pipeline.verifier.record_market_prediction.call_count == 4


def test_pipeline_verification():
    """Pipeline verification should check due items."""
    pipeline = _make_pipeline()
    pipeline.verifier.get_due_verifications.return_value = []
    pipeline.run_verification()
    pipeline.verifier.get_due_verifications.assert_called_once()


def test_lgb_predictions_degrade_when_latest_count_is_too_low(monkeypatch):
    """Low finite Qlib coverage should not be used as production scores."""
    pipeline = _make_pipeline()
    pipeline._lgb_predictions = None

    class FakeShortTermModel:
        @classmethod
        def load_from_pickle(cls, model_path):
            return cls()

        def predict_batch(self):
            return {"SH600519": 0.1}

    fake_module = types.SimpleNamespace(ShortTermModel=FakeShortTermModel)
    monkeypatch.setitem(sys.modules, "models.short_term", fake_module)
    import models.lgb_cache as lgb_cache

    monkeypatch.setattr(
        lgb_cache,
        "load_prediction_cache",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("no cache")),
    )

    preds = pipeline._load_lgb_predictions()

    assert preds == {}
    assert pipeline._lgb_status["status"] == "degraded"
    assert pipeline._lgb_status["count"] == 1
    assert pipeline._lgb_status["min_required"] == LGB_MIN_PREDICTIONS


def test_lgb_predictions_use_valid_cache_when_live_load_fails(monkeypatch):
    """A validated after-close LGB cache keeps production scoring live."""
    pipeline = _make_pipeline()
    pipeline._lgb_predictions = None

    class FakeShortTermModel:
        @classmethod
        def load_from_pickle(cls, model_path):
            raise ModuleNotFoundError("No module named 'qlib'")

    fake_module = types.SimpleNamespace(ShortTermModel=FakeShortTermModel)
    monkeypatch.setitem(sys.modules, "models.short_term", fake_module)
    import models.lgb_cache as lgb_cache

    monkeypatch.setattr(
        lgb_cache,
        "load_prediction_cache",
        lambda *args, **kwargs: (
            {"SH600519": 0.12, "SZ000001": -0.01},
            {"latest_date": "2026-05-07"},
        ),
    )

    preds = pipeline._load_lgb_predictions()

    assert preds["SH600519"] == 0.12
    assert pipeline._lgb_status["status"] == "ok"
    assert pipeline._lgb_status["source"] == "cache"
    assert pipeline._lgb_status["latest_date"] == "2026-05-07"


def test_horizon_recommendations_include_short_next_day_prediction():
    """Grouped recommendations should expose short/mid/long buckets."""
    pipeline = _make_pipeline()
    scorer = SignalScorer()
    recs = [
        scorer.score_stock("SH600519", "[A股] 贵州茅台", 0.9, 0.5, 0.6, mid_term_score=0.2, macro_score=0.1),
        scorer.score_stock("SZ300750", "[A股] 宁德时代", 0.6, 0.4, 0.5, mid_term_score=0.8, macro_score=0.2),
        scorer.score_stock("SH601318", "[A股] 中国平安", 0.4, 0.7, 0.5, mid_term_score=0.3, macro_score=0.7),
    ]
    recs[0].next_day_change_pct = 1.4
    recs[1].next_day_change_pct = 0.7
    recs[2].next_day_change_pct = 0.5

    groups = pipeline._classify_recommendations_by_horizon(recs, per_bucket=1)
    text = pipeline._format_horizon_recommendations(groups)

    # cx round 11 P1-1: title was "短线（明日）" / "明日预测";
    # relabelled to expose the 5-day horizon basis.
    assert "短线（5日窗口）" in text
    assert "中线（1-4周）" in text
    assert "长线（1-3月）" in text
    assert "5日均/日+1.40%" in text


def test_evening_outlook_records_structured_market_prediction():
    """22:00 outlook should prepend and store the index prediction."""
    pipeline = _make_pipeline()
    pipeline._fetch_geo_factors = MagicMock(return_value={
        "geo_risk_index": 0.1,
        "china_us_temperature": 0.2,
        "policy_signal": 0.3,
        "market_direction": 0.2,
    })
    pipeline._load_lgb_predictions = MagicMock(return_value={})
    pipeline._lgb_status = {"status": "degraded", "count": 30}
    pipeline.market_collector._load_spot_cache.return_value = None
    pipeline.crypto_collector.fetch_realtime.side_effect = [
        {"price": 100000, "change_pct": 1.0},
        {"price": 4000, "change_pct": 1.5},
    ]
    pipeline.gold_collector.fetch_realtime.return_value = {"price": 800, "change_pct": 0.1}
    pipeline.global_indices.fetch_all.return_value = {
        "沪深300": {"price": 4000, "change_pct": 0.2},
        "上证指数": {"price": 3100, "change_pct": 0.1},
        "深证成指": {"price": 9900, "change_pct": 0.3},
        "北证50": {"price": 900, "change_pct": -0.2},
        "科创50": {"price": 1000, "change_pct": 0.4},
        "创业板指": {"price": 1900, "change_pct": 0.3},
        "标普500": {"price": 5000, "change_pct": 0.4},
    }
    pipeline.global_indices.format_for_report.return_value = "沪深300: 4000 (+0.2%)"
    pipeline.llm_analyst.generate_outlook.return_value = "一、世界大事\nLLM outlook"
    pipeline.pusher.send_evening_outlook.return_value = True

    pipeline.run_evening_outlook()

    pipeline.verifier.record_market_prediction.assert_called_once()
    pushed_report = pipeline.pusher.send_evening_outlook.call_args.args[0]
    assert "【明日策略】" in pushed_report
    assert "一、世界大事" in pushed_report
    assert "四、明日A股大盘预测" in pushed_report
    assert "上证" in pushed_report
    assert "深证" in pushed_report
    assert "北证" in pushed_report
    assert "科创" in pushed_report
    assert "五、个股预测" in pushed_report
    # Production format uses "前十" (limit=10 at scheduler/jobs.py:2241).
    # Long-horizon section uses "长线观察榜前十（仅供参考...）" so we
    # broaden to "长线".
    assert "短线前十" in pushed_report
    assert "中线前十" in pushed_report
    assert "长线" in pushed_report
    assert "综合前十" in pushed_report
    assert "六、黄金预测" in pushed_report
    assert "七、加密货币预测" in pushed_report
    assert "LLM outlook" in pushed_report
    assert "Qlib" not in pushed_report
    assert "LGB" not in pushed_report


def test_user_facing_push_text_hides_model_implementation_names():
    text = _sanitize_push_text("Qlib / qlib / LightGBM / LGB / lgb")

    assert "Qlib" not in text
    assert "qlib" not in text
    assert "LightGBM" not in text
    assert "LGB" not in text
    assert "lgb" not in text
    assert "个股模型" in text
    assert "短线模型" in text


def test_intraday_decision_push_includes_indices_buys_and_mandatory_sells():
    pipeline = _make_pipeline()
    pipeline._fetch_geo_factors = MagicMock(return_value={
        "geo_risk_index": 0.1,
        "china_us_temperature": 0.1,
        "policy_signal": 0.2,
        "safe_haven_signal": 0.2,
        "market_direction": 0.3,
    })
    pipeline._load_lgb_predictions = MagicMock(return_value={
        "SH600519": 0.08,
        "SZ300750": 0.07,
        "SZ000001": 0.02,
    })
    pipeline._lgb_status = {"status": "ok", "count": 3, "latest_date": "2026-05-07"}
    pipeline.market_collector._spot_cache = pd.DataFrame([
        {"代码": "600519", "名称": "贵州茅台", "最新价": 110.0, "涨跌幅": 2.0, "成交量": 1000000, "最高": 112, "最低": 106},
        {"代码": "300750", "名称": "宁德时代", "最新价": 220.0, "涨跌幅": 1.5, "成交量": 900000, "最高": 225, "最低": 214},
        {"代码": "000001", "名称": "平安银行", "最新价": 12.0, "涨跌幅": 0.2, "成交量": 800000, "最高": 12.2, "最低": 11.8},
    ])
    pipeline.market_collector._load_spot_cache.return_value = None
    pipeline.market_collector.fetch_realtime.return_value = {
        "price": 110.0,
        "change_pct": 2.0,
        "volume": 1000000,
        "high": 112.0,
        "low": 106.0,
    }
    pipeline.crypto_collector.fetch_realtime.side_effect = [
        {"price": 100000, "change_pct": 1.0},
        {"price": 4000, "change_pct": 0.5},
    ]
    pipeline.gold_collector.fetch_realtime.return_value = {"price": 800, "change_pct": 0.1}
    pipeline.global_indices.fetch_all.return_value = {
        "沪深300": {"price": 4000, "change_pct": 0.2},
        "上证指数": {"price": 3100, "change_pct": 0.1},
        "深证成指": {"price": 9900, "change_pct": 0.3},
        "北证50": {"price": 900, "change_pct": -0.2},
        "创业板指": {"price": 1900, "change_pct": 0.4},
        "标普500": {"price": 5000, "change_pct": 0.4},
    }
    pipeline.global_indices.format_for_report.return_value = "指数数据"
    pipeline.verifier.get_recent_recommendations.return_value = [
        {
            "rec_date": "2026-05-06",
            "code": "SH600519",
            "name": "贵州茅台",
            "signal": "看多",
            "score": 0.8,
            "price_at_rec": 100.0,
        }
    ]
    pipeline.pusher.send_intraday_decision.return_value = True

    pipeline.run_sell_check()

    pipeline.pusher.send_intraday_decision.assert_called_once()
    pushed_report = pipeline.pusher.send_intraday_decision.call_args.args[0]
    assert "【14:30盘中决策】" in pushed_report
    assert "下一开盘日指数预测" in pushed_report
    assert "上证" in pushed_report
    assert "深证" in pushed_report
    assert "北证" in pushed_report
    assert "创业板" in pushed_report
    assert "强烈推荐" in pushed_report
    assert "下一开盘日预测" in pushed_report
    assert "必须卖出" in pushed_report
    assert "Qlib" not in pushed_report
    assert "LGB" not in pushed_report
    assert pipeline.verifier.record_recommendation.call_count >= 1


def test_daily_summary_includes_morning_final_prediction_comparison():
    pipeline = _make_pipeline()
    pipeline.run_verification = MagicMock()
    pipeline._fetch_geo_factors = MagicMock(return_value={
        "geo_risk_index": 0.1,
        "china_us_temperature": 0.1,
        "policy_signal": 0.2,
        "safe_haven_signal": 0.2,
        "market_direction": 0.3,
    })
    pipeline.crypto_collector.fetch_realtime.side_effect = [
        {"price": 100000, "change_pct": 1.0},
        {"price": 4000, "change_pct": 0.5},
    ]
    pipeline.gold_collector.fetch_realtime.return_value = {"price": 800, "change_pct": 0.1}
    pipeline.global_indices.fetch_all.return_value = {
        "沪深300": {"price": 4000, "change_pct": 0.2},
        "上证指数": {"price": 3100, "change_pct": 0.1},
        "深证成指": {"price": 9900, "change_pct": -0.2},
        "北证50": {"price": 900, "change_pct": -0.1},
        "创业板指": {"price": 1900, "change_pct": 0.5},
    }
    pipeline.global_indices.format_for_report.return_value = "指数数据"
    pipeline.verifier.verify_due_market_prediction_snapshots.return_value = [{"target_index": "上证指数"}]
    pipeline.verifier.generate_morning_prediction_error_report.return_value = "📌 早盘最终预测复盘\n误差主因：盘中资金改变了早盘动量信号"
    pipeline.verifier.verify_due_market_predictions.return_value = []
    pipeline.verifier.generate_market_prediction_report.return_value = ""
    pipeline.llm_analyst.generate_summary.return_value = "收盘总结正文"
    pipeline.pusher.send_daily_summary.return_value = True

    pipeline.run_daily_summary()

    pipeline.verifier.verify_due_market_prediction_snapshots.assert_called_once()
    pushed_report = pipeline.pusher.send_daily_summary.call_args.args[0]
    assert "早盘最终预测复盘" in pushed_report
    assert "误差主因" in pushed_report
    assert "收盘总结正文" in pushed_report


def test_evening_stock_forecasts_include_horizon_and_composite_top_five():
    pipeline = _make_pipeline()
    # 最新价 column required — sanitizer's invalid_price rule rejects
    # rows without a positive price regardless of require_quote setting.
    pipeline.market_collector._spot_cache = pd.DataFrame([
        {"代码": "600519", "名称": "贵州茅台", "最新价": 1800.0, "涨跌幅": 1.0, "成交量": 1000000},
        {"代码": "688981", "名称": "中芯国际", "最新价": 80.0, "涨跌幅": 2.0, "成交量": 900000},
        {"代码": "000001", "名称": "平安银行", "最新价": 12.0, "涨跌幅": 0.5, "成交量": 800000},
        {"代码": "300750", "名称": "宁德时代", "最新价": 220.0, "涨跌幅": 3.0, "成交量": 700000},
        {"代码": "002415", "名称": "海康威视", "最新价": 30.0, "涨跌幅": 0.2, "成交量": 600000},
        {"代码": "601318", "名称": "中国平安", "最新价": 45.0, "涨跌幅": -0.1, "成交量": 500000},
    ])
    pipeline.market_collector._load_spot_cache.return_value = None

    # Production calls _build_evening_stock_forecasts(lgb_preds, limit=10)
    # at scheduler/jobs.py:2241 and the label is hardcoded "前十". Test
    # the production default to keep the assertion aligned with the
    # actual evening report — previous "limit=5" + "前五" pair did not
    # match any real code path.
    groups = pipeline._build_evening_stock_forecasts(
        {
            "SH600519": 0.08,
            "SH688981": 0.07,
            "SZ000001": 0.06,
            "SZ300750": 0.05,
            "SZ002415": 0.04,
            "SH601318": 0.03,
        },
        limit=10,
    )
    text = pipeline._format_evening_stock_forecasts(groups)

    assert len(groups["短线"]) == 6
    assert len(groups["中线"]) == 6
    assert len(groups["长线"]) == 6
    assert len(groups["综合"]) == 6
    assert "短线前十" in text
    assert "中线前十" in text
    assert "长线" in text  # 长线观察榜前十
    assert "综合前十" in text
    # Production format renders "模型+0.0700" (no "分" suffix) — drift
    # from earlier "模型分" label. Assert on the broader substring.
    assert "模型" in text
    assert "Qlib" not in text
    assert "LGB" not in text


def test_stock_candidates_ignore_non_lgb_momentum_when_lgb_available():
    """Morning stock pool keeps all-A factor scores without letting them hijack model names."""
    pipeline = _make_pipeline()
    pipeline._lgb_status = {"status": "ok", "count": 2}
    pipeline.market_collector._spot_cache = pd.DataFrame([
        {"代码": "000001", "名称": "平安银行", "最新价": 12.0, "涨跌幅": 10.0, "成交量": 1000000},
        {"代码": "600519", "名称": "贵州茅台", "最新价": 110.0, "涨跌幅": 0.2, "成交量": 1000000},
        {"代码": "300750", "名称": "宁德时代", "最新价": 220.0, "涨跌幅": 5.0, "成交量": 1000000},
    ])
    pipeline.market_collector._load_spot_cache.return_value = None

    candidates = pipeline._build_stock_candidates(
        {
            "SH600519": 0.05,
            "SZ300750": -0.02,
        },
        stock_macro=0.1,
    )

    assert [item["code"] for item in candidates] == ["SH600519", "SZ000001"]
    assert candidates[0]["has_lgb"] is True
    assert candidates[0]["score_source"] == "ml_model"
    assert candidates[1]["has_lgb"] is False
    assert candidates[1]["score_source"] == "factor_fallback"


def test_evening_stock_forecasts_keep_all_a_factor_fallback_below_model_scores():
    """22:00 stock table scores the full spot pool while preferring model-covered names."""
    pipeline = _make_pipeline()
    pipeline.market_collector._spot_cache = pd.DataFrame([
        {"代码": "600519", "名称": "贵州茅台", "最新价": 110.0, "涨跌幅": 0.2, "成交量": 1000000},
        {"代码": "000001", "名称": "平安银行", "最新价": 12.0, "涨跌幅": 10.0, "成交量": 1000000},
    ])
    pipeline.market_collector._load_spot_cache.return_value = None

    groups = pipeline._build_evening_stock_forecasts({"SH600519": 0.05}, limit=2)

    assert [item["code"] for item in groups["短线"]] == ["SH600519", "SZ000001"]
    assert groups["短线"][0]["score_source"] == "ml_model"
    assert groups["短线"][1]["score_source"] == "factor_fallback"


def test_overnight_stock_snapshot_roundtrip_builds_morning_candidates(tmp_path, monkeypatch):
    """9:20 can reuse the 22:00 stock pool as its candidate baseline."""
    monkeypatch.setattr(
        "scheduler.jobs.OVERNIGHT_STOCK_SNAPSHOT_PATH",
        tmp_path / "overnight_stock_forecasts.json",
    )
    pipeline = _make_pipeline()
    pipeline._lgb_status = {"status": "ok", "count": 2}
    groups = {
        "短线": [
            {
                "code": "SH600519",
                "name": "贵州茅台",
                "price": 110.0,
                "lgb_score": 0.06,
                "change_pct": 0.5,
                "short_expected": 1.05,
                "mid_score": 0.08,
                "long_score": 0.07,
            }
        ],
        "中线": [
            {
                "code": "SZ300750",
                "name": "宁德时代",
                "price": 220.0,
                "lgb_score": 0.05,
                "change_pct": 0.2,
                "short_expected": 0.84,
                "mid_score": 0.09,
                "long_score": 0.06,
            }
        ],
        "长线": [],
        "综合": [],
    }

    pipeline._write_overnight_stock_snapshot(groups, target_date="2026-05-08")
    snapshot = pipeline._load_overnight_stock_snapshot(target_date="2026-05-08")
    candidates = pipeline._candidates_from_stock_snapshot(snapshot, stock_macro=0.2)

    assert [item["code"] for item in candidates] == ["SH600519", "SZ300750"]
    assert candidates[0]["next_day_change_pct"] == 1.05
    assert candidates[0]["mid_score_hint"] == 0.08
    assert pipeline._load_overnight_stock_snapshot(target_date="2026-05-09") is None


def test_morning_recommendation_uses_overnight_snapshot_flag():
    pipeline = _make_pipeline()
    pipeline.run_daily_recommendation = MagicMock()

    pipeline.run_morning_recommendation()

    pipeline.run_daily_recommendation.assert_called_once_with(use_overnight_snapshot=True)
