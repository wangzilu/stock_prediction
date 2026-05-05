import pytest
from signals.scorer import SignalScorer, Recommendation, RiskAlert
from signals.risk_monitor import RiskMonitor


# --- Multi-timeframe fusion tests ---

class TestMultiTimeframe:
    def setup_method(self):
        self.scorer = SignalScorer()

    def test_basic_score_still_works(self):
        """Backward compatible: works without mid/macro scores."""
        rec = self.scorer.score_stock(
            code="SH600519", name="贵州茅台",
            model_score=0.8, sentiment_score=0.5, sentiment_heat=0.6,
        )
        assert isinstance(rec, Recommendation)
        assert rec.signal in ("强烈看多", "看多", "观望", "看空", "强烈看空")

    def test_multi_timeframe_bullish(self):
        """All timeframes bullish should produce strong signal."""
        rec = self.scorer.score_stock(
            code="SH600519", name="贵州茅台",
            model_score=0.8, sentiment_score=0.6, sentiment_heat=0.7,
            mid_term_score=0.7, macro_score=0.5,
        )
        assert rec.final_score > 0.5
        assert "看多" in rec.signal
        assert not rec.has_divergence

    def test_divergence_detection(self):
        """Short bullish + mid bearish should flag divergence."""
        rec = self.scorer.score_stock(
            code="SH600519", name="贵州茅台",
            model_score=0.7, sentiment_score=0.0, sentiment_heat=0.5,
            mid_term_score=-0.6, macro_score=0.0,
        )
        assert rec.has_divergence is True
        # Score should be dampened
        assert abs(rec.final_score) < 0.5

    def test_divergence_appended_to_signal(self):
        """Divergence should append (分歧) to non-neutral signals."""
        rec = self.scorer.score_stock(
            code="SH600519", name="贵州茅台",
            model_score=0.9, sentiment_score=0.5, sentiment_heat=0.5,
            mid_term_score=-0.5, macro_score=0.3,
        )
        if rec.signal != "观望":
            assert "分歧" in rec.signal

    def test_macro_risk_suppresses_bullish(self):
        """Strongly negative macro should suppress bullish signals."""
        # Without macro risk
        rec_normal = self.scorer.score_stock(
            code="SH600519", name="贵州茅台",
            model_score=0.8, sentiment_score=0.5, sentiment_heat=0.5,
            mid_term_score=0.5, macro_score=0.0,
        )
        # With macro risk
        rec_risk = self.scorer.score_stock(
            code="SH600519", name="贵州茅台",
            model_score=0.8, sentiment_score=0.5, sentiment_heat=0.5,
            mid_term_score=0.5, macro_score=-0.7,
        )
        assert rec_risk.final_score < rec_normal.final_score

    def test_reason_includes_all_timeframes(self):
        """Reason should mention all significant timeframe signals."""
        rec = self.scorer.score_stock(
            code="SH600519", name="贵州茅台",
            model_score=0.8, sentiment_score=0.5, sentiment_heat=0.7,
            mid_term_score=0.6, macro_score=-0.5,
        )
        assert "短线看多" in rec.reason
        assert "中线看多" in rec.reason
        assert "宏观承压" in rec.reason

    def test_recommendation_has_timeframe_scores(self):
        """Recommendation should carry individual timeframe scores."""
        rec = self.scorer.score_stock(
            code="SH600519", name="贵州茅台",
            model_score=0.5, sentiment_score=0.3, sentiment_heat=0.4,
            mid_term_score=0.6, macro_score=-0.2,
        )
        assert rec.short_term_score == 0.5
        assert rec.mid_term_score == 0.6
        assert rec.macro_score == -0.2


# --- Risk alert tests ---

class TestRiskMonitor:
    def setup_method(self):
        self.monitor = RiskMonitor()

    def test_no_alerts_in_calm_market(self):
        """Calm market should produce no alerts."""
        alerts = self.monitor.check_all(
            geo_factors={
                "geo_risk_index": 0.2,
                "china_us_temperature": 0.1,
                "policy_signal": 0.0,
                "safe_haven_signal": 0.1,
            },
        )
        assert len(alerts) == 0

    def test_sentiment_spike_alert(self):
        """Negative sentiment spike should trigger alert."""
        alert = self.monitor.check_sentiment_spike(
            sentiment_score=-0.7,
            sentiment_heat=0.9,
            stock_code="SH600519",
            stock_name="贵州茅台",
        )
        assert alert is not None
        assert alert.severity in ("warning", "critical")
        assert "SH600519" in alert.affected_codes

    def test_no_sentiment_alert_when_heat_low(self):
        """Negative sentiment without high heat should not trigger."""
        alert = self.monitor.check_sentiment_spike(
            sentiment_score=-0.8,
            sentiment_heat=0.3,  # Low heat
        )
        assert alert is None

    def test_geo_escalation_alert(self):
        """Severe geo risk should trigger alert."""
        alert = self.monitor.check_geo_escalation({
            "geo_risk_index": -0.8,
            "china_us_temperature": -0.6,
            "policy_signal": 0.0,
            "safe_haven_signal": 0.7,
        })
        assert alert is not None
        assert "地缘风险" in alert.trigger
        assert "中美" in alert.trigger

    def test_policy_hawkish_alert(self):
        """Strongly hawkish policy should trigger alert."""
        alert = self.monitor.check_policy_shock({
            "geo_risk_index": 0.0,
            "china_us_temperature": 0.0,
            "policy_signal": -0.8,
            "safe_haven_signal": 0.0,
        })
        assert alert is not None
        assert "收紧" in alert.trigger

    def test_policy_dovish_alert(self):
        """Strongly dovish policy should trigger opportunity alert."""
        alert = self.monitor.check_policy_shock({
            "geo_risk_index": 0.0,
            "china_us_temperature": 0.0,
            "policy_signal": 0.9,
            "safe_haven_signal": 0.0,
        })
        assert alert is not None
        assert "宽松" in alert.trigger

    def test_check_all_multiple_alerts(self):
        """Multiple risk conditions should produce multiple alerts."""
        alerts = self.monitor.check_all(
            geo_factors={
                "geo_risk_index": -0.9,
                "china_us_temperature": -0.7,
                "policy_signal": -0.8,
                "safe_haven_signal": 0.9,
            },
            sentiment_by_stock={
                "SH600519": {
                    "name": "贵州茅台",
                    "sentiment_score": -0.8,
                    "heat": 0.95,
                },
            },
        )
        assert len(alerts) >= 2  # At least geo + policy or sentiment

    def test_alert_message_format(self):
        """Alert message should contain key info."""
        scorer = SignalScorer()
        alert = RiskAlert(
            timestamp="2026-05-05 10:32",
            trigger="地缘风险指数急剧恶化",
            impact="全市场风险偏好下降",
            suggestion="建议仓位降至3成",
            affected_codes=["SH600519"],
            severity="critical",
        )
        msg = scorer.generate_alert_message(alert)
        assert "🚨" in msg
        assert "地缘风险" in msg
        assert "SH600519" in msg
