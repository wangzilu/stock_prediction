import logging
from datetime import datetime
from typing import Optional

from signals.scorer import RiskAlert

logger = logging.getLogger(__name__)


class RiskMonitor:
    """Monitors for abnormal events that should trigger risk alerts.

    Checks:
    1. Sentiment spike: sudden surge in negative sentiment (heat > 3σ)
    2. Geopolitical escalation: geo risk index drops sharply
    3. Policy shock: unexpected hawkish/dovish policy signal
    """

    def __init__(
        self,
        sentiment_heat_threshold: float = 0.8,
        sentiment_negative_threshold: float = -0.5,
        geo_risk_threshold: float = -0.6,
        policy_shock_threshold: float = 0.7,
    ):
        self.sentiment_heat_threshold = sentiment_heat_threshold
        self.sentiment_negative_threshold = sentiment_negative_threshold
        self.geo_risk_threshold = geo_risk_threshold
        self.policy_shock_threshold = policy_shock_threshold

    def check_sentiment_spike(
        self,
        sentiment_score: float,
        sentiment_heat: float,
        stock_code: str = "",
        stock_name: str = "",
    ) -> Optional[RiskAlert]:
        """Check for sudden negative sentiment spike.

        Args:
            sentiment_score: Current sentiment score
            sentiment_heat: Current sentiment heat
            stock_code: Affected stock code
            stock_name: Affected stock name

        Returns:
            RiskAlert if triggered, None otherwise
        """
        if (
            sentiment_heat > self.sentiment_heat_threshold
            and sentiment_score < self.sentiment_negative_threshold
        ):
            severity = "critical" if sentiment_score < -0.7 else "warning"
            return RiskAlert(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                trigger=f"{stock_name}({stock_code}) 舆情异常：负面情绪飙升(热度{sentiment_heat:.1%}，情感{sentiment_score:+.2f})",
                impact=f"该标的短期可能承压，注意仓位控制",
                suggestion="建议降低仓位至3成以下" if severity == "critical" else "关注后续发展，适当减仓",
                affected_codes=[stock_code] if stock_code else [],
                severity=severity,
            )
        return None

    def check_geo_escalation(self, geo_factors: dict) -> Optional[RiskAlert]:
        """Check for geopolitical risk escalation.

        Args:
            geo_factors: Dict from GeopoliticalScorer.compute_all_factors

        Returns:
            RiskAlert if triggered, None otherwise
        """
        geo_risk = geo_factors.get("geo_risk_index", 0)
        china_us = geo_factors.get("china_us_temperature", 0)

        if geo_risk < self.geo_risk_threshold:
            severity = "critical" if geo_risk < -0.8 else "warning"
            trigger_parts = [f"地缘风险指数急剧恶化({geo_risk:+.2f})"]
            if china_us < -0.5:
                trigger_parts.append(f"中美关系紧张({china_us:+.2f})")

            return RiskAlert(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                trigger="，".join(trigger_parts),
                impact="全市场风险偏好可能下降，A股科技/出口板块首当其冲",
                suggestion="建议整体仓位降至3成，规避高风险敞口，考虑增配黄金",
                affected_codes=[],
                severity=severity,
            )
        return None

    def check_policy_shock(self, geo_factors: dict) -> Optional[RiskAlert]:
        """Check for unexpected policy direction change.

        Args:
            geo_factors: Dict from GeopoliticalScorer.compute_all_factors

        Returns:
            RiskAlert if triggered, None otherwise
        """
        policy = geo_factors.get("policy_signal", 0)

        if abs(policy) > self.policy_shock_threshold:
            if policy < 0:
                # Strongly hawkish
                return RiskAlert(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    trigger=f"央行政策明显收紧信号(政策指数{policy:+.2f})",
                    impact="加息/收紧预期升温，成长股和加密货币可能承压",
                    suggestion="减少成长股和加密货币仓位，关注高股息防御品种",
                    affected_codes=[],
                    severity="warning",
                )
            else:
                # Strongly dovish — opportunity, not risk
                return RiskAlert(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    trigger=f"央行政策明显宽松信号(政策指数{policy:+.2f})",
                    impact="降息/宽松预期升温，有利于股市和加密货币",
                    suggestion="可适当加仓，关注流动性敏感品种",
                    affected_codes=[],
                    severity="warning",
                )
        return None

    def check_all(
        self,
        geo_factors: dict,
        sentiment_by_stock: dict = None,
    ) -> list:
        """Run all risk checks.

        Args:
            geo_factors: From GeopoliticalScorer
            sentiment_by_stock: Dict of {code: {name, sentiment_score, heat}}

        Returns:
            List of RiskAlert objects (may be empty)
        """
        alerts = []

        # Geo checks
        geo_alert = self.check_geo_escalation(geo_factors)
        if geo_alert:
            alerts.append(geo_alert)

        policy_alert = self.check_policy_shock(geo_factors)
        if policy_alert:
            alerts.append(policy_alert)

        # Sentiment checks per stock
        if sentiment_by_stock:
            for code, data in sentiment_by_stock.items():
                sent_alert = self.check_sentiment_spike(
                    sentiment_score=data.get("sentiment_score", 0),
                    sentiment_heat=data.get("heat", 0),
                    stock_code=code,
                    stock_name=data.get("name", ""),
                )
                if sent_alert:
                    alerts.append(sent_alert)

        return alerts
