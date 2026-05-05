"""A-share market index judgment module.

Combines index price action + FinBERT news sentiment to judge overall market direction.
"""
import logging
import pandas as pd
from data.collectors.market import MarketCollector

logger = logging.getLogger(__name__)


class MarketJudge:
    """Judges overall A-share market direction.

    Considers:
    - CSI300/Shanghai Composite intraday performance
    - Global market sentiment (from FinBERT analysis)
    - Geopolitical factors
    """

    def __init__(self):
        self.collector = MarketCollector()

    def judge(self, geo_factors: dict = None, news_sentiment: dict = None) -> dict:
        """Judge overall market direction.

        Args:
            geo_factors: Dict with geo_risk_index, policy_signal, etc.
            news_sentiment: Dict from NewsSentimentAnalyzer with market_sentiment, etc.

        Returns:
            Dict with:
                - direction: str ("偏强", "中性", "偏弱", "强势", "弱势")
                - score: float [-1, 1]
                - reason: str
                - suggested_position: str (e.g., "7成", "5成", "3成")
                - index_change: float (CSI300 change %)
        """
        # Get CSI300 index realtime data
        index_change = self._get_index_change()

        # Score components
        index_score = max(-1, min(1, index_change / 3))  # ±3% = ±1

        # News/geo factors
        market_sent = 0.0
        if news_sentiment:
            market_sent = news_sentiment.get("market_sentiment", 0)

        geo_score = 0.0
        if geo_factors:
            geo_score = (
                geo_factors.get("geo_risk_index", 0) * 0.3
                + geo_factors.get("policy_signal", 0) * 0.3
                + geo_factors.get("china_us_temperature", 0) * 0.4
            )

        # Weighted combination
        final_score = (
            index_score * 0.5
            + market_sent * 0.25
            + geo_score * 0.25
        )
        final_score = max(-1, min(1, final_score))

        # Determine direction
        if final_score > 0.5:
            direction = "强势"
        elif final_score > 0.15:
            direction = "偏强"
        elif final_score > -0.15:
            direction = "中性"
        elif final_score > -0.5:
            direction = "偏弱"
        else:
            direction = "弱势"

        # Suggested position
        if final_score > 0.5:
            position = "8成"
        elif final_score > 0.2:
            position = "6-7成"
        elif final_score > -0.2:
            position = "5成"
        elif final_score > -0.5:
            position = "3-4成"
        else:
            position = "2成以下"

        # Reason
        reason = self._generate_reason(index_change, market_sent, geo_score, geo_factors)

        return {
            "direction": direction,
            "score": round(final_score, 3),
            "reason": reason,
            "suggested_position": position,
            "index_change": round(index_change, 2),
        }

    def _get_index_change(self) -> float:
        """Get CSI300 index change percentage."""
        try:
            import akshare as ak
            # Try to get CSI300 ETF (510300) as proxy
            quote = self.collector.fetch_realtime("sh510300")
            if quote:
                return quote.get("change_pct", 0)

            # Fallback: try shanghai composite
            quote = self.collector.fetch_realtime("sh000001")
            if quote:
                return quote.get("change_pct", 0)

        except Exception as e:
            logger.warning(f"Index fetch failed: {e}")
        return 0.0

    def _generate_reason(self, index_change, market_sent, geo_score, geo_factors) -> str:
        """Generate human-readable reason for market judgment."""
        parts = []

        if abs(index_change) > 0.5:
            if index_change > 0:
                parts.append(f"大盘上涨{index_change:+.1f}%")
            else:
                parts.append(f"大盘下跌{index_change:+.1f}%")

        if market_sent > 0.2:
            parts.append("全球市场情绪偏暖")
        elif market_sent < -0.2:
            parts.append("全球市场情绪偏冷")

        if geo_factors:
            if geo_factors.get("geo_risk_index", 0) < -0.3:
                parts.append("地缘风险偏高")
            if geo_factors.get("policy_signal", 0) < -0.3:
                parts.append("政策偏紧")
            elif geo_factors.get("policy_signal", 0) > 0.3:
                parts.append("政策偏松")
            if geo_factors.get("china_us_temperature", 0) < -0.3:
                parts.append("中美关系紧张")
            elif geo_factors.get("china_us_temperature", 0) > 0.3:
                parts.append("中美关系缓和")

        return "，".join(parts) if parts else "市场平稳"

    def format_for_report(self, judgment: dict) -> str:
        """Format judgment for inclusion in push report header."""
        return (
            f"大盘研判：{judgment['direction']}"
            f"（{judgment['reason']}）\n"
            f"建议整体仓位：{judgment['suggested_position']}"
        )
