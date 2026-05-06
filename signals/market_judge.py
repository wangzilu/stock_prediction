"""A-share market index judgment module.

Combines index price action + LLM geopolitical analysis to judge market direction.
Adjusts weights based on time of day:
- Early session (before 11:00): rely more on LLM/news analysis (盘中数据不稳定)
- Late session (after 13:00): rely more on actual price action (方向已明确)
"""
import logging
from datetime import datetime
from data.collectors.market import MarketCollector

logger = logging.getLogger(__name__)


class MarketJudge:
    """Judges overall A-share market direction."""

    def __init__(self):
        self.collector = MarketCollector()

    def judge(self, geo_factors: dict = None) -> dict:
        """Judge overall market direction.

        Adjusts weight allocation based on time of day:
        - Before 11:00: index 15%, geo 35%, LLM direction 50% (消息面驱动)
        - After 13:00: index 50%, geo 20%, LLM direction 30% (盘面驱动)
        """
        index_change = self._get_index_change()
        index_score = max(-1, min(1, index_change / 3))

        geo_score = 0.0
        llm_direction = 0.0
        if geo_factors:
            geo_score = (
                geo_factors.get("geo_risk_index", 0) * 0.3
                + geo_factors.get("policy_signal", 0) * 0.3
                + geo_factors.get("china_us_temperature", 0) * 0.4
            )
            llm_direction = geo_factors.get("market_direction", 0)

        # Time-adaptive weighting
        hour = datetime.now().hour
        if hour < 11:
            # Early session: price unreliable, trust news/LLM more
            w_index, w_geo, w_llm = 0.15, 0.35, 0.50
        elif hour < 13:
            # Mid-day: balanced
            w_index, w_geo, w_llm = 0.30, 0.30, 0.40
        else:
            # Afternoon: price action is clear
            w_index, w_geo, w_llm = 0.50, 0.20, 0.30

        final_score = (
            index_score * w_index
            + geo_score * w_geo
            + llm_direction * w_llm
        )
        final_score = max(-1, min(1, final_score))

        # Direction
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

        # Position
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

        reason = self._generate_reason(index_change, geo_score, geo_factors, hour)

        return {
            "direction": direction,
            "score": round(final_score, 3),
            "reason": reason,
            "suggested_position": position,
            "index_change": round(index_change, 2),
        }

    def _get_index_change(self) -> float:
        """Get CSI300 index change percentage.

        Uses Tencent single-stock API directly (instant, no full market load).
        """
        try:
            # Direct Tencent call for index - bypasses slow AKShare full market load
            quote = self.collector._fetch_realtime_tencent_single("sh000300")  # CSI300 index
            if quote and quote.get("change_pct"):
                return quote["change_pct"]

            quote = self.collector._fetch_realtime_tencent_single("sh000001")  # Shanghai Composite
            if quote and quote.get("change_pct"):
                return quote["change_pct"]
        except Exception as e:
            logger.warning(f"Index fetch failed: {e}")
        return 0.0

    def _generate_reason(self, index_change, geo_score, geo_factors, hour) -> str:
        """Generate human-readable reason."""
        parts = []

        if abs(index_change) > 0.3:
            parts.append(f"盘面{'上涨' if index_change > 0 else '下跌'}{index_change:+.1f}%")

        if geo_factors:
            md = geo_factors.get("market_direction", 0)
            if md > 0.3:
                parts.append("消息面偏多")
            elif md < -0.3:
                parts.append("消息面偏空")

            if geo_factors.get("geo_risk_index", 0) < -0.3:
                parts.append("地缘风险偏高")
            if geo_factors.get("china_us_temperature", 0) > 0.3:
                parts.append("中美关系缓和")
            elif geo_factors.get("china_us_temperature", 0) < -0.3:
                parts.append("中美关系紧张")
            if geo_factors.get("policy_signal", 0) < -0.3:
                parts.append("政策偏紧")
            elif geo_factors.get("policy_signal", 0) > 0.3:
                parts.append("政策偏松")

        if hour < 11:
            parts.append("早盘研判以消息面为主")

        return "，".join(parts) if parts else "市场平稳"

    def format_for_report(self, judgment: dict) -> str:
        """Format judgment for push report header."""
        return (
            f"大盘研判：{judgment['direction']}"
            f"（{judgment['reason']}）\n"
            f"建议整体仓位：{judgment['suggested_position']}"
        )
