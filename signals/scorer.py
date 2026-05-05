from dataclasses import dataclass, field
from datetime import datetime

from config.settings import HIGH_THRESHOLD, MID_THRESHOLD


@dataclass
class Recommendation:
    """A single stock recommendation."""
    code: str
    name: str
    final_score: float
    signal: str
    model_score: float
    sentiment_score: float
    sentiment_heat: float
    reason: str
    # Multi-timeframe scores (optional, filled when available)
    short_term_score: float = 0.0
    mid_term_score: float = 0.0
    macro_score: float = 0.0
    has_divergence: bool = False  # Short and mid disagree


@dataclass
class RiskAlert:
    """A risk alert triggered by abnormal events."""
    timestamp: str
    trigger: str          # What caused the alert
    impact: str           # Impact assessment
    suggestion: str       # Position suggestion
    affected_codes: list = field(default_factory=list)
    severity: str = "warning"  # "warning" or "critical"


class SignalScorer:
    """Combines multi-timeframe model predictions, sentiment, and geo factors."""

    def __init__(
        self,
        weight_short: float = 0.4,
        weight_mid: float = 0.3,
        weight_sentiment: float = 0.2,
        weight_macro: float = 0.1,
    ):
        self.weight_short = weight_short
        self.weight_mid = weight_mid
        self.weight_sentiment = weight_sentiment
        self.weight_macro = weight_macro

    def score_stock(
        self,
        code: str,
        name: str,
        model_score: float,
        sentiment_score: float,
        sentiment_heat: float,
        mid_term_score: float = 0.0,
        macro_score: float = 0.0,
    ) -> Recommendation:
        """Compute final signal with multi-timeframe fusion.

        Args:
            code: Stock code
            name: Display name
            model_score: Short-term model score [-1, 1]
            sentiment_score: Sentiment score [-1, 1]
            sentiment_heat: Sentiment heat [0, 1]
            mid_term_score: Mid-term model score [-1, 1]
            macro_score: Macro/geo composite score [-1, 1]

        Returns:
            Recommendation with fused signal
        """
        short = max(-1.0, min(1.0, model_score))
        mid = max(-1.0, min(1.0, mid_term_score))
        sent = max(-1.0, min(1.0, sentiment_score))
        macro = max(-1.0, min(1.0, macro_score))

        # Detect divergence: short and mid disagree on direction
        has_divergence = (short > 0.2 and mid < -0.2) or (short < -0.2 and mid > 0.2)

        # Weighted fusion
        final_score = (
            short * self.weight_short
            + mid * self.weight_mid
            + sent * self.weight_sentiment
            + macro * self.weight_macro
        )

        # Dampen score if timeframes diverge
        if has_divergence:
            final_score *= 0.6

        # Macro risk override: if macro is strongly negative, suppress bullish
        if macro < -0.5 and final_score > 0:
            final_score *= 0.5

        final_score = max(-1.0, min(1.0, final_score))

        signal = self._score_to_signal(final_score)

        # Append divergence warning to signal text
        if has_divergence and signal != "观望":
            signal += "(分歧)"

        reason = self._generate_reason(short, mid, sent, sentiment_heat, macro)

        return Recommendation(
            code=code,
            name=name,
            final_score=round(final_score, 2),
            signal=signal,
            model_score=round(short, 2),
            sentiment_score=round(sent, 2),
            sentiment_heat=round(sentiment_heat, 2),
            reason=reason,
            short_term_score=round(short, 2),
            mid_term_score=round(mid, 2),
            macro_score=round(macro, 2),
            has_divergence=has_divergence,
        )

    def _score_to_signal(self, score: float) -> str:
        """Convert numeric score to signal text."""
        if score > HIGH_THRESHOLD:
            return "强烈看多"
        elif score > MID_THRESHOLD:
            return "看多"
        elif score < -HIGH_THRESHOLD:
            return "强烈看空"
        elif score < -MID_THRESHOLD:
            return "看空"
        else:
            return "观望"

    def _generate_reason(
        self,
        short: float,
        mid: float,
        sentiment: float,
        heat: float,
        macro: float,
    ) -> str:
        """Generate human-readable reason for the signal."""
        parts = []

        # Short-term
        if short > 0.3:
            parts.append("短线看多")
        elif short < -0.3:
            parts.append("短线看空")

        # Mid-term
        if mid > 0.3:
            parts.append("中线看多")
        elif mid < -0.3:
            parts.append("中线看空")

        # Sentiment
        if sentiment > 0.3:
            parts.append("舆情偏正面")
        elif sentiment < -0.3:
            parts.append("舆情偏负面")

        if heat > 0.6:
            parts.append("讨论热度高")

        # Macro
        if macro > 0.3:
            parts.append("宏观利好")
        elif macro < -0.3:
            parts.append("宏观承压")

        return "，".join(parts) if parts else "信号中性"

    def generate_report(self, recommendations: list) -> str:
        """Generate formatted push report."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [f"📈 今日推荐 ({now})", "─────────────"]

        for i, rec in enumerate(recommendations, 1):
            score_display = round((rec.final_score + 1) * 5, 1)
            display_code = rec.code[2:] if rec.code[:2] in ("SH", "SZ") else rec.code
            lines.append(
                f"{i}. {rec.name}({display_code}) | {rec.signal} | 评分 {score_display}"
            )
            lines.append(f"   理由：{rec.reason}")

        lines.append("─────────────")

        if recommendations:
            best_score = recommendations[0].final_score
            if best_score > HIGH_THRESHOLD:
                position = "7-8成"
            elif best_score > MID_THRESHOLD:
                position = "5-6成"
            else:
                position = "3成以下"
            lines.append(f"建议仓位：{position}")

        return "\n".join(lines)

    def generate_alert_message(self, alert: RiskAlert) -> str:
        """Generate formatted risk alert message."""
        icon = "🚨" if alert.severity == "critical" else "⚠️"
        lines = [
            f"{icon} 风险警示 ({alert.timestamp})",
            "─────────────",
            f"触发原因：{alert.trigger}",
            f"影响评估：{alert.impact}",
            "─────────────",
            f"建议：{alert.suggestion}",
        ]
        if alert.affected_codes:
            lines.append(f"受影响标的：{', '.join(alert.affected_codes)}")
        return "\n".join(lines)
