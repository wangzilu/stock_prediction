from dataclasses import dataclass
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


class SignalScorer:
    """Combines model predictions and sentiment into final signals."""

    def __init__(
        self,
        weight_model: float = 0.6,
        weight_sentiment: float = 0.3,
        weight_heat: float = 0.1,
    ):
        self.weight_model = weight_model
        self.weight_sentiment = weight_sentiment
        self.weight_heat = weight_heat

    def score_stock(
        self,
        code: str,
        name: str,
        model_score: float,
        sentiment_score: float,
        sentiment_heat: float,
    ) -> Recommendation:
        """Compute final signal for a stock."""
        model_norm = max(-1.0, min(1.0, model_score))

        final_score = (
            model_norm * self.weight_model
            + sentiment_score * self.weight_sentiment
            + (sentiment_heat - 0.5) * self.weight_heat
        )
        final_score = max(-1.0, min(1.0, final_score))

        signal = self._score_to_signal(final_score)
        reason = self._generate_reason(model_norm, sentiment_score, sentiment_heat)

        return Recommendation(
            code=code,
            name=name,
            final_score=round(final_score, 2),
            signal=signal,
            model_score=round(model_norm, 2),
            sentiment_score=round(sentiment_score, 2),
            sentiment_heat=round(sentiment_heat, 2),
            reason=reason,
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
        self, model_score: float, sentiment_score: float, sentiment_heat: float
    ) -> str:
        """Generate human-readable reason for the signal."""
        parts = []

        if model_score > 0.3:
            parts.append("量化模型看多")
        elif model_score < -0.3:
            parts.append("量化模型看空")

        if sentiment_score > 0.3:
            parts.append("舆情偏正面")
        elif sentiment_score < -0.3:
            parts.append("舆情偏负面")

        if sentiment_heat > 0.6:
            parts.append("讨论热度高")

        return "，".join(parts) if parts else "信号中性"

    def generate_report(self, recommendations: list[Recommendation]) -> str:
        """Generate formatted push report."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [f"📈 今日推荐 ({now})", "─────────────"]

        for i, rec in enumerate(recommendations, 1):
            score_display = round((rec.final_score + 1) * 5, 1)
            # Stock codes like SH600519 -> 600519; crypto/gold codes stay as-is
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
