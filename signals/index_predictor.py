"""Next-trading-day A-share index prediction.

This is a transparent baseline for the 22:00 evening outlook. It is not a
trained alpha model yet; it produces a structured forecast that can be recorded
and verified daily while richer data/model work catches up.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
import math
from typing import Any


TARGET_INDEX = "沪深300"


@dataclass
class IndexPrediction:
    """Structured next-day index forecast."""

    pred_date: str
    target_date: str
    target_index: str
    direction: str
    expected_change_pct: float
    lower_bound_pct: float
    upper_bound_pct: float
    up_probability: float
    confidence: float
    drivers: list[str]
    risks: list[str]
    data_status: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def next_trading_day(current: date) -> date:
    """Return the next A-share trading day using Qlib calendar.

    Falls back to next weekday if Qlib calendar unavailable.
    """
    from pathlib import Path
    cal_path = Path(__file__).resolve().parents[1] / "data" / "storage" / "qlib_data" / "cn_data" / "calendars" / "day.txt"
    try:
        if cal_path.exists():
            cal_dates = [line.strip() for line in cal_path.read_text().splitlines() if line.strip()]
            current_str = current.strftime("%Y-%m-%d")
            for d in cal_dates:
                if d > current_str:
                    return datetime.strptime(d, "%Y-%m-%d").date()
    except Exception:
        pass
    # Fallback: next weekday
    target = current + timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target


# Keep old name for backward compatibility
next_weekday = next_trading_day


class OvernightIndexPredictor:
    """Baseline predictor for next-day CSI300 direction and return range."""

    def __init__(self, target_index: str = TARGET_INDEX):
        self.target_index = target_index

    def predict(
        self,
        *,
        as_of: datetime | None = None,
        global_indices: dict | None = None,
        geo_factors: dict | None = None,
        crypto_data: dict | None = None,
        gold_data: dict | None = None,
        top_bullish: list[dict] | None = None,
        top_bearish: list[dict] | None = None,
        lgb_status: dict | None = None,
        target_date: str | None = None,
    ) -> IndexPrediction:
        """Generate a structured next-trading-day market forecast."""
        as_of = as_of or datetime.now()
        global_indices = global_indices or {}
        geo_factors = geo_factors or {}
        crypto_data = crypto_data or {}
        lgb_status = lgb_status or {}

        a_momentum = self._a_share_momentum(global_indices)
        global_lead = self._global_lead(global_indices)
        risk_asset = self._risk_asset_signal(crypto_data, gold_data or {})
        geo_signal = self._geo_signal(geo_factors)
        lgb_breadth = self._lgb_breadth(top_bullish or [], top_bearish or [])

        lgb_weight = 0.10 if lgb_status.get("status") == "ok" else 0.0
        raw_expected = (
            a_momentum * 0.25
            + global_lead * 0.35
            + geo_signal * 0.25
            + risk_asset * 0.15
            + lgb_breadth * lgb_weight
        )
        if lgb_weight:
            raw_expected /= 1.10

        expected = round(_clamp(raw_expected, -3.0, 3.0), 2)
        direction = self._direction(expected)
        probability = self._up_probability(expected)
        confidence = self._confidence(
            expected=expected,
            global_indices=global_indices,
            crypto_data=crypto_data,
            geo_factors=geo_factors,
            lgb_status=lgb_status,
        )
        half_width = round(_clamp(1.05 - confidence * 0.55, 0.35, 0.90), 2)
        lower = round(expected - half_width, 2)
        upper = round(expected + half_width, 2)

        return IndexPrediction(
            pred_date=as_of.strftime("%Y-%m-%d"),
            target_date=target_date or next_weekday(as_of.date()).strftime("%Y-%m-%d"),
            target_index=self.target_index,
            direction=direction,
            expected_change_pct=expected,
            lower_bound_pct=lower,
            upper_bound_pct=upper,
            up_probability=probability,
            confidence=confidence,
            drivers=self._drivers(a_momentum, global_lead, geo_signal, risk_asset, lgb_breadth, lgb_status),
            risks=self._risks(geo_factors, lgb_status, confidence),
            data_status={
                "global_indices_count": len(global_indices),
                "has_crypto": bool(crypto_data),
                "has_gold": bool(gold_data),
                "lgb_status": lgb_status.get("status", "unknown"),
                "lgb_count": lgb_status.get("count", 0),
                "model_type": "rule_baseline_v1",
            },
        )

    def format_prediction(self, prediction: IndexPrediction) -> str:
        """Format a forecast block that can be prepended to the evening report."""
        return (
            "【明日大盘量化预测】\n"
            f"目标：A股大盘（{prediction.target_index}）{prediction.target_date}\n"
            f"方向：{prediction.direction}，预计涨跌幅 "
            f"{prediction.expected_change_pct:+.2f}% "
            f"（区间 {prediction.lower_bound_pct:+.2f}% ~ "
            f"{prediction.upper_bound_pct:+.2f}%）\n"
            f"上涨概率：{prediction.up_probability:.0%}，"
            f"置信度：{prediction.confidence:.0%}\n"
            f"主要驱动：{'；'.join(prediction.drivers) if prediction.drivers else '暂无'}\n"
            f"主要风险：{'；'.join(prediction.risks) if prediction.risks else '暂无'}"
        )

    def predict_a_share_segments(
        self,
        prediction: IndexPrediction,
        global_indices: dict | None = None,
        targets: list[tuple[str, str, float]] | None = None,
        calibration: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Split broad A-share forecast into Shanghai/Shenzhen/Beijing/STAR views."""
        global_indices = global_indices or {}
        calibration = calibration if isinstance(calibration, dict) else {}
        average_momentum = self._a_share_momentum(global_indices)
        targets = targets or [
            ("上证", "上证指数", 1.00),
            ("深证", "深证成指", 1.10),
            ("北证", "北证50", 1.30),
            ("科创", "科创50", 1.25),
        ]
        result = []
        for market, index_name, volatility in targets:
            quote = global_indices.get(index_name, {})
            if not quote and index_name == "深证成指":
                quote = global_indices.get("创业板指", {})

            has_quote = bool(quote)
            current_change = _finite_float(quote.get("change_pct")) if has_quote else 0.0
            relative_strength = current_change - average_momentum if has_quote else 0.0
            expected = prediction.expected_change_pct * volatility + relative_strength * 0.25
            calibration_item = calibration.get(index_name) or calibration.get(market) or {}
            calibration_bias = _finite_float(calibration_item.get("bias_pct") if isinstance(calibration_item, dict) else calibration_item)
            expected += calibration_bias
            expected = round(_clamp(expected, -5.0, 5.0), 2)

            half_width = max(
                0.35,
                (prediction.upper_bound_pct - prediction.lower_bound_pct) / 2 * volatility,
            )
            confidence = prediction.confidence if has_quote else max(0.20, prediction.confidence - 0.12)
            result.append(
                {
                    "market": market,
                    "index": index_name,
                    "direction": self._direction(expected),
                    "expected_change_pct": expected,
                    "lower_bound_pct": round(expected - half_width, 2),
                    "upper_bound_pct": round(expected + half_width, 2),
                    "up_probability": self._up_probability(expected),
                    "confidence": round(_clamp(confidence, 0.20, 0.85), 2),
                    "quote_change_pct": current_change if has_quote else None,
                    "data_status": "live_quote" if has_quote else "broad_market_proxy",
                    "calibration_bias_pct": round(calibration_bias, 2),
                    "calibration_samples": (
                        int(calibration_item.get("sample_count", 0))
                        if isinstance(calibration_item, dict) else 0
                    ),
                }
            )
        return result

    def format_segment_predictions(
        self,
        predictions: list[dict[str, Any]],
        title: str | None = None,
    ) -> str:
        """Format Shanghai/Shenzhen/Beijing/STAR next-day forecast lines."""
        market_names = "/".join(item.get("market", "") for item in predictions if item.get("market"))
        lines = [title or f"四、明日A股大盘预测（{market_names or '上证/深证/北证/科创'}）"]
        if not predictions:
            lines.append("暂无足够指数数据，按震荡处理。")
            return "\n".join(lines)

        for item in predictions:
            quote_note = ""
            if item.get("quote_change_pct") is not None:
                quote_note = f"，今日{item['quote_change_pct']:+.2f}%"
            elif item.get("data_status") == "broad_market_proxy":
                quote_note = "，用全市场代理"
            calibration_note = ""
            if item.get("calibration_bias_pct"):
                calibration_note = f"，误差校准{item['calibration_bias_pct']:+.2f}%"
            lines.append(
                f"{item['market']}（{item['index']}）：{item['direction']}，"
                f"预计{item['expected_change_pct']:+.2f}% "
                f"（{item['lower_bound_pct']:+.2f}%~{item['upper_bound_pct']:+.2f}%），"
                f"上涨概率{item['up_probability']:.0%}，置信度{item['confidence']:.0%}"
                f"{quote_note}{calibration_note}"
            )
        return "\n".join(lines)

    def _a_share_momentum(self, indices: dict) -> float:
        csi300 = _finite_float(indices.get("沪深300", {}).get("change_pct"))
        sh = _finite_float(indices.get("上证指数", {}).get("change_pct"))
        sz = _finite_float(indices.get("深证成指", {}).get("change_pct"))
        cyb = _finite_float(indices.get("创业板指", {}).get("change_pct"))
        star = _finite_float(indices.get("科创50", {}).get("change_pct"))
        sz_growth = sz if sz else cyb
        growth = sz_growth * 0.70 + star * 0.30 if star else sz_growth
        return _clamp(csi300 * 0.38 + sh * 0.28 + growth * 0.34, -3.0, 3.0)

    def _global_lead(self, indices: dict) -> float:
        spx = _finite_float(indices.get("标普500", {}).get("change_pct"))
        nasdaq = _finite_float(indices.get("纳斯达克", {}).get("change_pct"))
        dow = _finite_float(indices.get("道琼斯", {}).get("change_pct"))
        hsi = _finite_float(indices.get("恒生指数", {}).get("change_pct"))
        hstech = _finite_float(indices.get("恒生科技", {}).get("change_pct"))
        return _clamp(
            spx * 0.25 + nasdaq * 0.25 + dow * 0.10 + hsi * 0.20 + hstech * 0.20,
            -3.0,
            3.0,
        )

    def _risk_asset_signal(self, crypto_data: dict, gold_data: dict) -> float:
        btc = _finite_float(crypto_data.get("BTC/USDT", {}).get("change_pct"))
        eth = _finite_float(crypto_data.get("ETH/USDT", {}).get("change_pct"))
        gold = _finite_float(gold_data.get("change_pct"))
        crypto = btc * 0.60 + eth * 0.40
        # Gold strength helps only mildly; sharp gold rallies can also signal risk aversion.
        return _clamp(crypto * 0.12 + gold * 0.05, -1.5, 1.5)

    def _geo_signal(self, geo: dict) -> float:
        score = (
            _finite_float(geo.get("market_direction")) * 0.35
            + _finite_float(geo.get("policy_signal")) * 0.30
            + _finite_float(geo.get("china_us_temperature")) * 0.20
            + _finite_float(geo.get("geo_risk_index")) * 0.15
        )
        return _clamp(score * 1.2, -1.5, 1.5)

    def _lgb_breadth(self, bullish: list[dict], bearish: list[dict]) -> float:
        scores = [_finite_float(item.get("score")) for item in bullish + bearish]
        if not scores:
            return 0.0
        return _clamp(sum(scores) / len(scores) * 10.0, -1.5, 1.5)

    def _up_probability(self, expected: float) -> float:
        probability = 1.0 / (1.0 + math.exp(-expected / 0.55))
        return round(_clamp(probability, 0.08, 0.92), 2)

    def _confidence(
        self,
        *,
        expected: float,
        global_indices: dict,
        crypto_data: dict,
        geo_factors: dict,
        lgb_status: dict,
    ) -> float:
        coverage = 0.0
        if global_indices:
            coverage += min(len(global_indices) / 8.0, 1.0) * 0.35
        if crypto_data:
            coverage += 0.10
        if geo_factors:
            coverage += 0.15
        if lgb_status.get("status") == "ok":
            coverage += 0.15
        strength = min(abs(expected) / 1.2, 1.0) * 0.15
        return round(_clamp(0.25 + coverage + strength, 0.20, 0.85), 2)

    def _direction(self, expected: float) -> str:
        if expected >= 0.35:
            return "看涨"
        if expected <= -0.35:
            return "看跌"
        return "震荡"

    def _drivers(
        self,
        a_momentum: float,
        global_lead: float,
        geo_signal: float,
        risk_asset: float,
        lgb_breadth: float,
        lgb_status: dict,
    ) -> list[str]:
        candidates = [
            (abs(a_momentum), f"A股收盘动量{a_momentum:+.2f}%"),
            (abs(global_lead), f"全球/港股映射{global_lead:+.2f}%"),
            (abs(geo_signal), f"消息与政策因子{geo_signal:+.2f}"),
            (abs(risk_asset), f"风险资产夜盘{risk_asset:+.2f}%"),
        ]
        if lgb_status.get("status") == "ok":
            candidates.append((abs(lgb_breadth), f"个股模型广度{lgb_breadth:+.2f}"))
        return [text for _, text in sorted(candidates, reverse=True)[:3]]

    def _risks(self, geo_factors: dict, lgb_status: dict, confidence: float) -> list[str]:
        risks: list[str] = []
        if lgb_status.get("status") != "ok":
            risks.append("个股模型广度降级，指数预测未使用有效个股模型")
        if _finite_float(geo_factors.get("geo_risk_index")) < -0.3:
            risks.append("地缘风险偏高，隔夜跳空风险上升")
        if confidence < 0.45:
            risks.append("有效数据源不足，预测只适合作为低置信度参考")
        return risks
