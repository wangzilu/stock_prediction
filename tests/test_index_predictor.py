from datetime import datetime
import math

from signals.index_predictor import OvernightIndexPredictor, next_weekday


def test_next_weekday_skips_weekend():
    assert next_weekday(datetime(2026, 5, 8).date()).isoformat() == "2026-05-11"


def test_predict_returns_structured_finite_forecast():
    predictor = OvernightIndexPredictor()

    pred = predictor.predict(
        as_of=datetime(2026, 5, 7, 22, 0),
        global_indices={
            "沪深300": {"change_pct": 0.5},
            "上证指数": {"change_pct": 0.3},
            "创业板指": {"change_pct": 0.8},
            "标普500": {"change_pct": 0.4},
            "纳斯达克": {"change_pct": 0.6},
            "道琼斯": {"change_pct": 0.2},
            "恒生指数": {"change_pct": 0.5},
            "恒生科技": {"change_pct": 0.7},
        },
        geo_factors={
            "market_direction": 0.4,
            "policy_signal": 0.3,
            "china_us_temperature": 0.1,
            "geo_risk_index": 0.2,
        },
        crypto_data={"BTC/USDT": {"change_pct": 1.0}, "ETH/USDT": {"change_pct": 1.5}},
        gold_data={"change_pct": 0.2},
        top_bullish=[{"code": "SH600519", "score": 0.04}],
        top_bearish=[{"code": "SH600000", "score": -0.02}],
        lgb_status={"status": "ok", "count": 150},
    )

    assert pred.target_index == "沪深300"
    assert pred.target_date == "2026-05-08"
    assert pred.direction in ("看涨", "看跌", "震荡")
    assert pred.lower_bound_pct <= pred.expected_change_pct <= pred.upper_bound_pct
    assert 0.0 <= pred.up_probability <= 1.0
    assert 0.0 <= pred.confidence <= 1.0
    assert math.isfinite(pred.expected_change_pct)
    assert pred.data_status["model_type"] == "rule_baseline_v1"
    assert all("Qlib" not in text and "LGB" not in text for text in pred.drivers + pred.risks)


def test_format_prediction_contains_required_numbers():
    predictor = OvernightIndexPredictor()
    pred = predictor.predict(as_of=datetime(2026, 5, 7, 22, 0))

    text = predictor.format_prediction(pred)

    assert "明日大盘量化预测" in text
    assert pred.target_date in text
    assert "预计涨跌幅" in text


def test_predict_can_target_same_day_for_morning_final_forecast():
    predictor = OvernightIndexPredictor()
    pred = predictor.predict(
        as_of=datetime(2026, 5, 8, 9, 20),
        target_date="2026-05-08",
    )

    assert pred.pred_date == "2026-05-08"
    assert pred.target_date == "2026-05-08"


def test_a_share_segment_predictions_include_four_markets():
    predictor = OvernightIndexPredictor()
    pred = predictor.predict(as_of=datetime(2026, 5, 7, 22, 0))

    segments = predictor.predict_a_share_segments(
        pred,
        {
            "上证指数": {"change_pct": 0.1},
            "深证成指": {"change_pct": 0.3},
            "北证50": {"change_pct": -0.2},
            "科创50": {"change_pct": 0.5},
        },
    )
    text = predictor.format_segment_predictions(segments)

    assert [item["market"] for item in segments] == ["上证", "深证", "北证", "科创"]
    assert "四、明日A股大盘预测" in text
    assert "上证" in text
    assert "深证" in text
    assert "北证" in text
    assert "科创" in text


def test_a_share_segments_can_target_chinext_for_intraday_report():
    predictor = OvernightIndexPredictor()
    pred = predictor.predict(as_of=datetime(2026, 5, 7, 14, 30))

    segments = predictor.predict_a_share_segments(
        pred,
        {
            "上证指数": {"change_pct": 0.1},
            "深证成指": {"change_pct": 0.3},
            "北证50": {"change_pct": -0.2},
            "创业板指": {"change_pct": 0.5},
        },
        targets=[
            ("上证", "上证指数", 1.00),
            ("深证", "深证成指", 1.10),
            ("北证", "北证50", 1.30),
            ("创业板", "创业板指", 1.25),
        ],
    )
    text = predictor.format_segment_predictions(
        segments,
        title="一、下一开盘日指数预测（上证/深证/北证/创业板）",
    )

    assert [item["market"] for item in segments] == ["上证", "深证", "北证", "创业板"]
    assert "下一开盘日指数预测" in text
    assert "创业板" in text
    assert "科创" not in text


def test_segment_predictions_apply_error_calibration():
    predictor = OvernightIndexPredictor()
    pred = predictor.predict(as_of=datetime(2026, 5, 8, 9, 20), target_date="2026-05-08")

    raw = predictor.predict_a_share_segments(
        pred,
        {"上证指数": {"change_pct": 0.1}},
        targets=[("上证", "上证指数", 1.00)],
    )[0]
    calibrated = predictor.predict_a_share_segments(
        pred,
        {"上证指数": {"change_pct": 0.1}},
        targets=[("上证", "上证指数", 1.00)],
        calibration={"上证指数": {"bias_pct": 0.35, "sample_count": 6}},
    )[0]

    assert calibrated["expected_change_pct"] == round(raw["expected_change_pct"] + 0.35, 2)
    assert calibrated["calibration_bias_pct"] == 0.35
    text = predictor.format_segment_predictions([calibrated])
    assert "误差校准+0.35%" in text
