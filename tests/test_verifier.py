import pytest
from pathlib import Path
from tracker.verifier import Verifier


@pytest.fixture
def verifier(tmp_path):
    """Create a verifier with temp database."""
    db_path = tmp_path / "test_tracker.db"
    return Verifier(db_path=str(db_path))


def test_record_recommendation(verifier):
    """Should store a recommendation in the database."""
    verifier.record_recommendation(
        date_str="2026-05-05",
        code="SH600519",
        name="贵州茅台",
        signal="看多",
        score=0.75,
    )
    pending = verifier.get_pending_verifications()
    assert len(pending) == 1
    assert pending[0]["code"] == "SH600519"


def test_verify_recommendation(verifier):
    """Should mark recommendation as verified with result."""
    verifier.record_recommendation(
        date_str="2026-04-28",
        code="SH600519",
        name="贵州茅台",
        signal="看多",
        score=0.75,
    )
    verifier.verify(
        date_str="2026-04-28",
        code="SH600519",
        price_at_rec=1800.0,
        price_at_verify=1860.0,
        high_price=1880.0,
        low_price=1780.0,
    )
    verified = verifier.get_verified(date_str="2026-04-28")
    assert len(verified) == 1
    assert verified[0]["return_pct"] == pytest.approx(3.33, rel=0.01)
    assert verified[0]["is_correct"] == 1


def test_get_due_verifications(verifier):
    """Should return recommendations due for verification (5 trading days)."""
    verifier.record_recommendation(
        date_str="2026-04-25",
        code="SH600519",
        name="贵州茅台",
        signal="看多",
        score=0.75,
    )
    due = verifier.get_due_verifications(today="2026-05-05")
    assert len(due) == 1


def test_get_cumulative_stats(verifier):
    """Should calculate cumulative win rate."""
    verifier.record_recommendation("2026-04-20", "SH600519", "贵州茅台", "看多", 0.8)
    verifier.record_recommendation("2026-04-20", "SZ300750", "宁德时代", "看多", 0.7)

    verifier.verify("2026-04-20", "SH600519", 1800.0, 1850.0, 1870.0, 1790.0)
    verifier.verify("2026-04-20", "SZ300750", 200.0, 195.0, 210.0, 190.0)

    stats = verifier.get_cumulative_stats()
    assert stats["total"] == 2
    assert stats["correct"] == 1
    assert stats["win_rate"] == pytest.approx(50.0)


def test_market_prediction_record_and_verify(verifier):
    """Should record and verify next-day market predictions."""
    verifier.record_market_prediction({
        "pred_date": "2026-05-07",
        "target_date": "2026-05-08",
        "target_index": "沪深300",
        "direction": "看涨",
        "expected_change_pct": 0.35,
        "lower_bound_pct": -0.1,
        "upper_bound_pct": 0.8,
        "up_probability": 0.65,
        "confidence": 0.6,
        "drivers": ["全球市场偏强"],
        "risks": [],
        "data_status": {"model_type": "rule_baseline_v1"},
    })

    due = verifier.get_due_market_predictions(today="2026-05-08")
    assert len(due) == 1

    verified = verifier.verify_due_market_predictions(
        {"沪深300": {"change_pct": 0.42}},
        today="2026-05-08",
    )

    assert len(verified) == 1
    assert verified[0]["direction_correct"] == 1
    assert verified[0]["interval_hit"] == 1

    report = verifier.generate_market_prediction_report(verified)
    assert "大盘预测印证" in report
    assert "沪深300" in report


def test_morning_final_market_predictions_verify_all_indices(verifier):
    """Should compare 9:20 final forecasts with after-close index quotes."""
    for target_index, expected in [
        ("上证指数", 0.20),
        ("深证成指", 0.35),
        ("北证50", -0.15),
        ("创业板指", 0.50),
    ]:
        verifier.record_market_prediction({
            "pred_date": "2026-05-08",
            "target_date": "2026-05-08",
            "target_index": target_index,
            "direction": "看涨" if expected > 0 else "震荡",
            "expected_change_pct": expected,
            "lower_bound_pct": expected - 0.4,
            "upper_bound_pct": expected + 0.4,
            "up_probability": 0.6,
            "confidence": 0.55,
            "source": "morning_final",
            "drivers": ["早盘动量偏强"],
            "risks": ["盘中资金可能反复"],
            "quote_change_pct": 0.1,
            "data_status": {"model_type": "rule_baseline_v1"},
        })

    verified = verifier.verify_due_market_prediction_snapshots(
        {
            "上证指数": {"change_pct": 0.10},
            "深证成指": {"change_pct": -0.20},
            "北证50": {"change_pct": -0.05},
            "创业板指": {"change_pct": 1.30},
        },
        today="2026-05-08",
        source="morning_final",
    )

    assert len(verified) == 4
    report = verifier.generate_morning_prediction_error_report(verified)
    assert "早盘最终预测复盘" in report
    assert "上证指数" in report
    assert "创业板指" in report
    assert "误差主因" in report


def test_verified_market_prediction_errors_feed_calibration(verifier):
    """Verified after-close errors should produce next-run calibration bias."""
    for i, actual in enumerate([0.8, 0.6, 0.7, 0.5, 0.9], 1):
        verifier.record_market_prediction({
            "pred_date": f"2026-05-0{i}",
            "target_date": f"2026-05-0{i}",
            "target_index": "上证指数",
            "direction": "震荡",
            "expected_change_pct": 0.1,
            "lower_bound_pct": -0.2,
            "upper_bound_pct": 0.4,
            "up_probability": 0.5,
            "confidence": 0.5,
            "source": "morning_final",
            "drivers": [],
            "risks": [],
            "data_status": {},
        })
        due = verifier.get_due_market_predictions(
            today=f"2026-05-0{i}",
            source="morning_final",
        )
        verifier.verify_market_prediction(due[0]["id"], actual, verify_date=f"2026-05-0{i}")

    calibration = verifier.get_market_prediction_calibration(source="morning_final")

    assert calibration["上证指数"]["sample_count"] == 5
    assert calibration["上证指数"]["mean_error_pct"] == pytest.approx(0.6)
    assert calibration["上证指数"]["bias_pct"] == pytest.approx(0.3)
