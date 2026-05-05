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
