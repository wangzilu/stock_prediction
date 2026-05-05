import pytest
from factors.geopolitical import GeopoliticalScorer


@pytest.fixture
def scorer():
    return GeopoliticalScorer()


def test_geo_risk_index_empty(scorer):
    """Empty articles should return neutral score."""
    assert scorer.compute_geo_risk_index([]) == 0.0


def test_geo_risk_index_negative_tone(scorer):
    """Negative tone articles should give lower risk index."""
    articles = [
        {"title": "Military conflict escalation in region", "tone": -8.0},
        {"title": "War threatens economic stability", "tone": -6.0},
        {"title": "Crisis deepens as sanctions imposed", "tone": -7.0},
    ]
    score = scorer.compute_geo_risk_index(articles)
    assert -1.0 <= score <= 0.0  # Should be negative (high risk)


def test_geo_risk_index_positive_tone(scorer):
    """Positive tone articles should give higher risk index."""
    articles = [
        {"title": "Peace talks progress well", "tone": 5.0},
        {"title": "Economic cooperation strengthens", "tone": 6.0},
    ]
    score = scorer.compute_geo_risk_index(articles)
    assert score > 0.0  # Should be positive (low risk)


def test_china_us_temperature(scorer):
    """China-US temperature should reflect tone of articles."""
    hostile = [
        {"title": "Trade war escalates", "tone": -7.0},
        {"title": "Sanctions imposed on China", "tone": -5.0},
    ]
    assert scorer.compute_china_us_temperature(hostile) < 0

    friendly = [
        {"title": "Trade deal progress", "tone": 5.0},
        {"title": "Cooperation agreement signed", "tone": 6.0},
    ]
    assert scorer.compute_china_us_temperature(friendly) > 0


def test_policy_signal(scorer):
    """Policy signal should detect hawkish vs dovish news."""
    hawkish_news = [
        {"title": "Fed signals rate hike ahead", "description": "tightening policy"},
        {"title": "Inflation remains elevated", "description": "restrictive stance"},
    ]
    assert scorer.compute_policy_signal(hawkish_news) < 0

    dovish_news = [
        {"title": "Fed hints at rate cut", "description": "easing monetary policy"},
        {"title": "Stimulus package announced", "description": "support growth"},
    ]
    assert scorer.compute_policy_signal(dovish_news) > 0


def test_safe_haven_signal(scorer):
    """Safe haven signal should be high during crisis."""
    crisis_articles = [
        {"title": "War escalation drives gold demand", "tone": -8.0, "description": ""},
        {"title": "Flight to safety as recession fears grow", "tone": -6.0, "description": ""},
    ]
    macro_news = [
        {"title": "Gold hits record high on uncertainty", "description": "safe haven demand surges"},
    ]
    signal = scorer.compute_safe_haven_signal(crisis_articles, macro_news)
    assert signal > 0.3  # Should indicate meaningful safe haven demand


def test_safe_haven_signal_calm(scorer):
    """Safe haven signal should be low during calm times."""
    calm_articles = [
        {"title": "Markets rally on economic optimism", "tone": 6.0, "description": ""},
        {"title": "Growth outlook improves", "tone": 5.0, "description": ""},
    ]
    signal = scorer.compute_safe_haven_signal(calm_articles, [])
    assert signal < 0.3


def test_compute_all_factors(scorer):
    """compute_all_factors should return all factor scores."""
    result = scorer.compute_all_factors(
        conflict_articles=[{"title": "test", "tone": -3.0}],
        relation_articles=[{"title": "test", "tone": 2.0}],
        macro_news=[{"title": "rate cut", "description": "easing"}],
    )
    assert "geo_risk_index" in result
    assert "china_us_temperature" in result
    assert "policy_signal" in result
    assert "safe_haven_signal" in result
    for key, val in result.items():
        assert isinstance(val, float)


def test_compute_all_factors_empty(scorer):
    """All empty inputs should return all zeros."""
    result = scorer.compute_all_factors([], [], [])
    assert result["geo_risk_index"] == 0.0
    assert result["china_us_temperature"] == 0.0
    assert result["policy_signal"] == 0.0
    assert result["safe_haven_signal"] == 0.0
