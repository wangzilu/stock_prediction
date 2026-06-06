"""Tests for the A5-3 fix in signals/market_judge.py.

Pre-fix the early-session block weighted geo + LLM at 0.25 + 0.15 =
0.40 of the final_score, contradicting the same comment block's
declaration that LLM/geo are "for report text, NOT for scoring".
These tests pin the post-fix behaviour: index price is the only
contributor to final_score, regardless of session hour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _add_project_root_to_path():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    yield


def _make_judge(index_pct: float, monkeypatch):
    """Return a MarketJudge with _get_index_change() pinned to `index_pct`.

    MarketJudge's constructor instantiates a live MarketCollector which
    can make network calls — patching the single helper that reads
    that collector is enough to isolate the score-aggregation logic.
    """
    from signals.market_judge import MarketJudge
    # Patch at the bound-method level so the constructor still runs.
    judge = MarketJudge.__new__(MarketJudge)  # skip __init__
    judge.collector = None
    monkeypatch.setattr(judge, "_get_index_change", lambda: index_pct)
    return judge


# -------------------------------------------------------------------
# Direction does NOT shift when geo / LLM disagree with the index
# -------------------------------------------------------------------

@pytest.mark.parametrize("index_pct", [+1.5, -1.5])
def test_judgment_score_sign_follows_index_not_geo(index_pct, monkeypatch):
    """Before the fix, a strong opposing geo signal could flip the
    direction at early session because of the 40% LLM/geo weight.
    After the fix the score sign follows the index price."""
    judge = _make_judge(index_pct, monkeypatch)
    # Strong geo signal in the opposite direction of the index.
    verdict = judge.judge(geo_factors={
        "geo_risk_index": -index_pct,
        "policy_signal": -index_pct,
        "china_us_temperature": -index_pct,
        "market_direction": -index_pct,
    })
    score = verdict.get("score", 0.0)
    assert (score > 0) == (index_pct > 0), (
        f"index_pct={index_pct} should drive score sign; got score={score}"
    )


def test_judgment_score_unchanged_when_only_geo_changes(monkeypatch):
    """Two calls with identical index but opposite geo/LLM signals
    must produce the SAME score — geo+LLM weight is 0."""
    j1 = _make_judge(+0.6, monkeypatch)
    a = j1.judge(geo_factors={
        "geo_risk_index": +1.0, "policy_signal": +1.0,
        "china_us_temperature": +1.0, "market_direction": +1.0,
    })
    j2 = _make_judge(+0.6, monkeypatch)
    b = j2.judge(geo_factors={
        "geo_risk_index": -1.0, "policy_signal": -1.0,
        "china_us_temperature": -1.0, "market_direction": -1.0,
    })
    assert a["score"] == b["score"], (
        f"geo flipped from +1 to -1, score must not change. "
        f"got a={a['score']}, b={b['score']}"
    )


def test_judgment_with_empty_geo_does_not_crash(monkeypatch):
    """Sanity: empty geo_factors handled cleanly."""
    judge = _make_judge(+0.5, monkeypatch)
    verdict = judge.judge(geo_factors={})
    assert "direction" in verdict
    assert "score" in verdict


def test_judgment_with_none_geo_does_not_crash(monkeypatch):
    """Sanity: None handled cleanly."""
    judge = _make_judge(-0.5, monkeypatch)
    verdict = judge.judge(geo_factors=None)
    assert "direction" in verdict
    assert "score" in verdict


def test_index_score_clamped_to_minus_one_one(monkeypatch):
    """The score normalisation should keep extreme index moves in
    [-1, 1] — protection against absurd inputs from the collector."""
    judge = _make_judge(+20.0, monkeypatch)
    verdict = judge.judge(geo_factors={})
    assert -1.0 <= verdict["score"] <= 1.0
