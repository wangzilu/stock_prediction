"""Tests for Phase C.3 (L1) fact-count factor columns in
scripts/build_llm_event_factors.py.

The new columns must be emitted alongside the legacy
``llm_impact_*_decayed`` / ``llm_sentiment_score`` outputs so consumers
can adopt them gradually. The legacy synthesized-impact path stays for
one release as a deprecation window.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _add_project_root_to_path():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    yield


def _build_factors():
    import importlib
    return importlib.import_module("scripts.build_llm_event_factors").build_factors


def _stub_events(signal_date: str, *, stock="SH600519") -> list[dict]:
    """Build a small event list that exercises each fact-count column."""
    sd = datetime.strptime(signal_date, "%Y-%m-%d")
    def at(days_ago, **kwargs):
        ts = (sd - timedelta(days=days_ago)).strftime("%Y-%m-%d 09:00:00")
        return {
            "qlib_code": stock, "stock_code": stock[2:],
            "publish_time": ts, "file_date": ts[:10],
            "source": "上交所", "source_tier": "official",
            "source_quality": 1.0,
            "extractor_version": "v2",
            "confidence": 0.8,
            "relevance": 1.0, "novelty": 0.9,
            **kwargs,
        }
    return [
        # 3 positive in last 3 days
        at(0, event_type="share_buyback", direction=1, impact_1d=0.05, impact_5d=0.03,
           is_price_sensitive=True, is_official_disclosure=True, is_new_information=True,
           is_repeated_news=False),
        at(1, event_type="order_win", direction=1, impact_1d=0.04, impact_5d=0.024,
           is_price_sensitive=True, is_official_disclosure=False, is_new_information=True,
           is_repeated_news=False),
        at(2, event_type="dividend_increase", direction=1, impact_1d=0.05, impact_5d=0.03,
           is_price_sensitive=False, is_official_disclosure=True, is_new_information=True,
           is_repeated_news=True),
        # 1 negative in last 3 days
        at(2, event_type="regulatory_penalty", direction=-1, impact_1d=-0.05, impact_5d=-0.03,
           is_price_sensitive=True, is_official_disclosure=True, is_new_information=True,
           is_repeated_news=False),
        # 1 stale event (outside 3 days but inside 5 days)
        at(4, event_type="routine_announcement", direction=0, impact_1d=0.0, impact_5d=0.0,
           is_price_sensitive=False, is_official_disclosure=False, is_new_information=False,
           is_repeated_news=True),
    ]


def _patch_loader(monkeypatch, events):
    """Replace the JSONL loader so build_factors sees our fake events."""
    import scripts.build_llm_event_factors as mod
    df = pd.DataFrame(events)

    def _fake_load_events(*args, **kwargs):
        return df.copy()

    monkeypatch.setattr(mod, "load_events", _fake_load_events)


def test_fact_count_columns_are_emitted(monkeypatch):
    build_factors = _build_factors()
    _patch_loader(monkeypatch, _stub_events("2026-06-05"))
    result = build_factors(signal_date="2026-06-05", lookback_days=10)
    assert result is not None
    required = [
        "llm_event_count_5d",
        "llm_event_count_3d",
        "llm_positive_event_count_3d",
        "llm_negative_event_count_3d",
        "llm_price_sensitive_count_3d",
        "llm_official_event_count_3d",
        "llm_repeated_ratio_3d",
        "llm_event_intensity",
    ]
    missing = [c for c in required if c not in result.columns]
    assert not missing, f"missing columns: {missing}"


def test_positive_event_count_3d_matches_fixture(monkeypatch):
    build_factors = _build_factors()
    _patch_loader(monkeypatch, _stub_events("2026-06-05"))
    result = build_factors(signal_date="2026-06-05", lookback_days=10)
    row = result[result["qlib_code"] == "SH600519"].iloc[0]
    # Three positive direction events within 3 days.
    assert int(row["llm_positive_event_count_3d"]) == 3


def test_negative_event_count_3d_matches_fixture(monkeypatch):
    build_factors = _build_factors()
    _patch_loader(monkeypatch, _stub_events("2026-06-05"))
    result = build_factors(signal_date="2026-06-05", lookback_days=10)
    row = result[result["qlib_code"] == "SH600519"].iloc[0]
    assert int(row["llm_negative_event_count_3d"]) == 1


def test_price_sensitive_count_3d_matches_fixture(monkeypatch):
    build_factors = _build_factors()
    _patch_loader(monkeypatch, _stub_events("2026-06-05"))
    result = build_factors(signal_date="2026-06-05", lookback_days=10)
    row = result[result["qlib_code"] == "SH600519"].iloc[0]
    # The 3 events in the last 3 days marked price-sensitive: order_win +
    # share_buyback + regulatory_penalty.
    assert int(row["llm_price_sensitive_count_3d"]) == 3


def test_official_event_count_3d_matches_fixture(monkeypatch):
    build_factors = _build_factors()
    _patch_loader(monkeypatch, _stub_events("2026-06-05"))
    result = build_factors(signal_date="2026-06-05", lookback_days=10)
    row = result[result["qlib_code"] == "SH600519"].iloc[0]
    # share_buyback + dividend_increase + regulatory_penalty are official
    # disclosures within 3 days.
    assert int(row["llm_official_event_count_3d"]) == 3


def test_repeated_ratio_3d_is_fraction_of_three_day_events(monkeypatch):
    build_factors = _build_factors()
    _patch_loader(monkeypatch, _stub_events("2026-06-05"))
    result = build_factors(signal_date="2026-06-05", lookback_days=10)
    row = result[result["qlib_code"] == "SH600519"].iloc[0]
    # 1 repeated of 4 within 3 days.
    assert pytest.approx(float(row["llm_repeated_ratio_3d"]), abs=1e-6) == 0.25


def test_event_intensity_is_events_per_day(monkeypatch):
    build_factors = _build_factors()
    _patch_loader(monkeypatch, _stub_events("2026-06-05"))
    result = build_factors(signal_date="2026-06-05", lookback_days=10)
    row = result[result["qlib_code"] == "SH600519"].iloc[0]
    # 4 events in 3 days → 4/3
    assert pytest.approx(
        float(row["llm_event_intensity"]), abs=1e-6
    ) == 4.0 / 3.0


def test_legacy_impact_columns_still_emitted_for_backcompat(monkeypatch):
    """L1 keeps the synthesized impact factors for one release. Consumers
    that haven't migrated yet must not break."""
    build_factors = _build_factors()
    _patch_loader(monkeypatch, _stub_events("2026-06-05"))
    result = build_factors(signal_date="2026-06-05", lookback_days=10)
    assert "llm_impact_1d_decayed" in result.columns
    assert "llm_impact_5d_decayed" in result.columns
    assert "llm_sentiment_score" in result.columns
