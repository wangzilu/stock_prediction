"""Tests for the L5 daily LLM event-factor quality report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _add_project_root_to_path():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    yield


def _mod():
    import importlib
    return importlib.import_module("scripts.llm_factor_quality_report")


def _events():
    """Hand-crafted event mix exercising every metric the report tracks."""
    return [
        # Two routine + one downgraded earnings.
        {"qlib_code": "SH600519", "event_type": "routine_announcement",
         "direction": 0, "is_repeated_news": True,
         "is_official_disclosure": True, "is_price_sensitive": False,
         "title": "贵州茅台发布公告A",
         "publish_time": "2026-06-05 09:00:00",
         "source": "上交所"},
        {"qlib_code": "SH600519", "event_type": "routine_announcement",
         "direction": 0, "is_repeated_news": True,
         "is_official_disclosure": True, "is_price_sensitive": False,
         "title": "贵州茅台发布公告A",  # duplicate title
         "publish_time": "2026-06-05 09:01:00",
         "source": "上交所"},
        {"qlib_code": "SZ000333", "event_type": "routine_announcement",
         "event_type_original": "earnings_beat",
         "event_type_downgrade_reason": "downgraded: no keyword in earnings_beat family matched",
         "direction": 1, "is_repeated_news": False,
         "is_official_disclosure": False, "is_price_sensitive": True,
         "title": "券商：美的集团业绩预测上调",
         "publish_time": "2026-06-05 10:00:00",
         "source": "证券时报网"},
        # PIT-invalid: publish_time in the future.
        {"qlib_code": "SZ000001", "event_type": "share_buyback",
         "direction": 1, "is_repeated_news": False,
         "is_official_disclosure": True, "is_price_sensitive": True,
         "title": "平安回购股份",
         "publish_time": "2026-06-10 09:00:00",  # future
         "source": "深交所"},
        # Negative direction, official disclosure.
        {"qlib_code": "SH600000", "event_type": "regulatory_penalty",
         "direction": -1, "is_repeated_news": False,
         "is_official_disclosure": True, "is_price_sensitive": True,
         "title": "浦发被警示函",
         "publish_time": "2026-06-04 16:00:00",
         "source": "上交所"},
    ]


# -------------------------------------------------------------------
# build_report — schema and arithmetic
# -------------------------------------------------------------------

def test_events_count_matches_input():
    rep = _mod().build_report(_events(), target_date="2026-06-05")
    assert rep["events_count"] == 5


def test_stock_coverage_is_distinct_qlib_codes():
    rep = _mod().build_report(_events(), target_date="2026-06-05")
    assert rep["stock_coverage"] == 4  # 600519, 000333, 000001, 600000


def test_event_type_distribution_counts():
    rep = _mod().build_report(_events(), target_date="2026-06-05")
    assert rep["event_type_distribution"]["routine_announcement"] == 3
    assert rep["event_type_distribution"]["share_buyback"] == 1
    assert rep["event_type_distribution"]["regulatory_penalty"] == 1


def test_repeated_ratio():
    rep = _mod().build_report(_events(), target_date="2026-06-05")
    # 2 of 5 marked repeated.
    assert rep["repeated_ratio"] == 0.4


def test_schema_downgrade_metrics():
    rep = _mod().build_report(_events(), target_date="2026-06-05")
    # 1 of 5 has event_type_original set.
    assert rep["schema_downgrade_count"] == 1
    assert rep["schema_downgrade_ratio"] == 0.2


def test_pit_invalid_count_catches_future_publish_time():
    rep = _mod().build_report(_events(), target_date="2026-06-05")
    assert rep["pit_invalid_count"] == 1
    assert 0 < rep["pit_invalid_ratio"] < 1


def test_top_duplicate_titles_lists_only_duplicates():
    rep = _mod().build_report(_events(), target_date="2026-06-05")
    titles = [r["title"] for r in rep["top_duplicate_titles"]]
    assert "贵州茅台发布公告A" in titles
    assert all(r["count"] >= 2 for r in rep["top_duplicate_titles"])


def test_direction_distribution_three_buckets():
    rep = _mod().build_report(_events(), target_date="2026-06-05")
    assert rep["direction_distribution"]["0"] == 2
    assert rep["direction_distribution"]["1"] == 2
    assert rep["direction_distribution"]["-1"] == 1


def test_source_distribution_includes_top_sources():
    rep = _mod().build_report(_events(), target_date="2026-06-05")
    assert rep["source_distribution"]["上交所"] >= 2


# -------------------------------------------------------------------
# write_report — file lifecycle + atomic semantics
# -------------------------------------------------------------------

def test_write_report_round_trips_through_disk(tmp_path, monkeypatch):
    mod = _mod()
    # Redirect DATA_DIR so the report lands in tmp.
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    report = mod.build_report(_events(), target_date="2026-06-05")
    out = mod.write_report(report, "2026-06-05")
    parsed = json.loads(out.read_text())
    assert parsed["events_count"] == report["events_count"]
    assert parsed["target_date"] == "2026-06-05"


# -------------------------------------------------------------------
# empty-input safety
# -------------------------------------------------------------------

def test_empty_events_does_not_crash():
    rep = _mod().build_report([], target_date="2026-06-05")
    assert rep["events_count"] == 0
    assert rep["repeated_ratio"] == 0.0
    assert rep["schema_downgrade_ratio"] == 0.0
    assert rep["top_duplicate_titles"] == []
