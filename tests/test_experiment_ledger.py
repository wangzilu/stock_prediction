"""Tests for tracker.experiment_ledger.

Covers the 2026-06-05 contract — every experiment writes one ledger
line, the line round-trips back to a dict, filter_runs slices by the
project lead's expected columns (model_profile / split_config /
data_end / code_commit).
"""
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


def _ledger():
    from tracker import experiment_ledger as L
    return L


# -------------------------------------------------------------------
# LedgerEntry contract
# -------------------------------------------------------------------

def test_entry_requires_model_profile():
    """Lead's framing: 'each candidate's evidence must say which model'.
    Empty model_profile must fail loudly, not silently produce a row
    that future filter_runs can't group."""
    L = _ledger()
    with pytest.raises(ValueError):
        L.LedgerEntry(
            experiment_id="x", model_profile="", code_commit="abc",
            feature_count=242, data_end="2026-05-19",
            split_config="standard_24split", cache_path="/x.parquet",
        )


def test_entry_requires_experiment_id():
    L = _ledger()
    with pytest.raises(ValueError):
        L.LedgerEntry(
            experiment_id="", model_profile="xgb_242", code_commit="abc",
            feature_count=242, data_end="2026-05-19",
            split_config="standard_24split", cache_path="/x.parquet",
        )


def test_entry_jsonl_round_trips(tmp_path):
    L = _ledger()
    entry = L.LedgerEntry(
        experiment_id="xgb_242_24split_20260605_220000",
        model_profile="xgb_242",
        code_commit="abc123def456",
        feature_count=242,
        data_end="2026-05-19",
        split_config="standard_24split",
        cache_path="data/storage/feature_cache_242_production.parquet",
        feature_groups=["fundamental", "capital_flow"],
        dropped_groups=[],
        metrics={"rank_ic_mean": 0.0785, "spread_top20": 226.61},
        artifact_dir="data/storage/experiments/xgb_242_24split_20260605_220000",
    )
    line = entry.to_jsonl()
    parsed = json.loads(line)
    assert parsed["model_profile"] == "xgb_242"
    assert parsed["feature_count"] == 242
    assert parsed["metrics"]["rank_ic_mean"] == 0.0785
    assert parsed["feature_groups"] == ["fundamental", "capital_flow"]


# -------------------------------------------------------------------
# record_run + read_ledger lifecycle
# -------------------------------------------------------------------

def test_record_then_read_round_trip(tmp_path):
    L = _ledger()
    ledger = tmp_path / "ledger.jsonl"
    L.record_run(
        experiment_id="xgb_174_24split_20260605_221630",
        model_profile="xgb_174",
        feature_count=174,
        data_end="2026-05-19",
        split_config="standard_24split",
        cache_path=str(tmp_path / "feature_cache_174.parquet"),
        feature_groups=["capital_flow"],
        metrics={"rank_ic_mean": 0.058, "spread_top20": 226.61},
        code_commit="testcommit123",
        ledger_path=ledger,
    )
    rows = L.read_ledger(ledger)
    assert len(rows) == 1
    assert rows[0]["model_profile"] == "xgb_174"
    assert rows[0]["metrics"]["rank_ic_mean"] == 0.058
    assert rows[0]["code_commit"] == "testcommit123"


def test_record_run_appends_not_overwrites(tmp_path):
    """The cross-experiment ledger is append-only — a second run for
    the same model_profile must NOT clobber the first run's row, only
    add a new one."""
    L = _ledger()
    ledger = tmp_path / "ledger.jsonl"
    L.record_run(
        experiment_id="xgb_242_24split_run1",
        model_profile="xgb_242",
        feature_count=242, data_end="2026-05-19",
        split_config="standard_24split",
        cache_path="/data/cache_v1.parquet",
        metrics={"rank_ic_mean": 0.05},
        code_commit="commit_v1",
        ledger_path=ledger,
    )
    L.record_run(
        experiment_id="xgb_242_24split_run2",
        model_profile="xgb_242",
        feature_count=242, data_end="2026-05-19",
        split_config="standard_24split",
        cache_path="/data/cache_v2.parquet",
        metrics={"rank_ic_mean": 0.06},
        code_commit="commit_v2",
        ledger_path=ledger,
    )
    rows = L.read_ledger(ledger)
    assert len(rows) == 2
    assert rows[0]["code_commit"] == "commit_v1"
    assert rows[1]["code_commit"] == "commit_v2"


def test_read_ledger_skips_malformed_lines(tmp_path):
    """A truncated/corrupt line must not break the whole ledger read.
    The next run depends on read_ledger to bring back a usable
    cross-experiment view even after a partial write incident."""
    L = _ledger()
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        '{"experiment_id":"good","model_profile":"xgb_174","code_commit":"abc","feature_count":174,"data_end":"2026-05-19","split_config":"standard_24split","cache_path":"/c.parquet","feature_groups":[],"dropped_groups":[],"metrics":{},"artifact_dir":"","ts":"2026-06-05T22:00:00","extra":{}}\n'
        'this is not json\n'
        '{"experiment_id":"good2","model_profile":"xgb_242","code_commit":"def","feature_count":242,"data_end":"2026-05-19","split_config":"standard_24split","cache_path":"/c.parquet","feature_groups":[],"dropped_groups":[],"metrics":{},"artifact_dir":"","ts":"2026-06-05T22:01:00","extra":{}}\n'
    )
    rows = L.read_ledger(ledger)
    assert len(rows) == 2
    assert {r["experiment_id"] for r in rows} == {"good", "good2"}


# -------------------------------------------------------------------
# filter_runs — the slice tool the comparator will use
# -------------------------------------------------------------------

def test_filter_runs_by_model_profile_and_split(tmp_path):
    L = _ledger()
    ledger = tmp_path / "ledger.jsonl"
    for prof, split_cfg in (
        ("xgb_174", "standard_24split"),
        ("xgb_174", "fast_6split"),
        ("xgb_242", "standard_24split"),
        ("xgb175", "standard_24split"),
    ):
        L.record_run(
            experiment_id=f"{prof}_{split_cfg}",
            model_profile=prof, feature_count=200, data_end="2026-05-19",
            split_config=split_cfg, cache_path="/c.parquet",
            metrics={"rank_ic_mean": 0.05}, code_commit="abc",
            ledger_path=ledger,
        )
    # Three-way head-to-head row set — exactly what the project lead
    # asked for as the entry point to "who wins on the same exam".
    head_to_head = L.filter_runs(split_config="standard_24split", ledger_path=ledger)
    profiles = sorted(r["model_profile"] for r in head_to_head)
    assert profiles == ["xgb175", "xgb_174", "xgb_242"]


def test_filter_runs_by_code_commit_prefix(tmp_path):
    """The comparator should be able to ask 'all runs from commit
    72aa580...' even when callers stored the full 40-char hash. Prefix
    match keeps short hashes (12-char) compatible with long hashes."""
    L = _ledger()
    ledger = tmp_path / "ledger.jsonl"
    L.record_run(
        experiment_id="r1", model_profile="xgb_174",
        feature_count=174, data_end="2026-05-19",
        split_config="standard_24split", cache_path="/c.parquet",
        metrics={}, code_commit="72aa580b0f8341bca533",
        ledger_path=ledger,
    )
    L.record_run(
        experiment_id="r2", model_profile="xgb_174",
        feature_count=174, data_end="2026-05-19",
        split_config="standard_24split", cache_path="/c.parquet",
        metrics={}, code_commit="122f4220abcdef",
        ledger_path=ledger,
    )
    hits = L.filter_runs(code_commit="72aa580", ledger_path=ledger)
    assert len(hits) == 1
    assert hits[0]["experiment_id"] == "r1"
