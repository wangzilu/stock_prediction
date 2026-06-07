"""Unit tests for cx batch G additions to scheduler.job_deps."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from scheduler.job_deps import (
    JOB_DEPS,
    JOB_DEPS_PREV_BDAY,
    _is_status_fresh,
    _prev_business_day,
    _status_path,
    check_upstream,
    check_upstream_full,
    mark_complete,
)


def _redirect_status_dir(tmp_path, monkeypatch):
    """Point STATUS_DIR at a per-test tmp dir."""
    from scheduler import job_deps as jd
    monkeypatch.setattr(jd, "STATUS_DIR", tmp_path)
    return tmp_path


# --- cx batch G P1 #2 ----------------------------------------------------

def test_prev_business_day_walks_back_one_bday():
    # Mon → Fri
    assert _prev_business_day("2026-06-08") == "2026-06-05"
    # Tue → Mon
    assert _prev_business_day("2026-06-09") == "2026-06-08"


def test_prev_business_day_handles_weekend_input():
    # Sunday → Friday (pandas BDay treats Sun as 0-day-past-Fri so
    # subtracting 1 BDay returns Friday).
    assert _prev_business_day("2026-06-07") == "2026-06-05"


def test_job_deps_prev_bday_has_morning_chain():
    assert "morning_recommendation" in JOB_DEPS_PREV_BDAY
    assert "lgb_after_close_smoke" in JOB_DEPS_PREV_BDAY["morning_recommendation"]
    assert "champion_cache_rebuild" in JOB_DEPS_PREV_BDAY["morning_recommendation"]


def test_check_upstream_full_treats_missing_prev_bday_as_not_ready(tmp_path, monkeypatch):
    _redirect_status_dir(tmp_path, monkeypatch)
    # No status files anywhere — morning_recommendation should not be ready.
    res = check_upstream_full("morning_recommendation", "2026-06-08")
    assert not res["ready"]
    assert "lgb_after_close_smoke" in res["prev_bday_missing"]
    assert res["prev_bday_date"] == "2026-06-05"


def test_check_upstream_full_ready_when_prev_bday_succeeded(tmp_path, monkeypatch):
    _redirect_status_dir(tmp_path, monkeypatch)
    # Write successful status files for yesterday.
    for dep in JOB_DEPS_PREV_BDAY["morning_recommendation"]:
        mark_complete(dep, "2026-06-05", success=True, details="ok")
    res = check_upstream_full("morning_recommendation", "2026-06-08")
    assert res["ready"], f"unexpected: {res}"


# --- cx batch G P2 #6 ----------------------------------------------------

def test_shadow_paper_trade_in_job_deps():
    assert "shadow_paper_trade_generate" in JOB_DEPS
    assert "shadow_paper_trade_backfill" in JOB_DEPS


def test_shadow_paper_trade_prev_bday_deps():
    assert JOB_DEPS_PREV_BDAY["shadow_paper_trade_generate"] == [
        "lgb_after_close_smoke", "champion_cache_rebuild",
    ]
    assert JOB_DEPS_PREV_BDAY["shadow_paper_trade_backfill"] == [
        "lgb_after_close_smoke",
    ]


# --- cx batch G P2 #7 ----------------------------------------------------

def test_is_status_fresh_accepts_legacy_row_without_output_paths():
    # Pre-G rows have no output_paths field — we trust the success flag.
    fresh, reason = _is_status_fresh(
        {"success": True, "completed_at": "2026-06-07T18:30:00"}
    )
    assert fresh
    assert reason == ""


def test_is_status_fresh_flags_missing_output_path():
    fresh, reason = _is_status_fresh({
        "success": True,
        "completed_at": "2026-06-07T18:30:00",
        "output_paths": ["/nonexistent/path.parquet"],
    })
    assert not fresh
    assert "missing" in reason


def test_is_status_fresh_flags_stale_mtime(tmp_path):
    """File whose mtime is much older than completed_at trips the gate."""
    artifact = tmp_path / "stale.parquet"
    artifact.write_text("x")
    # Push mtime to one hour ago.
    one_hour_ago = (datetime.now() - timedelta(hours=1)).timestamp()
    os.utime(artifact, (one_hour_ago, one_hour_ago))

    now_iso = datetime.now().isoformat(timespec="seconds")
    fresh, reason = _is_status_fresh({
        "success": True,
        "completed_at": now_iso,
        "output_paths": [str(artifact)],
    })
    assert not fresh
    assert "older than completed_at" in reason


def test_is_status_fresh_accepts_recent_mtime(tmp_path):
    artifact = tmp_path / "fresh.parquet"
    artifact.write_text("x")
    now_iso = datetime.now().isoformat(timespec="seconds")
    fresh, reason = _is_status_fresh({
        "success": True,
        "completed_at": now_iso,
        "output_paths": [str(artifact)],
    })
    assert fresh, f"unexpected: {reason}"


def test_mark_complete_persists_output_paths(tmp_path, monkeypatch):
    _redirect_status_dir(tmp_path, monkeypatch)
    mark_complete(
        "fake_job", "2026-06-07", success=True, details="ok",
        output_paths=["/tmp/a.parquet", "/tmp/b.parquet"],
    )
    path = _status_path("fake_job", "2026-06-07")
    payload = json.loads(path.read_text())
    assert payload["output_paths"] == ["/tmp/a.parquet", "/tmp/b.parquet"]


def test_mark_complete_defaults_output_paths_to_empty(tmp_path, monkeypatch):
    _redirect_status_dir(tmp_path, monkeypatch)
    mark_complete("fake_job", "2026-06-07", success=True, details="ok")
    path = _status_path("fake_job", "2026-06-07")
    payload = json.loads(path.read_text())
    assert payload["output_paths"] == []


def test_check_upstream_demotes_stale_upstream(tmp_path, monkeypatch):
    """Upstream success row + stale artifact → upstream demoted to missing."""
    status_dir = _redirect_status_dir(tmp_path, monkeypatch)
    # Stage a stale artifact + a "fresh" success row pointing at it.
    artifact = tmp_path / "stale.parquet"
    artifact.write_text("x")
    one_hour_ago = (datetime.now() - timedelta(hours=1)).timestamp()
    os.utime(artifact, (one_hour_ago, one_hour_ago))
    mark_complete(
        "qlib_data_update", "2026-06-08", success=True, details="manual rerun",
        output_paths=[str(artifact)],
    )
    # evening_outlook gates on qlib_data_update via JOB_DEPS.
    res = check_upstream("evening_outlook", "2026-06-08")
    assert not res["ready"]
    assert "qlib_data_update" in res["missing"]
