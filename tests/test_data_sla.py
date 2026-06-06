"""Tests for the Phase A.7 source-specific SLA gate.

Covers:
  - config/data_sla.py contract (frequency / max_age validation,
    every PRODUCTION_GROUP_TO_HEALTH_SOURCE entry has an SLA).
  - scheduler/data_health.is_fresh_sla — per-source budget enforcement,
    fail-closed vs exempt policies for unregistered sources.
  - scheduler/data_health.sla_verdict — multi-source verdict with
    fresh / stale / exempt buckets.
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


# -------------------------------------------------------------------
# config/data_sla.py contract
# -------------------------------------------------------------------

def test_sla_dataclass_rejects_bad_frequency():
    from config.data_sla import SourceSLA
    with pytest.raises(ValueError):
        SourceSLA(frequency="yearly", max_age_trading_days=180)  # type: ignore[arg-type]


def test_sla_dataclass_rejects_negative_budget():
    from config.data_sla import SourceSLA
    with pytest.raises(ValueError):
        SourceSLA(frequency="daily", max_age_trading_days=-1)


def test_sla_map_covers_every_production_group_source():
    """Every PRODUCTION_GROUP_TO_HEALTH_SOURCE entry must have an SLA.
    Without this assertion a new production group could silently
    bypass the freshness gate."""
    from config.data_sla import SLA_BY_SOURCE
    from scheduler.data_health import PRODUCTION_GROUP_TO_HEALTH_SOURCE
    referenced = set(PRODUCTION_GROUP_TO_HEALTH_SOURCE.values())
    missing = referenced - set(SLA_BY_SOURCE)
    assert not missing, (
        f"PRODUCTION_GROUP_TO_HEALTH_SOURCE references health sources "
        f"with no SLA entry: {sorted(missing)}. Add them to "
        f"config.data_sla.SLA_BY_SOURCE."
    )


def test_quarterly_sources_have_at_least_one_quarter_budget():
    """quarterly sources must allow >= 60 trading days of lag.
    Anything tighter would force the gate red between disclosures."""
    from config.data_sla import SLA_BY_SOURCE
    for source, sla in SLA_BY_SOURCE.items():
        if sla.frequency == "quarterly":
            assert sla.max_age_trading_days >= 60, (
                f"quarterly source {source} has budget "
                f"{sla.max_age_trading_days}d, too tight for "
                f"a one-quarter cycle"
            )


# -------------------------------------------------------------------
# is_fresh_sla — per-source budget enforcement
# -------------------------------------------------------------------

def _patch_health_dir(monkeypatch, tmp_path):
    """Redirect data_health's HEALTH_DIR to tmp so tests do not write
    into the real data/storage tree."""
    import scheduler.data_health as dh
    monkeypatch.setattr(dh, "HEALTH_DIR", tmp_path)
    return dh


def _write_health(dh, tmp_path, source, *, date, latest_date,
                  success=True, partial=False):
    day_dir = tmp_path / date
    day_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "date": date,
        "finished_at": f"{date}T00:00:00",
        "success": success,
        "partial": partial,
        "n_items": 1,
        "latest_date": latest_date,
        "error_type": "",
        "error_message": "",
        "retry_count": 0,
        "network_profile": "domestic",
        "coverage": 0.0,
        "extra": {},
    }
    (day_dir / f"{source}.json").write_text(json.dumps(payload))


def test_daily_source_within_budget_is_fresh(monkeypatch, tmp_path):
    dh = _patch_health_dir(monkeypatch, tmp_path)
    _write_health(dh, tmp_path, "qlib_data_update",
                  date="2026-06-05", latest_date="2026-06-05")
    assert dh.is_fresh_sla("qlib_data_update", "2026-06-05") is True


def test_daily_source_one_day_late_is_stale(monkeypatch, tmp_path):
    """SLA(daily, 1) rejects a 2-trading-day-old latest_date."""
    dh = _patch_health_dir(monkeypatch, tmp_path)
    _write_health(dh, tmp_path, "qlib_data_update",
                  date="2026-06-05", latest_date="2026-06-03")
    # 2026-06-03 to 2026-06-05 is 2 CN trading days (Wed→Fri)
    assert dh.is_fresh_sla("qlib_data_update", "2026-06-05") is False


def test_weekly_source_3_day_lag_still_fresh(monkeypatch, tmp_path):
    """SLA(weekly, 7) accepts a Monday gate reading Friday data."""
    dh = _patch_health_dir(monkeypatch, tmp_path)
    _write_health(dh, tmp_path, "fundamental_update",
                  date="2026-06-08",  # Mon
                  latest_date="2026-06-05")  # Fri
    assert dh.is_fresh_sla("fundamental_update", "2026-06-08") is True


def test_quarterly_source_30_day_lag_still_fresh(monkeypatch, tmp_path):
    """SLA(quarterly, 65) accepts a 30-trading-day lag — typical
    between disclosure windows."""
    dh = _patch_health_dir(monkeypatch, tmp_path)
    _write_health(dh, tmp_path, "shareholder_update",
                  date="2026-06-05", latest_date="2026-04-20")
    assert dh.is_fresh_sla("shareholder_update", "2026-06-05") is True


def test_failed_health_is_stale(monkeypatch, tmp_path):
    dh = _patch_health_dir(monkeypatch, tmp_path)
    _write_health(dh, tmp_path, "qlib_data_update",
                  date="2026-06-05", latest_date="2026-06-05",
                  success=False)
    assert dh.is_fresh_sla("qlib_data_update", "2026-06-05") is False


def test_partial_health_is_stale(monkeypatch, tmp_path):
    dh = _patch_health_dir(monkeypatch, tmp_path)
    _write_health(dh, tmp_path, "qlib_data_update",
                  date="2026-06-05", latest_date="2026-06-05",
                  partial=True)
    assert dh.is_fresh_sla("qlib_data_update", "2026-06-05") is False


def test_unregistered_source_fail_closed_default(monkeypatch, tmp_path):
    dh = _patch_health_dir(monkeypatch, tmp_path)
    # Even with a green health row, an unregistered source returns False
    # under the default fail-closed policy.
    _write_health(dh, tmp_path, "fake_source",
                  date="2026-06-05", latest_date="2026-06-05")
    assert dh.is_fresh_sla("fake_source", "2026-06-05") is False


def test_unregistered_source_exempt_policy(monkeypatch, tmp_path):
    dh = _patch_health_dir(monkeypatch, tmp_path)
    assert dh.is_fresh_sla(
        "fake_source", "2026-06-05", if_unregistered="exempt"
    ) is True


# -------------------------------------------------------------------
# sla_verdict — multi-source buckets
# -------------------------------------------------------------------

def test_sla_verdict_buckets(monkeypatch, tmp_path):
    dh = _patch_health_dir(monkeypatch, tmp_path)
    _write_health(dh, tmp_path, "qlib_data_update",
                  date="2026-06-05", latest_date="2026-06-05")
    _write_health(dh, tmp_path, "valuation_update",
                  date="2026-06-05", latest_date="2026-06-03")  # stale
    _write_health(dh, tmp_path, "fundamental_update",
                  date="2026-06-05", latest_date="2026-06-04")  # weekly OK
    verdict = dh.sla_verdict(
        ["qlib_data_update", "valuation_update", "fundamental_update",
         "fake_source"],
        date="2026-06-05",
    )
    assert verdict["all_fresh"] is False
    assert set(verdict["fresh"]) == {"qlib_data_update", "fundamental_update"}
    assert set(verdict["stale"]) == {"valuation_update"}
    assert set(verdict["exempt"]) == {"fake_source"}
    # Details carries the per-source provenance for the audit doc.
    assert verdict["details"]["qlib_data_update"]["budget"] == 1
    assert verdict["details"]["fundamental_update"]["frequency"] == "weekly"
    assert verdict["details"]["valuation_update"]["reason"] == "exceeds_budget"
