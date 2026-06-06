"""Tests for the truthful-health aggregator in scripts/update_regime_daily.py.

2026-06-06 Phase A.6 fix A6-3. Pre-fix the script wrote
``HealthStatus(success=True, n_items=5)`` no matter what — every
downstream freshness gate that anchored to ``regime_daily_update``
was lying. These tests pin the new contract:

  - Every critical sub-source ok → success=True, partial=False.
  - Any critical sub-source failed → success=False, error message
    cites the failed source(s).
  - Critical ok but a non-critical sub-source failed → success=True,
    partial=True (the row remains honest about the degradation).
  - latest_date is the MIN of critical sub-sources' own latest_dates
    so the gate cannot read a row that is fresh-by-one-source but
    stale-by-another.
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


def _mod():
    import importlib
    return importlib.import_module("scripts.update_regime_daily")


def _r(source, ok=True, n=1, latest_date="2026-06-05", error=""):
    SubResult = _mod().SubResult
    return SubResult(source=source, ok=ok, n_rows=n,
                     latest_date=latest_date, error=error)


# -------------------------------------------------------------------
# All-green case
# -------------------------------------------------------------------

def test_all_critical_ok_all_non_critical_ok():
    mod = _mod()
    results = [
        _r("margin_detail"), _r("limit_list_d"), _r("moneyflow_hsgt"),
        _r("ak_futures"), _r("ak_usdcny"),
    ]
    agg = mod._aggregate_health(results, today="2026-06-05")
    assert agg["success"] is True
    assert agg["partial"] is False
    assert agg["latest_date"] == "2026-06-05"
    assert agg["error_message"] == ""
    assert agg["n_items"] == 5


# -------------------------------------------------------------------
# Critical failure cases — the 2026-06-05 production bug
# -------------------------------------------------------------------

def test_any_critical_failure_flips_success_false():
    """The 2026-06-05 bug pattern: ST_CLIENT failed → all three critical
    sub-sources errored → aggregator still wrote success=True. The fix
    must flip success=False and surface the failing source name."""
    mod = _mod()
    results = [
        _r("margin_detail", ok=False, error="ST_CLIENT init failed"),
        _r("limit_list_d", ok=False, error="ST_CLIENT init failed"),
        _r("moneyflow_hsgt", ok=False, error="ST_CLIENT init failed"),
        _r("ak_futures"),
        _r("ak_usdcny"),
    ]
    agg = mod._aggregate_health(results, today="2026-06-05")
    assert agg["success"] is False
    assert "margin_detail" in agg["error_message"]
    assert "limit_list_d" in agg["error_message"]
    assert "moneyflow_hsgt" in agg["error_message"]


def test_single_critical_failure_still_flips_success_false():
    """One critical source down is still a hard failure — the gate
    should not be allowed to degrade silently."""
    mod = _mod()
    results = [
        _r("margin_detail"), _r("limit_list_d"),
        _r("moneyflow_hsgt", ok=False, error="api timeout"),
        _r("ak_futures"), _r("ak_usdcny"),
    ]
    agg = mod._aggregate_health(results, today="2026-06-05")
    assert agg["success"] is False
    assert "moneyflow_hsgt" in agg["error_message"]


def test_critical_failure_clears_latest_date():
    """Pre-fix code wrote latest_date=today even when every fetch
    silently failed — the gate then thought the data was fresh. With
    the fix a failed critical source yields latest_date='' so the gate
    cannot mistake the row for fresh."""
    mod = _mod()
    results = [
        _r("margin_detail"), _r("limit_list_d"),
        _r("moneyflow_hsgt", ok=False, error="api"),
        _r("ak_futures"), _r("ak_usdcny"),
    ]
    agg = mod._aggregate_health(results, today="2026-06-05")
    assert agg["latest_date"] == ""


# -------------------------------------------------------------------
# Partial degradation — non-critical failures only
# -------------------------------------------------------------------

def test_non_critical_failure_sets_partial_true_keeps_success():
    """AKShare futures going down should not flip success to False —
    futures are not in CRITICAL_SOURCES. partial=True keeps the gate
    aware that something degraded."""
    mod = _mod()
    results = [
        _r("margin_detail"), _r("limit_list_d"), _r("moneyflow_hsgt"),
        _r("ak_futures", ok=False, error="akshare timeout"),
        _r("ak_usdcny"),
    ]
    agg = mod._aggregate_health(results, today="2026-06-05")
    assert agg["success"] is True
    assert agg["partial"] is True
    assert "ak_futures" in agg["extra"]["non_critical_failures"]


def test_latest_date_is_min_of_critical_sources():
    """If margin reports 2026-06-04 but limit_list and hsgt report
    2026-06-05, latest_date is the MIN — the row should never claim
    fresher than the slowest critical source."""
    mod = _mod()
    results = [
        _r("margin_detail", latest_date="2026-06-04"),
        _r("limit_list_d", latest_date="2026-06-05"),
        _r("moneyflow_hsgt", latest_date="2026-06-05"),
        _r("ak_futures"), _r("ak_usdcny"),
    ]
    agg = mod._aggregate_health(results, today="2026-06-05")
    assert agg["success"] is True
    assert agg["latest_date"] == "2026-06-04"


# -------------------------------------------------------------------
# Mixed degradation — one critical ok, two critical failed
# -------------------------------------------------------------------

def test_critical_partial_mix_flips_success_false_but_partial_true():
    """When SOME critical sources are ok and some failed, success is
    False (any critical down = no green light) but ``partial=True``
    so the gate knows it can read sub-source extras for the
    still-fresh sources rather than treating the whole row as dead."""
    mod = _mod()
    results = [
        _r("margin_detail"),                                          # ok
        _r("limit_list_d", ok=False, error="api"),                    # failed
        _r("moneyflow_hsgt", ok=False, error="api"),                  # failed
        _r("ak_futures"), _r("ak_usdcny"),
    ]
    agg = mod._aggregate_health(results, today="2026-06-05")
    assert agg["success"] is False
    assert agg["partial"] is True


# -------------------------------------------------------------------
# Extra metadata round-trip
# -------------------------------------------------------------------

def test_extra_carries_per_source_latest_date():
    """The gate must be able to pick out per-source freshness from
    one health row. extra.latest_date_<source> is how it does that."""
    mod = _mod()
    results = [
        _r("margin_detail", latest_date="2026-06-05"),
        _r("limit_list_d", latest_date="2026-06-04"),
        _r("moneyflow_hsgt", latest_date="2026-06-05"),
        _r("ak_futures", latest_date="2026-06-04"),
        _r("ak_usdcny", latest_date="2026-06-05"),
    ]
    agg = mod._aggregate_health(results, today="2026-06-05")
    assert agg["extra"]["latest_date_margin_detail"] == "2026-06-05"
    assert agg["extra"]["latest_date_limit_list_d"] == "2026-06-04"
    assert agg["extra"]["latest_date_ak_futures"] == "2026-06-04"


def test_extra_lists_critical_sources_and_failures():
    """The aggregate extra must be self-describing — a future reader
    should be able to ask 'which sources are considered critical?'
    without re-grepping the script."""
    mod = _mod()
    results = [
        _r("margin_detail"), _r("limit_list_d"), _r("moneyflow_hsgt"),
        _r("ak_futures", ok=False, error="x"),
        _r("ak_usdcny"),
    ]
    agg = mod._aggregate_health(results, today="2026-06-05")
    assert set(agg["extra"]["critical_sources"]) == {
        "margin_detail", "limit_list_d", "moneyflow_hsgt"
    }
    assert agg["extra"]["non_critical_failures"] == ["ak_futures"]
