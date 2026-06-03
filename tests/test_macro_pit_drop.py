"""PIT-safety pinning: macro features must NOT enter training data.

Per cx code review round 3 (2026-06-03) P1: the previous
`_load_macro` implementation loaded macro_features.parquet, took
df.iloc[-1] (latest snapshot), and broadcast that snapshot to every
historical (date, stock) row in the training index. Every training row
therefore saw the LATEST macro values, not the macro values that were
known at that row's prediction time — classic look-ahead bias.

The fix drops macro from training entirely until daily as-of macro
data is available. These tests pin the contract so a future PR that
silently re-enables it will fail CI loudly.

Re-enable contract (must satisfy ALL before flipping these tests):
  1. macro_features.parquet has multiple rows with an explicit
     `available_date` column (T+1 publication conservatism).
  2. _load_macro joins via asof on available_date <= trade_date.
  3. A new test asserts each training row's macro_* value is drawn
     from a row with available_date <= that training row's trade_date.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from models.feature_merger import FeatureMerger


@pytest.fixture
def merger_with_macro_parquet(tmp_path):
    """A FeatureMerger pointed at a tmp data_dir that DOES contain a
    valid-looking macro_features.parquet. The point: even when the
    parquet exists with what looks like usable data, _load_macro MUST
    still return None."""
    macro_df = pd.DataFrame({
        "date": ["2026-06-01", "2026-06-02", "2026-06-03"],
        "cpi": [102.1, 102.2, 102.3],
        "ppi": [99.5, 99.6, 99.7],
        "m2_yoy": [0.083, 0.082, 0.081],
    })
    macro_df.to_parquet(tmp_path / "macro_features.parquet")

    m = FeatureMerger.__new__(FeatureMerger)
    m.data_dir = tmp_path
    # Reset the class-level warn-once flag so test order doesn't matter
    if hasattr(FeatureMerger, "_macro_drop_warned"):
        delattr(FeatureMerger, "_macro_drop_warned")
    return m


# -----------------------------------------------------------------------------
# Core contract — _load_macro returns None unconditionally
# -----------------------------------------------------------------------------

def test_load_macro_returns_none_even_with_valid_parquet(merger_with_macro_parquet):
    """Even when macro_features.parquet exists with multi-row data, the
    method must return None until the asof-merge re-enable contract is
    satisfied. The single-row broadcast that this guards against would
    re-appear if the function silently switched to iloc[-1] again."""
    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-05-01", "2026-06-03"), ["SH600519", "SZ000001"]],
        names=["datetime", "instrument"],
    )
    result = merger_with_macro_parquet._load_macro(index)
    assert result is None, (
        "_load_macro returned a non-None frame — macro features have been "
        "silently re-enabled without the daily as-of upgrade. Either revert "
        "this re-enable or update the re-enable contract tests."
    )


def test_load_macro_returns_none_with_missing_parquet(tmp_path):
    """Sanity: when the parquet doesn't exist, return None (was already
    the case pre-fix). Pins that the early-return short-circuit didn't
    accidentally introduce a different failure mode."""
    m = FeatureMerger.__new__(FeatureMerger)
    m.data_dir = tmp_path
    if hasattr(FeatureMerger, "_macro_drop_warned"):
        delattr(FeatureMerger, "_macro_drop_warned")
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-06-01"), "SH600519")],
        names=["datetime", "instrument"],
    )
    assert m._load_macro(index) is None


def test_warn_once_per_session(merger_with_macro_parquet, caplog):
    """Confirm we don't spam cron logs: warn once across calls, then go
    silent. The class-level flag means re-instantiating the merger does
    NOT reset it — we want one warning per process, not per object."""
    import logging
    caplog.set_level(logging.WARNING)
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-06-01"), "SH600519")],
        names=["datetime", "instrument"],
    )

    merger_with_macro_parquet._load_macro(index)
    merger_with_macro_parquet._load_macro(index)
    merger_with_macro_parquet._load_macro(index)

    macro_warns = [r for r in caplog.records
                   if "macro features DROPPED" in r.getMessage()]
    assert len(macro_warns) == 1, (
        f"warn-once contract broken: emitted {len(macro_warns)} warnings "
        "across 3 calls (should be 1)"
    )


# -----------------------------------------------------------------------------
# Integration — _load_supplementary returns no macro_* columns
# -----------------------------------------------------------------------------

def test_load_supplementary_has_no_macro_columns(merger_with_macro_parquet):
    """The training-facing aggregator MUST NOT yield any macro_* columns.
    This is the column-level contract a downstream model would otherwise
    see (and learn from)."""
    # Mock all the OTHER loaders so we isolate the macro behaviour.
    # We use real method names so if any of these get renamed, the test
    # fails loudly and a developer revisits the macro contract too.
    for loader in (
        "_load_fundamental", "_load_capital_flow", "_load_shareholder",
        "_load_valuation", "_load_northbound", "_load_quality",
        "_load_st_daily_basic", "_load_event_factors", "_load_v2_event_factors",
        "_load_meta_event_factors", "_load_geo_factors",
        "_load_cross_market_regime", "_load_supply_chain",
        "_load_alphaforge", "_load_overlay_factors",
    ):
        if hasattr(merger_with_macro_parquet, loader):
            setattr(merger_with_macro_parquet, loader,
                    MagicMock(return_value=None))

    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-06-01", "2026-06-03"), ["SH600519"]],
        names=["datetime", "instrument"],
    )
    supp = merger_with_macro_parquet._load_supplementary(index)

    if supp is not None:
        macro_cols = [c for c in supp.columns if str(c).startswith("macro_")]
        assert macro_cols == [], (
            f"_load_supplementary returned macro_ columns: {macro_cols}. "
            "Macro features are supposed to be dropped from training until "
            "daily as-of data is available."
        )


# -----------------------------------------------------------------------------
# Registry — config/data_availability.py reflects the drop
# -----------------------------------------------------------------------------

def test_data_availability_registry_has_empty_allowed_usage_for_macro():
    """The registry serves as the policy boundary for what data may
    enter training. Macro must not be allowed in any usage until
    re-enabled by contract."""
    from config.data_availability import DATA_REGISTRY

    spec = DATA_REGISTRY.get("macro_features")
    assert spec is not None, (
        "macro_features spec missing from DATA_REGISTRY — was the registry "
        "renamed or deleted? Audit the registry contract."
    )
    assert spec.allowed_usage == [], (
        f"data_availability.DATA_REGISTRY['macro_features'].allowed_usage = "
        f"{spec.allowed_usage!r}, expected []. Macro is dropped from "
        "training until daily as-of data lands."
    )
    assert spec.pit_safe_level == "unsafe", (
        "macro_features pit_safe_level changed away from 'unsafe' without "
        "the as-of upgrade contract being met."
    )
