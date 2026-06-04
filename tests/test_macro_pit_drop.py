"""PIT-safety pinning: macro features are zero-baseline (no leakage).

History:
- Round 3 cx review (2026-06-03 morning) caught look-ahead in the old
  `_load_macro` (iloc[-1] broadcast of LATEST macro values to every
  historical row). First fix: return None to drop the columns entirely
  from training.
- Then commit d0b4240 (2026-06-03 night, during the 22:00 0-rec
  incident response) found that dropping the columns broke train/serve
  dim alignment: the trained champion expects 242 features and was
  getting 158 (Alpha158 only) at inference. The new contract is:
  `_load_macro` returns a frame of the same shape as before but with
  ALL ZEROS, preserving the column-count contract while keeping the
  look-ahead leak closed.

What these tests now pin:
  1. _load_macro returns a frame with the 3 contracted macro_* cols
     (macro_cpi / macro_ppi / macro_m2_yoy).
  2. EVERY value in those cols is exactly 0.0 — never the real
     CPI/PPI/M2 from the parquet, even when the parquet looks valid.
  3. `_load_supplementary` still yields zero macro_* columns when
     queried via the production contract groups (because the contract
     itself lists `macro_zero_baseline`, the zero-cols are kept; the
     no-leak property is row-level).

Re-enable contract (must satisfy ALL before flipping these tests to
expect REAL macro values):
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
# Core contract — _load_macro returns a ZERO-BASELINE frame (no leakage)
# -----------------------------------------------------------------------------

EXPECTED_MACRO_COLS = ("macro_cpi", "macro_ppi", "macro_m2_yoy")


def test_load_macro_returns_zero_baseline_even_with_valid_parquet(
    merger_with_macro_parquet,
):
    """Even when macro_features.parquet exists with multi-row REAL data
    (CPI 102.x, PPI 99.x, M2 8.x), `_load_macro` must return a frame
    whose values are ALL ZERO. This preserves the trained champion's
    242-dim contract while keeping the look-ahead leak closed. The
    single-row broadcast this guards against would reappear if the
    function silently switched back to iloc[-1] over the parquet."""
    index = pd.MultiIndex.from_product(
        [pd.date_range("2026-05-01", "2026-06-03"), ["SH600519", "SZ000001"]],
        names=["datetime", "instrument"],
    )
    result = merger_with_macro_parquet._load_macro(index)
    assert result is not None, (
        "_load_macro returned None — the dim-preserving zero-baseline "
        "contract regressed. Champion model expects 242 features at "
        "inference (158 Alpha158 + 84 supplementary); dropping the 3 "
        "macro_* cols puts inference back at 158→242 default-leaf land "
        "(see 2026-06-03 22:00 incident)."
    )
    assert list(result.columns) == list(EXPECTED_MACRO_COLS), (
        f"_load_macro column names drifted: {list(result.columns)} "
        f"!= {list(EXPECTED_MACRO_COLS)}. The trained model named these "
        f"three columns; renaming them silently is a train/serve skew."
    )
    assert (result.values == 0.0).all(), (
        "_load_macro returned non-zero values — look-ahead leak has "
        "re-opened. Every cell must be 0.0 until the daily as-of "
        "macro contract lands (see module docstring)."
    )
    assert len(result) == len(index), (
        f"_load_macro returned {len(result)} rows for an index of "
        f"{len(index)} — broadcast contract violated."
    )


def test_load_macro_returns_none_with_missing_parquet(tmp_path):
    """Degraded-path sanity: without the parquet, _load_macro cannot
    synthesize the column-name template, so it returns None. The live
    production cron always has the parquet (see fetch_macro_features.py
    job), so this branch only fires for fresh environments / tests.

    NOTE: when this branch fires in production, the downstream dim
    check in `inject_supplementary_into_handler` + the contract-fail
    gate in `short_term.py` will catch the resulting 232 vs 242
    mismatch and refuse to ship predictions. That's the defense-in-
    depth pin the 22:00 incident bought us."""
    m = FeatureMerger.__new__(FeatureMerger)
    m.data_dir = tmp_path
    if hasattr(FeatureMerger, "_macro_drop_warned"):
        delattr(FeatureMerger, "_macro_drop_warned")
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-06-01"), "SH600519"),
         (pd.Timestamp("2026-06-02"), "SH600519")],
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

    # Implementation logs "macro features DIMENSION-PRESERVED as zero
    # baseline" (see d0b4240). Accept either the historical "DROPPED"
    # phrasing or the current "DIMENSION-PRESERVED" phrasing so this
    # test pins the warn-once invariant, not the exact wording.
    macro_warns = [
        r for r in caplog.records
        if "macro features DROPPED" in r.getMessage()
        or "macro features DIMENSION-PRESERVED" in r.getMessage()
    ]
    assert len(macro_warns) == 1, (
        f"warn-once contract broken: emitted {len(macro_warns)} warnings "
        "across 3 calls (should be 1)"
    )


# -----------------------------------------------------------------------------
# Integration — _load_supplementary preserves macro_* columns as ZEROS
# -----------------------------------------------------------------------------

def test_load_supplementary_emits_macro_columns_as_zero(merger_with_macro_parquet):
    """The training-facing aggregator MUST yield the contracted macro_*
    columns (dim-preserving) BUT every value must be 0.0 (PIT-safe).

    Was: "must NOT yield macro_*". Flipped 2026-06-03 night when the
    pure drop broke 158→242 dim alignment and produced 0 stock recs
    at 22:00. The new contract is: keep the column shape, zero out
    the values."""
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
    # P0-e (2026-06-04): _load_supplementary no longer accepts groups=None.
    # Use the explicit "research / all-loaders" sentinel so this test
    # still exercises the full loader fan-out the way it did before the
    # production-contract gate landed.
    from config.production_features import RESEARCH_ALL_LOADERS
    supp = merger_with_macro_parquet._load_supplementary(
        index, groups=RESEARCH_ALL_LOADERS,
    )

    assert supp is not None, (
        "_load_supplementary returned None when only macro should be live "
        "— other loaders may have stopped being mocked."
    )
    macro_cols = [c for c in supp.columns if str(c).startswith("macro_")]
    assert macro_cols, (
        "_load_supplementary returned NO macro_ columns. The dim-preserving "
        "zero-baseline contract requires the columns to be present "
        "(see _load_macro doc) even if values are zero."
    )
    macro_block = supp[macro_cols]
    assert (macro_block.values == 0.0).all(), (
        "macro_ columns contain non-zero values — look-ahead leak has "
        "reopened. Until the daily as-of macro contract lands, every "
        "macro_ value must be 0.0."
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
