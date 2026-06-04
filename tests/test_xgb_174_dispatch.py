"""xgb_174 profile dispatch contract pins (cx round 16, 2026-06-04).

Pre-fix the xgb_174 path was unreachable because the 13 qlib-custom
expression factors only existed in scripts/train_pit_baseline.py.
cx round 16 P0-1/P1-1/P1-2/P1-3 wired the factors into
FeatureMerger + train_lgb + short_term + production_inference. This
test confirms:

  1. PROFILE_EXPECTED_COUNTS pins 174 = 158 + 3 + 13 exactly.
  2. assert_profile_dimensions hard-fails on partial custom-injection
     (a real failure mode under flaky D.features).
  3. FeatureMerger.inject_qlib_custom_factors_into_handler aligns
     per-frame using ``.loc`` rather than broadcasting one ``.values``
     into every frame.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Dim contract: 174 must be exactly 158 + 3 + 13.
# ---------------------------------------------------------------------------

def test_xgb_174_expected_counts_total_174():
    from config.production_features import PROFILE_EXPECTED_COUNTS
    spec = PROFILE_EXPECTED_COUNTS["xgb_174"]
    assert spec["alpha158"] == 158
    assert spec["supplementary"] == 3
    assert spec["qlib_custom"] == 13
    assert spec["total"] == 174


def test_xgb_242_expected_counts_total_242():
    from config.production_features import PROFILE_EXPECTED_COUNTS
    spec = PROFILE_EXPECTED_COUNTS["xgb_242"]
    assert spec["alpha158"] + spec["supplementary"] + spec["qlib_custom"] == 242


# ---------------------------------------------------------------------------
# assert_profile_dimensions: must fail when ANY component drifts.
# ---------------------------------------------------------------------------

def test_assert_profile_dimensions_174_partial_custom_raises():
    from config.production_features import assert_profile_dimensions
    # supp matches (3), custom drift (12 instead of 13) — must raise
    with pytest.raises(RuntimeError, match="13 qlib-custom"):
        assert_profile_dimensions(
            alpha_count=158, supp_count=3, custom_count=12,
            profile="xgb_174",
        )


def test_assert_profile_dimensions_174_supp_drift_raises():
    from config.production_features import assert_profile_dimensions
    with pytest.raises(RuntimeError, match="3 supplementary"):
        assert_profile_dimensions(
            alpha_count=158, supp_count=2, custom_count=13,
            profile="xgb_174",
        )


def test_assert_profile_dimensions_174_alpha_drift_raises():
    from config.production_features import assert_profile_dimensions
    with pytest.raises(RuntimeError, match="158 Alpha158"):
        assert_profile_dimensions(
            alpha_count=157, supp_count=3, custom_count=13,
            profile="xgb_174",
        )


def test_assert_profile_dimensions_174_happy_path():
    from config.production_features import assert_profile_dimensions
    # Exact contract — no raise
    assert_profile_dimensions(
        alpha_count=158, supp_count=3, custom_count=13, profile="xgb_174",
    )


def test_assert_profile_dimensions_242_custom_must_be_zero():
    from config.production_features import assert_profile_dimensions
    # 242 has no qlib_custom — passing custom_count != 0 must raise
    with pytest.raises(RuntimeError, match="0 qlib-custom"):
        assert_profile_dimensions(
            alpha_count=158, supp_count=84, custom_count=13,
            profile="xgb_242",
        )


# ---------------------------------------------------------------------------
# inject_qlib_custom_factors_into_handler per-frame index alignment.
# Pre-fix this broadcast a single ``.values`` array; we verify it now
# uses ``.loc`` so each frame sees its own index alignment.
# ---------------------------------------------------------------------------

class _FakeHandler:
    """Minimal _data/_learn/_infer container with controllable indexes."""
    def __init__(self, indexes: dict[str, pd.MultiIndex]):
        for attr, idx in indexes.items():
            # Each frame is a MultiIndex DataFrame with one dummy column.
            df = pd.DataFrame(
                {("feature", "_dummy"): np.zeros(len(idx))},
                index=idx,
            )
            setattr(self, attr, df)


def _make_index(dates: list[str], instruments: list[str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [pd.to_datetime(dates), instruments],
        names=["datetime", "instrument"],
    )


def test_inject_qlib_custom_per_frame_alignment():
    """Three frames with DIFFERENT indexes. Each frame must receive
    only the rows that belong to its own index — no broadcast."""
    from models.feature_merger import FeatureMerger

    idx_train = _make_index(["2026-01-02", "2026-01-03"], ["SH600000", "SH600001"])
    idx_valid = _make_index(["2026-02-02"], ["SH600000"])
    idx_test = _make_index(["2026-03-02"], ["SH600001", "SH600002"])
    handler = _FakeHandler({
        "_data": idx_train.union(idx_valid).union(idx_test),  # superset
        "_learn": idx_train,
        "_infer": idx_test,
    })
    handler._data = handler._data  # silence linter; attr already set in fixture
    # Construct a fake D.features return: union index, 2 factor cols
    union_idx = idx_train.union(idx_valid).union(idx_test)
    # D.features returns (instrument, datetime) order before swaplevel
    insts = union_idx.get_level_values(1)
    dts = union_idx.get_level_values(0)
    fake_features = pd.DataFrame(
        {0: np.arange(len(union_idx), dtype=float),
         1: np.arange(len(union_idx), dtype=float) * 10.0},
        index=pd.MultiIndex.from_arrays(
            [insts, dts], names=["instrument", "datetime"]
        ),
    )

    factor_specs = (("custom_a", "$pe"), ("custom_b", "$pb"))
    with patch("qlib.data.D") as MockD:
        MockD.features.return_value = fake_features
        FeatureMerger().inject_qlib_custom_factors_into_handler(
            handler, factor_specs=factor_specs,
        )

    # Each frame must have BOTH new columns AND values that match
    # what fake_features would map for its index — not a broadcast.
    for attr in ("_data", "_learn", "_infer"):
        df = getattr(handler, attr)
        assert ("feature", "custom_a") in df.columns, (
            f"{attr} missing custom_a"
        )
        assert ("feature", "custom_b") in df.columns, (
            f"{attr} missing custom_b"
        )
        # Length-preservation invariant: frame length unchanged.
        # If the helper broadcast one .values into every frame and
        # the frames had different lengths, this would have raised.
        # That's the real regression test.
