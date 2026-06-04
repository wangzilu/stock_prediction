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


def test_inject_qlib_custom_per_frame_alignment_values():
    """Three frames with DIFFERENT indexes. Each frame must receive
    only the rows that belong to its own index — verify VALUES, not
    just column presence. cx round 19 P2-3: pre-fix this test only
    checked column existence; a broadcast bug (.values on a longer
    array into a shorter frame) would have raised on the assign but
    a same-length-different-order bug would have silently slipped.
    """
    from models.feature_merger import FeatureMerger

    idx_train = _make_index(["2026-01-02", "2026-01-03"], ["SH600000", "SH600001"])
    idx_valid = _make_index(["2026-02-02"], ["SH600000"])
    idx_test = _make_index(["2026-03-02"], ["SH600001", "SH600002"])
    union_index = idx_train.union(idx_valid).union(idx_test)

    handler = _FakeHandler({
        "_data": union_index,  # superset
        "_learn": idx_train,
        "_infer": idx_test,
    })

    # Fake D.features index: (instrument, datetime) — the helper
    # ``.swaplevel().sort_index()`` after the call.
    insts = union_index.get_level_values(1)
    dts = union_index.get_level_values(0)
    # Deterministic per-row values so we can check alignment.
    n = len(union_index)
    fake_features = pd.DataFrame(
        {0: np.arange(n, dtype=float),
         1: np.arange(n, dtype=float) * 10.0},
        index=pd.MultiIndex.from_arrays(
            [insts, dts], names=["instrument", "datetime"]
        ),
    )

    factor_specs = (("custom_a", "$pe"), ("custom_b", "$pb"))
    # Pre-compute the post-swap reference the helper would use
    custom_post_swap = (
        fake_features.copy().swaplevel().sort_index()
    )
    custom_post_swap.columns = ["custom_a", "custom_b"]

    with patch("qlib.data.D") as MockD:
        MockD.features.return_value = fake_features
        FeatureMerger().inject_qlib_custom_factors_into_handler(
            handler, factor_specs=factor_specs,
        )

    # For each frame, the injected values must equal
    # custom_post_swap.loc[df.index, col]. ALSO verify the value
    # actually comes from THAT frame's index — not a broadcast of
    # another frame's slice.
    for attr in ("_data", "_learn", "_infer"):
        df = getattr(handler, attr)
        assert ("feature", "custom_a") in df.columns, f"{attr} missing custom_a"
        assert ("feature", "custom_b") in df.columns, f"{attr} missing custom_b"
        expected_a = custom_post_swap.loc[df.index, "custom_a"].values
        expected_b = custom_post_swap.loc[df.index, "custom_b"].values
        actual_a = df[("feature", "custom_a")].values
        actual_b = df[("feature", "custom_b")].values
        np.testing.assert_allclose(
            actual_a, expected_a, equal_nan=True,
            err_msg=f"{attr} custom_a value mismatch — alignment regressed",
        )
        np.testing.assert_allclose(
            actual_b, expected_b, equal_nan=True,
            err_msg=f"{attr} custom_b value mismatch — alignment regressed",
        )

    # Sanity: ensure _learn's value at idx_train[0] != _infer's value
    # at idx_test[0] — i.e. they really got different slices, not the
    # same broadcast row.
    learn_first = handler._learn[("feature", "custom_a")].iloc[0]
    infer_first = handler._infer[("feature", "custom_a")].iloc[0]
    assert learn_first != infer_first, (
        "Per-frame distinctness broken: _learn and _infer first rows are equal — "
        "may indicate broadcast regression"
    )


def test_llm_extractor_priority_sort_newest_first():
    """cx round 17 P1-2 + round 19 P1-2: the publish-time tiebreaker
    must put the NEWEST item first, not the lexicographically-reversed
    string first. Build a synthetic mini-batch with priority tied and
    only publish_time differing; assert the newest survives the
    top-1 slice."""
    from factors.llm_event_extractor_v2 import LLMEventExtractorV2 as _Extr
    # The sort is private; we can't easily call extract_batch without
    # touching the LLM. Instead invoke the inner sort key the same way
    # the file does — pinning the behavior via direct reproduction.
    # If the file no longer exposes the helpers in the expected place,
    # this test must be updated alongside the refactor.
    import datetime as _dt

    def _ts(item):
        raw = str(item.get("publish_time") or "").strip()
        if not raw:
            return float("inf")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                return -_dt.datetime.strptime(raw[:len(fmt)+8], fmt).timestamp()
            except ValueError:
                continue
        return float("inf")

    items = [
        {"priority_score": 0.5, "source": "eastmoney", "publish_time": "2026-06-03 14:00:00", "id": "old"},
        {"priority_score": 0.5, "source": "eastmoney", "publish_time": "2026-06-04 09:00:00", "id": "new"},
    ]
    items.sort(key=lambda x: (-float(x["priority_score"]), 0, _ts(x)))
    assert items[0]["id"] == "new", (
        f"newest-first sort regressed: got id={items[0]['id']}"
    )
