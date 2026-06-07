"""Regression tests for the case-mismatch / zero-coverage gates that
prevent the B.6.3 -> B.8 silent-failure pattern from recurring.

The gates protect every offline ``scripts/build_feature_cache_209_*.py``
joiner from quietly shipping constant-zero factor columns.
"""
from __future__ import annotations

import pandas as pd
import pytest

from factors.feature_cache_utils import (
    assert_join_coverage,
    normalize_instrument_index,
)


def _make_multiindex_df(
    dates: list[str],
    insts: list[str],
    cols: list[str],
    fill: float = 1.0,
) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), insts],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame(
        {c: [fill] * len(idx) for c in cols},
        index=idx,
    )


class TestNormalizeInstrumentIndex:
    def test_uppercase_to_lowercase(self):
        df = _make_multiindex_df(
            ["2026-01-01"], ["SH600000", "SZ300750"], ["val"],
        )
        normed = normalize_instrument_index(df, source_name="t")
        got = list(normed.index.get_level_values("instrument"))
        assert got == ["sh600000", "sz300750"]

    def test_already_lowercase_is_noop(self):
        df = _make_multiindex_df(
            ["2026-01-01"], ["sh600000", "sz300750"], ["val"],
        )
        normed = normalize_instrument_index(df, source_name="t")
        # exact same object reference if no change needed
        assert normed.index.equals(df.index)

    def test_mixed_case_normalizes_all(self):
        df = _make_multiindex_df(
            ["2026-01-01"], ["SH600000", "sz300750", "Sh000001"], ["val"],
        )
        normed = normalize_instrument_index(df, source_name="t")
        got = sorted(normed.index.get_level_values("instrument"))
        assert got == ["sh000001", "sh600000", "sz300750"]

    def test_non_multiindex_returns_unchanged(self):
        df = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
        out = normalize_instrument_index(df, source_name="t")
        assert out is df


class TestAssertJoinCoverage:
    def test_full_coverage_passes(self):
        src = _make_multiindex_df(
            ["2026-01-01"], ["sh600000", "sz300750"], ["x", "y"],
        )
        # reindexed = same as source (perfect match)
        reidx = src.copy()
        # should not raise
        assert_join_coverage(
            source_df=src, reindexed=reidx,
            factor_cols=["x", "y"], source_name="t",
        )

    def test_case_mismatch_zero_match_raises(self):
        """The exact B.6.3 LLM bug: source UPPERCASE, base lowercase."""
        src = _make_multiindex_df(
            ["2026-01-01"], ["SH600000", "SZ300750"], ["x"],
        )
        base_idx = pd.MultiIndex.from_product(
            [pd.to_datetime(["2026-01-01"]), ["sh600000", "sz300750"]],
            names=["datetime", "instrument"],
        )
        reidx = src.reindex(base_idx)
        # reindex returns all NaN because case differs
        assert reidx.notna().any(axis=1).sum() == 0
        with pytest.raises(RuntimeError, match="coverage-gate"):
            assert_join_coverage(
                source_df=src, reindexed=reidx,
                factor_cols=["x"], source_name="llm_event",
            )

    def test_date_range_miss_raises(self):
        """The B.8 guba bug: source dates don't overlap base."""
        src = _make_multiindex_df(
            ["2026-06-01", "2026-06-02"], ["sh600000"], ["x"],
        )
        base_idx = pd.MultiIndex.from_product(
            [pd.to_datetime(["2026-05-01"]), ["sh600000"]],
            names=["datetime", "instrument"],
        )
        reidx = src.reindex(base_idx)
        assert reidx.notna().any(axis=1).sum() == 0
        with pytest.raises(RuntimeError, match="coverage-gate"):
            assert_join_coverage(
                source_df=src, reindexed=reidx,
                factor_cols=["x"], source_name="guba",
            )

    def test_legitimately_sparse_passes(self):
        """B.7 chain pipeline (~0.01 % non-zero) is legitimately sparse
        — the gate must NOT cry wolf on it.
        """
        dates = pd.bdate_range("2026-01-01", periods=20, freq="B")
        insts = [f"sh{600000+i}" for i in range(100)]
        base_idx = pd.MultiIndex.from_product(
            [dates, insts], names=["datetime", "instrument"],
        )
        # Source has only 1 matching row out of 2000 — that's 0.05 %,
        # above the 0.001 % floor.
        src = pd.DataFrame(
            {"x": [1.0]},
            index=pd.MultiIndex.from_tuples(
                [(dates[0], "sh600000")],
                names=["datetime", "instrument"],
            ),
        )
        reidx = src.reindex(base_idx)
        # should NOT raise — sparse is OK, only "0 match despite
        # non-empty source" raises
        assert_join_coverage(
            source_df=src, reindexed=reidx,
            factor_cols=["x"], source_name="chain",
        )

    def test_empty_source_warns_not_raises(self):
        """If source itself is empty (e.g. xwlb on a day with no
        themes), we want a WARNING — not a hard fail."""
        src = pd.DataFrame(
            {"x": []},
            index=pd.MultiIndex.from_tuples(
                [], names=["datetime", "instrument"],
            ),
        )
        base_idx = pd.MultiIndex.from_product(
            [pd.to_datetime(["2026-01-01"]), ["sh600000"]],
            names=["datetime", "instrument"],
        )
        reidx = src.reindex(base_idx)
        # should NOT raise
        assert_join_coverage(
            source_df=src, reindexed=reidx,
            factor_cols=["x"], source_name="xwlb",
        )

    def test_missing_factor_cols_raises_clearly(self):
        src = _make_multiindex_df(
            ["2026-01-01"], ["sh600000"], ["x"],
        )
        reidx = src.copy()
        # request a col that doesn't exist in reindexed
        with pytest.raises(RuntimeError, match="none of the requested"):
            assert_join_coverage(
                source_df=src, reindexed=reidx,
                factor_cols=["y", "z"], source_name="t",
            )

    def test_normalize_then_assert_round_trip(self):
        """The combined pattern joiners use: normalize then assert.
        Verifies the gate accepts a case-mismatched source after
        normalization."""
        src = _make_multiindex_df(
            ["2026-01-01"], ["SH600000", "SZ300750"], ["x"],
        )
        src = normalize_instrument_index(src, source_name="t")
        base_idx = pd.MultiIndex.from_product(
            [pd.to_datetime(["2026-01-01"]), ["sh600000", "sz300750"]],
            names=["datetime", "instrument"],
        )
        reidx = src.reindex(base_idx)
        # after normalization the reindex matches both rows
        assert_join_coverage(
            source_df=src, reindexed=reidx,
            factor_cols=["x"], source_name="t",
        )
