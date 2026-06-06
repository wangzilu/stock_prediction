"""Tests for Phase D / SC-A1: global_chain_alpha split into
company-level vs industry-level sub-alphas, with industry-level
zscore + shrink + clip applied at the writer.

The previous code wrote the raw propagation score into a single
``global_chain_alpha`` column. The raw distribution had mean ~ −14
and span [-47, +15] which the project lead's 2026-06-06 critique
flagged as the source of the production buy-block leak A.5-1 closed.

After this fix:
  * ``company_level_alpha`` holds the higher-confidence company-level
    score (unchanged scale, since this path is direct propagation
    from supply_chain_edges).
  * ``industry_level_alpha`` holds a *zscore-by-date × 0.2 × clip[±3]*
    transform of the industry mapper output.
  * ``global_chain_alpha`` keeps the company-level value for company
    rows and the *shrunk* industry value for industry rows — anyone
    reading this column gets a sane scale.
  * a new ``level`` column tags rows ``"company"`` / ``"industry"`` so
    downstream consumers can decide what to consume.

These tests do NOT exercise the propagation network; they call
``build_factors(demo=True)`` (synthetic events) and check the writer.
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


def _build():
    import importlib
    return importlib.import_module("scripts.build_global_chain_factors").build_factors


def test_new_columns_present_on_demo_run():
    """Smoke: the demo path produces a non-empty DataFrame with the
    new ``company_level_alpha`` / ``industry_level_alpha`` / ``level``
    columns."""
    build_factors = _build()
    df = build_factors(target_date="2026-06-05", demo=True)
    assert df is not None and not df.empty
    for col in ("company_level_alpha", "industry_level_alpha",
                 "level", "global_chain_alpha"):
        assert col in df.columns, f"missing column: {col}"


def test_company_rows_have_zero_industry_alpha():
    """Company-level propagation rows must have
    industry_level_alpha == 0 (the industry-level path injects its own
    value for industry rows only)."""
    build_factors = _build()
    df = build_factors(target_date="2026-06-05", demo=True)
    company_rows = df[df["level"] == "company"]
    if not company_rows.empty:
        # All company rows should have industry_level_alpha == 0
        assert (company_rows["industry_level_alpha"] == 0.0).all(), (
            "company rows must not carry industry_level_alpha"
        )


def test_industry_rows_have_zero_company_alpha():
    build_factors = _build()
    df = build_factors(target_date="2026-06-05", demo=True)
    industry_rows = df[df["level"] == "industry"]
    if not industry_rows.empty:
        assert (industry_rows["company_level_alpha"] == 0.0).all()


def test_industry_alpha_is_shrunk_to_clip_range():
    """Industry-level alpha must lie within [-3, 3] after shrink+clip,
    regardless of the raw mapper output. Without this guarantee the
    consumer that read raw industry values (mean ~ -14) would still
    misfire."""
    build_factors = _build()
    df = build_factors(target_date="2026-06-05", demo=True)
    industry_rows = df[df["level"] == "industry"]
    if industry_rows.empty:
        pytest.skip("demo run produced no industry rows")
    assert industry_rows["industry_level_alpha"].min() >= -3.0
    assert industry_rows["industry_level_alpha"].max() <= 3.0
    assert industry_rows["global_chain_alpha"].min() >= -3.0
    assert industry_rows["global_chain_alpha"].max() <= 3.0


def test_global_chain_alpha_equals_company_for_company_rows():
    """Back-compat: for company-level rows, global_chain_alpha must be
    the company_level_alpha value (since the original code wrote that
    column without distinguishing levels)."""
    build_factors = _build()
    df = build_factors(target_date="2026-06-05", demo=True)
    company_rows = df[df["level"] == "company"]
    if company_rows.empty:
        pytest.skip("demo run produced no company rows")
    assert (
        company_rows["global_chain_alpha"]
        == company_rows["company_level_alpha"]
    ).all()


def test_global_chain_alpha_equals_industry_for_industry_rows():
    build_factors = _build()
    df = build_factors(target_date="2026-06-05", demo=True)
    industry_rows = df[df["level"] == "industry"]
    if industry_rows.empty:
        pytest.skip("demo run produced no industry rows")
    assert (
        industry_rows["global_chain_alpha"]
        == industry_rows["industry_level_alpha"]
    ).all()
