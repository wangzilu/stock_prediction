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


# ── cx review 2026-06-06 — three regression guards ────────────────────

def test_zero_variance_industry_alpha_is_exactly_zero(monkeypatch):
    """When the industry-only score pool is constant (std ≈ 0), the
    output must collapse to zero — NOT raw_score * 0.2. The earlier
    fallback ``mu=0, sigma=1`` would leave ``score * SHRINK`` in the
    column, which is exactly the raw-magnitude leak A.5-1 closed."""
    import importlib
    mod = importlib.import_module("scripts.build_global_chain_factors")

    # Stub the propagate path so company-level frame is empty and
    # the SupplyChainMapper returns a constant-score industry frame.
    monkeypatch.setattr(mod, "load_edges", lambda *_a, **_kw: [])
    monkeypatch.setattr(mod, "generate_demo_events", lambda *_a, **_kw: [])

    class _ConstantMapper:
        def __init__(self, *args, **kwargs):
            pass

        def get_all_affected_stocks(self, events):
            # 4 stocks, all with the same raw alpha so std == 0.
            return {
                "sh600000": -14.0,
                "sh600001": -14.0,
                "sh600002": -14.0,
                "sh600003": -14.0,
            }

    import factors.supply_chain_mapper as scm
    monkeypatch.setattr(scm, "SupplyChainMapper", _ConstantMapper)
    # Inject a single event so the pipeline continues past the
    # ``not events`` guard. The mapper's output is what matters.
    monkeypatch.setattr(mod, "_load_pre_extracted_events",
                         lambda *_a, **_kw: [{"src_entity": "Nvidia",
                                              "topic": "AI_server"}])
    monkeypatch.setattr(mod, "propagate_scores",
                         lambda *_a, **_kw: __import__("pandas").DataFrame())

    df = mod.build_factors(target_date="2026-06-05", demo=False)
    industry_rows = df[df["level"] == "industry"]
    assert not industry_rows.empty
    assert (industry_rows["industry_level_alpha"] == 0.0).all(), (
        "zero-variance industry pool must produce 0 for every row, "
        "got: " + str(industry_rows["industry_level_alpha"].tolist())
    )
    assert (industry_rows["global_chain_alpha"] == 0.0).all()


def test_company_stocks_excluded_from_industry_zscore(monkeypatch):
    """The mu/sigma used for industry-level zscore must be computed
    AFTER filtering out company-level overlap. Otherwise a high-weight
    company-level hit can skew the surviving industry rows' zscores —
    the cx review P2 case."""
    import importlib
    import pandas as pd
    mod = importlib.import_module("scripts.build_global_chain_factors")

    monkeypatch.setattr(mod, "load_edges", lambda *_a, **_kw: [])

    class _SkewMapper:
        def __init__(self, *args, **kwargs):
            pass

        def get_all_affected_stocks(self, events):
            return {
                # the company-level overlap stock — gets dropped
                "sh600519": +20.0,
                # surviving industry stocks have constant raw = 0
                "sh600001": 0.0,
                "sh600002": 0.0,
                "sh600003": 0.0,
            }

    import factors.supply_chain_mapper as scm
    monkeypatch.setattr(scm, "SupplyChainMapper", _SkewMapper)
    monkeypatch.setattr(mod, "_load_pre_extracted_events",
                         lambda *_a, **_kw: [{"src_entity": "Nvidia",
                                              "topic": "AI_server"}])
    # Inject a company-level row for sh600519 so the dedup logic drops
    # it from the industry frame.
    dt = pd.Timestamp("2026-06-05")
    company_df = pd.DataFrame([{
        "datetime": dt, "instrument": "SH600519",
        "global_chain_alpha": 5.0,
        "global_chain_event_count": 1,
        "global_chain_pos_score": 5.0,
        "global_chain_neg_score": 0.0,
        "company_level_alpha": 5.0,
        "industry_level_alpha": 0.0,
        "level": "company",
    }]).set_index(["datetime", "instrument"])
    monkeypatch.setattr(mod, "propagate_scores",
                         lambda *_a, **_kw: company_df)

    df = mod.build_factors(target_date="2026-06-05", demo=False)
    industry_rows = df[df["level"] == "industry"]
    # The three surviving industry stocks have constant raw 0, so
    # the zero-variance branch fires and every value is 0.
    # If sh600519 had stayed in the pool (raw=20.0), the surviving
    # stocks would have a non-zero negative zscore — which is what
    # this guard prevents.
    assert (industry_rows["industry_level_alpha"] == 0.0).all()
    # And the SH600519 company row carries its original value, not
    # an industry-derived one.
    company_rows = df[df["level"] == "company"]
    assert (company_rows["company_level_alpha"] == 5.0).all()
