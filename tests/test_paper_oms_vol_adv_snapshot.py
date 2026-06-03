"""Pin: PaperOMS fill sites actually use vol/ADV snapshot for sqrt_adv.

Per cx code review round 3 P2 #84: the previous PR added
_compute_slippage as a chokepoint but all 4 fill sites passed
vol=None, adv=None. With cost_model=CostModel(impact_model="sqrt_adv")
the CostModel internally fell back to bare slippage_rate — sqrt_adv
was dead code in production paper.

This PR adds vol_adv_snapshot wiring. The tests here pin:

  1. PaperOMS._lookup_vol_adv returns (None, None) when snapshot
     missing or code not present (graceful fallback).
  2. _lookup_vol_adv supports both dict-of-dict AND dict-of-tuple
     shape (so callers can choose either).
  3. With a snapshot AND cost_model in sqrt_adv mode, _compute_slippage
     receives non-None inputs and produces a different slippage than
     the bare-rate path.
  4. paper.cost_inputs.build_vol_adv_snapshot produces the expected
     dict from a synthetic price panel.
  5. paper.cost_inputs cache round-trip preserves the dict.

Tests do NOT exercise the live qlib loader — they inject a fixture
loader to keep tests offline.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.cost_model import CostModel


# -----------------------------------------------------------------------------
# PaperOMS._lookup_vol_adv
# -----------------------------------------------------------------------------

def test_lookup_returns_none_pair_when_snapshot_missing(tmp_path):
    from paper.oms import PaperOMS
    oms = PaperOMS(state_dir=str(tmp_path))
    assert oms._lookup_vol_adv("SH600519") == (None, None)


def test_lookup_returns_none_pair_when_code_not_in_snapshot(tmp_path):
    from paper.oms import PaperOMS
    snapshot = {"SH600519": {"vol": 0.02, "adv": 1e9}}
    oms = PaperOMS(state_dir=str(tmp_path), vol_adv_snapshot=snapshot)
    assert oms._lookup_vol_adv("SZ000001") == (None, None)


def test_lookup_accepts_dict_of_dict_shape(tmp_path):
    from paper.oms import PaperOMS
    snapshot = {"SH600519": {"vol": 0.02, "adv": 1e9}}
    oms = PaperOMS(state_dir=str(tmp_path), vol_adv_snapshot=snapshot)
    assert oms._lookup_vol_adv("SH600519") == (0.02, 1e9)


def test_lookup_accepts_dict_of_tuple_shape(tmp_path):
    from paper.oms import PaperOMS
    snapshot = {"SH600519": (0.025, 5e8)}
    oms = PaperOMS(state_dir=str(tmp_path), vol_adv_snapshot=snapshot)
    assert oms._lookup_vol_adv("SH600519") == (0.025, 5e8)


# -----------------------------------------------------------------------------
# Slippage actually changes when snapshot is provided
# -----------------------------------------------------------------------------

def test_slippage_uses_snapshot_when_cost_model_sqrt_adv(tmp_path):
    """End-to-end check: with cost_model=sqrt_adv AND a populated
    snapshot, _compute_slippage at a specific code's amount produces
    a different value than the bare-rate fallback path."""
    from paper.oms import PaperOMS

    cm = CostModel(impact_model="sqrt_adv", impact_coefficient=0.5)
    snapshot = {"SH600519": {"vol": 0.02, "adv": 1e8}}
    oms_with = PaperOMS(state_dir=str(tmp_path / "with"),
                          cost_model=cm, vol_adv_snapshot=snapshot)
    oms_without = PaperOMS(state_dir=str(tmp_path / "without"),
                             cost_model=cm)

    # NB: parameter triplet must be chosen so the sqrt_adv math does
    # NOT coincidentally land on the bare-rate value. With vol=0.02,
    # adv=1e8, coefficient=0.5, an amount of 1e7 yields slip_rate ≈
    # 0.00316 (sqrt_adv) vs 0.001 (bare) — comfortably distinct.
    amount = 10_000_000.0
    vol, adv = oms_with._lookup_vol_adv("SH600519")
    slip_with = oms_with._compute_slippage(amount, daily_volatility=vol, adv=adv)
    slip_without = oms_without._compute_slippage(amount)
    assert not math.isclose(slip_with, slip_without, rel_tol=1e-9), (
        f"vol/adv snapshot did not change slippage: with={slip_with}, "
        f"without={slip_without}. The wiring is still dead code."
    )
    # And concretely, sqrt_adv slippage should be HIGHER than bare for
    # this trade size (10% of ADV is significant).
    assert slip_with > slip_without


def test_fill_loops_call_lookup_for_code(tmp_path, monkeypatch):
    """Source-level anti-regression: the 4 fill-site lines in paper/
    oms.py must call _lookup_vol_adv(code) immediately before
    _compute_slippage. If a future refactor removes the lookup, the
    sqrt_adv wiring goes dead again — this test fails loudly."""
    src = (Path(__file__).resolve().parents[1] / "paper" / "oms.py").read_text(
        encoding="utf-8")
    # Count occurrences of the paired pattern
    lookup_count = src.count("_lookup_vol_adv(code)")
    fill_calls = src.count(
        "_compute_slippage(amount, daily_volatility=_vol, adv=_adv)"
    )
    assert lookup_count >= 4, (
        f"expected >= 4 _lookup_vol_adv(code) calls in fill sites, got "
        f"{lookup_count}"
    )
    assert fill_calls >= 4, (
        f"expected >= 4 vol/adv-aware _compute_slippage calls, got {fill_calls}"
    )


# -----------------------------------------------------------------------------
# paper.cost_inputs.build_vol_adv_snapshot
# -----------------------------------------------------------------------------

def _make_synthetic_panel(seed=0, n_days=30, codes=None):
    """Build a (date, code)-indexed frame with `close` + `amount`."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-05-01", periods=n_days)
    codes = codes or ["SH600519", "SZ000001", "SZ300750"]
    idx = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])
    n = len(idx)
    # Random walk closes
    raw = pd.DataFrame({
        "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, size=n))),
        "amount": rng.uniform(5e7, 1e9, size=n),
    }, index=idx)
    return raw


def test_build_snapshot_returns_expected_shape():
    from paper.cost_inputs import build_vol_adv_snapshot
    panel = _make_synthetic_panel(n_days=30)

    def _loader(start, end, universe):
        return panel

    snap = build_vol_adv_snapshot("2026-05-30", lookback_days=20,
                                    qlib_loader=_loader)
    assert isinstance(snap, dict)
    assert len(snap) > 0
    for code, val in snap.items():
        assert "vol" in val
        assert "adv" in val
        assert val["vol"] > 0
        assert val["adv"] > 0


def test_build_snapshot_excludes_codes_without_enough_lookback():
    """If only 5 days of data exist but lookback=20, the snapshot
    must be empty (callers fall back to bare rate per-code)."""
    from paper.cost_inputs import build_vol_adv_snapshot
    panel = _make_synthetic_panel(n_days=5)
    snap = build_vol_adv_snapshot("2026-05-30", lookback_days=20,
                                    qlib_loader=lambda *a, **k: panel)
    assert snap == {}


def test_build_snapshot_skips_codes_with_zero_amount():
    """Stocks whose ADV is 0 / NaN over the window must be omitted —
    sqrt_adv math would divide by zero otherwise."""
    from paper.cost_inputs import build_vol_adv_snapshot
    panel = _make_synthetic_panel(n_days=30)
    # Zero out one code's amount
    panel.loc[(slice(None), "SZ000001"), "amount"] = 0.0
    snap = build_vol_adv_snapshot("2026-05-30", lookback_days=20,
                                    qlib_loader=lambda *a, **k: panel)
    assert "SH600519" in snap
    assert "SZ000001" not in snap, (
        "code with zero ADV must be omitted to avoid div-by-zero downstream"
    )


def test_cache_round_trip(tmp_path, monkeypatch):
    from paper import cost_inputs as ci
    monkeypatch.setattr(ci, "CACHE_DIR", tmp_path)
    panel = _make_synthetic_panel(n_days=30)
    snap = ci.load_or_build_snapshot(
        "2026-05-30", lookback_days=20,
        qlib_loader=lambda *a, **k: panel,
    )
    assert len(snap) > 0

    # Second call reads from cache (loader injection NOT used; raise
    # if called so we know cache was hit).
    def _should_not_be_called(*a, **k):
        raise AssertionError("loader should NOT be called on cache hit")

    snap2 = ci.load_or_build_snapshot(
        "2026-05-30", lookback_days=20,
        qlib_loader=_should_not_be_called,
    )
    assert snap2 == snap


def test_force_rebuild_bypasses_cache(tmp_path, monkeypatch):
    from paper import cost_inputs as ci
    monkeypatch.setattr(ci, "CACHE_DIR", tmp_path)
    panel_v1 = _make_synthetic_panel(seed=0, n_days=30)
    panel_v2 = _make_synthetic_panel(seed=99, n_days=30)

    # First call populates cache with v1
    snap_v1 = ci.load_or_build_snapshot(
        "2026-05-30", qlib_loader=lambda *a, **k: panel_v1,
    )
    assert snap_v1, "first build should produce a non-empty snapshot"

    # force_rebuild=True overwrites cache with v2
    snap_v2_forced = ci.load_or_build_snapshot(
        "2026-05-30", qlib_loader=lambda *a, **k: panel_v2,
        force_rebuild=True,
    )
    assert snap_v2_forced != snap_v1, (
        "force_rebuild did not bypass cache — got v1 back"
    )

    # Subsequent non-forced call should read the now-updated cache (v2)
    snap_cached = ci.load_or_build_snapshot(
        "2026-05-30", qlib_loader=lambda *a, **k: panel_v1,  # ignored
    )
    assert snap_cached == snap_v2_forced, (
        "cache was not updated by force_rebuild"
    )
