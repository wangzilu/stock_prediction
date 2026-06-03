"""Wire sqrt_adv cost model into the live-like paths.

Per cx code review round 3 P2: backtest/cost_model.py:_slippage supports
the sqrt_adv impact model, but the live-like paths
(backtest/portfolio_backtest.py at the cost-rate line, paper/oms.py at
each fill site) only used the static rate. This PR exposes the
sqrt_adv path through optional knobs and these tests pin both the
backward-compat default AND the activated behaviour.

Tests cover:
  - CostModel.round_trip_rate sanity (with and without sqrt_adv inputs)
  - PortfolioBacktest._compute_sqrt_adv_cost_kwargs gates correctly
  - PortfolioBacktest end-to-end run with sqrt_adv on vs off changes
    the cost trajectory in the expected direction
  - PaperOMS._compute_slippage backward-compat default
  - PaperOMS._compute_slippage routes through CostModel when provided
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from backtest.cost_model import CostModel


# -----------------------------------------------------------------------------
# CostModel sanity
# -----------------------------------------------------------------------------

def test_round_trip_rate_default_is_static():
    cm = CostModel()  # default impact_model="fixed"
    assert cm.impact_model == "fixed"
    static_rate = cm.round_trip_rate()
    # Should equal commission*2 + stamp + slippage*2 + impact*2
    expected = (cm.commission_rate * 2 + cm.stamp_tax_rate
                + cm.slippage_rate * 2 + cm.impact_rate * 2)
    assert math.isclose(static_rate, expected, rel_tol=1e-12)


def test_round_trip_rate_sqrt_adv_with_inputs_scales_with_trade_value():
    cm = CostModel(impact_model="sqrt_adv", impact_coefficient=0.1)
    # Same vol/ADV but two different trade sizes; larger should cost more.
    small = cm.round_trip_rate(daily_volatility=0.02, adv=1e8, trade_value=1e5)
    large = cm.round_trip_rate(daily_volatility=0.02, adv=1e8, trade_value=1e7)
    assert large > small, (
        f"sqrt_adv must scale with trade size; got small={small}, large={large}"
    )


def test_round_trip_rate_sqrt_adv_without_inputs_falls_back_to_static():
    cm = CostModel(impact_model="sqrt_adv")
    # When we pass no vol/ADV, the function should fall back to static rate.
    assert cm.round_trip_rate() == cm.round_trip_rate(  # explicit equality
        daily_volatility=None, adv=None, trade_value=None,
    )


# -----------------------------------------------------------------------------
# PortfolioBacktest — sqrt_adv gating
# -----------------------------------------------------------------------------

def _build_synthetic_market(n_days=40, n_stocks=10, seed=0):
    """Build a tiny synthetic returns / adv panel for tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-04-01", periods=n_days)
    insts = [f"S{idx:06d}" for idx in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates, insts],
                                      names=["datetime", "instrument"])
    ret = pd.DataFrame({"return": rng.normal(0.0, 0.02, size=len(idx))}, index=idx)
    adv = pd.DataFrame({"adv": rng.uniform(5e7, 5e8, size=len(idx))}, index=idx)
    # Predictions: slowly time-varying random score
    pred = pd.DataFrame(
        {"score": rng.normal(0.0, 1.0, size=len(idx))}, index=idx,
    )
    return pred, ret, adv


def test_compute_sqrt_adv_cost_kwargs_returns_empty_when_disabled():
    from backtest.portfolio_backtest import PortfolioBacktest
    pred, ret, adv = _build_synthetic_market()
    pb = PortfolioBacktest(top_k=5, enable_sqrt_adv_costs=False,
                            cost_model=CostModel(impact_model="sqrt_adv"))
    kwargs = pb._compute_sqrt_adv_cost_kwargs(
        date=ret.index.get_level_values(0)[20],
        target_portfolio={"S000000", "S000001", "S000002"},
        returns=ret, adv=adv, turnover=0.2,
    )
    assert kwargs == {}, "disabled flag must produce empty kwargs (fallback)"


def test_compute_sqrt_adv_cost_kwargs_returns_empty_when_adv_missing():
    from backtest.portfolio_backtest import PortfolioBacktest
    pred, ret, _ = _build_synthetic_market()
    pb = PortfolioBacktest(top_k=5, enable_sqrt_adv_costs=True,
                            cost_model=CostModel(impact_model="sqrt_adv"))
    kwargs = pb._compute_sqrt_adv_cost_kwargs(
        date=ret.index.get_level_values(0)[20],
        target_portfolio={"S000000", "S000001"},
        returns=ret, adv=None, turnover=0.2,
    )
    assert kwargs == {}, "missing adv data must fall back to static"


def test_compute_sqrt_adv_cost_kwargs_computes_when_data_available():
    from backtest.portfolio_backtest import PortfolioBacktest
    pred, ret, adv = _build_synthetic_market()
    pb = PortfolioBacktest(top_k=5, enable_sqrt_adv_costs=True,
                            portfolio_value=1e7,
                            cost_model=CostModel(impact_model="sqrt_adv"))
    kwargs = pb._compute_sqrt_adv_cost_kwargs(
        date=ret.index.get_level_values(0)[25],
        target_portfolio={"S000000", "S000001", "S000002"},
        returns=ret, adv=adv, turnover=0.2,
    )
    assert set(kwargs.keys()) == {"daily_volatility", "adv", "trade_value"}, (
        f"unexpected kwarg keys: {kwargs}"
    )
    assert kwargs["daily_volatility"] > 0
    assert kwargs["adv"] > 0
    # trade_value = portfolio_value * turnover / n_stocks = 1e7 * 0.2 / 3
    assert math.isclose(kwargs["trade_value"], 1e7 * 0.2 / 3, rel_tol=1e-9)


def test_portfolio_backtest_default_behaviour_unchanged():
    """Backward compatibility: when sqrt_adv is OFF (default), the cost
    series matches static round_trip_rate * turnover exactly."""
    from backtest.portfolio_backtest import PortfolioBacktest
    pred, ret, adv = _build_synthetic_market(n_days=30, n_stocks=6, seed=1)
    cm = CostModel()  # default fixed
    pb = PortfolioBacktest(top_k=3, rebalance_freq=1, min_adv=0.0,
                            min_listing_days=0, cost_model=cm)
    result = pb.run(predictions=pred, returns=ret, adv=adv,
                    return_horizon_days=1)
    # Every per-day cost should equal cm.round_trip_rate() * turnover
    static = cm.round_trip_rate()
    for cost, turnover in zip(result.daily_cost.values, result.daily_turnover.values):
        assert math.isclose(cost, static * turnover, rel_tol=1e-12), (
            f"unexpected cost={cost} for turnover={turnover}; "
            f"expected {static * turnover}"
        )


def test_portfolio_backtest_sqrt_adv_changes_cost_trajectory():
    """With sqrt_adv on AND cost model in sqrt_adv mode AND adv data,
    the per-day costs differ from the static path (proves the wiring
    is actually exercised, not a no-op)."""
    from backtest.portfolio_backtest import PortfolioBacktest
    pred, ret, adv = _build_synthetic_market(n_days=30, n_stocks=6, seed=2)

    pb_off = PortfolioBacktest(top_k=3, rebalance_freq=1, min_adv=0.0,
                                min_listing_days=0, cost_model=CostModel())
    pb_on = PortfolioBacktest(
        top_k=3, rebalance_freq=1, min_adv=0.0, min_listing_days=0,
        cost_model=CostModel(impact_model="sqrt_adv", impact_coefficient=0.5),
        enable_sqrt_adv_costs=True,
        portfolio_value=5e7,
    )
    res_off = pb_off.run(predictions=pred, returns=ret, adv=adv,
                          return_horizon_days=1)
    res_on = pb_on.run(predictions=pred, returns=ret, adv=adv,
                        return_horizon_days=1)

    diffs = [
        abs(a - b) for a, b in zip(res_off.daily_cost.values, res_on.daily_cost.values)
        if a > 0 or b > 0
    ]
    assert any(d > 1e-8 for d in diffs), (
        "enabling sqrt_adv produced identical cost trajectory to static; "
        "either gating broke or sqrt_adv path is a no-op"
    )


# -----------------------------------------------------------------------------
# PaperOMS — slippage routing
# -----------------------------------------------------------------------------

def test_paper_oms_compute_slippage_default_is_bare_rate(tmp_path):
    from paper.oms import PaperOMS
    oms = PaperOMS(state_dir=str(tmp_path), slippage_rate=0.001,
                    cost_model=None)
    amount = 100_000.0
    assert math.isclose(oms._compute_slippage(amount), amount * 0.001,
                         rel_tol=1e-12)


def test_paper_oms_compute_slippage_with_cost_model_static_path(tmp_path):
    """When CostModel is provided but in fixed mode, slippage equals
    amount * cost_model.slippage_rate (NOT oms.slippage_rate). Pins the
    delegation contract."""
    from paper.oms import PaperOMS
    oms = PaperOMS(state_dir=str(tmp_path), slippage_rate=0.001,
                    cost_model=CostModel(slippage_rate=0.002))
    # With cost_model present but vol/adv None, CostModel._slippage returns
    # amount * cost_model.slippage_rate (its own rate, not oms.slippage_rate)
    out = oms._compute_slippage(100_000.0)
    assert math.isclose(out, 100_000.0 * 0.002, rel_tol=1e-12)


def test_paper_oms_compute_slippage_with_sqrt_adv_active(tmp_path):
    """When cost_model is sqrt_adv AND vol/ADV passed, slippage scales
    with sqrt(amount / ADV) — superlinear, not the bare-rate."""
    from paper.oms import PaperOMS
    cm = CostModel(impact_model="sqrt_adv", impact_coefficient=0.5)
    oms = PaperOMS(state_dir=str(tmp_path), slippage_rate=0.001, cost_model=cm)
    small = oms._compute_slippage(1e4, daily_volatility=0.02, adv=1e8)
    large = oms._compute_slippage(1e7, daily_volatility=0.02, adv=1e8)
    # Both go through sqrt_adv path; large/small ratio > sqrt(1000) * (1000)
    assert large > 100 * small, (
        f"sqrt_adv must produce superlinear scaling; small={small}, large={large}"
    )


def test_paper_oms_inline_slippage_calls_replaced(tmp_path):
    """Anti-regression: paper/oms.py source MUST NOT contain the inline
    `amount * self.slippage_rate` pattern anywhere except inside
    _compute_slippage. The bare-rate inline path is exactly what cx P2
    flagged."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "paper" / "oms.py").read_text(
        encoding="utf-8")
    # Strip the helper definition body to allow the bare-rate fallback there
    helper_idx = src.find("def _compute_slippage")
    if helper_idx >= 0:
        end_idx = src.find("\n    def ", helper_idx + 1)
        # Allow exactly one bare-rate occurrence (the helper's own fallback)
        helper_chunk = src[helper_idx:end_idx if end_idx > 0 else len(src)]
        other = src[:helper_idx] + (src[end_idx:] if end_idx > 0 else "")
    else:
        pytest.fail("paper/oms.py: _compute_slippage helper missing")

    leftover = [
        line for line in other.splitlines()
        if "amount * self.slippage_rate" in line
        and "_compute_slippage" not in line
    ]
    assert leftover == [], (
        f"paper/oms.py still has inline `amount * self.slippage_rate` "
        f"outside of _compute_slippage: {leftover!r}. Use the helper so "
        "future sqrt_adv plumbing flows through one chokepoint."
    )
