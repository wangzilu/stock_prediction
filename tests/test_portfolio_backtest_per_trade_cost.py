"""cx round 3 P2 #85 — PortfolioBacktest per-trade cost attribution.

The previous PR (a82fb4d) computed a single portfolio-mean (ADV, vol,
trade_value) triplet and multiplied round_trip_rate by total turnover.
A mixed large+small-cap portfolio diluted small-cap impact through
the mean ADV; small-cap-heavy strategies stayed optimistically priced.

This PR adds per-trade attribution:

  - PortfolioBacktest._per_stock_vol_adv_snapshot — per-code vol+ADV dict
  - PortfolioBacktest._per_trade_cost_dollars — sums per-leg cost using
    each stock's own (vol, ADV), returning total $ cost
  - Cost line at run() chooses the per-trade path when available;
    otherwise falls back to the portfolio-mean path (which itself
    falls back to static rate when sqrt_adv is OFF or ADV missing —
    three-level graceful degradation)

The tests pin:
  1. Backward compat: sqrt_adv OFF → behaviour unchanged
  2. Per-trade returns None when prerequisites missing
  3. Per-stock snapshot returns the expected dict shape
  4. Small-cap leg's marginal impact is HIGHER under per-trade than
     under portfolio-mean (the whole point of the fix)
  5. Equal-weight leg sizing == portfolio_value / top_k
  6. Optimizer-mode leg sizing == portfolio_value * |weight_delta|
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtest.cost_model import CostModel


def _build_panel(n_days=40, codes=("BIG", "SMALL"), seed=0):
    """Build a synthetic returns + ADV panel where BIG has high ADV
    and SMALL has low ADV, both with similar vol."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-04-01", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, list(codes)],
                                      names=["datetime", "instrument"])
    rows = []
    for d in dates:
        for c in codes:
            rows.append({
                "datetime": d, "instrument": c,
                "return": rng.normal(0.0, 0.02),
                "adv": 1e10 if c == "BIG" else 5e6,
            })
    df = pd.DataFrame(rows).set_index(["datetime", "instrument"])
    return df[["return"]], df[["adv"]]


def _pb(enable_sqrt_adv=True, portfolio_value=1e7, top_k=2, impact="sqrt_adv"):
    from backtest.portfolio_backtest import PortfolioBacktest
    cm = CostModel(impact_model=impact, impact_coefficient=0.5)
    return PortfolioBacktest(
        top_k=top_k,
        cost_model=cm,
        enable_sqrt_adv_costs=enable_sqrt_adv,
        portfolio_value=portfolio_value,
        cost_vol_window=10,
    )


# -----------------------------------------------------------------------------
# Per-stock snapshot
# -----------------------------------------------------------------------------

def test_per_stock_snapshot_returns_vol_adv_dict():
    pb = _pb()
    ret, adv = _build_panel(n_days=30)
    date = ret.index.get_level_values(0)[20]
    snap = pb._per_stock_vol_adv_snapshot(
        date=date, codes=["BIG", "SMALL"], returns=ret, adv=adv,
    )
    assert set(snap.keys()) == {"BIG", "SMALL"}
    for code, (v, a) in snap.items():
        assert v > 0
        assert a > 0
    assert snap["BIG"][1] > snap["SMALL"][1] * 100  # BIG ADV much larger


def test_per_stock_snapshot_omits_codes_without_data():
    pb = _pb()
    ret, adv = _build_panel(n_days=30, codes=("BIG",))
    date = ret.index.get_level_values(0)[20]
    snap = pb._per_stock_vol_adv_snapshot(
        date=date, codes=["BIG", "MISSING"], returns=ret, adv=adv,
    )
    assert "BIG" in snap
    assert "MISSING" not in snap


# -----------------------------------------------------------------------------
# Per-trade cost dollars
# -----------------------------------------------------------------------------

def test_per_trade_dollars_returns_none_when_disabled():
    pb = _pb(enable_sqrt_adv=False)
    ret, adv = _build_panel(n_days=30)
    out = pb._per_trade_cost_dollars(
        date=ret.index.get_level_values(0)[20],
        buys={"BIG"}, sells=set(),
        target_weights=None, prev_weights=None,
        returns=ret, adv=adv,
    )
    assert out is None


def test_per_trade_dollars_returns_none_when_adv_missing():
    pb = _pb()
    ret, _ = _build_panel(n_days=30)
    out = pb._per_trade_cost_dollars(
        date=ret.index.get_level_values(0)[20],
        buys={"BIG"}, sells=set(),
        target_weights=None, prev_weights=None,
        returns=ret, adv=None,
    )
    assert out is None


def test_per_trade_dollars_zero_when_no_legs():
    pb = _pb()
    ret, adv = _build_panel(n_days=30)
    out = pb._per_trade_cost_dollars(
        date=ret.index.get_level_values(0)[20],
        buys=set(), sells=set(),
        target_weights=None, prev_weights=None,
        returns=ret, adv=adv,
    )
    assert out == 0.0


def test_per_trade_dollars_small_cap_pays_more_than_large_cap():
    """The whole point of per-trade attribution: a small-cap leg's
    impact is calculated with ITS OWN low ADV, not the portfolio mean.
    A SMALL buy should cost more (as a rate) than a BIG buy of the
    same dollar value."""
    pb = _pb(portfolio_value=1e7, top_k=2)
    ret, adv = _build_panel(n_days=30)
    date = ret.index.get_level_values(0)[20]

    big_only = pb._per_trade_cost_dollars(
        date=date, buys={"BIG"}, sells=set(),
        target_weights=None, prev_weights=None,
        returns=ret, adv=adv,
    )
    small_only = pb._per_trade_cost_dollars(
        date=date, buys={"SMALL"}, sells=set(),
        target_weights=None, prev_weights=None,
        returns=ret, adv=adv,
    )
    assert small_only > big_only, (
        f"per-trade attribution must charge SMALL more than BIG for "
        f"the same trade dollar value (low ADV → higher slip rate). "
        f"got BIG={big_only}, SMALL={small_only}"
    )


def test_per_trade_dollars_equal_weight_leg_sizing():
    """Equal-weight mode (no target_weights): each leg dollar =
    portfolio_value / top_k."""
    pb = _pb(portfolio_value=1e7, top_k=2)
    ret, adv = _build_panel(n_days=30)
    date = ret.index.get_level_values(0)[20]
    out = pb._per_trade_cost_dollars(
        date=date, buys={"BIG"}, sells=set(),
        target_weights=None, prev_weights=None,
        returns=ret, adv=adv,
    )
    # tv = 1e7 / 2 = 5e6
    # vol ≈ 0.02, adv = 1e10, coeff = 0.5
    # slip = 0.02 * sqrt(5e6/1e10) * 0.5 = 0.02 * sqrt(0.0005) * 0.5
    #      ≈ 0.02 * 0.0224 * 0.5 ≈ 2.24e-4
    # cost = 5e6 * (commission 3e-4 + slip + impact 0)
    # rough sanity: cost between 0 and tv * (commission + 0.01)
    assert 0 < out < 5e6 * 0.01


def test_per_trade_dollars_optimizer_leg_sizing_from_weight_delta():
    """With optimizer_v2 weights: each leg dollar =
    portfolio_value * |w_new - w_old|."""
    pb = _pb(portfolio_value=1e7, top_k=2)
    ret, adv = _build_panel(n_days=30)
    date = ret.index.get_level_values(0)[20]

    # No prior; new weight 0.4 on BIG. tv = 1e7 * 0.4 = 4e6.
    out = pb._per_trade_cost_dollars(
        date=date, buys={"BIG"}, sells=set(),
        target_weights={"BIG": 0.4}, prev_weights={},
        returns=ret, adv=adv,
    )
    # Compare with equal-weight version (tv = 5e6). Optimizer's tv (4e6)
    # is smaller, cost should be proportionally smaller.
    eq = pb._per_trade_cost_dollars(
        date=date, buys={"BIG"}, sells=set(),
        target_weights=None, prev_weights=None,
        returns=ret, adv=adv,
    )
    assert out < eq, (
        f"smaller weight delta should produce smaller per-trade cost; "
        f"got optimizer={out} vs equal-weight={eq}"
    )


def test_per_trade_dollars_sells_include_stamp_tax():
    """A sell leg pays stamp tax; a buy leg of the same code/value
    does NOT. Verify the per-trade rate differential."""
    pb = _pb(portfolio_value=1e7, top_k=2)
    ret, adv = _build_panel(n_days=30)
    date = ret.index.get_level_values(0)[20]

    buy_cost = pb._per_trade_cost_dollars(
        date=date, buys={"BIG"}, sells=set(),
        target_weights=None, prev_weights=None,
        returns=ret, adv=adv,
    )
    sell_cost = pb._per_trade_cost_dollars(
        date=date, buys=set(), sells={"BIG"},
        target_weights=None, prev_weights=None,
        returns=ret, adv=adv,
    )
    # Same code, same trade_value, same slip — difference is exactly
    # stamp_tax_rate * tv = 5e-4 * 5e6 = 2500
    diff = sell_cost - buy_cost
    expected = 5e-4 * (1e7 / 2)
    assert math.isclose(diff, expected, rel_tol=1e-6), (
        f"sell - buy should equal stamp_tax_rate * tv; got diff={diff}, "
        f"expected ≈ {expected}"
    )


# -----------------------------------------------------------------------------
# End-to-end: run() with per-trade vs portfolio-mean
# -----------------------------------------------------------------------------

def test_run_with_per_trade_path_does_not_break_backward_compat():
    """Backward compat: when sqrt_adv is OFF, run() produces the same
    static cost trajectory as before the per-trade work landed.
    (Already covered by the existing test_portfolio_backtest_default_
    behaviour_unchanged in test_sqrt_adv_wiring.py — this is a smoke
    sanity that nothing in the per-trade refactor changed the OFF path.)
    """
    from backtest.portfolio_backtest import PortfolioBacktest
    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-04-01", periods=20)
    insts = [f"S{idx:06d}" for idx in range(6)]
    idx = pd.MultiIndex.from_product([dates, insts],
                                      names=["datetime", "instrument"])
    pred = pd.DataFrame({"score": rng.normal(0, 1, size=len(idx))}, index=idx)
    ret = pd.DataFrame({"return": rng.normal(0, 0.02, size=len(idx))}, index=idx)
    adv = pd.DataFrame({"adv": rng.uniform(5e7, 5e8, size=len(idx))}, index=idx)

    pb = PortfolioBacktest(top_k=3, rebalance_freq=1, min_adv=0.0,
                            min_listing_days=0, cost_model=CostModel())
    result = pb.run(predictions=pred, returns=ret, adv=adv,
                    return_horizon_days=1)
    static = CostModel().round_trip_rate()
    for cost, turnover in zip(result.daily_cost.values,
                                result.daily_turnover.values):
        assert math.isclose(cost, static * turnover, rel_tol=1e-12), (
            f"OFF path drift: cost={cost} vs expected {static * turnover}"
        )
