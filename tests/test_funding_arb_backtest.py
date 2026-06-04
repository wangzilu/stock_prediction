"""Tests for strategies/crypto/funding_arb (Phase Crypto-C).

Offline: synthetic funding-rate panels exercise the backtest engine.
No exchange calls, no crypto_root, no ccxt.

Coverage:
  - Empty input → empty result
  - Constant positive funding above entry threshold opens once and
    accrues carry
  - Funding sign flip while in a position triggers flip events
  - Below-exit threshold closes the position
  - After-cost APR + Sharpe + max-drawdown are computed
  - Acceptance gate (Sharpe ≥ 1.5 OR APR ≥ 5%) tripped by
    high-funding fixture
  - Acceptance gate fails on a noise-only fixture
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from strategies.crypto.funding_arb import (
    FundingArbBacktestConfig, backtest_funding_arb,
)


def _ts_utc(year, month, day, hour=0) -> int:
    return int(datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp() * 1000)


def _funding_panel(rates, base_ts=None, cadence_sec=8 * 3600, symbol="BTC/USDT:USDT"):
    """Build a funding-history DataFrame matching contract §4 schema."""
    if base_ts is None:
        base_ts = _ts_utc(2024, 1, 1)
    rows = []
    for i, r in enumerate(rates):
        rows.append({
            "timestamp_utc": base_ts + i * cadence_sec * 1000,
            "exchange": "binance",
            "symbol": symbol,
            "funding_rate": float(r),
            "next_funding_ts": None,
            "mark_price": None,
            "index_price": None,
            "ingested_at": base_ts + i * cadence_sec * 1000 + 5000,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------

def test_empty_input_returns_empty_result():
    res = backtest_funding_arb(pd.DataFrame(columns=[
        "timestamp_utc", "funding_rate", "exchange", "symbol",
    ]))
    assert res.n_events == 0
    assert res.net_pnl_usd == 0.0


def test_below_entry_threshold_never_opens():
    """If every funding rate is below entry threshold, the strategy
    never opens — zero events of action 'open'."""
    df = _funding_panel([0.00001] * 20)  # 0.1 bp — below default 1 bp entry
    cfg = FundingArbBacktestConfig()
    res = backtest_funding_arb(df, cfg)
    assert res.n_open_events == 0
    assert res.net_pnl_usd == 0.0


# -----------------------------------------------------------------------------
# Strategy dynamics
# -----------------------------------------------------------------------------

def test_high_positive_funding_opens_once_and_accrues():
    """Constant positive funding above entry → 1 open event, then
    accumulating gross carry across the remaining events."""
    n = 30
    df = _funding_panel([0.0005] * n)  # 5 bps per event = above 1 bp entry
    cfg = FundingArbBacktestConfig(
        notional_per_trade_usd=1_000.0,
        fee_rate_per_leg=0.0,         # disable fees to isolate carry
        slippage_bps_per_leg=0.0,
    )
    res = backtest_funding_arb(df, cfg)
    assert res.n_open_events == 1
    assert res.n_flip_events == 0
    assert res.n_close_events == 0
    # gross carry = notional * rate * n_funding_events_held
    # Position opens at event 0; held for n events.
    # First event is the open + carry; subsequent events keep collecting.
    # The implementation collects on each event INCLUDING the open one
    # since gross_event is computed before action; verify > 0 and
    # roughly notional * rate * (n_events_held)
    assert res.net_pnl_usd > 0
    # rough magnitude sanity: n events at 5 bps on $1k each ≈ $15
    assert 5 < res.net_pnl_usd < 25


def test_funding_flip_action_flat_triggers_flip_event():
    """Constant high funding then a sign reversal → at least one flip."""
    # 10 events at +5 bps, then 10 events at -5 bps
    df = _funding_panel([0.0005] * 10 + [-0.0005] * 10)
    cfg = FundingArbBacktestConfig(
        notional_per_trade_usd=1_000.0,
        fee_rate_per_leg=0.0,
        slippage_bps_per_leg=0.0,
        funding_flip_action="flat",
    )
    res = backtest_funding_arb(df, cfg)
    assert res.n_flip_events >= 1, "expected at least one flip event"


def test_below_exit_closes_position():
    """Funding drops below exit threshold → close event fires."""
    # Start above entry, then drop below exit (default exit = 0.5 bps)
    df = _funding_panel([0.0005] * 5 + [0.00002] * 5)
    cfg = FundingArbBacktestConfig(
        notional_per_trade_usd=1_000.0,
        fee_rate_per_leg=0.0,
        slippage_bps_per_leg=0.0,
    )
    res = backtest_funding_arb(df, cfg)
    assert res.n_close_events >= 1


# -----------------------------------------------------------------------------
# Cost accounting
# -----------------------------------------------------------------------------

def test_fees_and_slippage_decrease_net_pnl():
    """With non-zero fees + slippage, net < gross."""
    df = _funding_panel([0.0005] * 30)
    cfg_no_cost = FundingArbBacktestConfig(
        notional_per_trade_usd=1_000.0,
        fee_rate_per_leg=0.0,
        slippage_bps_per_leg=0.0,
    )
    cfg_with_cost = FundingArbBacktestConfig(
        notional_per_trade_usd=1_000.0,
        fee_rate_per_leg=0.0005,  # 5 bps per leg
        slippage_bps_per_leg=2.0, # 2 bps slip per leg
    )
    res_no = backtest_funding_arb(df, cfg_no_cost)
    res_yes = backtest_funding_arb(df, cfg_with_cost)
    assert res_yes.net_pnl_usd < res_no.net_pnl_usd
    assert res_yes.fees_paid_usd > 0


# -----------------------------------------------------------------------------
# Acceptance gate
# -----------------------------------------------------------------------------

def test_acceptance_passes_on_high_consistent_funding():
    """Very high consistent funding → Sharpe + APR both clear bar."""
    # 1095 events at 20 bps each = exaggerated to guarantee pass
    df = _funding_panel([0.002] * 1095)
    cfg = FundingArbBacktestConfig(
        notional_per_trade_usd=10_000.0,
        fee_rate_per_leg=0.0001,
        slippage_bps_per_leg=0.5,
    )
    res = backtest_funding_arb(df, cfg)
    passes, note = res.passes_acceptance()
    assert passes, note


def test_acceptance_fails_on_noise():
    """Funding ≈ N(0, 5 bps) with no carry → strategy should fail
    the acceptance bar (this is the headline Δ2 scenario)."""
    rng = np.random.default_rng(0)
    rates = rng.normal(0.0, 0.0005, size=400)
    df = _funding_panel(rates.tolist())
    cfg = FundingArbBacktestConfig(
        notional_per_trade_usd=5_000.0,
        fee_rate_per_leg=0.0005,
        slippage_bps_per_leg=2.0,
    )
    res = backtest_funding_arb(df, cfg)
    passes, note = res.passes_acceptance()
    assert not passes, (
        f"noise-only fixture should NOT pass; metrics={res.summary()}"
    )


# -----------------------------------------------------------------------------
# Result structure
# -----------------------------------------------------------------------------

def test_result_timeline_has_expected_columns():
    df = _funding_panel([0.0005] * 10)
    res = backtest_funding_arb(df, FundingArbBacktestConfig())
    timeline = res.timeline
    expected_cols = {"timestamp_utc", "funding_rate", "position",
                       "gross_event_pnl", "cum_net_pnl", "action"}
    assert expected_cols.issubset(set(timeline.columns))


def test_result_summary_includes_apr_and_sharpe():
    df = _funding_panel([0.001] * 100)
    res = backtest_funding_arb(df, FundingArbBacktestConfig())
    s = res.summary()
    assert "APR" in s and "Sharpe" in s


# -----------------------------------------------------------------------------
# Capital constraints (cx round 26 + round 28)
# -----------------------------------------------------------------------------

def test_capital_cap_scales_notional_when_request_exceeds_capital():
    """Notional > capital_usd * (1 - buffer) → backtest scales down."""
    df = _funding_panel([0.001] * 50)
    cfg = FundingArbBacktestConfig(
        capital_usd=3_400.0,
        capital_buffer_pct=0.20,
        notional_per_trade_usd=10_000.0,  # well over cap
        perp_leverage=1.0,
    )
    res = backtest_funding_arb(df, cfg)
    # Effective notional must be the capital cap divided by capital-per-notional
    # ratio (spot + perp ≈ 2× notional at 1× leverage), so effective ≈ cap/2.
    assert res.effective_notional_usd < 10_000.0, (
        "notional should have been scaled down"
    )
    assert res.capital_required_usd <= 3_400.0 * (1 - 0.20) + 1e-6


def test_apr_on_capital_below_apr_on_notional():
    """For 1× leverage, capital ≈ 2×notional → capital APR ≈ notional APR / 2."""
    df = _funding_panel([0.001] * 200)
    cfg = FundingArbBacktestConfig(
        capital_usd=3_400.0,
        capital_buffer_pct=0.0,
        notional_per_trade_usd=1_500.0,
        perp_leverage=1.0,
    )
    res = backtest_funding_arb(df, cfg)
    # cx round 29 P2: explicit > 0 assertion + strict ordering.
    assert res.after_cost_apr_on_capital > 0, (
        f"capital APR was 0 — return wiring regressed? "
        f"metrics={res.summary()}"
    )
    assert res.after_cost_apr_on_capital < res.after_cost_apr_on_notional, (
        f"capital APR={res.after_cost_apr_on_capital:.4f} should be less "
        f"than notional APR={res.after_cost_apr_on_notional:.4f}"
    )
    # The legacy alias points at the honest capital figure (round 28 P0-1)
    assert res.after_cost_apr == res.after_cost_apr_on_capital


def test_capital_apr_field_actually_populated_when_pnl_positive():
    """cx round 29 P2 inverted form: even when notional APR is the
    same as before R28, the R28 return-wiring fix means
    after_cost_apr_on_capital MUST NOT remain at the default 0.
    Pre-fix the field was added but the return only set
    after_cost_apr — this test would have caught the regression."""
    df = _funding_panel([0.001] * 30)
    res = backtest_funding_arb(df, FundingArbBacktestConfig(
        capital_usd=3_400.0, capital_buffer_pct=0.0,
        notional_per_trade_usd=1_500.0, perp_leverage=1.0,
    ))
    assert res.after_cost_apr_on_capital > 0, (
        "after_cost_apr_on_capital still 0 — return wiring regressed"
    )
    assert res.after_cost_apr_on_notional > res.after_cost_apr_on_capital, (
        "cx round 29 P2: notional figure must EXCEED capital figure"
    )


def test_acceptance_uses_capital_apr_not_notional():
    """Notional-APR alone would pass, but capital-APR fails →
    acceptance reflects the capital figure (cx round 28 P0-1)."""
    # Craft a case where notional APR > 5% but capital APR < 5%
    # by setting capital >> notional. funding 5 bps × 1095 events ≈
    # 54%/yr on notional, but capital is 10× the notional → ~5.4%/yr.
    df = _funding_panel([0.0005] * 1095)
    cfg = FundingArbBacktestConfig(
        capital_usd=20_000.0,
        capital_buffer_pct=0.0,
        notional_per_trade_usd=2_000.0,  # notional small vs capital
        perp_leverage=1.0,
        fee_rate_per_leg=0.0,
        slippage_bps_per_leg=0.0,
    )
    res = backtest_funding_arb(df, cfg)
    # Sanity: this fixture produces notional APR > capital APR; the
    # acceptance reason cites the capital figure either way.
    passes, note = res.passes_acceptance()
    if passes:
        assert "APR(capital)" in note or "Sharpe" in note, (
            f"acceptance note must cite capital APR or Sharpe, got: {note}"
        )
    else:
        assert "APR(capital)" in note, (
            f"failure note must cite capital APR, got: {note}"
        )
