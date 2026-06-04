"""Funding-rate carry strategy (Phase Crypto-C).

Captures the perpetual-futures funding spread by holding offsetting
positions across two legs:

  - 2026-06-04 cx round 26 P1-1 fix — direction was documented backwards.
    When funding_rate > 0, perp longs PAY funding; the carry profit
    comes from being SHORT the perp + LONG the spot (which neutralises
    price exposure). When funding_rate < 0, do the reverse:
        funding > 0:  short perp + long spot   (RECEIVE funding)
        funding < 0:  long perp + short spot   (RECEIVE funding)
    ``position_sign`` in the loop below tracks the perp leg's direction;
    the carry math at line 191 is correct (we receive funding when
    position_sign matches sign(funding_rate)).

Net P&L per funding event = `position_size_usd * funding_rate * sign`.
Net of fees: subtract per-leg trading fees on rebalance + slippage
(currently a fixed bps; sqrt_adv on OI to be added when we have
liquid OI snapshots — see follow-up below).

Operates on **closed-book backtest** only: it reads parquet funding
history and simulates the strategy day by day. Live execution would
go through the (future) Phase D paper daemon.

Per `plans/crypto-dev-phases.md` Δ2 (cx 6/3 freshness sweep): after-
cost reality on majors is ~3-12% APR on Hyperliquid, not the
gross 8-10% APR / 92%-positive number the 5/30 baseline assumed
([MDPI 14:2:346](https://www.mdpi.com/2227-7390/14/2/346) shows
only 40% of top opportunities net positive). The backtest's
acceptance gate is therefore:

  after-cost Sharpe ≥ 1.5  OR
  after-cost APR ≥ 5% on majors
  OR explicit fail verdict with evidence (Δ2 stance).

No live trade execution. No API key. No leverage > 1×.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Funding events on majors are ~3/day (8h cadence) on most venues.
FUNDING_EVENTS_PER_YEAR_DEFAULT = 3 * 365


@dataclass
class FundingArbBacktestConfig:
    """All knobs for the backtest. Defaults match the conservative
    starting envelope per crypto-data-contract.md §11."""

    # ---- Position sizing & capital ----
    # 2026-06-04 cx round 26 P0-1 + P0-2: a funding-arb trade needs
    # capital for BOTH legs (spot cash + perp margin) plus a buffer
    # for funding events. Reporting APR against ``notional`` alone
    # over-stated returns by ~2× because real capital_required is
    # spot_notional + perp_margin + buffer. Now the config models both.
    capital_usd: float = 3_400.0
    """User's total liquid USD-equivalent. Default 1.7 ETH @ $2,000 =
    $3,400 — matches the actual paper-only constraint and is the
    correct denominator for ``apr_on_capital``."""

    spot_cash_usd: float | None = None
    """Cash actually parked in the spot leg. None → derive from
    notional + min(spot_share, 0.6) of capital_usd."""

    perp_margin_usd: float | None = None
    """Initial margin posted on the perp leg. None → derive as
    notional / perp_leverage."""

    perp_leverage: float = 1.0
    """Leverage applied to the perp leg. The paper-only constraint
    says no leverage > 1× so margin = notional in the default path."""

    capital_buffer_pct: float = 0.20
    """Fraction of capital reserved for funding flushes / venue
    haircuts (default 20%)."""

    notional_per_trade_usd: float = 1_500.0
    """USD-equivalent leg size. Pre-fix default was $5,000 which is
    unrealistic for a 1.7 ETH ($3,400) account. New default $1,500
    fits the user's ``capital_usd`` default with 20% buffer + 1×
    leverage. The backtest also enforces a HARD cap of
    ``capital_usd * (1 - capital_buffer_pct)`` so any caller that
    bumps this above their capital is auto-clipped (see
    ``effective_notional`` in backtest)."""

    # ---- Trading-cost model ----
    # Maker / taker per-leg, applied at every flip event.
    fee_rate_per_leg: float = 0.00045
    """Per-leg fee as a fraction. 0.00045 == 4.5 bps == 0.045% (cx
    round 26 P1-2 unit fix; previous comment said "0.045 bps" which
    is 100× off). Hyperliquid taker is ~4.5 bps; Binance taker is
    ~5 bps. Both fit this default. Per-leg, two legs at open + two
    legs at close → 4 × fee_rate per round trip."""

    slippage_bps_per_leg: float = 1.0
    """Conservative bid-ask spread cost on entry/exit. 1 bp is tight
    for majors; bump higher for long-tail."""

    # ---- Entry / exit policy ----
    abs_funding_entry_bps: float = 1.0
    """Only enter when |funding_rate| in bps exceeds this threshold."""

    abs_funding_exit_bps: float = 0.5
    """Exit when funding magnitude drops below this (mean reversion of
    the spread eats the carry edge)."""

    funding_flip_action: str = "flat"
    """When funding sign flips while in a position: 'flat' (close +
    reopen on the new sign) or 'hold' (let it run). Conservative
    default is 'flat'."""

    # ---- Backtest window ----
    start_date_utc: Optional[str] = None
    end_date_utc: Optional[str] = None

    # ---- Reporting ----
    funding_events_per_year: int = FUNDING_EVENTS_PER_YEAR_DEFAULT


@dataclass
class FundingArbResult:
    """Backtest output. All metrics computed net of fees + slippage."""
    n_events: int = 0
    n_open_events: int = 0
    n_close_events: int = 0
    n_flip_events: int = 0
    gross_pnl_usd: float = 0.0
    net_pnl_usd: float = 0.0
    fees_paid_usd: float = 0.0
    slippage_paid_usd: float = 0.0
    # 2026-06-04 cx round 28 P0-1 + P1-2: expose BOTH denominators so
    # consumers can't accidentally read the rosier notional figure.
    # ``after_cost_apr`` is an ALIAS of ``after_cost_apr_on_capital``
    # (the honest one); ``after_cost_apr_on_notional`` is kept for
    # comparing against the legacy reporting. ``passes_acceptance``
    # uses the capital figure.
    after_cost_apr_on_notional: float = 0.0
    after_cost_apr_on_capital: float = 0.0
    after_cost_apr: float = 0.0  # = after_cost_apr_on_capital
    after_cost_sharpe: float = 0.0
    max_drawdown: float = 0.0  # legacy alias
    max_drawdown_on_notional: float = 0.0
    max_drawdown_on_capital: float = 0.0
    capital_required_usd: float = 0.0
    effective_notional_usd: float = 0.0
    timeline: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Per-event timeline: timestamp, funding_rate, position, gross_pnl,
    net_pnl, cumulative_net_pnl, action."""

    def summary(self) -> str:
        return (
            f"FundingArbResult({self.n_events} events, "
            f"net=${self.net_pnl_usd:,.2f}, "
            f"APR(cap)={self.after_cost_apr_on_capital*100:.2f}% / "
            f"APR(not)={self.after_cost_apr_on_notional*100:.2f}%, "
            f"Sharpe={self.after_cost_sharpe:.2f}, "
            f"max_dd(cap)={self.max_drawdown_on_capital*100:.1f}%)"
        )

    def passes_acceptance(self) -> tuple[bool, str]:
        """Apply the Phase Crypto-C acceptance gate.

        cx round 28 P0-1: gate uses the CAPITAL-denominated APR, not
        the rosier notional-denominated figure. A funding arb that
        looks like ~10% APR on notional collapses to ~5% APR on
        capital when capital_required ≈ 2× notional under 1× leverage.
        """
        if self.after_cost_sharpe >= 1.5:
            return True, f"Sharpe {self.after_cost_sharpe:.2f} ≥ 1.5"
        if self.after_cost_apr_on_capital >= 0.05:
            return True, f"APR(capital) {self.after_cost_apr_on_capital*100:.1f}% ≥ 5%"
        return False, (
            f"Sharpe {self.after_cost_sharpe:.2f} < 1.5 AND "
            f"APR(capital) {self.after_cost_apr_on_capital*100:.2f}% < 5% — "
            "strategy doesn't clear the acceptance bar at this size. "
            "(APR on notional would have been "
            f"{self.after_cost_apr_on_notional*100:.2f}%, kept for "
            "comparison; do not rely on the rosier number.) "
            "See Δ2 fail verdict path."
        )


def backtest_funding_arb(
    funding_df: pd.DataFrame,
    config: Optional[FundingArbBacktestConfig] = None,
) -> FundingArbResult:
    """Run the backtest against a funding-history frame.

    Args:
        funding_df: parquet rows in the §4 schema with columns
            (timestamp_utc, exchange, symbol, funding_rate, ...).
            Must be sorted ascending by timestamp_utc (we sort
            defensively).
        config: knob bundle. Default to FundingArbBacktestConfig().

    Returns:
        FundingArbResult with summary metrics + per-event timeline.
    """
    if config is None:
        config = FundingArbBacktestConfig()
    if funding_df.empty:
        return FundingArbResult()

    df = funding_df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    if config.start_date_utc:
        start_ms = int(pd.Timestamp(config.start_date_utc).timestamp() * 1000)
        df = df[df["timestamp_utc"] >= start_ms]
    if config.end_date_utc:
        end_ms = int(pd.Timestamp(config.end_date_utc).timestamp() * 1000)
        df = df[df["timestamp_utc"] < end_ms]

    if df.empty:
        return FundingArbResult()

    entry_bps = config.abs_funding_entry_bps / 1e4  # to fraction
    exit_bps = config.abs_funding_exit_bps / 1e4
    # cx round 26 P0-1: cap notional by available capital. Pre-fix the
    # backtest would happily simulate $5,000-per-trade against a
    # $3,400 account, producing fictitious returns.
    requested_notional = float(config.notional_per_trade_usd)
    spot_cash = (
        float(config.spot_cash_usd)
        if config.spot_cash_usd is not None
        else requested_notional
    )
    perp_margin = (
        float(config.perp_margin_usd)
        if config.perp_margin_usd is not None
        else (requested_notional / max(float(config.perp_leverage), 1e-9))
    )
    capital_required = spot_cash + perp_margin
    available_capital = float(config.capital_usd) * (1.0 - float(config.capital_buffer_pct))
    if capital_required > available_capital and available_capital > 0:
        # Scale notional down proportionally so the implied capital
        # requirement matches available_capital. Log loud — a silent
        # downsize would understate why APR drops vs the request.
        scale = available_capital / capital_required
        effective_notional = requested_notional * scale
        capital_required = available_capital
        spot_cash *= scale
        perp_margin *= scale
        logger.warning(
            "funding_arb: requested notional $%.0f exceeds available capital "
            "$%.0f (cap_usd=%.0f × (1-buf=%.2f)); scaled to $%.0f.",
            requested_notional, available_capital,
            config.capital_usd, config.capital_buffer_pct,
            effective_notional,
        )
        notional = effective_notional
    else:
        notional = requested_notional
    fee_per_leg = config.fee_rate_per_leg
    slip_per_leg = config.slippage_bps_per_leg / 1e4
    # cx round 28 P2-4: removed the unused ``cost_per_flip`` local
    # (computed but never referenced). Costs are calculated inline at
    # each open/close/flip site below.

    # State
    # cx round 28 P1-3: comment direction fix (was reversed). +1 means
    # the SHORT-perp leg (we collect funding when funding_rate > 0).
    position_sign = 0   # +1 short perp + long spot, -1 long perp + short spot
    n_open = 0
    n_close = 0
    n_flip = 0
    fees_total = 0.0
    slip_total = 0.0
    timeline_rows: list[dict] = []
    cum_net = 0.0

    for _, row in df.iterrows():
        ts = int(row["timestamp_utc"])
        funding_rate = float(row["funding_rate"])
        abs_fr = abs(funding_rate)

        # Carry P&L of CURRENT position over this funding event.
        # Long perp earns -funding when funding > 0; the offsetting
        # short-perp position EARNS +funding when funding > 0. So if
        # position_sign = +1 (we short the perp), we collect
        # +funding_rate * notional per event.
        gross_event = position_sign * funding_rate * notional

        # Decide next action
        action = "hold"
        if position_sign == 0:
            # Out of market — enter on threshold
            if abs_fr >= entry_bps:
                position_sign = 1 if funding_rate > 0 else -1
                n_open += 1
                fees_event = notional * 2 * fee_per_leg
                slip_event = notional * 2 * slip_per_leg
                fees_total += fees_event
                slip_total += slip_event
                cum_net -= (fees_event + slip_event)
                action = "open"
        else:
            same_sign = (position_sign > 0 and funding_rate > 0) or \
                        (position_sign < 0 and funding_rate < 0)
            if abs_fr < exit_bps:
                # Mean reversion — close
                n_close += 1
                fees_event = notional * 2 * fee_per_leg
                slip_event = notional * 2 * slip_per_leg
                fees_total += fees_event
                slip_total += slip_event
                cum_net -= (fees_event + slip_event)
                position_sign = 0
                action = "close"
            elif not same_sign and config.funding_flip_action == "flat":
                # Funding flipped sign — close + reopen on the new side
                n_close += 1
                n_open += 1
                n_flip += 1
                fees_event = notional * 4 * fee_per_leg  # close 2 + open 2
                slip_event = notional * 4 * slip_per_leg
                fees_total += fees_event
                slip_total += slip_event
                cum_net -= (fees_event + slip_event)
                position_sign = 1 if funding_rate > 0 else -1
                action = "flip"

        cum_net += gross_event
        timeline_rows.append({
            "timestamp_utc": ts,
            "funding_rate": funding_rate,
            "position": position_sign,
            "gross_event_pnl": gross_event,
            "cum_net_pnl": cum_net,
            "action": action,
        })

    timeline = pd.DataFrame(timeline_rows)
    n_events = len(timeline)

    # Returns per event (PnL / notional) for Sharpe / APR
    per_event_net = timeline["cum_net_pnl"].diff().fillna(timeline["cum_net_pnl"].iloc[0]) if n_events else pd.Series(dtype=float)
    per_event_return = per_event_net / notional if notional > 0 else per_event_net * 0

    mean_ret = float(per_event_return.mean()) if n_events else 0.0
    std_ret = float(per_event_return.std()) if n_events > 1 else 0.0
    events_per_year = config.funding_events_per_year
    apr = mean_ret * events_per_year
    sharpe = mean_ret / std_ret * np.sqrt(events_per_year) if std_ret > 0 else 0.0

    # cx round 26 P0-2: report APR on REAL CAPITAL EMPLOYED, not on
    # notional. capital_required = spot_cash + perp_margin (already
    # capped above). For a 1× leverage funding-arb trade,
    # capital_required ≈ 2 × notional → apr_on_capital ≈ apr_on_notional / 2.
    per_event_return_on_capital = (
        per_event_net / capital_required if capital_required > 0
        else per_event_return * 0.0
    )
    apr_on_capital = (
        float(per_event_return_on_capital.mean()) * events_per_year
        if n_events else 0.0
    )

    # Drawdown — cx round 28 P1-2: report on capital AND notional so
    # downstream can pick the right denominator. capital is the
    # honest one for paper-account viability.
    if n_events:
        eq = timeline["cum_net_pnl"].values
        peak = np.maximum.accumulate(eq)
        dd_not = (eq - peak) / max(notional, 1.0)
        dd_cap = (eq - peak) / max(capital_required, 1.0)
        max_dd_notional = float(abs(dd_not.min()))
        max_dd_capital = float(abs(dd_cap.min()))
    else:
        max_dd_notional = 0.0
        max_dd_capital = 0.0

    return FundingArbResult(
        n_events=n_events,
        n_open_events=n_open,
        n_close_events=n_close,
        n_flip_events=n_flip,
        gross_pnl_usd=float(timeline["gross_event_pnl"].sum()) if n_events else 0.0,
        net_pnl_usd=float(cum_net),
        fees_paid_usd=float(fees_total),
        slippage_paid_usd=float(slip_total),
        after_cost_apr_on_notional=float(apr),
        after_cost_apr_on_capital=float(apr_on_capital),
        after_cost_apr=float(apr_on_capital),  # alias — see dataclass note
        after_cost_sharpe=float(sharpe),
        max_drawdown_on_notional=max_dd_notional,
        max_drawdown_on_capital=max_dd_capital,
        max_drawdown=max_dd_capital,  # alias
        capital_required_usd=float(capital_required),
        effective_notional_usd=float(notional),
        timeline=timeline,
    )
