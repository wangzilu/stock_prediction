# fix/sqrt-adv-to-backtest-oms — PR Self-Review

Date: 2026-06-03
Branch: `fix/sqrt-adv-to-backtest-oms` (1 commit ahead of `master`)
Commit: `bb1c07f feat: wire sqrt_adv cost model into portfolio backtest + paper OMS`
Tests: 12 new + 68 adjacent passing
Merge gate: **safe to merge** — default OFF, opt-in

## What this PR does

Exposes the `sqrt_adv` impact model that already lived in
`backtest/cost_model.py` so it can actually be used by the two
live-like paths cx flagged:

| Path | Before | After |
|---|---|---|
| `backtest/portfolio_backtest.py:481` | `cost = round_trip_rate() * turnover` (static) | When `enable_sqrt_adv_costs=True` AND ADV passed AND vol computable, calls `round_trip_rate(vol, adv, trade_value)` |
| `paper/oms.py` 4 fill sites | `slippage = amount * slippage_rate` (inline) | `slippage = self._compute_slippage(amount, vol, adv)` — delegates to `CostModel._slippage` when `cost_model=` injected |

Default behaviour: unchanged. Both new paths are opt-in.

## Why default OFF

Activating sqrt_adv changes per-day cost rates in a backtest. Cost
trajectories shift, which propagates to Sharpe / drawdown / cost
attribution. cx's earlier guidance applies: any change that affects
production cost numbers needs an explicit promotion via paired
backtest, not silent inclusion. Default OFF keeps existing crons and
notebooks identical until a deliberate opt-in.

The plumbing is the point of this PR — once it's in place, activating
it for crypto-B or A-share production is one parameter flip.

## Files changed

```
backtest/portfolio_backtest.py  | 110 ++ helper + cost-line wire
paper/oms.py                    |  35 ++ helper + 4 site replacements
tests/test_sqrt_adv_wiring.py   | 251 ++ 12 tests, new file
```

## Test coverage (12 cases)

| Group | Test | Pins |
|---|---|---|
| CostModel sanity | `..._default_is_static` | round_trip_rate baseline |
| CostModel sanity | `..._sqrt_adv_with_inputs_scales_with_trade_value` | sqrt_adv math |
| CostModel sanity | `..._sqrt_adv_without_inputs_falls_back_to_static` | graceful fallback |
| PB gating | `..._returns_empty_when_disabled` | flag respected |
| PB gating | `..._returns_empty_when_adv_missing` | data gate |
| PB gating | `..._computes_when_data_available` | helper happy path |
| PB end-to-end | `..._default_behaviour_unchanged` | backward compat |
| PB end-to-end | `..._sqrt_adv_changes_cost_trajectory` | wiring exercised |
| OMS routing | `..._default_is_bare_rate` | backward compat |
| OMS routing | `..._with_cost_model_static_path` | delegation contract |
| OMS routing | `..._with_sqrt_adv_active` | superlinear scaling |
| OMS routing | `..._inline_slippage_calls_replaced` | anti-regression |

## Suggested merge call

**Safe to merge now.**

- Default OFF means no production behaviour change on merge.
- The 12 tests pin both backward-compat and activated behaviour.
- 68/68 adjacent suites pass (capital flow target-date, backtest
  compile, crypto quarantine, scheduler, macro PIT drop).
- Crypto-B has a hard dependency on this plumbing existing — see
  `plans/crypto-dev-phases.md` Δ6 action.

## Follow-ups (NOT in this PR)

1. **Wire per-stock vol/ADV into paper OMS fill loops**: the helper
   accepts `daily_volatility=None, adv=None` today. A daily snapshot
   feed would let sqrt_adv activate at fill time. Separate PR.
2. **Promote `enable_sqrt_adv_costs=True` as PortfolioBacktest
   default** after a paired backtest confirms cost rates are sensible
   at our sizing. Separate PR.
3. **Crypto cost model uses this**: `models/crypto_feature_pipeline.py`
   (Phase Crypto-B) will instantiate `CostModel(impact_model="sqrt_adv")`
   and pass it to whichever execution layer crypto uses.

## Honest scope clarification — cx review round 3 (2026-06-03)

cx independently reviewed this PR after merge and called out that the
original commit title "wire sqrt_adv cost model into portfolio
backtest + paper OMS" overstated closure. The honest scope is:

**Backtest side**: a portfolio-MEAN approximation of vol + ADV + a
uniform per-stock trade value. Not a per-stock cost attribution.
A mixed large+small-cap portfolio will dilute small-cap impact
through the portfolio-average ADV, so paper Sharpe may still be
optimistic on small-cap-heavy strategies. This PR proves the wiring
is not dead code; it does NOT promise faithful per-trade cost.

**Paper OMS side**: `_compute_slippage(amount)` is now the chokepoint,
but the four real fill sites all pass `vol=None, adv=None`. With
the current call sites, even `cost_model=CostModel(impact_model=
"sqrt_adv")` falls back to bare slippage_rate. Production paper is
still effectively fixed 0.1% — by design, because we don't have
per-fill vol/ADV plumbed in yet. The chokepoint is ready for that
plumbing to land in a follow-up.

cx round-3 findings tracked separately:
  - P2: Paper OMS — pipe vol/ADV into the 4 fill sites + into the
    paper_trading runner / shadow runner (Task #84).
  - P2: Backtest — per-trade attribution by (buy, sell, weight delta,
    per-stock ADV) instead of portfolio-mean (Task #85).
  - P3: portfolio_value docstring — call out RMB-vs-USD-vs-USDT
    semantics + that research scripts MUST pass explicitly (FIXED
    in commit on `fix/sqrt-adv-cx-round3`).

This honest framing is what should land in any future "sqrt_adv
fully closed" PR title.
