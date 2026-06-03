# CX Crypto System Design Review 2026-05-30

## Bottom Line

Current design score: **8.0 / 10**.

The design is good enough to start **only after legacy crypto quarantine**.
It is not yet good enough to be treated as a trusted trading system.

The strongest parts are:

- Paper-only and no-leverage constraint is now the highest gate.
- A-share isolation is explicitly designed.
- Funding/OI is correctly moved into Phase Crypto-A data foundation.
- Phase C model is shadow-only while the universe is small.
- Nautilus, RL, event/on-chain, and frontier models are deferred.

The remaining gap is not "more alpha". The remaining gap is trust:

```text
Can the crypto system collect data, generate paper signals, and report results
without touching or degrading the existing A-share production pipeline?
```

Until this is proven, no model work should be considered meaningful.

## Why Not 9/10 Yet

### 1. Isolation Is Designed But Not Yet Implemented

The CC implementation spec now correctly identifies the existing legacy crypto
coupling in `scheduler/jobs.py`, `config/watchlist.py`, and
`data/collectors/crypto.py`.

But the code still has that coupling today:

- `scheduler/jobs.py` imports `CryptoCollector`.
- `DailyPipeline.__init__()` instantiates it.
- BTC/ETH can be appended to recommendation candidates.
- Crypto data is fetched inside the same daily scheduler/report path as A-share.

This means the project is not yet isolated at runtime. It is only isolated in
the intended design.

Required improvement:

```text
PRE-Phase-0a:
  Implement legacy crypto quarantine.
  Default legacy crypto context OFF.
  Observe A-share cron GREEN for 3 days.
```

The quarantine PR should be tiny. It should not introduce the new crypto
pipeline. Its only goal is to prove A-share can run without touching legacy
crypto code.

Acceptance:

- `LEGACY_MARKET_CONTEXT_ENABLED=false` by default.
- `scheduler/jobs.py` does not import `CryptoCollector` at module import time.
- `DailyPipeline` initializes without constructing any crypto collector.
- BTC/ETH are not added to A-share recommendation candidates when the flag is
  off.
- A-share recommendation output is unchanged except for removed crypto
  background text.
- A-share daily cron is GREEN for 3 production days with flag off.

### 2. Data Reliability Is Still Too Single-source

The Phase A plan uses Binance as primary and OKX/Bybit as fallback reachability
checks. That is a useful start, but it is not enough for a system that will
eventually trust crypto signals.

Crypto exchange data can differ across venues because of:

- local venue liquidity and outages.
- different symbol/instrument conventions.
- temporary API lag.
- funding timestamp differences.
- exchange-specific OHLCV revisions.
- maintenance windows and withdrawal suspensions.

Required improvement:

```text
Phase A:
  Collect Binance primary data.
  Also sample OKX/Bybit comparison data for BTC/ETH/SOL.
  Store comparison metrics, not necessarily full duplicate history at first.
```

Minimum cross-source checks:

- latest 1h close difference between Binance and OKX/Bybit.
- latest 1h volume availability.
- missing bar count by exchange.
- funding timestamp alignment for BTC/ETH/SOL perps.
- funding-rate sign agreement across venues.
- API response latency and failure rate.

Health status should include:

```json
{
  "cross_source": {
    "BTC_USDT_1h_close_spread_bps": 3.2,
    "BTC_USDT_funding_sign_agree": true,
    "binance_latency_ms": 420,
    "okx_latency_ms": 510,
    "bybit_latency_ms": 480
  }
}
```

Suggested health rule:

- RED: primary exchange unreachable or primary latest bar stale.
- YELLOW: fallback unreachable or cross-source close spread exceeds threshold.
- GREEN: primary and at least one fallback healthy.

Do not train or backtest on cross-source merged data in Phase A. Just measure
whether the primary data is sane.

### 3. Phase A Universe Is Fine For Plumbing, Not For Alpha

The 5-coin universe is the right Phase A plumbing universe:

- BTC
- ETH
- SOL
- BNB
- XRP

But it is too small for serious cross-sectional statistics. With five names,
RankIC, top-bottom spread, and decile logic are too noisy.

Required improvement:

```text
Phase A:
  Use 5 coins for data plumbing and paper reporting.

Phase B/C:
  Treat alpha statistics as diagnostic only.

Phase E+:
  Consider top-20/top-30 expansion only after data health is stable.
```

Universe expansion should not be "just config". It changes risk surface.

Before top-20/top-30:

- delisted coin handling exists.
- stablecoin/depeg filter exists.
- scam/meme/liquidity filter exists.
- min dollar volume and listed-days filters exist.
- exchange coverage is measured.
- survivorship tests run.

Acceptance before expansion:

- 7 consecutive GREEN data-health days.
- gap rate below threshold for current universe.
- fallback exchange health verified.
- no A-share cron degradation during crypto collection.

### 4. Paper OMS Needs Its Own Serious Spec

The current CC document has good Phase D acceptance bullets, but the paper OMS
is the hardest part of the future system. It deserves its own design doc before
implementation.

Crypto paper OMS must model:

- T+0 immediate settlement.
- partial fills or conservative no-partial-fill assumption.
- maker/taker fee.
- min notional.
- lot size and tick size precision.
- quote currency balance.
- funding payment accrual for perps.
- stale quote refusal.
- exchange maintenance windows.
- failed order simulation.
- reconciliation against a synthetic statement.

Even paper-only mode can lie if OMS accounting is wrong. If paper PnL is wrong,
all downstream model evaluation is fake.

Required improvement:

```text
Before Phase D:
  Write plans/crypto-paper-oms-spec-YYYY-MM-DD.md
```

Minimum state model:

```text
Account:
  base_ccy: USDT
  cash_balance
  equity
  realized_pnl
  unrealized_pnl
  fee_paid
  funding_paid

Position:
  symbol
  quantity
  avg_entry_price
  mark_price
  notional
  unrealized_pnl
  opened_at_utc

Order:
  order_id
  symbol
  side
  quantity
  requested_price
  fill_price
  fill_time_utc
  fee
  status
  reject_reason
```

Minimum refusal rules:

- refuse if data health is RED.
- refuse if latest quote stale.
- refuse if notional below min_notional.
- refuse if quantity violates lot precision.
- refuse if target weight exceeds max weight.
- refuse if stablecoin/depeg RiskGuard active.

Paper OMS acceptance:

- BUY then SELL same bar works under `InstantSettlementModel`.
- min-notional and precision tests pass.
- fee and slippage are included.
- funding payment line appears for perps.
- daily paper report reconciles account equity to positions + cash.
- no live keys exist or are read.

### 5. RiskGuard Should Be More Crypto-native

The current RiskGuard direction is good, but it should be strengthened before
any paper strategy becomes "trusted".

Crypto-specific RiskGuard layers to add:

- stale primary exchange data.
- fallback exchange unavailable.
- cross-exchange close spread abnormal.
- funding-rate z-score extreme.
- OI surge with price divergence.
- market-wide liquidation cascade proxy.
- stablecoin depeg basket risk.
- withdrawal halt / exchange incident flag.
- BTC dominance or ETH/BTC breakdown.
- high volatility circuit breaker.

RiskGuard should first reduce or block paper actions, not try to improve alpha.

Suggested rules:

```text
RED:
  no new paper orders, report only

YELLOW:
  cap max weight
  reduce turnover
  refuse new perp/funding-arb paper exposure

GREEN:
  normal paper policy
```

Minimum acceptance:

- Every blocked action appears in daily report.
- Every RiskGuard trigger has timestamp, rule name, input values, and action.
- A repeated trigger does not spam duplicate report entries.

### 6. Daily Report Should Be Built For Observation, Not Just Snapshot

The report is the user's main interface during the paper-only period. It should
make change over time obvious.

Add a "What Changed Since Yesterday" section:

```text
What Changed Since Yesterday
- Position changes: BTC 20% -> 25%, ETH 20% -> 15%
- New signals: SOL moved from hold to buy
- Removed signals: XRP rejected by liquidity/stale-data guard
- Data health: OKX fallback changed GREEN -> YELLOW
- RiskGuard: funding_extreme triggered for BTC perp
- Model: no active model change; shadow model v003 differed on ETH
```

Add a "Paper vs Shadow" section:

```text
Paper Policy:
  rule baseline / active model version

Shadow Candidates:
  supervised model
  funding-arb shadow
  event/on-chain shadow

Decision:
  what paper portfolio did
  what shadow would have done
```

Add a "Data Trust" section:

```text
Data Trust
- Primary exchange: Binance GREEN
- Fallback: OKX GREEN, Bybit YELLOW
- Latest closed 1h bar age: 7 min
- Gap rate 30d: 0.1%
- Cross-source BTC close spread: 2.8 bps
```

The daily report should help the user build intuition. It should not merely
dump portfolio rows.

### 7. Research Scope Should Stay Narrow Through Phase D

The current design includes many possible research directions:

- funding paper strategy.
- rule-based baselines.
- XGB/LGB supervised model.
- on-chain factors.
- LLM event extraction.
- Kronos / RD-Agent / GraphSAGE.
- Nautilus.
- RL.

These are valid later, but the early system should answer one question:

```text
Can we produce reliable, reproducible, paper-only crypto signals without
degrading A-share?
```

Recommended research freeze:

```text
Before Phase D:
  no Kronos
  no RD-Agent
  no GraphSAGE
  no RL
  no DeFi
  no Nautilus
  no live/testnet
```

Allowed before Phase D:

- OHLCV data.
- funding/OI data.
- data health.
- rule-based baselines.
- shadow XGB/LGB if Phase C starts.
- daily report.

This prevents the project from expanding faster than its trust foundation.

## Revised Phase Recommendations

### PRE-Phase-0a: Legacy Crypto Quarantine

Goal:

Prove the A-share scheduler can run without touching legacy crypto code.

Tasks:

- Add `LEGACY_MARKET_CONTEXT_ENABLED=false` default.
- Lazy-import legacy `CryptoCollector` only when flag is true.
- Do not add BTC/ETH candidates when flag is false.
- Skip crypto forecast/report context when flag is false.
- Add A-share smoke test with flag false.

Acceptance:

- A-share daily cron GREEN for 3 production days with flag false.
- A-share stock recommendation list unchanged.
- No crypto network calls made during A-share daily run.

### Phase 0a: Contracts And Measurement

Goal:

Create contracts and measure external data constraints without touching A-share
production code.

Tasks:

- `plans/crypto-data-contract.md`
- `plans/numeric_claims_audit.md`
- `scripts/crypto_phase0_spike.py`
- exchange accessibility measurement.
- rate-limit measurement.
- history-depth measurement.
- funding-depth measurement.
- network profile validation.

Acceptance:

- Measurement results saved.
- No A-share production file changed.
- No live keys.
- No leverage.
- No trading.

### Phase 0b: Minimal Core Protocols

Goal:

Create asset-neutral interfaces without changing existing behavior.

Tasks:

- `core/asset.py`
- `core/instrument.py`
- `core/calendar.py`
- `core/settlement.py`
- `core/cost.py`

Acceptance:

- Import isolation lint passes.
- A-share does not yet depend on new core code unless behind adapters.
- A-share cron remains GREEN.

### Phase A: Data Foundation

Goal:

Collect trustworthy closed-bar crypto data.

Tasks:

- `data/collectors/crypto_market.py`
- `data/collectors/crypto_derivatives.py`
- `scripts/crypto_update_market_data.py`
- `scripts/crypto_update_derivatives.py`
- `scripts/crypto_data_health.py`
- `data/storage/crypto/**`

Acceptance:

- 1y OHLCV for 5 symbols.
- 1y funding for 3 perp symbols.
- 3 days OI samples.
- 7 GREEN health days.
- fallback exchange reachability measured.
- no A-share cron degradation.

Recommended addition:

- cross-source sanity metrics for BTC/ETH/SOL.

### Phase B: Rule Baselines

Goal:

Validate feature and backtest machinery without model risk.

Tasks:

- equal-weight baseline.
- momentum baseline.
- reversal baseline.
- BTC-beta-neutral baseline.
- fee/slippage model.

Acceptance:

- manual hand-calc sanity check passes.
- no model trained.
- feature cache is reproducible.
- all numbers are `[validated-on-local]`.

### Phase C: Shadow Supervised Model

Goal:

Validate model pipeline, not production alpha.

Tasks:

- XGB/LGB training.
- walk-forward split.
- weekly retrain shadow.
- predictions saved to shadow report.

Acceptance:

- no hard IC/ICIR gate for N=5.
- no model controls paper portfolio without user sign-off.
- shadow output appears next to rule baseline in daily report.

### Phase D: Paper OMS

Goal:

Make paper PnL trustworthy.

Before implementation:

- write dedicated paper OMS spec.

Acceptance:

- account ledger reconciles.
- fees/slippage/funding accounted.
- T+0 settlement works.
- precision/min-notional rules enforced.
- no live keys.
- no leverage.

## Highest Priority Punch List

1. Implement legacy crypto quarantine.
2. Add no-crypto-network-call test for A-share daily run.
3. Add `crypto_global` network profile fail-fast behavior.
4. Add cross-source data sanity metrics.
5. Write dedicated Phase D paper OMS spec.
6. Improve daily report with "what changed since yesterday".
7. Keep frontier models frozen until paper data and OMS are trusted.

## Final View

The current crypto design is directionally strong and now has the right
engineering temperament. It should not chase more alpha yet.

The next milestone is not:

```text
make crypto model profitable
```

The next milestone is:

```text
prove crypto can run in paper mode for weeks without touching A-share,
without stale data, without fake PnL, and without hidden operational coupling.
```

If that is achieved, the design moves from **8.0/10** to roughly **9.0/10**.

