# Crypto Quant Roadmap on Current Project

Date: 2026-05-30

## Executive Summary

This project can expand into cryptocurrency quant, but it should not simply add
BTC/ETH to the existing A-share recommender. Crypto is a different market:
24/7 trading, exchange fragmentation, derivatives funding, leverage/liquidation
flows, on-chain activity, social attention, and far higher regime turnover.

Current project state:

- Existing support is thin: `data/collectors/crypto.py` fetches Binance spot
  OHLCV/realtime via CCXT, with AKShare fallback.
- `scheduler/jobs.py` adds BTC/ETH as simple candidates using `change_pct / 10`.
- `signals/index_predictor.py` uses BTC/ETH change as a cross-market risk
  appetite input for A-share index prediction.
- There is no crypto-specific data store, feature cache, label definition,
  backtest engine, execution simulator, exchange risk model, or promotion gate.

Recommended direction:

1. Build crypto as a separate asset-class pipeline under the same production
   discipline, not as another stock factor.
2. Start with spot-only BTC/ETH/SOL/BNB/XRP and liquid majors, then add
   perpetual futures only after data, backtest, and paper trading are stable.
3. Use supervised ranking/forecasting first. Keep RL for Phase C/D as a sizing
   and execution/risk controller, not as the first alpha engine.
4. Reuse project infrastructure: `run_network_job.py`, `run_with_status.py`,
   `data_health.py`, paper OMS concepts, factor gate, shadow comparison, and
   daily reports.
5. Avoid overfitting: crypto has abundant bars but few independent regimes.
   Require walk-forward validation, exchange fee/slippage, funding costs, and
   paper trading before any live trading.

## Research Takeaways

### 1. Useful open-source projects

| Project | Best Use | Why It Matters |
|---|---|---|
| Freqtrade + FreqAI | ML-driven crypto strategy sandbox | FreqAI supports adaptive retraining, large feature sets, threaded retraining, and live inference separation. Good reference for feature/label/retrain workflows. |
| NautilusTrader | Production-grade backtest/live execution engine | Strongest candidate for serious execution simulation and live parity. It supports crypto venues, event-driven backtests, live nodes, account modes, order simulation, and exchange adapters. |
| Hummingbot | Market making and arbitrage | Best reference if later doing spread capture, cross-exchange arbitrage, liquidity provision, or CEX/DEX bots. Less suitable as the first directional alpha stack. |
| vectorbt / backtrader | Fast research/backtest | Useful for early signal sweeps, but weaker for production exchange semantics than NautilusTrader. |
| Qlib | Current project research style | Good for cross-sectional supervised factor research if we convert crypto bars to Qlib-like instruments, but Qlib is not enough for 24/7 execution/perps semantics. |

Primary references:

- FreqAI docs: https://docs.freqtrade.io/en/2025.1/freqai/
- NautilusTrader docs: https://nautilustrader.io/docs/
- Nautilus live trading: https://nautilustrader.io/docs/latest/concepts/live/
- Nautilus Bybit integration: https://nautilustrader.io/docs/latest/integrations/bybit
- Hummingbot docs: https://hummingbot.org/docs/

### 2. Crypto factors are not just Alpha158

Academic and practitioner evidence points to several factor families:

- Market, size, and momentum are common crypto risk factors.
- Short-term reversal exists broadly, but large/liquid coins often show momentum
  while small/illiquid coins reverse.
- Liquidity and volume strongly condition whether momentum or reversal works.
- Network attention and social hype can explain part of crypto returns.
- On-chain factors add information, especially network activity,
  scale-adjusted activity, valuation ratios, and token distribution.
- Perpetual futures introduce crypto-native signals: funding rate, open
  interest, basis, liquidation clusters, long/short imbalance, and forced
  deleveraging risk.

Useful papers:

- Liu, Tsyvinski, Wu, "Common Risk Factors in Cryptocurrency"
  https://www.nber.org/papers/w25882
- "Up or down? Short-term reversal, momentum, and liquidity effects in
  cryptocurrency markets"
  https://www.sciencedirect.com/science/article/pii/S1057521921002349
- "Impact of Size and Volume on Cryptocurrency Momentum and Reversal"
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4378429
- "On-Chain Factors and Cryptocurrency Asset Pricing"
  https://papers.ssrn.com/sol3/Delivery.cfm/6670521.pdf?abstractid=6670521&mirid=1
- "Deep reinforcement learning applied to statistical arbitrage investment
  strategy on cryptomarket"
  https://www.sciencedirect.com/science/article/abs/pii/S1568494624000292
- "Online probabilistic knowledge distillation on cryptocurrency trading using
  Deep Reinforcement Learning"
  https://www.sciencedirect.com/science/article/abs/pii/S0167865524002939
- "Revisiting Ensemble Methods for Stock Trading and Crypto Trading Tasks at
  ACM ICAIF FinRL Contest 2023-2024"
  https://arxiv.org/abs/2501.10709

## Strategic Positioning

Crypto should be split into three tracks:

### Track 1: Directional Spot Alpha

Goal: rank liquid crypto assets and forecast 4h/24h returns.

Universe:

- Phase A: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT.
- Phase B: top 30 USDT spot pairs by rolling dollar volume.
- Exclude stablecoins, wrapped duplicates, leveraged tokens, illiquid listings,
  obvious exchange-maintained synthetic products.

Labels:

- `ret_4h = close[t+4h] / close[t] - 1`
- `ret_24h = close[t+24h] / close[t] - 1`
- Optional classification labels:
  - `up_24h_after_fee`
  - `top_quantile_24h`
  - `drawdown_next_24h`

Model:

- Start with XGB/LGB ranking and regression.
- Use rank IC, ICIR, top-bottom spread, turnover, fee-adjusted return.
- Later add ensemble: XGB + LGB + linear/ridge + temporal CNN/Transformer only
  after simple models pass.

### Track 2: Perpetual Futures Risk/Carry

Goal: use derivatives data for risk control and optional carry overlay.

Data:

- Funding rate
- Open interest
- Long/short ratio
- Basis vs spot
- Liquidation volume
- Mark price/index price
- Insurance/funding anomalies if available

Signals:

- Crowded long risk: funding high + OI rising + price momentum exhausted.
- Short squeeze risk: funding negative + OI high + price breaks up.
- Carry: positive expected funding only if price risk and liquidation risk pass.
- Deleveraging risk flag: OI collapse + high liquidation + volatility spike.

Use:

- First as RiskGuard/regime, not direct alpha.
- Later as futures-specific overlay.

### Track 3: Event/Sentiment/On-chain

Goal: crypto-native external signals.

Data:

- Crypto news RSS/GDELT
- Exchange announcements: Binance, OKX, Bybit, Coinbase
- Regulatory/news events: ETF, SEC/CFTC, MiCA, stablecoin legislation
- On-chain: active addresses, transaction count, fees, exchange inflow/outflow,
  realized cap/MVRV style features, holder concentration
- Social attention: Google Trends, Reddit/X if accessible, Telegram/Discord only
  if a reliable source exists

Use:

- LLM event extraction only for high-impact items:
  - ETF/regulatory approvals
  - exchange listing/delisting
  - hack/exploit
  - stablecoin depeg
  - protocol upgrade
  - token unlock
  - founder/legal event
- Do not feed raw LLM sentiment directly into the main model until shadow
  validation shows IC contribution.

## Architecture

Target architecture:

```text
short term: light core protocols + greenfield crypto package
medium term: adapters/facades while A-share production paths stay stable
long term: core/ + ashare/ + crypto/ namespaces
```

The `crypto_` file names in the early implementation plan are transitional, not
the intended final architecture. The final design should separate shared
asset-agnostic interfaces from A-share-specific and crypto-specific
implementations.

Core abstractions to introduce gradually:

- `AssetClass`
- `InstrumentClass`
- venue-aware `Symbol`
- `TimeAxis` / `Always24x7Calendar`
- `ISettlementModel`
- `CommissionModel`
- `TaxModel`
- `ImpactModel`
- `UniverseFilter`
- `Instrument.round_qty`
- multi-currency ledger

### Crypto Time And Settlement Semantics

Crypto must not inherit the A-share T+1 mental model.

Key differences:

- Crypto trades 24/7, including weekends and holidays.
- Spot crypto is effectively T+0: after a buy fills, the position can be sold
  immediately.
- Perpetual futures are also continuous and margin-driven, with funding
  settlement cycles.
- There is no daily open/close auction equivalent to A-share execution.
- There is no ST, limit-up/limit-down, or 100-share lot concept.

Production implication:

```text
A-share:
  signal_date -> next trading-day open fill -> T+1-style reconcile

Crypto:
  signal_ts -> next closed bar / next bar open fill -> immediate settlement
```

The first crypto implementation should still avoid tick/HFT complexity:

- use closed candles only
- start with 1h / 4h / 1d bars
- refresh data every 30 minutes or at 4h bar boundaries
- refuse to trade on stale or incomplete bars
- use UTC as canonical time
- model immediate fills in paper OMS, or next-bar-open fills for conservative
  research consistency

So yes, crypto needs more real-time discipline than A-share, but Phase A should
be "bar-real-time", not millisecond order-book trading.

Recommended directory additions:

```text
data/collectors/crypto_market.py
data/collectors/crypto_derivatives.py
data/collectors/crypto_onchain.py
data/storage/crypto/
  raw/ohlcv/{exchange}/{timeframe}/{symbol}.parquet
  raw/funding/{exchange}/{symbol}.parquet
  raw/open_interest/{exchange}/{symbol}.parquet
  features/crypto_feature_cache.parquet
  predictions/crypto_predictions_latest.json
  paper/oms_state.json
scripts/crypto_update_market_data.py
scripts/crypto_build_features.py
scripts/crypto_train_model.py
scripts/crypto_predict.py
scripts/crypto_backtest.py
scripts/run_crypto_paper_trading.py
scripts/crypto_daily_report.py
models/crypto_feature_pipeline.py
models/crypto_model.py
paper/crypto_oms.py
backtest/crypto_backtest.py
plans/crypto-quant-roadmap-2026-05-30.md
```

Keep crypto separate from A-share feature cache. The A-share system has trading
day assumptions, T+1 execution assumptions, ST/BJ filters, Qlib instrument
format, and daily close semantics. Crypto is 24/7, continuous, and exchange
specific.

## Data Plan

### Evidence Tags

All numeric claims used for production decisions must be tagged:

- `[paper-reported]`
- `[exchange-dashboard]`
- `[validated-on-local]`

Rule:

```text
Any number used for sizing, risk thresholds, or promotion must be
[validated-on-local].
```

Paper-reported Sharpe, IC, funding rates, t-stats, or return spreads may
motivate research, but they cannot justify live capital.

### Phase A Data: Spot OHLCV

Collector:

- Use CCXT first.
- Exchanges: Binance if accessible, fallback OKX/Bybit.
- Timeframes: `1h`, `4h`, `1d`.
- Store as parquet with schema:

```text
timestamp_utc
exchange
symbol
open
high
low
close
volume_base
volume_quote
is_closed_bar
ingested_at
```

Important:

- Only use closed candles for labels/features.
- Use UTC as canonical time.
- Maintain gap detection: expected timestamp grid per symbol/timeframe.
- Run collector under `network=global` or `network=crypto_global` depending on
  exchange accessibility.

Acceptance:

- 365 days of 1h bars for BTC/ETH/SOL/BNB/XRP.
- Gap rate < 0.5%.
- No duplicate `(exchange, symbol, timeframe, timestamp_utc)`.
- Data health file written daily.

### Phase A Data: Derivatives / Funding / OI

Collector:

- Funding rate history
- Open interest
- Long/short ratio
- Mark/index price
- Liquidation feed if available

Schema:

```text
timestamp_utc
exchange
symbol
funding_rate
next_funding_time
open_interest
long_short_ratio
basis
mark_price
index_price
```

Acceptance:

- Funding and OI coverage for BTC/ETH/SOL perps.
- Funding timestamp aligned to exchange schedule.
- Missing derivative fields do not break spot alpha.

Positioning:

- Funding/OI data belongs in Phase Crypto-A with OHLCV because it is a
  crypto-native data axis.
- Funding arbitrage as a strategy remains gated to later paper trading.

### Phase C Data: On-chain/Event

Start small:

- BTC/ETH network-level series.
- Exchange inflow/outflow if free/reliable.
- Token unlock/calendar for altcoins.
- Major exchange announcements.

Do not spend too much time here before spot+derivatives backtest exists.

## Feature Plan

### Hard Gotchas

Every crypto factor imported from A-share or traditional equity research must
pass a forced sign and robustness check.

Known traps:

- IVOL sign may differ from A-share. Do not port the A-share sign.
- MAX / lottery-style behavior may differ from A-share.
- Traditional value/book-to-price does not transfer cleanly to crypto.
- BTC/ETH cointegration is not a safe default; structural breaks such as the
  Ethereum Merge can invalidate pair logic.
- Survivorship bias is severe; historical cross-section tests must preserve
  delisted/dead coins where possible.
- Alpha decay is likely faster than A-share; monitoring windows and retraining
  cadence should be shorter.
- Funding/carry signals are not the same as safe arbitrage. Strategy use
  requires margin, basis, collateral, and liquidation simulation.

### Core Price/Volume Features

For each symbol/timeframe:

- Returns: 1h, 4h, 12h, 24h, 3d, 7d
- Momentum: ROC, distance to rolling high/low
- Reversal: last-bar return, 4h reversal, volatility-conditioned reversal
- Volatility: realized vol 12h/24h/7d, Parkinson/Garman-Klass if high/low clean
- Volume: volume z-score, quote-volume rank, volume shock
- Liquidity: dollar volume, Amihud illiquidity, spread proxy if order book absent
- Trend: MA crossover, price/MA ratios, breakout
- Drawdown: distance from rolling peak, intraday range shock

### Cross-sectional Features

- Market beta to BTC
- Relative strength vs BTC/ETH
- Sector/category if available: L1, meme, DeFi, exchange token, infra
- Rank features across universe:
  - rank return 24h
  - rank volume shock
  - rank volatility
  - rank liquidity

### Perp Features

- funding_rate current / rolling mean / z-score
- funding sign persistence
- OI change 1h/4h/24h
- price up + OI up = leveraged trend
- price down + OI up = crowded short or panic buildup
- price up + OI down = short covering
- liquidation shock
- basis z-score

### Event/Attention Features

- exchange listing/delisting flag
- hack/exploit flag
- regulatory positive/negative flag
- ETF/stablecoin/liquidity policy flag
- news count 24h
- LLM event severity
- Google/news attention z-score

## Modeling Plan

### Baseline Models

1. Naive momentum/reversal rules:
   - Liquid majors: momentum more likely.
   - Illiquid alts: reversal more likely.
2. XGB/LGB regression:
   - Predict `ret_24h_after_fee`.
3. XGB/LGB ranking:
   - Rank symbols each 4h or daily.
4. Crash/downside classifier:
   - Predict probability of next 24h drawdown > threshold.

### Metrics

Research metrics:

- RankIC by timestamp
- ICIR
- Top-bottom spread
- Hit rate
- Long-only top-k return after fees
- Turnover
- Max drawdown
- Exposure to BTC beta
- Performance by regime:
  - BTC uptrend
  - BTC downtrend
  - high volatility
  - high funding
  - low liquidity weekend

Production metrics:

- Fee-adjusted daily PnL
- Slippage-adjusted PnL
- Number of trades
- Average holding period
- Worst 1d/3d drawdown
- Health coverage
- Data gap rate

Promotion gate:

- At least 90 calendar days historical backtest for Phase A.
- At least 30 calendar days paper trading.
- Positive fee-adjusted Sharpe and max drawdown under threshold.
- Stable IC across at least 4 market regimes.
- No single token contributes more than 40% of total PnL.

## Backtest and Execution

### Early Backtest

Use internal vectorized backtest first:

- Rebalance every 4h or daily.
- Long-only spot.
- Fee: Binance/OKX taker/maker conservative default.
- Slippage: 5-20 bps depending liquidity.
- No leverage.
- Position cap: 20-40% for BTC/ETH, 10-15% for alts.
- Cash allowed.

### Serious Backtest

Evaluate NautilusTrader after Phase A:

- It gives better research-to-live parity.
- It models orders, accounts, venues, and execution more rigorously.
- It is more appropriate before real exchange trading.

### Execution

Do not live trade immediately.

Paper stages:

1. Signal-only dashboard.
2. Paper OMS with simulated fills at next closed bar.
3. Exchange testnet/demo if using perps.
4. Tiny capital spot-only live canary, only after paper passes.

Risk:

- No leverage in Phase A/B.
- Hard position cap.
- Stop trading on data gap.
- Stop trading on exchange API anomaly.
- Stop trading if BTC 1h realized vol > threshold.
- Stablecoin depeg guard.
- Exchange maintenance/withdrawal halt guard.

## Cron Plan

Crypto is 24/7. Do not force it into A-share 9:20/14:30/22:00 only.

Recommended jobs:

```text
# Data
*/30 * * * * crypto_update_market_data --timeframe 1h --network global --timeout 600
5 */4 * * * crypto_update_market_data --timeframe 4h --network global --timeout 900
10 */4 * * * crypto_build_features --timeout 900
20 */4 * * * crypto_predict --timeout 600
25 */4 * * * crypto_paper_trading --timeout 600

# Reports
0 9,21 * * * crypto_daily_report --timeout 300

# Weekly
0 5 * * 0 crypto_train_model --timeout 7200
```

Integrate into `scripts/install_crontab.py` only after the scripts have health
checks. Use `run_network_job.py` with a crypto/global profile. Global exchange
APIs may need proxy; domestic collectors should remain proxy-free.

## Phase Roadmap

### Phase Crypto-0: Design and Data Audit

Duration: 1-2 weeks (revised from initial "1-2 days" per CC Implementation
Punch List item A, 2026-05-30). The scope grew over cc↔cx convergence
to include §14.1 audit version-locking, numeric claims tagging,
target-architecture decision, ssproxy / external-volume / legacy-quarantine
prerequisites, and the measurement spike — 1-2 days is unrealistic
for this deliverable set. Realistic estimate aligned with CC spec §1
Phase 0a/0b/0c sub-phases.

Tasks:

- Decide exchanges: Binance/OKX/Bybit.
- Decide initial symbols: BTC, ETH, SOL, BNB, XRP.
- Define UTC bar schema.
- Define data health contract.
- Define settlement-vs-data-latency split:
  - T+0 settlement is a Phase Crypto-0 OMS state-machine requirement.
  - WebSocket-class real-time data is deferred to Phase Crypto-G+.
  - Phase A-F use REST closed bars with stale/incomplete-bar guards.
- Write `plans/crypto-data-contract.md`.
- Create asset-implicit assumption checklist from the CC audit.
- Add evidence tags to numeric assumptions.
- Define target architecture as `core/ + ashare/ + crypto/`, with physical
  migration delayed.
- Write "library over framework" rule.
- Write DeFi out-of-scope paragraph.

Acceptance:

- One schema doc.
- One collector design.
- No model work yet.
- Every data field has dtype, timestamp semantics, and PIT availability rule.
- The design explicitly separates T+0 settlement from real-time data latency.
- A-share production paths remain untouched.

### Phase Crypto-A: Spot Data Foundation

Duration: 3-5 days.

Scripts:

- `scripts/crypto_update_market_data.py`
- `data/collectors/crypto_market.py`
- `data/collectors/crypto_derivatives.py`
- `scripts/crypto_data_health.py`

Acceptance:

- 1h/4h/1d bars saved to parquet.
- Funding/OI history saved for BTC/ETH/SOL perps where available.
- Gap detection works.
- Daily health file exists.
- Collector is idempotent.

### Phase Crypto-B: Feature Cache and Baseline Backtest

Duration: 5-7 days.

Scripts:

- `models/crypto_feature_pipeline.py`
- `scripts/crypto_build_features.py`
- `scripts/crypto_backtest_baseline.py`

Acceptance:

- Feature cache exists.
- Baseline momentum/reversal backtest runs.
- Fee/slippage included.
- Report has Sharpe, max drawdown, turnover, top-bottom spread.

### Phase Crypto-C: Supervised Model

Duration: 7-10 days.

Scripts:

- `models/crypto_model.py`
- `scripts/crypto_train_model.py`
- `scripts/crypto_predict.py`
- `scripts/crypto_factor_gate.py`

Acceptance:

- XGB/LGB model trained on rolling split.
- Predictions saved to `crypto_predictions_latest.json`.
- RankIC/ICIR reported by split.
- Shadow recommendation report generated.

### Phase Crypto-D: Paper OMS

Duration: 5-7 days.

Scripts:

- `paper/crypto_oms.py`
- `scripts/run_crypto_paper_trading.py`
- `scripts/crypto_daily_report.py`

Acceptance:

- Paper orders generated every 4h or daily.
- Positions and PnL tracked.
- Fee/slippage accounted.
- No live exchange keys required.
- **Collateral ledger reconciliation prototype** against a synthetic
  exchange statement schema (per CC Implementation Punch List item C,
  2026-05-30) — even paper-only mode requires the ledger to be
  reconciliation-shaped so Phase E/G+ stress simulations have a real
  surface to exercise.
- Paper observation gated by **user written sign-off** rather than a
  fixed "30 days" trigger (per §−1 paper-only constraint). The 30-day
  marker remains as a soft target for when CC asks the user "ready to
  discuss next step?", not as auto-promotion.

### Phase Crypto-E: Derivatives RiskGuard and Paper Funding Strategy

Duration: 1-2 weeks.

Scripts:

- `data/collectors/crypto_derivatives.py`
- `models/crypto_derivative_features.py`
- `risk/crypto_risk_guard.py`

Acceptance:

- Funding/OI features available for majors.
- High-crowding risk flag tested.
- RiskGuard can reduce exposure or block entries.
- Funding-arbitrage strategy exists only in paper, leverage=1.0
  (no margin, no real testnet — per §−1).
- **6 minimum-evidence items before any live funding-arb canary** (per
  CC Implementation Punch List item C, 2026-05-30):
  1. Venue-specific funding history (not only aggregate dashboard)
  2. Net-of-fee, net-of-slippage, net-of-borrow/collateral-cost backtest
  3. Stress windows: consecutive negative funding + large spot/perp moves
  4. Simulated withdrawal halt + venue outage + stale WebSocket behavior
  5. Collateral ledger reconciliation against synthetic exchange statement
  6. Max-loss calculation under liquidation-buffer assumptions
- Funding-arb paper PnL appears in §11 daily report (cc spec) as a
  separate line so user can compare to spot-only PnL during observation.
- No real leverage until local validation and paper evidence pass.

### Phase Crypto-F: Event/On-chain Overlay

Duration: 1-2 weeks.

Scripts:

- `data/collectors/crypto_events.py`
- `factors/crypto_event_store.py` or extend `EventStore` with asset type
- `scripts/crypto_build_event_factors.py`

Acceptance:

- Exchange announcement events parsed.
- Hack/listing/regulatory events tagged.
- Overlay runs in shadow only.
- No main-model inclusion until gate passes.
- RD-Agent/Kronos/CryptoTrade/GraphSAGE are evaluated only as shadow/research
  backlog items.

### Phase Crypto-G: Production Engine Evaluation

Duration: 1-2 weeks.

Tasks:

- Prototype NautilusTrader backtest for BTC/ETH spot.
- Compare internal backtest vs Nautilus fills/costs.
- Decide whether to use Nautilus for live/testnet execution.
- Begin physical `core/ + ashare/ + crypto/` migration only if A-share
  regression snapshots and production cron stability pass.

Acceptance:

- One working Nautilus backtest.
- One documented migration decision.

## What Not To Do Yet

- Do not trade leverage first.
- Do not use RL as the first alpha model.
- Do not rely on raw LLM sentiment for buy/sell.
- Do not merge crypto into A-share `feature_cache_174`.
- Do not treat Binance close as a universal crypto close; exchange matters.
- Do not ignore fees. Many crypto signals disappear after fees.
- Do not use minute-level prediction until bar data and paper OMS are stable.
- Do not put DeFi yield automation into the quant pipeline.

### DeFi Carve-out

DeFi yield is out of scope for the crypto quant pipeline. It may be tracked in
a separate capital-management note as a benchmark or treasury option, but it
should not affect model promotion, paper OMS, or signal validation.

### Library Over Framework

The production pipeline should prefer composable libraries over full trading
frameworks:

- Prefer: CCXT, Polars, DuckDB, internal vectorized backtests.
- Use as references: FreqAI, Hummingbot, FinRL.
- Evaluate later: NautilusTrader for exchange-realistic execution.

Reason:

- Frameworks impose process models that can conflict with the existing
  cron/parquet/jsonl/data-health workflow.
- Libraries preserve the current project's modular discipline.

## Recommended First Implementation Sprint

The first sprint should be boring and infrastructure-heavy:

1. Create crypto data schema and parquet store.
2. Build robust OHLCV collector with gap detection.
3. Build features for BTC/ETH/SOL/BNB/XRP.
4. Run baseline momentum/reversal backtest.
5. Produce a daily crypto report, no trading.

Only after this should the project add ML, paper OMS, derivatives, or RL.

## Success Criteria Before Live Money

- 30+ days paper trading.
- All jobs health-gated.
- Data gap rate < 0.5%.
- Backtest and paper use same signal timing.
- Fee/slippage included.
- No position opens if data stale.
- Strategy remains profitable after doubling fee/slippage assumptions.
- No single token dominates PnL.
- Manual kill switch exists.

## Bottom Line

Crypto is feasible on this Mac Studio and this codebase, but it should be a
parallel asset-class system. The best immediate edge is not fancy RL; it is
disciplined spot OHLCV + funding/OI data, fee-aware factor testing,
derivatives crowding signals, and strict paper trading. Once the supervised
spot model and crypto RiskGuard are stable, NautilusTrader is the strongest
candidate for serious exchange-grade execution, while FreqAI is the best
reference for ML feature/retrain workflow and Hummingbot is best saved for
later market-making/arbitrage experiments.
