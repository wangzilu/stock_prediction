# Crypto Quant Literature and Engineering Review

Date: 2026-05-30

Companion document:

- `plans/crypto-quant-roadmap-2026-05-30.md`

This document supplements the roadmap with a more systematic review of recent
papers, open-source systems, and direct implications for this project.

Important caveat: this is not literally every paper or every repository in the
world. It is a practical, engineering-oriented review of the major evidence
clusters most relevant to building a crypto quant extension on this project.

## Bottom Line

The strongest immediate path is:

1. Build crypto data infrastructure first.
2. Start with spot OHLCV + funding/OI + fee-aware supervised ranking.
3. Add derivatives crowding/funding/open-interest data as RiskGuard and regime
   features.
4. Add event/on-chain/social features only after baseline alpha and backtest
   discipline are stable.
5. Keep RL as Phase D/E for allocation and risk control, not Phase A alpha.
6. Use NautilusTrader as the serious execution/backtest reference, FreqAI as the
   supervised ML workflow reference, and Hummingbot as the later market-making
   and arbitrage reference.

## Evidence Map

| Area | Evidence Strength | Project Action |
|---|---:|---|
| Spot momentum/reversal/liquidity | High | Phase Crypto-B baseline factors |
| Size/liquidity/factor models | High | Cross-sectional crypto ranking |
| On-chain metrics | Medium-high | Phase Crypto-F, after baseline |
| Perpetual futures funding/OI | High practical value, medium academic maturity | Phase Crypto-E RiskGuard first |
| News/social/LLM sentiment | Medium, noisy | Shadow overlay only |
| Deep learning price prediction | Medium | Only after XGB/LGB benchmark |
| RL trading | Experimental/high overfit risk | Later as sizing/control, not initial alpha |
| Market making/arbitrage | Engineering-heavy, venue-dependent | Later, Hummingbot/Nautilus track |

Numeric evidence tags used in this document:

- `[paper-reported]`: reported by a paper or external study, not reproduced
  locally.
- `[exchange-dashboard]`: reported by an exchange/dashboard/vendor, not
  reproduced locally.
- `[validated-on-local]`: reproduced on this project's data and code.

Production rule:

```text
Any number used for sizing, risk limits, or promotion must be
[validated-on-local].
```

## Paper Cluster 1: Systematic Reviews

### Quantitative Alpha in Crypto Markets, 2025

Reference:

- William Mann, "Quantitative Alpha in Crypto Markets: A Systematic Review of
  Factor Models, Arbitrage Strategies, and Machine Learning Applications",
  SSRN, 2025.
- Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5225612

Key points:

- Reviews factor models, arbitrage, and machine learning applications.
- Highlights size, momentum, and liquidity as statistically meaningful crypto
  factors.
- Notes that non-linear ML architectures can capture patterns missed by
  classical models, but implementation quality remains difficult.

Implication for this project:

- Start with factor + supervised ML, not raw RL.
- Build modular feature/backtest/execution code first.
- Treat factor families as hypotheses that must pass the same gate discipline
  as A-share factors.

### Cryptocurrency Trading: A Systematic Mapping Study, 2024

Reference:

- "Cryptocurrency trading: A systematic mapping study",
  International Journal of Information Management Data Insights, 2024.
- Link: https://www.sciencedirect.com/science/article/pii/S2667096824000296

Key points:

- Crypto trading research uses many data types: OHLCV, technical indicators,
  social media, sentiment, blockchain/on-chain data, order books.
- Neural networks and deep learning are common, but research designs vary
  widely.

Implication:

- Do not trust any single paper's headline PnL.
- Our implementation must compare against simple baselines and include fees.
- Data contract and leakage control matter more than model glamour.

### Bitcoin Price Forecasting ML Systematic Review, 2025

Reference:

- "Forecasting the Bitcoin price using the various Machine Learning: A
  systematic review in data-driven marketing", 2025.
- Link: https://www.sciencedirect.com/science/article/pii/S2772941925000274

Key points:

- ML/DL approaches are widely used for BTC forecasting.
- Many studies mix technical indicators and sentiment.
- Data quality, local optima, and weak generalization remain persistent issues.

Implication:

- Require walk-forward validation and paper trading.
- BTC-only success does not automatically generalize to altcoins.

## Paper Cluster 2: Crypto Factor Models

### Common Risk Factors in Cryptocurrency

Reference:

- Liu, Tsyvinski, Wu, "Common Risk Factors in Cryptocurrency", NBER.
- Link: https://www.nber.org/papers/w25882

Key points:

- Crypto can be studied with factor-model logic.
- Market, size, and momentum-style factors have explanatory power.

Implication:

- Build crypto cross-sectional factors:
  - market beta to BTC
  - size/liquidity
  - momentum/reversal
  - volatility
  - volume/liquidity shock

### Momentum/Reversal/Liquidity

Reference:

- "Up or down? Short-term reversal, momentum, and liquidity effects in
  cryptocurrency markets".
- Link: https://www.sciencedirect.com/science/article/pii/S1057521921002349

Key points:

- Momentum and reversal differ by liquidity and size.
- Large/liquid coins may trend; smaller/illiquid coins often reverse.

Implication:

- Do not use one universal crypto momentum signal.
- Model interactions:
  - `momentum_24h * liquidity_rank`
  - `reversal_4h * volatility_rank`
  - `volume_shock * size_rank`

### Size and Volume Effects

Reference:

- "Impact of Size and Volume on Cryptocurrency Momentum and Reversal".
- Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4378429

Implication:

- Universe construction is alpha-critical.
- Avoid thin alts until slippage model is realistic.
- Use dollar volume and exchange coverage as eligibility gates.

## Paper Cluster 3: On-chain Factors

### On-Chain Factors and Cryptocurrency Asset Pricing

Reference:

- "On-Chain Factors and Cryptocurrency Asset Pricing".
- Link: https://papers.ssrn.com/sol3/Delivery.cfm/6670521.pdf?abstractid=6670521&mirid=1

Key factor families:

- Network activity
- Scale-adjusted activity
- Valuation ratios
- Distribution/holder concentration
- Transaction fees and congestion
- Exchange flow/inflow/outflow if available

Implication:

- Add on-chain only after spot baseline:
  - BTC/ETH first.
  - Then token-specific metrics where reliable.
- On-chain metrics are slower-moving; they are better for 1d/3d/7d horizons
  than 1h prediction.

Implementation caution:

- Free on-chain APIs can be rate-limited or incomplete.
- Vendor data can be expensive.
- On-chain timestamps need UTC alignment and availability-time tracking.

## Paper Cluster 4: Perpetual Futures and Microstructure

### Perpetual Futures and Market Quality

Reference:

- Cornell/SC Johnson summary of "Perpetual Futures Contracts and
  Cryptocurrency Market Microstructure", 2025.
- Link: https://business.cornell.edu/article/2025/02/perpetual-futures-contracts-and-cryptocurrency/

Key points:

- Perpetual futures dominate crypto derivatives volume.
- Funding mechanisms and open interest affect market quality and price
  discovery.

Implication:

- Perp data is not optional if we want serious crypto quant.
- Funding/open interest should first enter RiskGuard, then alpha.

### High-frequency Bitcoin Futures Microstructure, 2025

Reference:

- "High-frequency dynamics of Bitcoin futures: An examination of market
  microstructure", Borsa Istanbul Review, 2025.
- Link: https://www.econstor.eu/handle/10419/340643

Key points:

- Bitcoin perpetual futures have stylized facts similar to but distinct from
  traditional futures.
- Volume/volatility dynamics matter.

Implication:

- Intraday/perp strategies need separate treatment from daily spot alpha.
- Use 1h/4h first; do not jump straight to tick/order-book ML.

### Deep Learning for VWAP Execution in Crypto Markets, 2025

Reference:

- Remi Genet, "Deep Learning for VWAP Execution in Crypto Markets: Beyond the
  Volume Curve", SSRN, 2025.
- Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5150912

Implication:

- Execution alpha can matter as much as predictive alpha.
- Later phases should separate:
  - alpha model
  - portfolio sizing
  - execution schedule

For this project:

- Phase A/B should use conservative slippage.
- Execution ML only after paper trading proves signal value.

## Paper Cluster 5: Deep Learning and RL

### Multi-level Deep Q Networks for Bitcoin Trading, 2024

Reference:

- Scientific Reports, "Multi-level deep Q-networks for Bitcoin trading
  strategies", 2024.
- Link: https://www.nature.com/articles/s41598-024-51408-w

Key points:

- Uses historical BTC data and Twitter sentiment.
- DRL can learn trading policies, but robustness and market regime shift remain
  difficult.

Implication:

- Sentiment + RL is possible, but should be shadow-only.
- Need benchmark against simple momentum/reversal and XGB.

### Ensemble DRL for Crypto Trading

References:

- "Automated cryptocurrency trading approach using ensemble deep reinforcement
  learning: Learn to understand candlesticks".
  Link: https://www.sciencedirect.com/science/article/abs/pii/S0957417423018754
- "An Ensemble Method of Deep Reinforcement Learning for Automated
  Cryptocurrency Trading".
  Link: https://arxiv.org/abs/2309.00626

Implication:

- Ensemble RL is a real research direction.
- But for this project, it should be Phase Crypto-H:
  - after data
  - after supervised baseline
  - after paper OMS
  - after risk guard

### FinRL Contest and Ensemble Methods

Reference:

- "Revisiting Ensemble Methods for Stock Trading and Crypto Trading Tasks at
  ACM ICAIF FinRL Contest 2023-2024", arXiv 2025.
- Link: https://arxiv.org/abs/2501.10709

Implication:

- Ensemble methods matter.
- Practical lesson: ensemble should be used to stabilize model disagreement,
  not to maximize backtest curve fitting.

### FinRL-DeepSeek and LLM-infused RL

References:

- FinRL-DeepSeek: https://arxiv.org/abs/2502.07393
- SecureFinAI LLM + RL high-frequency crypto trading:
  https://papers.ssrn.com/sol3/Delivery.cfm/5714622.pdf?abstractid=5714622&mirid=1

Implication:

- LLM signals + RL is a frontier direction.
- Not a Phase A implementation item.
- In this project, LLM crypto events should first be structured facts and
  shadow overlays, mirroring the discipline already built for A-share events.

## Paper Cluster 6: Blockchain ML Beyond Trading

Reference:

- "Machine Learning on Blockchain Data: A Systematic Mapping Study", arXiv 2024.
- Link: https://arxiv.org/abs/2403.17081

Implication:

- Blockchain ML covers anomaly detection, scam detection, network analysis,
  transaction classification, and risk.
- Useful later for:
  - hack/exploit risk
  - token scam filters
  - abnormal flow detection
  - wallet concentration risk

Not first priority for directional spot alpha.

## Paper Cluster 7: Modern Frontier Backlog

These methods are worth tracking, but they should be shadow/research items, not
Phase A production dependencies.

### RD-Agent

Reference:

- Microsoft RD-Agent GitHub: https://github.com/microsoft/RD-Agent
- Microsoft Research article:
  https://www.microsoft.com/en-us/research/articles/rd-agent-an-open-source-solution-for-smarter-rd/

Use:

- factor hypothesis generation
- automated experiment loops
- research assistant for crypto factor zoo expansion

Project constraint:

- Must run behind local validation and promotion gates.
- No direct production promotion based on generated hypotheses alone.

### Kronos

Reference:

- "Kronos: A Foundation Model for the Language of Financial Markets".
- arXiv: https://arxiv.org/abs/2508.02739
- Model reference: https://tsfm.ai/models/NeoQuasar/Kronos-base

Use:

- shadow predictor or embedding generator for BTC/ETH/SOL OHLCV.
- compare to XGB/LGB after fees and slippage.

Project constraint:

- Do not replace tabular supervised baseline.

### CryptoTrade / LLM Trading Benchmarks

Reference:

- "CryptoTrade: A Reflective LLM-based Agent to Guide Zero-shot
  Cryptocurrency Trading", EMNLP 2024.
- arXiv: https://arxiv.org/abs/2407.09546
- ACL anthology: https://aclanthology.org/2024.emnlp-main.63/

Use:

- benchmark for event reasoning and LLM market narratives.

Project constraint:

- LLM produces structured facts, risk flags, and shadow overlays.
- No autonomous LLM order execution.

### GraphSAGE / On-chain GNN

Reference:

- Example blockchain GNN risk-classification work:
  https://link.springer.com/article/10.1186/s42400-023-00194-5

Use:

- wallet clustering
- scam/phishing risk
- whale-flow embeddings

Project constraint:

- Stronger evidence for risk classification than direct return prediction.
- Treat as feature discovery/risk overlay, not first-line alpha.

## Open-source Engineering Review

Engineering decision rule:

```text
Use libraries for the main production pipeline.
Use frameworks as references or later execution prototypes.
```

Reason:

- The current project is cron/parquet/jsonl/data-health driven.
- Full trading frameworks impose process models that can conflict with this.
- Libraries preserve composability.

### Freqtrade and FreqAI

References:

- FreqAI docs: https://docs.freqtrade.io/en/2025.1/freqai/
- Freqtrade GitHub: https://github.com/freqtrade/freqtrade

What it is good for:

- ML workflow reference for crypto.
- Adaptive retraining.
- Large feature sets.
- Backtest/live separation.
- Outlier cleaning and normalization.
- Dry/live deployment mechanics.

What to borrow:

- Feature naming convention.
- Model lifecycle:
  - train
  - cache model
  - live inference
  - periodic retraining
  - crash recovery
- Outlier detection around train/inference.

What not to copy blindly:

- Strategy-level feature generation can become messy.
- It is bot-oriented; this project is research/pipeline-oriented.
- Keep our own Alpha Factory gate and data-health discipline.

### NautilusTrader

References:

- NautilusTrader docs: https://nautilustrader.io/docs/
- Binance integration: https://nautilustrader.io/docs/latest/integrations/binance/
- Bybit integration: https://nautilustrader.io/docs/latest/integrations/bybit
- Tardis/data integration: https://nautilustrader.io/docs/latest/integrations/tardis/

What it is good for:

- Serious backtest/live parity.
- Event-driven trading.
- Exchange adapters.
- Spot and derivatives.
- Order and account semantics.
- Live data and execution clients.
- Demo/testnet support.

What to borrow:

- Later production execution model.
- Venue/instrument identity discipline:
  - spot BTCUSDT and perp BTCUSDT are not the same instrument.
- Testnet/demo workflow.

Recommendation:

- Do not integrate Nautilus in Phase A.
- Build internal research pipeline first.
- Prototype Nautilus in Phase Crypto-G after paper backtest is meaningful.

### Hummingbot

References:

- Official docs: https://hummingbot.org/docs/
- Hummingbot home: https://hummingbot.org/
- Spot-perpetual arbitrage:
  https://hummingbot.org/strategies/v1-strategies/spot-perpetual-arbitrage/

What it is good for:

- Market making.
- Cross-exchange market making.
- Spot/perp arbitrage.
- Funding rate arbitrage.
- Connector ecosystem.

What to borrow:

- Later arbitrage/market-making experiments.
- Funding/carry strategy reference.
- Connector ideas.

What not to use for first phase:

- Directional supervised alpha.
- Cross-sectional ML ranking.

### FinRL

References:

- FinRL home: https://www.finrl.ai/
- FinRL GitHub: https://github.com/AI4Finance-Foundation/FinRL

What it is good for:

- RL environment ideas.
- PPO/SAC/DDPG/A2C templates.
- Portfolio management examples.
- Contest baselines.

What to borrow:

- Environment/reward shaping prototypes.
- RL agent comparison.
- Ensemble RL evaluation ideas.

What not to do:

- Do not adopt FinRL as the production execution engine.
- Do not trust RL backtests without strict fees, slippage, and walk-forward.

### Qlib

Current project already uses Qlib-style A-share research. For crypto:

Useful:

- Dataset-style feature/label separation.
- Cross-sectional rank IC.
- Factor cache discipline.

Limitations:

- 24/7 market.
- Exchange-specific bars.
- Perpetual funding/OI.
- Continuous execution.

Recommendation:

- Reuse Qlib concepts, not necessarily Qlib storage.
- If converting to Qlib-like format, keep it as a research cache, not the only
  source of truth.

### vectorbt / backtrader

Useful for:

- Quick signal sweeps.
- Vectorized portfolio experiments.
- Simple fee/slippage comparison.

Limitations:

- Less exchange-realistic than NautilusTrader.
- Must be careful with 24/7 timestamp alignment.

Recommendation:

- Use for early Phase Crypto-B only if implementation is faster than internal
  vectorized backtest.

## Data Source Review

### Exchange OHLCV

Priority:

1. CCXT REST for OHLCV.
2. WebSocket later for live/paper.
3. Multiple exchanges only after one exchange is clean.

Initial exchanges:

- Binance if accessible.
- OKX/Bybit as fallback.

Engineering requirements:

- UTC timestamps.
- Closed candles only.
- Gap detection.
- Duplicate key:
  `(exchange, symbol, timeframe, timestamp_utc)`.
- Store `ingested_at` for PIT discipline.

### Derivatives Data

Priority fields:

- Funding rate.
- Next funding time.
- Open interest.
- Mark price.
- Index price.
- Long/short ratio if reliable.
- Liquidation data if available.

Use first as risk/regime:

- High funding + high OI + weakening momentum = crowded long risk.
- Negative funding + high OI + breakout = short squeeze risk.
- OI collapse + vol spike = deleveraging regime.

### On-chain Data

Potential sources:

- Public APIs.
- Exchange flow providers.
- Blockchain explorers.
- Vendor APIs if later budget allows.

Start with:

- BTC/ETH network activity.
- Transaction count.
- Active addresses.
- Fees.
- Exchange inflow/outflow if reliable.

Use horizon:

- 1d/3d/7d, not 1h.

### News/Event Data

Sources:

- GDELT/Google RSS.
- Exchange announcements.
- CoinDesk/Cointelegraph style crypto media.
- SEC/CFTC/regulatory feeds.
- Protocol governance announcements.

LLM extraction schema:

```text
asset
event_type
direction
confidence
source_quality
event_time
publish_time
available_time
signal_time
execution_time
magnitude
```

Event types:

- listing
- delisting
- hack/exploit
- ETF approval/rejection
- regulatory enforcement
- stablecoin depeg
- protocol upgrade
- token unlock
- exchange outage
- founder/legal event
- bridge exploit
- chain halt

Use:

- Shadow overlay only at first.

## Model Design Review

### Baseline Factor Model

Features:

- Return/momentum: 1h, 4h, 24h, 7d.
- Reversal: last 1h/4h return.
- Volatility: realized vol, high-low range.
- Volume: quote-volume z-score.
- Liquidity: dollar volume, Amihud.
- Relative strength vs BTC/ETH.
- BTC beta.
- Weekend/time-of-day seasonality.

Labels:

- 4h forward return.
- 24h forward return.
- 24h drawdown.
- Top-quintile classification.

### Supervised ML

First models:

- XGBoost.
- LightGBM.
- Ridge/ElasticNet baseline.

Why:

- Interpretable enough.
- Runs on Mac Studio.
- Stable with tabular factors.
- Matches current project style.

Do not start with:

- Transformer.
- LSTM.
- RL.
- Order-book neural networks.

### Ensemble

Use after individual models pass:

- rank mean
- robust z-score mean
- disagreement penalty
- regime-conditioned weights

Borrow from current project's ensemble discussion.

### RL

Use cases:

- position sizing
- risk reduction
- execution timing
- allocation among model signals

Do not use first for:

- raw price prediction
- direct buy/sell policy from OHLCV only

Why:

- Trading RL often overfits.
- Market observations are mostly not affected by the agent unless trading size
  is huge.
- Reward shaping can hide costs and drawdown.

## Project Integration Plan

Target architecture:

```text
short term: core protocols + greenfield crypto package
medium term: adapters/facades
long term: core/ + ashare/ + crypto/ namespaces
```

Physical migration of A-share files should be delayed until regression
snapshots and production cron stability pass.

Crypto time/settlement contract:

- Crypto is 24/7 and should use UTC timestamps.
- Crypto spot is T+0/immediate settlement, not A-share T+1.
- Perpetual futures are continuous and funding-cycle driven.
- The initial research/live-paper path should use closed candles only:
  - 1h
  - 4h
  - 1d
- Paper OMS can model either immediate fill or next-bar-open fill, but it must
  not reuse the A-share next-trading-day-open reconcile logic.
- Stale/incomplete bars should block signals.
- Tick/order-book/HFT data is explicitly out of scope for Phase A.

### New Phase Crypto-0: Evidence and Data Contract

Deliverables:

- `plans/crypto-data-contract.md`
- `data/storage/crypto/README.md`
- universe list and exchange choice
- asset-implicit audit checklist from the CC review
- hard-gotcha checklist
- evidence tags on numeric assumptions
- target architecture decision

Acceptance:

- UTC schema approved.
- symbol/exchange identity rules written.
- no trading code yet.
- every numeric sizing/risk assumption is tagged

### Hard Gotchas To Enforce From Phase Crypto-0

- IVOL sign may differ from A-share; force local sign validation.
- MAX / lottery-style behavior may differ from A-share.
- Traditional value/book-to-price does not transfer cleanly to crypto.
- BTC/ETH cointegration is not a safe default after structural breaks.
- Survivorship bias is severe; keep delisted/dead coins where possible.
- Alpha decay is likely faster than A-share; use shorter monitoring windows.
- All imported A-share factors require forced sign checks.

### New Phase Crypto-A: Data Foundation

Files:

- `data/collectors/crypto_market.py`
- `data/collectors/crypto_derivatives.py`
- `scripts/crypto_update_market_data.py`
- `scripts/crypto_data_health.py`

Acceptance:

- 1h/4h/1d OHLCV for 5 symbols.
- funding/OI history for BTC/ETH/SOL where available.
- idempotent parquet writes.
- health file.
- gap report.

### New Phase Crypto-B: Feature and Baseline

Files:

- `models/crypto_feature_pipeline.py`
- `scripts/crypto_build_features.py`
- `scripts/crypto_backtest_baseline.py`

Acceptance:

- feature cache.
- baseline momentum/reversal backtest.
- fees and slippage included.
- RankIC/spread/PnL report.

### New Phase Crypto-C: Supervised ML

Files:

- `models/crypto_model.py`
- `scripts/crypto_train_model.py`
- `scripts/crypto_predict.py`
- `scripts/crypto_factor_gate.py`

Acceptance:

- rolling split.
- XGB/LGB model.
- latest prediction artifact.
- ICIR and spread gates.

### New Phase Crypto-D: Paper Trading

Files:

- `paper/crypto_oms.py`
- `scripts/run_crypto_paper_trading.py`
- `scripts/crypto_daily_report.py`

Acceptance:

- no live keys.
- fee/slippage.
- position cap.
- stale data block.
- 30 calendar days paper before promotion.

### New Phase Crypto-E: Perp RiskGuard

Files:

- `data/collectors/crypto_derivatives.py`
- `risk/crypto_risk_guard.py`
- `scripts/crypto_update_derivatives.py`

Acceptance:

- funding/OI features.
- crowding risk flags.
- exposure reduction in paper mode.

### New Phase Crypto-F: Events/On-chain

Files:

- `data/collectors/crypto_events.py`
- `factors/crypto_event_store.py` or extend `EventStore` with `asset_class`
- `scripts/crypto_build_event_factors.py`

Acceptance:

- exchange announcement collector.
- LLM fact extraction.
- shadow overlay only.

### New Phase Crypto-G: Nautilus Prototype

Files:

- `experiments/nautilus_crypto_backtest/`
- `plans/crypto-nautilus-integration-notes.md`

Acceptance:

- BTC/ETH spot backtest runs.
- internal backtest vs Nautilus comparison.
- decision whether to integrate for live/testnet.

### New Phase Crypto-H: RL / Advanced Allocation

Files:

- `experiments/crypto_rl_env.py`
- `scripts/crypto_rl_train.py`

Acceptance:

- only after supervised paper has 30+ days.
- RL beats supervised baseline after fees in walk-forward.
- drawdown not worse.

## Current Project Gaps

Existing crypto code:

- `data/collectors/crypto.py`
- `scheduler/jobs.py`
- `signals/index_predictor.py`
- `config/watchlist.py`
- `tests/test_crypto_collector.py`

Gaps:

- no historical crypto data store
- no 24/7 UTC schedule
- no crypto feature cache
- no crypto labels
- no crypto backtest
- no fee/slippage model for crypto
- no exchange-specific universe
- no derivatives data
- no crypto paper OMS
- no crypto risk guard
- no promotion gate
- no event/on-chain crypto factor path

## What To Avoid

Avoid:

- running live leverage early
- treating BTC/ETH simple change_pct as a real quant signal
- using A-share daily Qlib assumptions
- using A-share `CandidateSanitizer`
- skipping fees
- skipping stale/gap checks
- treating LLM sentiment as alpha before shadow validation
- using RL as a shortcut around bad data
- mixing spot and perpetual symbols as if they were the same instrument

DeFi carve-out:

- DeFi yield is out of scope for the crypto quant pipeline.
- It can be tracked as a separate capital-management benchmark.
- It should not affect model promotion, paper OMS, or signal validation.

## Recommended Immediate Implementation Checklist

1. Create `data/storage/crypto/` layout.
2. Write data contract.
3. Replace or extend `CryptoCollector` into a production collector with:
   - exchange
   - symbol
   - timeframe
   - closed candle only
   - gap detection
   - health write
4. Add `scripts/crypto_update_market_data.py`.
5. Add `scripts/crypto_build_features.py`.
6. Add baseline factor backtest.
7. Only then add model training.

## Phase Placement

This should be a new branch of the roadmap:

- Phase Crypto-0/A/B can run now in parallel with A-share Phase 4 stabilization.
- Phase Crypto-C should wait until crypto data has at least 90 days historical
  coverage or enough backfilled bars.
- Phase Crypto-D should start only after baseline backtest exists.
- Phase Crypto-E/F should not block the baseline spot model.
- Phase Crypto-H RL is explicitly late-stage.

## Final Recommendation

For this project and one Mac Studio, the highest expected value crypto path is:

```text
spot OHLCV + funding/OI data → feature cache → fee-aware baseline
→ XGB/LGB ranking → paper OMS → derivatives RiskGuard
→ event/on-chain/frontier-model shadow overlays
→ NautilusTrader execution prototype + namespace migration
→ RL allocation only if justified
```

This matches the project's current strengths: disciplined factor research,
promotion gates, paper trading, cron/data health, and gradual productionization.
It avoids the main crypto quant traps: overfitting, fee blindness, unreliable
social signals, leverage too early, and confusing backtest alpha with executable
edge.
