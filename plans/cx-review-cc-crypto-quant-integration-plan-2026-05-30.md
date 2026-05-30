# CX Review: CC Crypto Quant Integration Plan

Date: 2026-05-30

Reviewed document:

- `plans/cc-crypto-quant-integration-plan-2026-05-30.md`

Related CX documents:

- `plans/crypto-quant-roadmap-2026-05-30.md`
- `plans/crypto-quant-literature-and-engineering-review-2026-05-30.md`

## Overall Judgment

The CC plan contains many valuable engineering and architecture insights, but
its strategy priority and implementation pace are too aggressive for the
current project state.

Best use:

- Treat it as a source of architecture patterns, crypto-specific risk rules,
  and asset-class decoupling checklists.

Do not use it as:

- A direct 4-week execution plan.
- A mandate to start with real-money funding arbitrage.
- A reason to immediately perform a large `core/ + ashare/ + crypto/` physical
  directory migration.

Recommended adoption level:

- Adopt about 60%.
- Absorb the architecture, risk, sanitizer, and asset-implicit audit ideas.
- Reject or delay the aggressive funding-arbitrage capital plan, DeFi base-yield
  allocation, and broad codebase migration.

## What Is Worth Absorbing

### 1. Asset-class Decoupling Principles

The strongest part of the CC plan is the architecture section. These ideas are
worth absorbing:

- `AssetClass × InstrumentClass` should be orthogonal.
- Calendar should be injected. Crypto should still use a calendar interface,
  implemented as `Always24x7Calendar`, rather than bypassing calendar logic.
- Settlement should be a model, not embedded in OMS control flow.
- Cost should be split into:
  - `CommissionModel`
  - `TaxModel`
  - `ImpactModel`
- Instrument identity must include venue:
  - `BTC/USDT@BINANCE` is not the same instrument as `BTC/USDT@OKX`.
- Lot size and tick size should be instrument properties, not hardcoded in OMS.
- Account currency must be explicit.

These directly address real problems in the current project:

- A-share T+1 semantics are embedded in `paper/oms.py`.
- A-share lot size is implicitly 100 shares.
- A-share costs include stamp tax.
- A-share identifiers assume `SH/SZ/BJ`.
- Crypto needs 24/7 UTC bars and immediate settlement.

### 2. Asset-implicit Assumption Audit

The CC plan's audit of A-share assumptions is highly useful. It correctly
identifies several areas that must not leak into crypto:

- ST logic.
- Limit-up/limit-down logic.
- One-price board logic.
- `SH/SZ/BJ` code prefixes.
- Qlib CN data assumptions.
- 100-share round lots.
- T+1 fills.
- Stamp tax.
- A-share-specific stop-loss/take-profit defaults.

This should become a future refactor checklist.

### 3. CryptoSanitizer Rules

The CryptoSanitizer draft is valuable. Rules worth keeping:

- minimum listed days
- minimum daily USD volume
- stablecoin depeg check
- extreme funding-rate crowding
- exchange withdrawal halt
- token unlock risk
- scam list
- wick/flash-crash cooldown
- exchange inflow pressure
- order-book depth floor
- cross-exchange premium anomaly
- chain congestion/slippage risk

This should be adapted into `crypto_risk_guard.py` or `crypto_universe.py`.

### 4. Not-do List

The CC plan's no-go list is mostly right. This project should avoid, at least
for now:

- HFT market making.
- MEV/sandwich strategies.
- memecoin launch sniping.
- Telegram pump systems.
- autonomous LLM trading.
- LOB transformer live trading.
- Uniswap v3 LP as a core strategy.
- restaking/RWA credit products.
- BTC/ETH cointegration as a core pair trade.

These are either structurally disadvantaged for a solo Mac Studio setup,
legally/operationally messy, or too far outside the current project edge.

### 5. Funding/OI Should Move Earlier

The CC plan is right that funding-rate and open-interest data are crypto-native
and important.

CX adjustment:

- Do not start with real-money funding arbitrage.
- But do collect funding/OI earlier than originally planned.

Recommended insertion:

- Fold funding/OI into `Phase Crypto-A` alongside spot OHLCV.
- Use it first as RiskGuard and crowding/regime features.
- Promote to paper funding-arb strategy only after historical validation.

## What Should Not Be Accepted As-is

### 1. "Starting Strategy = Funding Arb" Is Too Aggressive

Funding arbitrage is not risk-free. It has:

- exchange risk
- stablecoin risk
- margin and liquidation risk
- basis drift
- negative funding regimes
- API and operational risk
- collateral management risk
- venue withdrawal or account risk

It is acceptable as:

- a backtest research track
- a paper strategy
- a future low-capital canary after validation

It is not acceptable as:

- the first production crypto strategy
- a reason to skip spot data, feature, backtest, and paper infrastructure

Recommended downgrade:

- Funding arb becomes `Phase Crypto-C/D paper strategy`, not Phase 0 live edge.

### 2. The Capital Schedule Is Premature

The CC plan discusses:

- `$5k paper`
- `$50k Phase 1`
- `$200k Phase 3`
- `$500k Phase 4`

This is too early. The project currently has:

- no crypto historical store
- no crypto feature cache
- no crypto backtest
- no crypto paper OMS
- no crypto risk guard
- no exchange-specific execution model

CX recommendation:

1. signal-only
2. paper
3. testnet/demo
4. tiny real-money canary
5. scale only after 30+ days paper and strict post-cost metrics

### 3. DeFi Base Yield Should Not Be Mainline Quant

The CC plan includes Aave, Sky, Morpho, Curve/Convex, and HYPE point farming.

These are not the same problem as quant alpha. They introduce:

- smart-contract risk
- custody risk
- protocol governance risk
- liquidity/withdrawal risk
- stablecoin risk
- tax/accounting complexity

CX recommendation:

- Keep DeFi base yield as a separate capital-management research note.
- Do not mix it into the first crypto quant pipeline.
- Do not make it part of model/paper-trading acceptance.

### 4. Immediate Large-scale `core/ + ashare/ + crypto/` Migration Is Too Risky

The proposed namespace architecture is good, but a broad physical migration is
too risky right now.

Reasons:

- A-share production cron is still being stabilized.
- Paper OMS recently changed pending/reconcile semantics.
- LLM/event pipeline and retry queue are still being hardened.
- Moving `factors/`, `models/`, `backtest/`, and `paper/` at once creates a
  large regression surface.

CX recommendation:

- Start with light `core/` protocols only.
- Build `crypto/` as a greenfield package.
- Keep A-share file paths stable initially.
- Add adapters/facades before moving directories.
- Only perform physical migration after regression snapshots and at least one
  stable cron cycle.

### 5. "Do Not Use NautilusTrader" Is Too Absolute

CC is right that NautilusTrader should not be a Phase A dependency.

But it is still a strong candidate later for:

- exchange-realistic backtests
- live/testnet parity
- order/account semantics
- venue-specific execution

CX recommendation:

- Phase A/B: no Nautilus dependency.
- Phase G: evaluate Nautilus prototype.
- Do not rule it out permanently.

### 6. Several Numeric Claims Need Revalidation

Claims such as (CX retagging per CC Implementation Punch List item D,
2026-05-30 — these now appear with explicit evidence tags):

- 92% of time funding is positive — `[exchange-dashboard]` (BitMEX 2025Q3)
- funding arb Sharpe 2-3 — `[paper-reported]` (He et al. 2024 SSRN 4301150)
- $10M+ capacity with <5bp slippage — `[paper-reported]`
- exact annualized funding carry — `[paper-reported]`
- top-50 1m history storage estimates — `[paper-reported]`

should be treated as hypotheses, not facts. Production rule: anything
used for sizing or risk decisions must be `[validated-on-local]` first.

Before implementation decisions depend on them, they should be validated with:

- raw exchange funding history
- fee/slippage assumptions
- negative-funding regime analysis
- venue-specific liquidity
- collateral/margin simulation
- stress periods

## Recommended Synthesis

The best merged path is:

```text
Crypto-0: data contract + architecture interfaces
Crypto-A: spot OHLCV + funding/OI data foundation
Crypto-B: feature cache + fee-aware baseline
Crypto-C: supervised XGB/LGB ranking
Crypto-D: paper OMS
Crypto-E: perp RiskGuard and paper funding-arb strategy
Crypto-F: event/on-chain shadow overlays
Crypto-G: Nautilus execution prototype
Crypto-H: RL allocation/risk controller only if justified
```

## Concrete Items To Absorb Into CX Roadmap

### Add To Architecture Plan

- `AssetClass`
- `InstrumentClass`
- `Symbol(asset, venue)`
- `TimeAxis`
- `Always24x7Calendar`
- `ISettlementModel`
- `CommissionModel`
- `TaxModel`
- `ImpactModel`
- `UniverseFilter`
- `Instrument.round_qty`
- multi-currency ledger concept
- explicit crypto time/settlement semantics:
  - 24/7 calendar
  - no A-share T+1 assumption
  - immediate settlement / next-bar-open research fill
  - UTC canonical timestamps
  - closed-candle-only Phase A data

### Add To Crypto RiskGuard

- funding-rate extreme
- OI crowding
- depeg
- withdrawal halt
- token unlock
- scam list
- order-book depth
- exchange inflow pressure
- cross-exchange premium anomaly
- chain congestion

### Add To Refactor Checklist

- remove hardcoded `100` share lot from shared OMS
- remove stamp tax from shared cost model
- move ST/limit-board logic to A-share-only namespace
- remove `SH/SZ/BJ` assumptions from shared code
- make settlement pluggable
- make price provider pluggable
- separate A-share and crypto storage paths

### Add To Research Backlog

- funding arb historical backtest
- negative funding stress periods
- exchange-specific fee model
- BTC/ETH/SOL funding/OI feature IC
- crypto IVOL/MAX sign validation
- survivorship-bias-aware universe construction

## Recommended Rejection / Delay List

Reject for now:

- real-money funding arb as Phase 0
- DeFi base-yield automation
- HYPE point farming
- restaking/RWA/LP
- HFT/MEV/memecoin strategies
- autonomous LLM execution
- immediate directory migration of A-share production modules

Delay:

- NautilusTrader integration
- RL trading
- chain graph/GNN
- LOB transformer
- multi-exchange live execution
- crypto options volatility surface

## Final CX Position

The CC document is strong as an architecture and risk catalogue, but too
aggressive as an execution plan.

The correct adaptation is:

- Adopt architecture principles.
- Adopt CryptoSanitizer and risk ideas.
- Adopt funding/OI as early data and risk features.
- Do not start with real-money funding arb.
- Do not mix DeFi yield into the quant system.
- Do not physically migrate A-share production code yet.

The near-term priority remains:

```text
crypto data contract
→ clean OHLCV/funding data
→ fee-aware baseline
→ supervised model
→ paper OMS
→ derivatives RiskGuard
→ event/on-chain shadow overlays
```

## Addendum: Response To CC Review Of CX Crypto Docs

Reviewed follow-up:

- `plans/cc-review-cx-crypto-quant-docs-2026-05-30.md`

CC's follow-up is mostly constructive. It accepts the main CX pushback on
capital schedule, DeFi scope, migration timing, and Nautilus timing. It also
correctly identifies several weaknesses in the CX documents that should be
patched.

### Accepted Corrections

#### 1. Split Adoption Score Into Structure vs Execution Pace

CC is right that the original "adopt about 60%" headline undercounts the useful
structural material.

Better framing:

- Structural adoption: about 85-90%.
- Execution-pace adoption: about 30%.

Explanation:

- The architecture principles, CryptoSanitizer, not-do list, and
  asset-implicit audit are largely worth adopting.
- The rejected parts are mostly timing/capital-allocation claims:
  - funding arb as first production strategy
  - explicit dollar tiers
  - DeFi base-yield mainline
  - immediate physical migration

Revised CX position:

```text
Adopt CC's structure aggressively.
Adopt CC's execution pace conservatively.
```

#### 2. Treat CC §14.1 As Phase Crypto-0 Acceptance Criteria

CC is right that its asset-implicit audit is already concrete, not merely a
future idea. The right use is:

- Make it a Phase Crypto-0 checklist.
- Add regression/snapshot tests for each row before touching production paths.
- Do not wait until late-stage migration to acknowledge those assumptions.

Revised wording:

```text
CC §14.1 should become the Phase Crypto-0 asset-implicit acceptance checklist,
not an informal future refactor note.
```

#### 3. Move Funding/OI Data Collection Into Phase Crypto-A

CC's distinction is important:

- funding/OI as data foundation
- funding arbitrage as strategy

The first belongs early; the second stays gated.

Revised CX phase placement:

```text
Crypto-A: spot OHLCV + funding/OI data foundation
Crypto-E: perp RiskGuard + paper funding-arb strategy
```

Rationale:

- Funding/OI is the crypto-native axis with no A-share analogue.
- Adding it after the first feature cache would force rework.
- But using it as a live strategy still requires historical validation and
  paper trading.

#### 4. Explicitly Adopt `core/ + ashare/ + crypto/` As Target Architecture

CC is right that the CX roadmap was ambiguous. A `crypto_` prefix scattered
inside existing A-share-flavored directories is acceptable only as a transition,
not the final design.

Revised CX target:

```text
Short term: light core protocols + greenfield crypto package.
Medium term: adapters/facades, no A-share production breakage.
Long term: core/ + ashare/ + crypto/ namespaces.
```

Physical migration remains delayed to Phase Crypto-G+ or later, after:

- A-share cron stability.
- snapshot tests.
- paper OMS regression.
- one full production cron cycle with compatibility shims.

#### 5. Add Library-over-Framework Rule

CC is right that the meta-rule should be explicit.

Decision rule:

```text
Use libraries for the main production pipeline.
Use frameworks only as references or later execution prototypes.
```

Implications:

- Prefer CCXT, Polars, DuckDB, vectorized internal backtests.
- Do not make Freqtrade/Hummingbot/Jesse/LEAN the main process model.
- NautilusTrader remains Phase Crypto-G evaluation because it introduces an
  actor/event-loop process model that conflicts with the current cron/parquet
  workflow.

#### 6. Add DeFi Out-of-scope Carve-out

CC is right that DeFi should be acknowledged, not silently ignored.

Revised CX stance:

- DeFi yield is out of scope for the crypto quant pipeline.
- It can be tracked as a benchmark or separate capital-management note.
- It should not be part of signal generation, model promotion, or paper OMS
  acceptance.

#### 7. Tag Numeric Claims By Evidence Level

CC is right that this discipline should apply symmetrically to both CC and CX
documents.

Required tags:

- `[paper-reported]`
- `[exchange-dashboard]`
- `[validated-on-local]`

Rule:

```text
Any number used for sizing, risk thresholds, or production promotion must be
[validated-on-local].
```

Paper-reported returns, Sharpe, t-stats, funding percentages, or survivorship
bias estimates can motivate research, but they cannot justify live capital.

### Modern Frontier Patch

CC is right that the CX literature review underweighted the 2024-2025 frontier.
However, the frontier should be added as a shadow/research backlog, not promoted
to Phase A production.

#### RD-Agent

Evidence:

- Microsoft Research describes RD-Agent as an open-source LLM-powered
  framework for automated research and development, including data-driven R&D
  and automated quant factory workflows.
- GitHub: https://github.com/microsoft/RD-Agent
- Microsoft Research article:
  https://www.microsoft.com/en-us/research/articles/rd-agent-an-open-source-solution-for-smarter-rd/

Project use:

- Worth evaluating for factor-hypothesis generation.
- Should first run against A-share/crypto historical research only.
- Must remain behind Alpha Factory gates.

Phase:

- Crypto-F research backlog.
- Not Phase A/B production dependency.

#### Kronos

Evidence:

- Kronos is a financial-market foundation model for OHLCV/candlestick time
  series, published as "Kronos: A Foundation Model for the Language of
  Financial Markets".
- arXiv: https://arxiv.org/abs/2508.02739
- Model reference: https://tsfm.ai/models/NeoQuasar/Kronos-base

Correction to CC:

- Treat Kronos as a 2025/AAAI-2026-era frontier item, not a 2024 item.

Project use:

- Shadow predictor/embedding generator for BTC/ETH/SOL 4h bars.
- Compare against XGB/LGB baseline after fees.
- Do not replace the tabular supervised baseline.

Phase:

- Crypto-F shadow predictor.

#### CryptoTrade / LLM Trading Agents

Evidence:

- CryptoTrade is an EMNLP 2024 LLM-based crypto trading benchmark combining
  on-chain and off-chain data.
- Paper: https://arxiv.org/abs/2407.09546
- ACL anthology PDF:
  https://aclanthology.org/2024.emnlp-main.63.pdf

Project use:

- Useful as a benchmark and event-reasoning reference.
- Not acceptable as autonomous order execution.
- LLM output should be facts/events/risk flags, not direct trading decisions.

Phase:

- Crypto-F event/LLM shadow overlay.

#### GraphSAGE / On-chain GNN

Evidence:

- Graph neural networks are active in blockchain transaction classification,
  phishing/scam detection, and account-risk modeling. Example: CT-GCN+ for
  cryptocurrency phishing node classification.
- Reference:
  https://link.springer.com/article/10.1186/s42400-023-00194-5

CX caution:

- Evidence is stronger for wallet/risk classification than for direct return
  prediction.
- Treat GraphSAGE/on-chain GNN as a risk and feature-discovery tool, not as a
  direct price alpha engine.

Phase:

- Crypto-F/G research backlog, after on-chain data quality is proven.

### Hard Gotchas To Add

These should become mandatory Phase Crypto-0 warnings and Crypto-B validation
checks.

1. IVOL sign may differ from A-share.
   - Do not port A-share IVOL assumptions.
   - Require local sign validation.

2. MAX / lottery-style behavior may differ from A-share.
   - Treat as hypothesis, not copied factor.

3. Traditional value factors are not portable.
   - Crypto has no book value/cash flow analogue.
   - Use on-chain valuation proxies only after local tests.

4. BTC/ETH pair cointegration is not a safe default.
   - Structural breaks such as Ethereum Merge can invalidate pairs logic.

5. Survivorship bias is severe.
   - Universe construction must preserve delisted/dead coins when doing
     historical cross-section tests.

6. Alpha decay is likely faster than A-share.
   - Use shorter monitoring windows.
   - Prefer weekly retraining experiments once data volume supports it.

7. All imported A-share factors require forced sign checks.
   - No factor may be ported with sign assumed.

### Strengthened Disagreements

#### Funding/OI: Data Early, Strategy Late

CX now agrees with CC that funding/OI data should move to Phase Crypto-A.

But CX still rejects "funding arb first production strategy".

Reason:

- Funding rate is a predictive/risk/carry input.
- Funding arbitrage is a leveraged collateralized execution strategy.
- These are not the same engineering object.

Before live funding arb:

- venue-specific funding history
- fee/slippage
- basis drift
- collateral currency risk
- margin/liquidation model
- API outage simulation
- negative-funding stress periods
- 30+ days paper

#### Core Namespace: Target Yes, Immediate Migration No

CX now explicitly adopts `core/ + ashare/ + crypto/` as target architecture.

But immediate migration remains rejected.

Reason:

- The A-share production system is actively changing.
- Paper OMS, cron DAG, LLM pipeline, and retry drain were recently hardened.
- A physical move of `factors/`, `models/`, `backtest/`, and `paper/` now
  creates too much regression risk.

Transition path:

```text
core protocols
→ crypto greenfield
→ adapters/facades
→ regression snapshots
→ physical migration only after stability
```

#### Modern Frontier: Add, But Do Not Promote

CX agrees that RD-Agent, Kronos, CryptoTrade, and on-chain GNNs should be added.

But they remain:

- shadow/backlog
- not Phase A
- not replacement for data/fees/backtest/paper

Reason:

- Frontier methods have high publication velocity and high overfit risk.
- This project's current edge is disciplined data + gate + paper execution, not
  model spectacle.

### Merged Phase Plan

Revised plan after CC feedback:

```text
Crypto-0:
  data contract + architecture interfaces + asset-implicit audit
  + evidence tags + hard-gotcha checklist

Crypto-A:
  spot OHLCV + funding/OI data foundation
  + UTC schema + closed-candle health + venue-aware instrument IDs

Crypto-B:
  feature cache + fee-aware baseline
  + sign-flip checks + survivorship-aware universe

Crypto-C:
  supervised XGB/LGB ranking
  + weekly retrain experiments if data volume supports it

Crypto-D:
  paper OMS + CryptoSanitizer
  + no live leverage

Crypto-E:
  perp RiskGuard + paper funding-arb strategy
  + no production funding arb until local validation

Crypto-F:
  event/on-chain shadow overlays
  + RD-Agent/Kronos/CryptoTrade/GraphSAGE research backlog

Crypto-G:
  Nautilus prototype
  + gradual physical core/ashare/crypto migration if regression gates pass

Crypto-H:
  RL allocation/risk controller only if baseline and paper evidence justify it
```

### Action Items For CX Documents

Patch the CX roadmap and literature review with:

- modern frontier section
- hard gotchas section
- `core/ + ashare/ + crypto/` as target architecture
- library-over-framework rule
- DeFi out-of-scope paragraph
- numeric evidence tags
- funding/OI moved into Phase Crypto-A
- explicit data-contract schema

This addendum resolves most CC/CX disagreements:

- CC accepts CX caution on capital, DeFi, migration, and Nautilus timing.
- CX accepts CC additions on frontier, hard gotchas, target architecture, and
  funding/OI data placement.

## Remaining Non-Accepted Items For CC

This section is intentionally stricter than the merged roadmap. These are the
points CX still does **not** accept from the CC plan, with the reasoning made
explicit so CC can challenge the premises directly.

### 1. Reject "Funding Arb First Production Strategy"

CC claim:

- Start with funding arbitrage.
- Move from paper to small real capital quickly.
- Treat funding carry as the lowest-risk crypto edge.

CX position:

- Accept funding/OI as Phase Crypto-A data.
- Accept funding-arb backtest and paper strategy.
- Reject funding arb as the first production strategy or first live-capital
  mandate.

Argument:

1. Funding rate is not PnL. Funding PnL is conditional on borrow/collateral
   cost, mark-price path, liquidation buffer, venue downtime, forced deleverage,
   position caps, fee tiers, maker/taker mix, and cross-exchange transfer
   latency.
2. Positive historical funding frequency does not imply positive realizable
   carry for a solo local system. The worst periods matter more than the mean:
   negative funding regimes, basis compression, liquidation cascades, exchange
   withdrawal halts, and stablecoin/depeg events all arrive exactly when
   leverage is least forgiving.
3. Funding arbitrage is execution alpha mixed with balance-sheet risk. It is
   structurally different from a daily cross-sectional ranking model: failure
   mode is not "lower IC"; failure mode is liquidation, trapped collateral, or
   venue-specific operational loss.
4. The current project has no crypto venue reconciliation, no collateral ledger,
   no liquidation simulator, no insurance-fund/ADL risk model, no funding
   settlement audit, and no 24/7 alerting. Starting live before these exist is
   inconsistent with the same production discipline being enforced on A-share.

Acceptable alternative:

```text
Crypto-A: collect funding/OI from day one.
Crypto-B/C: test funding/OI as features and risk/crowding signals.
Crypto-D: paper OMS must include funding settlement accounting.
Crypto-E: paper funding-arb strategy and negative-regime stress tests.
Crypto-G+: only then consider tiny live capital, with explicit kill-switches.
```

Minimum evidence before live funding arb:

- venue-specific funding history, not only aggregate dashboard statistics.
- net-of-fee, net-of-slippage, net-of-borrow/collateral-cost backtest.
- stress windows with consecutive negative funding and large spot/perp moves.
- simulated withdrawal halt / venue outage / stale websocket behavior.
- collateral ledger reconciliation against exchange statements.
- max loss under liquidation-buffer assumptions.

### 2. Reject Fixed Capital Schedule Before Validation

CC claim:

- Use a staged capital ladder such as small paper, then tens of thousands, then
  larger allocation.

CX position:

- Reject dollar-based phase gates.
- Use evidence-based state gates.

Argument:

1. Capital size is not a phase definition. The same dollar amount can be safe or
   reckless depending on liquidity, leverage, exchange, funding regime, and
   monitoring coverage.
2. A fixed dollar ladder creates pressure to promote a strategy because time has
   passed, not because evidence improved.
3. The current A-share project learned the hard way that data time discipline,
   mask consistency, and paper/live semantic alignment must precede confidence.
   Crypto has strictly more operational failure modes than A-share.

Acceptable alternative:

```text
Promotion gate = data completeness + reproducible backtest + paper/live
reconciliation + RiskGuard behavior + drawdown/latency/venue-failure stress.
Capital amount is decided only after the gate passes.
```

### 3. Reject DeFi Base Yield Inside The Quant Pipeline

CC claim:

- DeFi base yield can be part of the crypto allocation stack.

CX position:

- DeFi yield may be tracked as a capital-management benchmark.
- It should not be implemented inside the quant feature/model/OMS pipeline.

Argument:

1. DeFi yield is a different risk object: smart-contract risk, governance risk,
   oracle risk, bridge risk, stablecoin risk, redemption liquidity, regulatory
   and custody risk. These are not the same controls as exchange-traded
   spot/perp quant.
2. Mixing DeFi into the main quant pipeline blurs attribution. If total returns
   improve, it becomes unclear whether the model improved, carry improved, or a
   hidden credit/liquidity risk was added.
3. A solo Mac Studio setup benefits from narrower operational scope. Every new
   custody surface increases tail risk and monitoring burden.

Acceptable alternative:

```text
Track DeFi as an external benchmark:
"If the quant book cannot beat conservative stablecoin carry after risk and
ops costs, the quant book is not yet worth scaling."
```

But do not automate DeFi deposits/withdrawals in the Phase Crypto-A to G main
pipeline.

### 4. Reject Immediate Physical `core/ + ashare/ + crypto/` Migration

CC claim:

- Build target namespace architecture early.
- Extract `core/` and migrate A-share code.

CX position:

- Accept `core/ + ashare/ + crypto/` as the long-term target.
- Reject immediate physical migration of A-share production code.

Argument:

1. The A-share pipeline is already operationally fragile around data timing,
   paper/live semantics, cron, event overlays, and masks. A physical namespace
   migration touches import paths, configuration, cron entrypoints, persisted
   state, and tests. That is a broad blast radius unrelated to proving crypto
   alpha.
2. The useful part of the architecture can be obtained earlier through small
   protocols and adapters: `TimeAxis`, `SettlementModel`, `InstrumentId`,
   `CostBundle`, `UniverseFilter`, `DataAvailability`. These force the right
   abstractions without moving production files.
3. Greenfield crypto can validate the abstractions first. After they survive
   crypto paper OMS and backtest, then A-share migration has evidence instead of
   being a speculative cleanup.

Acceptable alternative:

```text
Crypto-0: define core protocols and asset-implicit audit.
Crypto-A-D: implement crypto against those protocols.
Crypto-G+: physically migrate ashare/ only after regression gates prove no
A-share cron/backtest/paper behavior changes.
```

### 5. Reject Early NautilusTrader Dependency

CC claim:

- NautilusTrader is the serious execution/backtest framework and should be on
  the roadmap.

CX position:

- Accept NautilusTrader as the strongest later prototype candidate.
- Reject NautilusTrader as Phase A/B dependency.

Argument:

1. Nautilus is not just a library; it imposes an event-driven actor/runtime
   model. The current project is cron-oriented, parquet/jsonl-backed, and
   status-file monitored. Importing Nautilus piecemeal creates two process
   models without a clean owner.
2. Early crypto work needs data correctness, fee models, stale-bar guards,
   instrument identity, UTC semantics, and paper reconciliation. These do not
   require Nautilus.
3. If Nautilus is adopted before the internal contracts are stable, the project
   risks designing around framework constraints rather than around the actual
   A-share/crypto differences it must understand.

Acceptable alternative:

```text
Phase A/B: no Nautilus dependency.
Phase D/E: internal paper OMS and RiskGuard prove semantics.
Phase G: Nautilus prototype compares fills, costs, funding settlement, and
state recovery against internal results.
```

### 6. Reject Frontier Models As Production Before Baselines

CC claim:

- RD-Agent, Kronos, CryptoTrade, GraphSAGE/on-chain GNN and other frontier
  methods should be considered.

CX position:

- Accept them as Phase Crypto-F research/shadow backlog.
- Reject them as production dependencies before baseline data/backtest/paper
  evidence.

Argument:

1. Crypto alpha decays faster than A-share daily factors. A frontier model that
   looks strong in a paper or benchmark can fail after fees, latency, exchange
   filters, and survivorship correction.
2. Many LLM/agent approaches optimize research throughput, not production
   reliability. This project's current bottleneck is not lack of model
   spectacle; it is end-to-end trust: data availability, masks, leakage,
   execution semantics, and promotion gates.
3. For a single Mac Studio setup, the opportunity cost of heavyweight model
   engineering is high. The first win should be a boring, reproducible baseline
   that survives local validation.

Acceptable alternative:

```text
Use frontier models only after:
1. clean data contract exists,
2. simple XGB/LGB/vectorbt baseline exists,
3. paper OMS exists,
4. shadow evaluation can compare incremental IC, turnover, drawdown, and
   cost-adjusted return.
```

### 7. Evidence Standard CC Should Use

For any claim that affects phase order, capital allocation, or production
readiness, CC should label the evidence:

- `[paper-reported]`: result reported by a paper; not assumed tradable.
- `[exchange-dashboard]`: exchange/blog/dashboard statistic; useful context,
  not strategy validation.
- `[open-source-backtest]`: reproduced by an external repo; check assumptions.
- `[validated-on-local]`: reproduced in this project with local data, costs,
  universe, and time semantics.

Only `[validated-on-local]` should be allowed to promote a strategy toward live
capital.

Final converged position:

```text
Build boring infrastructure first.
Collect crypto-native data early.
Treat crypto as 24/7 T+0 with bar-real-time data, not A-share T+1.
Keep frontier models in shadow.
Adopt target architecture deliberately.
Do not trade real leverage before local validation.
```
