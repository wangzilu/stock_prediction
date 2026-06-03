# Crypto Quant Development Phases

**Single-source roadmap.** Consolidates the 5/30 research bundle into one
canonical phase doc, plus a 2026-06-03 freshness delta. When you start
crypto work, read THIS file — not the 8 underlying plans.

| Doc | Use when |
|---|---|
| **`crypto-dev-phases.md`** (this file) | Roadmap + acceptance gates. **Start here.** |
| `crypto-quant-literature-and-engineering-review-2026-05-30.md` | Paper-by-paper evidence trail |
| `cc-crypto-implementation-spec-2026-05-30.md` | File-level implementation spec |
| `crypto-data-contract.md` | UTC schema + symbol identity |
| `cc-crypto-quant-integration-plan-2026-05-30.md` + `cx-*review*.md` | Original cc/cx design dialogue (archive) |

Memory anchors: [[crypto-quant-research-20260530]] (core conclusions),
[[crypto-paper-only-2026-05-30]] (paper-only constraint),
[[crypto-capital-reality-2026-05-30]] (1.7 ETH capital reality),
[[ashare-isolation-during-crypto-dev-2026-05-30]] (isolation invariant),
[[crypto-quarantine-soak-2026-05-30]] (current quarantine status).

---

## Constraints That Override Everything

1. **Paper-only, no leverage** until user explicitly lifts the rule.
   Source: `feedback_crypto_paper_only`.
2. **A-share isolation invariant**: no crypto bug, slow import, network
   call, or cron may slow / break A-share production.
   Source: `feedback_ashare_isolation`.
3. **All crypto data through `ssproxy`** (mainland network reality).
4. **All crypto data stored on `/Volumes/DATA/crypto`** (external disk).
5. **1.7 ETH (~$3,430) capital reality** drives strategy choice — small
   size means MM / HFT / large-tick paths are non-starters.

---

## Phase Crypto-0 — Evidence and Data Contract (3-5 days)

Pure documentation phase. **No trading code.**

**Deliverables**
- `plans/crypto-data-contract.md` ✅ (exists)
- `data/storage/crypto/README.md` — symbol identity, UTC convention,
  exchange precedence
- Universe list (initial 5 symbols: BTC / ETH / SOL / BNB / XRP) with
  justification
- Hard-gotcha checklist embedded in `feedback_crypto_quant_research`
  memory (already done)
- Evidence-tagged constants in spec — each numeric assumption (slippage,
  funding net-of-cost rate, IVOL sign) gets `[evidence:<source>]` tag

**Acceptance**
- UTC schema reviewed and merged
- No imports of A-share modules in any crypto-namespace file
- Every numeric sizing / risk assumption has an evidence tag
- A-share cron jobs continue passing daily 5/5 GREEN check
- Quarantine PR (`fix/legacy-crypto-quarantine`) is merged to master

**Dependencies**
- Crypto quarantine soak completes GREEN. **Day 1 = 2026-06-02 ✅.
  Day 2 = 2026-06-03. Day 3 = 2026-06-04 → merge to master.**

---

## Phase Crypto-A — Data Foundation (1-2 weeks)

Pull OHLCV + perp funding + open interest. **No alpha. No models.**

**Deliverables**
- `data/collectors/crypto_market.py` (OHLCV via CCXT, paper-friendly
  endpoints first)
- `data/collectors/crypto_derivatives.py` (funding + OI via exchange
  REST, anonymous, no API key)
- `scripts/crypto_update_market_data.py` (idempotent parquet writes to
  `/Volumes/DATA/crypto/...`)
- `scripts/crypto_data_health.py` (gap report + freshness gate)
- Cron entries with `ssproxy` wrapper

**Acceptance**
- 1h / 4h / 1d OHLCV for 5 symbols, 60+ days history
- Funding + OI history for BTC / ETH / SOL (Binance + Bybit + OKX +
  **Hyperliquid** — see 6/3 delta below)
- Idempotent writes (re-run yields same parquet)
- Health file flags any > 2h gap, any spot/perp price divergence > 0.5%
- A-share cron jobs continue 5/5 GREEN

**Dependencies**
- Phase Crypto-0 closed.

---

## Phase Crypto-B — Feature and Baseline Backtest (1-2 weeks)

First non-trivial code. Establishes IC baseline and cost model.

**Deliverables**
- `models/crypto_feature_pipeline.py`:
  - Top 5 momentum / reversal / IVOL / MAX / volume features
  - **Forced sign validation** at construction time — if A-share sign
    fires opposite of crypto sign, fail loudly (per 6/3 hard-gotcha
    enforcement)
- `scripts/crypto_build_features.py` (feature cache, weekly walk-forward)
- `scripts/crypto_backtest_baseline.py` (momentum / reversal baseline)
- `backtest/crypto_cost_model.py` reusing `backtest/cost_model.py`
  sqrt_adv + crypto-specific fees (maker / taker per venue, **funding
  cost included in perp legs**)
- RankIC / spread / PnL daily report

**Acceptance**
- Baseline beats buy-and-hold on Sharpe (after-cost) AND beats random
- Cost model exhibits sqrt_adv scaling, NOT static slippage
- Each imported A-share factor's sign is validated in code (assertion)
- Survivorship bias controlled — dead coins kept in universe
- A-share cron jobs continue 5/5 GREEN

**Dependencies**
- Phase Crypto-A closed.
- A-share Batch 4 P2 sqrt_adv must be wired to OMS first (Task #63),
  otherwise the cost model can't share infrastructure.

---

## Phase Crypto-C — Funding Arb Backtest (1-2 weeks)

The first concrete strategy. Funding rate carry, market-neutral.

**Deliverables**
- `strategies/crypto/funding_arb.py` (long-perp / short-spot or
  cross-venue perp/perp basis)
- `scripts/crypto_backtest_funding_arb.py`
- `paper/crypto_oms.py` (T+0 state machine, NOT A-share reconcile)
- Reporting: net-of-cost APR, drawdown, funding-flip frequency, capacity

**Acceptance**
- After-cost Sharpe ≥ 1.5 on 60+ day backtest at $5k position size
- After-cost APR ≥ 5% on majors **OR** explicit "doesn't work at this
  size" verdict with evidence (per 6/3 delta — only 40% of top opps net
  positive)
- Capacity estimate: max position before funding-rate self-impact
- Funding-flip detection: rule that flat-or-flips when sign reverses
- A-share cron jobs continue 5/5 GREEN

**Dependencies**
- Phase Crypto-B closed.

---

## Phase Crypto-D — Paper Trading (30 calendar days minimum)

No live keys. No leverage. Paper OMS only.

**Deliverables**
- `scripts/run_crypto_paper_trading.py` (cron'd)
- `scripts/crypto_daily_report.py` (PnL, position, basis, risk metrics)
- Stale-data block (no signal if data > 30min stale on any venue)
- Position cap enforcement (no single position > 30% paper book)
- Daily push to user with one-line summary

**Acceptance**
- 30 calendar days continuous paper run, zero crashes
- After-cost Sharpe consistent with backtest (no > 30% degradation)
- A-share cron jobs continue 5/5 GREEN throughout
- User reviews logs at least weekly and signs off

**Dependencies**
- Phase Crypto-C closed.

---

## Phase Crypto-E and beyond — Frozen pending paper soak

On-chain factor overlay (Glassnode + CryptoQuant) → cross-section LGB →
event/sentiment integration → multi-venue → live decision.

**Live trading is OUT of scope** until user explicitly lifts paper-only
constraint AND > 30 days continuous Phase Crypto-D GREEN.

---

# 2026-06-03 Freshness Delta (vs 5/30 baseline)

5-day sweep produced **6 material findings** that update the baseline.
Inline into Phase Crypto-A/B/C as flagged.

## Δ1 — Hyperliquid is now the dominant perp venue (Phase A/C)

**5/30 review**: Hyperliquid not mentioned.

**6/3 finding**: Hyperliquid holds **70%+ perp DEX market share**, ~$21.8B
24h volume, ~$7.3B OI. Maker 0.015% / taker 0.045% (cheaper than
Binance / Bybit). Own L1 → zero gas. Funding arb on majors yields
**3-12% net APR**; long-tail (HYPE / XPL / new listings) **20-60%+ APR**.

**Action**: Add Hyperliquid as **first-class venue** in Phase Crypto-A
collectors, NOT a follow-up. Funding arb in Phase Crypto-C should
include Hyperliquid ↔ CEX legs from day one.

**Risk**: DEX-specific gotchas — Hyperliquid auto-deleveraging, no
SDK-level circuit-breaker, settlement on own L1 means custody risk if
chain stalls. Add to hard-gotcha list.

Source: [Hyperliquid Trading Guide 2026](https://www.altrady.com/blog/crypto-trading-tools/hyperliquid-trading-guide),
[Hyperliquid vs CEXs perp arb after fees](https://www.neuralarb.com/2026/04/24/hyperliquid-vs-cexs-perp-arbitrage-after-fees-funding-slippage/),
[Wall Street weekend Hyperliquid](https://www.blockhead.co/2026/06/03/hyperliquid-becomes-the-default-venue-for-wall-streets-weekend-derivatives-trading/),
[MEXC Hyperliquid funding strategy 2026](https://www.mexc.com/learn/article/hyperliquid-funding-rate-strategy-earning-passive-income-in-2026/1).

## Δ2 — Funding arb after-cost reality: 40%, not 92% (Phase C)

**5/30 review claim**: "BitMEX Q3 2025: 92% of time positive funding;
BTC perp ~11% annualized; net carry 7-10%."

**6/3 correction**: MDPI study on CEX/DEX funding markets finds **17%
of observations have ≥20bps arb spreads, but only 40% of top
opportunities generate positive returns after transaction costs and
spread reversals**.

**Action**: Phase Crypto-C acceptance gate downgraded from "Sharpe 2-3"
to **"Sharpe ≥ 1.5 OR explicit fail verdict with evidence"**. After-cost
APR target reduced from 7-10% to **5%+ on majors / 20%+ on long-tail
(Hyperliquid Δ1) — pick one path, not both naively averaged**.

Source: [Two-Tiered Structure of Cryptocurrency Funding Rate Markets, MDPI Mathematics 14(2):346](https://www.mdpi.com/2227-7390/14/2/346).

## Δ3 — Funding rates are engineered, not exogenous (Phase C risk)

**6/3 finding**: arxiv 2506.08573 (June 2026, Bocconi BSDE framework)
proves funding rates can be designed by exchanges as endogenous control
variables for peg stability, using replicating portfolios.

**Action**: Add to hard-gotcha list — exchanges may **deliberately
re-tune funding curves to shrink arb windows**. The 3-year backtest
window may not predict the next 6 months if a major venue updates its
funding formula. Phase Crypto-C should include a "funding-formula drift
detector" that flags when realized funding diverges from a naive
predictor by > 3σ.

Source: [Designing funding rates for perpetual futures, arxiv 2506.08573](https://arxiv.org/abs/2506.08573).

## Δ4 — Alpha decay has a concrete form (Phase B/C calibration)

**5/30 review claim**: "Alpha decay 5-10× faster than A-share".

**6/3 refinement**: arxiv 2512.11913 derives **hyperbolic decay
α(t) = K / (1 + λt)** from a game-theoretic equilibrium. Mechanical
factors (momentum, reversal) fit; judgment-based factors (value,
quality) don't. **Crowded reversal factors show 1.7-1.8× higher crash
probability**.

**Action**: Calibrate λ from each factor's IC time series in Phase
Crypto-B. Refuse to promote a factor where the fitted λ implies
half-life < 30 days. Add crowding-tail-risk reduction: trim reversal
factor exposure on crowded regimes.

Source: [Not All Factors Crowd Equally, arxiv 2512.11913](https://arxiv.org/abs/2512.11913).

## Δ5 — LLM factor agent achieves 44% OOS Sharpe 1.55 (Phase E candidate)

**6/3 finding**: arxiv 2604.26747 (April 2026) — Constrained LLM Agent
proposes factor hypotheses in DSL, deterministic engine evaluates with
PIT splits / costs / portfolio tests. Ridge-combined portfolio trained
2020-2022 achieves **44.55% annualized, Sharpe 1.55** on 2024-2026 pure
OOS after 5bps one-way cost.

**Action**: We already have an LLM event pipeline (MiniMax). Add to
Phase Crypto-E backlog: adapt our `factors/llm_event_extractor_v2.py`
prompt path to propose crypto factor candidates from a PIT DSL. Keep
deterministic gate strict — the paper's trick is that the gate, not the
LLM, does the science.

Source: [From Hypotheses to Factors: Constrained LLM Agents, arxiv 2604.26747](https://arxiv.org/abs/2604.26747).

## Δ6 — Framework reality check (Phase A/C tool choice)

**6/3 finding**:
- **Hummingbot 2.13 (Feb/Mar 2026)** — added Backpack / Aevo / Pacifica
  connectors + **MCP server integration for Claude Code**. Direction:
  AI-controlled bots via Telegram + Condor interface.
- **Freqtrade 2026.3** (April 2026) — moved to year-based versioning.
  FreqAI integrated for ML strategies. 30+ exchanges via CCXT.
- **NautilusTrader** — still the production-grade Rust-core async path.
  No 2026 inflection point.
- **CCXT** — remains the de-facto exchange abstraction for non-Rust
  paths.

**Action**:
- Phase Crypto-A: collectors **CCXT-based**, not framework-locked
- Phase Crypto-C: paper OMS = **own minimal T+0 state machine** (per
  5/30 decision), NOT Hummingbot/Freqtrade
- Phase Crypto-D: monitoring can borrow Freqtrade FreqAI patterns but
  not the runtime
- Phase Crypto-E backlog: Hummingbot MCP integration is interesting for
  the LLM factor agent path (Δ5)

Source: [Hummingbot GitHub](https://github.com/hummingbot/hummingbot),
[Freqtrade vs Hummingbot 2026 comparison](https://gainium.io/compare/freqtrade-vs-hummingbot),
[Top Hyperliquid bot frameworks](https://coincodecap.com/best-hyperliquid-bot-frameworks-sdks-hummingbot-ccxt).

---

## Hard Gotcha Catalogue (consolidated, kept here for quick scan)

Cross-reference: [[crypto-quant-research-20260530]] memory has the
prior baseline list. Below is the **merged 5/30 + 6/3 list**, marked
with origin.

1. `[5/30]` IVOL has POSITIVE sign in crypto (Zhang-Li 2020), vs
   NEGATIVE in A-share. Force sign assertion at factor construction.
2. `[5/30]` MAX has POSITIVE sign in crypto (Li 2021). Same.
3. `[5/30]` Traditional Value (B/P) does not transfer.
4. `[5/30]` BTC-ETH pairs cointegration broken since the Merge (47-day
   structural decay). Don't run cointegration on majors.
5. `[5/30]` Survivorship bias is severe (Ammann 2023: 62% overstatement
   on equal-weight). Keep dead coins with last K-line.
6. `[5/30, refined 6/3 Δ4]` Alpha decay 5-10× faster — now quantified
   as hyperbolic K/(1+λt). Reject factors with fitted λ implying
   half-life < 30d.
7. `[5/30]` Funding rate basis is crypto-native alpha — keep in
   factor zoo.
8. `[6/3 Δ3]` Funding rates may be engineered by exchanges. Run a
   formula-drift detector; tighten window after any venue update.
9. `[6/3 Δ1]` Hyperliquid auto-deleveraging + own-L1 settlement adds
   custody risk not present in CEXs. Position-size separately.
10. `[6/3 Δ2]` After-cost funding arb success rate is ~40% of top
    opps, not 92%. Backtest must include realistic slippage + spread
    reversal.
11. `[6/3 Δ4]` Crowded reversal factors → 1.7-1.8× crash probability.
    Trim reversal exposure when crowding signal flags.

---

## Capital Plan (unchanged, kept for traceability)

| Phase | Strategy | Paper / Live | Capital | Goal |
|---|---|---|---|---|
| 0 (1-2 wk) | Data contract + audit | n/a | $0 | Acceptance gate |
| A (1-2 wk) | OHLCV + perp + funding pull | n/a | $0 | Data foundation |
| B (1-2 wk) | Baseline backtest | n/a | $0 | IC + cost calibration |
| C (1-2 wk) | Funding arb backtest | n/a | $0 | Strategy verdict |
| D (30+ days) | Funding arb **paper** | paper | $0 | OMS soak |
| E (1-3 mo) | On-chain overlay | paper | $0 | + IC 0.04 |
| F (3-6 mo) | Cross-section LGB | paper | $0 | Multi-factor |
| Live | **User decision only**, not auto | live | TBD | After paper-only lifted |

**$0 capital** until user lifts paper-only constraint after >30 days
green paper. The 1.7 ETH stays in cold storage as a training stake, not
a trading bankroll.

---

## Status (today)

- **Crypto quarantine soak**: Day 2 of 3 in progress (2026-06-03).
  Soak GREEN → merge to master → Phase Crypto-0 may begin.
- **Frozen sequence**: A-share Batch 3 (#62) + Batch 4 (#63) precede
  crypto Phase A entry. Crypto-B has hard dependency on Batch 4 P2
  sqrt_adv being wired (Δ6 action).
- **No crypto code in production**: only quarantine flag + lazy import
  + paper collector + tests.

To start Phase Crypto-0, user signals "开 Crypto-0". This file is the
roadmap from that moment.
