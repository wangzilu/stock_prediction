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
| **`crypto-daemon-architecture-2026-06-03.md`** | **24/7 event-driven daemon spec — supersedes Phase D's cron assumption** |
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

## Phase Crypto-A — Data Foundation (2-3 weeks, was 1-2)

Two parallel data paths: REST for backfill / health (steps 2-3, **done**),
WebSocket for live event stream (new steps 4a / 4b). **No alpha. No
models.**

**Deliverables**
- ✅ step 2: `data/collectors/crypto_market.py` (REST OHLCV via CCXT)
- ✅ step 3: `data/collectors/crypto_derivatives.py` (REST funding + OI)
- **NEW step 4a**: `crypto/market_stream.py` — WS subscribe + append-only
  event log per `crypto-daemon-architecture-2026-06-03.md` §1 / §2 / §3
  (replay-safe log, seq-gap detection + REST resync, ssproxy keepalive).
- **NEW step 4b**: `crypto/replay.py` — consume WS event log, produce
  byte-identical book/bar/fill state. Contract test pins determinism.
- `scripts/crypto_update_market_data.py` (idempotent parquet writes to
  `/Volumes/DATA/crypto/...`)
- `scripts/crypto_data_health.py` (gap report + freshness gate, reads
  BOTH REST parquet and WS event log)
- **Three** cron entries with `ssproxy` wrapper:
  watchdog (5m), backfill (hourly), daily report (23:55 UTC).
  NO trading-loop cron (per daemon architecture decision).

**Acceptance**
- 1h / 4h / 1d OHLCV for 5 symbols, 60+ days history (REST backfill)
- **1m / 5m / 15m** live bars aggregated from WS for the same 5 symbols,
  ≥ 24h continuous, all closed bars match REST backfill within 1 bp.
- Funding + OI history for BTC / ETH / SOL (Binance + Bybit + OKX +
  **Hyperliquid** — see 6/3 delta below)
- Idempotent writes (re-run yields same parquet)
- WS event log replay produces identical bar / book state vs live
  (determinism contract test)
- Health file flags any > 2h gap (REST), any > 5min gap (WS), any
  spot/perp price divergence > 0.5%
- A-share cron jobs continue 5/5 GREEN

**Dependencies**
- Phase Crypto-0 closed.

---

## Phase Crypto-B — Feature and Baseline Backtest (2-3 weeks, was 1-2)

First non-trivial code. Establishes IC baseline and cost model. Both
the **batch feature pipeline** (research / historical backtest) and
the **online feature pipeline** (daemon hot loop) land here, with a
contract test that they produce the same vector for the same
(timestamp, symbol).

**Deliverables**
- `models/crypto_feature_pipeline.py`:
  - Top 5 momentum / reversal / IVOL / MAX / volume features
  - **Forced sign validation** at construction time — if A-share sign
    fires opposite of crypto sign, fail loudly (per 6/3 hard-gotcha
    enforcement)
  - **Timeframes: 1m / 5m / 15m + 1h / 4h / 1d** (was 1h+ only)
- `crypto/feature_online.py` — rolling-window features computed by
  daemon from live bar stream. Same math as batch pipeline.
- `scripts/crypto_build_features.py` (feature cache, weekly walk-forward)
- `scripts/crypto_backtest_baseline.py` (momentum / reversal / 1m-bar
  fast-bar baseline; published Sharpe targets 1.0-2.4 per 6/3 freshness)
- `backtest/crypto_cost_model.py` reusing `backtest/cost_model.py`
  sqrt_adv + crypto-specific fees (maker / taker per venue, **funding
  cost included in perp legs**)
- **Online/batch parity test**: same (timestamp, symbol) input → same
  feature vector from both paths. Blocking gate for Phase D.
- RankIC / spread / PnL daily report

**Acceptance**
- Baseline beats buy-and-hold on Sharpe (after-cost) AND beats random
  on BOTH bar horizons (1m/5m fast-bar AND 1h+ slow-bar)
- Cost model exhibits sqrt_adv scaling, NOT static slippage
- Each imported A-share factor's sign is validated in code (assertion)
- Survivorship bias controlled — dead coins kept in universe
- Online/batch feature parity test passes (max element-wise relative
  diff < 1e-6 on a 7-day window)
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

## Phase Crypto-D — Paper Trading (21+ calendar days, daemon-driven)

**REVISED 2026-06-03**: Phase D is no longer a cron-driven loop. It
is a 24/7 event-driven daemon plus three cron entries that handle
only watchdog / backfill / daily report. Full architecture spec in
`plans/crypto-daemon-architecture-2026-06-03.md` — read that first
before writing any Phase D code.

No live keys. No leverage. Paper OMS only.

**Deliverables**
- `scripts/run_crypto_daemon.py` (launchd target — Mac-native 24/7
  supervision; Linux systemd unit as future fallback)
- `scripts/install_crypto_daemon.py` (installs the launchd plist)
- `crypto/order_book.py` (in-memory book + WS-gap REST resync)
- `crypto/bar_aggregator.py` (live → 1m/5m/15m closed bars)
- `crypto/risk_guard.py` (stale / spread / vol / gap pre-trade gate,
  fail-closed)
- `paper/crypto_oms.py` (T+0 paper OMS, own state file,
  NOT A-share reconcile)
- `strategies/crypto_fast_bar.py` (1m/5m minute-bar momentum +
  spread filter; first soak strategy per architecture doc §3.3)
- `scripts/crypto_daemon_watchdog.py` (cron'd every 5 min: heartbeat
  + restart on miss)
- `scripts/crypto_backfill_ohlcv.py` (cron'd hourly: REST closed-bar
  resync, never the trading source)
- `scripts/crypto_daily_report.py` (cron'd 23:55 UTC: PnL, position,
  basis, latency, WS gaps, daemon restart count)
- Three cron entries (watchdog / backfill / report) — NONE is a
  trading loop.

**Acceptance** — per architecture doc §6:
- **21 calendar days minimum** continuous daemon uptime (was 30, but
  intraday daemon has more failure modes than a daily cron)
- At least one calendar day with the primary symbol moving ≥ 5%
  during the soak (risk-envelope pressure test)
- WS gap rate ≤ 5/day; P95 RTT ≤ 800ms; daemon restart ≤ 1/day
- After-cost Sharpe consistent with backtest replay (cumulative PnL
  within ±50% band; tighter band proves replay determinism)
- Stale-data block (no signal if last book update > 2s old)
- Position cap enforcement (no single position > 30% paper book)
- Spread filter (reject orders when spread > 25 bps per data
  contract §11)
- **A-share cron jobs continue 5/5 GREEN throughout** — and daemon
  CPU is throttled during A-share active windows (09:25-09:31,
  14:30-14:36, 18:00-18:55, 22:00-22:05 Asia/Shanghai)
- User reviews logs at least weekly and signs off

**Dependencies**
- Phase Crypto-C closed.

---

## Phase Crypto-E and beyond — Frozen pending paper soak

On-chain factor overlay (Glassnode + CryptoQuant) → cross-section LGB →
event/sentiment integration → multi-venue → live decision.

**Newly listed E.0 backlog items (moved out of Phase D per 2026-06-03
architecture pivot)**:

- **E.0a Order book L2 alpha**: book imbalance, micro-price,
  queue-position features. Phase D uses L1 (best bid/ask + last
  trade) only — L2 alpha needs separate research after the daemon
  proves stable on L1.
- **E.0b Hyperliquid daemon**: the Phase D first daemon targets
  Binance (rest of pipeline is Binance-anchored). Hyperliquid SDK
  has the cleanest 2026 Python integration but landing it before
  Binance soak passes risks double-debugging two venue runtimes.
- **E.0c LLM factor agent** on crypto (arxiv 2604.26747 path) —
  44.55% OOS Sharpe 1.55 published; reuse our MiniMax pipeline as
  the factor proposer, deterministic engine for evaluation.

**Live trading is OUT of scope** until user explicitly lifts paper-only
constraint AND ≥ 21 days continuous Phase Crypto-D GREEN.

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

## Δ7 — Daemon architecture decision (2026-06-03 architectural review)

**5/30 + 6/3 baseline assumption**: Phase D runs `run_crypto_paper_
trading.py` as a daily cron — same shape as A-share paper.

**2026-06-03 review finding**: cron is the wrong shape for anything
faster than 1h bars. Process-launch jitter, no continuous state for
order book / pending orders, no WS reconnect handling, REST polling
collapses stream data into samples. Crypto WS is event-driven by
spec (Binance kline 1-2s, futures depth 100-500ms; Bybit 20s
heartbeat).

**Decision**: Phase D pivots to a 24/7 launchd-supervised daemon
plus three cron entries that handle watchdog / backfill / daily
report only. NO trading-loop cron. Full design with seven sharp
details (WS log replay, seq-gap detection, ssproxy keepalive,
daemon resource limits during A-share windows, paper fill sim,
21-day soak bands, daemon/backfill collaboration) is in
**`plans/crypto-daemon-architecture-2026-06-03.md`**.

This pivot does NOT regress what is already shipped: Crypto-A step 2
(REST OHLCV collector, commit `12dabb8`) and step 3 (derivatives
collector, commit `e402a5c`) are kept and re-roled as the backfill
data path. The daemon runs alongside them, never instead of them.

Action:
- Phase A: existing REST collectors keep their role. Add step 4a
  (WS collector) and step 4b (replay engine). See Phase A above.
- Phase B: feature pipeline timeframes expand to include 1m/5m/15m
  alongside 1h/4h/1d. Add online/batch parity contract test. See
  Phase B above.
- Phase C: funding arb backtest unchanged (event-cadence is exchange-
  controlled; daemon switch doesn't affect funding-event spacing).
- Phase D: re-spec to daemon + three cron entries. 21-day soak with
  explicit acceptance bands. See Phase D above.
- Phase E.0 backlog: order book L2 alpha + Hyperliquid daemon + LLM
  factor agent moved here, after Binance L1 daemon proves stable.

Source: deep-research agent run 2026-06-03 plus primary refs:
[Binance Spot WebSocket](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md),
[Binance USD-M Futures Diff Book Depth](https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Diff-Book-Depth-Streams),
[Bybit WebSocket Connect](https://bybit-exchange.github.io/docs/v5/ws/connect).

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

## Status (2026-06-03 EOD)

- **Crypto quarantine**: merged to master 2026-06-03 (`a415605`) after
  soak Day 1+2 GREEN, Day 3 compressed per user decision.
- **A-share Batch 3 macro PIT drop**: merged (`5d01629`). Champion
  models from next train onwards exclude leaked macro_* columns.
- **A-share Batch 4 sqrt_adv plumbing**: merged (`a82fb4d`).
  PortfolioBacktest + paper OMS both have the chokepoint; activation
  is opt-in. cx round-3 P3 docstring fix merged (`2197328`).
- **Crypto-0**: closed (`9144b9e`).
- **Crypto-A step 1** (config modules): merged (`ab8dfba`).
- **Crypto-A step 2** (REST OHLCV collector): merged (`12dabb8`).
- **Crypto-A step 3** (REST derivatives collector): merged (`e402a5c`).
- **2026-06-03 architecture pivot**: Phase D shifts from cron to 24/7
  daemon (this commit). Phase A gains step 4a/4b for WS collector +
  replay engine. Phase B feature timeframes expand to 1m/5m/15m.
  Phase E.0 backlog absorbs L2 alpha + Hyperliquid daemon + LLM
  factor agent.

To start Phase Crypto-A step 4a (WS collector), user signals
"开 daemon" after this architecture doc is reviewed. Until then,
Phase A is paused at step 3 (REST collectors complete).

Outstanding A-share PR decisions (none block crypto code work
technically, but isolation prefers no outstanding A-share PRs while
crypto daemon is being written):
- `fix/llm-prefilter-dedup` (3 commits) — independent review pending
- `fix/train-lgb-use-feature-merger` — paired backtest pending
