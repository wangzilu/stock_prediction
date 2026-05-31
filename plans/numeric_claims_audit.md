# Numeric Claims Audit (4-tier evidence tagging)

Date: 2026-05-31
Status: **pre-Phase-0a audit** — centralizes every numeric claim
across the crypto-quant document set into one table so engineering
decisions never accidentally treat a paper-reported figure as a
locally-validated one.

## Why this exists

Research papers / exchange dashboards report headline numbers
("funding arb Sharpe 2-3", "92% of time funding is positive", "周
3.87% L/S return") that look like engineering inputs. Without
explicit tagging they leak into capital-sizing decisions and
production thresholds. The 4-tier system below is the spec rule
(`cc-crypto-implementation-spec-2026-05-30.md §1.6` + cx Addendum):

| Tier | Meaning | Justifies live capital? |
|---|---|---|
| `[paper-reported]` | Published in a peer-reviewed paper or working paper; results from author's data on author's sample | ❌ No |
| `[exchange-dashboard]` | Vendor / exchange statistic (BitMEX Funding Dashboard, Glassnode, CryptoQuant); not independently audited | ❌ No |
| `[open-source-backtest]` | Reproduced by an external open-source repo with documented assumptions | ❌ No |
| `[validated-on-local]` | Reproduced on THIS project's data with THIS project's costs, universe, time semantics, RiskGuard, paper OMS | ✅ Yes |

**Production rule**: any number used for sizing / risk thresholds /
promotion gates MUST be `[validated-on-local]` first. Anything else
is research motivation, not engineering assumption.

## Audit table

Every numeric claim in the crypto document set, sorted by criticality
to capital decisions.

### Funding / perpetuals economics

| Claim | Source | Tier | Used as sizing input? |
|---|---|---|---|
| BTC perp funding 11% annualized mean | BitMEX Q3 2025 derivatives report | `[exchange-dashboard]` | ❌ research motivation only |
| Net carry 7-10% annualized after fees | He et al. 2024 SSRN 4301150 | `[paper-reported]` | ❌ |
| Funding arb Sharpe 2-3 | He et al. 2024 | `[paper-reported]` | ❌ |
| 92% of time funding is positive | BitMEX 2025Q3 dashboard | `[exchange-dashboard]` | ❌ |
| Bear funding range -1% to -8% | He et al. 2024 | `[paper-reported]` | ❌ |
| Worst consecutive negative funding window 46 days (2022-06-19 to 2022-08-03) | BitMEX historical, cited in cc plan §2 | `[exchange-dashboard]` | ❌ |
| Capacity $10M+ with <5bp slippage at single exchange | cc plan §2 estimate | `[paper-reported]` (unsupported number) | ❌ |
| 8h funding settlement interval at Binance | Binance API spec | `[exchange-dashboard]` | ✅ schedule-only (cron alignment, not size) |

### Cross-sectional factor research

| Claim | Source | Tier | Used as sizing input? |
|---|---|---|---|
| CMOM weekly L/S ≈ 3% | Liu-Tsyvinski-Wu 2022 JoF | `[paper-reported]` | ❌ |
| CTREND weekly 3.87% | Fieberg 2024 JFQA | `[paper-reported]` | ❌ |
| t-stat > 3 for both above | same | `[paper-reported]` | ❌ |
| Equal-weight survivorship bias +62.19% | Ammann 2023 SSRN 4287573 | `[paper-reported]` | ❌ |
| IVOL sign POSITIVE in crypto (vs A-share NEGATIVE) | Zhang-Li 2020 | `[paper-reported]` | ❌ but **architectural rule**: any A-share factor ported to crypto requires forced local sign-validation before promotion |
| MAX (lottery momentum) sign POSITIVE | Li et al. 2021 | `[paper-reported]` | ❌ same architectural rule |
| Alpha decay 5-10x faster than A-share | cx Hard Gotchas synthesis | `[paper-reported]` (mostly anecdotal) | ❌ but **architectural rule**: factor IC monitoring windows shrink from 60-120D to 7-30D, walk-forward retrain from monthly to weekly |
| BTC-ETH cointegration broken since 2022 Merge | cx Hard Gotchas | `[paper-reported]` (correlation drop 0.95 → 0.75 within 47 days, ETH/BTC drifted 47%) | ❌ — used as **architectural rule**: no pairs cointegration on majors |

### On-chain alpha

| Claim | Source | Tier | Used as sizing input? |
|---|---|---|---|
| USDT exchange net inflow strongest signal, 1-6h IC significant | Chi 2025 arxiv 2411.06327 | `[paper-reported]` | ❌ |
| Glassnode Advanced $49/month covers Top-10 on-chain factors | Glassnode pricing page | `[exchange-dashboard]` | ✅ budget item (not signal sizing) |
| CryptoQuant + DefiLlama + Dune free tiers cover ~80% of needed on-chain | research synthesis | `[paper-reported]` (estimate) | ❌ |

### DeFi yield benchmarks

(Per CC self-correction #2: DeFi is OUT OF SCOPE for the quant
pipeline. These remain only as benchmark context for the separate
`capital-management-note.md` if it ever gets written.)

| Claim | Source | Tier | Used as sizing input? |
|---|---|---|---|
| Aave V3 USDC 3.45-5.2% APR | Aave dashboard | `[exchange-dashboard]` | ❌ |
| Sky sUSDS 3.75-4.75% | Sky.money dashboard | `[exchange-dashboard]` | ❌ |
| Morpho Blue vault 5-8% | Morpho dashboard | `[exchange-dashboard]` | ❌ |
| Uniswap v3 LP 51% loss rate | HAL hal-04214315v3 | `[paper-reported]` | ❌ → architectural rule: **no LP strategy** |

### Storage / data volume

| Claim | Source | Tier | Used as sizing input? |
|---|---|---|---|
| top-50 1m all-history 5y ≈ 650M rows | cc plan §3 estimate | `[paper-reported]` (back-of-envelope) | ❌ |
| Parquet compressed 8-12 GB | same | `[paper-reported]` | ❌ |
| Mac Studio 32GB ram sufficient for DuckDB full-scan | cc plan §3 estimate | `[paper-reported]` | ❌ |
| WS realtime top-50 trade+L2 ≈ 3-8 MB/s | cc plan §3 estimate | `[paper-reported]` | ❌ — **Phase 0a measurement spike** will reproduce |
| Polars 5min rolling features < 200ms | cc plan §3 estimate | `[paper-reported]` | ❌ |
| 1.7 ETH ≈ $3,430 at $2,018/ETH | user statement 2026-05-30 | `[exchange-dashboard]` | ✅ capital reality (drives paper-only constraint, not signal sizing) |

### Engineering thresholds (these ARE used as production gates)

| Claim | Source | Tier | Where used |
|---|---|---|---|
| `CLOSED_BUFFER_SEC = 120` | conservative default | `[paper-reported]` (rule-of-thumb) | crypto-data-contract §11 — **revise after Phase 0a measurement** |
| stale-1h `max_lag_sec = 5400` (90 min) | conservative | `[paper-reported]` | crypto-data-contract §11 |
| stale-4h `max_lag_sec = 18000` (5h) | conservative | `[paper-reported]` | crypto-data-contract §11 |
| stale-1d `max_lag_sec = 93600` (26h) | conservative | `[paper-reported]` | crypto-data-contract §11 |
| Cross-source spread YELLOW threshold 25 bps | conservative — normal spread 1-5 bps | `[exchange-dashboard]` | crypto-data-contract §7 |
| Min listed-days 60 (universe expansion only) | cc plan §11 | `[paper-reported]` | crypto-data-contract §11 |
| Min dollar-volume 50M USD/day rolling | cc plan §11 | `[paper-reported]` | crypto-data-contract §11 |
| Min CryptoSanitizer rules count = 12 | cc plan §11 | n/a (rule count, not measurement) | spec §6.5 audit |
| LEGACY_MARKET_CONTEXT_ENABLED default FALSE | user direction 2026-05-30 | n/a (boolean policy) | quarantine PR |
| MAX_LEVERAGE = 1.0 (paper hard cap) | user §−1 paper-only | n/a (boolean policy) | spec §−1 hard rule |

### Performance / risk metrics quoted in research

| Claim | Source | Tier | Used as sizing input? |
|---|---|---|---|
| BTC perp 1h realized vol > 3σ as liquidation cascade proxy | cx system design review §5 | `[paper-reported]` (heuristic) | ❌ — **RiskGuard threshold** for Phase E; must `[validated-on-local]` first |
| Funding rate z-score |>3 as extreme | cx system design review §5 | `[paper-reported]` (heuristic) | ❌ same |
| OI 24h change > 50% + price flat as crowding signal | cx system design review §5 | `[paper-reported]` (heuristic) | ❌ same |
| Stablecoin depeg > 50bps off peg as risk trigger | cx system design review §5 | `[paper-reported]` (heuristic) | ❌ same |
| BTC dominance 30d Z-score \|>2\| as regime change | cx system design review §5 | `[paper-reported]` (heuristic) | ❌ same |

### A-share baseline numbers (cross-referenced, not crypto)

These are NOT crypto claims but appear in crypto docs as comparison
anchors. Tagged here so they're not silently leaked.

| Claim | Source | Tier |
|---|---|---|
| A-share IVOL NEGATIVE sign | local A-share factor tearsheet (memory `feedback_factor_research.md`) | `[validated-on-local]` |
| A-share alpha decay window 60-120D | local A-share research | `[validated-on-local]` |
| A-share factor IC top performers KLEN +0.070 / ROC5_tsmin10 +0.047 / vol_compression +0.031 | memory `experiment_conclusions_20260525.md` | `[validated-on-local]` |
| A-share take-profit 8% / stop-loss 5% defaults | `config/settings.py` | `[validated-on-local]` (production-tuned) |

## Promotion path

Each `[paper-reported]` / `[exchange-dashboard]` row above is a
**research hypothesis to test**, not an engineering input. The
promotion path to `[validated-on-local]`:

1. Phase A: data foundation enables local measurement
2. Phase B: rule-based baselines test the simplest claims (e.g. is
   momentum a thing? does volatility-conditioned reversal exist?)
3. Phase C: shadow model adds the supervised-learning claims (factor
   IC / Sharpe)
4. Phase E: derivative-side claims (funding arb economics) tested
   under paper OMS with cost-aware simulation
5. Phase F: on-chain factor claims tested as shadow overlays
6. ONLY after a claim becomes `[validated-on-local]` AND user
   written sign-off → it may inform capital sizing

## Append-only protocol

This file is append-only after Phase Crypto-0 sign-off (mirror §14.1
discipline). New claims discovered later go in a §"Audit table v2"
section with date stamp; existing rows are not edited in place.

## Cross-doc consistency

When implementing Phase 0a / Phase A, code should NOT cite numbers
from `cc-crypto-quant-integration-plan` §2 / cx-roadmap / cx-lit-review
directly. Instead:

- **For schemas / paths / constants** → cite `crypto-data-contract.md` §11
- **For research claims** → cite this file with the tier explicit
- **For hard policy** (paper-only / leverage cap / quarantine) → cite spec §−1 / §−0.5 / §6.5

That keeps the numeric authority graph acyclic and auditable.
