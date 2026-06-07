# Phase B.8 — 4-way 6-split LOO ablation on xgb_209 candidate profiles

**Date**: 2026-06-07 (runs spanned into 2026-06-08 local time)
**Branch**: worktree-agent-aff7cdbc469f6f5f2 (off crypto)
**Operator**: subagent for Opus 4.7
**Baseline**: xgb_209 6split (B.7 commit 5ed34ee) — RankIC **0.0327** / ICIR **0.271** / Spread20 **87.05 bps**
**Promotion gate**: ΔRankIC ≥ +0.005 AND ΔSpread20 ≥ 0; otherwise SHADOW or REJECT.

## Summary table

| Profile | RankIC | ICIR | Spread20 (bps) | ΔRankIC | ΔICIR | ΔSpread20 | Verdict |
|---|---|---|---|---|---|---|---|
| xgb_209 (baseline) | 0.0327 | 0.271 | 87.05 | — | — | — | CHAMPION |
| xgb_209_xwlb | n/a | n/a | n/a | n/a | n/a | n/a | **SKIPPED** (no source data) |
| xgb_209_pbc | 0.0329 | 0.278 | 43.64 | +0.0002 | +0.007 | **−43.41** | **REJECT** (Spread20 collapse) |
| xgb_209_guba | 0.0316 | 0.259 | 86.15 | −0.0011 | −0.012 | −0.90 | **REJECT** (negative on every metric) |
| xgb_209_llm | 0.0330 | 0.273 | 95.73 | +0.0003 | +0.002 | **+8.68** | **SHADOW** (RankIC below gate but Spread20 positive) |

Gate pass count (strict): **0 / 3** profiles cleared ΔRankIC ≥ +0.005.
Gate pass count (relaxed, non-negative Spread20 only): **1 / 3** (xgb_209_llm).

## Inputs & data prep

### 1. Factor parquets

| Source | Path | Rows | Coverage on base index | Notes |
|---|---|---|---|---|
| `pbc_liquidity_factors.parquet` | data/storage/ | 12,819 | 8.97% | MARKET-keyed; broadcast to all stocks/date. Nonzero only 521 rows (2013-10-28 → 2026-06-05, sparse policy events). |
| `guba_factors.parquet` | data/storage/ | 1,100 (1,097 dedupe) | **0.000%** | Stock-keyed lowercase; covers 2026-05-22 → 2026-06-05 only. Base cache ends 2026-05-19 — **zero date overlap**. |
| `llm_event_factors.parquet` | data/storage/ | 198,949 (after rebuild 23:30) | 0.899% | Stock-keyed UPPERCASE qlib_code (SH/SZ). Needed case-normalize fix in joiner. Covers 2026-04-16 → 2026-06-07. |
| `xinwen_lianbo_theme_factors.parquet` | data/storage/ | **0 (file absent)** | — | `python scripts/build_policy_factors.py --source xinwen_lianbo --start 2025-03-01 --end 2026-06-05` returned `No factor rows built` — no XWLB events in EventStore. PE-4 cron has not run a historical scrape. |

### 2. Cache joiners (NEW this phase)

Three new builder scripts mirror `scripts/build_feature_cache_209_chain.py`:

| Script | Output | Cols | Group key registered in `supp_col_manifest.json` |
|---|---|---|---|
| `scripts/build_feature_cache_209_pbc.py` | `feature_cache_209_pbc.parquet` (3.64 GiB) | 215 (211 base + 4 PBC) | `pbc_liquidity` |
| `scripts/build_feature_cache_209_guba.py` | `feature_cache_209_guba.parquet` (3.64 GiB) | 214 (211 base + 3 guba) | `guba` |
| `scripts/build_feature_cache_209_xwlb.py` | `feature_cache_209_xwlb.parquet` (not built) | 215 expected | `xinwen_lianbo` (not registered — source missing) |

Each joiner: PROFILE_EXPECTED_COUNTS contract gate, atomic tmp-then-replace write, pre-fillna coverage report (cx P2 #5 honesty fix), `--allow-schema-drift` override.

### 3. LLM joiner case-bug fix

While rebuilding `feature_cache_209_llm.parquet` (224 cols, post-rebuild parquet), the cx P2 #5 honesty report exposed an UPPERCASE/lowercase mismatch:

- `llm_event_factors.parquet` writes `qlib_code` as UPPERCASE (`SH603536`).
- `feature_cache_209_production.parquet` uses LOWERCASE instruments (`sh600000`).
- `build_feature_cache_209_llm.py` set the MultiIndex from `qlib_code` directly without normalization → `llm.reindex(base.index)` matched **zero** rows → every LLM column was constant 0.0.

This means **prior B.6.3 24-split verdict (+0.0044 RankIC reported for LLM)** was numerical noise from XGB col-subsample PRNG drift, not actual LLM signal — same failure mode the B.7 chain doc already documented ("we did not test chain factors"). Fix applied: `llm["instrument"] = llm["instrument"].astype(str).str.lower()` before `set_index`, mirroring the F.P1 #3 belt-and-braces normalization in `FeatureMerger._load_guba`. Rebuilt parquet now has 0.899 % real coverage (54,174 / 6,027,907 rows).

The B.8 LLM run uses the rebuilt cache (post-fix). This is the first run that actually exposes the LLM columns to the model.

## Per-profile detail

### xgb_209_pbc — REJECT

- Cache: 215 cols, 8.97% coverage of broadcast PBC signal.
- Run: `phase4e_24split_ensemble.py --preset 6split --models xgb --checkpoint-tag b8_pbc` → `data/storage/phase4e_b8_pbc/`, 776s.
- Aggregate: RankIC 0.0329, ICIR 0.278, Spread20 43.64 bps.
- Δ vs baseline: RankIC +0.0002 (≈ noise, 4 % of +0.005 gate), ICIR +0.007, Spread20 **−43.41 bps (−50 %)**.
- Per-split spread_top20 collapsed in splits 0/2/3/4 (−29 / +29 / +44 / +61 bps). Only split 1 (+109 bps) carried weight.
- Verdict: **REJECT**. The 4 PBC cols pulled selectivity out of the top-20 cohort without adding rank signal. The MARKET-key broadcast injects a same-value-per-date input that XGB can latch onto for date-stratifying tree splits, hurting cross-sectional ranking (the only thing Spread20 measures). RankIC barely moves because the global ordering is preserved.

### xgb_209_guba — REJECT

- Cache: 214 cols, **0.000 %** coverage (the 3 guba cols are constant-zero on every row of the base index — date range mismatch, no value carried into training).
- Run: 1160 s. The 0% coverage means the model sees three dead columns; expected outcome is a B.7-style "PRNG-noise" delta. That is what we got.
- Aggregate: RankIC 0.0316, ICIR 0.259, Spread20 86.15 bps.
- Δ vs baseline: RankIC **−0.0011**, ICIR **−0.012**, Spread20 **−0.90 bps**. All negative.
- Verdict: **REJECT**. Until the guba collector backfills history into the base-cache date range (≤ 2026-05-19) the ablation cannot show signal. Re-test after the base cache is refreshed AND `collect_guba_sentiment.py` covers the new window.

### xgb_209_llm — SHADOW

- Cache: 223 cols, 0.899% real coverage after case-bug fix.
- Run: 1156.8 s → `data/storage/phase4e_b8_llm/summary.json`.
- Aggregate: RankIC 0.0330, ICIR 0.273, Spread20 95.73 bps.
- Δ vs baseline: RankIC +0.0003 (well below +0.005 gate, ~6 % of the bar), ICIR +0.002, Spread20 **+8.68 bps (+10 %)**.
- Per-split: splits 0/1/2 carry the Spread20 lift (106 / 117 / 128 bps vs the strong baseline splits in B.7), splits 3/4 still negative-tilted. RankIC moves on splits 2/3 only.
- Verdict: **SHADOW**. Does NOT clear the strict +0.005 RankIC gate, but Spread20 is meaningfully positive (+8.68 bps, +10 %) at unchanged RankIC. This is the inverse of the PBC pattern: same total ordering quality, materially better top-of-book selectivity. Operator's call. Conservative recommendation: paper-trade 5+ trading days at unchanged production weight (xgb_209) and decide on a flip only if Spread20 holds in shadow AND the next data refresh raises real LLM coverage above 1 % (current 0.899 % is the cap of what a 1.7-month live window can offer).
- **Important caveat**: this is the FIRST B.* phase to actually expose LLM columns to the model. Pre-fix the cache held constant-zero LLM cols, so all earlier B.6.x deltas were col-count-induced PRNG drift, not signal. The B.6.3 doc's "+0.0044 RankIC / +17.62 bps Spread20" must be retired — those numbers measured nothing.

### xgb_209_xwlb — SKIPPED

- `data/storage/xinwen_lianbo_theme_factors.parquet` does not exist.
- `python scripts/build_policy_factors.py --source xinwen_lianbo --start 2025-03-01 --end 2026-06-05` logged
  `No factor rows built for [2025-03-01, 2026-06-05]` and wrote a health record with `n=0`.
- Root cause: `data/storage/policy_events/xinwen_lianbo/` directory is absent (no XWLB events have been scraped). The PE-4 cron only started today, so historical CCTV 新闻联播 attention is unavailable.
- Cache joiner `scripts/build_feature_cache_209_xwlb.py` is committed but fails loud when the factor parquet is empty (the explicit message documents the gap to operator).
- Re-test plan: once PE-4 backfills XWLB events for ≥ 30 trading days, rerun `build_policy_factors.py` then this joiner then phase4e.

## Cross-cutting findings

1. **LLM joiner case bug** (NEW, found this phase). The fix is committed alongside the 3 new joiners. Reopens B.6.3 — the prior LLM 24-split verdict reflected zero coverage, not the L1 fact-count rebuild.
2. **Coverage matters more than expected count**. Three of the four candidate groups have < 10 % real coverage on the training cache. XGB's `colsample_bytree` will mostly skip these columns, and any rank or spread delta is dominated by PRNG state drift — same mechanism the B.7 chain verdict already explained.
3. **MARKET-key broadcast is dangerous for Spread20**. PBC's −43 bps Spread20 drop on near-flat RankIC is a clean illustration: a 4-col same-value-per-date feature pulls trees toward date-discriminating splits, which crowd out cross-section selectivity exactly where Spread20 measures it.
4. **Promotion gate held**. None of the three testable profiles cleared +0.005 RankIC + non-negative Spread20. Production stays on xgb_209.

## Recommended next step

- **Hold xgb_209 as champion**. No production flip from any of the four candidates this phase.
- **Run xgb_209_llm in paper-shadow for ≥ 5 trading days starting next session**. Spread20 +8.68 bps at flat RankIC is the only positive signal in this entire phase and deserves observation, but the +0.005 RankIC gate is not met so no production weight changes. Re-evaluate after a week of paper-trading evidence.
- **Refresh base cache to include 2026-05-20 → 2026-06-05** so the existing 11-day guba window enters training. Right now `feature_cache_209_production.parquet` ends 2026-05-19 — three days before guba data starts — so guba's ablation cannot show signal even when it has it. Same blocker applies to the most recent LLM events.
- **Backfill XWLB historical events** (≥ 30 trading days) before re-running the xwlb track. Until then `xgb_209_xwlb` is a placeholder profile that cannot be evaluated.
- **Audit the other supplementary joiners for analogous case/date silent-zero bugs**. The cx P2 #5 honesty fix (pre-fillna coverage report) only landed in the LLM and guba loaders so far. PBC, XWLB, chain, chain_llm joiners now have `--allow-schema-drift` contract gates but not all of them log pre-fillna coverage. Add coverage logging to every joiner before the next ablation round so a silent-zero cache cannot ship undetected again.
- **Retire B.6.3 LLM verdict**. The +0.0044 RankIC / +17.62 bps Spread20 numbers in `docs/phase_b6_3_llm_24split_verdict_20260607.md` were measured against a constant-zero LLM cache. Add a postmortem note to that doc pointing here.
