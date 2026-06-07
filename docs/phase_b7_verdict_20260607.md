# Phase B.7 verdict — chain rule vs LLM ablation (6-split)

**Date:** 2026-06-07
**Runner:** `scripts/phase4e_24split_ensemble.py --preset 6split --models xgb`
**Sequence:** baseline (17:20-17:39) → rule (17:39-18:00) → LLM (18:00-18:22), ~62 min total
**Outcome:** **REJECT both `xgb_209_chain` and `xgb_209_chain_llm` from production promotion.**

## TL;DR

Both chain factor sources contribute zero predictive value at current event density. XGBoost trees do not split on the chain columns because non-zero rows are < 0.01% of the cache. The pipeline needs a coverage uplift (event density × ~100), not a model change.

## Test setup

Three caches, same 6-split rolling-window splits over 2018-11-06 → 2026-06-08 :

| Profile | Cache | Cols | Aux | Non-zero chain rows |
|---|---|---|---|---|
| `xgb_209` (baseline) | `feature_cache_209_production.parquet` | 209 | 2 | n/a |
| `xgb_209_chain` (rule) | `feature_cache_209_chain.parquet` | 215 | 2 | **636 / 6,027,907 = 0.0106 %** |
| `xgb_209_chain_llm` (LLM) | `feature_cache_209_chain_llm.parquet` | 215 | 2 | **460 / 6,027,907 = 0.0076 %** |

The 6 chain columns are `global_chain_alpha / global_chain_event_count / global_chain_pos_score / global_chain_neg_score / company_level_alpha / industry_level_alpha`. The industry-level column is uniformly 0 in both caches (the SC-A1 shrink + clip stage zeroed it out for low-volume themes).

LLM events were re-built today after the C.P2 #6 confidence-formula fix (multiplicative `ev_weight × rel` instead of `max(floor, rel)`), giving 33 dates × 640 propagation rows. The rule-based source covers 14 dates.

## Results

### Aggregate (6-split mean)

| | RankIC | ICIR | Spread20 |
|---|---|---|---|
| baseline xgb_209 | **0.0327** | 0.271 | **87.05 bps** |
| +rule chain | 0.0333 | 0.275 | 86.02 bps |
| +LLM chain | 0.0333 | 0.275 | 86.02 bps |
| Δ rule vs baseline | +0.0005 | +0.004 | -1.03 bps |
| Δ LLM vs baseline | +0.0005 | +0.004 | -1.03 bps |
| Δ LLM vs rule | **0.0000** (bit-identical to 6 decimals) | **0.0000** | **0.00 bps** |

### Per-split

| Split | window | baseline | rule | LLM |
|---|---|---|---|---|
| 0 (1/6) | test 2025-07-08~2026-06-08 | 0.017186 | 0.015669 | 0.015669 |
| 1 (2/6) | test 2024-08-06~2025-07-07 | 0.041843 | 0.046664 | 0.046664 |
| 2 (3/6) | test 2023-09-05~2024-08-05 | 0.05084 | 0.0502 | 0.0502 |
| 3 (4/6) | test 2022-10-04~2023-09-04 | 0.055344 | 0.054453 | 0.054453 |
| 4 (5/6) | test 2021-11-02~2022-10-03 | -0.001911 | -0.001238 | -0.001238 |
| 5 (6/6) | test 2020-12-01~2021-11-01 | 0.032748 | 0.033258 | 0.033258 |

Every per-split metric for rule and LLM matches to 6 decimal places. The trees are effectively identical.

## Why rule == LLM exactly (and both ≈ baseline)

1. Both caches add 6 numeric columns. Both have non-zero values on < 0.011 % of rows.
2. XGBoost's column subsampling sees columns that are effectively constant at 0. `min_split_gain` filters them out at every split-decision node.
3. The chain columns are present in the feature matrix but unused at training time. Trees become equivalent to the `xgb_209` baseline modulo the column-count change (209 → 215) which slightly shifts the `colsample_bytree` PRNG state.
4. That tiny PRNG drift gives the +0.0005 RankIC / -1.03 bps Spread20 vs baseline — it's numerical noise, not signal.
5. Rule == LLM because their non-zero rows are sparse enough to be in the same PRNG-skipped distribution; both produce the same colsample sequence at every node.

## Implication for the chain pipeline

This is not "chain factors are bad" — this is "we did not test chain factors". The pipeline ran successfully and built non-zero values, but model never saw them.

For a real test the pipeline needs ~100× more density. Three orthogonal paths:

- **Propagation breadth.** Currently 33 event dates × ~20 affected stocks ≈ 660 rows. The industry-level shrink stage zeroed `industry_level_alpha` for every theme below a volume threshold, eliminating the cross-stock broadcast. Re-enabling industry-level propagation (with a tighter shrink target) would multiply density 10-50×.
- **Event coverage.** Backfill GDELT + Google RSS over 2024 fully (currently only sampled dates). 14 → 250+ dates is a 17× multiplier on temporal coverage.
- **Schema move.** SC-A2 v2 schema (relations not direction) was built and would let the propagation broadcast on relation type (supplier / customer / competitor) instead of single-stock entity match. Latent multiplier.

None of these are P0. The decision is to put chain-factor ablation back in the backlog pending a density uplift; do not retest until at least one of the three paths above is shipped and adds ≥ 1 % non-zero coverage.

## Decision

- **REJECT** `xgb_209_chain` for production promotion.
- **REJECT** `xgb_209_chain_llm` for production promotion.
- **KEEP** both as shadow profiles in `SHADOW_SUPPLEMENTARY_GROUPS`.
- **DO NOT** escalate to 24-split. The 6-split signal is unambiguous (bit-identical per-split metrics across rule and LLM = trees never split on chain columns).
- **OPEN BACKLOG**: chain-density uplift (one of the three paths above), retest after.

## Followups (not blocking)

- Investigate why `industry_level_alpha` is uniformly 0 in both caches. The SC-A1 shrink + clip stage may be over-aggressive.
- Consider event-density health metric: rolling 30-day non-zero ratio in `global_chain_factors*.parquet`. Add to LLM factor quality report.
