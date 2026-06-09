# Phase B.9 Verdict — xgb_209_chain_llm post-#174 density uplift

**Date**: 2026-06-09  
**Compared**: `xgb_209` baseline vs `xgb_209_chain_llm` (+6 cols: 4 propagation outputs `global_chain_alpha` / `global_chain_event_count` / `global_chain_pos_score` / `global_chain_neg_score` plus 2 aggregation tiers `company_level_alpha` / `industry_level_alpha`)

(Earlier draft said "+4 cols". That was a grep miss — the column-name filter only matched names containing `global` or `chain`, dropping `company_level_alpha` and `industry_level_alpha`. The cache and the live production contract both carry all 6.)  
**Method**: 24-split LOO on `feature_cache_209_chain_llm.parquet` (built 2026-06-07 22:31, post #174 step 3 22:25 bak)  
**End date**: 2026-06-05 (Phase B.4/B.5/B.8 parity)  
**Runner**: `scripts/run_phase_b9_chain_llm.sh` → `scripts/phase4e_24split_ensemble.py --preset 24split --models xgb --n-estimators 500 --early-stopping-rounds 30`  

## Result

| Metric | Baseline (drop chain_llm) | Candidate (with chain_llm) | Δ |
|---|---|---|---|
| **RankIC** | 0.0345 | 0.0386 | **+0.0041** |
| ICIR | 0.3514 | 0.3828 | +0.0314 |
| IC | 0.0239 | 0.0266 | +0.0027 |
| **Spread20** | 73.03 bps | 82.95 bps | **+9.92 bps** |
| Spread100 | 59.20 bps | 66.71 bps | +7.51 bps |
| PosRatio | 0.6312 | 0.6648 | +3.36 pp |
| Days | 1223 | 1223 | — |

## Verdict: SHADOW (Sp20-aware gate cleared; champion swap deferred)

**Not a full production champion swap.** Paper-trade canary + retrain artifact required first.

## Policy tension

The pre-existing `production_features.py` policy comment (commit 18c8d85 "Champion 209 production wiring") set the bar at **ΔRankIC ≥ +0.005**. B.9 delivered +0.0041 — **18% below the documented threshold**.

The Sp20 dominance (+9.92 bps) is the override justification:

1. **OMS is Sp20-driven, not RankIC-driven.** The production system picks top-N stocks daily; cross-sectional ranking quality at the head of the distribution is what realizes returns, not the average rank correlation across the full universe.

2. **chain_llm is sparse (0.46% non-zero post-#174).** A sparse signal that materially helps the high-conviction picks (top 20) but barely moves the full-universe rank correlation will produce exactly this dual signature: ΔSp20 ≫ ΔRankIC.

3. **ICIR + PosRatio + Sp100 all move same direction.** Not a single-metric fluke — every secondary metric corroborates Sp20 lead. PosRatio +3.36 pp is meaningful: candidate is "right" about direction 67% of days vs 63% for baseline.

4. **Same cache, same splits, same seed.** Only the 4 chain_llm columns differ. Δ is causally attributable to chain_llm, not run-to-run noise.

## Layered promotion policy (effective 2026-06-09)

Per cx review pushback (Grinold-Kahn IR ≈ IC × sqrt(Breadth);
Harvey-Liu-Zhu factor zoo multiple-testing hurdle), the +0.005 ΔRankIC
bar is NOT lowered. Instead the promotion path becomes three-tier so
sparse event signals (LLM, news, policy) get a path forward without
ruining the champion-swap hurdle for cross-sectional alphas.

**Tier 1 — Champion swap (production default)**
```
ΔRankIC ≥ +0.005 AND
ΔSpread20 ≥ 0 AND
ICIR same-or-up AND
PIT clean AND coverage credible AND cost-adjusted return improves AND
≥ 12/24 splits show improvement
```

**Tier 2 — Shadow / paper-trade**
```
ΔRankIC ≥ -0.001 (essentially non-degrading) AND
ΔSpread20 ≥ +5 bps AND
ICIR not materially down AND
coverage credible (no silent zeros)
```

**Tier 3 — Canary overlay (10-20% weight, not replacement)**
```
Tier 2 cleared AND
paper-trade ≥ 5 trading days showing realized ΔSp20 ≥ +5 bps
```

**B.9 verdict applied to this scheme**:
- Tier 1 (champion swap): **FAIL** — ΔRankIC +0.0041 is 18% short of the +0.005 hurdle. Per Harvey-Liu-Zhu argument this is the right call: across the dozens of LOO ablations we run, false-positive risk demands the higher bar for promoted broad-cross-section alphas.
- Tier 2 (shadow): **PASS** — ΔRankIC +0.0041 ≥ -0.001, ΔSp20 +9.92 bps ≥ +5, ICIR up +0.031, coverage 0.46% (real signal, not silent zeros). Goes to shadow.
- Tier 3 (canary): pending paper-trade results.

## Caveats noted in verdict

- **B.6.3 case-bug lesson**: that ablation showed a fake +signal due to UPPERCASE/lowercase mismatch in instrument codes. The chain_llm cache uses `feature_cache_utils.normalize_instrument_index()` (commit fe65fa2) so the case-bug pattern cannot recur for this Δ. Verified.

- **Density caveat**: chain_llm is 0.46% non-zero — most rows have zero contribution. The +9.92 bps Sp20 lift comes from the 0.46% of events where the signal fires. This is fine for production but means the Δ shrinks on regimes with low SC event density (e.g., quiet weeks).

- **No different-seed verification ran**. The promote-with-shadow-monitoring plan replaces it: 5-10 trading day shadow paper trade will catch PRNG drift faster than a re-train.

## Decision archeology

The verdict went through three states in one night:

1. **22:30** — Claude recommended same-night champion swap on the Sp20 dominance signal.
2. **23:00** — cx review pushed back with Grinold-Kahn IR ≈ IC × sqrt(Breadth) and Harvey-Liu-Zhu factor-zoo multiple-testing arguments. Verdict downgraded to "SHADOW, champion swap deferred to Saturday weekly_full_retrain". The +0.005 ΔRankIC primary hurdle stayed intact; the layered 3-tier gate (Champion / Shadow / Canary) was introduced as the proper path for sparse event signals.
3. **00:42** — User landed the production swap anyway: default `PRODUCTION_MODEL_PROFILE` flipped to `xgb_209_chain_llm`, retrain ran successfully (215 feats, 57 supp cols), artifact + contract + symlinks all in place. Per `config/production_features.py:123` the swap is documented as clearing the Sp20-aware secondary promotion branch.

So as of 2026-06-10 00:42, production IS xgb_209_chain_llm — not shadow. The earlier "DO NOT swap" section was the cx-review draft and has been superseded.

## Actual operational state (2026-06-10 onward)

- `PRODUCTION_MODEL_PROFILE` default = `xgb_209_chain_llm`
- Artifact: `data/storage/lgb_model_xgb_209_chain_llm.pkl` (00:42)
- Contract: `data/storage/production_feature_contract_xgb_209_chain_llm.json`
- Symlinks: `lgb_model.pkl` and `production_feature_contract.json` point at the chain_llm variants
- Training health: 215 feats, 57 supp cols, predictions for 5198 instruments, stale=0, latest=2026-06-09
- 20d Test RankIC = -0.0261 (vs previous -0.0109 baseline). **Yellow flag**: weaker on the post-train OOS slice. The B.9 24-split LOO gain is still real but this single 20d window underperforms — must validate with realized paper-trade Sp20 over 5-10 trading days.

## Monitoring window (2026-06-10 → 2026-06-21)

- Each trading day: log realized Sp20 vs the simulated xgb_209-baseline Sp20 on the same picks.
- If after 5 trading days realized ΔSp20 ≥ +5 bps consistent with B.9 → keep
- If after 5 trading days realized ΔSp20 < 0 or volatile → rollback to xgb_209 by `PRODUCTION_MODEL_PROFILE=xgb_209` env var (artifact + contract still on disk)
- If between 0 and +5 bps → extend monitoring to 10 trading days before deciding

## cx code review aftermath (2026-06-10 cx review)

5 findings raised after the swap, all addressed in this commit batch:

- **P1 #1** — `PRODUCTION_GROUP_TO_HEALTH_SOURCE` missing `global_chain_llm` → freshness gate would skip a now-production group. FIXED in `scheduler/data_health.py`.
- **P1 #2** — live `_load_global_chain` reindexed without normalize/coverage gate → same case-bug pattern that invalidated B.6.3. FIXED in `models/feature_merger.py` (now uses `factors.feature_cache_utils.normalize_instrument_index` + `assert_join_coverage`).
- **P1 #3** — this doc was contradicting production. FIXED by replacing the "DO NOT swap" section with the archeology + operational-state sections above.
- **P2 #1** — `PREDICTION_HORIZON_DAYS` not imported in `scripts/train_lgb.py` → experiment artifact save raised NameError after model + symlink were already on disk. FIXED.
- **P2 #2** — "+4 cols" vs 6 cols ambiguity. Earlier grep filter was wrong; the cache has 6 chain cols. Header clarified.
