# Phase B.4 — Four-Way 24-Split Verdict (2026-06-06)

## Summary

**xgb_209** (xgb_242 minus Phase B Bucket A trio: cross_market_regime,
capital_flow, shareholder) is the new champion candidate. It beats the
incumbent xgb_242 on every metric and beats the other two reference
configs on Spread20 by a large margin.

## Same-exam table

All four runs use the current code (commit `eed1509` for B.3 runs),
end-date `2026-05-19`, 24-split harness, xgb-only with
`n_estimators=500 / early_stopping_rounds=30`.

| Profile | Feat | RankIC | ICIR | **Spread20** | Days | Cache |
|---|---|---|---|---|---|---|
| xgb_205 (legacy 174-runner) | 205 | **+0.0419** | +0.346 | 25.58 bps | 777 | `feature_cache_174_holder_regime_ma.parquet` |
| **xgb_209 (242 − Bucket A)** | 209 | +0.0345 | **+0.351** | **73.03 bps** | **1223** | `feature_cache_242_production.parquet` |
| xgb_175 (5-26 stale) | 206 | +0.0288 | +0.244 | 35.74 bps | 867 | `feature_cache_175.parquet` |
| xgb_242 (current prod) | 242 | +0.0273 | +0.250 | 35.01 bps | 820 | `feature_cache_242_production.parquet` |

## xgb_209 vs xgb_242 (the head-to-head)

| Metric | xgb_242 | xgb_209 | Δ |
|---|---|---|---|
| RankIC | +0.0273 | +0.0345 | **+0.0072** ✅ |
| ICIR | +0.250 | +0.351 | **+0.101** (+40% stability) ✅ |
| Spread20 | 35.01 bps | 73.03 bps | **+38 bps (more than 2×)** ✅ |
| Days | 820 | 1223 | +403 days (50% more test coverage) ✅ |

Win across all four dimensions. Phase B 6-split LOO predicted Δ-sum
of +0.0151 from Bucket A drops; 24-split realises +0.0072, about half
the linear projection but in the same direction.

## Why xgb_209's Spread20 is so much better

Hypothesis: Bucket A's three loaders (cross_market_regime 27 cols,
capital_flow 3 cols, shareholder 3 cols) inject high-noise / late-feed
signals that pull the model toward false top-20 picks. Dropping them
sharpens the right tail. The 6-split LOO already flagged these three
as net-negative; 24-split confirms the effect is real and amplified
when all three are dropped together.

## Why xgb_205 has higher RankIC but loses on Spread20

xgb_205 covers only 777 days vs 209's 1223 — a narrower window where
RankIC tends to be inflated by less coverage of regime transitions.
Spread20 is the tradable metric: 25.58 bps vs 73.03 bps is
overwhelming. A model that predicts the long-short top-20 spread
better is the one that survives transaction costs.

## Why xgb_175 doesn't move the needle

xgb_175 trained on the stale 5-26 cache. It scores almost identically
to xgb_242 (RankIC +0.0288 vs +0.0273; Spread20 35.74 vs 35.01). This
confirms the legacy 175-cache config has no edge over the current 242
config — the gain comes from the Bucket A drop, not from going back
to the older feature set.

## Promotion decision

**Promote xgb_209 to next champion.** Phase B.4 gate (RankIC ≥ +0.005
over incumbent AND tighter Spread20) is met decisively:

- RankIC Δ = +0.0072 (1.4× the +0.005 threshold)
- Spread20 Δ = +38 bps (more than 100% improvement)
- ICIR Δ = +0.101 (40% improvement in stability)

Production rollout: regenerate the 209-feature production parquet
(drop cross_market_regime + capital_flow + shareholder from the supp
loader path), retrain on the most recent data window, and deploy as
the new daily model. Keep xgb_242 as the rollback target.

## Caveats

1. **End-date frozen at 2026-05-19**. The model has not yet been
   trained against the most recent ~12 trading days. Operator must
   confirm the gain persists when training to current data.
2. **Single random seed / hyperparameter set**. Phase B did not sweep
   xgb hyperparameters; the Spread20 doubling could partially reflect
   noise in the train/test split boundaries.
3. **Bucket B groups (macro_zero_baseline / st_holder_number /
   valuation / st_daily_basic, 25 cols total) were NOT touched**. If
   any of these have started decaying, 209 could still leave alpha on
   the table.
4. **Cross-market_regime drop loses the 2024+ Hang Seng / NASDAQ
   overlay**. Production decisions taken outside Asian hours (HK/US
   open) no longer have these features. If a future regime shift
   makes overseas correlation important again, this drop will need to
   be revisited.

## Recommended next steps

1. **Rebuild production cache as 209-feature parquet** (drop the
   three Bucket A groups at the cache writer, not at runtime).
2. **Retrain xgb_209 on data through latest available date** before
   any production deployment.
3. **Phase B.5 (optional)** — Bucket B sweep: drop the 4 marginal
   groups one at a time on 24-split to confirm none have decayed
   into the negative.
4. **Shadow paper-trading parallel** for 5-10 trading days: xgb_209
   as shadow vs xgb_242 as live. Compare actual top-20 long-short
   returns before flipping the production switch.

## Provenance

- Runner: `scripts/phase4e_24split_ensemble.py` (commit `eed1509`,
  with comma-separated `--drop-group` joint-drop support).
- Ledger rows (all `data_end=2026-05-19`):
  - `xgb_205_legacy_174_runner` — 2026-06-05 22:16
  - `xgb_242` — 2026-06-05 23:54
  - `xgb_175` (logged as `xgb_24split`) — 2026-06-06 13:42 approx
  - `xgb_24split_drop_cross_market_regime+capital_flow+shareholder` — 2026-06-06 13:31
- Phase B 6-split LOO audit: `docs/phase_b_loo_audit_20260606.md`
