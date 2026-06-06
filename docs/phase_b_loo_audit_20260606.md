# Phase B — LOO Ablation 6-Split Audit (2026-06-06)

Companion narrative to `docs/loo_ablation_20260606.md` (the
auto-generated ranking from `scripts/loo_analysis.py`). This doc adds
the bucket interpretation, caveats, and Phase B.2 recommendation.

## Scope

Phase B from `plans/ashare-phases-2026-06.md`. Six-split leave-one-out
ablation across the 9 supplementary-loader groups in
`PRODUCTION_SUPPLEMENTARY_GROUPS` that actually inject columns into
the production `feature_cache_242_production.parquet` (fundamental and
northbound return 0 cols today and are excluded).

Goal: identify which of the 84 supplementary columns are net-negative
on the production xgb_242 baseline, so the post-Phase-A.7 retrain can
drop them.

## Wall-clock summary

| Run | Duration | Notes |
|---|---|---|
| baseline (xgb_242 full 242 feats) | ~14 min | RankIC = +0.0309 |
| 9 LOO 6-split | 10:48 → 12:31 (~103 min) | mean ~11.5 min / run |
| Wrapper failures | 0 | cx-review-fix kept FAILED[] empty; all 9 rc=0 |

## Three buckets

**Bucket A — net negative, drop candidates (Δ > 0):**
- `cross_market_regime` (27 cols hsi/hstech/nasdaq) — Δ +0.0063
- `capital_flow` (3 cols flow_*) — Δ +0.0052
- `shareholder` (3 cols holder_total/liquid_*) — Δ +0.0036

Together = 33 cols. If all three are dropped at 24-split confirmation,
xgb_242 → xgb_209 (158 Alpha158 + 51 surviving supp).

**Bucket B — neutral / marginal (|Δ| < 0.005):**
- `macro_zero_baseline` (10 cols, zero-baseline by design) — Δ -0.0030
- `st_holder_number` (1 col) — Δ -0.0036
- `valuation` (7 cols val_*) — Δ -0.0044
- `st_daily_basic` (7 cols st_pe / pb / ps / turnover) — Δ -0.0044

25 cols, near-zero cost. Keep until 24-split disambiguates.

**Bucket C — real signal, keep (Δ < -0.005):**
- `quality` (8 cols qual_roe / margin / growth) — Δ -0.0063
- `st_moneyflow` (18 cols, strongest contributor) — Δ -0.0091

26 cols. Do NOT remove. st_moneyflow alone is 30% of baseline RankIC.

## Caveats

1. **6-split fast screen, not the final word**. Phase B.2 gate requires
   24-split LOO on any group flagged here before changing
   `PRODUCTION_SUPPLEMENTARY_GROUPS`. 24-split for the three Bucket-A
   groups is ~3 × 35 min = ~1.75 h compute.
2. **Single-window evidence**. End-date pinned to 2026-05-19. Alpha
   weights drift; a walk-forward LOO is the proper long-term check.
3. **Order independence not tested**. Removing all three Bucket-A
   groups simultaneously may interact — once the regime overlay no
   longer dominates training residual, the marginal value of
   capital_flow / shareholder may shift. Pair/triple-drop tests should
   precede final removal.

## Recommended next step

1. **Phase B.2 — 24-split confirmation** on the three Bucket-A
   candidates (cross_market_regime, capital_flow, shareholder).
   ETA ~1.75 h compute.
2. **Phase B.3 — joint-drop test**. Train xgb_209 (= xgb_242 minus
   Bucket A) on 24-split. If RankIC ≥ +0.04 (linear projection of
   Bucket A Δ sum ≈ +0.0151 over baseline +0.0309 → ~+0.046), this is
   a substantive improvement and becomes the next champion candidate.
3. **Phase B.4 — promotion gate**. Compare xgb_209 (new) vs xgb_242
   (current) on the same Phase A three-way harness. If xgb_209 wins
   by ≥ +0.005 RankIC AND has tighter Spread20 distribution, promote.

If 24-split refutes the 6-split signal for any group, roll back the
drop list and keep the 242 config.

## Provenance

- Wrapper: `scripts/run_loo_ablation_6split.sh` (cx-review-fix
  FAILED[] tracker + refuse-ALL-DONE-on-failure).
- Analyzer: `scripts/loo_analysis.py` (cx-review-fix expected-groups
  gate; exits 2 on partial sweep).
- Ledger entries: `data/storage/experiments_ledger.jsonl` rows
  `xgb_6split_loo_<group>_20260606_*` + baseline
  `xgb_6split_20260606_104415`.
- Baseline parquet: `data/storage/feature_cache_242_production.parquet`
  (built by `scripts/build_feature_cache_242.py --end 2026-05-19`).
- Code commit at baseline: `1852bf4`; cx-review fixes: `6294d30`.

## Sign-off

Phase B gate prerequisites met:
- A.0 (same-exam infrastructure) ✅
- A.5 (shadow containment) ✅
- A.6 (data health truthfulness) ✅
- A.7 (source-specific SLA gate) ✅
- Phase B 6-split LOO sweep complete (9/9 rc=0) ✅

Phase B.2 (24-split confirmation) can begin once the operator approves
the Bucket-A drop list above.
