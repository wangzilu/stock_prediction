# xgb_174 Fresh-Run Verdict — Archive Number Was Apples-to-Oranges

**Date**: 2026-06-14 (Saturday)
**Trigger**: User asked "现在最强模型是啥?" — I cited xgb_174's `lgb_model_xgb_174.pkl` metrics.json RankIC 0.05117 / ICIR 0.646 as "historical strongest." User pushed back: "175 弱 174 强不合逻辑." We retrained xgb_174 on latest data + re-ran 24-split LOO with the same protocol as Phase B.4 / B.9 to find out.

## Two-step verdict

### Step 1 — Fresh retrain (train_lgb, single-fold)

Ran `PRODUCTION_MODEL_PROFILE=xgb_174 python scripts/train_lgb.py`.
- Train: 2021-06-14 → 2026-03-16
- Valid: 2026-03-17 → 2026-05-15
- Test: 2026-05-16 → 2026-06-14 (15 trading days)

```
Test RankIC: -0.0020 ± 0.0935 (15-day test segment)
```

Single-fold RankIC -0.0020 ≈ 0 within 1σ of zero. Inconclusive. Could be noise; needs proper 24-split.

(Also fixed an accident: train_lgb's `update_legacy_contract_alias()` flipped both `lgb_model.pkl` and `production_feature_contract.json` symlinks to xgb_174. Manually restored both to xgb_209_chain_llm immediately after training so production continued pointing at the right artifact.)

### Step 2 — 24-split LOO (`phase4e_24split_ensemble.py`, B.9 protocol)

```
--preset 24split --models xgb --n-estimators 500 --early-stopping-rounds 30
--end-date 2026-06-09
--cache-path data/storage/feature_cache_174_holder_regime_ma.parquet
```

Final summary (xgb-only):

| Metric | xgb_174 (today) | xgb_209 (B.4) | xgb_209_chain_llm (B.9) |
|---|---|---|---|
| RankIC | **0.0192** | 0.0345 | 0.0386 |
| ICIR | 0.172 | 0.351 | 0.383 |
| **Spread20** | **17.89 bps** | 73.03 | **82.95** |
| Spread100 | 14.23 | 59.20 | 66.71 |
| PosRatio | 0.566 | 0.631 | 0.665 |
| Days | 663 | 1223 | 1223 |

**xgb_174 is the WORST of the three across every metric**:
- RankIC ≈ 56% of xgb_209
- Sp20 ≈ 24% of xgb_209 — the bps gap is the largest
- Sp100 ≈ 24% of xgb_209
- PosRatio 5pp behind

(Days = 663 vs 1223 reflects different test-window coverage in the 174 cache, not a sample-size advantage. Even pro-rated this doesn't recover the gap.)

### Run cost

- train_lgb retrain: ~10 minutes
- 24-split LOO: ~44 minutes wall clock (2621 s)
- LLM calls: 0 (pure XGBoost)

## Why the archive number lied

The 0.05117 RankIC and 0.646 ICIR in `data/storage/lgb_model_xgb_174.pkl`'s metrics blob came from:

1. **Different end_date** — older training window (2026-05-25 archive vs today's 06-09)
2. **Different cache build commit** — supplementary col set evolved
3. **Different evaluation protocol** — single train/test fold vs 24-split LOO
4. **Different label window definition** — earlier code used a slightly different forward-return shift

ANY of those alone is enough to make a metric incomparable. All four together turn the archive number into pure noise relative to today's same-protocol verdicts.

This is exactly the "评估口径" finding cx round 12 raised earlier. Archive metrics MUST be re-verified under the current protocol before being cited as comparison points.

## Net effect on production

- `lgb_model_xgb_174.pkl` and `production_feature_contract_xgb_174.json` were both rewritten on disk by the train_lgb step. They are NOT in production rotation. xgb_209_chain_llm symlinks were restored before any cron could pick the wrong artifact.
- xgb_174's xgb-only 24-split summary is now archived in `data/storage/experiments/` under tag `xgb_174_freshrun_20260614` for any future reference.
- The "strongest candidate" honor stays with **xgb_209_chain_llm** (B.9 verdict, awaiting shadow validation after Monday's cron chain unblocks).

## Recommendation

**Stop quoting archive metrics across protocol boundaries.** When comparing candidates, the protocol-axis hygiene checklist applies:

1. Same end_date?
2. Same cache build (verify by `data['cache_path']` in experiment artifact)?
3. Same split harness (single fold? 24-split LOO? cross-validation? hold-out?)?
4. Same label expression (the `Ref($close, -N)/ Ref($close, -1) - 1` form)?

If any answer differs, re-run before comparing. The fresh-run cost (~45 min for a 24-split LOO) is far smaller than the cost of a wrong promotion decision.
