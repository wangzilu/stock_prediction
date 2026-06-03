# fix/train-lgb-use-feature-merger — PR Self-Review

Date: 2026-06-03
Branch: `fix/train-lgb-use-feature-merger` (1 commit ahead of `master`)
Commit: `3d8fa2b fix: train_lgb applies merge_for_training preprocessing (cx review round 3 P1)`
Tests: 4 new + adjacent passing
Merge gate: **awaiting backtest verification** — see "Why this PR is held" below

## What this PR does

Fixes a production / research distribution drift bug. `scripts/train_lgb.py`
was calling `FeatureMerger._load_supplementary()` and injecting RAW
values into Qlib internals, skipping the per-day rank percentile
that `merge_for_training` applies by default. Production champion
therefore trained on different distributions than research / cache
for the same column names.

The fix: insert
`merger._preprocess_supplementary(supp, raw_data.index, mode="rank")`
before the injection loop.

## Files changed

```
scripts/train_lgb.py                     |   8 ++ (inline preprocess + comment)
tests/test_train_lgb_uses_preprocessing.py | 163 ++ (4 tests, new file)
```

## Why this PR is held (NOT for immediate merge)

This change **alters the production champion's training distribution**.
Same column names, but values shift from raw scales (e.g. PE in 5-5000
or fund flow in [-1e9, 1e9]) to per-day rank percentiles in [0, 1].

XGB / LGB are tree-based and rank-invariant within a single feature,
but their behaviour can differ when:

- A column previously had skewed raw values that informed split
  thresholds; after rank-normalisation those thresholds are uniform.
- Interactions between features that previously had different scales
  may shift importance.
- The model's `n_estimators=500` was tuned on the raw-scale champion.

Required before merge:
1. Run a backtest comparison: previous champion vs new training on
   the same train / valid / test split, both predicting the same
   period.
2. Verify IC / spread doesn't degrade (allow 10% regression
   tolerance — this is fixing a bug, not optimising).
3. Smoke test: `python scripts/smoke_lgb_predict.py` against a fresh
   model trained from this branch passes the existing IC gate.

The user's earlier guidance was explicit:
> 不推荐 train_lgb FeatureMerger 当第一个动：会改 champion 模型的训练
> 分布，要先验完整 backtest 才能上

This PR exists so the fix is auditable now; merge timing is held for
the backtest signal.

## Tests

```
tests/test_train_lgb_uses_preprocessing.py
├── test_rank_mode_per_day_unit_interval         ← behaviour
├── test_raw_mode_does_not_normalise             ← counter-test
├── test_train_lgb_source_calls_preprocess_supplementary  ← anti-regression
└── test_train_lgb_preprocess_call_happens_before_injection ← order check
```

The two source-level tests are intentionally brittle: if a future PR
removes the preprocess call (e.g. to "restore prior behaviour"), CI
fails before production drifts.

## Downstream audit

- `models/feature_merger.py:_preprocess_supplementary(mode="rank")`:
  per-day rank percentile, column names preserved. No suffix added.
  So nothing else needs to change (the Qlib injection loop iterates
  over `supp.columns` and works as before).
- The macro drop from PR `233fcd2` is upstream — `supp` no longer
  contains `macro_*` columns, so this preprocessing step has fewer
  inputs than before that fix landed. No interaction concern.
- `tests/test_macro_pit_drop.py` continues to pass on this branch.

## Suggested next steps

1. Run a backtest using a model trained from this branch on a recent
   60-day window vs the current production champion. Compare IC /
   spread / drawdown.
2. If results within tolerance: merge to master. Next midweek_train
   (Wed 18:15) or weekly_full_retrain (Sat 04:00) picks up the fix.
3. If results degrade beyond tolerance: investigate before merging —
   the preprocessing default `mode="rank"` may not be optimal for
   tree-based models, even if it's the correct semantic match.

## Follow-ups (NOT in this PR)

- **Batch 3 step C** (P2): ST mask historisation. Requires ST_CLIENT
  historical API.
- **Batch 4 sqrt_adv** (Task #79): wires sqrt_adv into portfolio
  backtest + paper OMS. Prerequisite for Crypto-B.
