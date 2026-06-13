# Phase B.9 Shadow Monitoring Plan

**Created**: 2026-06-13 Saturday
**Author**: Claude  
**Context**: B.9 production swap (xgb_209 → xgb_209_chain_llm) was committed in `config/production_features.py` on 2026-06-10 00:42 but has NOT actually served any picks yet because `champion_cache_rebuild` was blocked 4 trading days running by stale qlib_data_update.

## The reality so far (06-10 → 06-13)

| Day | morning_recommendation push? | Predictions cache backing it | Real served model |
|---|---|---|---|
| 06-10 | ✅ FORCE_CACHE mode | lgb_latest_predictions.json from 06-07 22:16 | **xgb_209** |
| 06-11 | ✅ FORCE_CACHE | same stale cache | **xgb_209** |
| 06-12 | ✅ FORCE_CACHE | same stale cache | **xgb_209** |
| 06-13 (Sat) | n/a — market closed | — | — |

So the "promotion" has been a paper exercise. Users got xgb_209 picks the entire window. This is actually GOOD news: we didn't ship the 20d Test RankIC -0.0261 yellow flag without knowing it.

## 2026-06-13 attempt to manually activate

Tonight (Sat) I ran `build_champion_cache.py --end-date 2026-06-09` successfully (12 min). All 3 feature caches refreshed (242 → 209 → 209_llm), but **NOT** the chain_llm cache (`feature_cache_209_chain_llm.parquet` still dated 06-07).

Then ran `smoke_lgb_predict.py` — REFUSED by training gate:

```
Critical sources stale/missing: [
  'qlib_data_update', 'fund_flow_update', 'valuation_update',
  'regime_daily_update', 'northbound_update', 'st_daily_basic_update',
  'st_moneyflow_update', 'global_chain_factors_llm'
]
expected latest_date >= 2026-06-12
```

The gate is doing its job — it will NOT let stale features drive new predictions. Conclusion: we cannot manually activate shadow tonight. Must wait for upstream data to refresh.

## What turns shadow ON

The shadow window starts the first day after ALL three things land:

1. `qlib_data_update` success — depends on 4 fixes I shipped today:
   - `b2d7719` ST consecutive-daily fail-fast
   - `5179422` ST cap 30d + cross-endpoint counter
   - `86a76db` baostock cap 30d
   - `0365daa` ST skip-and-continue (current strategy)
2. `champion_cache_rebuild` success — runs Mon-Fri 18:30, depends on (1)
3. `lgb_after_close_smoke` success — runs Mon-Fri 18:35, depends on (2), writes `lgb_latest_predictions.json` with xgb_209_chain_llm model + fresh `latest_date`

Realistically: **Monday 2026-06-15 18:35** is the earliest. First **Tuesday 09:20** morning_recommendation will serve xgb_209_chain_llm picks.

## Shadow window definition

**Start**: First trading day where `lgb_latest_predictions.json.model_path` resolves to `lgb_model_xgb_209_chain_llm.pkl` AND `latest_date` advances each day.

**Duration**: 5-10 trading days of realized Sp20 collection.

**Tracking metric**: realized Sp20 = (top-20 picks 5d forward return) − (bottom-20 picks 5d forward return), per day.

**Compare against**: B.9 LOO simulated baseline: ΔSp20 +9.92 bps over xgb_209. If realized ΔSp20 stays consistent ≥ +5 bps over 5+ trading days, candidate clears the Sp20-aware promotion bar.

## Decision rules (from layered gate policy)

| Outcome over 5-10 trading days | Action |
|---|---|
| Realized ΔSp20 ≥ +5 bps consistently | Stay on xgb_209_chain_llm. Canary tier passes. |
| Realized ΔSp20 ∈ [0, +5) bps | Extend monitoring to 15-20 trading days. Marginal. |
| Realized ΔSp20 < 0 consistently | **Rollback** by setting `PRODUCTION_MODEL_PROFILE=xgb_209` in cron env. The xgb_209 artifact + contract stay on disk for one-command rollback. |
| Mixed signals (single big-loss day + neutral rest) | Investigate which day's pick caused the drag; do not auto-decide. |

## Files / commands relevant to monitoring

- **Track served model each day**: 
  ```
  jq '.model_path' data/storage/lgb_latest_predictions.json
  jq '.latest_date' data/storage/lgb_latest_predictions.json
  ```
- **Daily morning push picks**: stored by `verifier` somewhere (need to locate — TODO link to that file)
- **5d forward returns**: realized via qlib calendar advancing — `data/storage/qlib_data/cn_data/calendars/day.txt`
- **One-command rollback**:
  ```
  PRODUCTION_MODEL_PROFILE=xgb_209 # patch cron via install_crontab or env_vars on CronJob
  ```

## Open follow-ups

- `feature_cache_209_chain_llm.parquet` still dates 06-07. Production inference uses live FeatureMerger so this stale train-time cache is not a runtime hazard; flag for the next weekly_full_retrain to refresh.
- Locate the verifier morning-pick log path and add automated daily Sp20 extraction script under `scripts/track_b9_shadow_realized.py`.
- After 5 trading days, write `docs/b9_shadow_realized_<date>.md` with the verdict and either GO/NO-GO on canary.

## Memory hook

The "production swap silently no-ops when champion_cache_rebuild blocks" pattern is worth promoting from this incident to a generic [[feedback_layered_promotion_gate]] entry: any future production-default flip must be paired with a manual one-shot champion_cache_rebuild verification before assuming the swap is live.
