# Paper-Trade Head-to-Head: xgb_209_chain_llm vs xgb_242

**Created**: 2026-06-16 | **Task**: #162 | **Ref**: `docs/b9_shadow_monitoring_plan_20260613.md`
**Status**: design only

## 1. Decision: Option B (dual-prediction shadow scoring)

**Pick Option B — extend `b9_shadow_sp20_tracker` to score both profiles; no second PaperOMS.**

- Tracker already snapshots `lgb_latest_predictions.json` daily + computes 5d-fwd Sp20. Adding a second prediction file is ~30 lines; a second OMS doubles `champion_cache_rebuild`, `lgb_after_close_smoke`, `oms_state.json`, and morning-push surface for an unvalidated candidate.
- Sp20 is the binding gate (B.9 + `feedback_layered_promotion_gate`: +0.005 ΔRankIC / +5 bps ΔSp20). PaperOMS PnL adds sqrt_adv + T+1 noise irrelevant to signal quality.
- Pushes read `lgb_latest_predictions.json` only; Option B keeps 242 in a sibling snapshot, eliminating legend confusion.

## 2. Implementation sketch (not built)

Extend `track_b9_shadow_sp20.py` (or fork): (1) run a second inference pass with `xgb_242` model+contract against the same `latest_date` features after smoke; (2) snapshot both to `data/storage/c1_209_vs_242_snapshots/{profile}_{date}.json`; (3) compute §3 metrics once 5d fwd returns land; (4) append to `c1_209_vs_242_running.csv`.

## 3. Comparison metrics (per trading day)

| Metric | Definition |
|---|---|
| **Sp20** | mean(top-20 5d fwd ret) − mean(bot-20 5d fwd ret), per profile |
| **Spread IR** | mean(Sp20_t) / std(Sp20_t) over window |
| **Top-20 overlap** | `|top20_209 ∩ top20_242|` / 20 |
| **Realized turnover** | `|top20_t Δ top20_{t-1}|` / 20, per profile |
| **Max drawdown** | min cumulative `Sp20_242 − Sp20_209` |

Rank-IC is secondary (Sp20 is binding per B.9).

## 4. Decision rule after 10 trading days

| Outcome | Action |
|---|---|
| `ΔIR ≥ +0.3` AND 242 maxDD ≤ 2× 209 | Swap to canary (manual `PRODUCTION_MODEL_PROFILE` flip) |
| 242 maxDD > 2× 209 at any point | Reject, archive |
| ΔIR ∈ [0, +0.3) | Extend to 20 days |
| ΔIR < 0 from day 5 onward | Reject early |
| Top-20 overlap > 0.85 | Models converged → 242 extras redundant; reject |

+0.3 IR floor ≈ Grinold-Kahn sampling-noise bound for n=10.

## 5. Cron integration

**Re-use `b9_shadow_sp20_tracker`** — extend to dual-write, or add sibling `c1_209_vs_242_tracker` at 18:39 (between smoke 18:35 and tracker 18:38). 242 inference reuses the already-loaded feature cache (~4 GB peak, same envelope as smoke). Runtime **2-3 min**, peak RSS **~5 GB**, no contention with 18:38/18:40/18:42 slots. Gated on `lgb_after_close_smoke`.

## 6. Risks

1. **xgb_242 artifact freshness** — model dates 2026-06-03, contract 06-04. By T+10 it'll be 16 trading days older than 209's. Run a one-shot `champion_cache_rebuild` against xgb_242 before launch, else we're scoring the 242 *artifact* not retrained-242. Recommend rebuild Monday, start clock Tuesday.
2. **Legend confusion** — mitigated by Option B (242 never reaches push).
3. **Double PaperOMS state** — eliminated by Option B.
4. **Feature-set asymmetry** — 242 has macro/cross_market cols dropped in B.4. A macro-shock day could flatter 242 on a non-generalizing sample; require regime-tag inspection before swap.
5. **Stale chain_llm cache caveat** — `feature_cache_209_chain_llm.parquet` dates 06-07; production uses live FeatureMerger so not a runtime hazard, but comparison is "stale 242 artifact vs live-merger 209" — document in verdict.

---

## 3-sentence summary

Picked **Option B** (dual-prediction Sp20 scoring inside an extended `b9_shadow_sp20_tracker`) because the binding promotion gate is signal-level Sp20/IR, not OMS PnL, and re-using the tracker avoids a second champion_cache + smoke + PaperOMS for zero incremental information. 242 wins only if Sp20 IR exceeds 209_chain_llm by ≥0.3 **and** maxDD stays under 2× over 10 trading days, else extend or reject per §4. **Biggest risk**: the xgb_242 artifact is already 13 days stale — without a one-shot rebuild before launch, we'd measure "stale 242 vs live 209", silently biasing the verdict against the candidate.
