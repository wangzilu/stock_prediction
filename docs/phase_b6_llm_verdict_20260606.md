# Phase B.6 — xgb_209_llm vs xgb_209 LLM Ablation Verdict (2026-06-06)

## Summary

**The current LLM event factor group makes the model slightly worse.**
On a 6-split fast-screen LOO at end-date 2026-05-19, adding the 5
LLM-derived columns to xgb_209 produces ΔRankIC −0.0012 and ΔSpread20
−1.44 bps. Not catastrophic, but not earning a production slot.
The promotion gate (ΔRankIC ≥ +0.005) is not met.

**Action**: Keep LLM factors in shadow. Do NOT promote `xgb_209_llm`
to production today. Focus on the upstream pipeline quality fixes
that the project lead's 2026-06-06 critique flagged:
fact-count rebuild (L1), EventStore canonicalisation (LLM L2),
source quality (sentiment / supply-chain published_at).

## 6-split table

Cache: `feature_cache_209_llm.parquet` (216 cols incl label + 5 LLM).
End-date: 2026-05-19. xgb-only, 500 estimators, early-stop 30.

| Run | Feat | RankIC | ICIR | Spread20 | Days |
|---|---|---|---|---|---|
| baseline (xgb_209 + LLM, full 214) | 214 | +0.0362 | +0.310 | 70.51 bps | 1114 |
| drop llm_event (= xgb_209) | 209 | +0.0374 | +0.323 | 71.95 bps | 1114 |

| Δ (LLM contribution) | |
|---|---|
| ΔRankIC | **−0.0012** |
| ΔICIR | −0.013 |
| ΔSpread20 | **−1.44 bps** |

Negative on every metric. LLM cost > LLM benefit on this window.

## Why this matches the project lead's critique

The 2026-06-06 data-pipeline critique called out three reasons the
LLM chain was not ready:

1. **JSONL was the default source** (mis-PIT relative to EventStore).
   Fixed in commit `0c3d6b8`, but the existing
   `llm_event_factors.parquet` was generated from the JSONL path
   before the fix, so this LOO ran on the older snapshot. A clean
   re-run after a rebuild from EventStore could shift the verdict.
2. **L1 fact-count columns are not yet on disk.** The cache joined
   the 5 legacy cols (impact_1d_decayed, impact_5d_decayed,
   sentiment_score, event_count_5d, avg_confidence). The new L1
   columns (positive/negative/price_sensitive/official counts,
   intensity, repeated_ratio) require a rebuild that runs
   `build_llm_event_factors` after the eventstore bool.astype fix
   in `48243dd`. The 7 fact-count columns are exactly the kind of
   high-signal features the critique recommended.
3. **Sentiment + guba + supply-chain factors are not in the model**
   at all (task #164 / #165), so the LLM event group is competing
   against ALL the noise without the rest of the alternative-signal
   stack to help it.

So a negative ΔRankIC on the current snapshot doesn't mean LLM is
worthless — it means the snapshot is not yet representative. A
proper re-ablation after the upstream fixes lands is filed as
follow-up.

## Recommended next steps

1. **Production action today**: flip the `PRODUCTION_MODEL_PROFILE`
   default from `xgb_242` to `xgb_209`. xgb_209 has 24-split
   evidence (Phase B.4 verdict) and a freshly retrained `.pkl`
   from data through 2026-06-05. Monday's cron should serve it.
2. **Keep LLM in shadow**. `xgb_209_llm` candidate profile stays in
   `config/production_features.py` for future LOO re-runs but is
   not loaded by default. Task #163 closes with this verdict.
3. **Upstream pipeline work** (from the 2026-06-06 critique):
   - P1 #3 follow-up: rebuild `llm_event_factors.parquet` from
     EventStore + L1 schema, then re-run B.6 — different evidence
     could promote xgb_209_llm next round.
   - P1 #4 (#164): build sentiment + guba factors into proper
     parquets, fix the docstring lie.
   - P1 #5 (#165): supply chain `published_at` instead of
     `target_date` — the cleanest single-file fix.
   - PE-1 steps 2 + 3 (#166): finish the PBOC chain so monetary
     policy can join the model as a regime input.

## Caveats

- **6-split is a fast screen.** A 24-split confirmation is needed
  before declaring LLM permanently shadow. But spending another
  ~35 min today on a 24-split for a profile that already lost on
  6-split is poor allocation; do it after one of the upstream
  fixes lands.
- **Single-window evidence.** End-date 2026-05-19. The 5 LLM cols
  could plausibly help on a different window (e.g. earnings season
  with high event density). Walk-forward LOO would resolve this.
- **The LLM cache snapshot is 2 weeks stale.** Rebuilding
  `llm_event_factors.parquet` after today's two bug fixes
  (`0c3d6b8` source flip + `48243dd` bool.astype) is the first
  step before claiming LLM is permanently negative.

## Provenance

- Wrapper: `scripts/run_phase_b6_llm_ablation.sh`
- Cache build: `scripts/build_feature_cache_209_llm.py`
- Ledger rows (end-date 2026-05-19, 6-split):
  - baseline (full 214): logged with `dropped_groups=[]`,
    feature_count=214
  - drop_llm_event (= 209): logged with
    `dropped_groups=['llm_event']`, feature_count=209
- Wall-clock: 28 min total (baseline 14 min, drop 14 min).
