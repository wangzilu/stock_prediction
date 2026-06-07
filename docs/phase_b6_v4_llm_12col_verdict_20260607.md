# Phase B.6.2 v4 — xgb_209_llm (12-col L1) vs xgb_209 Final 6-Split Verdict (2026-06-07)

## Summary

The L1 fact-count rebuild flipped Phase B.6 from "LLM hurts (-1.44 bps
Sp20)" to "LLM is RankIC-neutral but lifts top-20 Spread by +6.81 bps".
Promotion is NOT decided on 6-split alone — schedule a 24-split LOO
re-run before any production action.

## Result

Cache: `feature_cache_209_llm.parquet` rebuilt from EventStore source
with L1 fact-count cols (12 LLM cols total = 5 legacy + 7 fact-count).
6-split LOO, end-date 2026-05-19, xgb-only, 500 estimators, early-stop 30.

| Run | Feat | RankIC | ICIR | **Spread20** | Days |
|---|---|---|---|---|---|
| baseline (xgb_209 + 12 LLM) | 221 | +0.0371 | +0.322 | **78.76 bps** | 1114 |
| drop llm_event (= xgb_209) | 209 | +0.0374 | +0.323 | 71.95 bps | 1114 |

| Δ (LLM contribution) | |
|---|---|
| ΔRankIC | −0.0002 (neutral) |
| ΔICIR | −0.002 (neutral) |
| **ΔSpread20** | **+6.81 bps (+9.5%)** |

## Why this differs from B.6 v1

| Phase | LLM cols | ΔRankIC | ΔSpread20 |
|---|---|---|---|
| B.6 v1 (5-col legacy) | impact_1d/5d_decayed + sentiment + count_5d + confidence | −0.0012 | **−1.44 bps** |
| B.6.2 v4 (12-col L1) | 5 legacy + 7 fact-count: positive_3d / negative_3d / price_sensitive_3d / official_3d / count_3d / repeated_ratio_3d / event_intensity | −0.0002 | **+6.81 bps** |

The 7 L1 fact-count columns the project lead's critique asked for
(replace `direction * 0.05` synthesised impact with raw event counts)
materially improve Spread20 without moving RankIC. The classic
explanation: fact counts help the model surface high-conviction
extreme cases (right tail) but don't change the average ranking
quality across the full universe.

The IC backtest at `data/storage/llm_factor_ic_compare/2026-06-06_2126_summary.json`
also shows that EventStore + L1 cols give a NEGATIVE single-factor
sentiment_score IC (-0.0088 vs JSONL +0.0114). Single-factor IC is
not the LOO test, but the negative reading suggests the gain on
Spread20 comes from the fact-count cols, not from sentiment.

## Promotion decision

**Defer to 24-split confirmation.** The promotion gate is `ΔRankIC ≥
+0.005 AND tighter Spread20`. Here ΔRankIC is essentially zero so
the gate is not met. The +6.81 bps Spread20 is on a 6-split fast
screen — a 24-split run could either crystallise the lift or wash
it out as noise on a different split boundary.

Operational plan:

1. Keep production on **xgb_209** today (Monday cron will serve it
   without any further action — the default flip landed in
   commit `aec520d`).
2. Schedule **B.6.3 — xgb_209_llm 24-split** with the new 12-col
   cache (~35 min compute). Compare against the existing xgb_209
   24-split ledger row.
3. If 24-split ΔSpread20 stays above +5 bps AND ΔRankIC stays
   neutral (between −0.001 and +0.001), promote xgb_209_llm AND
   re-run shadow paper-trade against xgb_209.
4. If 24-split ΔSpread20 < +2 bps, file LLM as permanently shadow
   on current architecture and pivot to upstream pipeline work
   (#138 SC-A2 LLM relation extraction, #166 PE-1 step 2/3 already
   landed today).

## Caveats

- **Single-window evidence** (end-date 2026-05-19). The LLM event
  factors only became meaningful after sentiment models matured in
  early 2024, so test window matters more than usual.
- **L1 column quality**: the 7 fact-count cols come from
  `is_price_sensitive` / `is_official_disclosure` / `is_repeated_news`
  flags the LLM extractor sets. If those flags drift, the next
  rebuild's result could be quite different.
- **EventStore single-factor IC was negative** at -0.0088 for
  sentiment_score. The Spread20 lift here comes from the fact-count
  cols, not from sentiment per se. A factor-level decomposition
  (which of the 7 fact-count cols carries the lift) would help
  before locking the conclusion in.

## Provenance

- Wrapper: `scripts/run_phase_b6_llm_ablation.sh`
- Cache: `feature_cache_209_llm.parquet` (3.64 GB, 6,027,907 rows × 223 cols)
- Ledger rows (end-date 2026-05-19):
  - baseline (12 LLM): `xgb_6split_*` recorded 2026-06-07T10:07
  - drop llm_event: `xgb_6split_loo_llm_event_*` recorded 2026-06-07T10:20
- Manifest: `data/storage/supp_col_manifest.json` group `llm_event`
  = 12 cols (5 legacy + 7 fact-count).
- L1 fact-count rebuild: `data/storage/llm_event_factors.parquet`
  (189,840 rows × 14 cols, 12 numeric).
