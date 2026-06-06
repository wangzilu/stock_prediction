# Phase B.5 — Bucket B 24-Split LOO Verdict (2026-06-06)

## Summary

**Bucket B groups are all neutral.** None of the four Phase B 6-split
LOO marginal groups (macro_zero_baseline / st_holder_number /
valuation / st_daily_basic) show a clear DROP signal at 24-split.
Stay with **xgb_209** as the production champion — do NOT extend the
Phase B Bucket A drop set with any Bucket B candidates.

## 24-split table

Baseline: xgb_242 24-split full (rank_ic +0.0273, ICIR +0.250,
spread_top20 35.01 bps, days 820).

| Drop group | Feat | RankIC | ΔRIC | ICIR | Spread20 | ΔSp20 | Days |
|---|---|---|---|---|---|---|---|
| valuation | 235 | +0.0321 | +0.0048 | +0.307 | 43.96 bps | +9.0 | 777 |
| st_daily_basic | 235 | +0.0321 | +0.0048 | +0.307 | 43.96 bps | +9.0 | 777 |
| macro_zero_baseline | 232 | +0.0314 | +0.0041 | +0.300 | 42.86 bps | +7.9 | 749 |
| st_holder_number | 241 | +0.0266 | −0.0007 | +0.268 | 46.07 bps | +11.1 | 848 |

All ΔRankIC are within [-0.001, +0.005] — the 6-split B.5 "neutral"
classification holds at 24-split with stronger statistics.

## What this means relative to xgb_209

xgb_209 24-split RankIC was +0.0345 (Spread20 73.03 bps). The largest
Bucket-B single-drop gain at 24-split is only +0.0048 RankIC and
+11.1 bps Spread20 — far below the xgb_209 lift over xgb_242
(Δ +0.0072 RankIC, +38 bps Spread20). The Bucket A drop is doing all
the heavy lifting; the marginal Bucket B groups behave roughly like
zero-mean noise.

## Important observations

1. **valuation and st_daily_basic produced IDENTICAL metrics**
   (+0.0321 RankIC, 43.96 bps Sp20, 777 days). This is suspicious —
   they may share a high-correlation underlying signal (both pull from
   PE/PB/PS columns), or one is downstream of the other in the supp
   loader pipeline. Worth a future investigation but not blocking.
2. **st_holder_number is the ONLY group with ΔRIC < 0** (−0.0007).
   This means dropping it slightly hurts. Keep.
3. **All four groups improved Spread20 when dropped** (+7.9 to
   +11.1 bps), even when RankIC was flat. This is consistent with
   the broader Bucket A finding — many supplementary loaders inject
   noise that hurts the right tail more than the median predictor.

## Future work (NOT for this release)

- **Joint-drop ablation**: pairwise drops of Bucket B groups (e.g.
  drop valuation + macro_zero_baseline) on 24-split. The single-drop
  Δs don't compose linearly so this could either crystallise a real
  effect or confirm noise.
- **valuation vs st_daily_basic identity check**: compare the actual
  feature column overlap between these two loaders. If they're
  duplicating signals, one should be removed as dead code.
- **Walk-forward Bucket B sweep**: the single-window evidence here is
  end-date pinned at 2026-05-19. Re-run on 3-month / 6-month rolling
  windows to confirm none of these drop into the negative as alpha
  decays.

## Recommended next step

1. **Hold xgb_209 as champion**. Do NOT extend the drop set today.
2. **Run shadow paper-trade xgb_209 vs xgb_242** for 5-10 trading
   days (the existing task #162).
3. **Promote xgb_209 to production default** only after shadow
   confirms the 24-split Spread20 lift survives live trading.

## Provenance

- Wrapper: `scripts/run_phase_b5_24split.sh` (cx-style FAILED[] +
  refuse-ALL-DONE on partial)
- Runner commit: `eed1509` (comma-separated --drop-group)
- Ledger rows (all end-date 2026-05-19, 24-split, xgb-only,
  500 estimators / early-stop 30):
  - `xgb_24split_loo_macro_zero_baseline_20260606_161144`
  - `xgb_24split_loo_st_holder_number_20260606_165536`
  - `xgb_24split_loo_valuation_20260606_173103`
  - `xgb_24split_loo_st_daily_basic_20260606_181230`
- Baseline: `xgb_242_24split_20260605_235458`
- Phase B 6-split LOO predicted neutral/marginal: confirmed at
  24-split. The 6-split fast screen was accurate.

## Sign-off

Phase B.5 gate: complete (4/4 runs rc=0). No production action
required beyond the existing Phase B.4 xgb_209 promotion.
