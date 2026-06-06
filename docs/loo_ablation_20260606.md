# LOO 6-split ablation — 9 groups

- Generated: 2026-06-06T12:37:08
- Baseline (xgb_242 full, no drop) RankIC = **+0.0309**
- Baseline experiment: `xgb_6split_20260606_104415`
- Baseline commit: `1852bf4`
- Baseline cache: `feature_cache_242_production.parquet`

## Ranked by ΔRankIC (best LOO first = group that hurts most when present)

| Drop group | LOO RankIC | ΔRankIC vs baseline | Spread20 (bps) | Days | exp_id |
|---|---|---|---|---|---|
| `cross_market_regime` | +0.0372 | +0.0063 ✅ | +75.74 | 1114 | `xgb_6split_loo_cross_market_regime_20260606_123104` |
| `capital_flow` | +0.0361 | +0.0052 ✅ | +76.70 | 1079 | `xgb_6split_loo_capital_flow_20260606_110316` |
| `shareholder` | +0.0345 | +0.0036 ✅ | +55.29 | 1079 | `xgb_6split_loo_shareholder_20260606_112226` |
| `macro_zero_baseline` | +0.0280 | -0.0030  | +42.93 | 960 | `xgb_6split_loo_macro_zero_baseline_20260606_111127` |
| `st_holder_number` | +0.0273 | -0.0036  | +64.78 | 1042 | `xgb_6split_loo_st_holder_number_20260606_121916` |
| `valuation` | +0.0265 | -0.0044  | +24.92 | 926 | `xgb_6split_loo_valuation_20260606_113224` |
| `st_daily_basic` | +0.0265 | -0.0044  | +24.92 | 926 | `xgb_6split_loo_st_daily_basic_20260606_115455` |
| `quality` | +0.0246 | -0.0063 ⚠️ | +42.47 | 1041 | `xgb_6split_loo_quality_20260606_114343` |
| `st_moneyflow` | +0.0218 | -0.0091 ⚠️ | +54.02 | 1063 | `xgb_6split_loo_st_moneyflow_20260606_120809` |

## Interpretation guide

- **ΔRankIC > 0** (`✅`): dropping the group helped. The group is a net-negative loader and a candidate for removal from `PRODUCTION_SUPPLEMENTARY_GROUPS`.
- **ΔRankIC ≈ 0** (no symbol): the group is essentially noise; dropping it costs nothing but does not help either. Phase B.2 should 24-split confirm before keeping it.
- **ΔRankIC < −0.005** (`⚠️`): dropping the group hurt. The group carries real signal — keep.

All ΔRankIC values are on a 6-split FAST screen. A larger 24-split is required before changing PRODUCTION_SUPPLEMENTARY_GROUPS for real (Phase B.2).