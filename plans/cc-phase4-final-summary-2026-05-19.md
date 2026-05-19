# Phase 4 Final Summary

**Date:** 2026-05-19
**Author:** CC

---

## 1. Phase 4 Overview

All tracks completed. Champion model confirmed, paper trading live, infrastructure hardened.

| Track | Name | Status | Verdict |
|:---:|------|:---:|------|
| 4A | Rolling Gate | PASS | 24-split validation, all 5 gates clear |
| 4B | Execution Strategy | PASS | buffered_partial best candidate, shadow |
| 4C | Exposure Check | PASS | Stock 5%, industry 30% |
| 4D | Model Governance | DONE | xgb_174 champion, xgb_205 research_only |
| 4E | Alpha360 Contrast | DONE | No help for tree models |
| 4F | Paper Trading | LIVE | 18:42 daily crontab |
| 4G | Regime Signals | DONE | Regime controller, not stock selection |
| 4H | MA Timing | DONE | research_only |

---

## 2. Track A: 24-Split Rolling Gate

**Result: ALL GATES PASS**

| Gate | Value | Threshold | Pass |
|------|:---:|:---:|:---:|
| avg RankIC | +0.0513 | >=0.04 | Yes |
| avg Spread | +2.51% | >=1.2% | Yes |
| RankIC>0 ratio | 83.3% (20/24) | >=65% | Yes |
| Spread>0 ratio | 87.5% (21/24) | >=65% | Yes |
| Worst 20% avg Spread | -0.67% | >-1.5% | Yes |

Regime breakdown: alpha stable across all market environments. Bear-market Spread highest (+3.26%).

---

## 3. Track B: Execution Strategy

**Result: buffered_partial is best, but median annual return only +8.7%. Shadow candidate.**

| Strategy | avg Annual | avg Sharpe | Annual>0 |
|------|:---:|:---:|:---:|
| daily_rebal | +65.2% | +1.155 | 50% (6/12) |
| buffered_partial | +82.9% | +2.364 | 67% (8/12) |
| buffered+stop8% | +15.0% | -0.094 | 33% (4/12) |

Key caveats:
- avg inflated by extreme splits (+280%, +323%, +426%)
- Median annual return: **+8.7%**
- Stop-loss 8% too tight for A-share volatility

Production parameters: top_k=20, buffer=5, trade_rate=0.35, min_hold=2d, max_turnover=15%/day, vol_threshold=1.5.

---

## 4. Track C: Exposure Check

**Result: PASS**

| Constraint | Limit | Actual | Pass |
|------|:---:|:---:|:---:|
| Single stock | 5% | <=5% | Yes |
| Max industry | 40% | 30% | Yes |
| Capacity | sufficient | OK | Yes |

---

## 5. Track D: Model Governance

| Model | Status | Notes |
|------|:---:|------|
| xgb_174 | **champion** | Production model, 174 features |
| xgb_205 | research_only | Downgraded: regime neg-ctrl failed |

Registry: `data/storage/phase4/model_registry.json`. Shadow slot is null.

Governance rule: shadow 20d -> gate check -> cost backtest -> exposure check -> promote. No step may be skipped.

---

## 6. Track 4E: Alpha360 Contrast

Alpha360 feature set provided no incremental value for XGBoost tree models. The 174-feature hand-crafted set remains superior.

---

## 7. Track 4F: Paper Trading

- Crontab: daily at 18:42
- Model: xgb_174 champion
- Execution: buffered_partial
- Status: **running**

---

## 8. Track 4G: Regime (Cross-Market Signals)

### Pre-fix results (look-ahead bias present)

| Metric | 178-dim base | 205-dim +regime | Change |
|------|:---:|:---:|:---:|
| avg RankIC | +0.054 | +0.070 | +30% |
| avg Spread | +2.18% | +2.41% | +10% |

### Negative control results (post-fix, 8 splits)

| Control | avg RankIC | real > ctrl | Gate (>=70%) |
|------|:---:|:---:|:---:|
| real | +0.049 | -- | -- |
| date_shuffle | +0.034 | 75% | PASS |
| circular_60d | +0.030 | 62% | FAIL |
| circular_120d | +0.032 | 75% | PASS |
| circular_250d | +0.024 | 62% | FAIL |
| future_1d | +0.042 | 75% | PASS |
| **future_5d** | **+0.086** | **38%** | **FAIL** |
| future_20d | +0.041 | 50% | FAIL |

**Critical finding:** future_5d regime (+0.086) beat real regime (+0.049). This means HSI/NASDAQ are synchronous with A-shares, not leading indicators. The look-ahead version was the strongest signal.

**Conclusion:** RankIC gate 75% pass (3/4 controls), but Spread gate only 42% pass. Regime repositioned as risk/position controller, not stock selection feature. xgb_205 downgraded to research_only.

---

## 9. Track 4H: MA Timing

| Metric | Value |
|------|:---:|
| avg Annual | +61.2% |
| avg Sharpe | +1.995 |
| Median Annual | +13% |
| Annual>0 | 67% (8/12) |
| Worst split | -48% |

Extreme variance (one split +683% inflates average). **Status: research_only.** Useful as auxiliary filter, not main pipeline.

---

## 10. Phase 2 Factor Ablation Summary

**ALL new factors failed to beat the 174-feature baseline in rolling ablation.**

| Factor Group | RankIC Gate (>=70%) | Spread Gate | Verdict |
|------|:---:|:---:|------|
| regime (HSI/NQ) | 75% pass | 42% fail | Regime controller only |
| forecast (binary) | 62% | -- | Stale data (median age 1404d) |
| forecast (content) | 62% | -- | Only 0.3% fresh within 30d |
| moneyflow | <50% | <50% | No incremental value |
| cyq (chip distribution) | <50% | <50% | No incremental value |
| pledge | <50% | <50% | No incremental value |
| block_trade | <50% | <50% | No incremental value |
| top_inst | <50% | <50% | No incremental value |
| holder_num | 67% | residual IC -0.018 | Redundant with existing features |

### PIT (Point-in-Time) Audit

| Config | avg RankIC |
|------|:---:|
| no_flow | +0.035 |
| flow_lag0 (current) | +0.038 |
| flow_lag1 (safe) | +0.043 |
| flow_lag2 | +0.042 |

**174-dim baseline is PIT-safe.** lag0 -> lag1 drops only 12.7%. lag1 actually performs better. NASDAQ look-ahead bias identified and fixed.

---

## 11. Infrastructure Built

| Component | Detail |
|------|------|
| feature_cache | 6M rows x 207 cols, 3.8GB, one-time build |
| fast_rolling_gate | 24 splits in 33 min (was 5+ hours) |
| Cache V2 | Mixed format: parquet + npy sidecar |
| asof_merge vectorized | searchsorted-based, 20min -> 2s |
| model_registry | JSON-based champion/shadow governance |
| shadow inference | Daily automated scoring pipeline |
| unified JSON serializer | Handles numpy/datetime types |
| industry mapping | 5523 stocks x 110 industries |

---

## 12. Current Production Config

| Item | Value |
|------|------|
| Champion model | xgb_174 |
| Execution strategy | buffered_partial |
| Paper trading | Running (18:42 daily crontab) |
| Shadow model | None (xgb_205 downgraded) |
| Flow lag | lag1 (PIT-safe) |

---

## 13. Next Steps

| Priority | Task | Target |
|------|------|------|
| 1 | Paper trading observation | 20 trading days minimum |
| 2 | 4G.2 sector spillover | US sector -> A-share industry mapping |
| 3 | moneyflow/cyq derivatives | Second-order features from raw factors |
| 4 | Phase 5 RL controller | Reinforcement learning for position sizing |
