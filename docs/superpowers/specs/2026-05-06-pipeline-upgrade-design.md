# Pipeline Upgrade Design — LGB Integration + RL + Multi-Push Schedule

**Date:** 2026-05-06

## 1. Data Update & Model Retrain

- `update_qlib_data.py`: Run with conda tianshou env, update to today
- `train_lgb.py`: Dynamic dates — train = 5 years to 3 months ago, valid = 3 months to 1 month ago, test = 1 month to today. Remove hardcoded dates.

## 2. LightGBM Integration into Pipeline

**Problem:** `jobs.py` uses `change_pct / 10` as `short_score` instead of real model predictions.

**Solution:**
- Load `lgb_model.pkl` at pipeline startup
- Run Alpha158 inference on candidate stocks to get 5-day forward return predictions
- Replace `short_score` with LGB prediction value
- Fallback to `change_pct` for crypto/gold (no Qlib features)

## 3. RL Agent — Transformer + SAC

| Item | Design |
|------|--------|
| Framework | tianshou 2.0 + gymnasium |
| Algorithm | SAC (off-policy, auto entropy tuning) |
| Network | Transformer encoder (4 layers, 8 heads, d=128) → MLP → action |
| State | Alpha158 (158d) + 20-day price series (OHLCV 5×20=100d) + position (1d) + sentiment (1d) = ~260d |
| Action | Discrete 3: buy / hold / sell |
| Reward | `r = return - λ * max(drawdown, 0)`, λ=2.0 drawdown penalty |
| Training data | Qlib binary data, per-stock episodes |
| Output | `data/storage/rl_model.pt` |

## 4. Sell Check Logic (14:30)

Rules + model hybrid:
- **Take profit:** gain ≥ 8% since recommendation → sell
- **Stop loss:** loss ≥ 5% since recommendation → sell
- **LGB flip:** current LGB prediction < -0.02 (bullish → bearish) → sell
- Any trigger → push sell suggestion

## 5. Push Schedule

| Time | Job | Content |
|------|-----|---------|
| 9:20 | `run_morning_recommendation` | LGB + RL pre-market recommendations |
| 14:30 | `run_sell_check` | Take-profit/stop-loss + LGB flip detection |
| 15:30 | `run_daily_summary` | Market close summary (LLM generated) |
| 22:00 | `run_evening_outlook` | Next-day outlook (macro + model predictions) |

Replaces old 14:00 recommendation and 14:05 verification schedule. Verification merged into 15:30 summary.
