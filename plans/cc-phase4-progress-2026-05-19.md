# Phase 4 进展总结

**日期：** 2026-05-19
**作者：** CC

---

## Track 完成状态

| Track | 状态 | 关键结果 |
|:---:|:---:|------|
| 4A | ✅ PASS | 24-split: avg RankIC +0.051, Spread +2.51%, 87.5% Spread>0 |
| 4B | ✅ PASS | buffered_partial: 8/12 正, avg Sharpe +2.36 |
| 4C | ✅ PASS | 单票 5%, 行业最大 30%(<40%), 容量 OK |
| 4D | 待做 | Champion/shadow 治理 |
| 4E | 待做 | Alpha360 对照 |
| 4G | ✅ 验证完成 | 恒指/纳指 regime: RankIC +30%↑, 71% split 改善 |
| 4H | ✅ 验证完成 | MA timing: avg 年化+61%, Sharpe +2.0, 67% 胜率 |
| 4F | 待做 | Paper trading |

---

## 4G Cross-Market Regime 消融（24 splits）

| 指标 | 178 维 (base) | 205 维 (+regime) | 改善 |
|------|:---:|:---:|:---:|
| avg RankIC | +0.054 | +0.070 | **+30%** |
| avg Spread | +2.18% | +2.41% | +10% |
| RankIC>0 | 20/24 | 22/24 | 更稳定 |
| Δ RankIC>0 | — | 17/24 (71%) | 显著 |
| Δ Spread>0 | — | 13/24 (54%) | 中等 |

**结论：恒指/纳指 regime 信号有增量（avg RankIC +30%），但最近 split 反而恶化（+0.006→-0.082）。
建议 XGB 205 进入 shadow candidate，不直接升级 champion。需先跑 20 日 shadow + 成本后回测。**

Regime 特征包括（每个指数 9 个）：
- 1d/5d/20d return
- 5d/20d volatility
- 5d/20d momentum (close vs MA)
- 10-day up ratio (RSI proxy)
- 20-day drawdown from high

---

## 4H MA Timing Rolling（12 splits）

| 指标 | 值 |
|------|:---:|
| avg 年化 | +61.2% |
| avg Sharpe | +1.995 |
| 年化>0 | 8/12 (67%) |

策略逻辑（来自 lzz）：
- 入场：XGB Top50 候选 + 贴着 5MA（2%以内）+ 在 20MA 上方
- 止损：跌破 20MA
- 止盈：涨 20% 或远离 5MA 8%+

**结论：MA timing 有择时能力但波动极大（中位年化仅+13%，+683%极端拉高，最近split -48%）。
定位：research_only / 辅助过滤，不进主链。**

---

## 模型状态台账

| 模型 | 状态 | 说明 |
|------|:---:|------|
| XGB 174 | **champion** | 不动，继续生产 |
| XGB 205 + buffered_partial | **shadow candidate** | 等 20 日 shadow + 4D gate 全过才能谈 champion |
| MA timing | research_only | 辅助过滤，不进主链 |

**治理顺序：shadow 20 日 → 4D gate check → 成本后回测 → 暴露检查 → 才能 promote。
不允许跳过任何步骤直接升级 champion。**

---

## 工程成果

- feature_cache 预计算：600 万行 × 207 列，3.8GB，一次构建多次复用
- cross-market regime broadcast 向量化：20 分钟 → 2 秒
- fast_rolling_gate：24 splits 在 33 分钟完成（原来要 5+ 小时）
- daily return / label 口径分离，防呆检查
- crontab 时间修复（17:00→17:45）
- 行业映射表（5523 股票 × 110 行业）
