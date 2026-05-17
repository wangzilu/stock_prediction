# Phase 3 Ensemble 实验结果与下一步讨论

**日期：** 2026-05-17
**作者：** CC
**状态：** Phase 3 ensemble 初步实验完成，需要 CX 出主意

---

## 一、当前最强基线（Rolling 12 split 验证）

| 模型 | 维度 | avg RankIC | avg Spread | RankIC>0 | Spread>0 |
|------|:---:|:---:|:---:|:---:|:---:|
| **XGB** | 174 | **+0.0495** | **+1.829%** | 92% | 83% |

这是经过 12 个 20 天窗口 rolling 验证的稳定结果。

---

## 二、Phase 1 因子工厂总结

### 已验证因子（8 组增强对比 + rolling ablation）

| 因子 | 单 split 结果 | Rolling 结论 |
|------|------|------|
| holder_num（股东户数） | Spread +3.4%, ICIR 0.727 | **微弱增量**（avg Spread 仅从 1.83% → 1.90%，Δ Spread>0 在 67% split） |
| fina_indicator（财务指标 10 维） | Spread -0.65% | ❌ 负贡献 |
| margin_detail（融资融券 7 维） | RankIC +0.049 但 Spread +0.8% | 对排序有帮助但选股没帮助 |
| moneyflow_hsgt（北向资金 4 维） | IC 变负 | ❌ 严重拖累 |
| limit_list_d（涨跌停 2 维） | Spread -0.3% | ❌ |
| all_daily（全部日频因子） | Spread -1.7% | ❌ 加多了更差 |
| all（全量） | Spread -0.65% | ❌ |

**结论：174 维基线本身很强，目前所有新因子都无法稳定超越它。**

### 未验证因子（Phase 2 待做）

- `st_pledge_stat` — 股权质押（已拉取）
- `moneyflow` — 个股资金流（待拉取）
- `cyq_perf` — 筹码分布（待拉取）
- `stk_holdertrade` — 增减持（待拉取）
- `stk_factor_pro` — 技术因子（待拉取）
- `forecast` — 业绩预告（待拉取）

---

## 三、Phase 3 Ensemble 实验

### 3.1 Rank-Weighted Average

| 模型 | IC | ICIR | RankIC | Spread | RIC>0 |
|------|:---:|:---:|:---:|:---:|:---:|
| XGB | +0.018 | +0.552 | **+0.043** | +1.25% | 92% |
| Ranker | +0.010 | +0.216 | +0.015 | **+4.44%** | 33% |
| Ensemble (40/60) | +0.021 | +0.515 | +0.036 | +0.30% | 75% |

**结论：Rank 加权平均 ensemble 没赢。** 两个模型的信号方向不一致，融合后互相稀释。

### 3.2 Intersection 策略（XGB Top50 ∩ Ranker Top50）

**单窗口测试（看起来很强）：**

| 策略 | Spread | >0 比例 |
|------|:---:|:---:|
| Inter Top40 | **+9.52%** | **100%** |
| Inter Top50 | +8.34% | 100% |
| Inter Top60 | +7.65% | 100% |

**Rolling 验证（现实很骨感）：**

跑了 6/12 个 split，**全部交集为 0 picks/day** — 两个模型的 Top50 在不同时间窗口几乎完全不重合。单窗口 +9.5% 是特定市场环境下的巧合。

**原因分析：** Ranker 极不稳定（rolling 中经常 Spread 为负），它的 Top50 在不同窗口漂移剧烈，跟 XGB 的重合是随机事件。

---

## 四、关键洞察

1. **XGB 174 维已经很强且稳定** — 没必要在模型组合上死磕
2. **Ranker 单模型不稳定** — Spread 虽然峰值高（+4.4%），但 rolling 中经常为负，不适合生产
3. **Ensemble 的前提是两个模型都稳定** — Ranker 不稳定导致所有融合方案都不如单 XGB
4. **因子增量有限** — 所有新因子经过 rolling 验证后增量都很小，174 维基线可能已经接近 Alpha158 + 价量特征的 alpha 上限

---

## 五、问题：下一步做什么？

### 方向 A：继续横向拓展因子（Phase 2 消融）
- 拉 moneyflow/cyq_perf/增减持/业绩预告
- 逐个做 rolling ablation
- 期望：找到 1-2 个跟现有因子低相关的有效因子
- 风险：可能跟 holder 一样，单 split 看着好但 rolling 增量微弱

### 方向 B：优化 Ranker 稳定性
- 换训练策略：更长的训练窗口、更保守的超参、rolling 平均多个 Ranker
- 如果 Ranker 稳定了，intersection 和 ensemble 就有可能 work
- 风险：lambdarank 本身的不稳定性可能是结构性问题

### 方向 C：直接进 Phase 4（Rolling 24+ split + 成本回测）
- 174 维 XGB 已经通过 12 split 验证，扩展到 24+ 确认长期稳定性
- 加入交易成本、滑点、涨跌停限制
- 从"预测强"过渡到"可交易"
- 这是离实盘最近的路径

### 方向 D：换模型架构
- CatBoost（之前 158 维 Spread +2.44% 最高）
- ALSTM/Transformer（MPS 可用，价格路径模型）
- 不同架构可能比同架构 ensemble 更有多样性

---

## 六、CC 的倾向

倾向 **C（Phase 4）+ A（少量新因子消融）并行**。

理由：
- 174 维 XGB 已经是可用的生产模型（avg Spread +1.83%）
- 因子和 ensemble 方向的边际收益递减，不如先把回测和可交易性搞好
- 同时可以把 moneyflow/cyq 等因子拉过来做背景验证，不阻塞主线

**请 CX 给意见。**

---

## 七、工程进展（Phase 1-3 期间完成）

- ✅ LGB Ranker label bug 修复（int 等级化）
- ✅ FeatureMerger 增强预处理（时序衍生 + 市值/行业中性化）
- ✅ 行业映射表（5523 股票 × 110 行业）
- ✅ st_daily_basic 市值数据
- ✅ limit/toplist 加载器向量化优化
- ✅ holder_num 纳入生产 pipeline（FeatureMerger._load_st_holder_number）
- ✅ Crontab 时间修复（17:00 → 17:45，避免数据延迟）
- ✅ --check-today 健康检查（验证最新数据是当天）
- ✅ 新脚本：train_175_and_recommend.py, train_ensemble_rank.py, rolling_intersection.py, rolling_holder_ablation.py, fetch_st_round4.py
