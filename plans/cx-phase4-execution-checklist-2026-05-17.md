# Phase 4 执行清单：从预测强到可交易

**日期：** 2026-05-17  
**作者：** CX  
**目标：** 把当前 `XGB 174` 基线从“rolling 上有效”推进到“成本后、约束后、可解释、可灰度”的机构化研究状态。  
**适用范围：** 当前项目的日频 A 股选股主线，不包含高频、期货和 RL 交易控制器。

---

## 一、当前共识

基于 [cc-phase3-ensemble-results-2026-05-17.md](/Users/wangzilu/MyProjects/stockPrediction/plans/cc-phase3-ensemble-results-2026-05-17.md:1) 和现有代码状态，Phase 4 的统一口径如下：

1. `XGB 174` 是当前主基线，先按 champion 对待。
2. `175 + holder` 仍有研究价值，但 rolling 增量有限，先做 shadow，不直接定义为生产最优。
3. `Ranker` 暂不作为主模型，只保留一条受控实验线：`XGB 选池 -> Ranker rerank`。
4. 现在最缺的不是新模型，而是交易层、组合层、研究治理层。

---

## 二、Phase 4 总目标

Phase 4 不再回答“模型分数高不高”，而是回答下面 5 个问题：

1. 这个 alpha 在 `24+` 个 rolling split 下是否稳定。
2. 加入真实交易约束后，组合是否仍然赚钱。
3. 成本、换手、容量会不会把纸面收益吃掉。
4. 收益是否来自可解释的选股能力，而不是行业/风格漂移。
5. 模型是否达到进入 shadow 或生产的统一门槛。

---

## 三、主线与非主线

### 主线

- Champion: `XGB 174`
- Shadow: `XGB 175 holder`
- 受控增强实验: `XGB topN -> Ranker rerank`

### 暂缓

- naive rank average ensemble
- `Top50 ∩ Top50` 交集策略
- ALSTM / Transformer / RL 直接进入主线
- 大规模扩因子但不做 rolling gate

---

## 四、工作流拆分

把 Phase 4 拆成 4 条线并行，但只有一条主链：

### Track A：研究验证升级

目标：把当前 `12 split` 升级为正式 promotion gate。

涉及文件：

- [scripts/rolling_train.py](/Users/wangzilu/MyProjects/stockPrediction/scripts/rolling_train.py:39)
- [models/model_registry.py](/Users/wangzilu/MyProjects/stockPrediction/models/model_registry.py:1)
- 新增建议：`scripts/phase4_rolling_report.py`

任务：

1. 把 rolling 评估从 `12 split` 升级到 `24/36 split`。
2. 切窗从自然日改为交易日，避免不同 split 有效样本漂移。
3. 对比训练窗口：
   - `2 年`
   - `3 年`
   - `5 年`
   - `expanding`
4. 输出分市场状态结果：
   - 强势市场
   - 弱势市场
   - 高波动
   - 低波动
5. registry 记录每次 run 的：
   - 特征版本
   - 模型版本
   - 训练窗口
   - rolling aggregate
   - regime breakdown

交付物：

- `rolling_summary.json`
- `rolling_splits.csv`
- `rolling_regime_breakdown.csv`
- registry entry

验收门槛：

- `avg RankIC >= 0.035`
- `avg Spread >= 1.20%`
- `RankIC > 0` 的 split 占比 `>= 65%`
- `Spread > 0` 的 split 占比 `>= 65%`
- 最差 `20%` split 不出现明显失控，建议门槛：`avg Spread > -1.50%`

说明：

- 这些门槛不是“最终收益目标”，而是进入组合回测层的准入门槛。
- 如果 `XGB 175` 达不到这些门槛，只保留为 shadow。

### Track B：组合与回测升级

目标：把当前“TopK 排序表现”变成“可执行组合表现”。

涉及文件：

- [scripts/backtest_qlib_signal.py](/Users/wangzilu/MyProjects/stockPrediction/scripts/backtest_qlib_signal.py:42)
- [models/portfolio_policy.py](/Users/wangzilu/MyProjects/stockPrediction/models/portfolio_policy.py:19)
- [backtest/engine.py](/Users/wangzilu/MyProjects/stockPrediction/backtest/engine.py:1)
- 新增建议：`backtest/cost_model.py`
- 新增建议：`backtest/portfolio_report.py`

任务：

1. 明确唯一成交假设，建议先固定为：
   - `T 日收盘后出信号`
   - `T+1 日开盘或 VWAP 模拟成交`
2. 加交易限制：
   - `T+1`
   - 涨停不可买
   - 跌停不可卖
   - 停牌不可交易
   - ST 过滤
   - 最低成交额过滤
3. 加成本模型：
   - 买入佣金
   - 卖出佣金 + 印花税
   - 双边滑点
   - 冲击成本占位参数
4. 输出组合指标：
   - 净收益
   - 年化收益
   - 年化波动
   - Sharpe
   - Calmar
   - 最大回撤
   - 日/周换手
   - 成本占收益比例
5. 保留两套对照：
   - raw top20
   - constraint-aware portfolio

交付物：

- `backtest_result.json`
- `daily_pnl.csv`
- `turnover_report.csv`
- `cost_breakdown.csv`

验收门槛：

- 成本后年化收益 `> 0`
- 成本后 Sharpe `>= 0.8`
- 最大回撤 `<= 20%`
- 平均单期换手 `<= 35%`
- 成本吃掉的收益比例 `<= 35%`

说明：

- 这里的门槛是“小资金日频 alpha 是否值得继续”，不是百亿资金最终门槛。
- 如果 raw 很强但成本后塌掉，优先改组合与执行，不要急着换模型。

### Track C：风险暴露与容量校验

目标：证明收益不是靠隐含暴露偷来的。

涉及文件：

- [models/portfolio_policy.py](/Users/wangzilu/MyProjects/stockPrediction/models/portfolio_policy.py:32)
- 文档中已提到的行业映射表
- 新增建议：`backtest/exposure_report.py`

任务：

1. 把当前按代码前缀近似行业的逻辑替换成真实行业映射。
2. 输出组合暴露：
   - 行业
   - 市值
   - beta
   - 波动率
   - 流动性
   - momentum
   - value
3. 加组合硬约束：
   - 单票权重上限
   - 单行业权重上限
   - 流动性参与率上限
4. 做容量粗估：
   - 单票持仓占 `ADV` 比例
   - 组合成交额占当日市场成交额比例

交付物：

- `exposure_report.csv`
- `industry_weight_timeseries.csv`
- `capacity_report.csv`

验收门槛：

- 单票目标权重默认 `<= 8%`
- 单行业权重默认 `<= 25%`
- 单票计划成交额占 `ADV` 默认 `<= 2%`
- 不允许收益主要来自单一行业长期超配

说明：

- 如果收益高度集中在一个风格段，先标注为“风格 beta 策略”，不要误称通用 alpha。

### Track D：生产治理与 promotion gate

目标：建立 champion / shadow / reject 三态治理。

涉及文件：

- [models/model_registry.py](/Users/wangzilu/MyProjects/stockPrediction/models/model_registry.py:1)
- [tracker/verifier.py](/Users/wangzilu/MyProjects/stockPrediction/tracker/verifier.py:1)
- [scheduler/jobs.py](/Users/wangzilu/MyProjects/stockPrediction/scheduler/jobs.py:1)
- 新增建议：`scripts/promote_model.py`

任务：

1. 给每个模型 run 增加状态：
   - `research_only`
   - `shadow`
   - `champion`
   - `rejected`
2. 统一 promote 判定，不允许手工口头升级。
3. 每次升级必须带 4 类证据：
   - rolling
   - 成本后回测
   - 暴露报告
   - 数据覆盖和特征健康
4. 生产推送默认只用 champion。
5. shadow 每天同时跑，但只记录，不触发正式推荐替换。

交付物：

- `promotion_decision.json`
- `shadow_vs_champion.csv`
- `feature_health_report.json`

验收门槛：

- 必须同时通过 Track A/B/C
- shadow 连续 `20` 个交易日无明显劣化后，才可申请 champion 替换
- 任一关键健康项失败，则自动降级为 `research_only` 或 `shadow`

---

## 五、唯一保留的模型增强实验

Phase 4 只允许一条模型增强实验进入主线旁路：

### `XGB 选池 -> Ranker rerank`

设计：

1. 用 `XGB 174` 先选 `top100` 或 `top200`
2. 只在候选池内用 `Ranker` 二次排序
3. 最终取 `top20`

为什么只保留这条：

- 它不破坏 XGB 的稳定底盘
- 它避免 ranker 在全市场横截面直接主导结果
- 它比 naive 平均和交集更接近机构常见做法

进入下一轮的门槛：

- 成本后 Spread 或组合收益相对 champion 提升 `>= 10%`
- 换手恶化不超过 `15%`
- 暴露不显著恶化
- `24+ split` 下不是靠少数窗口撑出来

如果不通过：

- 冻结 Ranker 主线，不再继续投入主时间

---

## 六、推荐开发顺序

### Week 1：打通正式 rolling gate

目标：

- `24 split` 跑通
- 输出 split 级别 artifact
- champion / shadow 口径统一

本周完成算通过：

- `XGB 174` 正式 rolling 报告可复现
- `XGB 175` 有 shadow 报告
- registry 中能看到对照结果

### Week 2：升级真实回测

目标：

- 成交假设固定
- 成本模型落地
- 涨跌停 / 停牌 / ST / 流动性约束入回测

本周完成算通过：

- 能同时输出 raw 与 cost-adjusted 结果
- 能看到 turnover 和 cost breakdown

### Week 3：补风险暴露与容量

目标：

- 真行业映射接入
- 单票 / 行业 / ADV 约束接入
- 暴露报告自动输出

本周完成算通过：

- 回测报告中能解释收益、暴露和容量约束

### Week 4：promotion gate 与 shadow 运行

目标：

- 统一模型准入
- champion / shadow 自动化
- 每日 shadow 记录与对照

本周完成算通过：

- 新模型不再靠单次实验口头升级

---

## 七、Phase 4 完成定义

只有同时满足下面条件，才算 Phase 4 完成：

1. `XGB 174` 有 `24+ split` 正式 rolling 结果。
2. 成本后回测报告稳定输出，包含收益、回撤、换手、成本。
3. 风险暴露和容量报告稳定输出。
4. champion / shadow / reject 治理生效。
5. 每个模型升级都能追溯到 artifact，而不是只剩终端日志。

---

## 八、明确不做什么

1. 不用单窗口 `Spread +9%` 宣布模型升级。
2. 不因为 `RankIC` 更高就忽略成本后收益。
3. 不把 `175 holder` 的单次好结果当成已取代 `174` 的证据。
4. 不在 Phase 4 中途转去追 Transformer / RL 大工程。
5. 不让生产脚本和研究结论继续口径打架。

---

## 九、最小决策台账

当前建议的统一决策如下：

| 事项 | 决策 |
|---|---|
| 主模型 | `XGB 174` |
| Shadow 模型 | `XGB 175 holder` |
| Ranker 定位 | 仅做 rerank 实验 |
| Phase 4 主目标 | rolling + 成本后 + 暴露后可交易验证 |
| promote 依据 | artifact + gate，不是单次分数 |

---

## 十、一句话版

Phase 4 的本质不是“再找一个更高分模型”，而是把当前最强 `XGB 174` 变成一个**成本后仍有效、风险可控、可解释、可治理**的组合引擎。
