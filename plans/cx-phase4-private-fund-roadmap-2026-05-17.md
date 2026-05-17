# Phase 4 私募化改造路线图

**日期：** 2026-05-17  
**作者：** CX  
**定位：** 把当前 A 股日频选股系统，从“研究上有 alpha”推进到“更像百亿私募内部研究-组合-执行闭环”的工程版本。  
**依赖基线：** `XGB 174` 作为 champion，`XGB 175 holder` 作为 shadow，`Ranker` 仅保留 rerank 旁路实验。  
**配套文档：** `plans/cx-phase4-execution-checklist-2026-05-17.md`

---

## 一、目标定义

Phase 4 不是继续追求更漂亮的单次 `RankIC`，而是建立一个更接近机构工作流的闭环：

1. `rolling` 结果稳定，可重复。
2. 回测结果是**成本后、约束后、可成交**的。
3. 组合收益能分解到行业、风格、选股、成本。
4. 模型升级有统一 gate，不靠口头判断。
5. 生产推送和研究结论口径一致。

---

## 二、最终产物

Phase 4 完成时，项目应新增或稳定输出以下 8 类产物：

1. `24/36 split` rolling 报告
2. 成本后组合回测报告
3. 暴露报告（行业 / size / beta / liquidity）
4. 容量报告（ADV 参与率）
5. champion / shadow 对照报告
6. promotion decision artifact
7. 组合级 paper ledger 雏形
8. 每日模型与组合健康报告

---

## 三、路线图总览

### Phase 4A：研究口径升级

目标：把“12 split rolling”升级成正式准入门槛。

要改的脚本：

- `scripts/rolling_train.py`
- `models/model_registry.py`

建议新增：

- `scripts/phase4_rolling_report.py`
- `scripts/phase4_regime_tagging.py`

核心动作：

1. rolling 从 `12 split` 升级到 `24/36 split`
2. 切窗从自然日改为交易日
3. 支持 `2y / 3y / 5y / expanding` 训练窗口对照
4. 输出 split 级、aggregate 级、regime 级三层结果
5. registry 增加：
   - `feature_set`
   - `train_window`
   - `split_count`
   - `regime_breakdown`
   - `promotion_status`

输出物：

- `data/storage/phase4/rolling_summary.json`
- `data/storage/phase4/rolling_splits.csv`
- `data/storage/phase4/rolling_regime_breakdown.csv`

通过门槛：

- `avg RankIC >= 0.035`
- `avg Spread >= 1.20%`
- `RankIC > 0` split 占比 `>= 65%`
- `Spread > 0` split 占比 `>= 65%`
- 最差 `20%` split 的均值不低于 `-1.50%`

失败动作：

- 不进入组合升级
- 自动标记 `research_only`

### Phase 4B：回测从信号层升级到组合层

目标：让回测回答“这个策略能不能真的拿去配仓”。

要改的脚本：

- `scripts/backtest_qlib_signal.py`
- `backtest/engine.py`
- `models/portfolio_policy.py`

建议新增：

- `backtest/cost_model.py`
- `backtest/order_simulator.py`
- `backtest/portfolio_report.py`

核心动作：

1. 固定成交假设：
   - `T 日收盘后生成信号`
   - `T+1 日开盘` 或 `VWAP` 模拟成交
2. 从“直接吃 label 收益”升级到“持仓-调仓-成交-收益”
3. 加真实限制：
   - `T+1`
   - 涨停不可买
   - 跌停不可卖
   - 停牌不可成交
   - ST 不可持有
   - 最低成交额过滤
4. 加成本模型：
   - 佣金
   - 印花税
   - 双边滑点
   - 冲击成本占位参数
5. 同时保留两类组合：
   - `raw top20`
   - `constraint-aware portfolio`

输出物：

- `data/storage/phase4/backtest_result.json`
- `data/storage/phase4/daily_pnl.csv`
- `data/storage/phase4/turnover_report.csv`
- `data/storage/phase4/cost_breakdown.csv`

通过门槛：

- 成本后年化收益 `> 0`
- 成本后 Sharpe `>= 0.8`
- 最大回撤 `<= 20%`
- 平均调仓换手 `<= 35%`
- 成本吃掉收益比例 `<= 35%`

失败动作：

- 优先调组合约束和换手控制
- 暂不进入 production promotion

### Phase 4C：暴露、容量、可解释性

目标：把收益拆干净，避免“其实只是风格漂移”。

要改的模块：

- `models/portfolio_policy.py`

建议新增：

- `backtest/exposure_report.py`
- `backtest/capacity_report.py`
- `data/storage/industry_map/*.parquet` 或现有行业映射接入层

核心动作：

1. 用真实行业映射替换代码前缀近似行业
2. 输出组合暴露：
   - 行业
   - size
   - beta
   - momentum
   - value
   - volatility
   - liquidity
3. 计算容量粗估：
   - 单票计划成交额 / ADV
   - 组合计划成交额 / 组合成交容量
4. 加组合硬约束：
   - 单票权重上限
   - 单行业权重上限
   - ADV 参与率上限

输出物：

- `data/storage/phase4/exposure_report.csv`
- `data/storage/phase4/industry_weight_timeseries.csv`
- `data/storage/phase4/capacity_report.csv`

通过门槛：

- 单票权重 `<= 8%`
- 单行业权重 `<= 25%`
- 单票成交额占 ADV `<= 2%`
- 不允许收益主要由单一行业长期超配驱动

失败动作：

- 若暴露过大，先降权重和加约束
- 若容量过低，直接标注“小资金有效，不可放大”

### Phase 4D：治理与 promotion gate

目标：让 champion / shadow / reject 真正落地。

要改的模块：

- `models/model_registry.py`
- `scheduler/jobs.py`
- `tracker/verifier.py`

建议新增：

- `scripts/promote_model.py`
- `scripts/shadow_compare.py`
- `scripts/feature_health_report.py`

核心动作：

1. 模型状态标准化：
   - `research_only`
   - `shadow`
   - `champion`
   - `rejected`
2. promote 决策必须依赖 4 类 artifact：
   - rolling
   - 成本后回测
   - 暴露 / 容量
   - 数据 / 特征健康
3. production 只消费 champion
4. shadow 每日并跑，但只记录不推送
5. 连续 `20` 交易日 shadow 不劣于 champion，才允许申请替换

输出物：

- `data/storage/phase4/promotion_decision.json`
- `data/storage/phase4/shadow_vs_champion.csv`
- `data/storage/phase4/feature_health_report.json`

通过门槛：

- Track A/B/C 全通过
- shadow 连续 `20` 日无显著劣化
- 最新数据健康全部通过

失败动作：

- 自动回退到 champion
- 候选模型降级为 `shadow` 或 `research_only`

### Phase 4E：paper trading 雏形

目标：在不碰真实资金前，先验证整套交易闭环。

建议新增：

- `paper/oms.py`
- `paper/ledger.py`
- `paper/fill_engine.py`
- `scripts/run_paper_trading.py`

核心动作：

1. 建立 `target -> order -> fill -> position -> pnl` 基础链
2. 接入组合约束、成本、交易限制
3. 日终输出纸面持仓和收益
4. 留存每笔交易的模型来源、信号来源、约束来源

输出物：

- `data/storage/paper/orders.csv`
- `data/storage/paper/fills.csv`
- `data/storage/paper/positions.csv`
- `data/storage/paper/pnl.csv`

通过门槛：

- 连续 `20-40` 个交易日稳定跑通
- 无脏仓、负现金、非法成交
- 能追溯每笔交易的决策链

失败动作：

- 不进入灰度实盘

---

## 四、唯一保留的增强实验

Phase 4 期间只允许一条模型增强实验进入旁路：

### `XGB top100/top200 -> Ranker rerank -> final top20`

原因：

1. 保留 XGB 的稳定底盘
2. 只让 Ranker 在局部精修
3. 比 naive ensemble 更接近机构实战

需要新增脚本：

- `scripts/phase4_rerank_experiment.py`

输出物：

- `data/storage/phase4/rerank_compare.csv`

通过门槛：

- 成本后收益相对 champion 提升 `>= 10%`
- 换手恶化不超过 `15%`
- 暴露不恶化
- `24+ split` 下不是只靠少数窗口撑起来

失败动作：

- 冻结 Ranker 主线，不继续消耗主开发时间

---

## 五、脚本级拆分建议

### Sprint 1：先把研究 gate 做硬

本 sprint 只改：

- `scripts/rolling_train.py`
- `models/model_registry.py`

建议新增：

- `scripts/phase4_rolling_report.py`

验收：

- 能一键跑出 `24 split` 报告
- registry 里能同时看到 `XGB 174` 和 `XGB 175` 对照
- 文档口径和生产口径不再冲突

### Sprint 2：把回测升级到可交易层

本 sprint 只改：

- `scripts/backtest_qlib_signal.py`
- `backtest/engine.py`
- `models/portfolio_policy.py`

建议新增：

- `backtest/cost_model.py`
- `backtest/order_simulator.py`

验收：

- raw / cost-adjusted 两套结果同时输出
- turnover / cost / drawdown 都可见
- 至少一条真实成交假设固定下来

### Sprint 3：接暴露和容量

本 sprint 重点：

- 行业映射接入
- 暴露报告
- 容量报告

验收：

- 能回答“赚的钱是选股 alpha，还是行业超配”
- 能回答“小资金能不能做，大资金会不会撞容量”

### Sprint 4：champion/shadow 治理

本 sprint 重点：

- promote 脚本
- shadow 对照
- feature health artifact

验收：

- 任何模型升级都有 artifact 依据
- scheduler 只消费 champion

### Sprint 5：paper trading

本 sprint 重点：

- ledger
- fill engine
- 日终对账

验收：

- 连续 `20+` 交易日 paper 运行正常

---

## 六、验收指标总表

| 维度 | 指标 | 门槛 |
|---|---|---|
| Rolling | avg RankIC | `>= 0.035` |
| Rolling | avg Spread | `>= 1.20%` |
| Rolling | RankIC 正值 split 占比 | `>= 65%` |
| Rolling | Spread 正值 split 占比 | `>= 65%` |
| Backtest | 成本后 Sharpe | `>= 0.8` |
| Backtest | 最大回撤 | `<= 20%` |
| Backtest | 平均换手 | `<= 35%` |
| Backtest | 成本占收益比例 | `<= 35%` |
| Exposure | 单票权重 | `<= 8%` |
| Exposure | 单行业权重 | `<= 25%` |
| Capacity | 单票成交额 / ADV | `<= 2%` |
| Governance | shadow 跟踪期 | `>= 20` 交易日 |
| Paper | 稳定运行 | `20-40` 交易日 |

---

## 七、明确不做什么

1. 不在 Phase 4 里重新开 Transformer 主线。
2. 不在 Phase 4 里把 RL 提前拉进交易层。
3. 不再用单窗口高 spread 宣布模型升级。
4. 不让 `175 holder` 在 rolling 未证实前覆盖 champion。
5. 不再做 naive ensemble 和 `Top50 ∩ Top50` 主线实验。

---

## 八、一句话执行顺序

先把 `rolling gate` 做硬，再把 `backtest` 做真，再把 `exposure/capacity` 做透，最后把 `champion/shadow/paper` 跑顺。
