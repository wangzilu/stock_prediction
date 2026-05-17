# Phase 5 / 6 RL 策略控制器实施方案

**日期：** 2026-05-17  
**作者：** CX  
**定位：** 把 RL 放进当前项目的真实开发链路里，作为“组合控制器 / 风险控制器 / 执行控制器”，而不是端到端个股预测器。  
**前置条件：** Phase 4A-D 必须完成，尤其是日频回测口径修复、champion/shadow 治理、暴露与容量报告。  
**关联文档：**

- `plans/cx-phase4-execution-checklist-2026-05-17.md`
- `plans/cx-phase4-private-fund-roadmap-2026-05-17.md`
- `plans/cc-backtest-label-mismatch-2026-05-17.md`

---

## 一、先定边界

RL 在这套系统里只做三件事：

1. 决定**什么时候更激进、什么时候更保守**。
2. 决定**仓位怎么分配、什么时候换、换多少**。
3. 决定**执行方式怎么选**，比如更快换仓还是更少换手。

RL 不做这些事：

1. 不直接替代 `XGB 174` 这种 alpha 主模型。
2. 不直接在全市场里学“哪只股票明天涨”。
3. 不绕过交易成本、涨跌停、停牌、ST、容量约束。
4. 不在 Phase 4 没稳定前就进 production。

一句话：**先把 RL 做成 policy controller，再考虑它有没有资格碰实盘。**

---

## 二、当前代码状态

现在仓库里已经有 RL 雏形，但它更像 sandbox，不是机构化版本：

- `scripts/train_rl.py`
  - 当前是单票环境雏形，环境是 `buy / hold / sell`。
  - 用了 Alpha158 特征 + `qlib_score` + sentiment/regime 状态。
  - 训练逻辑更接近手写 DQN，不是成熟的 offline RL 流水线。
- `models/rl_agent.py`
  - 里面已经有 `StockTradingEnv`、`TransformerActor`、`TransformerCritic`、`RLAgent`。
  - 这个文件适合作为原型，但还没拆出 portfolio-level 环境。
- `models/short_term.py`
  - 可作为 alpha 基座。
- `models/mid_term.py`
  - 可作为中线状态输入。
- `models/portfolio_policy.py`
  - 现在已有组合约束雏形，可作为 RL 的 hard constraint 层。

结论：

- 这套 RL 代码不能直接扩成“自动赚钱机器人”。
- 但它足够作为 **Phase 5 的起点**。

---

## 三、RL 要解决的真正问题

最有价值的 RL 任务不是个股预测，而是下列三层：

### 层 1：组合控制

RL 决定：

- `topk` 取多少
- 是否收缩到更保守的候选池
- 当前是否应该降杠杆 / 降仓位 / 提高现金
- 当前是否应该加快或放慢换仓

### 层 2：风险控制

RL 决定：

- 在市场强势 / 弱势 / 高波动 / 低波动 regime 下，仓位如何变化
- 风险预算是否收紧
- 哪些行业暴露该压缩
- 是否触发“只减仓不加仓”的防御状态

### 层 3：执行控制

RL 决定：

- 这次调仓是更接近 `TWAP`、`VWAP` 还是缩短执行窗口
- 是否因为流动性或成本太差而降低调仓幅度

所以 RL 的正确位置是：

> `alpha 模型 -> 组合候选 -> RL policy controller -> 风控/执行 -> PnL`

不是：

> `RL 直接替代 alpha`

---

## 四、推荐的开发阶段

### Phase 4：必须先完成的前置条件

这个阶段不属于 RL，但没它就不该碰 RL。

必须先修好：

1. `phase4_backtest.py` 的 daily PnL 口径。
2. `PortfolioBacktest` 的 return horizon 防呆。
3. `XGB 174` / `XGB 175` 的 champion / shadow 口径。
4. 组合回测里的成本、换手、暴露、容量报告。

验收门槛：

- Track B 结果可复现。
- 组合回测不再出现 label / PnL 口径错位。
- 生产模型和研究模型口径一致。

### Phase 5A：RL Sandbox 改造成 Portfolio MDP

目标：把当前单票环境改造成组合级环境。

### Phase 5B：Offline RL 训练

目标：先在历史数据上做保守策略学习，不做在线探索。

### Phase 5C：Shadow / Paper Trading

目标：RL policy 先只跑 shadow，再进 paper。

### Phase 6：Execution RL / 细粒度调度

目标：只在前面都稳定后，再考虑把 RL 下沉到执行层。

---

## 五、推荐架构

### 5.1 三层系统

#### 第一层：Alpha Layer

职责：

- 产生个股预期收益分数
- 产生候选池排序

来源：

- `XGB 174` champion
- `XGB 175 holder` shadow
- `Ranker rerank` 旁路实验

#### 第二层：RL Policy Layer

职责：

- 决定今天到底更偏进攻还是防守
- 决定 topk / 仓位 / 换手 / 风险预算

输入：

- alpha 分数分布
- 组合暴露
- 市场 regime
- 流动性状态
- 最近收益和回撤
- 成本环境

#### 第三层：Execution Layer

职责：

- 把 policy 输出转成订单
- 模拟成交 / 约束 / 成本
- 最终形成日频 PnL

---

## 六、状态、动作、奖励

这是 RL 成败的核心。

### 6.1 状态设计

不建议一开始就喂全市场原始特征。  
建议用**压缩后的组合状态 + 候选池状态**。

#### 状态 A：组合状态

| 类别 | 内容 |
|---|---|
| 账户状态 | 现金、总仓位、净值、最近回撤、最近收益 |
| 持仓状态 | 持仓数、平均持有期、集中度、单票最大权重 |
| 风险状态 | 行业暴露、风格暴露、ADV 参与率、是否触发风控 |
| 成本状态 | 最近换手、最近成本、成本占收益比例 |
| 执行状态 | 上次调仓日、距离上次调仓天数、可交易标的数 |

#### 状态 B：市场状态

| 类别 | 内容 |
|---|---|
| 指数状态 | 沪深主指数 1/5/20 日收益、波动、回撤 |
| 宽度状态 | 上涨家数、涨停家数、跌停家数、停牌家数 |
| regime 状态 | 强势 / 弱势 / 高波动 / 低波动 |
| 流动性状态 | 成交额、换手率、ADV 分布、尾部流动性 |
| 宏观状态 | 利率 / 汇率 / 风险偏好 / 地缘风险 |

#### 状态 C：候选池状态

不看全市场，只看 champion 选出的候选池，例如 top100 或 top200。

| 类别 | 内容 |
|---|---|
| alpha 统计 | 分数均值、方差、分位差、top-bottom spread |
| candidate 风险 | 行业集中度、风格集中度、流动性集中度 |
| score 分布 | 分数排序、分层聚类、稳定性指标 |
| rerank 信息 | Ranker score、差异分布、与 champion 的分歧度 |

#### 状态版本建议

- `state_v1`: 组合摘要 + 市场摘要 + top100 候选统计
- `state_v2`: 加入候选池逐标的特征
- `state_v3`: 加入执行层特征和订单簿代理变量

先做 `state_v1`，不要一步到 `state_v3`。

---

### 6.2 动作设计

不要一开始就做“每只股票 buy / sell / hold”的大动作空间。  
先做**组合控制动作**，更稳，更像机构。

#### 动作空间 v1：离散控制器

动作由几个低维控制量组成：

| 动作维度 | 取值示例 | 含义 |
|---|---|---|
| `risk_scale` | 0.5 / 0.75 / 1.0 / 1.25 | 整体仓位强弱 |
| `topk_bucket` | 10 / 20 / 30 / 40 | 持仓集中度 |
| `rebalance_bucket` | 1 / 5 / 10 | 调仓频率 |
| `dropout_bucket` | 0 / 5 / 10 | 持有保护强度 |
| `sector_tightness` | loose / normal / tight | 行业约束强度 |

这一步的动作不是下单，而是**给组合引擎下策略参数**。

#### 动作空间 v2：目标权重控制

在 topN 候选池内，RL 输出：

- 每个候选的权重 bucket
- 持仓目标比例
- 行业预算分配

#### 动作空间 v3：执行控制

在权重目标确定后，RL 决定：

- `TWAP`
- `VWAP`
- `POV`
- 简化的一次性成交

#### 推荐顺序

1. 先做 v1。
2. 再做 v2。
3. 最后才碰 v3。

---

### 6.3 奖励设计

奖励必须是**净收益导向**，不能只看裸收益。

建议：

```text
reward_t
= log(1 + net_return_t)
- λ1 * turnover_t
- λ2 * max(0, drawdown_t - dd_budget)
- λ3 * exposure_violation_t
- λ4 * cost_ratio_t
- λ5 * action_change_penalty_t
```

### 推荐权重

| 项 | 初始权重 |
|---|---|
| turnover penalty | 0.05 ~ 0.20 |
| drawdown penalty | 1.0 ~ 3.0 |
| exposure penalty | 1.0 ~ 5.0 |
| cost ratio penalty | 0.5 ~ 1.5 |
| action change penalty | 0.01 ~ 0.10 |

### 原则

1. 不要把 reward hard clip 到太窄。
2. 不要让 RL 只学会“少动就好”。
3. 不要让 reward 跟训练 label 混成一锅。

---

## 七、数据管线

### 7.1 RL 数据来源

RL 不直接读原始全量特征，而是读**已经通过 Phase 4 gate 的研究产物**。

数据来源分三块：

1. `alpha` 输出
   - `XGB 174`
   - `XGB 175 holder`
   - `Ranker rerank`
2. `portfolio` 输出
   - 组合持仓
   - 日频 PnL
   - 换手
   - 暴露
3. `market` 输出
   - regime
   - 宽度
   - 波动
   - 流动性

### 7.2 轨迹格式

建议定义统一轨迹 schema：

| 字段 | 说明 |
|---|---|
| `date` | 决策日期 |
| `state` | 压缩后的状态向量 |
| `action` | RL 动作 |
| `reward` | 当日 reward |
| `next_state` | 下一日状态 |
| `done` | episode 是否结束 |
| `metadata` | 成本、换手、暴露、regime |

### 7.3 轨迹的来源

训练 RL 时，轨迹不要只来自随机探索。  
要从三个行为策略里采样：

1. `fixed TopK`
2. `risk-aware TopK`
3. `champion XGB + rerank`

这三类轨迹构成 offline dataset。

### 7.4 数据生成脚本

建议新增：

- `scripts/build_rl_dataset.py`
- `scripts/build_rl_trajectory_store.py`
- `scripts/rl_dataset_health.py`

必须输出：

- `rl_trajectories.parquet`
- `rl_state_stats.json`
- `rl_dataset_health.json`

---

## 八、训练路线

### 8.1 训练顺序

#### Stage 1：Behavior Cloning

先学规则策略，不直接上 RL。

目标：

- 让 policy 至少学会不要乱来
- 先复制 fixed TopK / risk-aware policy

#### Stage 2：Conservative Offline RL

再上保守算法：

- `IQL`
- `CQL`
- `TD3+BC`

原因：

- 这些方法更适合纯历史数据
- 不需要危险的在线探索
- 对金融这种低容错场景更稳

#### Stage 3：Sequence Policy

如果前两步稳定，再试：

- `Decision Transformer`
- `Trajectory Transformer`

用途：

- 学习 regime 切换下的策略序列
- 作为对照，不作为首发主模型

### 8.2 当前代码如何改

#### `scripts/train_rl.py`

建议拆成四个脚本：

1. `scripts/build_rl_dataset.py`
2. `scripts/train_rl_policy.py`
3. `scripts/evaluate_rl_policy.py`
4. `scripts/rl_shadow_runner.py`

#### `models/rl_agent.py`

建议拆成三个模块：

1. `models/rl_env_portfolio.py`
2. `models/rl_policy.py`
3. `models/rl_inference.py`

#### 现有 `StockTradingEnv`

保留，但只作为 sandbox / unit test 环境。

不要把它直接当成机构级主环境。

---

## 九、评估与门槛

RL 的评估不能只看 episode reward。

### 9.1 必须对照的 baseline

1. `fixed TopK`
2. `risk-aware TopK`
3. `XGB 174 champion`
4. `XGB 174 + Ranker rerank`
5. `random policy`

### 9.2 必须输出的指标

| 类别 | 指标 |
|---|---|
| 收益 | annual return, total return, net alpha |
| 风险 | sharpe, calmar, max drawdown |
| 交易 | turnover, cost ratio, fill proxy |
| 组合 | sector exposure, style exposure, concentration |
| 稳定性 | worst 20% split, regime breakdown |
| 容量 | ADV participation, capacity estimate |

### 9.3 通过门槛

进入 shadow 前，至少满足：

- 成本后 Sharpe 不低于 baseline
- 成本后年化收益不低于 baseline 或提升明显
- 最大回撤不明显恶化
- 换手不显著上升
- 暴露约束不突破
- 最差 20% split 不比 baseline 差太多

建议更硬一点的门槛：

| 门槛 | 建议值 |
|---|---|
| cost-adjusted Sharpe | `>= baseline` |
| annual return uplift | `>= +10%` relative 或绝对显著改善 |
| max drawdown | `<= baseline + 2pp` |
| turnover | `<= baseline + 15%` |
| exposure violations | `0` |
| worst regime split | 不失控 |

### 9.4 统计检验

建议加入：

- block bootstrap
- regime stratified comparison
- paired daily return test

否则 RL 很容易靠一两个 regime 的好运气显得很强。

---

## 十、和现有 Phase 的对应关系

### Phase 4

先做完：

- backtest 口径修复
- cost-aware portfolio
- exposure/capacity report
- champion/shadow governance

### Phase 5

启动 RL sandbox：

- build RL dataset
- offline BC
- conservative RL
- shadow compare

### Phase 6

进入 paper / execution：

- paper OMS
- fill simulator
- daily compliance log
- shadow to small-cap live

---

## 十一、文件级改造清单

### 必改

- `scripts/train_rl.py`
- `models/rl_agent.py`
- `scripts/phase4_backtest.py`
- `backtest/portfolio_backtest.py`
- `models/portfolio_policy.py`
- `tracker/verifier.py`
- `scheduler/jobs.py`

### 建议新增

- `scripts/build_rl_dataset.py`
- `scripts/train_rl_policy.py`
- `scripts/evaluate_rl_policy.py`
- `scripts/rl_shadow_runner.py`
- `scripts/rl_dataset_health.py`
- `models/rl_env_portfolio.py`
- `models/rl_policy.py`
- `models/rl_inference.py`
- `backtest/rl_policy_report.py`
- `paper/oms.py`
- `paper/fill_engine.py`
- `paper/ledger.py`

---

## 十二、推荐实施顺序

### Week 1-2

- 修好 Phase 4 回测口径
- 跑通 champion/shadow 资产链
- 冻结当前 RL 代码为 sandbox

### Week 3-4

- 定义 portfolio-level RL 状态
- 生成 offline trajectories
- 做 behavior cloning baseline

### Week 5-6

- 上 IQL / CQL
- 跑 24+ rolling split
- 和 fixed TopK / rerank 对照

### Week 7-8

- shadow runner 每日跑
- 记录 RL policy 与 champion 差异
- 评估是否值得 paper

### Week 9+

- paper trading
- 小资金灰度
- 只在稳定后考虑 execution RL

---

## 十三、明确不做什么

1. 不把 RL 先放到全市场选股主线。
2. 不把 `buy/hold/sell` 直接当机构级策略。
3. 不在没有 daily PnL 的情况下训练 policy。
4. 不用在线探索在真实账户里试错。
5. 不允许 RL 直接突破风控硬约束。
6. 不用单次 episode reward 代表真实可交易收益。

---

## 十四、最终结论

RL 可以做，而且值得做。  
但在这套项目里，它最合适的位置不是 alpha 发动机，而是：

> **基于现有 champion alpha 的组合控制器 + 风险控制器 + 执行控制器**

这样做的好处是：

1. 不会和 `XGB 174` 这种稳定 alpha 冲突。
2. 能在 Phase 4 的资产链上自然接入。
3. 训练目标更接近机构真实工作流。
4. 更容易做 shadow / paper / promotion gate。

如果按这个路线走，RL 才有机会从“实验好玩”变成“可用策略层”。 
