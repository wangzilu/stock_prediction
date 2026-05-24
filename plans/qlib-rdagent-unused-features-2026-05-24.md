# Qlib + RD-Agent 未用功能调研

**日期**: 2026-05-24
**作者**: CC
**请 CX 审阅**

---

## 1. 当前已用的 Qlib 模块

| 模块 | 用途 | 使用位置 |
|------|------|---------|
| `qlib.data.D` | OHLCV 数据加载、instrument 列表 | 多处 |
| `Alpha158` handler | 158 维价量因子 | `train_lgb.py`, 特征缓存 |
| `Alpha360` handler | 360 维原始序列特征 | `phase4m` 实验 |
| `LGBModel` / `XGBModel` | 模型训练预测 | `train_lgb.py`, workflow |
| `DatasetH` | 数据集构建 | 训练脚本 |
| `calc_ic` / `risk_analysis` | IC/RankIC 评估 | 因子评估 |
| `Recorder (R)` | 实验记录 | 仅 `run_qlib_workflow.py` |

**未用 Qlib 的部分**：回测引擎（自建 `backtest/engine.py`）、组合优化（自建 `backtest/optimizer.py`）、深度模型（自建绕过 MPS 兼容问题）、在线服务（自建 `main.py` + scheduler）。

---

## 2. 未用功能——第一梯队（直接可用，无依赖阻塞）

### 2.1 ShrinkCovEstimator — 收缩协方差估计

**模块**: `qlib.model.riskmodel.ShrinkCovEstimator`

**功能**: Ledoit-Wolf 收缩方法估计协方差矩阵。比样本协方差更稳定，特别是当持仓数接近或超过历史窗口长度时（10 只股票、60 天历史 → 样本协方差不稳定）。

**当前状态**: 项目的 `optimizer_v2` 使用 alpha-proportional 权重，没有用协方差矩阵。`barra_simple.py` 计算了 5 个风格因子但未接入优化器。

**价值**: 如果未来优化器要做均值-方差优化或风险平价，ShrinkCov 是最基础的输入。即使不做 MVO，也可以用它计算组合波动率预测，作为 RiskGuard 的补充信号。

**接入难度**: 低。无额外依赖，直接 import 可用。

### 2.2 StructuredCovEstimator — 因子结构化协方差

**模块**: `qlib.model.riskmodel.StructuredCovEstimator`

**功能**: 基于因子模型（类 Barra）估计协方差。将股票协方差分解为 `B * F * B' + D`（因子暴露 × 因子协方差 × 因子暴露转置 + 特异波动），比全样本协方差更稳定且可解释。

**当前状态**: 项目已有 `barra_simple.py`（5 style + 33 industry），但只用于暴露监控，没有接入协方差估计。

**价值**: 配合已有的 Barra 因子，可以做出工业级的组合风险模型。

**接入难度**: 中。需要把 `barra_simple` 的因子暴露矩阵对接到 Qlib 的接口格式。

### 2.3 POETCovEstimator — 高维协方差

**模块**: `qlib.model.riskmodel.POETCovEstimator`

**功能**: Principal Orthogonal complEment Thresholding。适合股票数远大于历史天数的场景（e.g. 100 只候选股、60 天历史）。

**价值**: 如果未来扩大持仓候选池（top100 → top200），这个比 ShrinkCov 更合适。

### 2.4 DDG-DA — 数据驱动泛化 + 域自适应

**模块**: `qlib.model.meta`

**功能**: 学习哪些历史数据与当前市场条件最相关。不是简单的滚动窗口（最近 N 天），而是根据市场状态动态加权历史样本。

**当前状态**: 项目用固定 480 天滚动窗口，rolling IC 跨窗口衰减是已知问题。

**价值**: 高。直接解决 concept drift / rolling IC 衰减。如果 2024.10 的市场状态更像 2015.06 而非 2024.09，DDG-DA 会自动增加 2015.06 附近数据的权重。

**接入难度**: 高。需要构建 meta-task、meta-dataset，训练 meta-model。但原理清晰，值得投入。

**风险**: 论文效果不一定在 A 股复现。需要先在 24-split 上验证。

### 2.5 ADARNN — 自适应非平稳 RNN

**模块**: `qlib.contrib.model.adarnn`

**功能**: 专为非平稳时间序列设计的 RNN。自动检测分布漂移并调整内部表示。

**当前状态**: 项目的 ALSTM/Transformer 实验效果不如 XGB（深度模型 IC 低且不稳定）。

**价值**: 中。如果 ADARNN 的自适应机制能改善深度模型在 A 股的稳定性，可能突破当前"深度模型不如 GBDT"的瓶颈。但也可能和之前的深度模型一样效果不佳。

**接入难度**: 中。需要 MPS 兼容性验证。

### 2.6 RollingGen — 自动滚动窗口生成

**模块**: `qlib.workflow.task.gen.RollingGen`

**功能**: 自动生成滚动训练/验证/测试窗口配置。支持 expanding window 和 sliding window。

**当前状态**: 每个训练脚本（`rolling_train.py`, `train_lgb.py`, `train_model_suite.py`）都手写滚动窗口逻辑，代码重复且参数硬编码。

**价值**: 中。消除重复代码，统一滚动窗口定义。但不会改变模型效果。

**接入难度**: 低。

### 2.7 公式化因子引擎

**模块**: `qlib.data.ops`（通过 `$close`, `Ref`, `Corr` 等表达式）

**功能**: 用字符串表达式定义因子，例如 `Corr($close, $volume, 10)` 或 `Rank(Mean($volume, 5) / Mean($volume, 20))`。不需要写 Python 函数。

**当前状态**: 项目只用表达式定义 label（`Ref($close, -1) / Ref($close, -2) - 1`），不用于因子构建。

**价值**: 中。快速原型验证新因子假设，不需要写完整的 Python pipeline。配合 RD-Agent 的自动因子挖掘效果更好。

### 2.8 SignalRecord / PortAnaRecord — 标准化实验记录

**模块**: `qlib.workflow.record_temp`

**功能**: 自动记录每次实验的预测信号、IC 分析、组合回测结果。支持 MLflow 后端。

**当前状态**: 大部分训练脚本把结果 dump 到 JSON 文件（`data/storage/factor_tearsheet_*.json`），没有统一的实验管理。

**价值**: 中。让模型版本对比和回归检测更系统化。但当前项目规模（1 个 champion + 1 个 shadow）还没到需要完整实验管理的程度。

---

## 3. 未用功能——第二梯队（有价值但有前置条件）

### 3.1 TopkDropoutStrategy — 自动换手控制

**模块**: `qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy`

**功能**: 持有 Top-K 股票，只有跌出 Top-K+N 才卖出（dropout buffer）。自动控制换手率，不需要手动设置 max_turnover。

**当前状态**: 项目用自建 `optimizer_v2`（opt_top100_to10, max_turnover=10%）。

**阻塞**: 依赖 `PortfolioOptimizer` → cvxpy → numpy>=2.0。当前环境 numpy 1.x 与 cvxpy 冲突。

**解决方案**: 升级到 numpy 2.x + 验证 Alpha158 兼容性，或者在虚拟环境中隔离。

### 3.2 Qlib Backtest Engine

**模块**: `qlib.backtest`

**功能**: 完整的事件驱动回测。Exchange 对象模拟交易所（涨跌停、成交量限制、佣金、滑点）。Account 跟踪持仓和现金。Position 管理个股仓位。

**当前状态**: 自建 `backtest/engine.py`，已实现 VWAP/open 执行、IPO 过滤、一字板过滤。

**阻塞**: 同 cvxpy/numpy 冲突。

**价值**: Qlib 的回测引擎更完整（支持日内多次交易、订单队列），但自建引擎已经覆盖了主要需求。除非要做日内执行优化，否则迁移的 ROI 不高。

### 3.3 Online Serving

**模块**: `qlib.contrib.online`

**功能**: `OnlineManager` 管理多策略在线运行，`PredUpdater` 自动刷新预测，`RollingStrategy` 管理模型滚动更新。

**当前状态**: 自建 `main.py` + `scheduler/jobs.py` + 23 个 crontab。

**价值**: 如果要做到"新数据到 → 自动重训练 → 自动切换模型 → 自动执行"的全自动闭环，Qlib 的 Online 模块比手写 crontab 更可靠。但当前手动管理还能应付。

### 3.4 HIST — 异构信息股票 Transformer

**模块**: `qlib.contrib.model.pytorch_hist`

**功能**: 用行业和概念关系图构建股票间的注意力机制。能捕捉"同行业联动"和"概念板块轮动"。

**当前状态**: 项目没有构建股票关系图。

**价值**: 中。A 股板块联动效应很强，HIST 理论上能利用这个特征。但需要先构建行业/概念图数据。

### 3.5 RL 订单执行

**模块**: `qlib.rl`

**功能**: 用强化学习优化订单执行策略（VWAP/TWAP），减少市场冲击。

**当前状态**: 项目有 `models/rl_agent.py`（独立 DQN/PPO），但用于选股不是执行。

**价值**: 低（当前阶段）。只有在做日内执行优化时才有意义。当前 T+1 open 执行不需要复杂的执行策略。

---

## 4. RD-Agent（完全未用）

### 4.1 是什么

Microsoft Research 开发的自动化研发框架。核心思路：用 LLM 作为"研究员"自动提出因子/模型假设，然后自动实现、测试、迭代。

### 4.2 核心功能

| 功能 | 说明 | 对我们的价值 |
|------|------|-------------|
| **Auto Factor Mining** | LLM 自动提出因子假设 → 生成 Python 代码 → 用 Qlib 回测 → 根据 IC/收益反馈迭代 | 高 — 可能发现 229 因子之外的有效信号 |
| **Auto Model Tuning** | 自动搜索模型架构和超参数 | 中 — 手动调参已经做了 |
| **Factor-Model Co-Optimization** | 因子和模型同时演化，联合优化 | 高 — 当前 pipeline 因子和模型是分开的 |
| **Financial Report Analysis** | 从财报中用 LLM 提取可交易因子 | 中 — 补充 fundamental 因子 |
| **Research Loop Automation** | R(Research) agent 提假设 → D(Development) agent 实现测试 → 自动循环 | 高 — 自动化目前手动的"提假设→写代码→跑回测→看结果"循环 |

### 4.3 公开性能

论文声称：用 70% 更少的因子，实现 2x 的年化收益。每个迭代周期成本 < $10。

### 4.4 限制

- **需要 Linux + Docker**：Mac Studio 上不能直接跑。需要一台 Linux 服务器或云实例。
- **LLM 成本**：每个 R&D 循环需要大量 LLM API 调用（GPT-4 或 Claude）。
- **验证风险**：论文效果可能在 A 股不成立（大部分量化论文在 A 股效果打折）。
- **与现有 pipeline 对接**：需要把现有的 Qlib 数据和因子格式对接到 RD-Agent 的接口。

---

## 5. 被阻塞的根因：cvxpy/numpy 冲突

多个高价值功能（TopkDropout、PortfolioOptimizer、Qlib Backtest）被同一个依赖冲突阻塞：

```
cvxpy 要求 numpy >= 2.0
当前环境 numpy 1.x（Alpha158 等依赖）
```

**解决方案选项**：

| 方案 | 风险 | 工作量 |
|------|------|--------|
| 升级 numpy 2.x + 验证全部代码兼容性 | 中（可能有隐式 API 变更） | 1-2 天 |
| 用独立虚拟环境跑需要 cvxpy 的部分 | 低 | 半天 |
| 不用 Qlib 的优化器，继续自建 | 无风险 | 已完成 |

当前自建 optimizer_v2 (Sharpe 4.5+) 效果已经很好，短期不需要 Qlib 的优化器。但如果要做均值-方差优化或风险平价，迟早要解决这个冲突。

---

## 6. CC 建议的优先级

### 现在可以做（P0/P1）

| 优先级 | 功能 | 理由 |
|--------|------|------|
| P1 | ShrinkCovEstimator 接入 optimizer | 无依赖问题，直接改善风险估计 |
| P1 | DDG-DA 原型验证 | 对抗 concept drift，24-split 验证 |
| P2 | RollingGen 替换手写滚动逻辑 | 消除重复代码 |
| P2 | SignalRecord 标准化实验记录 | 改善实验管理 |

### 中期值得探索（P2/P3）

| 优先级 | 功能 | 前置条件 |
|--------|------|---------|
| P2 | ADARNN / HIST 模型实验 | MPS 兼容性验证 |
| P2 | 解决 cvxpy 冲突 → 接入 TopkDropout | numpy 升级验证 |
| P3 | RD-Agent 自动因子挖掘 | Linux 服务器 + Docker |
| P3 | RL 订单执行 | 日内交易需求确认 |

### 不建议做

| 功能 | 理由 |
|------|------|
| 迁移到 Qlib Backtest Engine | 自建引擎已满足需求，迁移 ROI 不高 |
| Online Serving 替换 crontab | 当前手动管理足够，重构代价大 |
| HighFreqHandler | 项目是日频，不需要分钟/tick 数据 |

---

*请 CX 审阅，重点判断：DDG-DA 是否值得投入？RD-Agent 是否需要租 Linux 服务器跑？*
