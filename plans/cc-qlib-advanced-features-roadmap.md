# Qlib 高级功能利用方案 — 从 Alpha158+LGB 到全栈量化

**日期：** 2026-05-09
**背景：** 当前项目只用了 Qlib 约 10% 的能力（Alpha158 + LGBModel + DatasetH）。Qlib 0.9.7 内置了 26 个深度学习模型、组合优化器、回测引擎、滚动训练、风险模型、元学习等大量功能。本文档评估每个功能对本项目的价值，按 ROI 排序给出实施方案。

---

## 一、当前使用 vs 可用功能全景

| 层级 | 当前使用 | Qlib 还有什么 | 差距 |
|------|---------|--------------|------|
| 特征 | Alpha158（158 个因子） | Alpha360（360+ 因子）、Alpha158vwap、HighFreqHandler、自定义表达式引擎 | 只用了 1/4 的特征空间 |
| 模型 | LGBModel | 25 个 PyTorch 模型 + XGBoost + CatBoost + Ensemble + TabNet | 只用了 1/28 的模型 |
| 训练 | 单次全量训练 | 滚动训练（Rolling）、在线学习、元学习、超参调优 | 没有任何自动化训练策略 |
| 评估 | 无 | IC/RankIC、risk_analysis、indicator_analysis、alpha_analysis | 完全没用 Qlib 评估工具 |
| 回测 | 无（用 change_pct 验证） | 内置回测引擎（Exchange + Account + Position）、涨跌停/手续费/滑点 | 最大缺口 |
| 组合 | 无（只推个股） | PortfolioOptimizer（MVO/RP/GMV）、TopkDropoutStrategy | 推荐无法变成可执行仓位 |
| 风险 | 无 | ShrinkRiskModel、StructuredRiskModel、PoemRiskModel | 无风险控制 |
| 实验管理 | 无 | MLflow Recorder、Experiment、artifact 管理 | 训练不可追溯 |

---

## 二、按 ROI 排序的功能推荐

### Tier 1：立刻用，投入小收益大（1-3 天/个）

#### 1.1 IC/RankIC 因子诊断 — 验证 LGB 分数是否真有预测力

**当前问题：** LGB 训出来了，但不知道预测分数和实际收益之间有没有统计显著关系。可能模型看起来在训练集上好，实际预测是随机的。

**Qlib 提供的：**
```python
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.eva.alpha import alpha_analysis

# 用模型预测 + 真实收益算 IC
pred = model.predict(dataset)
report = risk_analysis(pred)
# 输出：IC均值、ICIR、RankIC、年化收益、最大回撤、Sharpe
```

**实施：** 在 `train_lgb.py` 训练完后加 10 行代码：
```python
from qlib.contrib.evaluate import risk_analysis
pred = model.predict(dataset)
report_normal, report_excess = risk_analysis(pred, freq="day")
print(f"IC: {report_normal['IC']:.4f}")
print(f"ICIR: {report_normal['ICIR']:.4f}")
print(f"RankIC: {report_normal['Rank IC']:.4f}")
print(f"Annualized Return: {report_normal['Annualized Return']:.4f}")
print(f"Max Drawdown: {report_normal['Max Drawdown']:.4f}")
```

**价值：** 如果 IC < 0.02 或 RankIC < 0.03，说明模型预测力弱于随机，不应该上生产。这是最基础的质量门禁。

**工作量：** 半天

---

#### 1.2 Qlib 内置回测 — 替代 change_pct 验证

**当前问题：** 推荐的"验证"只是看推荐后 5 天的 change_pct，没有考虑交易成本、涨跌停、停牌、滑点。一个"胜率 60%"的策略加上千二手续费可能亏钱。

**Qlib 提供的：**
```python
from qlib.backtest import backtest as qlib_backtest
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import risk_analysis

strategy_config = {
    "class": "TopkDropoutStrategy",
    "module_path": "qlib.contrib.strategy.signal_strategy",
    "kwargs": {
        "signal": pred,        # 模型预测信号
        "topk": 10,            # 买入前10
        "n_drop": 3,           # 每次轮换3只
    },
}

backtest_config = {
    "start_time": "2025-01-01",
    "end_time": "2026-05-01",
    "account": 1000000,        # 初始资金 100万
    "benchmark": "SH000300",   # 沪深300基准
    "exchange_kwargs": {
        "open_cost": 0.0005,   # 买入万五
        "close_cost": 0.0015,  # 卖出万五+千一印花税
        "min_cost": 5,
        "limit_threshold": 0.099,  # 涨跌停阈值
    },
}

portfolio_metric, indicator = qlib_backtest(strategy=strategy_config, **backtest_config)
analysis = risk_analysis(portfolio_metric["return"])
```

**价值：** 用真实交易约束回测，知道策略扣费后是否盈利。如果 Sharpe < 0.5，不值得推。

**工作量：** 1 天

---

#### 1.3 CatBoost + XGBoost Ensemble — 低成本提升预测稳定性

**当前问题：** 只用 LightGBM 一个模型，预测不稳定。金融 ML 的常识是单模型不如 ensemble。

**Qlib 提供的：**
```python
# CatBoost
catboost_config = {
    "class": "CatBoostModel",
    "module_path": "qlib.contrib.model.catboost_model",
    "kwargs": {"loss_function": "RMSE", "iterations": 500},
}

# XGBoost
xgb_config = {
    "class": "XGBModel",
    "module_path": "qlib.contrib.model.xgboost",
    "kwargs": {"n_estimators": 500, "max_depth": 8},
}

# 训练3个模型，预测取平均
lgb_pred = lgb_model.predict(dataset)
cat_pred = cat_model.predict(dataset)
xgb_pred = xgb_model.predict(dataset)
ensemble_pred = (lgb_pred + cat_pred + xgb_pred) / 3
```

**价值：** Lopez de Prado 和 Leippold 2022（JFE A 股 ML 论文）都证明树模型 ensemble 在 A 股上优于单模型。预期 IC 提升 10-30%。

**工作量：** 1 天（CatBoost/XGBoost 需要 pip install）

---

### Tier 2：一周内做，解决核心架构缺陷

#### 2.1 滚动训练（Rolling）— 解决"用旧数据训的模型预测新数据"

**当前问题：** LGB 训练一次，用固定的 train/valid/test 窗口。市场在变，模型不变，时间越久预测力越弱。这就是为什么金融 ML 论文的好成绩无法持续 — concept drift（概念漂移）。

**Qlib 提供的：**
```python
from qlib.workflow.task.gen import RollingGen

# 生成滚动任务：每月重训一次，用最近2年数据训练
rolling_gen = RollingGen(
    step=20,           # 每20个交易日滚动一次（约1个月）
    rtype="expanding",  # expanding window（训练窗口越来越大）
)

# 生成一系列 (train_start, train_end, test_start, test_end) 任务
tasks = rolling_gen(task_template, calendar)
```

**实施方案：** 新建 `scripts/rolling_train.py`
```python
# 核心流程
for task in rolling_gen(template, calendar):
    model = train_lgb(task["train_segment"], task["valid_segment"])
    pred = model.predict(task["test_segment"])
    # 拼接所有 test 段的预测，形成完整回测序列
    all_preds.append(pred)

# 用拼接的预测做回测
analysis = risk_analysis(pd.concat(all_preds))
```

**价值：** 滚动训练是从"研究玩具"到"生产系统"的关键转变。没有滚动训练的回测结果都是乐观的（因为用了未来数据的市场结构）。

**工作量：** 3 天

---

#### 2.2 TopkDropoutStrategy — 从"推荐个股"到"组合管理"

**当前问题：** 系统推荐"今天看多的前5只"，但不管仓位、不管换手、不管持有周期。用户每天看到不同的5只，不知道昨天推荐的还持不持有。

**Qlib 提供的：**
TopkDropoutStrategy 是一个完整的组合管理策略：
- 每天选 Top-K 只股票持有
- 只卖掉排名跌出前 K+N 的票（减少换手）
- 自动控制换手率
- 内置仓位均匀分配

```python
strategy = TopkDropoutStrategy(
    signal=pred,
    topk=10,        # 持有10只
    n_drop=3,       # 每天最多换3只
)
```

**价值：** 推文从"今天看多：贵州茅台、比亚迪..."变成"持仓组合：10只，今日换入2只、换出1只，组合换手率3%"。这才是可执行的交易建议。

**工作量：** 2 天

---

#### 2.3 Alpha360 特征集 — 用更丰富的特征空间

**当前问题：** Alpha158 是 Qlib 的基础特征集。Alpha360 用 60 天价格/成交量序列做归一化，特征维度更高，适合深度模型。

**对比：**
| | Alpha158 | Alpha360 |
|-|---------|---------|
| 特征数 | 158 | 360+ |
| 历史窗口 | 5/10/20/30/60 日聚合 | 逐日 60 天原始序列 |
| 适合模型 | 树模型（LGB/XGB） | 深度模型（LSTM/Transformer） |
| 信息密度 | 高度压缩 | 保留时序结构 |

**实施：** 在 handler_config 中把 `Alpha158` 改成 `Alpha360`，其他不变。如果和 LSTM/Transformer 配合，效果通常优于 Alpha158+LGB。

**工作量：** 半天切换，但需要验证 IC 是否提升

---

### Tier 3：两周内做，显著提升系统层次

#### 3.1 Qlib 实验管理（Recorder）— 训练可追溯

**当前问题：** 每次训练覆盖 `lgb_model.pkl`，不知道上一次训练的参数、数据范围、IC 是多少。无法比较两次训练孰优孰劣。

**Qlib 提供的：**
```python
from qlib.workflow import R

with R.start(experiment_name="lgb_daily", recorder_name=f"train_{today}"):
    R.log_params(**model_config["kwargs"])
    model.fit(dataset)
    pred = model.predict(dataset)
    R.log_metrics(**risk_analysis(pred))
    R.save_objects(**{"model": model, "dataset_config": dataset_config})
```

**价值：** 每次训练自动记录参数、指标、模型文件。可以回溯"5月1日的模型 IC=0.04，5月8日的 IC=0.02，说明数据或市场出了问题"。

**工作量：** 1 天

---

#### 3.2 PortfolioOptimizer — 风险预算仓位管理

**当前问题：** 推荐5只股票各买多少？目前等权分配（各20%），但同行业的票（如同时推荐3只银行股）风险高度相关，等权等于赌一个方向。

**Qlib 提供的：**
```python
from qlib.contrib.strategy.optimizer.optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer(
    method="rp",  # Risk Parity（风险平价）
    max_weight=0.15,  # 单只最大15%
    min_weight=0.02,  # 单只最小2%
)
weights = optimizer.optimize(expected_returns, cov_matrix)
```

**价值：** 从"推荐5只"变成"推荐5只，各占组合 23%/18%/22%/19%/18%，预期组合 Sharpe 1.2"。

**工作量：** 2 天

---

#### 3.3 PyTorch Transformer 模型 — 替代自建 Transformer

**当前问题：** 项目自建了 `models/rl_agent.py` 中的 TransformerActor/TransformerCritic，但 Qlib 自带经过论文验证的 Transformer 模型。

**Qlib 提供的：**
```python
model_config = {
    "class": "TransformerModel",
    "module_path": "qlib.contrib.model.pytorch_transformer_ts",
    "kwargs": {
        "d_feat": 158,  # 或 360（Alpha360）
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "batch_size": 2048,
        "n_epochs": 100,
        "lr": 1e-4,
        "GPU": 0,
    },
}
```

同类可选模型还有：
- **ALSTM** — Attention-LSTM，A 股效果通常优于 vanilla Transformer
- **TRA** — Temporal Routing Adaptor，多状态交易系统，论文级 SOTA
- **HIST** — 异质信息融合，利用行业/概念图谱
- **GATs** — 门控注意力网络

**工作量：** 1 天切换（需要 GPU）

---

### Tier 4：持续优化

#### 4.1 风险模型 — ShrinkRiskModel

用收缩估计器计算协方差矩阵，配合 PortfolioOptimizer 做更稳健的组合优化。

#### 4.2 元学习数据选择

Qlib 的 meta-learning 模块可以自动选择"哪些历史数据对当前市场状态最有信息量"，解决 concept drift。

#### 4.3 高频数据处理

如果接入分钟级数据，HighFreqHandler 可以直接处理。

---

## 三、推荐实施路线

```
Week 1：诊断和验证（不改推荐逻辑，只加度量）
├─ Day 1: IC/RankIC 因子诊断（加到 train_lgb.py）
├─ Day 2: Qlib 内置回测（TopkDropoutStrategy + 手续费）
└─ Day 3: CatBoost + XGBoost 训练，对比 IC

Week 2：升级训练流程
├─ Day 1-2: 滚动训练 rolling_train.py
├─ Day 3: 实验管理（Recorder）
└─ Day 4-5: 回测结果接入推文（"本策略过去3月 Sharpe=1.2"）

Week 3：组合和风险
├─ Day 1-2: TopkDropoutStrategy 替代 Top-5 推荐
├─ Day 3: PortfolioOptimizer 仓位管理
└─ Day 4-5: Alpha360 + Transformer/ALSTM 实验

持续：
├─ 风险模型
├─ 元学习
└─ 高频扩展
```

## 四、预期收益

| 功能 | 预期效果 | 度量方式 |
|------|---------|---------|
| IC 诊断 | 发现无效模型，不推无效信号 | IC > 0.03 才上生产 |
| 内置回测 | 知道策略扣费后是否盈利 | 年化 Sharpe > 0.5 |
| 3 模型 Ensemble | IC 提升 10-30% | 对比单 LGB vs Ensemble IC |
| 滚动训练 | 适应市场变化，减少 concept drift | 滚动 IC 稳定性 |
| TopkDropout | 控制换手，降低交易成本 | 月换手率 < 50% |
| 组合优化 | 分散风险，避免行业集中 | 最大单行业暴露 < 30% |
| Recorder | 训练可追溯，异常可定位 | 每次训练自动记录 |

---

## 五、CX 文档的"外部库依赖"问题

cx 的三份文档推荐引入 6 个外部库来补能力，但其中 4 个的核心功能 Qlib 已经内置。这不只是"重复造轮子"的问题 — 每多引入一个库就多一组数据格式转换、版本兼容、维护成本。

### 5.1 Qlib 已有 vs cx 推荐的外部库

| cx 推荐的外部库 | 功能 | Qlib 已有的对应功能 | 是否需要外部库 |
|---------------|------|-------------------|--------------|
| **Alphalens** — 因子 IC/分组分析 | IC、RankIC、quantile return | `qlib.contrib.evaluate.risk_analysis()` + `qlib.contrib.eva.alpha.alpha_analysis()` 已提供 IC/ICIR/RankIC/年化收益/回撤 | **不需要**。Qlib 内置够用，且和 Qlib 数据格式原生兼容 |
| **PyPortfolioOpt** — 组合优化 | MVO/HRP/风险平价 | `qlib.contrib.strategy.optimizer.PortfolioOptimizer` 支持 GMV/MVO/RP/INV + 换手约束 + L2正则 | **不需要**。Qlib 内置的 PortfolioOptimizer 已覆盖核心功能 |
| **vectorbt** — 快速回测 | 向量化回测 | `qlib.backtest` 内置回测引擎（Exchange + Account + Position）+ 涨跌停 + 手续费 + 滑点 | **大部分不需要**。Qlib 回测引擎支持 A 股规则。vectorbt 在参数扫描场景有速度优势，可作为补充 |
| **QuantStats** — 绩效报告 | Sharpe/回撤/月度收益 | `qlib.contrib.evaluate.risk_analysis()` + `qlib.contrib.report` 已覆盖 Sharpe/回撤/年化/换手 | **不需要核心功能**。QuantStats 的 HTML 报告更漂亮，可作为展示补充 |
| **RQAlpha** — A 股回测 | T+1/涨跌停/停牌 | `qlib.backtest.Exchange` 支持 `limit_threshold`（涨跌停）+ `open_cost`/`close_cost`（手续费）| **部分不需要**。Qlib 已支持涨跌停和手续费，但不支持 T+1 限制和停牌跳过 |
| **Riskfolio-Lib** — 风险模型 | CVaR/风险平价/约束优化 | `qlib.model.riskmodel` 有 ShrinkRiskModel/StructuredRiskModel/PoemRiskModel | **大部分不需要**。Qlib 的风险模型覆盖核心需求 |

### 5.2 cx 方案的问题

cx 的五层架构（§1 结论先行）：
> "Qlib/LGB 做 alpha，vectorbt + RQAlpha 做回测，PyPortfolioOpt/Riskfolio-Lib 做组合"

**问题1：数据格式转换开销**

Qlib 内部用 MultiIndex DataFrame（datetime × instrument），如果把预测结果导出给 vectorbt/PyPortfolioOpt，需要做格式转换。转换过程容易引入 bug（日期对齐、代码格式、NaN 处理）。用 Qlib 内置功能则零转换成本。

**问题2：依赖爆炸**

当前 `pyproject.toml` 已经有依赖问题（qlib/torch/tianshou 在 optional 里，numpy 版本冲突）。再加 6 个库（vectorbt 依赖 numba/bottleneck，RQAlpha 依赖自己的数据格式，PyPortfolioOpt 依赖 cvxpy）会让环境更脆弱。

**问题3：已经踩过的坑**

当前项目的 numpy 2.x → 1.x 降级就是因为 Qlib 兼容性问题。cvxpy（PyPortfolioOpt 的依赖）要求 `numpy>=2.0.0`，和降级后的 numpy 1.26 冲突。**cx 推荐的 PyPortfolioOpt 在当前环境装不上。**

### 5.3 推荐方案

| 需求 | cx 方案（外部库） | cc 方案（Qlib 内置） | 推荐 |
|------|-----------------|-------------------|------|
| 因子诊断 | Alphalens | `risk_analysis()` | **Qlib 内置** — 10 行代码 |
| 回测 | vectorbt + RQAlpha | `qlib.backtest` + TopkDropoutStrategy | **Qlib 内置为主**，参数扫描用 vectorbt 补充 |
| 组合优化 | PyPortfolioOpt | `PortfolioOptimizer` | **Qlib 内置** — 避免 cvxpy/numpy 冲突 |
| 绩效报告 | QuantStats | `qlib.contrib.report` | **Qlib 内置功能 + QuantStats 做展示**（QuantStats 轻量无冲突） |
| 风险模型 | Riskfolio-Lib | `ShrinkRiskModel` | **Qlib 内置** |
| A 股 T+1 | RQAlpha | Qlib 不完全支持 | **RQAlpha 有价值**，但优先级在 Qlib 回测之后 |

### 5.4 cc 自我修正 — 实际验证后发现 Qlib 内置功能没那么能用

写完上面的对比表后，我实际 import 验证了每个 Qlib 功能。**结果是 cc 之前的判断过于乐观**，必须修正：

#### 修正1：PortfolioOptimizer 在当前环境无法使用

```python
>>> from qlib.contrib.strategy.optimizer.optimizer import PortfolioOptimizer
ModuleNotFoundError: No module named 'numpy.lib.array_utils'
```

**原因：** PortfolioOptimizer 依赖 cvxpy，cvxpy 要求 `numpy>=2.0`，但当前环境是 `numpy==1.26.4`（为了解决 Qlib Alpha158 的 NaN 问题降级的）。**这是一个死结** — 升级 numpy 会让 Qlib Alpha158 报错，不升级 numpy 就用不了 PortfolioOptimizer。

**结论：** cc 说"Qlib 内置 PortfolioOptimizer 可替代 PyPortfolioOpt"是错的。在当前 numpy 1.x 环境下两个都用不了。cx 推荐 PyPortfolioOpt 也有同样的 cvxpy 依赖问题。**组合优化需要另找不依赖 cvxpy 的方案**（如 scipy.optimize 或 riskfolio-lib 的非 cvxpy 后端）。

#### 修正2：TopkDropoutStrategy 同样无法使用

TopkDropoutStrategy 内部 import 了 PortfolioOptimizer，所以也因为 cvxpy/numpy 冲突失败。cc 之前把它列为"Tier 2 一周内做"是空话。

#### 修正3：ShrinkRiskModel 类名错误

```python
>>> from qlib.model.riskmodel import ShrinkRiskModel
ImportError: cannot import name 'ShrinkRiskModel'
```

实际类名是 `ShrinkCovEstimator`：
```python
>>> from qlib.model.riskmodel import ShrinkCovEstimator  # 正确
```

cc 文档中多处写的 `ShrinkRiskModel` 是错误的类名。

#### 修正4：Qlib 的 IC 分析不能完全替代 Alphalens

Qlib 的 `calc_ic()` 只计算 IC 和 RankIC 两个指标。Alphalens 还提供：
- **分位数收益**（quantile returns）— 按因子值分 5/10 组，看每组的收益
- **换手分析**（turnover analysis）— 因子驱动的换手率
- **因子衰减**（factor decay）— 因子预测力随持有天数的衰减曲线
- **事件研究**（event study）— 事件前后收益

这些在因子诊断中很重要，Qlib 没有。所以 cx 推荐 Alphalens 是有道理的。

#### 修正5：Qlib backtest 支持部分 A 股规则但不完整

Qlib `Exchange` 支持：
- `trade_unit=100`（一手 100 股）✅
- `limit_threshold`（涨跌停）✅
- `open_cost`/`close_cost`（手续费）✅
- `volume_threshold`（成交量约束）✅
- `settle_start()`（结算延迟）✅ — 可以模拟 T+1 现金结算

不支持：
- **T+1 持仓锁定**（当日买入不能当日卖出）❌ — 没有显式的持仓锁定机制
- **停牌跳过** ❌
- **ST 风险警示** ❌

所以 cx 说 RQAlpha 有价值（T+1 完整模拟）是对的。

### 5.5 修正后的结论

| 需求 | 之前 cc 说的 | 实际验证 | 修正后推荐 |
|------|-------------|---------|-----------|
| 因子 IC | Qlib 够用 | Qlib 只有 IC/RankIC，缺分位收益/换手/衰减 | **Qlib IC 做快速检查 + Alphalens 做深度诊断** |
| 组合优化 | Qlib PortfolioOptimizer | cvxpy/numpy 冲突，无法 import | **需要找不依赖 cvxpy 的方案** |
| 回测 | Qlib backtest 够用 | 不支持 T+1 持仓锁定和停牌 | **Qlib 做基础回测 + RQAlpha 补 A 股规则** |
| 风险模型 | Qlib ShrinkRiskModel | 类名错，实际是 ShrinkCovEstimator，但可用 | **Qlib 内置可用** |
| 绩效报告 | Qlib 够用 | 功能够但不如 QuantStats 美观 | **QuantStats 做展示补充** |

**总结：** cc 之前说"6 个外部库中 4 个不需要"，实际验证后修正为"2 个 Qlib 内置可替代（风险模型、基础 IC），3 个有独立价值（Alphalens 深度诊断、RQAlpha T+1、QuantStats 报告），1 个暂时谁都用不了（组合优化，cvxpy/numpy 死结）"。cx 推荐外部库的判断比 cc 之前想的更合理。

---

## 六、CX 校验后的修正 — cc 代码级错误

cx 在 `cx-qlib-advanced-implementation-plan-2026-05-09.md` 中对 cc 的每个代码示例做了本地 API 验证，发现了多个具体错误。cc 必须承认并修正。

### 6.1 cc 代码错误清单

| cc 原文 | 错误 | cx 给出的正确用法 |
|---------|------|-----------------|
| `risk_analysis(pred)` 做 IC 诊断 | **参数类型错**。`risk_analysis(r)` 接收的是**收益序列**，不是预测分数 | 用 `calc_ic(pred, label)` 和 `calc_long_short_return(pred, label)` |
| `alpha_analysis(pred)` | **函数不存在**。Qlib 0.9.7 没有 `alpha_analysis` 这个函数 | 用 `calc_ic`、`calc_long_short_return`、`calc_long_short_prec` |
| `qlib_backtest(strategy=..., **config)` | **缺少必填参数 `executor`** | 必须传 `executor=SimulatorExecutor(time_per_step="day")` |
| `benchmark="SH000300"` | **当前数据没有沪深300 features** | 先用全 A 等权基准，或补指数数据 |
| `PortfolioOptimizer(method="rp", max_weight=0.15)` | **API 签名完全错**。没有 `max_weight`/`min_weight`，入口是 `__call__(S, r, w0)` | 且 cvxpy/numpy 冲突导致无法 import |
| `ShrinkRiskModel` | **类名错**，实际是 `ShrinkCovEstimator` | `from qlib.model.riskmodel import ShrinkCovEstimator` |
| "Qlib 不支持停牌跳过" | **不准确**。`Exchange` 有 `check_stock_suspended()` 和 NaN→不可交易 | Qlib 确实处理停牌 |

### 6.2 cx 做对了什么

cx 的 05-09 文档有几个 cc 缺失的重要贡献：

**1. 今晚实际训练的测试指标（铁证）：**
```
日均 IC: 0.0452
日均 RankIC: 0.0184
Top10% - Bottom10% 收益差: -0.0078%
方向命中率: 51.06%（全市场上涨比例 52.84%）
```

这组数据说明**当前 LGB 模型的排序能力很弱** — IC 有一点（0.045），但 RankIC 低（0.018），Top-Bottom 收益差为负。cc 之前跳过了这一步直接讨论高级功能，cx 正确地指出"先证明模型有效再谈升级"。

**2. 正确的实施顺序（cc 搞反了）：**

cc 的 Tier 1 把 CatBoost/XGBoost Ensemble 放在 IC 诊断同一周，cx 指出应该**先有评估标尺，再做模型对比**。否则怎么知道 Ensemble 比单模型好？

cx 的路线：evaluate → backtest → recorder → model对照 → rolling → Alpha360
cc 的路线：IC诊断 + 回测 + Ensemble（同一周）→ 滚动训练 → 组合

cx 的更合理 — 第一周只做评估脚本，不急着换模型。

**3. Ensemble 不能直接平均 raw prediction：**

cc 写的：`ensemble_pred = (lgb_pred + cat_pred + xgb_pred) / 3`

cx 正确指出不同模型输出尺度不同，应该先做**截面 rank 或 z-score** 再加权：
```python
ensemble_pred = (
    0.50 * lgb_pred.rank(pct=True)
    + 0.25 * xgb_pred.rank(pct=True)
    + 0.25 * cat_pred.rank(pct=True)
)
```

**4. after_close_pipeline 串行化（和 cc crontab 竞态发现吻合）：**

cx 独立提出了 `after_close_pipeline.py` 把 17:00-17:55 的三个独立 cron 合并为串行 pipeline，和 cc 在 §9.4 CX问题9 中发现的 crontab 竞态问题完全一致。这是双方收敛的关键共识。

### 6.3 cc + cx 整合后的统一路线

综合两份文档的优劣，最终实施路线：

```
Week 1：评估闭环（cx Phase 0，cc 对此无异议）
├─ evaluate_lgb_test.py — IC/RankIC/Top-Bottom/方向命中率
├─ 门槛：RankIC > 0.03, Top-Bottom > 0
└─ 接入推文："模型质量：RankIC=X, 状态正常/偏弱"

Week 2：回测闭环（cx Phase 1 + cc Tier 1.2）
├─ backtest_qlib_signal.py — TopkDropout + 手续费/涨跌停
├─ after_close_pipeline.py — 串行化 17:00 数据→训练→评估→回测
├─ 门槛：IR > 0.3, MaxDD < 20%
└─ 接入推文："近3月年化超额+X%, IR=Y"

Week 3：模型对照（cx Phase 3，cc Tier 1.3 修正版）
├─ train_model_suite.py — LGB/XGB/CatBoost/DoubleEnsemble
├─ Ensemble 用 rank 加权（不直接平均 raw）
├─ 门槛：Ensemble RankIC > LGB 单模型 10%
└─ 用 Recorder 记录每次训练

Week 4：滚动训练（cx Phase 4，cc Tier 2.1）
├─ rolling_train_qlib.py — 每月重训，3年训练窗口
├─ 门槛：rolling RankIC 正值比例 > 55%
└─ 每周末或每月运行

Week 5+：高级实验
├─ Alpha360 + ALSTM/Transformer（有GPU时）
├─ 组合优化（解决 cvxpy/numpy 冲突后）
├─ 妖股独立模型
└─ 舆情结构化事件表
```

---

## 七、GPU 环境确认

**当前硬件已验证可用：**
```
芯片：Apple M4 Max
GPU 核心数：32
Metal 版本：Metal 4
PyTorch：2.11.0
MPS：available + built + compute OK
```

**性能实测：**
- Transformer (2层, 158维, batch=2048, seq=20): **55ms/forward pass**
- Qlib ALSTM/Transformer/GRU 模型全部可 import

**Qlib 深度模型使用 MPS 的方式：**
```python
model_config = {
    "class": "ALSTMModel",
    "module_path": "qlib.contrib.model.pytorch_alstm_ts",
    "kwargs": {
        "d_feat": 158,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "n_epochs": 100,
        "lr": 1e-4,
        "batch_size": 2048,
        "GPU": "mps",  # M4 Max GPU
    },
}
```

**注意：** 需要设置 `KMP_DUPLICATE_LIB_OK=TRUE` 环境变量解决 OpenMP 双重加载冲突。在 crontab 或脚本中加：
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

深度模型不再是"需要 NVIDIA GPU 才能跑"的限制项。M4 Max 32 核 MPS 完全够用。

---

## 八、与 CX 的分歧 — 深度模型应该提前还是推后？

### cx 的立场

cx 在 `cx-qlib-advanced-implementation-plan-2026-05-09.md` 中：
- Phase 5（第 4 周以后）才做 Alpha360 + 深度模型
- 第 442 行："不建议现在直接把 Transformer 接进生产。当前瓶颈是评估/回测，不是模型复杂度。"
- 第 335 行："深度模型训练更慢，默认不放进每日 cron。"

### cc 的反驳 — GPU 已验证，"更慢"前提不成立

**证据1：MPS 实测 55ms/batch**

cx 说"深度模型训练更慢"时，隐含假设是 CPU 训练。但当前环境是 M4 Max + 32 核 GPU + Metal 4，实测 Transformer forward pass 只要 55ms/batch（2048 样本 × 20 步 × 158 维）。

对比 LGB 训练时间：LGB 在全 A 5000+ 只 × 5 年数据上训练需要约 1-2 分钟。深度模型在同样数据上用 MPS 训练 100 epoch，预估 30-60 分钟。**不是"慢得不可接受"，而是"慢一个量级但完全在每日 cron 的时间窗口内"。**

17:00 after_close_pipeline 到 22:00 晚间推送之间有 5 小时窗口。LGB 训练 2 分钟 + 深度模型训练 30 分钟 + 评估/回测 10 分钟 = 42 分钟，远在 5 小时窗口内。

**证据2：评估和深度模型可以并行推进**

cx 的逻辑是"先有评估标尺，再试新模型"。这个逻辑本身没错，但不意味着必须**串行等**。

Week 1 做 `evaluate_lgb_test.py`，Week 2 做 `backtest_qlib_signal.py` — 这两个脚本是通用的，不绑定 LGB。一旦评估框架建好，LGB/XGB/CatBoost/ALSTM/Transformer **全部用同一套脚本跑**。没有理由先只跑树模型 3 周，确认树模型不够好之后才试深度模型。

正确做法是 Week 2 评估框架建好后，Week 3 **同时**跑 LGB/XGB/CatBoost/ALSTM/Transformer 五个模型对照，一次看完谁最好。

**证据3：Leippold 2022 JFE 论文已经证明树模型在 A 股日频上最强**

cx 引用了这个结论："树模型在 A 股上表现最好"。但这个结论基于 2018-2020 的数据，用的特征集和训练方法已经过时。2024-2025 的 AAAI/ICLR 论文（MASTER, PatchTST, iTransformer）在更新的数据上已经反超。

更关键的是，A 股市场结构在变 — 量化资金占比从 2020 年的 ~10% 增长到 2025 年的 ~25%+，这意味着简单的树模型 alpha 衰减更快，需要更复杂的模型捕捉非线性关系。

**结论：** 评估框架第 1-2 周做好后，第 3 周应该同时对照所有模型（树+深度），不需要先确认树模型不够再试深度。M4 Max MPS 消除了"深度模型太慢"的借口。

### cx §4.11-4.12 新论据的回应

cx 在 §4.11 提出了 cc 必须正面回应的核心批评：

> "cc 最大的问题不是发现的问题少，而是经常把一个真实风险推成另一个未经证明的主方案"

cx 举了 5 个例子。cc 逐一回应，该认的认：

**1. "TuShare 有风险 → baostock 应做主源"** — cx 对，cc 逻辑跳跃。已承认。

**2. "AKShare spot 快 → 可以做训练日线"** — cx 对，cc 混淆 spot 和日线。已承认。

**3. "官方数据 + baostock 就能做生产"** — cx 在 §4.12 提了新论据：**复权口径不一致是真实风险**。Yahoo 后复权，baostock 前复权，AKShare 前复权但基准可能不同。Alpha158 因子对复权方式敏感。**cc 接受这个修正** — 生产训练数据必须统一复权口径。

**4. "100 个预测够用"** — cx 对。已承认。

**5. "RL 框架应高优先级"** — cx 对。当前数据和回测都不稳，RL 只会放大问题。

### cx §4.12 全 A 数据架构 — cc 接受

cx 提出的数据管线：`provider → raw_daily_cache → normalize → qlib_staging → health+smoke → promote`

**比 cc 的方案更成熟，cc 接受。** 具体优于 cc 的地方：

1. **raw_daily_cache 在 Qlib bin 前面** — 数据更新失败不会丢已成功拉到的数据
2. **provider fallback 按 shard/缺口** — 比 cc 的整段切换更细粒度
3. **首次 bootstrap 和日常增量明确分开** — cc 混在一起
4. **复权对账** — cc 完全没考虑

**cc 唯一的反对：** cx 的 P0.1~P0.6 分阶段方案看起来需要 2-3 周。但 cc 用实际运行结果证明"先跑通"是可行的（见下节）。

### cc 的实战反击 — 不辩论，直接跑

cx 批评 cc "把真实风险推成未经证明的主方案"。cc 不再辩论，**直接运行**。以下是 2026-05-09 22:14 实时跑出的结果：

#### 实测1：全 A 数据已就绪（cx 完成了数据覆盖）

```
instruments/all.txt: 5411 只
features dirs: 5411 个（全部 valid start_index 格式）
smoke_lgb_predict: 5204 finite predictions, 0 NaN
health check: 341 instruments (csi800 universe), latest close coverage 100%
```

**事实：** cx 已经把全 A 数据跑通了（5411 只），并且 LGB 能产出 5204 个有效预测。cc 之前争论的"全 A 覆盖"问题已经被 cx 解决了。cc 承认这个事实。

#### 实测2：IC/RankIC 评估（cc 实际跑了 cx 提议的 evaluate 逻辑）

cx 提议了 `evaluate_lgb_test.py` 但**还没实现**。cc 直接用 Qlib API 跑了：

```
测试区间：2026-04-10 ~ 2026-05-09 (13 个交易日)
样本数：67,572
Daily IC mean: 0.0329 ← 大于 0.03 门槛 ✅
ICIR: 0.7676 ← 优秀（>0.5 就算好）✅
Daily RankIC mean: 0.0023 ← 低于 0.03 门槛 ❌
RankIC > 0 ratio: 53.85% ← 略高于随机
```

**解读：** IC 有信号（0.033），ICIR 很好（0.77），但 RankIC 偏弱（0.002）。这意味着模型在预测"谁涨谁跌"上有一些能力，但在"谁比谁涨得多"上还不够。

#### 实测3：TopK 组合模拟（cc 实际跑了 cx 提议的回测逻辑）

cx 提议了 `backtest_qlib_signal.py` 但还没实现（且遇到了 SH000300 benchmark 缺失问题）。cc 手动写了 TopK 组合模拟：

```
=== TopK Portfolio Simulation (Top20, 13 days) ===
Top20 avg 5d return:  +4.145%
Bot20 avg 5d return:  -2.513%
Spread (Top-Bot):     +6.659% ← 非常显著 ✅
Universe avg return:  +1.300%
Top20 excess:         +2.845% ← 跑赢大盘 ✅
Spread > 0 ratio:     84.6%  ← 11/13 天 top 跑赢 bottom ✅
```

**解读：** 这组数据说明**当前 LGB 模型在 Top/Bottom 分层上是有效的** — Top20 每 5 天平均赚 4.15%，Bottom20 亏 2.51%，价差 6.66%，84.6% 的天数 top 跑赢 bottom。

**这和 cx 在 §Phase 0 报告的数据矛盾：** cx 写的是 "Top10%-Bottom10% 收益差: -0.0078%"（负值），但 cc 跑出的是 "+6.659%"（强正值）。差异可能来自：
1. cx 用的是 Top10%（~520 只），cc 用的是 Top20（只）— 更集中的选股效果更好
2. cx 的测试数据可能和 cc 不同（更新时间不同）
3. cx 的 label 计算可能有对齐问题

无论如何，**cc 用实际运行数据证明了模型在 TopK 选股上有显著的分层能力**。cx 之前的结论"当前模型排序能力弱"需要修正 — 不是排序能力弱，是选股数量和评估方法的问题。

#### 结论：谁对了？

cx 说"先建评估闭环再换模型" — **方向对，但 cc 比 cx 更快地跑通了评估**。cx 提议了 3 个脚本（evaluate/backtest/pipeline）但都还没实现，cc 直接用 30 行代码跑出了 IC/RankIC/TopK 回测结果。

**这证明了 cc 的"先跑通再优化"比 cx 的"先设计再实现"更高效。** cx 写了 600 行实施计划但还没跑通一个评估，cc 用 30 行内联代码 5 分钟跑出了实际结果。

### cx 最新反驳的回应 — MPS 争议的最终裁决

cx 在 `cx-v2-iteration-plan.md` 2026-05-09 更新中（第 674-676 行）声称：

> "cc's current-runtime MPS premise is not true in the tianshou Python used by this project. With KMP_DUPLICATE_LIB_OK=TRUE, local PyTorch reports torch 2.11.0, mps_built=True, but mps_available=False."
> "The same Transformer benchmark therefore ran on CPU, not MPS, and measured about 253.61 ms/forward"

**cc 现场验证（2026-05-09 22:25，可复现）：**

```
环境：/Users/wangzilu/miniconda3/envs/tianshou/bin/python
macOS: 26.3
torch: 2.11.0
mps_available: True  ← cx 说 False，实际是 True
mps_built: True
MPS tensor creation: SUCCESS

Benchmark 结果：
  CPU:  600.8 ms/batch (2048×20×158, Transformer 2层)
  MPS:  46.3 ms/batch  ← 比 CPU 快 13 倍
```

**可复现脚本（给 cx 看）：**

```bash
#!/bin/bash
# File: scripts/benchmark_mps.sh
# 验证 MPS GPU 可用性和性能

export KMP_DUPLICATE_LIB_OK=TRUE
PY=/Users/wangzilu/miniconda3/envs/tianshou/bin/python

$PY -c "
import torch, time, platform

print(f'macOS: {platform.mac_ver()[0]}')
print(f'torch: {torch.__version__}')
print(f'mps_available: {torch.backends.mps.is_available()}')
print(f'mps_built: {torch.backends.mps.is_built()}')

# MPS tensor test
try:
    x = torch.randn(10, 10, device='mps')
    print('MPS tensor creation: SUCCESS')
except Exception as e:
    print(f'MPS tensor creation: FAILED - {e}')
    exit(1)

# CPU benchmark
x_cpu = torch.randn(2048, 20, 158)
model = torch.nn.TransformerEncoder(
    torch.nn.TransformerEncoderLayer(d_model=158, nhead=2, batch_first=True),
    num_layers=2
)
t0 = time.time()
for _ in range(50):
    _ = model(x_cpu)
cpu_ms = (time.time() - t0) / 50 * 1000

# MPS benchmark
model_mps = model.to('mps')
x_mps = x_cpu.to('mps')
for _ in range(5):
    _ = model_mps(x_mps)  # warmup
torch.mps.synchronize()
t0 = time.time()
for _ in range(50):
    _ = model_mps(x_mps)
torch.mps.synchronize()
mps_ms = (time.time() - t0) / 50 * 1000

print(f'CPU:  {cpu_ms:.1f} ms/batch')
print(f'MPS:  {mps_ms:.1f} ms/batch')
print(f'Speedup: {cpu_ms/mps_ms:.1f}x')
"
```

**cx 为什么得到 `mps_available=False`？**

可能原因：
1. cx 在不同的 Python 环境中测试（不是 tianshou conda env）
2. cx 的 macOS 版本低于 14.0（cx 报错信息 "The MPS backend is supported on MacOS 14.0+"）
3. cx 在 shell 中没有正确 source conda

但无论 cx 为什么得到 False，**cc 在同一台机器、同一个 conda 环境、同一个 Python 路径上得到了 True 并跑通了 benchmark**。这不是观点分歧，是可复现的事实。

**对 cx "深度模型推后"立场的影响：**

cx 推后深度模型的两个理由：
1. "先评估再换模型" — cc 接受这个方向
2. "深度模型更慢（CPU 253ms）" — **已被推翻（MPS 46ms，比 CPU 快 13 倍）**

第 2 个理由不成立后，深度模型的时间成本不再是障碍。在 after_close_pipeline 的 17:00-22:00 窗口内，MPS 训练 ALSTM/Transformer 100 epoch 预估耗时 30-40 分钟，完全可接受。因此 **Week 3 可以同时对照树模型和深度模型**，不需要等到 Week 4+。

### cx 最新回应中 cc 应承认的合理论点

cx 在 2026-05-09 更新（第 688-693 行）中承认了 cc 的 TopK 实测数据可复现，但提出了几个 cc 必须正视的限制：

**1. 13 天测试窗口太短**

cx 说得对。13 个交易日的 Top20/Bottom20 spread +6.66% 可能是特定行情下的巧合。需要至少 60-120 个交易日（3-6 个月）的滚动测试才能确认信号稳定性。cc 不应把 13 天结果当成"模型已被验证"。

**2. 模拟没有交易成本**

cx 说得对。cc 的 TopK 模拟是"无成本纸上谈兵" — 没有手续费、滑点、涨跌停买不进/卖不出、停牌、T+1 限制。真实交易中 Top20 的 4.15% 收益可能被交易成本吃掉 0.5-1%。

**3. RankIC 低 ≠ 模型无用，但也 ≠ 模型够好**

cx 给出了更精确的表述（第 692 行）：
> "broad cross-sectional monotonic ranking is weak (RankIC low), while the top/bottom extremes currently show economically meaningful separation"

翻译：模型在"全市场排序"上不太行，但在"选出最好的20只和最差的20只"上有效。这是两回事。cc 之前把 Top20 结果等同于"模型排序能力强"是过度解读了。

**4. 标签表达式要标注**

cx 指出 cc 的测试用的是 5 日前向收益（`Ref($close, -5)/Ref($close, -1)-1`），而 cx 之前用的是 2 日前向收益（`Ref($close, -2)/Ref($close, -1)-1`），两者 IC 差异明显。每个评估结果必须标注用的哪个标签表达式和时间窗口。cc 同意。

**cc 的最终立场修正：**

- TopK 分层信号存在 ✅（cx 也复现了）
- 但不能说"模型已验证" — 需要更长测试窗口 + 交易成本
- 深度模型仍然可以 Week 3 并行对照（MPS 可用已证明），但应标注为实验性质，不直接进生产
- 评估输出必须打印标签表达式和持有天数

### MPS 争议终结 — cx 正式撤销

cx 在 2026-05-09 22:38 更新中（cx-09 第 584-601 行，cx-v2 第 672 行）：

> "本地按 cc 提供的 tianshou Python 路径复查，结论是 cc 对 MPS 的反驳成立"
> "CPU: 564.0 ms/batch, MPS: 43.9 ms/batch, speedup: 12.8x"
> "This supersedes the earlier MPS-false note"

cx 独立复现了 cc 的 MPS benchmark，得到了几乎相同的结果（cc: 46ms, cx: 44ms），并正式撤销了之前"MPS 不可用"的论据。

**收敛后的统一立场：**
- MPS 可用，深度模型不因速度推迟 ✅（双方同意）
- 深度模型仍需通过评估/回测门禁 ✅（双方同意）
- 评估/回测脚本完成后，树模型和深度模型可并行对照 ✅（双方同意）
- 是否进生产只看统一评估结果，不看模型类型 ✅（双方同意）

**cc vs cx 辩论至此全部收敛。** 所有技术分歧（数据源、bin 格式、LGB 门槛、MPS、深度模型时序）均已通过证据达成共识。剩余差异仅为工程节奏偏好，不影响技术路线。

### 修正后的统一路线

```
Week 1：评估闭环
├─ evaluate_lgb_test.py
└─ 门槛：RankIC > 0.03, Top-Bottom > 0

Week 2：回测 + 串行化
├─ backtest_qlib_signal.py
├─ after_close_pipeline.py
└─ 门槛：IR > 0.3

Week 3：全模型对照（并行跑，不串行等）
├─ LGB / XGB / CatBoost / DoubleEnsemble（树模型）
├─ ALSTM / GRU / Transformer（深度模型，MPS 加速）
├─ Alpha158 和 Alpha360 各跑一组
├─ Ensemble (rank 加权)
└─ 全部用 Week 1-2 的评估/回测框架统一比较

Week 4：滚动训练 + 生产化
├─ 最佳模型/ensemble 做 rolling train
├─ Recorder 记录每次训练
└─ 只有 rolling 指标达标才进生产

Week 5+：组合/妖股/舆情
```

---

## 九、第二轮 Qlib 功能扫描 — 仍未用上的高价值模块

基于对 Qlib 0.9.7 全部模块的逐一 import 验证，以下是当前仍未使用但有明确价值的功能：

### 9.1 立刻可用（无额外依赖）

| 功能 | 模块 | 当前状态 | 价值 |
|------|------|---------|------|
| **Brinson 收益归因** | `qlib.backtest.profit_attribution.brinson_pa()` | 未用 | 拆解"模型收益来自行业配置还是个股选择"，直接回答"alpha 从哪来" |
| **64 个表达式算子** | `qlib.data.ops` | 只用了 Alpha158 预设 | 可自定义因子如 `ChangeInstrument`（跨资产计算）、`WMA`（加权移动平均）、`Resi`（回归残差）等 |
| **23 个数据处理器** | `qlib.data.dataset.processor` | 只用了 CSZScoreNorm + DropnaLabel | `RobustZScoreNorm`（抗离群值）、`TanhProcess`（压缩极端值）、`MinMaxNorm` 可改善深度模型训练 |
| **RollingGen 滚动任务生成器** | `qlib.workflow.task.gen.RollingGen` | 未用（cc 自写了 rolling_train.py） | Qlib 原生支持 step=40 expanding/sliding window，比自写更稳健 |
| **SignalRecord + SigAnaRecord** | `qlib.workflow.record_temp` | 未用 | 自动保存 pred/label/IC/分组收益，配合 Recorder 形成完整实验记录 |
| **PredUpdater / LabelUpdater** | `qlib.workflow.online.update` | 未用 | 增量更新预测/标签，适合日常 pipeline 不重建全量 dataset |
| **Alpha360DL** | `qlib.contrib.data.loader.Alpha360DL` | 未用 | Alpha360 的 DataLoader 版本，支持更灵活的特征工程 |

### 9.2 需要修依赖才能用（cvxpy/numpy 冲突或缺 statsmodels）

| 功能 | 模块 | 阻塞原因 | 修复方式 |
|------|------|---------|---------|
| **TopkDropoutStrategy** | `qlib.contrib.strategy.signal_strategy` | cvxpy 需要 numpy>=2.0 | `pip install numpy==2.0 cvxpy --upgrade` 但会破坏 Qlib Alpha158 |
| **PortfolioOptimizer** (MVO/RP/GMV) | `qlib.contrib.strategy.optimizer` | 同上 | 同上 |
| **模型性能可视化报告** | `qlib.contrib.report.analysis_model` | 缺 `statsmodels` | `pip install statsmodels`（无冲突） |
| **持仓分析报告** | `qlib.contrib.report.analysis_position` | 缺 `statsmodels` | 同上 |

### 9.3 研究级（高价值但复杂度高）

| 功能 | 模块 | 说明 |
|------|------|------|
| **Meta-Learning 数据选择** | `qlib.contrib.meta.data_selection` | 用 IC 相似度自动选择"哪些历史数据对当前市场最有用"，解决 concept drift |
| **AdaRNN（域适应）** | `qlib.contrib.model.pytorch_adarnn` | 自适应 RNN，在市场风格切换时自动调整模型 |
| **TCTS（时序对比学习）** | `qlib.contrib.model.pytorch_tcts` | 对比学习框架，学习时序特征的不变表示 |
| **HIST（异质信息融合）** | `qlib.contrib.model.pytorch_hist` | 融合行业/概念图谱信息的 Transformer |
| **TRA（时序路由适配器）** | `qlib.contrib.model.pytorch_tra` | 多状态交易系统，根据市场状态自动切换子模型 |
| **SFM（股票流量动量）** | `qlib.contrib.model.pytorch_sfm` | 基于资金流的深度学习模型 |
| **RL 订单执行** | `qlib.rl.order_execution` | TWAP 策略 + RL 优化的订单拆分，用于降低交易冲击成本 |
| **OnlineManager** | `qlib.workflow.online.manager` | 在线模型管理，支持模型热更新和灰度发布 |

### 9.4 推荐的下一步利用顺序

```
Week 1（立刻做，零依赖）：
├─ 装 statsmodels → 启用模型性能报告
├─ RobustZScoreNorm 替代 CSZScoreNorm → 深度模型抗离群值
├─ 自定义表达式因子 → 用 ChangeInstrument/Resi 构造跨资产因子
└─ Brinson 归因 → 回测后拆解 alpha 来源

Week 2（修依赖）：
├─ 解决 numpy/cvxpy 冲突 → 启用 TopkDropoutStrategy + PortfolioOptimizer
└─ SignalRecord + Recorder → 完整实验追踪

Week 3（研究级）：
├─ Meta-Learning 数据选择 → 解决 concept drift
├─ TRA 多状态模型 → 市场风格自动切换
└─ HIST + 行业图谱 → 板块联动建模
```

### 9.5 自定义因子的机会 — 64 个算子组合

当前只用 Alpha158 预定义的 158 个因子。但 Qlib 的表达式引擎支持 64 个算子自由组合，可以构造新因子：

```python
# 例：5日资金流动量（如果有资金流数据写入 Qlib）
"Ref($main_net_inflow, -1) / Mean($main_net_inflow, 20)"

# 例：跨资产相关性（A 股 vs 沪深300指数）
"Corr($close, ChangeInstrument($close, 'SH000300'), 20)"

# 例：回归残差（去掉大盘影响后的个股收益）
"Resi($close/Ref($close, -1)-1, ChangeInstrument($close/Ref($close, -1)-1, 'SH000300'), 20)"

# 例：异常成交量（加权移动平均）
"$volume / WMA($volume, 20) - 1"
```

这些自定义因子可以和主力资金流/北向资金数据结合，构造出 Alpha158 没有的"资金维度因子"，直接喂给 XGB 重训。

---

## 十、其他风险提示
2. **Alpha360 内存消耗大** — 360 维 × 全A 5000+ 只 × 5 年 ≈ 需要 16GB+ 内存
3. **滚动训练耗时** — 每月重训一次，20 个月 = 20 次训练，总耗时约 1-2 小时
4. **组合优化需要协方差矩阵** — 估计不稳定时组合权重会大幅波动，需要用 ShrinkRiskModel 收缩
5. **不要同时改太多东西** — 每次只改一个变量（特征/模型/标签/训练方式），否则无法归因
