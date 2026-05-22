# Institutional Quant Development Roadmap

**Date:** 2026-05-20  
**Author:** CX  
**Scope:** 把当前 `xgb_174` champion 从“强信号研究原型”推进到更接近百亿私募口径的机构化量化系统。  
**原则:** 不追神秘模型，不靠单窗口漂亮数字。先把评估、组合、风控、容量和治理做硬，再扩因子、深度模型、LLM 和 RL。

---

## 0. 当前基线与核心判断

当前 champion 是 `xgb_174`。本地证据：

- `data/storage/phase4/model_registry.json`：`xgb_174` 是 champion；`xgb_205` 因 regime negative control 失败降为 `research_only`。
- `plans/cc-phase4-final-summary-2026-05-19.md`：24-split rolling gate 通过，avg RankIC `+0.0513`，avg Spread `+2.51%`，RankIC>0 `20/24`，Spread>0 `21/24`。
- `logs/phase4_rolling_gate_174_3yr.log`：24 split 日志里，IC 均值约 `+0.0456`，ICIR 均值约 `+0.71`，RankIC 均值约 `+0.0513`。
- `data/storage/phase4_backtest_xgb_174_top20.json`：最新 6 个月成本后年化 `+37.6%`、Sharpe `1.79`，但平均持仓只有 `4.1`，更像集中 TopK 策略，不是可百亿化指增策略。
- `plans/cc-phase4-final-summary-2026-05-19.md` Track B：rolling 执行策略中位年化只有 `+8.7%`，均值被极端 split 拉高。

判断：

1. **信号强度已经接近公开强基准**，不是弱模型。
2. 离百亿私募差距主要不是“再调一点 XGB 参数”，而是 **评估口径、可交易组合、容量、风险中性、因子正交化、研究治理**。
3. 后续应从 `Top20 spread` 转向 `benchmark-relative excess IR`，否则无法对标真实指增产品。

---

## 1. 外部参考工程怎么用

这里不是建议一次性引入所有库，而是借鉴它们的工程边界和评估口径。

| 参考工程/文档 | 用在本项目哪里 | 具体借鉴 | 不建议照搬 |
|---|---|---|---|
| Microsoft Qlib：<https://github.com/microsoft/qlib> | 数据、模型、workflow、Alpha158/Alpha360、回测 | 继续作为主研究框架；参考 examples/benchmarks 的 Alpha158/Alpha360 对照；使用 Qlib recorder 思路统一实验记录 | 不要盲目换成 Qlib 内置全套策略，当前已有自定义 PIT/成本逻辑 |
| Qlib benchmarks：<https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md> | 公开基准对照 | 用 Alpha158/Alpha360 的 IC、RankIC 量级判断自己是否真的强 | 不把 Qlib benchmark 当百亿私募真实指标 |
| Qlib RL / order execution | Phase 6 执行控制 | 借鉴订单执行 RL，而不是个股预测 RL | 不在 Phase 5 前让 RL 接管选股 |
| Alphalens：<https://github.com/quantopian/alphalens> | 因子诊断 | 分位收益、IC decay、turnover、factor tearsheet | 不必强依赖安装；可以先在 `scripts/evaluate_factor_ic.py` 自实现核心指标 |
| cvxportfolio：<https://github.com/cvxgrp/cvxportfolio> | 组合优化 | 目标函数：收益、风险、成本、换手约束；多期组合优化思想 | 当前环境可能有 cvxpy/numpy 冲突，第一版用 `scipy.optimize` 或启发式优化 |
| MLflow：<https://mlflow.org/docs/latest/tracking/> | 实验治理 | run_id、params、metrics、artifacts、model version | 本仓库已有 `models/model_registry.py` 和 `mlruns/`，先统一现有 registry，不急着重构 |
| DVC：<https://dvc.org/doc> | 数据版本 | 数据快照、特征缓存版本、可复现训练 | 大数据文件很多，先做 manifest/hash，不急着全量 DVC 化 |
| Feast：<https://docs.feast.dev/> | PIT feature store | point-in-time join、offline/online feature 语义 | 暂不引入服务化 feature store，先把 `asof_merge` 和 cache manifest 做规范 |
| FinRL / FinRL-Meta：<https://github.com/AI4Finance-Foundation/FinRL> | RL sandbox | Gym 环境、交易成本、状态/动作/奖励设计 | 不直接用 FinRL 的股票环境替代本项目真实回测 |
| FinGPT：<https://github.com/AI4Finance-Foundation/FinGPT> | LLM 舆情/事件因子 | 金融文本情绪、事件抽取、低相关另类数据方向 | 不把 LLM 分数直接当买卖信号，必须做事件表和 rolling ablation |
| vn.py：<https://github.com/vnpy/vnpy> | Paper/实盘 OMS 思路 | 事件驱动、订单/成交/账户/风控模块边界 | 本项目仍以研究和 paper trading 为主，不急着接真实柜台 |

---

## 2. 总体 Phase 切分

| Phase | 名称 | 目标 | 状态 |
|---|---|---|---|
| 4J | Institutional Metric Gate | 从信号指标升级到机构产品指标 | 最高优先级 |
| 4K | Portfolio & Capacity Engine | 从 TopK 策略升级到约束组合 | 最高优先级 |
| 4L | Weak Factor Factory v2 | 把失败因子做正交化/事件化/残差化 | 高优先级 |
| 4M | Alpha360 & Model Diversity | 表格 + 深度模型分层验证 | 中高优先级 |
| 4N | LLM Event Alpha | 新闻/公告/舆情结构化为 PIT 事件因子 | 中优先级 |
| 5A | Research Governance | 数据/实验/模型/报告治理 | 与 4J 并行 |
| 5B | Deep Sequence Models | HIST/ALSTM/Transformer 类模型研究 | 4J/4K 后 |
| 5C | RL Portfolio Controller | RL 只做仓位/换手/风险预算控制 | 组合引擎稳定后 |
| 6 | Execution/Paper OMS | 订单、成交、风控、paper ledger | 最后推进 |

---

## 3. Phase 4J：Institutional Metric Gate

### 3.1 为什么先做

现在的 gate 主要是：

- IC / RankIC
- Top20 spread
- 简单成本后 Sharpe
- 简单 exposure

百亿私募更关心：

- 相对指数超额收益
- 信息比率 IR
- tracking error
- 行业/风格/市值暴露
- 换手和容量
- 成本后、约束后、滚动稳定

如果这个 gate 不补上，后面所有因子和模型实验都会继续围绕 `Top20 spread` 优化，容易越做越像小资金集中策略。

### 3.2 本仓库参考文件

现有脚本：

- `scripts/phase4_rolling_gate.py`
- `scripts/fast_rolling_gate.py`
- `scripts/phase4_backtest.py`
- `scripts/phase4_exposure.py`
- `scripts/backtest_qlib_signal.py`
- `scripts/evaluate_factor_ic.py`
- `backtest/portfolio_backtest.py`
- `backtest/exposure_report.py`
- `backtest/cost_model.py`
- `models/feature_pipeline.py`

现有结果：

- `data/storage/phase4_rolling_gate_xgb_174_3yr.json`
- `logs/phase4_rolling_gate_174_3yr.log`
- `data/storage/phase4_backtest_xgb_174_top20.json`
- `data/storage/phase4/exposure_report.json`
- `plans/cc-phase4-final-summary-2026-05-19.md`

外部参考：

- Qlib recorder / workflow / risk report
- Alphalens factor tear sheet
- MLflow Tracking 的 run/metric/artifact 结构

### 3.3 建议新增文件

```text
metrics/
  __init__.py
  signal_metrics.py
  portfolio_metrics.py
  benchmark_metrics.py
  factor_tearsheet.py

scripts/
  phase4j_institutional_gate.py
  phase4j_daily_metric_snapshot.py
  phase4j_compare_champion_shadow.py

data/storage/phase4j/
  institutional_gate_xgb_174.json
  institutional_gate_xgb_174.csv
  daily_ic_series_xgb_174.parquet
  benchmark_excess_report_xgb_174.json
  exposure_timeseries_xgb_174.parquet
```

### 3.4 指标口径

信号层：

| 指标 | 说明 | 当前状态 | 目标 |
|---|---|---|---|
| daily IC mean/std/ICIR | Pearson IC 日序列 | 日志有 split ICIR，但 JSON 不完整 | 正式落 parquet/json |
| daily RankIC mean/std/RankICIR | Spearman RankIC 日序列 | 有 RankIC mean，无 RankICIR | 必须新增 |
| IC decay | 1/3/5/10/20 日预测衰减 | 未系统化 | 新增 |
| 分位收益 | Q1-Q10 或 Top20/50/100/300 | 主要 Top20 | 扩到多层 |
| turnover of signal | 因子/分数换手 | 未系统化 | 新增 |

组合层：

| 指标 | 说明 | 目标 |
|---|---|---|
| annual excess return | 相对 CSI300/500/1000/A500 | `8%-15%+` 是可研究目标，`15%-20%+` 是强产品目标 |
| excess IR | 年化超额 / tracking error | `1.0+` 起步，`1.5+` 强，`2.0+` 很强 |
| tracking error | 相对基准波动 | 按策略定位设置 |
| max drawdown / excess drawdown | 净值和超额回撤 | 必须可见 |
| turnover / cost drag | 换手和成本占收益比 | 成本不超过收益 `25%-35%` |
| capacity | 按 ADV 参与率估算容量 | Top20 不够，至少评估 Top100/300 |

暴露层：

| 指标 | 目标 |
|---|---|
| industry active weight | 单行业偏离可控，默认 `<5%-10%` |
| size/style exposure | 不能全靠小盘/微盘暴露 |
| single-name weight | 单票上限 `1%-3%` 用于指增；TopK 研究可单独保留 |
| liquidity bucket exposure | 小成交额桶占比必须可控 |

### 3.5 验收门槛

第一阶段通过门槛：

- 输出完整 `institutional_gate_xgb_174.json`。
- `xgb_174` 有 daily IC、RankIC、ICIR、RankICIR。
- 同时输出 Top20/50/100/300 spread。
- 至少支持一个基准：CSI500 或 CSI1000。
- 报告明确显示：绝对收益、相对收益、成本后收益、行业暴露、换手。

第二阶段通过门槛：

- 24 split 每个 split 都有产品层指标。
- 不再只说 `avg RankIC`，而是能说：
  - `RankIC = X`
  - `RankICIR = Y`
  - `Top100 cost-adjusted excess IR = Z`
  - `turnover = T`
  - `capacity under 5% ADV = C`

### 3.6 不通过怎么办

- RankIC 好但 IR 差：优先修组合和成本，不急着修模型。
- Top20 好但 Top100/300 差：说明是小容量集中 alpha，不能按百亿私募路线吹。
- 超额收益好但行业暴露大：上组合优化器和行业约束。
- ICIR 低：做因子正交化、降噪、模型集成，不盲目上 RL。

---

## 4. Phase 4K：Portfolio & Capacity Engine

### 4.1 目标

把现在的 `TopK equal weight / buffered_partial` 升级为：

```text
alpha score
-> candidate pool
-> risk/cost/capacity aware optimizer
-> target weights
-> executable orders
-> daily portfolio ledger
```

### 4.2 本仓库参考文件

现有：

- `backtest/portfolio_backtest.py`
- `backtest/cost_model.py`
- `backtest/optimizer.py`
- `backtest/exposure_report.py`
- `scripts/phase4_backtest.py`
- `scripts/rolling_backtest_configs.py`
- `scripts/run_paper_trading.py`
- `paper/oms.py`
- `models/portfolio_policy.py`

相关文档：

- `plans/cx-phase4-private-fund-roadmap-2026-05-17.md`
- `plans/cc-track-b-final-2026-05-18.md`
- `plans/cc-backtest-label-mismatch-2026-05-17.md`
- `plans/cc-portfolio-execution-research-2026-05-17.md`

外部参考：

- cvxportfolio：目标函数、交易成本、约束设计。
- Qlib backtest / strategy：研究框架参考。
- vn.py：paper OMS 的订单、成交、账户、风控边界。

### 4.3 建议新增/改造文件

```text
backtest/
  constraints.py
  capacity.py
  benchmark.py
  risk_model.py
  optimizer_v2.py
  ledger.py

scripts/
  phase4k_portfolio_optimizer_backtest.py
  phase4k_capacity_report.py
  phase4k_benchmark_excess_backtest.py
```

### 4.4 第一版组合优化器不要太复杂

先不要直接上 cvxpy。建议先实现一个可控的启发式/`scipy.optimize` 版本：

目标函数：

```text
maximize:
  expected_alpha
  - lambda_risk * factor_risk
  - lambda_turnover * turnover
  - lambda_cost * estimated_cost
  - lambda_concentration * concentration_penalty
```

硬约束：

- 单票权重：研究 `<=5%`，指增 `<=1%-2%`。
- 行业主动偏离：`<=5%-10%`。
- 单日换手：`<=10%-20%`。
- 个股 ADV 参与率：默认 `<=3%-5%`。
- ST/停牌/涨跌停不可交易。
- 持仓数：Top20 研究保留；机构化必须支持 Top100/Top300。

### 4.5 容量报告

新增 `capacity.py`，输入目标权重和 ADV：

| 输出 | 说明 |
|---|---|
| capacity_at_1pct_adv | 每只股票交易额不超过 ADV 1% 的容量 |
| capacity_at_3pct_adv | 默认可交易容量 |
| capacity_at_5pct_adv | 激进容量 |
| crowded_names | 容量瓶颈个股 |
| small_liquidity_bucket_weight | 低流动性股票权重 |

输出文件：

```text
data/storage/phase4k/capacity_xgb_174_top100.json
data/storage/phase4k/capacity_xgb_174_top300.json
data/storage/phase4k/optimizer_backtest_xgb_174.json
```

### 4.6 验收门槛

- Top100/Top300 组合能跑完整 24 split。
- 成本后 excess IR 不低于 Top20 集中策略太多。
- 单票权重和行业偏离可控。
- 组合持仓数中位数大于 `80`，才算进入“可规模化”路线。
- 输出 capacity，不再只说 `Capacity OK`。

---

## 5. Phase 4L：Weak Factor Factory v2

### 5.1 当前问题

`plans/cc-phase4-final-summary-2026-05-19.md` 已经说明：新增因子直接 rolling ablation 大多失败。

失败不是坏事，说明 baseline 已经很强。但下一步不能再“原字段拼接”。要把因子工程升级为：

```text
raw field
-> PIT alignment
-> coverage / freshness
-> winsorize / rank / zscore
-> industry/size neutralization
-> residual IC against champion
-> event/window decay
-> two-stage residual model
-> rolling ablation
```

### 5.2 本仓库参考文件

脚本：

- `scripts/build_moneyflow_v2.py`
- `scripts/ablation_moneyflow_v2.py`
- `scripts/build_event_factors.py`
- `scripts/ablation_event_factors.py`
- `scripts/build_block_trade_v2.py`
- `scripts/ablation_block_trade_v2.py`
- `scripts/build_derived_factors.py`
- `scripts/ablation_derived_factors.py`
- `scripts/phase2_residual_ic.py`
- `scripts/evaluate_factor_ic.py`
- `scripts/phase2_pit_audit.py`

文档：

- `plans/cc-weak-factor-enhancement-research-2026-05-19.md`
- `plans/cc-factor-merge-debug-log.md`
- `plans/cx-pit-safe-performance-boost-plan-2026-05-13.md`

数据：

- `data/storage/st_moneyflow.parquet`
- `data/storage/st_cyq_perf.parquet`
- `data/storage/st_forecast.parquet`
- `data/storage/st_top_list.parquet`
- `data/storage/st_top_inst.parquet`
- `data/storage/st_block_trade.parquet`
- `data/storage/st_pledge_stat.parquet`
- `data/storage/st_fund_portfolio.parquet` 如后续拉齐

外部参考：

- Alphalens 的 IC decay、quantile return、turnover。
- Feast 的 PIT feature join 思想。

### 5.3 建议新增统一处理器

```text
factors/
  processors.py
  neutralize.py
  decay.py
  residual.py
  event_windows.py

scripts/
  phase4l_build_factor_v2.py
  phase4l_factor_tearsheet.py
  phase4l_residual_ablation.py
  phase4l_two_stage_residual_model.py
```

### 5.4 每类因子的处理方法

#### moneyflow

不要再拼 18 个原始列。保留 6-8 个正交特征：

- `net_flow_zscore_60d`
- `flow_persistence_10d`
- `flow_acceleration_5_20d`
- `large_small_divergence`
- `industry_relative_flow_rank`
- `size_neutral_flow_z`
- `flow_x_low_vol`
- `flow_x_momentum`

验收：

- residual RankIC > `0.005`
- 12 split 中 `>=7` 个 split delta RankIC 为正
- Top100/300 组合收益不能恶化

#### cyq/chip distribution

不要用绝对成本价格。改成：

- `price_vs_weighted_avg_cost`
- `winner_rate_zscore_60d`
- `winner_rate_extreme_reversal`
- `cost_concentration_zscore`
- `chip_pressure_x_turnover`
- `industry_relative_winner_rate`

验收：

- 必须在低换手/突破/反转 regime 分层看，不要求全市场无条件有效。

#### block trade / top list / top institution

这类是事件，不是连续因子。处理方式：

- event date PIT 对齐。
- 事件类型 one-hot。
- 买方/卖方席位强度。
- 事件后 1/3/5/10/20 日 decay。
- 同一股票事件簇聚合。
- 行业同日事件强度。

验收：

- 单事件窗口收益曲线必须合理。
- 不直接拼进主模型，先作为 rerank 或 blacklist/whitelist。

#### forecast / earnings

当前 forecast 数据新鲜度差。处理方式：

- 只保留公告后 `0-60` 天事件。
- 用 surprise：预告净利增速 vs 行业中位数、公司历史。
- 对上修/下修做方向。
- 把 stale forecast 全部置空。

验收：

- coverage 和 freshness 必须入报告。
- stale 数据不能参与训练。

#### fund holdings / top fund crowdedness

放在 Phase 4L 后半段。数据来自 `scripts/fetch_st_round6.py --api fund_portfolio`。

建议因子：

- 基金持仓市值占流通市值比例。
- 持有基金数量变化。
- 前十大重仓集中度。
- 明星基金/高胜率基金持仓变化。
- 行业拥挤度。
- 拥挤反转：高拥挤 + 高估值 + 资金流转弱。

验收：

- 只按季度披露，必须考虑披露滞后。
- 不允许按报告期当天使用，必须按公告/披露可得日期 asof。

### 5.5 统一门槛

一个新因子包进入 shadow 的条件：

- PIT audit 通过。
- coverage > `60%`，事件因子可单独设覆盖门槛。
- residual RankIC > `0.005` 或 Top100 spread 有稳定正增量。
- 12 split 里 delta RankIC 正比例 >= `60%`。
- 不能显著增加换手和行业偏离。
- 进入 `model_registry` 状态必须是 `research_only`，跑满 20 个 paper days 才可升 shadow。

---

## 6. Phase 4M：Alpha360 & Model Diversity

### 6.1 当前结论

`plans/cx-phase4-private-fund-roadmap-2026-05-17.md` 已经定义了 Alpha360 的正确位置：不是直接全量拼生产，而是 feature-set 对照。

当前已有：

- `scripts/build_alpha360_cache.py`
- `scripts/phase4e_feature_set_compare.py`
- `data/storage/feature_cache_alpha360.parquet`
- `data/storage/phase4/phase4i_baseline_v2_compare.json`

### 6.2 参考工程

- Qlib Alpha158 / Alpha360 benchmark。
- Qlib HIST / ALSTM / Transformer examples。

### 6.3 开发动作

特征集固定为：

| Feature Set | 内容 | 用途 |
|---|---|---|
| FS-174 | 当前 champion | 主基线 |
| FS-360 | Alpha360 | 价量路径独立基线 |
| FS-534 | FS-174 + Alpha360 | 拼接验证 |
| FS-174-resid360 | Alpha360 预测 FS-174 residual | 正交增量验证 |

新增脚本：

```text
scripts/phase4m_alpha360_full_gate.py
scripts/phase4m_alpha360_residual_model.py
scripts/phase4m_rank_fusion.py
```

模型顺序：

1. XGB / LGB / CatBoost 表格模型。
2. Rank fusion，不做 raw score 平均。
3. 只有 FS-360 独立有效时，再考虑 ALSTM/HIST。

验收：

- FS-360 单模型 RankIC 不低于 `0.045` 才值得继续。
- FS-534 必须稳定优于 `max(FS-174, FS-360)`，否则不拼接。
- rank fusion 24 split delta RankIC 正比例 >= `65%` 才能 shadow。
- 如果 Alpha360 只在深度模型有效，放入 Phase 5B。

---

## 7. Phase 4N：LLM Event Alpha

### 7.1 目标

LLM 不直接给“买/卖”。LLM 做：

```text
新闻/公告/研报/社媒
-> 结构化事件
-> 事件强度/方向/置信度/主体映射
-> PIT event factor
-> rolling ablation
```

### 7.2 本仓库参考文件

已有：

- `scripts/collect_daily_news.py`
- `scripts/run_llm_event_pipeline.py`
- `scripts/build_llm_event_factors.py`
- `factors/llm_event_extractor.py`
- `factors/event_impacts.py`
- `factors/news_sentiment.py`
- `data/collectors/sentiment.py`
- `data/collectors/gdelt.py`
- `plans/cc-量化与舆情深度调研.md`
- `plans/cx-quant-sentiment-deep-research-2026-05-07.md`

外部参考：

- FinGPT：金融文本情绪和事件抽取方向。
- GDELT：全球新闻事件数据。
- Qlib event-driven features 思路。

### 7.3 事件 schema

建议统一落表：

```text
data/storage/events/llm_events.parquet
```

字段：

| 字段 | 说明 |
|---|---|
| event_time | 文本发布时间，必须带时区 |
| available_date | 可交易使用日期，A股收盘后新闻默认次日 |
| symbol | 股票代码 |
| entity | 公司/行业/人物/产品 |
| event_type | earnings, policy, lawsuit, supply_chain, order, buyback, risk 等 |
| polarity | -1/0/+1 |
| intensity | 0-1 |
| confidence | 0-1 |
| source | news/announcement/social/research |
| summary_hash | 去重 |
| llm_model | 模型版本 |
| prompt_version | prompt 版本 |

### 7.4 因子化方式

不要只做情绪均值。做：

- `event_intensity_decay_3d/5d/20d`
- `negative_risk_event_count_20d`
- `positive_event_cluster_score`
- `industry_event_breadth`
- `event_surprise_vs_history`
- `policy_event_industry_spillover`
- `source_weighted_confidence`

### 7.5 验收门槛

- LLM 输出必须可复现：prompt version、model、temperature、原文 hash。
- 事件表必须 PIT-safe。
- 单独做 event study：事件后 1/3/5/10/20 日收益曲线。
- rolling ablation 必须过 12 split，不能只拿单日案例。
- 对成本后组合收益有增量，才进入主线；否则只用于解释和风控。

---

## 8. Phase 5A：Research Governance

### 8.1 为什么重要

现在项目已经有很多结果文件、日志和模型，但仍有几个风险：

- JSON 有过序列化失败，导致结果文件不完整。
- 同一个指标在不同脚本口径可能不同。
- 模型文件命名还不够稳定。
- 数据缓存版本和训练结果没有强绑定。

### 8.2 本仓库参考文件

- `models/model_registry.py`
- `models/registry.py`
- `scripts/phase4_promote.py`
- `scripts/shadow_daily_compare.py`
- `scripts/shadow_daily_inference.py`
- `scripts/run_with_status.py`
- `data/storage/phase4/model_registry.json`
- `mlruns/`

外部参考：

- MLflow Tracking。
- DVC data versioning。
- Feast feature registry。

### 8.3 建议新增

```text
tracker/
  data_manifest.py
  experiment_schema.py
  metric_schema.py
  artifact_store.py

scripts/
  validate_experiment_artifacts.py
  freeze_data_manifest.py
  compare_model_registry.py
```

### 8.4 每次实验必须记录

| 类别 | 字段 |
|---|---|
| data | qlib data path、calendar hash、feature cache path/hash、label expr |
| split | train/valid/test dates、n_splits、test_days |
| model | model type、params、seed、feature_set |
| metric | IC/ICIR/RankIC/RankICIR/spread/backtest/exposure/capacity |
| artifact | model path、predictions path、report path |
| governance | status: research_only/shadow/champion/deprecated |

### 8.5 验收

- 任意一条推荐股票能追溯到：数据版本、模型版本、特征版本、训练窗口、评分时间。
- 任意模型 promotion 都必须有 gate report。
- JSON 写入统一走 serializer，禁止再次出现 `np.bool_` 导致半截 JSON。

---

## 9. Phase 5B：Deep Sequence Models

### 9.1 什么时候做

只有当 4J/4K 已经能证明：

- 当前表格模型在机构指标下的真实强弱；
- Alpha360 是否有独立信息；
- Top100/300 组合能稳定评估；

才启动深度模型。

### 9.2 本仓库参考

- `models/deep_models.py`
- `data/storage/alstm_model.pt`
- `data/storage/transformer_model.pt`
- `scripts/train_model_suite.py`
- `scripts/build_alpha360_cache.py`
- `plans/cc-qlib-advanced-features-roadmap.md`

外部参考：

- Qlib ALSTM / Transformer / HIST benchmark。
- MASTER / SFM / TCN 类序列模型可作为研究方向，但不要一次性铺开。

### 9.3 推荐顺序

1. ALSTM + Alpha360。
2. Transformer + Alpha360。
3. HIST 类 market graph / concept relation。
4. 表格 champion + deep model rank fusion。

### 9.4 验收

- 深度模型单模型 RankIC >= FS-174。
- rank fusion 提升 RankIC 或成本后 excess IR。
- 24 split 中至少 `65%` split 改善。
- 推理速度可满足每日收盘后运行。

---

## 10. Phase 5C：RL Portfolio Controller

### 10.1 边界

沿用 `plans/cx-phase5-rl-control-roadmap-2026-05-17.md` 的判断：

RL 只做 policy controller，不做个股 alpha。

正确结构：

```text
xgb_174 / shadow alpha
-> top100/top300 candidate
-> optimizer baseline
-> RL adjusts risk budget / topk / turnover / cash
-> hard constraints
-> paper trading
```

### 10.2 本仓库参考

- `scripts/train_rl.py`
- `models/rl_agent.py`
- `models/portfolio_policy.py`
- `data/storage/rl_metrics.json`
- `data/storage/rl_model.pt`
- `plans/cx-phase5-rl-control-roadmap-2026-05-17.md`

外部参考：

- FinRL / FinRL-Meta：Gym environment、transaction cost、offline train/evaluate。
- Qlib RL order execution：后续执行层 RL。

### 10.3 状态、动作、奖励

状态：

- 当前组合收益、回撤、仓位、换手。
- 行业/风格/市值暴露。
- top100 分数分布、分数分化、候选池稳定性。
- 市场 regime、波动、流动性、涨跌停数量。
- 最近成本和成交压力。

动作：

- cash ratio：`0%-50%`
- topk：`50/100/200/300`
- turnover budget：`5%/10%/20%`
- risk aversion：低/中/高
- rebalance aggressiveness：保守/正常/激进

奖励：

```text
excess_return
- cost_penalty
- turnover_penalty
- drawdown_penalty
- exposure_violation_penalty
- capacity_violation_penalty
```

### 10.4 验收

- 必须和 rule-based controller 对比。
- 24 split 中成本后 Calmar 或 excess IR 有稳定提升。
- 不能增加极端回撤。
- 不能绕过 hard constraints。
- 只允许先进入 shadow，不允许直接 production。

---

## 11. Phase 6：Execution / Paper OMS

### 11.1 目标

把研究回测输出变成真实可审计 paper ledger：

```text
target_weights
-> orders
-> simulated fills
-> holdings
-> cash
-> pnl
-> risk logs
```

### 11.2 本仓库参考

- `paper/oms.py`
- `scripts/run_paper_trading.py`
- `data/storage/paper/oms_state.json`
- `scripts/install_crontab.py`
- `scripts/after_close_pipeline.py`

外部参考：

- vn.py：事件驱动 OMS 模块边界。
- Qlib executor/backtest：订单执行抽象。

### 11.3 新增模块

```text
paper/
  broker_sim.py
  order.py
  fill_engine.py
  account.py
  risk_check.py
  ledger.py

scripts/
  paper_reconcile.py
  paper_daily_report.py
```

### 11.4 验收

- 每天 14:30 / 22:00 的推荐差异可以解释：数据可得性、信号时间、成交假设、风险状态。
- 每笔订单有生成原因、模型版本、目标权重、风控结果。
- 每日 paper report 包含：持仓、换手、成本、行业暴露、未成交原因。

---

## 12. 推荐优先级

### 立即做，1-3 天

1. `Phase 4J institutional gate`
   - 新增 `phase4j_institutional_gate.py`
   - 补 daily IC/RankIC/ICIR/RankICIR
   - 补 Top20/50/100/300 分层
   - 补 benchmark excess metrics

2. `Phase 4K portfolio v1`
   - Top100/Top300 组合回测
   - 单票/行业/换手/ADV 约束
   - capacity report

3. `Research governance hotfix`
   - 统一 JSON serializer
   - 修复半截 JSON 风险
   - 模型文件命名规范

### 接着做，3-7 天

4. `Weak Factor Factory v2`
   - moneyflow/cyq/block/toplist/forecast 全部按 processor + residual IC 重跑。

5. `Alpha360 residual/fusion`
   - 不直接拼接生产。
   - 先做 FS-360、FS-534、resid360、rank fusion。

6. `Paper report`
   - 每日 14:30/22:00 推荐解释、持仓、风险、交易成本。

### 再做，1-3 周

7. `LLM Event Alpha`
   - 事件 schema + event study + rolling ablation。

8. `Deep sequence model`
   - ALSTM/HIST/Transformer 只作为 shadow/fusion。

9. `RL controller`
   - 只调仓位/换手/风险预算，不碰选股 alpha。

---

## 13. 最重要的三条红线

1. **任何新模型、新因子不许绕过 PIT、rolling、成本、暴露、容量。**
2. **任何单窗口爆炸收益都只能进入 research_only，不能 promotion。**
3. **百亿私募化不是 RankIC 单指标竞赛，而是 signal -> portfolio -> execution -> governance 的整条链。**

---

## 14. 一句话路线

先用 Phase 4J/4K 把 `xgb_174` 的真实机构级产品能力量出来；然后用 Phase 4L/4M 找低相关增量；再用 Phase 5A 固化研究治理；最后才让 LLM、深度模型和 RL 进入 shadow。这样走，项目会从“能选几只强票”变成“能解释、能复现、能约束、能扩容的量化系统”。
