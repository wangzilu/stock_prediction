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
| 4J | Institutional Metric Gate | 从信号指标升级到机构产品指标 | ✅ 已完成 |
| 4K | Portfolio & Capacity Engine | 从 TopK 策略升级到约束组合 | ✅ 已完成 (opt_top100_to10, Sharpe 4.5+) |
| 4L | Weak Factor Factory v2 | 把失败因子做正交化/事件化/残差化 | 高优先级 |
| 4M | Alpha360 & Model Diversity | 表格 + 深度模型分层验证 | 中高优先级 |
| 4N | LLM Event Alpha | 新闻/公告/舆情结构化为 PIT 事件因子 | 中优先级 (B'+C' overlay 已做) |
| **4S** | **实验治理 + 组合风险 + Alpha Factory** | **统一产物契约、ShrinkCov、Barra报告、regime-weighted sampler** | **当前最高优先级** |
| 4T | LLM 结构化事件库 | 统一 JSON schema，PIT-safe 事件存储 | 4S 后 |
| 4U | 事件研究校准 | 每类事件 1/3/5/10/20 日 CAR | 4T 事件库 60+ 天 |
| 4V | 政策/情绪 overlay shadow | 不改 XGB，rerank 对照 30 天 | 4U 校准通过 |
| 5A | Research Governance | 数据/实验/模型/报告治理 | 合并入 4S |
| 5B | Deep Sequence Models | HIST/ALSTM/Transformer 类模型研究 | 4S 后 |
| 5C | RL Portfolio Controller | RL 只做仓位/换手/风险预算控制 | 组合引擎稳定后 |
| 5D | 行业政策暴露矩阵 | 政策主题 × 行业/概念映射 | 4V shadow 验证 |
| 5E | Regime Policy Controller | 政策不确定性/支持/监管压力进风控 | 5D |
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

先用 Phase 4J/4K 把 `xgb_174` 的真实机构级产品能力量出来；然后用 Phase 4L/4M 找低相关增量；再用 Phase 5A 固化研究治理；最后才让 LLM、深度模型和 RL 进入 shadow。这样走，项目会从”能选几只强票”变成”能解释、能复现、能约束、能扩容的量化系统”。

---

## 15. Phase 4S：实验治理 + 组合风险 + Alpha Factory Lite（2026-05-24 新增）

**来源**: CX 审阅 Qlib/RD-Agent 调研后拍板。核心判断：当前缺的不是更多高级模块，而是研究闭环的纪律。

### 15.1 总体优先级（CX 终版）

| 优先级 | 任务 | 阶段 |
|--------|------|------|
| P0 | 统一实验产物契约 + promotion gate | Mac Studio 第一阶段 |
| P1 | Recorder/SignalRecord 补主训练链 | Mac Studio 第一阶段 |
| P1 | ShrinkCov 作为组合风险报告和 rerank penalty（不做 MVO） | Mac Studio 第一阶段 |
| P1/P2 | StructuredCov 接 barra_simple，接入组合报告 | Mac Studio 第一阶段 |
| P2 | RollingGen 统一 rolling split | Mac Studio 第一阶段 |
| P2 | 简化版 DDG-DA：regime-weighted training sampler | Mac Studio 第一阶段 |
| P2 | Alpha Factory Lite：候选因子自动生成→验证→晋级 | Mac Studio 第一阶段 |
| P3 | RD-Agent / ADARNN / HIST / Qlib Online | 未来云/Linux |

### 15.2 P0：统一实验产物契约

**问题**：每个脚本各自 dump JSON，格式不统一。新因子/模型的验收全靠人肉看 log。

**目标**：所有训练/回测必须输出同结构结果。

统一产物：

```text
{experiment_id}/
  config.json        # 超参数、数据窗口、因子集、预处理方式
  pred.pkl           # 预测值 (datetime × instrument)
  label.pkl          # 实际标签
  metrics.json       # IC/RankIC/ICIR/RankICIR/spread/cost_adjusted
  backtest.json      # Sharpe/annual_return/max_dd/turnover/cost_drag
  factor_health.json # coverage/freshness/autocorr/persistence
  exposure.json      # 行业/风格/市值暴露
```

Promotion gate 检查清单：
- PIT audit 通过
- 24 split metrics.json 全部生成
- RankIC > 0.005（残差 IC 对 champion）
- 12+ split delta RankIC 为正
- 换手和行业偏离不显著恶化
- negative control 通过（shuffle 后 IC ≈ 0）

**新增文件**：

```text
tracker/
  artifact_contract.py   # 定义统一产物 schema + 写入/验证
  promotion_gate.py      # 自动检查 gate 条件
```

### 15.3 P1：ShrinkCov 组合风险（不改 optimizer）

**原则**：optimizer_v2 (Sharpe 4.5+) 不动。ShrinkCov 只做三件事：

1. **组合预测波动率**：每日输出 portfolio predicted volatility，作为 RiskGuard L2 的补充信号。
2. **高相关持仓惩罚**：如果两只持仓股相关性 > 0.8，在 reranker 中加 diversification penalty。
3. **风险预算监控**：组合中每只股票的边际风险贡献（MCTR），识别尾部集中风险。

**新增文件**：

```text
backtest/
  risk_model.py     # ShrinkCov + 组合波动率 + MCTR
```

### 15.4 P1/P2：StructuredCov 接 Barra

**前置**：`backtest/barra_simple.py` 已有 5 style + 33 industry 因子暴露。

**目标**：
- 把 exposure 接入 `risk_model.py`，用 `B * F * B' + D` 结构估计协方差。
- 每日输出 exposure report（行业主动偏离、风格偏离）。
- 暴露超限时在 daily_health_check 中报警。

### 15.5 P2：regime-weighted training sampler

**替代直接上 DDG-DA**。思路：

1. 用 `regime_controller.compute(date)` 计算每个历史交易日的 regime 向量（12 维）。
2. 计算当前日期和每个历史日期的 regime 相似度（余弦距离）。
3. 训练样本权重 = f(相似度)，越像当前市场的历史数据权重越高。
4. 传给 XGB 的 `sample_weight` 参数。

**验收**：24 split 中 regime-weighted vs uniform 的 RankIC 对比。

### 15.6 P2：Alpha Factory Lite

**替代直接上 RD-Agent**。本地自动化：

```text
candidate_factors/
  {factor_name}/
    build.py          # 因子构建脚本
    config.json       # 参数
    tearsheet.json    # IC/RankIC/spread/coverage/negative_control
    verdict: pass/fail/pending
```

流程：
1. 手动或脚本生成候选因子 → 写入 `candidate_factors/`。
2. 自动跑 tearsheet pipeline（IC、RankIC、spread、coverage、negative control）。
3. 通过 gate → 进入 `model_registry` 状态 `research_only`。
4. 跑满 20 paper days → 可升 shadow。

### 15.7 不做的事

| 功能 | 理由 |
|------|------|
| 迁移到 Qlib Backtest Engine | 自建引擎已满足需求 |
| Online Serving 替换 crontab | 当前手动管理足够 |
| RD-Agent 租 Linux 服务器 | 先把本机研究闭环做硬 |
| DDG-DA 完整 meta 框架 | 先做简化版 regime-weighted sampler |
| 均值-方差优化替代 optimizer_v2 | Sharpe 4.5+ 不需要动 |

### 15.8 与 Regime Controller P0 的关系

2026-05-24 同日完成的 regime_controller.py P0 改动：
- ✅ 修 inflation_score CPI 列（nt_yoy）
- ✅ 修 fx_risk_score 百倍报价
- ✅ 加 hard_break / soft_break 击穿逻辑
- ✅ futures_basis_score 改真基差（IC 期货 / CSI500 现货）
- ✅ policy_support / theme_breadth PIT 过滤

这些改动服务于 P0 实验治理中的 regime 相关 gate 检查。

---

## 16. Phase 4T-4V + 5D-5E：政策/情绪因子体系（2026-05-24 CX 设计）

**核心原则**：政策和情绪不是普通价量因子，必须拆成三层，不能一股脑塞进 XGB174。

### 16.1 三层架构

| 层级 | 用途 | 进模型方式 |
|------|------|-----------|
| **市场级 regime** | 决定仓位、风险、是否开新仓 | 进 regime_controller，不进 XGB |
| **行业/主题级 rotation** | 决定哪些板块被政策或情绪推着走 | 进 reranker 行业加权 |
| **个股级事件/舆情** | 只对有新闻/公告/讨论的股票做 overlay | 进 event overlay / rerank |

### 16.2 8 类因子定义

| 因子 | 层级 | 数学形式 | 用途 |
|------|------|---------|------|
| `policy_uncertainty_score` | 市场级 | 中文 EPU 文本频率 z-score，越高越负 | regime 降风险 |
| `policy_support_score` | 市场/行业 | 政策支持事件强度 × 衰减 | risk-on / 行业加权 |
| `regulatory_pressure_score` | 行业/个股 | 监管、处罚、反垄断、窗口指导文本强度 | 降低相关行业/个股 |
| `sector_policy_momentum` | 行业级 | 近 N 日政策正向事件加权和 | 板块轮动 |
| `retail_sentiment_score` | 个股级 | 股吧/雪球/东财情绪 z-score | rerank / overlay |
| `attention_shock_score` | 个股级 | 新闻数/帖子数/搜索热度异常 z-score | 事件候选过滤 |
| `sentiment_disagreement_score` | 个股级 | 正负观点分歧/评论方差 | 高波动/反转风险 |
| `speculative_heat_score` | 市场/个股 | 涨停/连板/炸板/换手/讨论热度 | 防追高 |

### 16.3 核心公式

```text
event_score_i,t =
  sum_k impact(event_k) * confidence_k * exp(-age_k / half_life_k)

policy_industry_score_j,t =
  sum_topic policy_intensity_topic,t * exposure_industry_j,topic

stock_policy_score_i,t =
  sum_j industry_weight_i,j * policy_industry_score_j,t
  + firm_specific_policy_event_i,t

sentiment_score_i,t =
  zscore_rolling(positive_prob_i,t - negative_prob_i,t)

attention_shock_i,t =
  (log(1 + news_count_i,t) - rolling_mean_60d) / rolling_std_60d
```

**生产用 rerank overlay（不改 XGB 主模型）**：

```text
final_rank_score =
  rank(xgb_score) * 0.75
  + rank(event_policy_score) * 0.10
  + rank(sentiment_score) * 0.05
  + rank(risk_penalty) * 0.10
```

### 16.4 LLM 输出 schema（统一事件表）

LLM 只做结构化抽取，不做买卖判断。输出统一 JSON：

```json
{
  "date": "2026-05-24",
  "stock": "sh600000",
  "source": "announcement/news/forum/policy",
  "event_type": "policy_support",
  "topic": "AI算力/低空经济/地产/券商/新能源",
  "direction": 1,
  "magnitude": 0.6,
  "confidence": 0.82,
  "affected_industries": ["计算机", "通信"],
  "horizon_days": 5,
  "is_policy": true,
  "is_regulatory": false,
  "is_rumor": false,
  "summary": "..."
}
```

收益影响必须由历史事件研究校准，不信 LLM 的 magnitude 分数。

### 16.5 Phase 分解

| Phase | 内容 | 目标 | 前置 |
|-------|------|------|------|
| **4T** | LLM 结构化抽取 | 新闻/公告/政策/股吧统一 JSON schema | 4S artifact contract |
| **4U** | 事件研究校准 | 每类事件测 1/3/5/10/20 日 CAR | 4T 事件库 60+ 天 |
| **4V** | sentiment/event overlay shadow | 不改 XGB，只做 rerank 对照 | 4U 校准通过 |
| **5D** | 行业政策暴露矩阵 | 政策主题 × 行业/概念映射 | 4V shadow 验证 |
| **5E** | regime policy controller | 政策不确定性/政策支持/监管压力进风控 | 5D |

### 16.6 现有代码对接

| CX 设计 | 现有代码 | 状态 |
|---------|---------|------|
| LLM 结构化事件 | `factors/llm_event_extractor_v2.py` | ✅ schema 基本匹配 |
| 事件校准 | `scripts/build_event_calibration.py` | ⚠️ 14天数据太少，需 60+ 天 |
| rerank overlay | `scripts/build_event_overlay.py` (B'+C') | ✅ 在 shadow 中 |
| 政策支持 regime | `regime_controller._policy_support()` | ✅ 已有 |
| 散户情绪 | guba 人气榜 | ⚠️ 只有 1 个文件，需持续拉 |
| EPU 指数 | 无 | ❌ 可引入 CBADE 外部数据 |
| 行业政策映射 | 无 | ❌ Phase 5D |

### 16.7 数据源优先级

| 数据 | 用途 | 可行性 |
|------|------|--------|
| 国务院/发改委/财政部/央行/证监会/交易所公告 | 政策/监管主数据 | 可通过东财公告 API |
| 东方财富公告 API | 个股事件 | ✅ 已在用 |
| 东方财富股吧 | 散户情绪 | ✅ AKShare 人气榜作弱代理 |
| 财联社/证券时报/上证报/中证报 | 政策解释和市场叙事 | ⚠️ 需要 RSS 或 API |
| 涨跌停/连板/炸板/成交额 | 情绪热度市场确认 | ✅ 已有 limit_list_d |
| 北向/融资余额/ETF 份额 | 情绪转化为资金 | ✅ 已有 |
| 雪球 | 散户情绪 | ❌ 短期不爬，法律风险 |

### 16.8 验收指标（10 条）

政策/情绪因子不能只看 IC（覆盖率低，IC 可能不好看，但 overlay 可能有效）。必须看：

1. 单因子 RankIC
2. Top20 / Bottom20 spread
3. 事件后 CAR: 1d, 3d, 5d, 10d, 20d
4. 覆盖率 coverage
5. 命中股票的换手和滑点
6. 分行业表现
7. 牛市/熊市/震荡市分段表现
8. negative control
9. 24 split OOS 表现
10. 加入 overlay 后组合收益、回撤、换手是否改善

### 16.9 7 条防坑红线

1. **PIT-safe**：新闻/公告/政策发布时间精确到日期/时间。晚上 10 点能用，下午 2:30 不一定能用。
2. **不用未来解释文本**：事后媒体复盘不能作为当天因子。
3. **coverage neutralization**：大票新闻多、小票少，必须做覆盖率中性化，防止变成隐性市值因子。
4. **LLM 分数要校准**：LLM 只负责分类和摘要，收益影响必须由历史数据学习。
5. **"新政策超预期" vs "旧政策反复表态"**：必须区分，否则持续利好信号会失效。
6. **极端正面 ≠ 买入**：极端正面可能是拥挤和见顶信号。
7. **市场级政策不当个股 alpha**：很多政策只适合控制仓位，不适合选股票。

### 16.10 MVP 最小可行版本

先做 4 个因子，shadow 30 个交易日 + 回放 2020-2026：

1. `policy_support_market_score`（市场级）
2. `regulatory_pressure_industry_score`（行业级）
3. `stock_event_sentiment_score`（个股级）
4. `retail_attention_shock_score`（个股级）

通过后进入 reranker，不直接塞 XGB174。

**学术参考**：
- Baker/Bloom/Davis EPU + CBADE Mainland China EPU
- Du/Huang/Wermers/Wu 中文金融情绪词典
- FinBERT 中文金融 BERT
- A-share sentiment and efficiency
- Weibo mood and Chinese stock market
- BERT sentiment + TVP-VAR
