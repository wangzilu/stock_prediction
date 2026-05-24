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
| 4S | 实验治理 + 组合风险 + Alpha Factory | 统一产物契约、ShrinkCov、Barra报告、regime-weighted sampler | ✅ 已完成 |
| 4W | 可信度收敛 | 数据时间字典、Paper OMS pending order、统一时间语义 | ✅ 已完成 |
| 4E | Model Ensemble 基建 | 模型 zoo artifact、rank/zscore fusion、rolling IC weighting、disagreement | ⚠️ 单 split 完成，24-split 待跑 |
| 4G | Feature Path 统一 + Factor Inventory | 官方因子路径、自动清单、晋级流程 | ✅ 已完成 |
| 4R | Meta-filter / Reranker | Top100 二阶段过滤、risk-aware rerank、optimizer_v2 shadow | 4E/4G 后 |
| 4T | LLM 结构化事件库 | 统一 JSON schema，PIT-safe 事件存储 | ✅ EventStore 基础完成 |
| 4N-1 | EventStore 统一 + 5 时间字段 | 废弃 legacy 双轨 | ✅ 已完成 |
| 4N-2 | 事件 surprise 特征 | direction/confidence/novelty/attention | ✅ 初版完成（需更多数据） |
| 4N-3 | 历史校准表 | event_type × 行业 × 市值 × regime | ⏳ 需 60+ 天事件数据 |
| 4N-4 | overlay/reranker rolling PIT 验证 | 不改 XGB，rerank 对照 | ⏳ 需 rolling pred artifact |
| 4O | Downside Risk Layer | crash label + 负面事件因子 + 踩踏/退潮因子 | ⚠️ crash labels 已建，模型待训练 |
| 4P | RiskGuard 接入 | crash_prob → cannot_buy / reduce_weight / force_sell | 4O 后 |
| 4T-1~7 | LLM pipeline 收敛 | V2 默认/EventStore 唯一/规则筛选/公告正文/校准表 | ⚠️ 4T-1/4T-2 已完成 |
| 4U | Global Supply Chain Overlay | 全球产业事件 → 供应链映射 → A 股因子 | 设计完成，待实施 |
| **4X** | **网络分层 + 数据稳定性** | **domestic/global/none profile, proxy wrapper, cron 分类** | **待实施** |
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

---

## 17. Phase 4W：可信度收敛（2026-05-24 CC 提出，基于 CX 总体评价）

**背景**：CX 总体评价项目 5.5-7.5 分，核心短板是"时间语义不统一"。研究能力已经很强，工程可信度还没完全跟上。离"可信实盘系统"差的不是某个神奇模型，而是三个纪律：数据时点、研究晋级、实盘执行。

**目标**：不再加新因子/模型，专注把已有系统的可信度从 5.5-6 分拉到 7.5-8 分。

### 17.1 CX 评分基线（2026-05-24）

| 维度 | 分数 | 评价 |
|------|------|------|
| 数据覆盖 | 7/10 | 已经很多，需要更严格的 availability date |
| 因子研究 | 7.5/10 | 方向好，防止堆料 |
| 模型训练 | 7/10 | XGB 主线正确，gate 口径要统一 |
| 回测组合 | 6.5/10 | optimizer_v2 不错，执行/PnL 时序要修 |
| 风控 | 6/10 | 有框架，阈值和实盘接入需谨慎 |
| 生产调度 | 6.5/10 | crontab 可用，不是强一致生产系统 |
| LLM/event | 5.5/10 | 有潜力，还在 shadow/calibration 阶段 |
| 实盘可信度 | 5.5-6/10 | 可以 paper，暂不建议自动真金白银闭环 |

### 17.2 已修的问题（2026-05-24 当天）

| 问题 | 修复 |
|------|------|
| daily_basic / northbound 无 T+1 lag | ✅ BDay(1) |
| ann_date 无可用性延迟 | ✅ BDay(1) |
| 财报 fallback 过于乐观 | ✅ Q1/Q3 30→45d, H1 60→75d |
| regime weights 和 1.11 | ✅ 归一化 |
| inflation 不是 U 型 | ✅ U 型映射 |
| promotion gate split 不 fail | ✅ hard failure |
| MCTR sum≠1 | ✅ PCTR 除以 variance |
| 回测 PnL 日期错位 | ✅ realized dates |
| OMS T+1 open 语义混乱 | ✅ _price_type 标记 |
| LLM JSON 解析脆弱 | ✅ first/last brace |
| 两套 gate 口径 | ✅ 旧 gate 标 legacy |
| event overlay 用 latest 回测历史 | ✅ WARNING + TODO |

### 17.3 P0：全项目数据时间字典

**问题**：同一个数据源在不同模块里的可用时间假设不同。

**方案**：创建 `docs/data_time_dictionary.md`，为每个数据源定义 5 个时间点：

```text
| 数据源 | event_date | publish_time | available_time | signal_time | execution_time |
|--------|------------|--------------|----------------|-------------|----------------|
| Alpha158 (Qlib) | T 收盘 | T 收盘 | T 收盘 | T+1 信号 | T+1 open |
| st_daily_basic | T 收盘 | T 17:00~ | T+1 BDay | T+1 信号 | T+1 open |
| st_moneyflow | T 收盘 | T 17:00~ | T+1 BDay | T+1 信号 | T+1 open |
| northbound HSGT | T 收盘 | T 17:00~ | T+1 BDay | T+1 信号 | T+1 open |
| st_margin_detail | T 收盘 | T 18:00~ | T+1 BDay | T+1 信号 | T+1 open |
| 财报 (有 ann_date) | 报告期末 | ann_date | ann_date+1 BDay | 下一交易日 | 下下交易日 open |
| 财报 (无 ann_date) | 报告期末 | 未知 | 法定截止+N天 | 截止后首交易日 | 再下一交易日 |
| LLM event | 新闻时间 | publish_time | >=15:00 → T+1 | T+1 信号 | T+1 open |
| Shibor | T 11:00 | T 11:00 | T (盘中可用) | T 信号 | T 交易 |
| CPI/PMI/M2 | 公布日 | 公布时间 | 公布日 | 下一交易日 | 下下交易日 |
```

**验收**：所有 `feature_merger._load_*` 和 `regime_controller._*` 方法的 lag 与字典一致。

### 17.4 P0：Paper OMS 改 pending order 模式

**问题**：当天运行 paper trading 时 `Ref($open, -1)` 拿的是未来价格（如果 Qlib 数据还没更新，会拿到 NaN 然后 fallback 到 close）。不同时间运行结果不同。

**方案**：

```text
当前：signal(T) → 立即计算 fill price → 立即更新持仓
改成：signal(T) → 生成 pending_orders → T+1 开盘后获取真实 open → 撮合 → 更新持仓
```

具体改动：
1. `run_daily()` 拆成两步：`generate_orders(date)` 和 `reconcile(date)`
2. `generate_orders` 只生成目标组合和订单列表，不执行
3. `reconcile` 在 T+1 数据可用后执行，用真实 open 价格撮合
4. 订单文件存 `data/storage/paper/pending_orders_{date}.json`
5. RiskGuard 在 `generate_orders` 阶段用 T close 做初步检查，在 `reconcile` 阶段用真实 open 做最终检查

**新增 crontab**：
- 16:00 `generate_orders` → 生成订单
- 次日 10:00 `reconcile` → 用真实 open 撮合

### 17.5 P1：现有 event overlay 接入 Alpha Factory gate

**问题**：B'+C' event overlay 在 shadow 中运行，但没有通过 Alpha Factory 的标准 gate 流程。

**方案**：
1. 把 `build_event_overlay.py` 的 `gated_event_score` 注册为 `CandidateFactor`
2. 用 Alpha Factory 跑标准 tearsheet（IC/RankIC/spread/coverage/negative control）
3. 结果写入 `data/storage/candidate_factors/event_overlay_bpc/tearsheet.json`
4. 通过 gate → 保持 shadow；不通过 → 标记 fail 并停止 shadow

### 17.6 P1：regime_controller 历史 replay PIT 审计

**问题**：CX 指出 policy_support 和 theme_breadth 虽然加了文件名 PIT 过滤，但整个 replay 还没有系统审计过。

**方案**：
1. 写 `scripts/regime_pit_audit.py`
2. 对每个分数，比较"当天 compute() 结果"和"用未来数据 compute() 结果"
3. 差异 > 0 的分数标记为 PIT-unsafe，输出报告
4. 报告格式：`data/storage/regime_pit_audit.json`

### 17.7 P1：统一 rolling split 定义

**问题**：`rolling_train.py`、`train_lgb.py`、`phase4_rolling_gate.py`、`fast_rolling_gate.py` 各自定义滚动窗口，参数硬编码，代码重复。

**方案**：
1. 创建 `config/rolling_splits.py`，定义标准 24-split 窗口配置
2. 提供 `generate_splits(n_splits, train_days, valid_days, test_days, end_date)` 函数
3. 所有滚动训练/评估脚本从这里读取窗口配置
4. 不修改现有脚本的核心逻辑，只替换窗口定义部分

### 17.8 P2：backtest engine 时间语义显式化

**问题**：`portfolio_backtest.py` 的 PnL 日期虽然已修为 realized dates，但 signal date / formation date / return date 没有显式区分。

**方案**：在 `PortfolioResult` 中增加字段：

```python
signal_dates: list      # 信号生成日期
formation_dates: list   # 组合形成日期（= signal_dates for T+0）
return_dates: list      # 收益实现日期（= signal_dates shifted by 1）
```

### 17.9 P2：生产调度健壮性

**问题**：23 个 crontab 之间没有依赖管理。如果上游数据拉取失败，下游训练/推荐仍会运行。

**方案**：
1. 每个 crontab job 在完成后写 `data/storage/job_status/{job_name}_{date}.json`
2. 下游 job 启动前检查上游 status 文件
3. 如果上游失败或缺失 → 跳过执行 + 发送告警
4. 不做复杂的 DAG 调度器，只做简单的文件锁检查

### 17.10 Phase 4W 优先级总览

| 优先级 | 任务 | 工作量 | 可信度提升 |
|--------|------|--------|-----------|
| **P0** | 全项目数据时间字典 | 小（文档） | 高 — 消除歧义 |
| **P0** | Paper OMS pending order 模式 | 中 | 高 — live/backtest 语义统一 |
| **P1** | event overlay 接入 Alpha Factory | 小 | 中 — 研究晋级纪律 |
| **P1** | regime PIT 审计脚本 | 小 | 中 — 发现残留未来信息 |
| **P1** | 统一 rolling split 定义 | 小 | 中 — 消除脚本分叉 |
| **P2** | backtest 时间语义显式化 | 小 | 低 — 可读性 |
| **P2** | 生产调度依赖检查 | 中 | 中 — 防止脏数据传播 |

### 17.11 完成标准

Phase 4W 完成时，项目应该满足：

1. **任意数据源**都能在时间字典中查到 5 个时间点
2. **Paper trading** 的 PnL 和 live 执行的 PnL 语义完全一致
3. **任意因子/模型/overlay** 的晋级都通过 `tracker/promotion_gate.py`
4. **regime controller** 的历史 replay 有 PIT 审计报告
5. **所有滚动训练脚本** 共享同一套窗口配置

达到这些标准后，CX 的"实盘可信度"评分应该从 5.5-6 提升到 7-7.5。

---

## 18. Phase 4E：Model Ensemble 基建（2026-05-24 CX 设计）

**核心判断**：之前 XGB + Ranker 简单 rank 加权没赢，原因不是 ensemble 方向错，而是方式太浅。前沿做法是分层 ensemble：模型层、样本/特征层、regime routing 层、二阶段 rerank 层各司其职。

**原则**：XGB174/optimizer_v2 主线不动，ensemble 先 shadow。

### 18.1 Model Zoo Artifact 统一

所有模型必须输出标准 artifact（通过 `tracker/artifact_contract.py`）：

| 模型 | 特征集 | 状态 |
|------|--------|------|
| XGB174 | FS-174 | ✅ champion |
| XGB175 | FS-175 | 待训练 |
| LGB regression | FS-174 | 待训练 |
| CatBoost regression | FS-174 | 待训练 |
| LGBMRanker | FS-174 | 已有结果，待标准化 |
| DoubleEnsemble | FS-174 | 待实现 |
| Alpha360-XGB | FS-360 | ✅ artifact 已建 |

### 18.2 Ensemble Fusion

**新增文件**：`models/ensemble_fusion.py`

三种融合方式（不允许 raw score mean）：

```text
rank_mean:       mean(rank_percentile_per_model)
robust_z_mean:   mean(winsorized_zscore_per_model)
rolling_ic_weighted:
  weight_m,t ∝ max(0, rolling_rank_ic_m,t) / vol(rank_ic_m,t)
```

**约束**：
- 单模型权重 ≤ 60%
- Ranker 权重 ≤ 25%
- 深度模型权重 ≤ 15%
- LLM/event overlay 不参与主 ensemble，只参与 rerank

### 18.3 Model Disagreement 特征

```text
model_disagreement_i,t = std(rank_xgb, rank_lgb, rank_cat, rank_ranker)
model_consensus_i,t = mean(top_decile_votes)
```

不直接进主模型，先作为 ensemble report + Phase 4R meta-filter 特征。

### 18.4 验收门槛（硬）

ensemble 必须和**最强单模型**比，不是和平均模型比：
- `ensemble_rank_ic > best_single_rank_ic * 1.05`
- `ensemble_top20_spread > best_single_spread * 1.10`
- 24 split 中 ≥ 16 split 净收益优于 XGB175
- avg turnover 不恶化超过 15%
- max drawdown 不恶化
- 行业/市值暴露不明显漂移
- negative control 不通过则直接否决

### 18.5 参考文献

- Qlib DoubleEnsemble: 样本重加权 + 特征选择
- Qlib TRA: 市场模式 routing
- Pooling/winsorizing ML forecasts (多市场实证)
- SABER: ranking 不确定性

---

## 19. Phase 4R：Meta-filter / Reranker（接 `phase4r-rerank-meta-label-spec.md`）

**目标**：不重新选全市场，判断 XGB/ensemble Top100 里哪些更可信。

### 19.1 流程

```text
XGB175 or ensemble_score
→ Top100 candidate pool
→ meta-filter (binary classifier)
→ 去掉 meta_prob 最低 20%-30%
→ optimizer_v2
```

### 19.2 Meta Label

```text
meta_label = future_5d_return > median(candidate_pool_return) + cost_threshold
```

### 19.3 Meta 特征（不重复 174 维）

```text
xgb_rank_pct              # 主模型排名百分位
ensemble_rank_pct          # ensemble 排名百分位
model_disagreement         # 模型分歧度
rank_gap_to_cutoff         # 距离入选边界的距离
volatility_20d             # 20 日波动率
adv_20d                    # 20 日日均成交额
market_cap_bucket          # 市值分档
industry                   # 行业
event_alpha                # LLM 事件得分
regime_risk_on             # regime 综合分数
correlation_penalty        # 与持仓相关性惩罚
```

### 19.4 上线方式

保守版（初始）：只过滤 `meta_prob` 最低 20%，不重排。

进阶版：`0.80 * rank(base_score) + 0.20 * rank(meta_prob)`

### 19.5 验收

- 24 split OOS meta-filter 后组合收益 > 未过滤版
- 过滤掉的 20% 确实表现更差（事后验证）
- 换手不因过滤而大幅增加

---

## 20. Phase 5E：Regime-aware Ensemble（4E/4R 通过后）

**目标**：不同市场状态用不同模型权重。

```text
normal:          XGB/CatBoost 权重大
policy_bull:     event overlay 权重上升（设 cap）
microcap_crash:  liquidity/risk penalty 权重上升
external_shock:  event 降权，risk model 权重上升
high_vol:        ranker/追涨模型降权
```

实现先用规则 + rolling 表现，不上复杂 neural routing：

```text
weight_m,t ∝ rolling_rank_ic_m,t / rolling_vol_m,t
```

长期再考虑 Qlib TRA routing。

---

## 21. 不优先做的 Ensemble 方向

| 方向 | 理由 |
|------|------|
| RL ensemble | 当前执行层未成熟 |
| 复杂 stacking neural net | 过拟合风险高，可解释性差 |
| Transformer ensemble 生产化 | MPS 不稳定 |
| RD-Agent 自动改模型 | 需要 Linux + Docker |

### 推荐 Phase 排序（终版）

```text
Phase 4W：可信度收敛             ✅ 已完成
Phase 4E：Model Ensemble 基建     ← 当前
Phase 4R：Meta-filter / Reranker
Phase 4T-V：政策/情绪因子
Phase 5B：Deep Sequence Models
Phase 5E：Regime-aware Ensemble
Phase 6：Execution / Paper OMS 生产
```

---

## 22. Phase 4G：Feature Path 统一 + Factor Inventory（2026-05-24 CX 深度审查）

**背景**：CX 重新扫描代码后发现核心认知偏差——XGB174 实际只吃 Alpha158 + 3 flow + 13 custom + holder + regime 广播。大量"已有"因子（ST moneyflow 大小单、northbound per-stock、CYQ、block trade、龙虎榜、LLM event）只有 loader 和数据，没有进入 champion 主链路。

**核心问题**：项目有两条 feature path，语义不统一。

```text
路径 A（XGB174/175 主链路）:
  models/feature_pipeline.py → prepare_features_174()
  Alpha158 + 3 flow + 13 custom + holder + cross-market regime
  → feature_cache_174_holder_regime_ma.parquet (207 列)

路径 B（LGB/production supplement）:
  scripts/train_lgb.py → _load_supplementary()
  Alpha158 + 大量补充因子（ST daily_basic, moneyflow, northbound, quality...）
  → 没有统一 cache，每次训练临时加载
```

这导致：XGB gate 测出来的结论 ≠ LGB production 链路的结论。

### 22.1 当前主模型真实覆盖

| 类别 | 实际进入 cache | 数量 |
|------|---------------|------|
| Alpha158 基础价量技术 | ✅ | 158 |
| 资金流简化 (flow_net_mf) | ✅ | 3 |
| 自定义估值/换手/成交 | ✅ | 13 |
| 股东户数 | ✅ | 1 |
| 恒生/恒科/纳指 regime 广播 | ✅ | 27 |
| MA 辅助列 | ✅ | 3 |
| **合计** | | **205 feature + 2 label** |

### 22.2 项目有但未进主模型的因子

| 因子类 | 数据状态 | 主模型状态 | 建议 |
|--------|---------|-----------|------|
| ST daily_basic (PE/PB/MV) | ✅ 有 parquet | ❌ 未进 cache | P1 晋级候选 |
| ST moneyflow 大单/小单 | ✅ 有 parquet | ❌ 未进 cache | P1 晋级候选 |
| northbound per-stock | ✅ 有 parquet | ❌ 未进 cache | P2 |
| CYQ 筹码分布 | ✅ 有 parquet | ❌ 未进 cache | P3（高自相关） |
| block trade 大宗交易 | ✅ 有 parquet | ❌ 未进 cache | P3（事件型） |
| 龙虎榜/机构席位 | ✅ 有 parquet | ❌ 未进 cache | P2 overlay |
| LLM event factors | ✅ 有 pipeline | ❌ 未进 cache | overlay/rerank |
| sector_spillover | ✅ 有脚本 | ❌ 未进 cache | P1 优先 |
| fundamental quality | ✅ 有 parquet | ❌ 未进 cache | P2 |
| pledge 股权质押 | ✅ 有 parquet | ❌ 未进 cache | P3 |
| guba 情绪 | ⚠️ 1 个文件 | ❌ | 等数据积累 |
| fund_portfolio 公募重仓 | ⚠️ 部分数据 | ❌ | P3 |

### 22.3 A 股大涨逻辑缺口

当前主模型最强的是"价量交易结构 + 基础估值换手"。距离完整 A 股大涨逻辑还缺：

| 大涨逻辑 | 覆盖状态 | 优先级 |
|---------|---------|--------|
| 趋势/动量/反转/波动 | ✅ Alpha158 覆盖 | - |
| 成交量/换手异常 | ✅ 已有 | - |
| 基础估值 PE/PB | ✅ 已有 | - |
| 简单资金流 | ✅ 3 个 flow | - |
| **事件 surprise** | ❌ LLM 有但未校准 | **P1** |
| **板块扩散/主题强度** | ❌ 有脚本没生产化 | **P1** |
| **龙虎榜/游资** | ❌ 有数据没生产化 | **P2** |
| **公募/ETF 拥挤度** | ❌ 部分数据 | **P2** |
| **涨停连板情绪** | ❌ monster_stock 未生产化 | **P2** |
| 分钟级/盘口信号 | ❌ 仅日频 | P3 |

### 22.4 P0：Factor Inventory 自动生成

创建 `scripts/generate_factor_inventory.py`，自动扫描并输出：

```json
{
  "column_name": "flow_net_mf_5d",
  "source_parquet": "feature_cache_174_holder_regime_ma.parquet",
  "factor_group": "capital_flow",
  "in_feature_cache_174": true,
  "in_train_lgb_supplementary": true,
  "in_champion_xgb": true,
  "pit_safe": true,
  "last_gate_result": "pass",
  "rank_ic": 0.012,
  "notes": ""
}
```

### 22.5 P0：Official Champion Feature Path 定义

创建 `config/feature_path.py`，明确定义：

```python
CHAMPION_PATH = {
    "cache_file": "feature_cache_174_holder_regime_ma.parquet",
    "builder": "models.feature_pipeline.prepare_features_174",
    "feature_count": 205,
    "label": "__label_5d",
    "status": "champion",
}

SUPPLEMENT_PATH = {
    "loader": "models.feature_merger.FeatureMerger._load_supplementary",
    "status": "research_only",
    "note": "NOT used by champion; only for ablation/exploration",
}

# 因子晋级流程：
# candidate_factors/ → Alpha Factory gate → supplement ablation → cache rebuild → 24-split gate → shadow → champion
```

### 22.6 P1：优先晋级三类因子

1. **moneyflow_v2**：大单/小单/主力流，比现有 3 个 flow 更细
2. **sector_spillover / sector heat**：A 股板块效应强
3. **event surprise overlay**：重大合同、业绩预告、回购、增持 → overlay/reranker

### 22.7 CX 总评

> 这个项目现在不是玩具了，已经进入"研究体系初成，但生产语义需要收敛"的阶段。
> 当前真正强的是价量模型和研究框架；距离"百亿私募味道"的差距，不在于再随手加 100 个因子，而在于把候选因子纳入统一晋级制度。

---

## 23. Phase 4N 重构：LLM/Event 三层架构（2026-05-24 CX LLM 深度审查）

**核心问题**：当前 LLM 被要求直接预测收益 impact（`impact_1d/impact_5d`），把"信息抽取"和"收益定价"混在一起。LLM 对"发生了什么"很强，但对"1 日涨跌多少"不稳定。

**正确架构**：LLM 做信息抽取，quant 模型做收益定价。

### 23.1 当前链路问题清单

| 问题 | 位置 | 影响 |
|------|------|------|
| LLM 直接预测 impact_1d/5d | `llm_event_extractor.py` prompt | impact 数值尺度不稳定 |
| build_llm_event_factors 直接用 impact×confidence×... | `build_llm_event_factors.py:171` | LLM 主观判断当 alpha 本体 |
| legacy llm_events/ + unified events/ 双轨并行 | pipeline + validate | 同一事件两个版本、口径不同 |
| 时间语义三套逻辑 | EventStore/build_factors/validate | 不是同一个"唯一真相" |
| overlay backtest 承认不是 PIT-safe | `build_event_overlay.py:150` | 不能作为晋级依据 |
| max_news_per_stock=1 太粗 | `run_llm_event_pipeline.py:173` | 可能漏掉重要公告 |

### 23.2 三层架构

```text
第一层：结构化事件抽取（LLM 只做这一层）
  输入：新闻/公告/政策原文
  输出：event_type, direction, magnitude_raw, amount, is_first_time, is_rumor, source_quality
  不输出：impact_1d, impact_5d

第二层：事件收益校准（quant 模型做这一层）
  输入：event_type × 行业 × 市值分层 × 市场 regime × 是否超预期
  输出：calibrated_alpha_1d, calibrated_alpha_5d
  来源：历史校准表，不是 LLM

第三层：overlay / reranker / risk flag
  XGB174 给 base score
  event_model 给 event_alpha
  只对高置信事件的股票调分
  低覆盖事件不填 0
```

### 23.3 LLM 应输出的字段（替代 impact_1d/5d）

```json
{
  "event_type": "major_contract",
  "direction": 1,
  "magnitude_raw": "large",
  "amount_wan": 50000,
  "amount_ratio_to_revenue": 0.15,
  "amount_ratio_to_mcap": 0.03,
  "customer_quality": "government",
  "is_first_time": true,
  "is_repeat_announcement": false,
  "is_policy_related": false,
  "is_rumor": false,
  "source_quality": 0.9,
  "publish_time": "2026-05-24T17:30:00",
  "summary": "..."
}
```

### 23.4 事件 Surprise 因子（P1 优先）

| 因子 | 公式 | 用途 |
|------|------|------|
| `evt_contract_revenue_ratio` | 合同金额 / 最近年营收 | 合同规模相对性 |
| `evt_buyback_mcap_ratio` | 回购金额 / 市值 | 回购力度 |
| `evt_insider_buy_float_ratio` | 增持金额 / 自由流通市值 | 增持信号强度 |
| `evt_forecast_surprise` | 业绩预告中值 - 历史基线 | 业绩超预期 |
| `evt_penalty_np_ratio` | 处罚金额 / 净利润 | 处罚严重度 |
| `evt_calibrated_alpha_1d` | 校准表输出 | 历史同类事件 T+1 超额 |
| `evt_calibrated_alpha_5d` | 校准表输出 | 历史同类事件 T+5 超额 |
| `evt_event_count_20d` | 近 20 日事件数 | 关注度/催化密度 |
| `evt_novelty_score` | 是否首次 × 非重复 | 信息新鲜度 |
| `evt_source_quality` | 来源权重 | 可信度 |

接入方式：**先 overlay/reranker，不进 XGB 主特征。**

### 23.5 Phase 4N 细分

| Phase | 内容 | 前置 | 可现在做？ |
|-------|------|------|-----------|
| **4N-1** | EventStore 统一：废弃 legacy 双轨，统一 5 时间字段 | 无 | ✅ |
| **4N-2** | 事件 surprise 特征工程 | 4N-1 | ✅ |
| **4N-3** | 历史校准表：event_type × 行业 × 市值 × regime 分桶 | 60+ 天事件数据 | ⏳ |
| **4N-4** | overlay/reranker rolling PIT 验证 | 4N-3 + rolling pred artifact | ⏳ |

### 23.6 板块扩散因子（P1 第二优先）

A 股大涨逻辑：政策/主题 → 板块龙头涨停 → 二线扩散 → 补涨。

| 因子 | 说明 |
|------|------|
| `sector_ret_1d/3d/5d` | 行业近期收益 |
| `sector_volume_zscore` | 行业成交量异常 |
| `sector_limit_up_count` | 行业涨停家数 |
| `sector_up_ratio` | 行业上涨比例 |
| `sector_turnover_zscore` | 行业换手异常 |
| `sector_moneyflow_rank` | 行业资金流排名 |
| `stock_beta_to_hot_sector` | 个股对热门板块的弹性 |
| `stock_relative_strength_in_sector` | 个股在行业内相对强弱 |
| `sector_heat` | 综合板块热度 |
| `sector_breadth` | 板块扩散宽度 |

接入方式：可进主 cache，但必须经过 24-split gate。

### 23.7 龙虎榜/游资（P2 reranker/risk tag）

| 因子 | 说明 |
|------|------|
| `ti_net_buy_mcap_ratio` | 净买入/市值 |
| `ti_inst_buy_ratio` | 机构席位占比 |
| `ti_hot_money_buy_ratio` | 游资席位占比 |
| `ti_repeat_seat_count` | 重复席位次数 |
| `ti_famous_seat_score` | 知名游资评分 |
| `ti_after_limit_up_flag` | 涨停后龙虎榜 |
| `ti_lhb_5d_decay` | 5 日衰减 |

**注意**：龙虎榜容易变成"已涨后追高"信号。用于 risk/加速识别，不直接用于选股 alpha。

### 23.8 涨停情绪（P2 独立模型）

| 因子 | 说明 |
|------|------|
| `limit_up_chain_len` | 连板天数 |
| `open_board_count` | 炸板次数 |
| `sealed_board_strength` | 封板强度 |
| `first_limit_up_time` | 首次封板时间 |
| `sector_limit_up_breadth` | 板块涨停扩散 |
| `yesterday_limit_up_premium` | 昨日涨停今日溢价 |
| `dragon_head_score` | 龙头评分 |

**不混入全市场回归**——分布太极端，会污染普通选股信号。用于短线加速/涨停后风险模型。

### 23.9 公募拥挤度（Phase 5，中期模型）

| 因子 | 说明 |
|------|------|
| `fund_holding_pct` | 基金持仓占比 |
| `fund_holding_change_qoq` | 季度环比变化 |
| `num_funds_holding` | 持有基金数 |
| `fund_crowding_zscore` | 拥挤度 z-score |
| `crowding_unwind_risk` | 抱团瓦解风险 |

中期选股和风格暴露控制用，不抢 Phase 4 主线。

### 23.10 新闻筛选改进

当前 `max_news_per_stock=1` 太粗。改为规则优先筛选：

```text
1. 公告优先于新闻
2. 标题命中关键词：重大合同/中标/回购/增持/减持/业绩预告/处罚/诉讼/股权激励
3. 来源优先级排序（交易所公告 > 财联社 > 东财 > 其他）
4. 去重后 top K
5. LLM 只吃"候选高信息密度文本"
```

### 23.11 开发顺序（终版）

```text
1. EventStore 统一 + 5 时间字段 → 可现在做
2. LLM prompt 改成"结构化事实"（去掉 impact_1d/5d） → 可现在做
3. 事件 surprise overlay → 可现在做（用已有事件数据）
4. 板块热度/扩散 candidate factor → 可现在做
5. 龙虎榜/涨停做 reranker/risk tag → P2
6. 公募拥挤度 → Phase 5
7. 历史校准表 → 需 60+ 天数据
8. overlay rolling PIT 验证 → 需 rolling pred artifact
```

---

## 24. Phase 4O：Downside Risk Layer（2026-05-24 CX 设计）

**核心判断**：大跌比大涨更应该优先工程化。大涨因子提高收益上限，大跌因子降低回撤和踩雷概率。当前 regime（市场层）+ RiskGuard（止损）有框架，但个股级暴雷/踩踏/退潮因子明显不够。

### 24.1 Downside Model 架构

```text
输入：负面事件因子 + 交易踩踏因子 + 板块退潮因子 + 拥挤瓦解因子
模型：LightGBM/XGBoost classifier
Label：
  crash_1d: 未来 1 日跌幅 < -5%
  crash_5d: 未来 5 日跌幅 < -10%
  max_dd_5d: 未来 5 日最大回撤
  underperform_5d: 未来 5 日跑输行业 > 8%

输出：
  crash_prob_1d / crash_prob_5d
  cannot_buy flag
  reduce_weight / force_sell 建议

接入方式：
  final_score = xgb_alpha_score - lambda * crash_prob_5d
  crash_prob_5d > 0.65 → cannot_buy
  crash_prob_5d > 0.80 且已持仓 → reduce_weight
  负面重大事件 + 跌破 MA20 + 资金流出 → force_sell
```

### 24.2 四阶段因子

**第一阶段：事件型大跌因子**
```text
neg_regulatory_penalty          处罚金额 / 净利润
neg_lawsuit                     诉讼金额 / 净资产
neg_investigation               立案调查
neg_audit_opinion               审计非标
neg_earnings_miss               业绩不及预期
neg_forecast_down                业绩预告大幅下修
neg_shareholder_sell             大股东减持 / 市值
neg_unlock_pressure              解禁金额 / 自由流通市值
neg_pledge_risk                  质押率
neg_debt_default                 债务违约
```

**第二阶段：交易踩踏因子**
```text
limit_down_today                今日跌停
limit_down_chain_len            连续跌停天数
volume_collapse                 成交额骤降
turnover_spike_then_drop        换手冲高后价格下跌
main_flow_adv_negative          主力资金大幅净流出
block_trade_discount             大宗交易折价率
block_trade_volume_ratio        大宗交易占比
margin_balance_drop              融资余额快速下降
```

**第三阶段：板块退潮因子**
```text
sector_limit_up_count_drop       板块涨停家数骤降
sector_limit_down_count          板块跌停家数
sector_breadth_collapse          板块上涨比例崩塌
leader_drawdown                  龙头回撤
yesterday_limit_up_premium_neg   昨日涨停今日折价
theme_heat_decay                 主题热度衰减
```

**第四阶段：拥挤瓦解因子**（Phase 5）
```text
fund_crowding_zscore             基金持仓拥挤度
crowded_stock_drawdown           高拥挤股回撤
northbound_holding_drop          北向持仓骤降
holder_num_increase              股东户数暴增
```

### 24.3 最优先改的 5 点

1. `_check_limit_down()` 做实：用前收盘 × 涨跌停规则判断 cannot_sell
2. 建 crash label：1d<-5%、5d<-10%、5d max drawdown
3. 负面 EventStore 校准：不依赖 LLM 直接给 impact
4. moneyflow_v2 + block_trade_v2 + pledge + forecast 组成 downside candidate set
5. 推荐列表后加 risk reranker：高 alpha 但高 crash_prob 降权或剔除

---

## 25. Phase 4T-1~7：LLM Pipeline 收敛（2026-05-24 CX LLM 全链路审查）

**核心判断**：LLM pipeline 方向对，但处在"多代方案并存"状态。建设性建议是把 V2/Calibration/EventStore 扶正，V1 降级。

### 25.1 七步收敛计划

| 步骤 | 内容 | 状态 |
|------|------|------|
| **4T-1** | 默认切到 V2 extractor | 可现在做 |
| **4T-2** | EventStore 成为唯一正式事件源 | ✅ 今天已实现 |
| **4T-3** | 事件候选筛选器（规则优先） | 可现在做 |
| **4T-4** | 公告正文增强（高价值公告拉详情） | 需要 API 开发 |
| **4T-5** | 历史校准表升级 | ⏳ 需 60+ 天数据 |
| **4T-6** | overlay rolling PIT validation | ⏳ 需 rolling pred |
| **4T-7** | 接入推荐后处理 | 4T-6 通过后 |

### 25.2 新闻筛选规则（4T-3）

```text
高优先级关键词：
  重大合同、中标、回购、增持、减持、业绩预告、亏损、扭亏、
  处罚、立案、诉讼、仲裁、解禁、质押、债务、评级、重组、
  并购、政府补贴、订单、市占率

来源优先级：
  交易所公告 > 巨潮/上交所/深交所 > 权威财经媒体 > 普通媒体

路由规则：
  official/high-impact → 全部送 LLM
  普通新闻 → 每股最多 1 条
  social/duplicate → 默认不送
```

### 25.3 validation 升级方向

事件因子不用全市场 IC 验收（低覆盖天然吃亏）。正确验收：

1. Top20 spread uplift（加 overlay 后 vs 不加）
2. event-stock hit rate
3. negative-event drawdown reduction
4. coverage by source
5. false positive rate
6. overlay turnover impact

### 25.4 health check 升级

当前只看 news_count / event_count。应看质量：

```text
official_event_count
price_sensitive_count
routine_announcement_ratio
parse_fail_rate
source_distribution
positive/negative/neutral ratio
EventStore stored count
factor_rows_generated
```

---

## 26. Phase 4U：Global Supply Chain Event Overlay（CX 设计）

**核心思路**：A 股很多公司是全球产业链的影子资产。全球产业新闻 → 供应链映射 → A 股受益/受损公司 → LLM 抽事实 → 校准成因子/overlay。第一阶段做轻量 edge-weight propagation，不上 GNN。

### 26.1 四层架构

```text
Layer 1: 全球产业事件采集 (GDELT / Google RSS / 海外财报)
Layer 2: 供应链实体映射 (200-500 条高置信边，半人工)
Layer 3: LLM 抽事实 (V2 extractor 风格，不预测收益)
Layer 4: 因子/overlay 验证
```

### 26.2 供应链边表

```text
data/storage/supply_chain_edges.parquet

示例：
Nvidia  → 中际旭创/沪电股份/工业富联  (AI server / PCB / optical)
Apple   → 立讯精密/歌尔股份/蓝思科技  (consumer electronics)
Tesla   → 拓普集团/三花智控/旭升集团  (EV parts)
TSMC    → 北方华创/中微公司/沪硅产业  (semiconductor equipment)
```

### 26.3 全球事件因子

```text
global_chain_alpha_1d/5d
global_chain_event_count_5d
global_chain_customer_shock
global_chain_supplier_shock
global_chain_policy_risk
global_chain_commodity_pressure
```

### 26.4 验收

不用全市场 IC。按 topic 分别看：AI 算力、苹果链、特斯拉链、锂电、光伏、半导体设备。

### 26.5 学术参考

- Cohen & Frazzini: Economic Links and Predictable Returns
- Herskovic et al.: Firm Volatility in Granular Networks
- FinDKG: LLM + Dynamic Knowledge Graph
- Temporal Relational Ranking for Stock Prediction

---

## 27. 网络分层架构（CX 设计）

**问题**：国内 API 走代理超时，全球 API 不走代理失败。现在每个脚本零散处理，不可靠。

### 27.1 三档网络 profile

```text
domestic_no_proxy:  清空 http_proxy/https_proxy/ALL_PROXY
                    用于 baostock/ST_CLIENT/东财/AKShare
global_with_proxy:  检查代理端口 → 设置 proxy → 连通性检测
                    用于 GDELT/Google RSS/BBC/海外 API
none:               不联网，纯计算
                    用于 factor build/merge/train/predict
```

### 27.2 统一 wrapper

```text
scripts/run_network_job.py --network domestic|global|none -- command...
```

### 27.3 cron 任务分类

```text
国内链 (domestic):  morning_rec / collect_news / qlib_update / regime / valuation
全球链 (global):    gdelt / global_supply_chain / overnight_context
计算链 (none):      event_store / factor_build / train / predict / paper_trading
LLM 链:            拆成 collect(domestic/global) + extract(看 LLM API) + build(none)
```

### 27.4 代理健康检查

```text
1. 本地端口检测: connect 127.0.0.1:proxy_port
2. 外网检测: curl https://www.google.com/generate_204
3. 失败: 尝试 zsh -ic 'ssproxy' → 再测 → 仍失败则 job fail
```

---

## 28. Phase 4X：网络分层 + 数据拉取稳定性（CX 详细设计）

**目标**：拉数据万无一失。国内链不碰 VPN，全球链自动确保 ssproxy，混合任务拆开。任何联网环节都有硬超时、网络 profile、失败降级、下游熔断。

**硬规则**（不可违反）：
1. 任何数据任务都不允许无限等待
2. 任何 global/proxy 任务失败，不能拖死 domestic 主链
3. 任何 domestic/no_proxy 任务不能继承代理环境
4. 任何混合任务必须拆开
5. 任何下游任务必须检查上游 data_health

### 28.1 实施计划（10 步）

| 步骤 | 内容 | 优先级 |
|------|------|--------|
| 1 | 新增 `run_network_job.py` 统一 wrapper | P0 |
| 2 | 业务脚本不自己管网络 | P0 原则 |
| 3 | `install_crontab.py` CronJob 加 `network` 字段 | P0 |
| 4 | 现有 job 分类标注 domestic/global/none/llm | P0 |
| 5 | 拆 `run_llm_event_pipeline.py` 成 4 个独立 job | P1 |
| 6 | 代理启动策略：先检测 → 再启动 → 等待 → 重试 | P1 |
| 7 | 日志 + health check 增加网络链路状态 | P1 |
| 8 | 失败隔离：global 失败不影响 domestic | P1 |
| 9 | cron 时间排布优化（含 retry 时段） | P2 |
| 10 | 清理各脚本内部零散 proxy unset | P2 |

### 28.2 网络配置文件

```python
# config/network_profiles.py
PROXY_URL = "http://127.0.0.1:7890"
PROXY_START_CMD = ["zsh", "-ic", "ssproxy"]
GLOBAL_CHECK_URLS = [
    "https://api.gdeltproject.org/api/v2/doc/doc",
    "https://www.google.com/generate_204",
]
DOMESTIC_CHECK_URLS = [
    "https://emappdata.eastmoney.com",
]
```

### 28.3 Cron 任务分类（终版）

**国内链 (domestic)**：
```text
09:20 morning_recommendation
14:30 sell_check
16:25 domestic_event_collect (公告+新闻)
16:35 guba_popularity
17:45 qlib_data_update
17:55 fund_flow_update
18:00 valuation_update
18:05 regime_daily_update
```

**全球链 (global)**：
```text
16:30 global_macro_news_update
16:40 global_supply_chain_news
21:30 global_news_retry
```

**计算链 (none)**：
```text
17:05 event_factor_build
18:35 smoke_lgb_predict
18:40 shadow_optimizer (pending order generate)
18:42 paper_trading (pending order generate)
18:55 daily_health_check
次日 10:00 reconcile (pending order fill)
```

**LLM 链 (llm)**：
```text
16:50 llm_event_extract
22:10 llm_event_retry
```

### 28.4 LLM pipeline 拆分

```text
现在：run_llm_event_pipeline.py = collect + extract + build (混合)
改成：
  domestic_event_collect.py   → domestic
  global_event_collect.py     → global
  llm_event_extract.py        → llm (读 collected raw, 写 EventStore)
  event_factor_build.py       → none (读 EventStore, 生成因子)
```

### 28.5 验收标准

1. crontab dry-run 输出每个 job 都带 network wrapper
2. domestic job 日志 `proxy_env_set=false`
3. global job 日志 `proxy_env_set=true`
4. ssproxy 未启动时，global job 能自动启动或明确失败
5. ssproxy 启动失败，不影响 qlib_data_update
6. daily_health_check 区分国内/全球/LLM 链路状态
7. 连跑 5 个交易日，国内数据更新成功率不下降

### 28.6 逐环节防卡死要求

**国内核心链**（baostock / ST_CLIENT / AKShare / 东财）：
- network=domestic，强制 unset proxy
- 每个 requests timeout ≤ 15s
- 整个 job global timeout 30-45min
- 失败写 data_health
- **核心行情失败 → 禁止训练/推荐**

**全球链**（GDELT / RSS / 海外产业链）：
- network=global，先检测 ssproxy
- 检测失败最多启动一次 ssproxy，启动后最多等 10-15s
- 仍失败：**fail fast**，不进入业务拉取
- 业务请求 timeout ≤ 15s，整个 job timeout ≤ 10min
- **失败只关闭 global overlay，不影响国内主链**

**LLM 链**（MiniMax / future OpenAI）：
- network=llm，按配置走 domestic 或 global
- 单次 LLM timeout ≤ 15-30s，每条新闻最多 retry 1-2 次
- 整个 extraction job timeout ≤ 60-120min
- 超时保留 partial file，标记 `partial=true`
- **factor build 只使用 complete 或 quality≥threshold 的事件**

**通知链**（pushplus / WeChat）：
- timeout ≤ 10s
- **失败只写日志，绝不影响数据 job 成败**

### 28.7 run_network_job.py 防卡死逻辑

```python
if network == "domestic":
    unset_proxy_env()
    run(command, timeout=job_timeout)

elif network == "global":
    set_proxy_env()
    if not proxy_health_ok(timeout=5):
        start_ssproxy(timeout=15)
        sleep_retry(max_wait=10)
    if not proxy_health_ok(timeout=5):
        fail_fast("proxy unavailable")  # 15 秒内退出
    run(command, timeout=job_timeout)

elif network == "none":
    unset_proxy_env()
    run(command, timeout=job_timeout)
```

### 28.8 CronJob 扩展字段

```python
@dataclass(frozen=True)
class CronJob:
    job_id: str
    schedule: str
    target: list[str]
    log_name: str
    network: str = "none"           # domestic/global/none/llm/push
    timeout_sec: int = 0            # 0 = no limit
    critical: bool = False          # True = 失败阻断下游
```

推荐配置：
```text
qlib_data_update:       domestic, timeout=3600, critical=true
fund_flow_update:       domestic, timeout=1800, critical=false
global_news_update:     global,   timeout=600,  critical=false
llm_event_extract:      llm,      timeout=7200, critical=false
smoke_lgb_predict:      none,     timeout=900,  critical=true
daily_health_check:     none,     timeout=300,  critical=false
push jobs:              push,     timeout=60,   critical=false
```

### 28.9 Job Lock 防重复启动

```text
data/storage/locks/{job_id}.lock

规则：
- job 启动先拿 lock（写入 pid + 启动时间）
- 拿不到 → 检查 pid 是否活着
  - 活着：本轮 skip
  - 死了：清理 stale lock，重新拿
- job 结束释放 lock
```

防止：global proxy 卡住导致 global job 堆积、LLM extraction 超时导致 retry 重复启动、baostock 更新没结束下一个训练就开始。

### 28.10 Data Health 文件

每个数据 job 完成后写：`data/storage/data_health/YYYY-MM-DD/{source_name}.json`

```json
{
  "source": "qlib_data_update",
  "network_profile": "domestic",
  "started_at": "...",
  "finished_at": "...",
  "success": true,
  "rows_fetched": 5200,
  "symbols_covered": 5173,
  "latest_date": "2026-05-24",
  "proxy_used": false,
  "error": null
}
```

### 28.11 依赖熔断规则

```text
training 前：
  qlib_data_update.success == true
  latest_date == today
  instrument_count >= 4500
  否则：不训练

event overlay 前：
  event_factor_build.success == true
  event_count >= threshold
  partial == false
  否则：关闭 event overlay

global overlay 前：
  global_news_update.success == true
  否则：关闭 global overlay

paper trading 前：
  smoke_lgb_predict.success == true
  prediction_count >= threshold
  否则：使用昨日预测 + 标记 stale
```

### 28.12 防卡死验收清单（8 条）

1. 关闭 ssproxy → 跑 global job → **15 秒内失败退出**，不挂住
2. 开全局代理 → 跑 domestic job → 日志 `proxy_env_cleared=true`
3. 国内 API 超时 → job 在 timeout_sec 内退出
4. LLM 卡住 → 到 7200 秒自动 kill，标记 partial
5. 上一轮 job 未结束 → 下一轮 cron 不重复启动（lock）
6. global job 失败后 → qlib_data_update 仍正常运行
7. qlib_data_update 失败后 → train_lgb 不训练或标记 data stale
8. pushplus 失败 → 不影响任何数据 job
