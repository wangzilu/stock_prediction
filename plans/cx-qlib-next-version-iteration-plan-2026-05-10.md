# Qlib 下一版本迭代方案

日期：2026-05-10  
目标：把当前项目从“全 A 数据 + 单模型预测 + 初版评估回测”升级为“可追溯研究平台 + 多源影响因子 + 统一模型对照 + 组合级验证”。

## 结论

下一版不应该只换更复杂模型。真正的路线是：

1. 先把 Qlib Workflow/Recorder 接起来，让每次训练、预测、评估、回测都有实验记录。
2. 把日频资金流、北向资金、板块热度、基金重仓、舆情事件统一成影响因子层，再通过严格滞后和回测门槛进入模型。
3. 用同一套评估和回测比较 Alpha158、Alpha360、XGB/CatBoost/DoubleEnsemble、ALSTM/Transformer 等模型。
4. 用 Rolling/Online/Registry 做模型版本治理，只有通过门槛的模型才能进入生产推送。

核心判断：当前系统和头部量化私募的最大差距，不是 XGB/LGB/Transformer 这类模型名字，而是有效因子宽度。Alpha158 主要覆盖价量维度；头部机构的优势来自更多低相关信息源：基本面、资金流、北向、行业/主题、事件、公告、舆情、交易行为和组合风控。后续追平甚至超越，不能靠盲目堆到 500+ 因子，而要建立“有效、低相关、可持续、可追溯”的因子工厂。

因此，下一阶段的战略优先级是：先扩展新信息维度，再升级模型复杂度。基本面估值/质量、资金流、北向和板块资金应作为第一批增量因子同步研究；基金重仓和公告事件是披露滞后更强的中长期慢因子；Transformer/HIST/TRA 等深度模型只有在这些新维度进入标准评估闭环后，才有更大概率真正胜过 XGB。

Qlib 还有不少功能没用上，但优先级不同。最该马上用的是 Recorder/SignalRecord/PortAnaRecord、RollingGen、Alpha360/Alpha158DL、自定义表达式算子、Brinson 归因、RobustZScoreNorm/TanhProcess 处理器、更多模型和 Dataset/Expression Cache；暂时不该硬上的，是当前环境会被 cvxpy/numpy 卡住的 TopkDropoutStrategy/PortfolioOptimizer，以及 Qlib RL/高频执行。

## 证据

官方文档确认 Qlib Workflow 覆盖 Data、Model、Evaluation、Backtest，并用 Recorder 追踪训练、推理、评估阶段产物：<https://qlib.readthedocs.io/en/latest/component/workflow.html>。官方示例也把 `SignalRecord` 和 `PortAnaRecord` 放进同一个任务配置，配合 `TopkDropoutStrategy` 和交易成本做组合分析。

官方模型文档说明 Qlib Model Zoo 提供 LightGBM、MLP、LSTM 等基线模型，并支持 `SignalRecord` 保存预测结果：<https://qlib.readthedocs.io/en/latest/component/model.html>。

官方数据文档说明 Qlib 有中国股票模式、涨跌停阈值、DataHandler 处理链、ExpressionCache 和 DatasetCache：<https://qlib.readthedocs.io/en/latest/component/data.html>。

本地环境校验：

- `qlib==0.9.7`。
- 本地 handler 有 `Alpha158`、`Alpha158DL`、`Alpha158vwap`、`Alpha360`、`Alpha360DL`、`Alpha360vwap`。
- 本地 record template 有 `SignalRecord`、`SigAnaRecord`、`PortAnaRecord`、`MultiPassPortAnaRecord`、`HFSignalRecord`。
- 本地模型模块包含 `xgboost`、`catboost_model`、`double_ensemble`、`pytorch_alstm`、`pytorch_gru`、`pytorch_hist`、`pytorch_localformer`、`pytorch_tabnet`、`pytorch_tcn`、`pytorch_transformer`、`pytorch_tra` 等。
- `RollingGen(step=40, rtype='expanding')` 可用；`OnlineManager` 模块可导入。
- `qlib.backtest.profit_attribution.brinson_pa(positions, bench='SH000905', group_field='industry', ...)` 可导入，用于行业配置/选股收益归因。
- `qlib.data.ops` 本地暴露 78 个非私有名称，可用于自定义表达式因子；`qlib.data.dataset.processor` 本地可用 `RobustZScoreNorm`、`TanhProcess`、`MinMaxNorm`、`CSRankNorm` 等处理器。
- `statsmodels` 当前未安装；Qlib 可视化 report 相关模块需要先补依赖，但这不是主链阻塞项。
- cc 最新 `cc-factor-gap-analysis.md` 指出基本面估值/质量因子应和资金流同等前置，这个修正成立。本地 AKShare 校验显示 `stock_financial_analysis_indicator_em`、宏观、股东/质押/解禁/回购接口可用；但 cc 写的 `stock_a_indicator_lg` 在当前版本不存在，估值接口需要用 `stock_a_ttm_lyr`、`stock_a_all_pb`、`stock_value_em`、`stock_zh_valuation_baidu` 等可用接口做兼容适配。
- `TopkDropoutStrategy` 当前不可直接用：导入会经过 `EnhancedIndexingOptimizer -> cvxpy`，在当前 numpy/cvxpy 组合下报 `No module named 'numpy.lib.array_utils'`。

## 当前状态

已具备：

- 全 A Qlib 数据更新、health gate、staging promotion 和 4500+ 生产覆盖门槛。
- `after_close_pipeline.py` 串行执行 data update -> health -> train -> smoke -> evaluate。
- `evaluate_lgb_test.py` 初版 IC/RankIC/TopK 评估。
- `backtest_qlib_signal.py` 初版本地 TopK/TopK-dropout 近似回测。
- `train_model_suite.py` 初版多模型对照框架。
- `rolling_train.py` 初版 rolling 训练评估。
- `models/model_registry.py` 本地 JSON registry。
- `LimitUpCollector` 和妖股相关因子雏形。

还缺：

- Qlib Recorder/SignalRecord/PortAnaRecord 没接入，训练产物仍主要靠 JSON 和覆盖写模型文件。
- 影响因子层还没落库：资金流、北向、板块热度、基金重仓、舆情事件还没有统一 schema。
- Alpha360/Alpha158DL 还没进入标准模型对照。
- 多模型结果还没有统一 registry、promotion gate 和自动回滚。
- 回测仍是近似组合回测，缺 Qlib executor 的正式成交约束，也缺 RQAlpha 级别的 T+1/涨跌停/停牌细节验证。
- OnlineManager 还没用于模型灰度、更新和热切换。

## 下一版架构

```text
Qlib OHLCV / Alpha158 / Alpha360
        |
        +-- 日频资金流 / 北向资金 / 行业资金流
        +-- 板块热度 / 涨停板 / 龙虎榜
        +-- 基金季报重仓 / 绩优基金共识
        +-- 新闻公告 / 股吧雪球 / 地缘政策事件
        |
        v
factor_impacts 日频影响因子层
        |
        v
DatasetH / custom DataLoader / model suite
        |
        v
SignalRecord + SigAnaRecord + evaluate_lgb_test
        |
        v
TopK backtest + PortAnaRecord/custom backtest
        |
        v
ModelRegistry / OnlineManager / production cache
```

影响因子层必须统一字段：

| 字段 | 说明 |
|---|---|
| `date` | 因子实际生效交易日 |
| `target_type` | `market` / `sector` / `stock` |
| `target_id` | 市场、行业或股票代码 |
| `source_type` | `fund_flow` / `northbound` / `sector_heat` / `fund_holding` / `news_event` / `limit_up` |
| `factor_name` | 具体因子名 |
| `value` | 原始值 |
| `zscore` | 截面或时序标准化值 |
| `confidence` | 置信度 |
| `effective_date` | 可用于训练和推理的最早日期 |
| `decay_days` | 衰减周期 |

红线：任何因子都只能从 `effective_date` 开始使用。基金重仓必须用披露日后一个交易日；盘后资金流不能用于当天收盘前决策；新闻和舆情必须按原始时间戳入库。

## 系统能力补齐策略

头部私募强的不只是模型，而是完整闭环。下一版要把系统能力拆成六条主线，每条都有指标、失败动作和落地模块。

### 1. 数据快且干净

目标：每天盘后稳定得到可训练、可回测、可解释的数据，不让半更新数据污染模型。

应对策略：

- 数据源分层：日线主链使用 provider auto，按可用性走 TuShare/AKShare/baostock/vendor；spot 只用于推文展示和盘中动量，不写训练日线。
- Raw cache 先行：所有外部数据先写 raw cache，再转 staging，再 health gate，通过后才 promote 到生产 Qlib 或 feature store。
- Manifest 断点续跑：每个 symbol、每类因子记录 `last_success_date`、`source`、`row_count`、`checksum`、失败原因，失败后只补缺口。
- 数据口径审计：OHLCV、复权、成交额、流通市值、财报披露日、基金披露日都要有 source metadata，跨源混合前做抽样对账。
- 交易日历统一：所有预测、训练、复盘、因子 forward-fill 都以 Qlib calendar 或交易所 calendar 为准。

落地模块：

- `data/storage/raw_cache/`
- `data/storage/factor_impacts/`
- `data/storage/update_manifest.json`
- `scripts/check_qlib_data_health.py`
- `scripts/check_factor_store_health.py`

关键指标：

- 最新交易日 OHLCV 覆盖率 >= 95%。
- 生产 LGB/XGB 有效预测数 >= 4500。
- 因子覆盖率按因子族统计：资金流、基本面、宏观、北向、行业分别报告。
- 数据更新失败时生产数据不推进，模型不训练。

失败动作：

- coverage 不足：保留上一版生产数据，写失败报告。
- 某个因子族缺失：该因子族本轮禁用，模型进入 `factor_degraded` 状态。
- 复权/字段尺度异常：拒绝 promote，进入人工审计列表。

### 2. 因子每天监控衰减

目标：知道每个因子什么时候有效、什么时候变弱、什么时候应该降权或剔除。

应对策略：

- 建 `scripts/evaluate_factor_ic.py`，对每个因子计算日频 IC、RankIC、分位收益、换手、覆盖率和 1/5/10/20 日衰减曲线。
- 建因子 registry：记录因子定义、数据源、频率、effective_date 规则、当前状态、最近 20/60/120 日表现。
- 分组监控：基本面、资金流、北向、行业、事件、基金重仓分别看 IC 和相关性，避免一堆同质因子重复计权。
- 因子相关性管理：高相关因子聚类，保留信息量最高或最稳定的代表因子。
- 因子降权规则：连续窗口 RankIC 转负、换手过高或覆盖率恶化时，自动降权进入观察。

落地模块：

- `scripts/evaluate_factor_ic.py`
- `models/factor_registry.py`
- `data/storage/factor_registry/index.json`
- `data/storage/factor_diagnostics/`

关键指标：

- 20/60/120 日 IC 均值和 t-stat。
- RankIC 正值比例。
- Top quantile - bottom quantile spread。
- 因子半衰期和最优持有期。
- 因子覆盖率、异常率、换手率。

失败动作：

- 连续 3 个评估窗口 RankIC < 0：标记 `decayed`，训练时默认禁用。
- 覆盖率低于阈值：只用于解释层，不进模型。
- 与已有因子相关性过高且增量 IC 不显著：合并或删除。

### 3. 模型有 Rolling 验证

目标：不再相信单次 train/valid/test，而是用 walk-forward 证明模型跨市场阶段有效。

应对策略：

- 用 Qlib `RollingGen` 生成 expanding/sliding 任务，替代单次静态切分。
- 每个 split 都记录 train/valid/test 日期、模型参数、因子集版本、预测、IC、回测结果。
- 同一 rolling 框架对比 Alpha158、Alpha360、基本面+资金流增强特征、XGB/LGB/CatBoost/DoubleEnsemble/深度模型。
- Purged/embargoed validation 用于中长期标签和重叠窗口，避免未来信息泄漏。
- Rolling 结果决定 promotion，不由单次最新测试窗口决定。

落地模块：

- `scripts/run_rolling_workflow.py`
- `qlib.workflow.task.gen.RollingGen`
- `models/model_registry.py`
- Qlib Recorder experiment：`rolling_<model>_<feature_set>`

关键指标：

- split 级 IC/RankIC 均值、标准差、正值比例。
- 成本后 IR、Sharpe、最大回撤。
- 各市场状态下表现：强势、震荡、弱势、高波动。
- 模型稳定性：最差 split 不能过差，不能只靠单个窗口贡献。

失败动作：

- 最新模型 rolling 不达标：保留上一版生产模型。
- 新因子只在一个窗口有效：留在 research，不进 production。
- 深度模型不优于 XGB/CatBoost 基线：不进入生产，只保留实验记录。

### 4. 组合有换手、行业暴露、回撤控制

目标：从“个股清单”升级到“可执行组合”，控制交易成本和风格暴露。

应对策略：

- 初版继续用本地 TopK/TopK-dropout，控制 `topk`、`max_drop`、再平衡周期和单票权重。
- 加行业暴露约束：单行业权重上限、主题拥挤惩罚、同产业链过度集中惩罚。
- 加换手约束：每日/每周最大换手，低增益换仓不执行。
- 加风控过滤：ST、停牌、涨跌停不可交易、流动性过低、解禁临近、质押高风险。
- 加回撤控制：组合回撤超过阈值时降低仓位；市场情绪周期弱时降低 TopK 暴露。
- 依赖修复后再评估 Qlib `TopkDropoutStrategy` / `PortfolioOptimizer`，当前先不让 cvxpy/numpy 阻塞主链。

落地模块：

- `scripts/backtest_qlib_signal.py`
- `models/portfolio_policy.py`
- `signals/market_judge.py`
- `data/storage/lgb_backtest_latest.json`
- 后续：RQAlpha 或 Qlib executor 细规则验证。

关键指标：

- 成本后年化收益、Sharpe/IR、最大回撤。
- 平均换手、换手收益比。
- 行业暴露、单票集中度、Top holdings 稳定性。
- 相对基准超额收益和胜率。

失败动作：

- 换手高但收益不提升：提高 dropout/换仓门槛。
- 行业暴露贡献过高：降低行业上限或加行业中性版本。
- 回撤超过阈值：候选模型暂停 promotion，仓位建议降级。

### 5. 生产模型能灰度、回滚

目标：新模型先影子运行，证明稳定后再切生产；坏模型能自动回退。

应对策略：

- 模型 registry 记录所有候选：run id、feature set、label、训练窗口、rolling 指标、回测指标、artifact path。
- 生产模型、候选模型、影子模型分层：候选模型每天产预测但不影响推文，用于 live paper comparison。
- promotion gate：data health、prediction coverage、rolling metrics、backtest metrics、factor health、模型新鲜度全部通过才升级。
- rollback：生产模型异常、覆盖不足或连续质量退化时自动回滚到上一版 stable。
- OnlineManager 可作为后续模型热更新/灰度发布框架；短期先用本地 registry 实现核心语义。

落地模块：

- `models/model_registry.py`
- `scripts/promote_model.py`
- `scripts/check_model_quality.py`
- `data/storage/model_registry/`
- `data/storage/production_model.json`

关键指标：

- 生产模型 run id、数据日期、模型年龄。
- 生产 vs 影子模型的日度 IC、TopK overlap、模拟收益差。
- 连续退化天数。
- 回滚次数和原因。

失败动作：

- smoke 失败：不写生产 cache。
- evaluation/backtest 失败：模型留在 candidate。
- 连续 3 次 quality degraded：回滚上一版 stable，并在推文状态中标注。

### 6. 每天知道收益来自哪里

目标：每天回答“今天赚/亏是因为行业配置、个股选择、市场 beta、还是交易成本”。

应对策略：

- 接 Brinson 归因：拆行业配置收益、个股选择收益、交互项。
- 接因子归因：按基本面、资金流、北向、行业、事件、价量分组看贡献。
- 接交易归因：开仓收益、持仓收益、换仓成本、滑点/手续费。
- 推文和日报分开：推文只展示简洁状态，日报/周报保存完整归因表。
- 归因反哺因子治理：如果收益长期只来自行业 beta，而不是个股选择，要调整模型目标或加行业中性版本。

落地模块：

- `scripts/portfolio_attribution.py`
- Qlib `brinson_pa()`
- `data/storage/attribution/latest.json`
- `data/storage/attribution/history.csv`

关键指标：

- 行业配置贡献。
- 个股选择贡献。
- 交易成本贡献。
- 因子组贡献。
- TopK overlap 和换仓贡献。

失败动作：

- 收益长期来自单一行业：降低行业上限，跑行业中性对照。
- 交易成本吞噬收益：降低换手，提高持仓周期。
- 个股选择贡献持续为负：模型降级，回到更简单的基线或观察榜。

## Phase 1：Qlib Workflow/Recorder 正式接入

目标：把当前散落的 train/evaluate/backtest JSON 变成可追溯实验，同时吸收 cc 最新文档中“零依赖 Qlib 功能先用起来”的建议。

任务：

1. 新建 `scripts/run_qlib_workflow.py` 或 YAML config 生成器。
2. 每次训练用 `R.start(experiment_name=...)`。
3. 用 `SignalRecord` 保存 `pred.pkl` 和预测元数据。
4. 用 `SigAnaRecord` 或自写指标把 IC、RankIC、TopK spread 写入 Recorder。
5. 用 `PortAnaRecord` 能跑则接入；若仍被策略依赖卡住，继续用本地 backtest 输出 artifact。
6. 把 `ModelRegistry` 和 Qlib Recorder 对齐：registry 只保存“可生产候选”的摘要，Recorder 保存完整研究产物。
7. 在回测结果上接 `brinson_pa()`，拆解收益来自行业配置还是个股选择。
8. 为模型对照增加处理器实验：`CSZScoreNorm` vs `RobustZScoreNorm` vs `TanhProcess`，先在 research branch 跑，不能直接替换生产处理链。
9. 建立自定义表达式因子清单，优先做去大盘残差、异常成交量、资金流动量和跨资产相关性。

验收：

- 每个模型训练 run 都能查到参数、数据窗口、label、预测、评估、回测结果。
- `lgb_model.pkl` 不再是唯一事实来源；生产 cache 能指向具体 run id。
- 失败 run 不覆盖上一版生产模型。
- Brinson 归因能回答 TopK 收益中行业配置贡献和个股选择贡献各占多少。
- 处理器/表达式实验只在同一模型和同一数据窗口下比较，避免同时改特征、模型和标签导致无法归因。

## Phase 2：基本面/宏观 + 日频资金流/北向/板块因子研究

目标：把 cc 新增的基本面估值/质量、宏观、资金行为因子变成可回测 research features。基本面估值不是 Phase 3 以后才做，它和资金流一样是当前最大信息缺口之一。

任务：

1. 新建 `data/collectors/fundamental.py`：
   - 估值：PE/PB/PS/股息率/总市值/流通市值，接口需版本兼容。
   - 质量：ROE/ROA/毛利率/净利率/资产负债率/现金流质量。
   - 成长：营收增速/利润增速/利润加速度。
2. 新建 `data/collectors/macro_factors.py`：
   - 利率、国债收益率、汇率、大宗商品、PMI/CPI/M2 等全市场 broadcast 因子。
3. 新建 `data/collectors/fund_flow.py`：
   - `ak.stock_individual_fund_flow`
   - `ak.stock_sector_fund_flow_rank`
   - `ak.stock_hsgt_individual_em`
   - `ak.stock_hsgt_hold_stock_em`
4. 新建 `factors/fundamental.py`：
   - `pe_ttm_zscore`
   - `pb_industry_pctile`
   - `ps_ttm_zscore`
   - `roe_ttm`
   - `gross_margin`
   - `debt_ratio`
   - `cashflow_quality`
   - `market_cap_pctile`
5. 新建 `factors/institutional.py`：
   - `main_flow_zscore`
   - `sell_pressure_zscore`
   - `northbound_holding_delta_10d`
   - `sector_flow_rank`
6. 新建 `models/feature_merger.py`，按 `(date, instrument)` 合并 Alpha158、基本面、宏观、资金流、行业/板块因子。第一版先在模型输入前外部 merge，不改 Qlib 内部 bin；验证稳定后再考虑写 Qlib 自定义 feature bin。
7. 新建 `data/storage/factor_impacts/`，先用 parquet/csv 做研究落库。
8. 写 `scripts/evaluate_factor_ic.py`，单因子和增量模型都要跑：
   - 单因子 IC/RankIC
   - 分位收益
   - 衰减 1/5/10/20 日
   - 换手和覆盖率
9. 把有效因子接入 `train_model_suite.py` 的可选 feature branch，不直接污染生产 Alpha158。
10. 只有通过研究门槛后，才把基本面/资金流字段写入 Qlib 自定义 feature bin 或 DatasetH 自定义 loader。cc 提议的“Day 3 写入 Qlib bin、Day 5 更新生产模型”可以作为最快路径，但必须插入 eval/backtest/rolling gate。

验收：

- 至少 6-12 个月 rolling 窗口。
- 加入基本面估值/质量、资金流/北向因子后，RankIC、Top20 spread 或回测 Sharpe 至少一项稳定优于 Alpha158/XGB 基线。
- 基本面、股东和财报类因子必须使用披露日/公告日后的 `effective_date`，不能用报告期末日期直接 forward-fill。
- 资金流因子有容量和延迟说明，不能把研报 Sharpe 当生产承诺。
- 即使短窗 IC 达到 `0.04+`，也不能直接更新生产模型；还要通过成本、换手、最大回撤和连续 rolling split 稳定性检查。

## Phase 3：基金重仓和舆情事件慢因子

目标：把中长期机构偏好、主题确认和新闻事件结构化。

任务：

1. 新建 `data/collectors/fund_holdings.py`，采集基金净值、绩优基金池、季度 Top10 持仓、披露日期。
2. 新建 `factors/fund_holdings.py`：
   - `fund_star_weight`
   - `fund_holding_delta`
   - `fund_consensus_count`
   - `fund_crowding_risk`
   - `fund_theme_exposure`
3. 新建 `data/storage/event_impacts/`，统一新闻、公告、股吧、地缘政策事件。
4. 把板块热度和基金主题暴露合并成 `sector_theme_score`。
5. 长线栏目在真正模型前继续叫“观察榜”，不能暗示长期价值判断。

验收：

- 单元测试证明任意日期只使用当时已披露/已发布的信息。
- 基金因子主要服务 20-60 日和 3-24 月模型；短线模型只允许作为背景确认或拥挤度惩罚。
- 输出解释不能写“当前基金持仓”，只能写“最近已披露季报显示”。

## Phase 4：Alpha360、DL 和模型对照

目标：用 Qlib 未用上的特征和模型扩展预测能力，但全部走同一评估门槛。

任务：

1. 在 `train_model_suite.py` 加 handler 参数：
   - `Alpha158`
   - `Alpha360`
   - `Alpha158 + factor_impacts`
   - `Alpha360 + factor_impacts`
2. 模型对照：
   - LGB
   - XGB
   - CatBoost
   - DoubleEnsemble
   - GRU / ALSTM
   - Transformer / Localformer
   - HIST / TabNet 作为研究候选
3. 深度模型用本地 MPS wrapper 或 Qlib CPU baseline，避免卡在 Qlib GPU 参数不支持 MPS。
4. Ensemble 必须做截面 rank/z-score 后融合，不能 raw prediction 直接平均。

验收：

- 每个模型都有同格式 `pred`、`label`、`metrics`、`backtest`。
- 新模型的 RankIC 或成本后回测 IR 必须显著优于当前生产模型，否则只能留在研究区。
- Alpha360 内存占用和运行时间必须记录，避免 cron 超时。

## Phase 5：RollingGen、OnlineManager 和生产治理

目标：把“训练一次模型”升级为“持续模型治理”。

任务：

1. 用 Qlib `RollingGen` 替代或补强当前自写 `rolling_train.py`。
2. 每个 rolling split 生成独立 Recorder run。
3. 用 `RollingEnsemble` 或本地 ensemble 汇总 OOS 预测。
4. 研究 `OnlineManager`：
   - 生产模型注册
   - 灰度候选模型
   - 自动更新预测
   - 回滚上一版
5. after-close pipeline 增加 promotion gate：
   - data health
   - smoke coverage
   - eval quality
   - backtest drawdown
   - registry promotion

验收：

- 生产模型只来自通过 promotion gate 的 run。
- 连续 3 次质量退化，自动保留上一版模型并标记 `model_quality_degraded`。
- 推文/报告能显示模型版本、数据日期、预测覆盖、质量状态。

## 暂缓项

| Qlib 功能 | 暂缓原因 | 重新评估条件 |
|---|---|---|
| `TopkDropoutStrategy` / `PortfolioOptimizer` | 当前 cvxpy/numpy import 链失败 | 新建隔离 env 或解决依赖冲突后再切换 |
| Qlib report 可视化 | 当前缺 `statsmodels`，且核心指标已由 JSON 输出覆盖 | 安装 `statsmodels` 后作为报告层接入，不作为 gate 前置依赖 |
| Qlib RL / Order Execution | 当前日频 alpha、成本、回测还没稳定 | 组合回测稳定后，用 RL 做仓位/执行辅助 |
| HighFreqHandler / 1min 数据 | 数据成本、存储和执行复杂度高 | 先证明日频策略稳定盈利 |
| Meta-learning / DoubleAdapt | 概念漂移有价值，但治理复杂 | Rolling/Recorder/Registry 稳定后再做 |
| PIT 财报数据库 | 长线模型必需，但数据准备重 | 基金/资金流/板块因子跑通后接 |

## 下一版交付清单

P0：

- `scripts/run_qlib_workflow.py`
- Recorder + SignalRecord + SigAnaRecord 接入
- model run id 写入 `lgb_latest_predictions.json`
- `scripts/evaluate_factor_ic.py`
- Brinson 归因接入回测输出
- 处理器对照实验：`CSZScoreNorm` / `RobustZScoreNorm` / `TanhProcess`

P1：

- `data/collectors/fundamental.py`
- `data/collectors/macro_factors.py`
- `data/collectors/fund_flow.py`
- `factors/fundamental.py`
- `factors/institutional.py`
- `models/feature_merger.py`
- `data/storage/factor_impacts/`
- `train_model_suite.py --features factor_impacts`

P2：

- `data/collectors/fund_holdings.py`
- `factors/fund_holdings.py`
- `data/storage/event_impacts/`
- Alpha360 model suite

P3：

- RollingGen workflow
- OnlineManager/registry promotion gate
- 隔离环境验证 Qlib TopkDropoutStrategy/PortfolioOptimizer

## 与 cc 文档的关系

采纳：

- cc 对资金流/北向资金和基金季报重仓的区分是正确的。日频资金行为因子应升到 P1 研究；基金重仓仍是 P2 慢因子。
- cc 对 Qlib 高级能力的方向判断大体正确：Recorder、Rolling、Alpha360、模型 zoo、TopK/组合验证都应该进入路线。
- cc 最新的 `cc-next-phase-roadmap.md` 提醒了 Brinson 归因、RobustZScoreNorm/TanhProcess 和自定义表达式算子，这些是低成本高价值项，应前移到 Phase 1。
- cc 最新的 `cc-factor-gap-analysis.md` 指出基本面估值/质量因子是低垂果实，这个分歧应采纳：PE/PB/PS/ROE/现金流质量应和资金流/北向同步进入 Phase 2，而不是排到基金重仓和舆情事件之后。

保留门槛：

- cc 引用的研报 Sharpe 只能作为优先级线索，不能直接写入生产收益预期。
- cc 引用的 AKShare API 名称需要按本地版本校验；`stock_a_indicator_lg` 当前不可用，不能写死为唯一实现。
- 官方 Qlib 的 TopkDropoutStrategy 是正确方向，但当前本地环境不能直接导入；下一版仍先用本地 TopK 回测，依赖修复后再替换。
- 深度模型可以并行研究，但不能绕过同一套 evaluation/backtest/registry gate。
- cc 的“3 天资金流接入、Day 5 更新生产模型”节奏过快。可以 3 天做完 research prototype，但生产 promotion 必须等 rolling、成本、换手、回撤和 registry gate 通过。

## 推荐执行顺序

1. 先接 Recorder、Brinson 归因、处理器对照和 `evaluate_factor_ic.py`，把研究结果可追溯。
2. 立刻做基本面估值/质量、宏观、日频资金流/北向因子，因为它们共同补上当前最大的有效因子宽度缺口。
3. 同步把基金重仓和事件影响层做成慢因子，不急着进生产。
4. 然后跑 Alpha360、自定义表达式和模型 zoo，对照是否真的提升。
5. 最后用 RollingGen/OnlineManager 把通过门槛的模型纳入生产治理。
