# 当前项目与百亿量化私募的差距调研及追赶路线图

日期：2026-05-12  
作者：cx  
范围：`stockPrediction` 当前代码、数据产物、已有 cc/cx 计划文档、Qlib 官方能力、公开可查的头部量化机构与监管资料。

## 一句话结论

当前项目已经从“脚本级股票预测”进入“有 Qlib 数据、有 LGB/XGB/深度模型、有初版 rolling/backtest/归因/生产管线”的研究原型阶段，但距离百亿量化私募的核心差距仍然很大。

最大差距不是单个模型。XGB/LGB/Transformer 这些模型本身并不稀缺；真正差距在：

1. **点时安全的数据与因子工厂**：机构有大量低相关、可追溯、可监控衰减的因子；当前项目的新增因子还没有完整 point-in-time 对齐，部分 `FeatureMerger` 逻辑存在把“最新值”广播到历史日期的未来函数风险。
2. **组合与交易闭环**：机构优化的是成本后、约束后、容量可承载的组合；当前项目主要还是信号分数和简化 TopK 回测。
3. **风险模型与归因体系**：机构每天知道收益来自行业、风格、个股、交易成本、流动性还是模型漂移；当前只有简化行业归因，且行业分类还用板块/代码段近似。
4. **研究平台与生产治理**：机构能并行跑成百上千个实验、因子、参数、市场状态切片，并通过灰度、回滚、审计上线；当前项目有雏形，但还没有 Qlib Recorder/MLflow 级实验闭环和严格 promotion gate。
5. **执行与合规能力**：百亿机构面对程序化交易监管、申撤单速率、交易所监控、券商柜台、风控限额；当前项目还没有真实交易执行、委托模拟、合规报送和订单级风控。

追赶路线不应是“马上换更高级模型”，而是先补齐 **PIT 数据层 -> 因子工厂 -> rolling/组合评估 -> 交易风控 -> 生产治理**。只有这些完成后，深度模型和强化学习才有可靠土壤。

## 证据边界

百亿私募的内部因子、交易系统和真实绩效不是公开资料能完整证明的。本报告对“百亿私募能力”的判断分三类：

- **公开事实**：来自机构官网、交易所/监管公开资料、Qlib 官方文档。
- **行业合理推断**：由公开事实、招聘/技术路线、量化业务常识推断，但不把它写成确定内部细节。
- **本项目证据**：直接来自本仓库文件、日志和 `data/storage/*.json` 产物。

参考来源：

- Qlib 官方 Workflow 文档说明一次工作流覆盖 Data、Model、Evaluation、Backtest，并能追踪训练、推理、评估产物：<https://qlib.readthedocs.io/en/stable/component/workflow.html>
- Qlib Recorder 文档说明 `Recorder`、`SignalRecord`、`SigAnaRecord`、`PortAnaRecord` 可用于实验记录、预测、IC 分析和回测记录：<https://qlib.readthedocs.io/en/stable/component/recorder.html>
- 幻方量化官方英文页公开提到使用海量数据、神经网络、NLP、AI 深度学习平台，并把基本面、另类数据、市场数据纳入专有模型：<https://www.high-flyer.cn/en/fund/>
- 上交所英文公开资料说明程序化交易监管中对高频交易有每秒 300 笔、单日 20000 笔申报/撤单等重点监控口径：<https://english.sse.com.cn/news/newsrelease/voice/c/c_20250710_10784499.shtml>
- 上交所 2024 年公开资料也提到交易所对异常程序化交易标准进行测试和征求意见：<https://english.sse.com.cn/news/newsrelease/voice/c/c_20240611_10758618.shtml>
- 《金融研究》关于中国量化对冲基金的论文摘要指出量化基金具备算法交易、高频策略、复杂做空与杠杆设计等更强技术能力：<https://www.jryj.org.cn/EN/abstract/abstract1518.shtml>
- SCMP 2026 年报道披露 2025 年中国大型私募排名中量化机构表现突出，可作为“头部量化仍是强竞争者”的公开市场背景，而非内部能力证明：<https://www.scmp.com/tech/tech-trends/article/3339633/deepseek-founders-high-flyer-ranks-among-chinas-top-hedge-fund-firms-2025>

## 本项目当前真实状态

### 已经具备的能力

从仓库文件和数据产物看，项目已经不只是 toy model：

- 数据层：
  - Qlib 全 A 数据目录已经存在，`update_qlib_data.py` 有 staging、health gate、promotion 逻辑。
  - `fund_flow_history.parquet`、`northbound_history.parquet`、`fundamental_valuation.parquet`、`macro_features.parquet` 已经出现，说明资金流、北向、估值、宏观数据正在接入。
  - `scripts/fetch_fund_flow_history.py`、`scripts/fetch_fundamental_valuation.py` 已经是数据采集脚本雏形。
- 模型层：
  - `train_lgb.py` 是主生产 LGB。
  - `train_model_suite.py` 覆盖 LGB/XGB/CatBoost/DoubleEnsemble/ALSTM/Transformer 的模型对照框架。
  - `models/deep_models.py` 和 `data/storage/alstm_model.pt`、`transformer_model.pt` 表明深度模型已经能训练出结果。
- 评估层：
  - `evaluate_lgb_test.py` 计算 IC、RankIC、Top20 spread。
  - `rolling_train.py` 做了初版 walk-forward。
  - `evaluate_factor_ic.py`、`train_factor_ablation.py` 做了单因子 IC 和 ablation 雏形。
- 组合与归因：
  - `backtest_qlib_signal.py` 做简化 TopK + dropout + 成本回测。
  - `attribution.py` 做简化 Brinson 风格归因。
- 生产治理：
  - `after_close_pipeline.py` 串起 data update -> health -> train -> smoke -> evaluate -> attribution -> decay -> promotion。
  - `promote_model.py` 和 `models/model_registry.py` 是初版模型 registry / promotion / rollback。
  - `monitor_factor_decay.py` 是初版信号衰减监控。

### 当前结果的亮点

从已有结果看，主模型不是完全没信号：

- `data/storage/lgb_eval_latest.json`：2026-04-12 至 2026-05-12，LGB 测试样本 67576，IC 0.0524，RankIC 0.0302，Top20-Bot20 spread 8.28%，质量标记 normal。
- `data/storage/lgb_rolling_results.json`：6 个 rolling split，平均 IC 0.0673，RankIC 0.0693，Top20 spread 5.45%，RankIC split 正值比例 100%。
- `data/storage/lgb_backtest_latest.json`：近 60 天简化回测 7 个调仓周期，总收益 26.17%，Sharpe 5.357，但样本太短，不能视为稳定生产绩效。
- `data/storage/lgb_attribution_latest.json`：简化归因显示超额主要来自“个股选择”，但行业分类粗糙，结论只能作初步参考。

### 不能过度乐观的地方

这些结果还不能直接对标百亿私募：

- rolling 只有 6 个 split，测试区间集中在 2026 年初至 5 月，市场状态覆盖不足。
- 回测只有 7 个调仓周期，且用 forward label 近似收益，没有真实成交约束、停牌、涨跌停不可买卖、滑点、盘口冲击。
- `xgb_enhanced_results.json` 显示 171 维增强 XGB 的 IC 为 -0.0036、Top20 spread 为 -0.44%，说明“加因子”并不会自动变强。
- `factor_ic_test.json` 是旧版本评估产物，里面部分 STRONG 标记与后续修复后的严格标准不一致，应重跑后再引用。
- `FeatureMerger` 当前对资金流、估值、宏观等多类补充特征采用“最新值广播到全部训练日期”的简化方式，这在训练中会引入未来信息，不能直接用于正式实验。

## 与百亿量化私募的核心差距

### 1. 数据差距：从“能拉数据”到“点时安全的数据资产”

百亿机构的关键不是有 OHLCV，而是有可追溯、可复现、点时安全的数据资产。公开资料显示头部 AI 量化会使用基本面、另类数据、市场数据，并投入自研深度学习平台。这里不能推断其具体因子，但可以确定它不是单一价量数据。

当前项目问题：

- OHLCV 主链逐步稳定，但过去日志出现过最新交易日覆盖率不足、写入失败、Qlib health gate 拒绝 promotion。
- 资金流和北向数据正在拉取，但 StockToday/AKShare 都有空响应、重试失败、限流和断点续跑问题。
- 基本面估值数据有 baostock 接口，但财报指标、披露日、修正公告、业绩预告还没有完整 point-in-time 版本。
- `FeatureMerger` 的最新值广播会把 2026 年最新资金流/估值信息塞回 2021 年训练样本，这是机构级研究绝对不能接受的。
- 没有统一 raw -> normalized -> PIT feature store -> training view 的数据契约。

追赶要求：

- 所有非价量数据必须有 `source_timestamp`、`publish_date`、`effective_date`、`asof_date`。
- 训练集只允许使用 `effective_date <= sample_date` 的数据。
- 外部数据先入 raw cache，不直接进入模型；清洗后入 staging；通过 coverage、异常值、时序单调、重复键校验后才进入 feature store。
- 用交易日历统一所有 forward-fill，不能用自然日随意填充。

### 2. 因子差距：从 Alpha158 到真正的因子工厂

当前项目主要依赖 Alpha158 价量因子。新增资金流、估值、宏观处于采集和初步合并阶段。百亿私募的优势通常来自大量低相关信息源和持续的因子迭代，而不是“某一个神奇模型”。

当前项目缺口：

- 估值：已有 PE/PB/PS/PCF，但尚未做行业中性、历史分位、截面 winsor/zscore、PIT 对齐。
- 质量：ROE、ROA、毛利率、现金流质量、杠杆、应计利润还没有稳定进入模型。
- 成长：营收/利润增长、盈利预期变化、业绩预告 surprise 还缺。
- 资金流：有 net_mf_amount 等原始数据，但缺规模标准化、成交额归一、滚动 z-score、买卖盘结构、主力/散户分解的研究闭环。
- 北向/机构：北向持仓变化、持股占流通盘比例、连续净买入、基金季报重仓与拥挤度还未系统化。
- 行业/主题：没有稳定申万/中信行业映射、行业中性因子、主题热度、板块资金轮动。
- 事件：公告、财报、解禁、回购、质押、龙虎榜、涨停板、监管处罚、舆情事件没有统一事件表。
- 因子元数据：没有每个因子的定义、版本、数据源、延迟、覆盖率、IC 半衰期、禁用状态。

追赶要求：

- 建 `factor_registry`，每个因子必须有 owner、公式、数据源、频率、延迟规则、覆盖率阈值、上线状态。
- 建 `factor_store`，以 `(date, instrument, factor_name)` 或宽表分区保存 PIT 特征。
- 每日跑因子健康：coverage、NaN/inf、分位、截面均值方差、行业暴露、RankIC、TopK spread、换手。
- 因子上线必须通过：单因子 IC、ablation、shuffled negative control、rolling split、组合回测、容量/换手检查。

### 3. 模型差距：不是没有模型，而是缺标准化模型研究流程

项目已经能训练 LGB、XGB、CatBoost、ALSTM、Transformer。模型种类不是最大短板。

当前问题：

- 模型对照结果还不完整，`model_suite_results.json` 只保存了 ALSTM/Transformer 最新结果，树模型对照未形成完整统一报告。
- 深度模型训练样本和特征仍以 Alpha158 为主，没有利用高质量新因子。
- `train_model_suite.py` 还不是 Qlib Recorder/qrun 标准工作流，实验记录、artifact、参数、数据版本不可完全追溯。
- 没有稳定的超参搜索、特征集版本对照、市场状态分层评估。
- 还没有集成 Qlib 的 `SignalRecord`、`SigAnaRecord`、`PortAnaRecord`，而这些正是 Qlib 官方用于预测、IC、回测产物管理的标准模块。

追赶要求：

- 所有模型训练统一成 `workflow_config.yaml` 或 `run_qlib_workflow.py`，接入 Qlib Recorder/MLflow。
- 每次实验记录：数据版本、feature set 版本、label、universe、训练窗口、处理器、模型参数、代码 git hash。
- 建模型排行榜，但 promotion 只看 rolling OOS + 成本后组合指标，不看单次 IC。
- 深度模型只在 PIT 多源特征稳定后再加码；否则复杂模型只是在拟合噪声。

### 4. 回测差距：从“信号收益”到“可交易组合收益”

当前 `backtest_qlib_signal.py` 是必要雏形，但还不是机构级回测。

关键缺口：

- 用 label 近似持有期收益，缺订单级成交、T+1、涨停买不进、跌停卖不出、停牌、复牌、除权除息处理。
- 成本只有固定 open/close cost，没有滑点、冲击成本、成交量参与率、盘口深度。
- 没有持仓约束：行业上限、风格暴露、单票权重、流动性、黑名单、涨跌停过滤。
- 没有容量评估：Top20 等权在小资金下可能有效，但百亿规模最关键是容量和冲击。
- 没有真实基准对齐：中证500/1000/全A/行业中性多基准比较。

追赶要求：

- 短期：把现有 TopK 回测升级为交易日级、可复现、可配置的 portfolio simulator。
- 中期：接入 Qlib `PortAnaRecord` 或 RQAlpha/自研 A 股回测引擎。
- 长期：加入订单簿/分钟级数据、成交量参与率模型、冲击成本模型、券商交易规则。

### 5. 风险模型差距：百亿机构赚的是风险调整后的钱

机构不是只买分数最高股票，而是控制暴露后拿 alpha。

当前缺口：

- 没有 Barra/CNE 风格风险模型：市值、价值、成长、动量、波动、流动性、杠杆、盈利质量等风格暴露。
- 行业映射粗糙，`attribution.py` 用代码段近似行业，不能支撑真实行业风险控制。
- 没有组合优化目标：最大化预期收益 - 风险 - 成本 - 换手。
- 没有实时风险限额：单票、行业、风格、换手、回撤、流动性、黑名单。
- 没有拥挤交易和策略相关性监控。

追赶要求：

- 建风险模型 v1：行业哑变量 + 市值 + beta + 波动 + 动量 + 流动性 + 估值。
- 每日输出组合暴露：行业、风格、个股集中度、换手、预估成本。
- 组合构建从 TopK 升级到 constrained optimizer；如果 cvxpy/numpy 冲突，先用 scipy/pandas 实现简化约束优化。
- 回测报告必须同时输出 raw alpha、行业中性 alpha、风格中性 alpha、成本后 alpha。

### 6. 交易执行差距：没有真实交易系统，就谈不上百亿规模

当前项目主要是预测和推送，没有券商交易接口、订单管理和执行算法。

百亿机构必须处理：

- 程序化交易报备、交易所重点监控、高频标准、异常交易行为。
- 券商柜台、行情延迟、订单拆分、撤单控制、成交回报、风控前置。
- 日内执行算法：TWAP/VWAP/POV、涨跌停处理、流动性过滤。
- 交易后分析：滑点、冲击、撤单率、成交率、未成交损失。

公开监管资料显示，A 股对程序化交易和高频交易有明确监控口径；这意味着追赶不是简单“提高下单速度”，而是要有合规、风控和执行质量体系。

追赶要求：

- 先做 paper trading OMS：目标仓位 -> 订单 -> 模拟成交 -> 持仓 -> 风控检查。
- 再接券商模拟/实盘接口：只允许小资金灰度。
- 建 compliance log：策略版本、信号、人工确认、订单、撤单、成交、风控触发都要可审计。

### 7. 生产治理差距：从脚本串行到机构级 MLOps

项目已有 `after_close_pipeline.py`，方向正确，但还不够。

缺口：

- pipeline 只是 subprocess 串行，缺 DAG 调度、重试策略、数据/模型 artifact 版本。
- registry 是 JSON 文件，不能完整追踪实验 lineage。
- promotion gate 主要看最近 IC/质量，没有强制检查 rolling、factor health、backtest、风险暴露、数据版本。
- 没有 shadow model 的 live paper comparison。
- 没有系统级监控：任务耗时、失败率、数据延迟、模型漂移、预测覆盖、组合异常。

追赶要求：

- 接 Qlib Recorder/MLflow：每次训练都有 run id 和 artifact。
- 建 production manifest：当前生产数据版本、模型版本、feature set 版本、回测报告、风险报告。
- 候选模型先 shadow 运行 2-4 周，和生产模型每日同场比较。
- rollback 不只看 IC，还看覆盖率、回撤、换手、异常交易风险。

## 差距评分

| 模块 | 当前成熟度 | 机构级目标 | 主要差距 |
|---|---:|---:|---|
| 日线数据主链 | 55/100 | 95/100 | 覆盖和稳定性已有 gate，但数据源仍不稳 |
| PIT 多源数据 | 20/100 | 95/100 | effective_date/asof_date 缺失，最新值广播有泄漏 |
| 因子工厂 | 25/100 | 95/100 | 有雏形，无 registry、衰减、分层上线流程 |
| 模型训练 | 45/100 | 90/100 | 模型种类够，但实验追踪和标准对照不足 |
| Rolling 验证 | 35/100 | 90/100 | 初版 rolling 太短，缺市场状态和生产 gate |
| 回测系统 | 25/100 | 95/100 | 简化 TopK，不是订单/成交级 |
| 风险模型 | 15/100 | 95/100 | 行业/风格/容量/成本控制不足 |
| 交易执行 | 5/100 | 95/100 | 无 OMS、无券商接口、无合规审计 |
| 生产治理 | 35/100 | 90/100 | 有 pipeline/registry 雏形，缺 artifact lineage |
| 算力平台 | 20/100 | 90/100 | 单机为主，无分布式因子/训练/搜索平台 |

## 追赶路线图

### Phase 0：先堵未来函数和数据污染

目标：让任何实验都能被相信。

任务：

1. 重构 `FeatureMerger`：禁止最新值广播到历史样本，所有补充因子按 `date <= sample_date` 做 asof join。
2. 为 `fund_flow_history.parquet`、`northbound_history.parquet`、`fundamental_valuation.parquet` 加数据字典和 `effective_date` 规则。
3. 每个数据文件增加 health report：覆盖率、最新日期、重复键、NaN/inf、极值、source。
4. 重跑 `evaluate_factor_ic.py` 和 `train_factor_ablation.py`，用最新修复后的 verdict 标准覆盖旧结果。
5. 将当前所有“增强因子模型”标记为 research-only，未通过 PIT 审计前不进入生产。

验收：

- 任何训练样本的补充因子都能追溯到当日或之前可获得的数据。
- factor audit 对随机样本能证明没有未来数据。
- 旧的 `STRONG` 因子结论全部重算。

### Phase 1：建立机构级因子工厂 v1

目标：把价量 158 维扩展为点时安全的 250-300 维有效候选特征。

优先因子族：

1. 估值：PE/PB/PS/PCF、EP/BP/SP、历史分位、行业相对估值。
2. 质量：ROE/ROA/毛利率/净利率/现金流质量/负债率。
3. 资金流：主力净流入/成交额、5/20 日流入 z-score、买卖盘结构、资金流反转。
4. 北向：持股比例变化、连续净买入、持仓拥挤、北向相对行业偏好。
5. 行业/主题：申万行业、行业动量、行业资金、行业估值分位。
6. 事件：解禁、回购、质押、涨停板、龙虎榜、财报披露日。

工程任务：

- 新建 `data/storage/factor_store/`，按日期分区保存训练宽表。
- 新建 `data/storage/factor_registry/index.json`。
- 新建 `scripts/build_factor_store.py` 和 `scripts/check_factor_store_health.py`。
- 每日输出 `factor_health_latest.json`。

验收：

- 每个因子族 coverage >= 80% 或有明确降级规则。
- 每个因子有 RankIC、TopK spread、coverage、turnover、correlation 报告。
- 因子加入模型必须通过 ablation 与 negative control。

### Phase 2：统一 Qlib Workflow/Recorder 与 rolling 研究

目标：每次训练和实验都可复现、可对比、可回滚。

任务：

1. 接入 Qlib `R.start` / Recorder，保存模型、预测、IC、回测 artifact。
2. 用 `SignalRecord`、`SigAnaRecord`、`PortAnaRecord` 替代散落 JSON 的部分职责。
3. 重写 `train_model_suite.py` 为统一任务配置：LGB/XGB/CatBoost/DoubleEnsemble/ALSTM/Transformer 同一数据版本。
4. rolling 从 6 个短 split 扩展到 24-36 个 split，覆盖至少 3-5 年，按牛市、熊市、震荡、高波动切片。
5. promotion gate 改成 rolling + backtest + factor health + coverage + risk exposure 联合门槛。

验收：

- 任意模型能按 run id 找到训练数据版本、feature set、参数、预测、回测和归因。
- 新模型必须在多数 rolling split 优于生产模型，且最差 split 不显著恶化。

### Phase 3：组合优化和风险模型

目标：从“股票排序”升级为“可执行组合”。

任务：

1. 建行业映射表：申万/中信行业 + 概念板块。
2. 建风险暴露矩阵：行业、size、beta、momentum、value、volatility、liquidity、quality。
3. 回测加入：
   - 单票权重上限
   - 行业权重上限
   - 风格暴露上限
   - 日/周换手上限
   - 流动性过滤
   - 涨跌停/停牌过滤
4. 建交易成本模型：固定成本 + 滑点 + 冲击成本 + 参与率约束。
5. 输出组合归因：行业配置、风格暴露、个股选择、交易成本、现金拖累。

验收：

- 报告中同时有 raw signal、约束后组合、成本后组合。
- TopK 策略和优化组合能在 rolling 中比较。
- 每日推送不只给股票清单，还给组合风险摘要。

### Phase 4：paper trading、执行系统和合规日志

目标：可以小资金灰度，而不是只在 notebook/脚本里看分数。

任务：

1. 建 paper OMS：target position -> order -> simulated fill -> position -> pnl。
2. 加交易规则：
   - T+1
   - 涨跌停不可买卖
   - 停牌不可交易
   - 最小交易单位
   - 现金约束
3. 加执行算法模拟：TWAP/VWAP/POV。
4. 加合规日志：信号版本、人工确认、订单、撤单、成交、风控触发。
5. 接券商模拟盘，再小资金实盘灰度。

验收：

- paper trading 至少连续 1-2 个月稳定运行。
- 实盘前能解释每笔交易来自哪个模型、哪个信号、哪个组合约束。
- 有撤单率、成交率、滑点、冲击成本日报。

### Phase 5：算力和研究平台升级

目标：让因子研究和模型实验速度接近机构工作流。

任务：

- 本地单机缓存：Qlib expression cache、dataset cache、factor store parquet 分区。
- 并行因子计算：按日期/股票 shard，失败可恢复。
- 超参搜索：Optuna/Ray Tune，先小规模。
- 深度模型训练：MPS/GPU batch pipeline，明确训练成本和收益。
- 研究看板：因子、模型、组合、生产状态统一 dashboard。

验收：

- 一个新因子从接入到 IC/ablation/rolling 报告不超过 1 天。
- 一个新模型从配置到完整 rolling 报告不超过 1-2 天。

### Phase 6：强化学习策略控制器

目标：不让强化学习直接承担“预测哪只股票涨”的职责，而是让它学习组合层面的动态控制：仓位、换手、TopK 数量、风险降级和再平衡节奏。

定位：

- 监督学习模型负责 alpha：LGB/XGB/CatBoost/Transformer 输出股票排序和预期收益。
- 强化学习负责 policy：在给定信号、市场状态和当前持仓下，决定怎么交易、持多少仓、换多少仓。
- RL 必须建立在 PIT 因子、可交易回测环境和成本模型之后；否则很容易学到未来函数或历史噪声。

状态空间 v1：

| 状态 | 说明 |
|---|---|
| `market_regime` | 市场强弱、指数动量、波动率、成交额、涨跌停家数 |
| `signal_quality` | 当日 LGB/XGB 分数分布、TopK 分数差、模型近期 IC/RankIC |
| `portfolio_state` | 当前仓位、行业暴露、单票集中度、浮盈浮亏、回撤 |
| `risk_state` | 近 5/20 日波动、最大回撤、换手、流动性压力 |
| `factor_state` | 资金流、北向、估值、行业热度、舆情事件聚合状态 |

动作空间 v1：

| 动作 | 范围 |
|---|---|
| 总仓位 | 0%、25%、50%、75%、100% |
| TopK 数量 | 10、20、30、50 |
| 最大换手 | 0%、20%、40%、80%、100% |
| 风险模式 | normal / defensive / cash |
| 行业集中度上限 | 20%、30%、40% |

奖励函数 v1：

```text
reward =
  next_period_portfolio_return
  - transaction_cost
  - slippage_cost
  - turnover_penalty
  - drawdown_penalty
  - exposure_violation_penalty
```

第一版不要追求复杂算法，先把环境做可信：

1. 基于 `backtest_qlib_signal.py` 抽象 `TradingEnv`：
   - 输入：每日候选股票、预测分数、真实可交易收益、停牌/涨跌停/流动性标记。
   - 输出：组合净值、持仓、交易成本、回撤、换手、风险暴露。
2. 先做 baseline policy：
   - 固定 Top20。
   - 市场弱势降到 50% 仓位。
   - 回撤超过阈值降仓。
   - 换手上限约束。
3. 再训练 RL：
   - 算法优先 PPO 或 DQN，使用项目已有 `models/rl_agent.py` / `scripts/train_rl.py` 雏形。
   - 训练只使用历史训练窗口，验证和测试必须 walk-forward。
   - 不允许在测试窗口调参。
4. 做对照实验：
   - Fixed TopK baseline。
   - Rule-based risk policy。
   - RL policy。
   - Random/noisy policy 负控。
5. 只在 RL 同时满足收益、回撤、换手、稳定性四类指标后，才进入 paper trading。

验收门槛：

- RL 在 24+ 个 rolling split 中，成本后 Sharpe 或 Calmar 明显优于 rule-based policy。
- 最大回撤低于固定 TopK。
- 平均换手不高于固定 TopK，或换手增加能被收益覆盖。
- 最差 20% split 不显著劣于 baseline。
- 行业/风格暴露不突破硬约束。
- 连续 1-2 个月 paper trading 后仍优于 rule-based policy，才允许小仓灰度。

明确禁区：

- 不用 RL 直接预测个股收益。
- 不用短窗口单次回测结果证明 RL 有效。
- 不在有未来函数风险的特征上训练 RL。
- 不允许 RL 绕过风控硬约束。
- 不把训练环境收益当作实盘收益承诺。

## 最优先的 10 个具体行动

1. **修 PIT 泄漏**：重构 `FeatureMerger` 的资金流、估值、宏观、股东因子对齐方式。
2. **重跑因子评估**：用修复后的 `evaluate_factor_ic.py` 和 `train_factor_ablation.py` 重算所有候选因子。
3. **建 factor registry**：每个因子有定义、延迟、覆盖、状态和最近表现。
4. **建 factor store health**：每天输出多源因子覆盖率和异常报告。
5. **升级 rolling**：至少 24 个 split，3 年以上，多市场状态。
6. **接 Qlib Recorder**：训练、预测、IC、回测 artifact 全部进入 run。
7. **行业映射和风险暴露**：先做行业 + size + beta + volatility + liquidity。
8. **组合约束回测**：行业上限、换手上限、流动性过滤、涨跌停停牌过滤。
9. **shadow model**：候选模型连续 2-4 周 paper run，再考虑 promotion。
10. **paper OMS**：先不实盘，先把目标仓位、订单、模拟成交、持仓、PnL 跑通。

强化学习排在这 10 件事之后。等 PIT、组合回测、paper OMS 稳定后，再启动 Phase 6；否则 RL 很可能只是在更复杂地过拟合。

## 对“追上甚至超过”的现实判断

短期追上百亿私募的全套能力不现实。它们的优势包含多年数据积累、专有数据、研究团队、交易系统、券商资源、风控合规、算力平台和资金规模经验。

但项目可以在 3 个方向形成可行突破：

1. **小资金灵活性**：不需要承载百亿容量，可以做机构因为容量限制做不了的小盘/事件/拥挤反转策略。
2. **AI 解释层和事件层**：用 LLM 做公告、新闻、舆情、政策事件结构化，形成比普通 Alpha158 更低相关的信息源。
3. **研究纪律**：如果严格执行 PIT、negative control、rolling、成本后组合和灰度上线，小团队也可以避免大多数“看起来很强、实盘失效”的陷阱。

真正的目标不是复制百亿私募，而是建立一个机构级纪律的小型量化系统：

- 数据不污染。
- 因子有证据。
- 模型有 rolling。
- 组合有约束。
- 交易有成本。
- 生产能回滚。
- 每天知道收益和亏损来自哪里。

只要这条线跑通，再叠加资金流、北向、基本面、行业、事件和舆情等低相关因子，当前系统才有可能从“预测模型”升级为“可持续量化研究与交易平台”。
