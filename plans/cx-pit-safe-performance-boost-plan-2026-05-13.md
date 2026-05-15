# PIT-safe 模型大幅提升实施计划

日期：2026-05-13  
作者：cx  
目标：在不重新引入未来函数的前提下，把当前可信但偏弱的 PIT-safe 结果，提升到可研究上线、可滚动验证、可成本后评估的水平。

## 当前判断

当前最重要的事实不是“XGB 不行”，而是：

| 指标 | 旧 XGB 171维 | 旧 LGB 158维 | 本次 XGB 202维 |
|---|---:|---:|---:|
| IC | -0.004 | +0.072 | +0.014 |
| ICIR | - | 2.09 | 0.50 |
| RankIC | - | +0.041 | +0.037 |
| Top20 spread | -0.4% | +7.3% | +0.22% |
| PIT | 泄漏 | 泄漏 | 修复 |

所以后续不能追旧 LGB 的虚高结果。当前 202 维 XGB 是更可信的起点，但强度不够。真正的提升路线应是：

1. 先重建 PIT-safe 的 LGB/XGB/CatBoost 同口径基线。
2. 再用可审计的因子工厂扩展信息量。
3. 再用排序目标和 ensemble 提升 TopK 选择能力。
4. 最后用 rolling、成本、风险暴露和灰度机制决定是否上线。

一句话：结果要大幅提升，不能靠堆模型赌运气，要靠“PIT-safe 因子宽度 + 严格消融 + 排序优化 + 生产闭环”。

## 量化目标

| 阶段 | 目标口径 | RankIC | ICIR | Top20 spread | 说明 |
|---|---|---:|---:|---:|---|
| 当前可信起点 | PIT-safe XGB 202 | 0.037 | 0.50 | 0.22% | 可相信，但不够强 |
| 短期目标 | PIT-safe LGB/XGB/ensemble 202-280 | 0.045-0.055 | 0.8+ | 1.5%-2.5% | 可进入重点研究 |
| 中期目标 | 因子工程 + 排序模型 + rolling | 0.060+ | 1.0+ | 3.0%+ | 可考虑 paper shadow |
| 生产目标 | 成本后、约束后、滚动稳定 | 0.060+ | 1.0+ | 3.0%+ | 还要看换手、回撤、容量 |

所有目标必须在同一 universe、同一 label、同一 split、同一数据版本下比较。单窗口漂亮数字不算通过。

## Phase 0：可信基线重建

周期：1-2 天  
目标：先知道真实起点在哪里，避免拿泄漏结果做目标。

任务：

1. 重跑 PIT-safe LGB 158。
2. 重跑 PIT-safe LGB 202。
3. 重跑 PIT-safe XGB 202。
4. 增加 CatBoost 202 对照。
5. 所有结果落统一报告，包含：
   - 数据版本；
   - feature set 版本；
   - label 表达式；
   - train/valid/test 日期；
   - 样本数、股票数、交易日数；
   - IC、ICIR、RankIC、Top20 spread、coverage；
   - 是否使用 PIT-safe as-of merge。

验收标准：

- 旧 LGB 泄漏结果从候选结论中剔除，只保留为历史对照。
- 至少有一张同口径基线表，能够回答“到底是模型差，还是因子差”。
- 任一模型如果强于 XGB 202，必须能复现且没有数据泄漏。

优先判断：

- 如果 PIT-safe LGB 202 明显强于 XGB 202，短期以 LGB 为主模型。
- 如果 XGB/LGB/CatBoost 各有不同强项，进入 rank ensemble。
- 如果所有 202 维模型都弱，优先补因子，不优先调复杂模型。

## Phase 1：PIT-safe 因子工厂 v1

周期：2-5 天  
目标：从“有一些外部字段”升级为“可审计、可消融、可监控的因子包”。

优先因子包：

| 因子包 | 数据 | 预期用途 | PIT 要求 |
|---|---|---|---|
| valuation_daily | PE/PB/PS/PCF/股息率/市值 | 价值、估值分位 | trade_date 当日或滞后一日 |
| moneyflow_daily | 主力/超大/大/中/小单净流入 | 资金偏好、短期拥挤 | trade_date 当日或滞后一日 |
| northbound | 北向净买入、持股变化、连续买入 | 外资偏好 | 公布日后可用 |
| quality_fundamental | ROE/ROA/毛利率/现金流质量 | 基本面质量 | ann_date/publish_date as-of |
| shareholder_fund | 基金重仓、股东户数、机构持仓 | 机构偏好、拥挤度 | 披露日后可用 |
| industry_theme | 申万/中信行业、概念、板块资金 | 行业轮动、主题热度 | 分类版本和生效日 |

每个因子包必须有数据字典：

- 原始数据源；
- 原始字段名；
- 生成公式；
- 频率；
- 可用时间规则；
- 缺失处理规则；
- 覆盖率阈值；
- 是否允许训练；
- 最近一次健康检查结果。

验收标准：

- 不允许把最新财务、股东、基金持仓快照广播到历史日期。
- 每个新增因子都有 `effective_date <= sample_date` 的证明。
- 每日输出 coverage、NaN、inf、极值、重复键、最新日期报告。
- 覆盖率低于阈值的因子只能进入观察池，不能进正式训练。

## Phase 2：截面预处理和因子变换

周期：2-4 天  
目标：让新增因子以模型能稳定利用的形式进入训练，而不是原始值硬塞。

必做处理：

1. 非法值处理：`inf -> NaN`，极端 PE/PB/turn 做业务 guard。
2. 每日截面 winsorize：默认 1%-99% 或 3 sigma，两套都保留可选。
3. 每日截面 zscore：避免量纲差异压制树模型分裂。
4. 每日截面 rank percentile：重点给排序模型使用。
5. 缺失指示列：对财务、北向、基金持仓这类非全覆盖数据尤其重要。
6. 行业中性化：估值、质量、资金流至少做行业内 rank/zscore 对照。
7. 市值中性化：对容易被 size 主导的因子做残差化。
8. 时序衍生：5/20/60 日均值、变化率、波动率、历史分位、连续流入天数。

优先产物：

- `raw`：原始 PIT 字段，仅用于审计。
- `xs_rank`：截面 rank 特征，优先给排序模型。
- `xs_zscore`：截面标准化特征，优先给树模型。
- `industry_neutral`：行业中性特征，用于验证纯 alpha。
- `missing_flags`：缺失指示，避免把“没数据”误当成 0。

验收标准：

- 任一新因子进入模型前，都能看到原始分布、处理后分布和每日覆盖率。
- winsor 和 zscore 使用训练期规则，valid/test 不参与拟合任何全局参数。
- 同一个因子至少比较 raw、rank、zscore 三种版本，保留最稳版本。

## Phase 3：严格消融和负控

周期：3-5 天  
目标：证明新增因子真的有边际 alpha，而不是偶然、泄漏或重复 Alpha158。

实验矩阵：

| 实验 | 目的 |
|---|---|
| Alpha158 baseline | 价格量基线 |
| Alpha158 + valuation | 估值增量 |
| Alpha158 + moneyflow | 资金流增量 |
| Alpha158 + northbound | 北向增量 |
| Alpha158 + quality | 基本面质量增量 |
| Alpha158 + shareholder/fund | 机构偏好增量 |
| Alpha158 + industry/theme | 行业主题增量 |
| Alpha158 + all selected | 综合增量 |
| Alpha158 + shuffled within date | 负控 |
| Alpha158 + random factor | 管线泄漏检查 |

每组必须输出：

- IC；
- ICIR；
- RankIC；
- RankIC 正值比例；
- Top20-Bot20 spread；
- spread 正值比例；
- coverage；
- 日度 paired spread 差异；
- 相对 baseline 的残差 IC。

关键规则：

- 负控必须按 date 内 shuffle，保留每日分布和覆盖率，只打乱 instrument 对应关系。
- 如果 shuffled 因子也提升，说明管线有问题或结果是偶然。
- 如果单因子 IC 好但 RankIC/TopK spread 差，不进入主模型。
- 如果因子对 baseline residual 没有 IC，说明大概率只是重复已有价量信息。

通过门槛：

- rolling 窗口中至少 70% 的 RankIC 或 spread 为正；
- 加入因子包后 RankIC 或 Top20 spread 至少一项稳定超过 baseline；
- 成本后回测不恶化；
- coverage 不低于生产阈值，建议日均 80% 起步；
- 负控提升必须消失。

## Feature Set V2：566 维候选池

目标特征池可以按下面口径定义：

| 特征组 | 维度 | 含义 | 使用方式 |
|---|---:|---|---|
| Alpha158 | 158 | Qlib 经典价量表达式 | 表格模型主基线 |
| Alpha360 | 360 | 60 日价量序列展开特征 | 树模型可直接用，深度模型可 reshape 成序列 |
| 补充因子 | 48 | PIT-safe 资金流、估值、北向、质量、股东/基金、行业主题 | 必须先通过 PIT 审计和消融 |
| 合计 | 566 | V2 候选特征池 | 只作为候选池，不默认全量上线 |

这个 566 维方向值得做，但不能直接假设“维度越多越强”。Alpha158 和 Alpha360 都来自价量，信息会重叠；补充因子虽然更接近私募的非价量信息，但也最容易出现披露日、覆盖率、极值和未来函数问题。

因此实验顺序必须拆开：

1. `Alpha158`：158 维基线。
2. `Alpha158 + 补充因子`：206 维左右，验证非价量增量。
3. `Alpha360`：360 维序列价量基线。
4. `Alpha158 + Alpha360`：518 维，验证 Alpha360 是否提供额外时序信息。
5. `Alpha360 + 补充因子`：408 维左右，验证序列价量和非价量组合。
6. `Alpha158 + Alpha360 + 补充因子`：566 维完整候选池。

保留标准：

- 如果 566 维只提升训练集、不提升 rolling OOS，说明过拟合，不能上线。
- 如果 `Alpha158 + Alpha360` 不如单独 Alpha360 或 Alpha158，说明价量冗余过高，应做特征筛选或降权 ensemble。
- 如果 `+补充因子` 的提升主要来自某一低覆盖字段，先降级为观察因子。
- 最终上线不一定使用全部 566 维，可以是 300-450 个通过筛选的稳定特征。

## 叠加是否一定更好

不一定。`Alpha158 + Alpha360 + 补充因子` 叠在一起只代表信息更多，不代表 OOS 收益更好。它可能出现三种结果：

| 现象 | 解释 | 应对 |
|---|---|---|
| 训练集更好，valid/test 不好 | 过拟合 | 降维、加正则、减少 Alpha360 冗余列 |
| IC 更好，RankIC/TopK spread 不好 | 解释能力提升，但排序能力没提升 | 改用 rank loss、rank ensemble 或直接剔除 |
| 单模型不强，ensemble 变强 | 信息源互补但同一模型难以同时吸收 | 分模型训练后 rank 融合 |
| 206 强于 566 | 补充因子有效，Alpha360 冗余或噪声 | 主力用 206，Alpha360 只做辅助模型 |
| 360 强于 158/206 | 原始价量路径有效 | 重点做 Alpha360 序列模型 |
| 566 稳定强于所有子集 | 信息互补且模型能吸收 | 再筛到 300-450 维候选上线 |

更好的目标不是“维度最大”，而是“每一类信息都能贡献 OOS 排序能力”。如果叠加后没有提升，就说明信息冗余、噪声、覆盖率、尺度处理或模型容量之间有冲突。

## 详细迭代方案

### Iteration 0：冻结评估口径

目的：先让所有实验可比。

固定项：

- universe：`all`；
- label：当前 5 日 forward return；
- train/valid/test 切分；
- 交易日历；
- 评估指标：IC、ICIR、RankIC、RankIC 正值比例、Top20 spread、spread 正值比例、coverage；
- 结果保存格式；
- 随机种子；
- 特征集版本号。

验收：

- 同一脚本重复跑，核心指标波动在可接受范围内；
- 任一结果都能追溯到数据版本、特征版本和 split。

### Iteration 1：表格基线六档对照

目的：判断 566 维里哪一类信息真的有效。

训练矩阵：

| 版本 | 特征 | 模型 | 结论要回答 |
|---|---|---|---|
| FS-158 | Alpha158 | LGB/XGB/CatBoost | 当前价量摘要基线 |
| FS-206 | Alpha158 + 补充因子48 | LGB/XGB/CatBoost | 非价量因子是否有增量 |
| FS-360 | Alpha360 | LGB/XGB/CatBoost | 原始价量路径是否有增量 |
| FS-408 | Alpha360 + 补充因子48 | LGB/XGB/CatBoost | Alpha360 与非价量是否互补 |
| FS-518 | Alpha158 + Alpha360 | LGB/XGB/CatBoost | 两套价量是否互补还是冗余 |
| FS-566 | Alpha158 + Alpha360 + 补充因子48 | LGB/XGB/CatBoost | 全量候选池是否真能被表格模型吸收 |

决策：

- 如果 `FS-206 > FS-158`：补充因子进入主线。
- 如果 `FS-360 > FS-158`：Alpha360 进入主线。
- 如果 `FS-518 <= max(FS-158, FS-360)`：Alpha158/Alpha360 冗余，后续不强行拼。
- 如果 `FS-566 <= max(FS-206, FS-360, FS-408, FS-518)`：全量拼接不是最佳，进入筛选和 ensemble。
- 如果 `FS-566` 在 rolling 上稳定第一：继续筛特征，避免上线过拟合全量版本。

### Iteration 2：补充因子包消融

目的：找出真正带来增量的 48 个补充因子。

因子包：

- valuation；
- moneyflow；
- northbound；
- quality；
- shareholder/fund；
- industry/theme；
- macro。

每个包跑：

1. `base`；
2. `base + factor_group`；
3. `base + shuffled_factor_group`；
4. `base + selected_factor_group`。

保留规则：

- `base + factor_group` 的 rolling RankIC 或 Top20 spread 稳定提升；
- `shuffled_factor_group` 不提升；
- group 内至少有一部分特征对 baseline residual 有 RankIC；
- coverage 达标；
- 没有 PIT 疑点。

如果某个因子包只在单窗口提升，不进入主线，只进观察池。

### Iteration 3：特征筛选

目的：从 566 维候选池筛到 300-450 个稳定特征。

筛选顺序：

1. 数据质量过滤：PIT 不确定、coverage 低、inf 多、stale 严重、近常数的先删。
2. 单因子 rolling：保留 RankIC、TopK spread、coverage 稳定的因子。
3. 残差 IC：保留对 `label - baseline_pred` 仍有信息的补充因子。
4. 相关性聚类：`abs(corr) > 0.85/0.90` 的特征组只留 1-3 个代表。
5. 模型重要性稳定性：保留多个 rolling split 都有贡献的特征。
6. ablation 验证：删除特征后不降指标的，继续删除。

输出三套特征：

| 特征集 | 维度 | 用途 |
|---|---:|---|
| compact | 250-300 | 稳定生产候选，低过拟合 |
| main | 300-450 | 主力研究候选，平衡信息量和稳定性 |
| full | 566 | 研究对照，不直接上线 |

### Iteration 4：模型分工

目的：不同信息源用不同模型吸收，不强迫一个模型吃掉所有信息。

模型组合：

| 模型 | 输入 | 角色 |
|---|---|---|
| LGB-206 | Alpha158 + selected supplement | 稳定表格主力 |
| LGB/CatBoost-main | 300-450 selected features | 主力增强模型 |
| XGB-main | 300-450 selected features | 多样性来源 |
| LGBRanker-main | selected features | 直接优化排序 |
| Transformer/ALSTM-360 | Alpha360 reshape 序列 | 价格路径模型 |
| Fusion model | Alpha360 embedding + 补充因子 embedding | 高级研究候选 |

融合方式：

- 先做 daily rank ensemble，不直接平均 raw score；
- 权重由 rolling 表现决定；
- 单模型权重上限不超过 50%-60%，避免一类信息源失效拖垮组合；
- 如果深度模型不稳定，只保留为辅助信号，不进生产主权重。

### Iteration 5：结果更好的判定标准

只有满足下面条件，才算“结果真的更好”：

- rolling OOS RankIC 提升，而不是单次 test 提升；
- Top20 spread 成本后提升；
- spread 正值比例提升；
- 换手没有显著恶化；
- 行业/风格暴露可解释，不是押单一行业；
- 负控实验不提升；
- shadow 期连续优于生产模型。

如果 IC 提升但 TopK spread 不提升，不能认为更好。这个项目最终是选股和组合，不是解释收益率。

### Iteration 6：上线决策

上线优先级：

1. `compact` 特征 + LGB/rank ensemble：最稳，优先 shadow。
2. `main` 特征 + LGB/CatBoost/ranker ensemble：主力候选。
3. `full 566`：只有 rolling 和成本后都稳定第一，才考虑 shadow。
4. Alpha360 深度模型：先做研究和辅助 ensemble，不抢第一上线优先级。

失败处理：

- 如果 566 不如 206：回到补充因子主线。
- 如果 360 不如 158：Alpha360 暂时只保留深度模型研究。
- 如果补充因子没有 residual IC：继续补数据源，而不是调模型。
- 如果 rolling 分化严重：按市场状态建 regime ensemble，而不是硬推一个模型。

## Phase 4：模型升级路线

周期：3-7 天  
目标：在因子可信后，再用模型榨出排序能力。

优先顺序：

1. PIT-safe LGB：先跑 158、206、360、518、566 五档，因为 LGB 对表格因子通常稳定。
2. XGB：保留作模型多样性来源，重点对比 206 和 566 是否过拟合。
3. CatBoost：处理非线性和缺失结构，重点看 RankIC 和 TopK spread。
4. Rank ensemble：按每日预测 rank 做加权，不直接平均 raw score。
5. LGBMRanker/LambdaRank：直接优化排序目标，而不是只回归收益。
6. Two-stage residual model：第一阶段 Alpha158/Alpha360，第二阶段用资金流/基本面预测残差。
7. ALSTM/Transformer：优先吃 Alpha360 序列，再和补充因子做 late fusion。
8. 强化学习：短期不用于选股 alpha，后续只用于组合调仓、换手控制和执行策略。

为什么不先上深度模型：

- 当前瓶颈不是模型种类不够，而是 PIT-safe 有效因子不够。
- 单台 Mac Studio 可以训练中小深度模型，但无法靠算力弥补数据质量。
- 深度模型在短测试窗口上容易给出漂亮但不稳的结果，必须排在 rolling 和因子审计之后。

模型通过门槛：

- 不只看 IC，必须看 RankIC、TopK spread、成本后收益。
- ensemble 必须显著优于最强单模型，不能只是平均稀释。
- 任何模型进入 shadow 前，必须通过至少 24 个 rolling split。

## Phase 5：Rolling、成本和可交易验证

周期：1-2 周  
目标：把“预测强”转换成“可交易、成本后、风险可控”。

rolling 设计：

- 至少 24 个 split；
- 覆盖 3 年以上；
- 每个 split 保留完整 train/valid/test；
- 市场状态分层：上涨、下跌、震荡、高波动、低波动；
- 训练窗口比较：2 年、3 年、5 年、expanding；
- 测试窗口比较：1 个月、2 个月、3 个月。

回测设计：

- Top20、Top50、Top100 都跑；
- dropout 和换仓阈值都跑；
- 加交易成本、滑点、停牌、涨跌停过滤；
- 加流动性过滤，剔除成交额太小股票；
- 输出换手率、最大回撤、胜率、容量粗估。

组合约束：

- 单票权重上限；
- 行业权重偏离上限；
- 市值暴露限制；
- 波动率暴露限制；
- ST、停牌、涨跌停、退市风险过滤；
- 黑名单和财报异常过滤。

验收标准：

- rolling 平均 RankIC 达标，且不是少数 split 拉高；
- TopK spread 成本后仍为正；
- 换手不把收益吃掉；
- 行业/风格暴露不是收益唯一来源；
- 回撤在可接受范围内。

## Phase 6：生产提升闭环

周期：持续  
目标：把研究结果变成每天能稳定运行、能灰度、能回滚的系统能力。

每日固定报告：

1. 数据健康：最新交易日、覆盖率、缺失、异常值、延迟。
2. 因子健康：IC、RankIC、spread、coverage、衰减、分布漂移。
3. 模型健康：预测覆盖、score 分布、与昨日相关性、TopK 变化。
4. 组合健康：行业暴露、风格暴露、换手、预估成本、流动性。
5. 归因报告：收益来自行业配置、个股选择、风格暴露还是交易成本。
6. promotion gate：候选模型是否超过生产模型，是否允许 shadow。

上线规则：

- 新模型先 shadow 2-4 周。
- shadow 每天和生产模型同场比较。
- 通过 rolling、paper、风险、数据健康后再 promotion。
- promotion 后保留一键 rollback。
- 任何数据源异常时，模型不得自动升级。

## 近期执行清单

P0：

- 重跑 PIT-safe LGB 158。
- 重跑 PIT-safe LGB 202。
- 重跑 PIT-safe XGB 202。
- 增加 CatBoost 202 同口径报告。
- 产出 `baseline_pit_safe_comparison`。

P1：

- 完成 ST_CLIENT daily_basic/moneyflow 的追加式历史文件。
- 为 fundamental、shareholder、quality 加 publish/effective date 审计。
- 建 factor registry 草表。
- 每个因子包输出 coverage 和 PIT 证明。

P2：

- 给新增因子统一做 winsor/zscore/rank/missing flag。
- 实现按 date 内 shuffle 的负控。
- 跑 valuation、moneyflow、northbound、quality、fund/shareholder 的单包消融。
- 输出 residual IC。

P3：

- 训练 LGB/XGB/CatBoost 的 158、206、360、518、566 五档版本。
- 做 rank ensemble。
- 尝试 LGBMRanker。
- 只保留 rolling 稳定的模型。

P4：

- 扩展 rolling 到 24+ split。
- 增加成本后 TopK 回测。
- 加行业/风格暴露报告。
- 候选模型进入 shadow，不直接 promotion。

## 不做什么

1. 不把旧 LGB 泄漏结果当成目标。
2. 不因为单次 IC 提高就上线。
3. 不把低覆盖因子直接喂模型。
4. 不把最新财报、股东、基金持仓广播到历史样本。
5. 不先押注 Transformer/RL 来掩盖数据和验证问题。
6. 不用单窗口 Top20 spread 证明模型变强。

## 最终路线

后续要大幅提升，我会按这个顺序推进：

1. 用 PIT-safe 基线把真实水平钉死。
2. 用数据和因子工厂扩宽有效信息。
3. 用消融、负控、残差 IC 证明增量。
4. 用 LGB/XGB/CatBoost/ranker/ensemble 提升排序。
5. 用 rolling、成本、风险、归因筛掉纸面强但不可交易的模型。
6. 用 shadow、promotion、rollback 把研究成果变成稳定生产能力。

这条路慢一点，但它能把 RankIC 和 spread 的提升变成可信收益，而不是再制造一个漂亮但泄漏的历史结果。
