# cx 量化平台、舆情分析与量化大师经验深度调研

日期：2026-05-07

目的：为当前 `stockPrediction` 项目的 V2/V3 迭代选型。重点回答三件事：

1. 除 Qlib 之外，还有哪些比较强的量化开源库值得研究或接入。
2. 舆情分析应该用哪些库、模型、数据源，怎样和 14:30 / 22:00 / 09:20 的推送逻辑结合。
3. 量化领域值得长期学习的书、名人经验和工程原则有哪些。

结论先行：当前项目不应该立刻替换 Qlib。更好的路线是保留 Qlib 做 A 股因子和短线预测核心，同时补三层能力：`vectorbt/RQAlpha` 做快速回测验证，`FinBERT/FinGPT + 中文 NLP` 做事件修正层，`PyPortfolioOpt/风险预算` 做仓位和组合层。真正实盘接口或多资产高频执行以后再考虑 Lean、NautilusTrader、vn.py。

---

## 一、量化开源库全景

### 1.1 和 Qlib 同级或互补的研究平台

| 库/平台 | 定位 | 优点 | 风险/限制 | 对本项目建议 |
|---|---|---|---|---|
| Qlib | AI-oriented 量化投资平台，覆盖数据、模型训练、回测、组合优化、执行链路 | 已经接入；天然适合 Alpha158、LightGBM、深度模型、A 股日频预测；微软官方论文和 GitHub 都还活跃 | 数据格式和日历容易踩坑；实盘执行不是最强项 | 保留为短线模型和研究主干 |
| FinRL / FinRL-X | 金融强化学习框架，面向 DRL agent、market env、portfolio allocation | 适合把 RL 做成可复现实验；FinRL 官方已把新方向转向 FinRL-X / FinRL-Trading | 原始 FinRL 更偏教学和基准；直接用于 A 股实盘还要大量适配 | 作为 RL 训练参考框架，不直接替换现有 DQN |
| TensorTrade | RL trading agent 框架，模块化 exchange/action/reward/agent | 适合快速实验 reward/action scheme | 官方说明仍偏 beta；生态弱于 FinRL | 只作为 reward 设计参考 |
| PyPortfolioOpt | 组合优化库，支持均值方差、Black-Litterman、协方差收缩、HRP | 和当前个股推荐互补，可解决“买多少”的问题 | 不能解决 alpha 预测，只解决组合构建 | 高优先级引入，用于仓位和组合层 |

来源：Microsoft Qlib 官方论文和 GitHub、FinRL GitHub、TensorTrade GitHub、PyPortfolioOpt PyPI/GitHub。

### 1.2 回测与研究框架

| 库/平台 | 定位 | 优点 | 风险/限制 | 对本项目建议 |
|---|---|---|---|---|
| vectorbt | 向量化量化研究和回测 | 极快，适合扫参数、做多股票横截面验证、验证 14:30/22:00/09:20 信号的历史胜率 | 更适合研究，不适合复杂撮合和实盘事件流 | 高优先级引入，用来验证当前推荐逻辑 |
| RQAlpha | RiceQuant 开源 Python 回测框架 | A 股语境更自然，支持多证券，插件式架构 | 需要整理本地数据适配；生态不如最火时 | 中高优先级，用于 A 股 T+1、涨跌停、手续费规则回测 |
| Backtrader | Python 回测老牌框架 | API 友好，教程多，适合学习事件驱动 | 维护偏慢，现代依赖兼容性有风险 | 不作为新核心，只可用于原型 |
| Zipline-reloaded | Zipline 继承分支 | Pipeline 因子研究经典，和 Stefan Jansen 书配套 | A 股适配成本较高 | 可读代码，不建议主接入 |
| Lean / QuantConnect | 开源专业级事件驱动交易引擎，支持 Python/C#，回测到实盘 | 模块化强，数据/券商/费用/滑点/持仓模型完整，适合多资产和生产级部署 | C# 内核重，接入成本高；A 股本地数据和交易通道仍要适配 | 未来做多市场实盘时考虑 |
| NautilusTrader | Rust + Python 的高性能事件驱动交易引擎 | 回测和实盘同一核心，性能和测试体系强 | 更偏专业实盘工程，当前项目过早 | 未来高性能、多市场执行层候选 |
| vn.py | 中国市场实盘交易平台 | 国内期货/股票/券商网关生态强，CTA 实盘经验多 | 研究/ML 管线不是主强项 | 如果做国内实盘下单，优先研究 |

来源：vectorbt 文档、RQAlpha GitHub 元数据、Backtrader 官方文档/PyPI、Lean 官方文档、NautilusTrader 官方页。

### 1.3 数据与研究辅助库

| 类别 | 候选 | 用法 |
|---|---|---|
| A 股数据 | AKShare、TuShare Pro、BaoStock、JQData | 当前已用 AKShare/BaoStock；妖股、龙虎榜、涨停板、融资融券最好补 TuShare/JQData |
| 因子与统计 | pandas、polars、NumPy、scikit-learn、LightGBM、CatBoost、XGBoost | 当前 LGB 有效，下一步要补 CatBoost/XGBoost 做 ensemble baseline |
| 组合风险 | PyPortfolioOpt、riskfolio-lib | 用 HRP、风险预算、最大回撤约束来控制推荐仓位 |
| 绩效归因 | pyfolio/empyrical、quantstats | 用于日报/周报：收益、回撤、胜率、Sharpe、Calmar、换手 |

### 1.4 对当前项目的量化选型建议

短期不换 Qlib。现在最缺的不是“再换一个模型平台”，而是证据闭环：

1. `Qlib` 继续负责日频特征、短线预测、模型缓存。
2. `vectorbt` 负责快速回测当前信号，比如“14:30 强烈推荐后下一开盘日收益”“22:00 指数预测命中率”“09:20 修正是否提升收益/降低回撤”。
3. `RQAlpha` 负责 A 股真实交易规则回测，尤其 T+1、涨跌停、手续费、停牌、不能买入一字板等规则。
4. `PyPortfolioOpt` 负责把“推荐列表”变成“仓位建议”，避免只给股票不给仓位。
5. `FinRL/FinRL-X` 作为 RL 的参考环境和 baseline，不要让 RL 直接上生产；先用离线验证它是否比规则融合更好。

---

## 二、舆情分析库与金融 NLP 选型

### 2.1 英文金融舆情

| 模型/库 | 定位 | 适合任务 | 风险 | 建议 |
|---|---|---|---|---|
| FinBERT / ProsusAI FinBERT | 金融文本三分类：positive / negative / neutral | 英文新闻、财报摘要、海外宏观新闻 | 英文为主；对中文和 A 股股吧无效 | 高优先级，用作英文新闻快速打分 |
| Hugging Face Transformers | NLP/LLM 标准加载和推理框架 | 统一加载 FinBERT、中文金融模型、zero-shot 分类 | 需要模型治理和本地缓存 | 必须作为统一推理层 |
| VADER | 词典+规则情绪分析，偏社交媒体 | 英文短文本、Twitter/Reddit 类噪声文本 | 金融语义弱，不能理解反讽/财报语境 | 只作低成本 fallback |
| Financial PhraseBank | 金融情绪训练/评估数据集 | 英文金融情绪基准 | 数据偏小、偏欧洲公司新闻，商业使用要看许可 | 用于模型 smoke/eval，不直接当 A 股训练集 |

来源：ProsusAI FinBERT Hugging Face/GitHub、Hugging Face Transformers 文档、VADER PyPI、Financial PhraseBank 数据集说明。

### 2.2 中文金融舆情

| 模型/库 | 定位 | 适合任务 | 风险 | 建议 |
|---|---|---|---|---|
| FinGPT | 开源金融大语言模型体系，覆盖情绪、预测、RAG、多智能体 | 金融新闻摘要、事件影响判断、中文/英文金融文本 instruction tuning | 部署成本高；不同模型质量差异大；需要结构化输出约束 | 中高优先级，用作 LLM 事件修正层参考 |
| SnowNLP | 轻量中文 NLP，含情感分数、分词、摘要 | 快速中文舆情 baseline；东方财富股吧/雪球短文本初筛 | 默认情感训练数据偏商品评论，金融效果会偏；必须用金融语料重训 | 可快速接入，但一定要重训 |
| HanLP | 多语言 NLP，中文分词/NER/依存句法等 | 公司名、人物、政策、地名、行业实体识别 | 不是金融情绪模型本身 | 用于实体抽取和事件归因 |
| LTP | 哈工大中文 NLP 平台，词法/句法/语义分析 | 中文新闻事件结构化、实体关系抽取 | 商业使用需注意授权；不是直接情绪模型 | 研究/原型可用，生产要确认许可 |
| PaddleNLP / ERNIE 系列 | 中文 NLP 工程生态 | 中文分类、实体识别、模型微调 | 项目依赖较重 | 若本地中文模型微调，可进入候选 |

来源：FinGPT 官网/GitHub/AI4Finance、SnowNLP GitHub、HanLP 官方文档、LTP 官网/GitHub。

### 2.3 舆情数据源优先级

| 数据源 | 信息价值 | 使用方式 | 注意点 |
|---|---|---|---|
| RSS / GDELT | 全球宏观、地缘、政策事件 | 已在用；继续做 22:00 和 09:20 宏观修正 | 要去重、分语种、过滤低质量媒体 |
| GDELT DOC/GKG Tone | 全球新闻 tone、主题、地理、实体 | 做地缘风险和国家/行业冲击指数 | tone 是宏观情绪代理，不等于股价方向 |
| 东方财富股吧 | A 股个股情绪、妖股热度 | 个股短文本情绪、热度、异常讨论量 | 噪声大，容易被水帖/情绪极端误导 |
| 雪球 | 投资者观点、机构/大 V 讨论 | 个股中期关注度、分歧度 | 抓取稳定性和合规性要注意 |
| 公告/研报/交易所公告 | 公司层面硬信息 | 停复牌、业绩、减持、监管函、重大合同 | 需要事件分类，而不是简单情绪 |
| 龙虎榜/涨停板/资金流 | A 股投机情绪和妖股信号 | 妖股识别、短线情绪温度计 | 需要 TuShare/JQData/AKShare 数据补齐 |

### 2.4 舆情在交易系统里的正确位置

舆情不应该直接替代量化模型，也不应该让 LLM 直接拍脑袋荐股。更稳的结构是：

```text
基础模型信号
  + 当日盘面动量
  + 舆情事件冲击
  + 大盘/宏观修正
  + 风控约束
  = 最终操作建议
```

22:00 和 09:20 的定位应该是“事件修正层”：

| 时间 | 作用 | 主要输入 | 输出 |
|---|---|---|---|
| 14:30 | 盘中执行决策 | 当日行情、短线模型、历史推荐记录 | 下一开盘日指数、强买、必卖 |
| 22:00 | 夜间第一次修正 | 收盘行情、模型缓存、晚间全球新闻、商品、美元、加密 | 明日策略预案 |
| 09:20 | 开盘前最终修正 | 美股收盘、凌晨突发、国内早间政策、隔夜外盘 | 开盘动作：买/不买/减仓/卖 |

推荐结构化 LLM 输出：

```json
{
  "event_severity": 0.0,
  "market_impact": 0.0,
  "sector_impacts": [{"sector": "半导体", "impact": 0.3, "reason": "..."}],
  "stock_impacts": [{"code": "SH600000", "impact": -0.5, "action_override": "avoid"}],
  "policy_signal": 0.0,
  "confidence": 0.0,
  "decay_hours": 24
}
```

关键工程原则：

1. LLM 只做修正，不做原始 alpha。
2. 每条舆情必须有时间衰减，突发新闻不能永久影响模型。
3. 结构化输出必须入库，后续验证“LLM 修正到底有没有提升胜率”。
4. 对公司级硬新闻设置 override 权限，例如监管处罚、重大减持、停牌、财务造假。
5. 对宏观/地缘新闻只调仓位和行业权重，不直接强推个股。

---

## 三、量化大师、书单与经验

### 3.1 名人经验

| 人物 | 核心贡献 | 对当前项目的启发 |
|---|---|---|
| Jim Simons | 文艺复兴科技和 Medallion 代表了数据驱动、数学建模、科学家团队的极致路线 | 不要迷信叙事；小优势、多标的、严格执行、持续验证，比单次“神预测”更重要 |
| Ed Thorp | 从 blackjack 到可转债/统计套利，被视为现代量化先驱之一 | 概率、赔率、Kelly、风险约束是交易系统底层；先活下来再复利 |
| Marcos Lopez de Prado | 金融机器学习工程体系：triple-barrier、meta-labeling、purged CV、HRP | 当前项目最该补的是标签、交叉验证、防泄漏、回测真实性 |
| Ernest Chan | 独立量化交易实战派，强调回测陷阱、交易成本、均值回归/动量策略 | 策略要简单、可解释、能扣掉费用后赚钱；每个信号都要能落地交易 |
| Robert Carver | 系统化交易、风险目标、仓位框架 | 推荐列表必须变成风险预算和仓位，不然无法执行 |
| Cliff Asness / AQR | 价值、动量、质量等因子跨资产研究 | 因子不要孤立看，动量和价值可能互补；组合比单因子更稳 |
| Rishi Narang | 把黑箱量化拆成 alpha、风险、成本、组合、执行等模块 | 项目架构要模块化，每个模块单独验证 |
| Larry Harris | 市场微观结构和交易机制 | A 股涨跌停、T+1、滑点、成交量约束必须进入回测 |
| Ray Dalio | 全天候/风险平价/宏观情景思维 | 大盘和宏观预测更适合影响仓位和风险预算，而不是直接决定某只股票 |

来源：Zuckerman 的 Simons 传记页面、Edward Thorp 官网、Wiley/O'Reilly/出版社页面、AQR 论文页、Oxford Larry Harris 页面、Harriman House Robert Carver 页面。

### 3.2 必读书单：按用途排序

#### 第一组：金融机器学习和回测真实性

| 书 | 作者 | 为什么读 | 读完要落地什么 |
|---|---|---|---|
| Advances in Financial Machine Learning | Marcos Lopez de Prado | 金融 ML 的标签、CV、防过拟合、特征重要性体系 | triple-barrier 标签、purged walk-forward、meta-labeling |
| Machine Learning for Asset Managers | Marcos Lopez de Prado | 组合构建和资产管理角度的 ML | HRP、协方差去噪、组合稳健性 |
| Machine Learning for Algorithmic Trading | Stefan Jansen | Python 量化 ML 全流程，覆盖替代数据、NLP、回测 | 把 NLP 舆情和预测模型接入回测 |

#### 第二组：实战策略和交易工程

| 书 | 作者 | 为什么读 | 读完要落地什么 |
|---|---|---|---|
| Quantitative Trading | Ernest Chan | 独立量化交易入门，重视回测陷阱和费用 | 每个信号加入手续费、滑点、容量约束 |
| Algorithmic Trading | Ernest Chan | 均值回归、动量、配对交易等策略实现 | 为短/中/长线分别建立 baseline 策略 |
| Systematic Trading | Robert Carver | 系统化交易和仓位框架 | forecast → position sizing → risk target |
| Trading Systems and Methods | Perry Kaufman | 交易系统百科 | 技术指标和系统测试参考，不直接照抄 |

#### 第三组：组合、因子和市场结构

| 书 | 作者 | 为什么读 | 读完要落地什么 |
|---|---|---|---|
| Active Portfolio Management | Grinold & Kahn | 主动管理基本定律：信息系数、宽度、信息比率 | 评估模型 IC、rank IC、覆盖宽度 |
| Inside the Black Box | Rishi Narang | 用模块化视角理解量化系统 | 拆出 alpha、risk、cost、portfolio、execution 五层 |
| Trading and Exchanges | Larry Harris | 交易机制和微观结构经典 | A 股回测加入涨跌停、停牌、成交约束 |
| A Man for All Markets | Ed Thorp | 概率思维、Kelly、风险控制 | 仓位上限和下注比例规则 |
| The Man Who Solved the Market | Gregory Zuckerman | Simons 和 Renaissance 的数据驱动思想 | 科学化团队和模型治理，而不是追热点 |

### 3.3 从大师经验抽象出的项目规则

1. 每个模型都必须有样本外验证，不看训练集表现。
2. 所有推荐必须记录：模型版本、数据日期、舆情修正、推送时间、价格。
3. 回测必须加入 A 股真实交易限制：T+1、涨跌停、停牌、手续费、滑点、成交额容量。
4. 22:00 和 09:20 的 LLM 修正必须可回放、可验证。
5. 只让 LLM 调整风险和事件冲击，不让它覆盖量化模型的原始分。
6. 仓位比股票名更重要：强烈推荐但仓位过大仍然会出问题。
7. 妖股策略要单独建模，不能和普通趋势/价值策略混在一起。
8. 组合层要限制相关性暴露，同一题材不要一口气买满。
9. 交易成本会杀死高换手策略，所以 14:30 强买必须有预期收益阈值。
10. 任何“看起来很聪明”的复杂模型，都要先赢过简单 baseline。

---

## 四、对当前项目的落地方案

### Phase A：先把证据闭环补起来

目标：验证当前 14:30、22:00、09:20 推送到底有没有价值。

任务：

1. 新增 `prediction_snapshots` 表：记录每次推送的原始模型分、舆情修正、指数预测、个股推荐、当时价格。
2. 新增 `signal_backtest.py`：用 vectorbt 回测最近 N 年信号。
3. 为 14:30 强买建立指标：下一开盘日收益、1日/3日/5日收益、最大回撤、胜率、盈亏比。
4. 为 22:00/09:20 修正建立指标：修正前后方向是否提升、是否降低回撤。
5. 加入费用和滑点：A 股至少按万分之几佣金、印花税、滑点估算。

推荐库：vectorbt、empyrical/quantstats。

### Phase B：接入结构化舆情修正层

目标：让 LLM 的贡献变成可验证的数值，而不是只生成文字。

任务：

1. 将 LLM 地缘/宏观输出改成严格 JSON schema。
2. 接入 FinBERT 做英文新闻初筛：正/负/中性 + 置信度。
3. 接入 SnowNLP 或中文模型做中文短文本初筛，但必须用金融语料重训。
4. 用 HanLP/LTP 做实体抽取：公司、行业、国家、政策、人物。
5. 给事件设置 `impact`、`confidence`、`decay_hours`，写入数据库。
6. 在 22:00 和 09:20 推送中显示“修正原因”，但不显示底层库名。

推荐库：Transformers、FinBERT、FinGPT、SnowNLP、HanLP/LTP。

### Phase C：组合和仓位层

目标：从“推荐股票”升级成“可执行组合”。

任务：

1. 引入 PyPortfolioOpt 的 HRP 或风险预算模型。
2. 对每只股票设置基础仓位、最大仓位、行业上限、题材相关性上限。
3. 大盘预测影响总仓位，个股预测影响相对权重。
4. 卖出规则分成：硬止损、模型翻空、舆情硬负面、时间止损、组合降风险。
5. 每日收盘总结中给出组合风险暴露。

推荐库：PyPortfolioOpt、riskfolio-lib。

### Phase D：A 股真实规则回测

目标：避免“理论能买、现实买不到”。

任务：

1. 用 RQAlpha 或自建撮合模拟 A 股 T+1。
2. 加入涨跌停不可成交、停牌不可成交、一字板买不进。
3. 妖股策略单独回测：连板、首板、换手、成交额、题材热度。
4. 区分普通推荐和妖股推荐，不共用风险参数。

推荐库：RQAlpha、AKShare/TuShare/JQData。

### Phase E：未来实盘层

目标：当策略验证稳定后再考虑。

候选：

1. 国内券商/期货/CTA：vn.py。
2. 多资产、海外市场、完整事件驱动：Lean。
3. 高性能专业执行和回测同核：NautilusTrader。

当前不建议马上做实盘自动下单。先把“推荐、验证、回测、仓位”闭环跑稳。

---

## 五、优先级清单

### 立刻做

1. `vectorbt` 回测当前 14:30 强买和 22:00/09:20 指数预测。
2. LLM 输出结构化，写入数据库。
3. 推荐记录增加模型版本、数据日期、舆情修正字段。
4. SnowNLP/FinBERT 做舆情 baseline，但不直接控制买卖。

### 下一步做

1. PyPortfolioOpt 做仓位建议。
2. RQAlpha 或自建 A 股撮合做真实规则回测。
3. 中文金融舆情数据集建设：股吧/雪球/公告/新闻，人工标注一小批高质量样本。
4. RL 训练引入 FinRL-style baseline，对比当前规则融合。

### 暂不做

1. 直接替换 Qlib。
2. 让 LLM 独立荐股。
3. 直接实盘自动下单。
4. 为了“高级”引入过重框架，导致当前 scheduler 不稳定。

---

## 六、参考来源

量化平台与回测：

- Qlib 论文与项目：[Microsoft Research Qlib paper](https://www.microsoft.com/en-us/research/publication/qlib-an-ai-oriented-quantitative-investment-platform/)，[Qlib GitHub](https://github.com/microsoft/qlib)
- Lean / QuantConnect：[Lean 官网](https://www.lean.io/)，[Lean 文档](https://www.quantconnect.com/docs/v2/lean-engine/getting-started)，[Lean GitHub](https://github.com/QuantConnect/Lean)
- NautilusTrader：[官方 Open Source 页面](https://nautilustrader.io/open-source/)
- vectorbt：[官方文档](https://vectorbt.dev/)
- Backtrader：[官方站点](https://www.backtrader.com/)，[PyPI](https://pypi.org/project/backtrader/)
- RQAlpha：[GitHub 元数据](https://repos.ecosyste.ms/hosts/GitHub/repositories/ricequant%2Frqalpha)
- FinRL：[FinRL GitHub](https://github.com/AI4Finance-Foundation/FinRL)，[AI4Finance](https://ai4finance.org/)
- TensorTrade：[TensorTrade GitHub](https://github.com/tensortrade-org/tensortrade)，[TensorTrade 文档](https://www.tensortrade.org/en/latest/)
- PyPortfolioOpt：[PyPI](https://pypi.org/project/pyportfolioopt/)，[GitHub](https://github.com/PyPortfolio/PyPortfolioOpt)

舆情与 NLP：

- FinBERT：[Hugging Face ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)，[GitHub](https://github.com/ProsusAI/finBERT)
- FinGPT：[官网](https://fingpt.io/)，[GitHub](https://github.com/AI4Finance-Foundation/FinGPT)，[AI4Finance 论文页](https://ai4finance.org/research/fingpt-open-source-finllm.html)
- Hugging Face Transformers：[pipeline 文档](https://huggingface.co/docs/transformers/main_classes/pipelines)，[text classification 文档](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- VADER：[PyPI](https://pypi.org/project/vaderSentiment/)
- SnowNLP：[GitHub](https://github.com/isnowfy/snownlp)
- HanLP：[官方文档](https://hanlp.hankcs.com/docs/index.html)
- LTP：[官网](https://ltp.ai/)，[GitHub](https://github.com/HIT-SCIR/ltp)
- Financial PhraseBank：[Hugging Face dataset](https://huggingface.co/datasets/takala/financial_phrasebank)
- GDELT：[GKG 2.0 介绍](https://blog.gdeltproject.org/introducing-gkg-2-0-the-next-generation-of-the-gdelt-global-knowledge-graph/)，[DOC API Tone 支持](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/amp/)

书籍与人物：

- Marcos Lopez de Prado：[Advances in Financial Machine Learning - Wiley](https://dev.store.wiley.com/en-us/Advances%2Bin%2BFinancial%2BMachine%2BLearning-p-00000140)
- Ernest Chan：[Algorithmic Trading - MathWorks/Wiley 信息页](https://www.mathworks.com/academia/books/algorithmic-trading-chan.html)，[Quantitative Trading - O'Reilly](https://www.oreilly.com/library/view/quantitative-trading-2nd/9781119800064/)
- Robert Carver：[Systematic Trading - Harriman House](https://harriman.house/books/systematic-trading/)
- Larry Harris：[Trading and Exchanges - Oxford Academic](https://academic.oup.com/book/52292)
- Rishi Narang：[Inside the Black Box - Wiley](https://newsroom.wiley.com/press-releases/press-release-details/2013/Inside-the-Black-Box-A-Simple-Guide-to-Quantitative-and-High-Frequency-Trading-2nd-Edition/default.aspx)，[2024 版 Wiley 信息](https://www.wiley-vch.de/en/areas-interest/finance-economics-law/finance-investments-13fi/investments-securities-13fi3/inside-the-black-box-978-1-119-93189-8)
- Perry Kaufman：[Trading Systems and Methods - Wiley](https://www.wiley-vch.de/en/areas-interest/finance-economics-law/finance-investments-13fi/trading-13fi4/trading-systems-and-methods-978-1-119-60535-5)
- Stefan Jansen：[Machine Learning for Algorithmic Trading - Apple Books](https://books.apple.com/us/book/machine-learning-for-algorithmic-trading/id1525046439)
- Edward Thorp：[A Man for All Markets - 作者官网](https://www.edwardothorp.com/books/a-man-for-all-markets/)
- Cliff Asness / AQR：[Value and Momentum Everywhere](https://www.aqr.com/Insights/Research/Journal-Article/Value-and-Momentum-Everywhere)
- Jim Simons：[The Man Who Solved the Market - Apple Books](https://books.apple.com/us/book/the-man-who-solved-the-market/id1452108275)
