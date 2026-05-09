# cx 量化库、舆情分析库与量化大师经验深度调研

日期：2026-05-08  
背景：在 `cx-quant-sentiment-deep-research-2026-05-07.md` 和 `cc-量化与舆情深度调研.md` 基础上补充最新开源生态、舆情工具链、书单和可落地迭代建议。  
目标：回答“除了 Qlib，还有哪些强库；舆情分析怎么做；量化大师、书和经验怎么转化为当前项目路线”。

> 结论先行：当前项目不应该替换 Qlib，而应该把系统拆成五层：`Qlib/LGB` 做 A 股日频 alpha，`vectorbt + RQAlpha` 做验证回测，`PyPortfolioOpt/Riskfolio-Lib/skfolio` 做仓位组合，`FinBERT/FinGPT/FinRobot + 中文 NLP + GDELT/股吧/公告` 做事件修正和研究助手，`vn.py/Lean/NautilusTrader` 只在实盘执行或跨市场执行成熟后再接。

---

## 1. 量化库全景：按系统层级选型

### 1.1 研究与 AI 量化平台

| 库/平台 | 当前活跃度与定位 | 强项 | 局限 | 对本项目建议 |
|---|---|---|---|---|
| [Qlib](https://github.com/microsoft/qlib) | 微软 AI-oriented quant 平台，覆盖数据、模型、回测、组合、RL，并强调 AI workflow；新版生态还接入 RD-Agent 自动化研发流程 | Alpha158、LGB/深度模型、A 股/日频研究、已有接入成本最低 | 数据格式、日历、全市场覆盖容易踩坑；实盘执行不是强项 | 继续作为 A 股短线模型主干，不替换 |
| [FinRL / FinRL-X](https://github.com/AI4Finance-Foundation/FinRL) | AI4Finance 金融强化学习生态；原 FinRL 更偏教学/研究，FinRL-X/Trading 偏新架构 | RL agent、portfolio allocation、市场环境、baseline | 直接套 A 股实盘不现实，需要交易规则和数据适配 | 作为 RL baseline 和 reward/env 设计参考 |
| [FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) | AI4Finance 金融 Agent 平台，强调 LLM、RL、量化分析和研究报告自动化 | 多 Agent 研究、财报/估值/同业比较、可作为舆情解释器和研究助手 | 更像研究/报告 Agent，不是日频 alpha 训练引擎；依赖外部数据与 LLM API | 用于 22:00/09:20 新闻解释、财报摘要、个股风险说明，不直接决定买卖 |
| [TradeMaster](https://github.com/TradeMaster-NTU/TradeMaster) | NTU 维护的 RL 量化平台，覆盖数据、环境、模拟器、算法、评估 | RL 论文复现、模型 zoo、Alpha158 支持、港股/期货数据样例 | 工程接入复杂，生产生态弱于 Qlib/Lean | 用作 RL 实验对照，不进生产主链 |
| [TensorTrade](https://github.com/tensortrade-org/tensortrade) | composable RL trading framework，仍提示 beta 风险 | exchange/action/reward/agent 模块化，适合试 reward/action scheme | 生产风险高，社区活跃度一般 | 只借鉴 reward/env 抽象 |
| [OpenBB](https://github.com/OpenBB-finance/OpenBB) | 金融数据平台，面向 analysts/quants/AI agents，Python/REST/MCP/Excel 多入口 | 数据集成、美国/全球资产、AI agent 接口、local-first | 不是模型训练框架；AGPL 许可要注意 | 做美股/港股/宏观数据补充层，暂不替代 Qlib |

判断：Qlib 最像当前项目的“研究主干”。FinRL/TradeMaster 更像 RL 实验台，OpenBB 更像数据集成层。真正要补的是数据覆盖、验证闭环和仓位层，不是换一个“更强 Qlib”。

### 1.2 回测与策略验证

| 库/平台 | 定位 | 强项 | 局限 | 对本项目建议 |
|---|---|---|---|---|
| [vectorbt](https://vectorbt.dev/) | pandas/NumPy/Numba/Rust 加速的向量化研究与回测 | 多股票、多参数、快速扫策略；适合验证信号胜率 | 不天然处理复杂撮合、涨跌停、T+1 | 第一优先级：验证 14:30/22:00/09:20 信号 |
| [RQAlpha](https://github.com/ricequant/rqalpha) | RiceQuant 开源 A 股回测/交易框架，支持多证券与 Mod 扩展 | A 股语境、扩展性、回测/模拟链路 | 商业使用许可要确认；本地数据适配要做 | 第二优先级：验证 T+1、涨跌停、停牌、手续费 |
| [Backtrader](https://www.backtrader.com/) | 老牌 Python 事件驱动回测框架 | 易学、示例多、指标丰富 | 维护节奏偏慢，现代 ML/大横截面不强 | 学习/原型可用，不做新核心 |
| [Zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) | Quantopian Zipline 继承分支，Stefan Jansen 维护 | Pipeline 因子研究、和 ML for Trading 书配套 | A 股适配成本高 | 作为因子研究参考，不接主链 |
| [Lean / QuantConnect](https://www.lean.io/) | 专业级开源算法交易引擎，Python/C#，研究、回测、优化、实盘 | 跨资产、券商模型、费用、组合、实盘部署完整 | C# 内核重，A 股本地化适配成本高 | 做美股/港股/加密实盘时再评估 |
| [NautilusTrader](https://nautilustrader.io/docs/latest/concepts/overview) | Rust-native 生产级事件驱动交易系统，研究到实盘语义一致 | 多市场、高性能、确定性模拟、live parity | 学习曲线陡，当前过早 | 未来高性能执行层候选 |
| [vn.py / VeighNa](https://github.com/vnpy/vnpy) | 国内流行开源量化交易平台 | 国内期货/券商网关、CTA、实盘生态 | ML 研究链不是主强项 | 如果要接国内实盘下单，优先研究 |

判断：当前最该补 `vectorbt` 和 `RQAlpha`。`Lean/Nautilus/vn.py` 是执行层，不应该在数据和模型还不稳定时接入。

### 1.3 组合、风控与绩效分析

| 库 | 定位 | 能解决的问题 | 对本项目建议 |
|---|---|---|---|
| [PyPortfolioOpt](https://github.com/PyPortfolio/PyPortfolioOpt) | 组合优化，均值方差、Black-Litterman、协方差收缩、HRP | “推荐哪些股”升级为“各买多少” | 高优先级，先做 HRP/风险预算 |
| [Riskfolio-Lib](https://pypi.org/project/riskfolio-lib/) | 更完整的组合优化与风险度量，支持大量风险指标、HRP/HERC/NCO | CVaR、回撤风险、风险平价、约束优化 | 中高优先级，用于复杂仓位和风险报告 |
| [skfolio](https://github.com/skfolio/skfolio) | sklearn 风格组合优化、CV、压力测试 | 组合模型调参、交叉验证、stress test | 可作为 PyPortfolioOpt 之后的升级项 |
| [Alphalens-reloaded](https://pypi.org/project/alphalens-reloaded/) | alpha 因子分析：收益、IC、换手、分组 | 判断模型分数是否真的有排序能力 | 高优先级，用于 Qlib/LGB 分数诊断 |
| [QuantStats](https://github.com/ranaroussi/quantstats) | 绩效报告、风险指标、HTML tear sheet | 日报/周报：Sharpe、回撤、胜率、月度收益 | 高优先级，用于验证和报告 |

判断：项目现在“会推荐”，但还不像交易系统。组合层要把每次推送转成：目标仓位、最大仓位、止损、行业上限、总风险预算。

---

## 2. 舆情分析库与金融 NLP 工具链

### 2.1 金融专用模型与框架

| 工具 | 适合任务 | 强项 | 风险 | 对本项目建议 |
|---|---|---|---|---|
| [FinBERT](https://github.com/ProsusAI/finBERT) | 英文金融新闻正/负/中三分类 | 金融语料预训练 + Financial PhraseBank 情绪分类 | 英文为主；代码基于较老 transformers 生态 | 英文新闻快速情绪基线 |
| [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | 金融 LLM、情绪、RAG、预测、报告、agent | 开源金融 LLM 生态，支持多任务金融 instruction 数据 | 模型质量差异大，本地部署成本高 | 作为 22:00/09:20 事件修正层参考 |
| [FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) | 金融 LLM Agent、研究自动化、报告生成 | 可把新闻、公告、财报、估值、同行比较组织成多 Agent 工作流 | 不应该把 Agent 输出直接当 alpha；要落结构化事件表并回测 | 作为“解释和归因层”，回答为什么模型要修正 |
| [FinNLP / FinGPT 数据集生态](https://huggingface.co/FinGPT) | 金融 NLP 数据、instruction、NER、headline、sentiment | 任务覆盖广，适合金融 LLM 微调/评估 | 中文 A 股场景仍需自建数据 | 用于评估集和训练样本设计 |
| [Transformers](https://huggingface.co/docs/transformers/index) | 统一加载 BERT/LLM/多模态模型 | 模型生态最大，便于接 FinBERT/中文模型 | 需要模型缓存、版本治理、推理资源 | 建议作为统一推理层 |
| [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) | 英文金融情绪基准数据 | 4840 条金融新闻句子，正/负/中性 | CC-BY-NC-SA，商业使用受限；非中文 | 用作英文情绪 smoke/eval，不直接训练商业模型 |

### 2.2 中文 NLP 与中文金融舆情

| 工具 | 适合任务 | 强项 | 风险 | 对本项目建议 |
|---|---|---|---|---|
| [SnowNLP](https://github.com/isnowfy/snownlp) | 中文短文本情绪快速 baseline | 简单、轻量、可重训 | 默认情感数据偏商品评论，金融场景会偏 | 可先接，但必须用股吧/财经语料重训 |
| [LTP](https://ltp.ai/) | 中文分词、词性、NER、句法、语义角色 | 中文基础 NLP 强，支持 6 类核心任务 | 商业使用许可要确认 | 用于公司/行业/政策实体抽取 |
| [HanLP](https://hanlp.hankcs.com/en/demos/ner.html) | 多语种 NER、分词、句法 | 工程化好，多语言实体识别 | 在线 API/auth 与许可要确认 | 备选实体抽取层 |
| [PaddleNLP / ERNIE](https://paddlenlp.readthedocs.io/en/latest/index.html) | 中文模型微调、分类、NER、ERNIE 系列 | 中文生态完整，工业化部署能力强 | 依赖较重 | 中文金融分类模型的中期方案 |
| [spaCy](https://github.com/explosion/spaCy) | 英文/多语种 NER、分类、流水线 | 生产级、高速、70+ 语言 | 中文金融效果不如专门中文模型 | 英文新闻实体抽取和生产 pipeline |
| [PyABSA](https://pyabsa.readthedocs.io/en/stable/0_intro/introduction.html) | Aspect-based sentiment analysis，方面级情绪分类和方面词抽取 | 能把“新能源车利好但锂矿利空”拆成不同 aspect，适合行业/主题舆情 | 金融中文仍需要自建标注集；直接套通用模型会误判 | 用作细粒度舆情研究，不做第一版生产依赖 |
| [VADER](https://github.com/cjhutto/vaderSentiment) | 社交媒体短文本情绪 | 快、可解释、规则词典法 | 英文为主，金融语义弱 | 只作为英文社媒 fallback |

### 2.3 舆情数据源优先级

| 数据源 | 信息类型 | 价值 | 接入建议 |
|---|---|---|---|
| RSS / 新闻源 | 全球宏观、地缘、政策、行业事件 | 已在用，是 22:00/09:20 的基础 | 加去重、可信源权重、事件时间衰减 |
| [GDELT DOC/Context API](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/amp/) | 全球新闻搜索、上下文、近实时事件 | 地缘风险和国家/行业冲击 | 做宏观/地缘风险指数，不直接荐股 |
| [HuggingFace 金融数据集](https://huggingface.co/datasets?search=financial) | 金融新闻、推文、财报、分类/情绪数据 | 可以快速构建英文情绪 smoke test 和回归测试 | 许可证、来源质量、语言和市场差异要逐个核对 | 只用作评估/预训练参考，不直接替代 A 股标注 |
| [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | 英文金融推文 bearish/bullish/neutral | MIT，约 1.2 万条，适合英文金融情绪 baseline | Twitter/X 语境和 A 股股吧差异很大 | 用于英文社媒模型验证 |
| 东方财富股吧 | 个股热度、散户情绪、妖股温度 | 短线/妖股很有价值 | 使用 AKShare/爬虫时注意限速与合规 |
| 雪球/微博财经 | 观点、分歧、热度、KOL 传播 | 对中短期主题热度有效 | 噪声大，必须做账号/重复内容过滤 |
| 公司公告/交易所公告 | 硬事件：减持、业绩、监管、停复牌 | 负面硬事件可一票否决 | 必须结构化事件分类，不只做情绪 |
| 涨停板、龙虎榜、资金流、融资融券 | A 股投机情绪和资金行为 | 妖股识别核心输入 | 建议补 TuShare Pro/JQData，AKShare 做备份 |
| 美股/港股新闻与公告 | 海外/港股扩展 | 未来多市场必要 | 可用 OpenBB + RSS + 交易所公告补充 |

### 2.4 推荐的舆情架构

```text
新闻/RSS/GDELT/公告/股吧/雪球
        |
        v
去重、可信源权重、语言识别、实体抽取
        |
        +--> 英文金融新闻：FinBERT / spaCy
        +--> 中文短文本：SnowNLP(重训) / PaddleNLP / PyABSA
        +--> 深度事件解释：LLM / FinGPT-style prompt
        +--> 多源研究归因：FinRobot-style agent workflow
        |
        v
结构化事件表 event_impacts
  - target_type: market / sector / stock
  - impact: -1~1
  - confidence: 0~1
  - decay_hours
  - source_quality
  - hard_override: avoid / sell / reduce / none
        |
        v
22:00/09:20 对量化信号做修正，不直接生成原始 alpha
```

关键原则：

1. LLM 只做事件修正和解释，不直接拍脑袋荐股。
2. 每条舆情必须结构化入库，后续能回测“修正前/修正后”。
3. 公司硬负面事件有 override 权限，宏观新闻只调仓位和行业权重。
4. 舆情有时效衰减，隔夜新闻不能永久影响模型。
5. 22:00 和 09:20 主要差异应来自“新增新闻/外盘变化”，股票池不应大幅漂移。

### 2.5 2026 年补充判断：舆情不是“情绪分”，而是“事件冲击表”

最新金融 NLP/Agent 生态的方向很明确：单纯给新闻打正负分已经不够，真正有交易价值的是把文本变成可回测的事件冲击。

建议把舆情系统拆成四类输出：

| 输出 | 示例 | 模型/工具 | 用法 |
|---|---|---|---|
| 情绪标签 | bullish / bearish / neutral | FinBERT、Twitter Financial News Sentiment 微调模型、SnowNLP 重训 | 快速粗筛新闻和社媒 |
| 实体与主题 | 公司、行业、国家、政策、商品 | LTP、HanLP、spaCy、PaddleNLP | 把新闻映射到指数、行业、个股 |
| 方面级冲击 | “AI 算力利好，消费电子中性，锂矿利空” | PyABSA、LLM JSON 抽取 | 防止一条新闻被粗暴套到所有相关股票 |
| 研究归因 | 为什么影响、影响多久、置信度多少 | FinGPT/FinRobot-style Agent + RAG | 给推文和盘后复盘提供可解释原因 |

这也解释了为什么 22:00 和 09:20 的推荐不应大幅变化：量化分数应该相对稳定，只有隔夜外盘、突发宏观、监管/公司公告、商品价格大波动这类新增事件才能改变权重。若两次股票池差异很大，要优先排查缓存、随机模型、新闻去重、排序 tie-breaker 和 LLM 非确定性。

---

## 3. 量化大师与经验

### 3.1 人物与可迁移经验

| 人物 | 代表意义 | 可迁移到当前项目的规则 |
|---|---|---|
| Jim Simons / Renaissance | 数据驱动、科学家团队、模型纪律 | 数据质量和严格执行比主观叙事重要；小优势、多标的、反复验证 |
| Ed Thorp | 概率、套利、Kelly、风险控制 | 每个信号都要问赔率；仓位必须受胜率/盈亏比约束 |
| Marcos Lopez de Prado | 金融 ML 工程体系 | 标签设计、purged CV、meta-labeling、防泄漏比换模型更重要 |
| Ernest Chan | 独立量化实战 | 简单策略、交易成本、容量、滑点是生死线 |
| Robert Carver | 系统化交易与仓位框架 | forecast 要转换为 position sizing；风险目标要稳定 |
| Rishi Narang | 量化系统模块化解释 | 系统拆成 alpha、risk、cost、portfolio、execution，每层单测 |
| Cliff Asness / AQR | 因子投资、价值/动量/质量 | 因子组合比单一神因子更稳；看 IC、宽度、相关性 |
| Larry Harris | 市场微观结构 | A 股 T+1、涨跌停、停牌、成交约束必须进回测 |
| Stefan Jansen | ML + alternative data + Python 工作流 | 机器学习、NLP、因子、回测要放在同一研究闭环里 |

### 3.2 书单：按落地价值排序

#### 第一梯队：必须读

| 书 | 作者 | 为什么读 | 立刻落地 |
|---|---|---|---|
| [Advances in Financial Machine Learning](https://dev.store.wiley.com/en-us/Advances%2Bin%2BFinancial%2BMachine%2BLearning-p-00000140) | Marcos Lopez de Prado | 金融 ML 标签、CV、防过拟合、特征重要性核心 | triple-barrier、purged walk-forward、meta-labeling |
| [Machine Learning for Algorithmic Trading](https://books.apple.com/us/book/machine-learning-for-algorithmic-trading/id1525046439) | Stefan Jansen | Python ML 量化全链路，含 NLP、回测、替代数据 | 统一研究流程和 notebook/report 模板 |
| [Quantitative Trading](https://www.wiley.com/) / Algorithmic Trading | Ernest Chan | 独立量化实战入门，强调成本、容量、回测陷阱 | 每个策略都加费用、滑点、容量约束 |
| [Systematic Trading](https://www.harriman-house.com/systematic-trading) | Robert Carver | 从预测到仓位、风险目标、交易频率 | 建立 forecast -> position sizing 模块 |
| [Inside the Black Box](https://www.thequantbook.com/) | Rishi Narang | 把量化系统拆开看 | 项目模块化：alpha/risk/cost/portfolio/execution |

#### 第二梯队：组合与风险

| 书 | 作者 | 用途 |
|---|---|---|
| Machine Learning for Asset Managers | Marcos Lopez de Prado | HRP、协方差去噪、组合稳健性 |
| Active Portfolio Management | Grinold & Kahn | 信息系数、宽度、信息比率，适合评估模型排名能力 |
| Trading and Exchanges | Larry Harris | 交易机制、滑点、流动性、订单执行 |
| Trading Systems and Methods | Perry Kaufman | 技术交易系统百科，用于策略 baseline |

#### 第三梯队：认知与历史

| 书 | 作者 | 用途 |
|---|---|---|
| [The Man Who Solved the Market](https://www.randomhousebooks.com/books/557104/) | Gregory Zuckerman | 理解 Simons 的数据驱动和模型纪律 |
| [A Man for All Markets](https://www.edwardothorp.com/books/a-man-for-all-markets/) | Ed Thorp | 概率思维、Kelly、套利和风险控制 |
| Principles | Ray Dalio | 系统化决策、宏观情景、风险平价思想 |
| Fooled by Randomness | Nassim Taleb | 警惕幸存者偏差和随机性误判 |

---

## 4. 对当前项目的具体路线

### 4.1 不换 Qlib，补四个关键短板

1. **数据覆盖层**  
   继续修全 A 数据覆盖。A 股日线优先 `TuShare Pro 批量交易日接口`，AKShare 做免费备选，baostock 只做缺口修复；美股/港股再评估 OpenBB。

2. **信号验证层**  
   接 `vectorbt` 验证三类推送：14:30、22:00、09:20。核心指标：下一交易日收益、3/5 日收益、胜率、盈亏比、最大回撤、换手、容量。

3. **A 股真实规则回测层**  
   接 `RQAlpha` 或自建撮合：T+1、涨跌停、停牌、手续费、滑点、一字板买不进。

4. **组合仓位层**  
   接 PyPortfolioOpt/Riskfolio-Lib：把短中长推荐转为仓位；大盘预测调总仓位，个股预测调相对权重。

### 4.2 舆情落地路线

Phase 1：结构化事件表

- 新增 `event_impacts`：市场、行业、个股三层影响。
- LLM 输出严格 JSON：`impact/confidence/decay_hours/source_quality/hard_override`。
- 所有 22:00/09:20 修正都记录“修正前分数、修正后分数、原因”。

Phase 2：模型组合

- 英文新闻：FinBERT + spaCy。
- 中文股吧/雪球：SnowNLP 重训做快速筛选。
- 公司/行业/政策实体：LTP 或 HanLP。
- 复杂事件解释：当前 LLM 或 FinGPT-style 金融 prompt。

Phase 3：验证

- 回测“没有舆情修正 vs 有舆情修正”。
- 单独评估硬负面 override：是否减少大亏。
- 单独评估宏观/地缘修正：是否降低组合回撤。

### 4.3 妖股/五倍股/十倍股单独建模

普通 LGB 日频预测不适合直接寻找 5 倍股/10 倍股。妖股要单独做“投机情绪模型”：

| 信号 | 数据 | 解释 |
|---|---|---|
| 连板/首板 | 涨停板池 | 首板、二连板、三连板，板块高度 |
| 板块热度 | 概念涨停数量 | 同题材涨停越集中，情绪越强 |
| 流通市值 | 基本面/行情 | 小盘更容易被资金推动 |
| 换手和成交额 | 日行情 | 高换手但未爆量出货，是短线核心 |
| 龙虎榜 | 游资席位 | 识别活跃资金和接力风险 |
| 舆情突增 | 股吧/雪球/微博 | 情绪扩散速度 |
| 监管/减持风险 | 公告/交易所 | 一票否决或大幅降权 |

妖股评分建议：

```text
monster_score =
  0.25 * limit_up_chain_score
+ 0.20 * volume_turnover_anomaly
+ 0.20 * sector_heat
+ 0.15 * float_market_cap_score
+ 0.10 * sentiment_spike
+ 0.10 * capital_flow_or_lhb_score
```

风控：

- 妖股池和普通推荐池分开展示。
- 单只妖股最大仓位不超过 3%~5%。
- 5 板以后默认不追，除非总龙头且市场情绪极强。
- 换手率异常放大、跌停开盘、监管关注、减持公告触发强制降权/卖出。

### 4.4 全 A 数据下载慢的解决方案

当前慢的根因不是 Qlib 本身，而是“全 A 首次/缺口回补”还在走 baostock 逐只串行请求。按当前本地运行状态，生产 Qlib `all.txt` 只有 280 只，后台全 A 回补需要拉 5000+ 只；如果继续用 baostock 每只股票单独请求多年历史，耗时会以小时计，而且容易遇到 `Broken pipe`、`网络接收错误`、会话掉线。

结论：把数据更新分成三条链路，不再混在一个慢任务里。

| 链路 | 目标 | 数据源优先级 | 频率 | 要点 |
|---|---|---|---|---|
| 首次全量/大缺口回补 | 建成 4500+ 全 A 可训练数据 | TuShare Pro > AKShare > baostock | 手动或周末 | 允许慢，但必须可断点续跑、分批重连、staging 后健康检查 |
| 每日盘后增量 | 只补最近 1-5 个交易日 | TuShare Pro `daily`/`daily_basic`/`adj_factor` 按交易日批量 > AKShare recent bars > baostock gap repair | 每个开盘日 17:00 后 | 不再全量重拉多年历史；只写缺口 |
| 推文/盘中行情 | 给 9:20、14:30、22:00 快速读取 | AKShare `stock_zh_a_spot_em()` + 磁盘缓存 + Tencent fallback | 17:05 预热，盘中 TTL | 不参与模型训练，只做行情展示、动量因子和兜底候选 |

推荐落地顺序：

1. **日常更新默认走 TuShare 批量日线**  
   如果配置了 `TUSHARE_TOKEN`，`update_qlib_data.py --provider auto` 应优先用 TuShare。TuShare 的优势是可以按 `trade_date` 拉全市场 `daily`，再合并 `daily_basic` 和 `adj_factor`，比 baostock “一只股票一个请求”更适合全 A 每日更新。没有 token 时再用 AKShare/baostock。

2. **baostock 从主链降级成 repair provider**  
   baostock 免费、字段够用，但全 A 串行历史回补太慢，且会话稳定性一般。它适合补少数失败股票、修缺口、校验特殊日期，不适合作为每天全市场历史主数据源。

3. **严格区分历史日线和 spot 快照**  
   AKShare `stock_zh_a_spot_em()` 很适合一次拿全 A 当前行情，已经可以用来做 17:05 缓存预热，避免每次 `main.py` 都现场下载 spot。但 spot 快照不能直接替代训练用的复权日线；它只能做当天行情展示、盘中动量、流动性 proxy。

4. **所有 Qlib 写入必须 staging + health gate**  
   每次更新先写 `data/storage/qlib_data_staging/cn_data`，通过以下检查再 promote：
   - `all.txt` 标的数 >= `LGB_MIN_DATA_INSTRUMENTS`。
   - feature bin 是 Qlib `[start_index, values...]` 格式，不能裸 `np.append`。
   - 最近 N 个交易日 close 覆盖率达标。
   - `smoke_lgb_predict.py` 至少跑出 `LGB_MIN_PREDICTIONS` 个有限预测。

5. **增量更新必须按“每只股票 last_success_date”续跑**  
   当前脚本已有 `update_manifest.json` 的思路，后续要把它变成硬约束：失败股票只记录失败，不推进生产 calendar；成功股票写入 staging；下一次只从缺口日期继续。这样全 A 首次回补中断后，不会从头再来。

6. **不要用多线程 baostock 当主方案**  
   baostock 的 session 更像全局状态，多线程 `login/logout` 容易互相影响。真正要并发，优先用进程级隔离、分 shard 写独立 staging，再合并；但这复杂度比直接接 TuShare/JQData 更高。短期更稳的做法是单进程分批重连，长期换批量源。

目标时间：

| 场景 | 当前问题 | 目标 |
|---|---|---|
| 首次全 A 建库 | baostock 串行可能数小时到十几小时 | 允许跑周末/夜间；必须可断点续跑 |
| 每日盘后增量 | 不应再扫 5000 只多年历史 | 5-15 分钟完成 |
| 22:00 推文 | 不应现场重新下载全 A spot | 直接读 17:05 after-close cache |
| 模型训练 | 不应在数据半残时训练 | 只有 health + smoke 通过才训练/发布 |

开发拆分和验收标准：

| 任务 | 实现要点 | 验收 |
|---|---|---|
| `--provider auto` | 有 `TUSHARE_TOKEN` 时优先 TuShare；无 token 时 AKShare；baostock 只补失败标的/失败日期 | 日常增量日志能看到 provider 决策，不再默认全量 baostock |
| TuShare 交易日批量增量 | 按 `trade_date` 拉全市场 `daily`、`daily_basic`、`adj_factor`，再转成 Qlib feature | 最近 1-5 个交易日增量不需要逐只股票请求 |
| manifest 断点续跑 | 记录每只股票 `last_success_date`、失败原因、失败日期；失败不推进生产 calendar | 中断后重跑只补缺口，不从头扫 5000+ 只 |
| staging 发布 | 先写 staging，health/smoke 过线后原子 promote 到生产目录 | `all.txt`、calendar、bin 文件不因半残更新污染生产数据 |
| spot 缓存预热 | 17:05 跑一次 AKShare 全 A spot 并写磁盘缓存；22:00/09:20 优先读缓存 | `main.py --run-now` 不再每次现场卡在 `Loading A-share spot data` |
| 数据状态报告 | 增加/完善状态命令，展示标的数、最近交易日覆盖率、有限预测数、失败标的 Top N | 训练前能一眼判断是否该训练，避免 NaN/低覆盖模型发布 |

关键提醒：Qlib 官方也强调可以下载示例 CN 数据，但 Yahoo 来源数据“不一定完美”，并且离线包不能直接做增量更新；所以它适合作为研究基线，不适合作为当前 A 股生产全量数据的唯一来源。生产链应以自己可控的 TuShare/AKShare/baostock 数据写入、健康检查和预测缓存为准。

### 4.5 Codex 补充意见：对当前方案和 cc 文档的校正

我的判断是：`cc-量化与舆情深度调研.md` 可以作为生态清单参考，但不能直接照着做工程路线。当前项目最要紧的不是“换更强模型”，而是把数据、预测、推文、复盘四件事做成稳定闭环。

必须优先坚持的工程决策：

1. **全 A 有限预测数是第一闸门**  
   现在的问题本质不是 LGB 不该用，而是全 A 数据覆盖不足导致 LGB 无法稳定覆盖 4500+ 标的。scheduler 可以在有限预测数不足时降级，但长期目标应该从数据链解决，让 LGB 一定有足够覆盖，而不是长期依赖 fallback。

2. **Qlib bin 写法不能随意改**  
   有些文档建议用 `np.append` 直接拼 bin，这个方向不可靠。当前正确格式是 Qlib 的 `[start_index, values...]`，写错会让 Qlib 读出错位、NaN 或低覆盖，最后表现成模型“能跑但结果不可信”。

3. **不要把 baostock 多线程当主要加速方案**  
   baostock 的登录会话更像全局状态，多线程 `login/logout` 容易互相踢掉。短期可以单进程分批重连、断点续跑；真正要解决速度，应切到 TuShare/JQData 这类按交易日批量拉全市场的源。

4. **vectorbt 不能被排除**  
   当前最缺的是“推荐是否真的有效”的快速验证。vectorbt 虽然不能完整模拟 A 股撮合，但非常适合先验证 14:30、22:00、09:20 三类信号的收益、胜率、回撤、换手。A 股真实规则再交给 RQAlpha 或自建撮合层。

5. **GDELT 不是主舆情链路**  
   GDELT 适合做全球宏观、地缘政治、国家/行业冲击，不适合作为 A 股个股推荐主舆情源。项目主链应该是 RSS/新闻、公告、股吧/雪球、涨停板/龙虎榜/资金流，再由 LLM 结构化成事件冲击。

6. **LLM 不直接荐股，只做事件修正**  
   22:00 和 09:20 的推送可以因为突发新闻、外盘、商品、监管公告而修正，但基础股票池应该来自同一份量化预测缓存。两次推荐差异很大时，优先怀疑缓存、排序、随机模型、新闻去重或 LLM 非确定性。

7. **RL 不能替代监督模型，先做二层决策器**  
   更靠谱的路线是让 RL 学习“在 Qlib/LGB 分数、动量、舆情冲击、大盘预测、波动率、流动性约束下如何调仓/降权/卖出”，而不是直接让 RL 从原始行情里猜明天涨跌。RL 的输出应先作为仓位/风险调整，不直接覆盖 LGB alpha。

8. **妖股模型必须单独建池**  
   5 倍股、10 倍股不是普通日频 alpha 的线性延伸，核心是连板高度、题材热度、流通盘、换手、龙虎榜、监管风险和社媒扩散。普通推荐池和妖股池必须分开打分、分开风控、分开展示。

9. **推文质量要服务交易动作**  
   推文不应该堆信息。顺序应固定为：世界大事 -> 世界格局影响 -> A 股/港美/商品/加密影响 -> 明日指数预测 -> 短中长线个股前五 -> 综合前五 -> 黄金和加密。14:30 推文更应该像交易指令：下一开盘日指数预测、强烈推荐买入、明确卖出/减仓。

10. **收盘复盘必须进入训练数据资产**  
   每个收盘后都要把早上/午后/昨晚预测和真实结果比对，记录错误原因：数据缺失、模型偏差、舆情误判、行业突发、市场风格切换、流动性问题。第一阶段先入库和报告，第二阶段再把这些标签用于 meta-labeling、样本权重和 RL reward。

推荐二版本路线：

| 版本 | 目标 | 关键交付 |
|---|---|---|
| V1 稳定版 | 让全 A 每天都有可信量化分数和清晰推文 | TuShare/AKShare/baostock repair 数据链、4500+ 有限预测、spot 缓存、三类推文模板、收盘复盘 |
| V2 进化版 | 让系统能自我评估并学习修正 | vectorbt/RQAlpha 回测、event_impacts 舆情表、meta-labeling、RL 仓位决策器、妖股独立模型 |

### 4.6 cc 文档详细审查：保留、修正和删除项

重新细读 `plans/cc-量化与舆情深度调研.md` 后，我的结论是：生态清单和旧设计审查有参考价值，但后半段实施方案不能直接照做。里面有些建议已经过期，有些会把当前工程路线带偏，还有少数会直接损坏 Qlib 数据。

| 优先级 | cc 文档问题 | 为什么有问题 | 解决方案 |
|---|---|---|---|
| P0 | 推荐“Qlib 官方 Yahoo 数据 + baostock 增量”作为主生产方案 | Qlib 官方示例 CN 数据适合研究基线，但 Yahoo 来源不一定完美，且离线包不能直接做增量；再拼 baostock 会混用复权口径、calendar 和字段尺度 | 生产链统一为 `TuShare Pro trade_date 批量 > AKShare 备选 > baostock repair`，全部先写 staging，health/smoke 过线后 promote |
| P0 | 用 `np.append(existing, new_values)` 追加 Qlib bin | Qlib feature bin 不是裸数组，当前正确格式是 `[start_index, values...]`；裸 append 会造成读数错位、NaN 或低覆盖 | 删除该方案；统一使用 `read_feature_series -> merge -> write_feature_bin`，由 `write_feature_bin()` 写 start index |
| P0 | 把多线程 baostock 当首次全量加速主方案 | baostock 的 `login/logout` 更像全局会话；当前单进程分批重连都能遇到 `10002007: 网络接收错误`，多线程更容易互相干扰 | baostock 只做缺口修复；真要并发用进程级 shard + 独立 staging，或直接换 TuShare/JQData 批量源 |
| P1 | 把 AKShare spot 近似写入训练日线 | `stock_zh_a_spot_em()` 适合盘中/盘后快照缓存，但没有统一复权、正式日线口径和历史补齐能力 | spot 只用于推文行情展示、动量因子和 17:05 缓存预热；训练日线必须来自正式日线接口 |
| P1 | 排除 VectorBT，理由是“当前不做系统化回测” | 当前最缺的正是验证 9:20、14:30、22:00 推荐是否有效；没有回测闭环，推文质量无法量化 | VectorBT 升为 P1，用来快速评估胜率、收益、回撤、换手；A 股 T+1/涨跌停/停牌再交给 RQAlpha 或自建撮合 |
| P1 | FinGPT 描述过于乐观，且版本/模型混乱 | 文档说 FinGPT v3.1 ChatGLM2-6B，但示例默认模型是 `fingpt-sentiment_llama2-13b_lora`；本地部署成本、显存、输出稳定性都被低估 | 不直接替代 MiniMax；先抽象 `SentimentProvider`，让 MiniMax/FinGPT/SnowNLP 同跑样本集，评估准确率和稳定性后再切 |
| P1 | SnowNLP 被描述成半天即可提升准确率 | SnowNLP 默认情感模型偏通用中文评论，不是金融/股吧语料，直接用于 A 股容易“自信地误判” | 先作为低权重 baseline；用股吧、公告、财经新闻样本重训或校准后再纳入正式评分 |
| P1 | Triple-barrier 改造被估成 1 天 | 在 Qlib handler 里换 label 不是加一个 Python 函数这么简单，还要处理 MultiIndex 对齐、防泄漏、分类/回归目标变化、walk-forward 验证 | 保留当前收益率预测主线；新增 meta-labeling 实验分支，先做离线评估，再决定是否替换训练目标 |
| P2 | 建议在普通 Stage 1 里额外计算 `monster_score` | 妖股信号是投机情绪模型，和普通短中长线 alpha 风险结构不同，混排会污染综合推荐 | 妖股池单独建模、单独展示、单独风控；普通综合前五不和妖股混排 |
| P2 | `INSERT OR REPLACE` 问题已经部分过期 | 当前代码已改成 `INSERT OR IGNORE`，但 `(rec_date, code)` 唯一约束仍无法保留 9:20、14:30、22:00 多次预测快照 | 推荐记录增加 `run_slot/source/prediction_id`，保留每次推送，用于收盘复盘和训练样本沉淀 |

对 cc 文档的处理建议：

- **保留**：量化库/舆情库生态清单、妖股信号维度、旧设计文档与现实差距审查。
- **修正**：数据下载主路线、FinGPT/SnowNLP 预期、Triple-barrier 工程量、VectorBT 优先级。
- **删除或降级**：`np.append` 追加 Qlib bin、AKShare spot 写训练日线、多线程 baostock 作为主加速、FinGPT 直接替代 MiniMax。

落地顺序应以当前生产风险为准：

1. 先保证全 A 数据覆盖和 4500+ 有限 LGB 预测。
2. 再做 spot 缓存、推文模板、收盘复盘。
3. 然后接 VectorBT/RQAlpha，把推荐质量变成可量化指标。
4. 最后再推进 FinGPT、Triple-barrier、RL 仓位决策器和妖股模型。

### 4.7 cc 文档二次审查：新增确认的问题

第二次细读时发现，`cc-量化与舆情深度调研.md` 已经扩展到 1200+ 行，后面新增了“问题19-24”。这些新增问题里有几条是实打实的代码死角，也有几条需要重新表述，避免变成错误修复方向。

本地验证结果：

- 当前 conda `tianshou` 环境里 `qlib` 版本是 `0.9.7`，`importlib.util.find_spec("qlib.run")` 返回 `None`，所以 `python -m qlib.run.get_data` 这条命令确实不可用。
- AKShare 当前环境里存在 `stock_comment_em`、`stock_comment_detail_zlkp_jgcyd_em`、`stock_zt_pool_em`、`stock_zt_pool_previous_em`、`stock_zt_pool_strong_em`、`stock_board_concept_name_em`、`stock_board_concept_cons_em` 这些接口名。
- `factors/quant.py` 仍然使用 `region_type="cn"` 和写死 `end_time="2026-05-01"`，而主训练链 `models/short_term.py`、`scripts/train_lgb.py` 已经使用正确的 `region=REG_CN`。
- `main.py --setup` 仍会调用 `factors.quant.prepare_qlib_data()`，所以这个死角不是完全无害；首次 setup 会踩到错误下载命令。

新增问题和处理意见：

| 优先级 | 位置/说法 | 判断 | 解决方案 |
|---|---|---|---|
| P0 | `factors/quant.py` 用 `python -m qlib.run.get_data` | cc 指认正确；当前 Qlib 0.9.7 没有 `qlib.run` 模块，`main.py --setup` 会失败 | 改为 `from qlib.tests.data import GetData`；但只作为研究/初始化路径，不作为生产数据链 |
| P1 | `factors/quant.py` 用 `region_type="cn"` | cc 指认正确；Qlib 会提示 unrecognized config，应该统一为 `region=REG_CN` | 修 `init_qlib()`，并给 `prepare_qlib_data()` 加 smoke test |
| P1 | `get_alpha158_handler()` 写死 `end_time="2026-05-01"` | cc 指认正确；虽然主链不用它，但会误导后续调用者 | 改为 `end_time: str | None = None`，默认取当天日期或最近交易日 |
| P1 | 网络测试没有标记 | cc 指认基本正确；`test_market_collector.py` 和 `test_sentiment_collector.py` 会直接访问 AKShare/雪球/东财 | 加 `@pytest.mark.network`，默认单元测试 mock 网络；把真实联网测试放到显式命令里跑 |
| P1 | 情绪采集仍依赖雪球和东财 HTML 正则 | cc 指认正确；`SentimentCollector.fetch_xueqiu()` 靠非登录请求，`fetch_eastmoney()` 靠 HTML 正则 | 短期改 AKShare 结构化接口；中期落 `event_impacts`；长期再评估 SnowNLP/FinGPT |
| P2 | `DATA_CUTOFF_TIME` 定义但未使用 | cc 指认正确，但它不是当前最大风险 | 要么删除该配置；要么在推文里明确展示“行情/新闻/模型数据截止时间” |
| P2 | pushplus 返回码和推送限额 | 返回码问题偏猜测，当前测试按 `code == 200`；真正问题是没有推送配额和去重控制 | 保留返回码测试；新增每日推送计数、同股票同类型告警去重、超限降级为日志 |
| P2 | pyproject 依赖声明 | cc 的问题意识对，但“全部放主依赖”不一定适合；用户实际用 conda `tianshou` 环境跑 | 更好方案是维护 `environment.yml` 或 `requirements-train.txt`，并让 `main.py`/scheduler 启动时检查 qlib、lightgbm、torch、tianshou |
| P2 | `SignalScorer` 权重和文档不一致 | cc 指认正确；设计、MVP、实际代码三套权重会让后续维护混乱 | 在 `signals/scorer.py` 写清当前权重来源；更进一步，把权重配置化并记录到每次推荐 payload |
| P3 | backtest 模块空壳 | cc 指认正确；但不建议直接删，因为当前正需要回测闭环 | 暂时标记 experimental；先接 VectorBT 做轻量验证，再决定保留/替换旧 backtest 模块 |

cc 文档还存在几个“文档质量问题”：

1. **问题编号错乱**  
   新增的“问题19-24”插在“问题17/18”前面，说明文档是拼接式更新。后续不适合作为唯一执行计划，应该用本 `cx` 文档承接决策。

2. **前后自相矛盾**  
   前面说 `GDELT` “已在用”，后面又承认 `GDELTCollector/GeopoliticalScorer` 没接 pipeline。前面列 `VectorBT` 为重要回测框架，后面又排除它。执行时必须以代码现状和当前需求为准。

3. **英文工具误用于中文舆情**  
   `VADER`、`TextBlob` 可以当英文社媒 baseline，不适合直接分析股吧/微博中文短文本。中文短文本应优先考虑 SnowNLP 重训、PaddleNLP/ERNIE、LTP/HanLP、LLM 结构化抽取。

4. **FinRL/Qbot/AlphaPy 优先级偏高**  
   当前瓶颈不是“缺更多模型框架”，而是全 A 数据覆盖、LGB 有限预测、回测复盘、推文一致性。FinRL/Qbot/AlphaPy 只能作为实验参考，不能进入近期主线。

5. **Wind/JQData 不能简单按成本排除**  
   当免费源导致全 A 回补十几个小时、session 掉线、复权口径不稳时，稳定付费数据反而可能更便宜。应以“节省的维护时间 + 数据质量 + 可增量能力”评估，而不是只看 API 是否付费。

6. **2025/2026 论文条目缺少可执行来源**  
   `TGNS`、`CNN-LSTM-GNN`、`Financial ML: An Engineering Problem` 这类条目没有给出论文链接、数据集、代码或可复现实验，不应进入开发优先级。可以放研究备忘，不进入迭代计划。

更新后的取舍：

- **立刻修**：`factors/quant.py` 的 setup 路径、Qlib `region` 参数、写死日期、联网测试标记。
- **近期做**：情绪采集结构化、推送配额/去重、SignalScorer 权重记录、推荐快照 schema。
- **暂缓**：FinRL/Qbot/AlphaPy、FinGPT 本地替代 MiniMax、Triple-barrier 直接替换 LGB label、英文社媒工具接中文股吧。

### 4.8 从 cc 文档吸收的部分，以及问题证据增强

这部分把 `cc-量化与舆情深度调研.md` 里的可吸收内容和需要否决/修正的内容分开。原则是：只吸收能直接改善当前系统稳定性、可验证性、推荐质量的内容；凡是可能破坏数据口径、增加随机性、或绕开当前最大瓶颈的建议，都必须先给证据和替代方案。

#### 可吸收项

| cc 内容 | 是否吸收 | 吸收方式 | 证据/理由 |
|---|---|---|---|
| 涨停板数据接入 | 吸收 | 新增 `LimitUpCollector`，但只服务妖股池和板块热度，不混入普通综合前五 | 本地 AKShare 环境已验证存在 `stock_zt_pool_em`、`stock_zt_pool_previous_em`、`stock_zt_pool_strong_em` |
| 板块热度追踪 | 吸收 | 用概念成分股 + 涨停池交叉计算 `sector_heat`，作为短线/妖股解释因子 | 本地 AKShare 环境已验证存在 `stock_board_concept_name_em`、`stock_board_concept_cons_em` |
| 东方财富/雪球采集脆弱 | 吸收 | 短期替换 HTML 正则和雪球裸请求，优先用 AKShare 结构化接口；输出统一进入 `event_impacts` | 当前 `data/collectors/sentiment.py` 中 `fetch_xueqiu()` 依赖非登录请求，`fetch_eastmoney()` 依赖 `re.findall` HTML 正则 |
| GDELT/地缘旧代码未接入 | 吸收其审查结论 | 保留 MiniMax/LLM 为主；GDELT 如使用，只做宏观 tone/time series 补充，不直接荐股 | 当前 `data/collectors/gdelt.py` 和 `factors/geopolitical.py` 存在，但主 pipeline 用 `signals/llm_analyst.py` |
| 推荐记录会丢多时段快照 | 吸收并升级 | 不是只改 `INSERT OR IGNORE`，而是给推荐记录增加 `run_slot`、`source`、`prediction_id` | 当前 `tracker/verifier.py` 已用 `INSERT OR IGNORE`，但唯一键仍是 `(rec_date, code)`，无法区分 9:20/14:30/22:00 |
| pushplus 推送缺少频控 | 吸收 | 增加每日推送计数、同股票同类型告警去重、超过阈值转日志 | 当前 `config/settings.py` 有 `MAX_PUSH_PER_STOCK_PER_DAY`，但 `scheduler/jobs.py`/`push/wechat.py` 没有使用它 |
| 依赖/运行环境不一致 | 吸收 | 不简单把所有东西塞进主依赖；维护 conda `tianshou` 的 `environment.yml` 或 `requirements-train.txt`，并加启动前依赖检查 | 当前用户明确用 conda `tianshou` 跑；`pyproject.toml` 把 qlib/torch/tianshou 放 optional，容易出现 `No module named qlib` |
| backtest 空壳 | 吸收但不删 | 标记旧 backtest 为 experimental，先引入 VectorBT 快速验证推送信号 | 当前系统最缺“推荐有效性验证”，VectorBT 比继续维护空壳更直接 |

#### 有问题项的证据增强

| 问题 | 证据 | 结论 | 替代方案 |
|---|---|---|---|
| Qlib 官方数据 + baostock 拼接作为生产主链 | cc 自己写该数据来自 Yahoo 且截止 2020-09-25；当前项目已经有自己的 Qlib 写入、manifest、staging、health gate；混拼会新增复权口径和 calendar 差异 | 不作为生产主链 | 统一源链：TuShare 批量日线优先，AKShare 备选，baostock 只 repair |
| 裸 `np.append` 追加 bin | 当前 `scripts/update_qlib_data.py` 的 `write_feature_bin()` 明确写 `[start_index, values...]`；`save_to_qlib_format()` 会先读已有 series、merge，再重写 bin | cc 的 append 方案会破坏 Qlib bin 格式 | 只允许 `read_feature_series -> merge -> write_feature_bin` |
| 多线程 baostock 作为主加速 | 当前后台日志已经在单进程场景出现 `10002007: 网络接收错误`，并需要分批 reconnect；baostock session 不是为线程内反复 login/logout 设计的稳定批量源 | 不做主方案 | 单进程断点续跑 + 重连；或进程级 shard；长期换 TuShare/JQData |
| AKShare spot 写训练日线 | 当前 `data/collectors/market.py` 已把 spot 定义成磁盘缓存，用于推文/实时行情；spot 缺复权、缺正式日线口径、无法补历史 | 不能写入训练 Qlib 日线 | 17:05 预热 spot cache；训练数据走日线接口 |
| FinGPT 直接替代 MiniMax | cc 文档说 v3.1 ChatGLM2-6B，但示例模型写 `FinGPT/fingpt-sentiment_llama2-13b_lora`；模型版本、底座、显存、输出稳定性都没在本项目验证 | 不能直接替代 | 先做 `SentimentProvider` 抽象和离线评测集，MiniMax/FinGPT/SnowNLP 同跑比较 |
| SnowNLP 半天提升准确率 | SnowNLP 默认情绪模型不是 A 股金融语料；当前股吧语境里“炸板、核按钮、反包、地天板”等词会让通用模型误判 | 不能高权重上线 | 先低权重 baseline，再用股吧/公告/财经新闻样本重训或校准 |
| VADER/TextBlob 分析股吧/微博 | VADER/TextBlob 主要面向英文文本；cc 将其放到中文股吧/微博语境不合适 | 不进入中文舆情主线 | 中文用 SnowNLP 重训、PaddleNLP/ERNIE、LTP/HanLP、LLM JSON |
| Triple-barrier 直接替换 LGB label | Qlib handler 标签涉及 MultiIndex、时序窗口、防泄漏、目标类型；不是加一个 `triple_barrier_label()` 函数即可 | 不直接替换主训练目标 | 新增 meta-labeling 实验分支，离线 walk-forward 评估通过后再升级 |
| FinRL/Qbot/AlphaPy 提高优先级 | 当前 `all.txt` 只有 280 行，LGB 有限预测目标是 4500+；模型框架不是当前瓶颈 | 暂缓 | 先做数据覆盖、预测缓存、回测复盘、推文一致性 |
| 仅因付费排除 Wind/JQData | 免费源当前导致全 A 回补慢、会话不稳、维护成本高；付费源如果减少维护和失败，实际成本可能更低 | 不能简单排除 | 以“增量能力、复权质量、稳定性、维护时间”评估数据源 |
| 2025/2026 论文条目直接进路线 | cc 对 `TGNS`、`CNN-LSTM-GNN`、`Financial ML: An Engineering Problem` 等条目没有给直接论文链接、代码或复现实验 | 只做研究备忘 | 进入开发前必须补论文链接、数据集、代码、可复现实验和本项目适配成本 |

#### 证据清单

- 本地环境验证：`/Users/wangzilu/miniconda3/envs/tianshou/bin/python -c "import qlib; import importlib.util; ..."` 显示 `qlib==0.9.7` 且 `qlib.run` 不存在。
- 本地 AKShare 验证：`stock_comment_em`、`stock_comment_detail_zlkp_jgcyd_em`、涨停池和概念板块接口名在当前环境存在。
- 本地数据覆盖：`data/storage/qlib_data/cn_data/instruments/all.txt` 当前只有 280 行，说明全 A 覆盖仍是第一瓶颈。
- 运行日志证据：`logs/all_share_nightly_train.log` 里单进程 baostock 已出现 `10002007: 网络接收错误`，说明多线程 baostock 风险不是理论问题。
- 代码证据：`factors/quant.py` 仍含 `region_type="cn"`、`end_time="2026-05-01"` 和 `python -m qlib.run.get_data`；`main.py --setup` 会调用它。
- 代码证据：`push/wechat.py` 只判断 pushplus 返回，未做每日配额；`config/settings.py` 的 `MAX_PUSH_PER_STOCK_PER_DAY` 未接入推送限流。
- 代码证据：`signals/scorer.py` 当前权重为 short 0.4、mid 0.3、sentiment 0.2、macro 0.1，和旧设计/MVP 文档不一致，需要记录到推荐 payload。

### 4.9 cc 第九章复读：对 CX 审查的吸收与反驳

`cc-量化与舆情深度调研.md` 新增了第九章“CX 文档审查”，这部分比前面的生态清单更有价值，因为它开始反向指出 `cx` 文档自身的问题。我的处理原则是：能降低当前工程风险的批评直接吸收；把风险判断推导成错误工程路线的地方，要保留风险、反驳路线。

#### 应该吸收的批评

| cc 批评 | 是否成立 | 吸收方式 |
|---|---|---|
| TuShare 不能作为唯一生产依赖 | 成立。2025-08 TuSharePro 停运事件说明单一数据源有商业/托管风险；积分/权限也需要配置成本 | 修改表述为“provider auto 优先使用可用的批量日线源”，而不是“无条件押注 TuShare”。生产必须多源 fallback + 本地缓存 + health gate |
| MiniQMT/xtquant 准入条件没写清 | 成立。它们通常依赖券商开户、终端、权限和本地运行环境 | 在数据源候选里标明“长期可选，需要券商/终端/权限”，不放入近期默认路线 |
| multi-bagger `max_forward_return` 有前视/可交易性偏差 | 成立。未来最高价标签适合“雷达/发现潜力”，不适合直接训练可执行买卖收益 | 多倍股模型改成研究雷达：标签同时记录 close-to-close、triple-barrier、time-to-target、max drawdown；训练/验证必须 purged/embargo |
| PyABSA 不能开箱即用中文金融 | 成立。可作为方面级情绪框架参考，但中文金融语料需要标注和评测 | 从“推荐实施库”降级为 P3 研究项；第一版用 LLM JSON + LTP/HanLP/PaddleNLP 做实体和事件抽取 |
| 三份 cx 文档有重复 | 成立。当前 `cx-quant-sentiment-deep-research`、本文件、`cx-v2-iteration-plan` 有大量重叠 | 后续以本文件做研究主文档，以 `cx-v2-iteration-plan` 做执行计划；旧调研只保留归档 |
| US/HK 扩展过早过重 | 基本成立。用户问过美股/港股需要做什么，所以保留方案合理，但不应抢 Phase 0/1 资源 | US/HK 移到 Phase 3+，当前只保留“市场隔离、单独模型、单独验证”的设计原则 |
| FinRobot 定位偏模糊 | 成立。它适合作研究/报告 agent 参考，不是近期生产依赖 | 从实施优先级降级为参考项目；不进入 P0-P2 |
| entry/stop/take-profit/invalidation 字段有价值 | 成立。它能把推荐从“看多”升级为可复盘交易计划 | 推荐快照 schema 增加 `entry_zone`、`stop_loss`、`take_profit`、`invalidation_condition`、`drivers`、`risks` |
| 妖股四象限比单一分数更可操作 | 成立 | 妖股池输出改为“潜伏型/加速型/兑现型/排除型”，而不是只给 `monster_score` |

#### 需要反驳或修正的结论

| cc 结论 | 问题 | 修正后的判断 |
|---|---|---|
| “TuShare 有停运/积分风险，所以日常增量应首选 baostock” | 风险判断对，但结论错。baostock 当前全 A 串行慢，日志已经出现 `10002007: 网络接收错误`；它不适合做全市场日常主源 | 正确路线是多源自动选择：TuShare/AKShare/JQData/本地 vendor 谁能批量、稳定、通过 health gate 就用谁；baostock 只做缺口 repair |
| “AKShare 免费稳定，有批量接口 `stock_zh_a_spot_em()`，应做日常增量主源” | `stock_zh_a_spot_em()` 是 spot 快照，不是训练日线；cc 自己也记录 AKShare spot 连接失败 | AKShare spot 只做推文/动量/缓存；训练日线必须走可复权、可补历史、可对齐 calendar 的日线接口 |
| “Qlib 官方 Yahoo 数据 + baostock 多线程补齐是确定可跑通路径” | 可以作为研究 bootstrap，但不适合生产主链。Yahoo/baostock 混用会引入复权口径和字段差异，多线程 baostock 也不稳定 | 官方数据只可临时 bootstrap；生产必须统一口径、staging、health/smoke、manifest |
| “TuShare 5000+ 积分才能用日线/日常数据” | 需要更精确。TuShare 官方日线行情文档显示 120 积分起可调用，5000 积分是更高频次/更高权限档，不应混为最低门槛 | 文档应写“TuShare 需要 token/积分/权限，权限档影响频次和可用字段”，而不是简单说新用户不可用 |
| “LGB ≥100 predictions 是当前目标” | 已过期。当前 scheduler 和数据健康目标已经提升到全 A，`LGB_MIN_PREDICTIONS=4500` | 任何新数据方案都必须以 4500+ 有限预测为验收，不再按 100 只判断 |
| “Triple-barrier Phase 2 直接替代简单前向收益” | 方向对，但太快。标签改造会影响模型目标、阈值、回测和推文解释 | 先做 meta-labeling/实验分支，不能直接替换主 LGB label |

#### 修正后的数据源原则

1. **不押单一数据源**  
   TuShare、AKShare、JQData、本地 vendor、baostock 都可能失败。生产链的核心不是“谁永远最好”，而是 provider auto、失败降级、缓存、manifest、staging、health gate。

2. **训练日线和 spot 快照严格分离**  
   `stock_zh_a_spot_em()` 快，但只能做行情快照、推文展示、当日动量和兜底候选，不能直接写 Qlib 训练日线。

3. **baostock 不做全 A 日常主源**  
   它适合修少数缺口和校验特殊日期，不适合每天重新扫 5000+ 标的。

4. **官方 Qlib 数据只做研究 bootstrap**  
   可以用来快速检查 Qlib pipeline，但生产模型要基于统一、可增量、可健康检查的数据链。

5. **长期 vendor 数据要写准准入成本**  
   MiniQMT/xtquant/通达信本地缓存等可以作为长期稳定方案，但前提是用户有终端、券商权限、数据订阅和本地自动化环境。

#### 修正后的路线图

| 阶段 | 目标 | 调整 |
|---|---|---|
| Phase 0 | 守护脚本、staging、LGB smoke、状态可见 | 已基本落地；继续补 `factors/quant.py` setup 死角和联网测试标记 |
| Phase 1 | 全 A 数据覆盖 + 快速推荐稳定 | 不押 TuShare；实现 provider auto、多源 fallback、manifest resume、4500+ 有限预测 |
| Phase 2 | 推文可交易化 + 复盘入库 | 加 `entry_zone/stop_loss/take_profit/invalidation_condition`，保存 9:20/14:30/22:00 快照 |
| Phase 3 | 回测和事件修正 | VectorBT/RQAlpha、event_impacts、舆情结构化 |
| Phase 4 | 妖股雷达和组合层 | 妖股四象限、板块热度、PyPortfolioOpt/HRP |
| Phase 5 | 多市场和研究型模型 | US/HK、FinRobot、PyABSA、FinGPT 本地化、RL 序列模型 |

### 4.10 cc 文档一致性审计：未收敛问题与论据

这轮重新读 `cc-量化与舆情深度调研.md` 后，我把问题分成两类：一类是 cc 指出的真实工程风险，应该吸收；另一类是 cc 自己内部仍然摇摆，或者证据已经被当前代码反驳，不能直接照单全收。

| 论点 | 矛盾或未收敛点 | 证据 | 结论与修法 |
|---|---|---|---|
| 收盘后数据、训练、smoke 必须串行 | cc 对 crontab 竞态的批评成立，而且是当前最危险的运行问题 | `scripts/install_crontab.py:50-69` 把 17:00 数据更新、17:35 LGB 训练、17:55 smoke 拆成三个独立 cron；`scripts/nightly_train.py:98-140` 反而是正确的串行失败即停止；`logs/data_update.log` 显示 2026-05-08 17:00 数据更新失败，`logs/lgb_after_close_train.log` 仍在 17:35 训练并保存模型 | 把 17:00-17:55 合并成 `after_close_pipeline.py`：update -> health -> train -> smoke -> cache，任一步失败就停止；或至少用同一把 `flock` 锁和状态文件阻断下游 |
| `provider auto` 还没真正成为生产闭环 | cc 说“写了多源 fallback，但实际没走完”有现实证据，不过它把 baostock 重新抬成主源的结论不对 | 当前 `fetch_data()` 已按 TuShare -> AKShare -> baostock 排序，但旧日志显示 AKShare 全失败后直接 `No data fetched`；当前代码 `fetch_with_akshare()` 已加低成功率抛错让 auto fallback，但 cron 仍用 `--universe-source baostock`，且没有把 provider 选择、成功率、stale 缓存写成验收标准 | 日线主链应是“批量日线 provider auto + 本地缓存 + manifest + staging + health gate”，不是单押 TuShare、AKShare 或 baostock。baostock 只做缺口修复和兜底慢链 |
| AKShare spot 不能写训练日线 | cc 前面承认 `stock_zh_a_spot_em()` 没有历史和复权，后面又建议它做日常增量主源，口径冲突 | cc §8.5 写 spot 缺少前复权、只有当天、可能非精确收盘价；但 §9.2 又建议 AKShare spot 做日常增量主源 | spot 可以做 17:05 缓存、推文展示、当日动量、紧急 fallback；训练日线必须来自可复权、可补历史、可对齐交易日历的日线接口 |
| Qlib bin 格式争议应以本地 Qlib 版本收敛 | cc 继续声称官方数据可能是裸数组，并推断 `[start_index, values...]` 会让 Qlib 读错；这个判断与本地 Qlib 0.9.7 代码不符 | 本地 `/Users/wangzilu/miniconda3/envs/tianshou/bin/python` 验证 `qlib==0.9.7`；`FileFeatureStorage.start_index` 读取第一个 float，`__getitem__` seek 时跳过 4 字节头，`write()` 新文件写 `np.hstack([index, data_array])`；本地 `close.day.bin` 第一项是 `4944.`，后面才是价格 | 对当前环境，`[start_index, values...]` 是正确格式。cc 的裸 `np.append(existing, new_values)` 方案应删除。若未来引入官方包或不同 Qlib 版本，先写版本化转换器，而不是混写两种格式 |
| LGB 验收口径还有 100 与 4500 的漂移 | cc 还在用“≥100 predictions”做路线目标，当前配置已经提升到全 A；但日志又显示 17:35/17:55 曾以 100 或 280 通过 | `config/settings.py:43-45` 是 `LGB_MIN_PREDICTIONS=4500`、`LGB_MIN_DATA_INSTRUMENTS=4500`；cc §9.6 仍写 `smoke 通过 ≥100 predictions`；`logs/lgb_after_close_smoke.log` 显示 280 finite predictions 也写入 cache | 统一验收：生产推荐必须 4500+ 最新有限预测；研究/临时 smoke 可允许 100，但必须标注 `research_only`，不能写生产 cache |
| 长线推荐还不是长线模型 | cc 对“长线概念混淆”的批评成立，当前实现只是长周期观察榜 | `scheduler/jobs.py:410-463` 的长线分数来自 `model_score * 0.45 + liquidity_score * 0.35 + change_pct * 0.20`；`scheduler/jobs.py:246-322` 的长线桶来自 `final_score/macro/sentiment`，没有 ROE、利润、估值、现金流等长期因子 | 短期把“长线”改名或标注为“观察榜”；中期接入财报/估值/机构持仓/行业景气，训练独立 3-12 月模型；在此之前不能给用户长期持有暗示 |
| 下一个交易日不能只跳周末 | cc 指出 `next_weekday()` 的节假日问题成立，当前只是 first pass | `signals/index_predictor.py:51-56` 只跳过周六周日，不处理春节、国庆、调休、临时休市 | 改成 `next_trading_day()`，优先读取 Qlib calendar 或交易所日历；预测、复盘、推文标题都用同一个交易日函数 |
| verification 旧问题已经部分过期 | cc 继续沿用“验证窗口还是 latest 10 bars”的旧批评，但当前代码已经改过 | `scheduler/jobs.py:1440-1458` 现在按 `df.index > rec_dt` 取推荐日之后的 `PREDICTION_HORIZON_DAYS` 根 bar；`tracker/verifier.py:144-160` 不再用 `today - 7 calendar days` 锁死结果 | 这条不应再作为当前缺陷。剩余问题是：交易日历函数要统一，验证报告应把实际 bar 日期写入快照，便于审计 |
| GDELT 的定位已经在 cx 文档里收敛，但代码仍未接 | cc 说 cx 对 GDELT 立场矛盾，部分成立；本文件已收敛成“宏观 tone/time series 补充”，但实现还缺口 | `scheduler/jobs.py:1170-1189` 当前宏观/地缘走 `llm_analyst.analyze_geopolitics()`；`data/collectors/gdelt.py` 存在但主 pipeline 未调用 | 文档结论保持：LLM 是解释与事件结构化主链，GDELT 只做可回测的宏观 tone 序列。下一步要么接入 `event_impacts`，要么标注 deprecated，不能继续“已在用” |
| VectorBT 与 FinRL 的优先级写法摇摆 | cc §1 把 FinRL 写高优先级、§5 写 P4；§6 排除 VectorBT，§9.6 又把 vectorbt 放 Phase 2 | 早期库清单、排除表、最终路线图三处口径不同 | 收敛为：VectorBT/QuantStats 是 P1 验证工具，先用于信号回测；FinRL/TradeMaster/TensorTrade 是 P3+ 研究对照，等数据、回测、交易成本模型稳定后再上 |
| cc 文档自身编号和结论重复 | cc 新增第九章后出现 `CX问题15-18`，随后又重复 `CX问题12-14` 和 `CX问题8`，读者很难判断最终结论 | `cc` 文档 1373 行后新增问题，1472 行后重复较早问题，末尾路线图仍写 100 predictions | 建一个“决策台账”：每个议题只保留 `采纳/否决/待验证`、证据、负责人、验收标准。调研材料可长，但执行文档必须唯一收敛 |

#### 这轮应立即吸收的改动

1. 先做 `after_close_pipeline.py`，用串行依赖替代 17:00/17:35/17:55 三个独立 cron。
2. 把所有生产 LGB 验收统一到 4500+ 最新有限预测，禁止 280 或 100 的 smoke 写生产 cache。
3. 用交易日历替换 `next_weekday()`，保证 22:00、09:20、14:30、收盘复盘都指向同一个真实下一个开盘日。
4. “长线推荐”在没有基本面模型前改成“长线观察榜”，避免给用户长期持有的错觉。
5. Qlib bin 格式按本地 0.9.7 的 `FileFeatureStorage` 收敛，删除裸 append 方案。

### 4.11 为什么 cc 的几个关键结论不成立

这里不是为了“站队”，而是把论证链拆开看：对方有些观察是对的，但从观察推出的工程结论不对。生产系统不能只看“哪个数据源免费/哪个方案最快”，要看是否满足四个硬条件：可重复、可审计、可健康检查、失败时不污染生产模型。

| cc 结论 | 为什么不对 | 证据 | 正确结论 |
|---|---|---|---|
| TuShare 有停运/积分风险，所以 baostock 应做日常主源 | 这是从“TuShare 不能单点依赖”跳到了“baostock 可以当主源”，中间少了吞吐、稳定性、失败恢复的证明。TuShare 有风险，只能推出“不能单押 TuShare”，不能推出“应该单押 baostock” | cc 自己在 §8.1 写 baostock 是逐只串行请求，800 只约 2.5 小时，全 A 约 16 小时；本项目日志 `logs/all_share_nightly_train.log` 出现过 `10002007: 网络接收错误`；`scripts/update_qlib_data.py:593-650` 需要分批 reconnect 才能维持会话 | 日常主链应是 `provider auto`：优先可批量日线源，失败 fallback，结果写 staging，health/smoke 过线才 promote。baostock 只能是 repair/慢兜底 |
| AKShare `stock_zh_a_spot_em()` 可以做日常训练增量 | 这是把“行情快照”当成“训练日线”。spot 快照快，但它不是复权日线，也不能补历史，更不能保证最终收盘后字段口径和 Qlib 训练字段一致 | cc §8.5 自己承认 spot 缺少前复权、只有当天、可能非精确收盘价；当前代码里训练日线用 `ak.stock_zh_a_hist(..., adjust="qfq")`，不是 `stock_zh_a_spot_em()`；本文件 §4.4 已把 spot 定位成 17:05 缓存和推文行情 | spot 只做展示、盘中动量、流动性 proxy、紧急 fallback；训练数据必须来自正式日线接口，并经过 calendar 对齐、复权口径确认和 health gate |
| Qlib 官方 Yahoo 数据 + baostock 增量是确定可跑通的生产路径 | “能跑通”不等于“适合生产”。Yahoo 基础包、baostock 增量、本地 TuShare/AKShare 增量混在一起，会引入复权口径、字段尺度、calendar/instrument 差异。即便短期不 NaN，也可能让模型学到混合口径噪声 | cc §8.3 写官方 Yahoo 数据只到 2020-09-25；本文件 §4.4 引用 Qlib 官方示例数据“不一定完美”；当前项目已经实现 `manifest + staging + health + smoke`，生产链应围绕自己的数据口径建立，而不是把示例包当底座 | 官方数据可以做 bootstrap、环境验证、Qlib pipeline smoke；生产训练主链必须统一数据来源和复权口径，所有数据更新都走 staging 和 health/smoke |
| Qlib feature bin 可能是裸 float32 数组，`[start_index, values...]` 可能会读错 | 这个判断直接被本地 Qlib 0.9.7 源码反驳。不是观点差异，是当前运行环境的读写协议问题 | 本地 conda 环境验证 `qlib==0.9.7`；`FileFeatureStorage.start_index` 从文件前 4 字节读第一个 float；`__getitem__` 读取数据时 `seek(... + 4)` 跳过头；`write()` 新文件写 `np.hstack([index, data_array])`；本地 `close.day.bin` 第一项是 `4944.`，后面才是价格 | 当前环境必须写 `[start_index, values...]`。裸 `np.append(existing, new_values)` 会破坏 offset。若将来换 Qlib 版本，应先写转换器和版本检查，不能混写 |
| LGB smoke 通过 100 个预测就能进入路线图 | 这是把早期 smoke 门槛当生产门槛。100 个预测只能证明模型没有完全坏，不能支撑“全 A 推荐” | `config/settings.py:43-45` 当前生产目标是 `LGB_MIN_PREDICTIONS=4500`、`LGB_MIN_DATA_INSTRUMENTS=4500`；cc §9.6 末尾仍写 `LGB 重训 + smoke 通过 ≥100 predictions`；历史 `logs/lgb_after_close_smoke.log` 曾出现 280 finite predictions 也写入 cache，说明低门槛确实会污染生产状态 | 100/280 只能标记为 `research_only` 或 `degraded`。生产推荐、RL 训练输入、推文个股榜必须要求 4500+ 最新有限预测 |
| crontab 用 `--universe-source baostock` 就说明 baostock 是行情主源 | 这里混淆了“股票列表来源”和“日线行情 provider”。这个观察能说明 cron 配置不收敛，但不能证明 baostock 已经是价格数据主源 | `scripts/install_crontab.py:55-64` 传的是 `--universe-source baostock`，没有显式传 `--provider baostock`；`scripts/update_qlib_data.py:782-807` 的价格数据 provider 是另一个参数，auto 时按 TuShare -> AKShare -> baostock；`logs/data_update.log` 显示实际运行是 `provider=auto` 且先尝试 AKShare | cc 指出的“cron 配置与文档不一致”成立，但“baostock 已经是行情主源”的解释不严谨。应该修 cron 为串行 pipeline，并在日志里明确 universe source、price provider、成功率 |
| 多线程 baostock 是首次全量加速主方案 | baostock 的瓶颈不是简单 CPU 并发，而是 API 会话、网络稳定性、服务端限流和失败重试。盲目多线程会把 session 和限流问题放大 | cc §8.4 自己写 `bs.login()` 是全局状态且 16 线程偶尔超时；本项目单进程都需要 `_reconnect()`，日志已有网络接收错误；全 A 数据写 Qlib 还要保证 staging 合并和 manifest 不被并发写坏 | 若必须并发，只能做进程级 shard、独立 staging、最后 merge + health；更优先的是接入按交易日批量拉全市场的源 |
| VectorBT 可以先排除，因为当前不做系统化回测 | 这和当前项目最缺的东西相反。现在最大风险不是“缺模型名字”，而是推荐到底有没有胜率、有没有回撤、9:20/14:30/22:00 哪个时点有效都没量化 | cc §6.1 把 VectorBT 排除，§9.6 又把 vectorbt 放 Phase 2，前后不收敛；当前已有推荐快照和验证需求，正好需要轻量回测先跑起来 | VectorBT/QuantStats 应进 P1 做快速验证；A 股 T+1、涨跌停、停牌等细规则再交给 RQAlpha 或自建撮合 |
| FinRL/组合级 RL 高优先级 | 数据、标签、回测、成本模型都没稳定前，RL 很容易学到数据漏洞或噪声。把 RL 提前不是增强判断，而是放大不稳定输入 | cc §1.1 写 FinRL 高优先级，§5 又写 P4；当前全 A 数据覆盖和 LGB 4500+ 仍是底座问题；`scripts/train_rl.py` 也还处在研究训练阶段 | FinRL/TradeMaster/TensorTrade 只能做 P3+ baseline。先把数据覆盖、回测、交易成本、舆情结构化做好，再让 RL 学“何时相信/不相信信号” |

#### 最核心的反驳

cc 最大的问题不是“发现的问题少”，而是经常把一个真实风险推成另一个未经证明的主方案：

1. TuShare 有单点风险，所以不能单押 TuShare；但这不等于 baostock 适合全 A 日常主源。
2. AKShare spot 很快，所以适合缓存和展示；但这不等于它可以写训练日线。
3. Qlib 官方数据能帮助 bootstrap；但这不等于它能和 baostock 增量混成生产主链。
4. 100 个预测能证明模型还活着；但这不等于可以支撑全 A 推荐。
5. RL 框架很强；但这不等于当前阶段应该把它放到数据和回测之前。

所以最终收敛口径是：**生产链以验收标准说话，不以库名、免费、速度或单次成功说话。** 只要没有 4500+ 最新有限预测、统一复权口径、真实交易日历、staging/health/smoke 通过，就不能进入正式推荐主链。

### 4.12 全 A 数据拉取专项复审：吸收 cc 的问题，但不能照搬 cc 的方案

这次重点重读 cc 文档的第八章后，我的结论更明确了：cc 对“全 A 下载慢”的痛点判断是对的，但它给出的 D+A 组合（Qlib 官方 Yahoo 数据 + baostock 增量）只能作为研究 bootstrap，不能直接作为生产主链。真正要解决的是“5000+ 只股票的可恢复、可验收、统一口径日线数据链”，不是单纯把某个下载接口换成另一个。

#### 当前事实

| 项目 | 当前状态 | 影响 |
|---|---|---|
| 生产 Qlib `data/storage/qlib_data/cn_data` | `all.txt` 只有 280 只，calendar 到 2026-05-07 | 不能支撑全 A LGB/RL，只能算 CSI300 级别覆盖 |
| staging Qlib `data/storage/qlib_data_staging/cn_data` | `all.txt` 352 只，calendar 到 2026-05-08 | 没有 promote 是对的，说明 health gate 挡住了半残数据 |
| 顶层 Qlib 示例包 `data/storage/qlib_data` | `all.txt` 3875 只，calendar 到 2020-09-25 | 可用于研究 smoke，但不是当前生产全 A 数据 |
| LGB 预测缓存 | `finite_prediction_count=280`，`min_predictions=100` | 这是过期低门槛缓存，不能代表全 A 可用 |
| 当前后台全 A baostock 任务 | 进程仍在，日志停在 `2026-05-08 23:34:22 Reconnecting baostock session` | baostock 长任务会卡住，不能叫“确定跑通” |
| 当前 conda `tianshou` 环境 | `akshare=True`，`baostock=True`，`tushare=False`，`TUSHARE_TOKEN=False` | TuShare-first 在当前机器上还不可执行 |

#### cc 这次应该吸收的地方

1. **不要每天全量重拉多年历史。** 这个判断完全正确。日常盘后只应补最近 1-5 个交易日，首次建库/大缺口回补才拉多年历史。
2. **必须有本地缓存。** cc 提到 parquet/HDF5 缓存是对的。当前脚本直接从 provider 拉到 Qlib bin，中间没有 raw daily cache；一旦 health gate 失败，成功拉到的 750 只也不能可靠复用，下一次又从头扫。
3. **当前 TuShare 方案还只是纸面方案。** 本机没有安装 `tushare`，也没有 token。把它写成“首选生产源”会误导执行。
4. **17:00/17:35/17:55 独立 cron 有竞态。** 数据没拉完或拉失败时，训练仍会继续跑旧数据，这会污染模型状态和推荐解释。
5. **全 A 首次拉取必须能断点续跑。** 这不是优化项，是 P0。没有断点，任何 5000+ 标的大任务都会被网络波动拖死。

#### cc 不能照搬的地方

| cc 方案 | 问题 | 正确处理 |
|---|---|---|
| Qlib 官方 Yahoo 数据 + baostock 增量做生产主链 | Yahoo/baostock/AKShare/TuShare 复权口径、字段尺度、calendar 都可能不同；混拼能避免 NaN，不等于能训练出可信 alpha | 官方包只做 bootstrap 和 smoke；生产数据必须统一复权口径，并在 health check 中抽样对账 |
| 多线程 baostock 作为全量加速主方案 | `bs.login()` 是全局会话风格，单进程已经出现 `10002007 网络接收错误` 和 reconnect 卡住；多线程会放大会话互踢和限流问题 | 如果并发，只做进程级 shard，每个 shard 独立 raw cache/staging/status，最后 merge + health |
| AKShare `stock_zh_a_spot_em()` 做盘后训练日线 | spot 是快照，不是复权历史日线；没有可靠前复权、历史补齐、停牌/复牌口径 | spot 只用于推文行情、盘中动量、17:05 缓存预热；训练日线必须用正式 daily/hist 接口 |
| 裸 `np.append(existing, new_values)` 追加 Qlib bin | 当前 Qlib 0.9.7 feature bin 需要 `[start_index, values...]` 头，裸 append 会破坏 offset | 继续使用当前 `write_feature_bin()` 逻辑；追加也必须先读 header/offset 再重写或版本化转换 |
| “baostock 确定能跑通” | 现在后台 baostock 任务已卡在 reconnect；它比 AKShare hist 更能拉到数据，但不具备长任务确定性 | baostock 只能作为慢兜底/repair，且必须加 login/reconnect timeout、no-output watchdog、shard resume |

#### 真正可落地的全 A 数据架构

```
provider daily/hist
  -> raw_daily_cache(parquet/sqlite, per provider, per symbol, per date)
  -> normalize/adjust/check
  -> qlib_staging
  -> qlib health + LGB smoke
  -> promote production qlib + production prediction cache
```

核心原则：

1. **raw cache 在 Qlib bin 前面。** 每只股票、每个交易日、每个字段先落 `raw_daily_cache`，记录 `provider/source_date/adjust/missing_reason`。Qlib bin 只是由 raw cache 生成的训练格式，不是唯一数据仓库。
2. **provider fallback 按 shard/缺口执行，不按整段一次性执行。** 现在 `fetch_data()` 是 TuShare -> AKShare -> baostock 整段切换；如果 AKShare 成功 112/5180，剩余 5068 只没有逐只 fallback。正确做法是：每个 shard 统计成功率，低于阈值立刻换源，只补 missing symbols/dates。
3. **首次 bootstrap 和日常增量分开。** 当前 `build_start_dates()` 对本地没有 feature 的新股票只回看 `new_symbol_days=365`，这对 Alpha158/LGB 的多年训练不够。首次全 A 建库应显式 `--bootstrap-missing-symbols --bootstrap-start 2020-01-01` 或 `--full-years 5/7`，日常增量才用最近 1-5 天。
4. **统一复权口径先于速度。** 当前 AKShare 用 `adjust="qfq"`，baostock 用 `adjustflag="2"`；TuShare 代码现在只是拿了 `adj_factor`，但没有把 OHLC 按 adj factor 折算成同一口径。TuShare 真上线前必须补复权转换和跨源抽样对账。
5. **长任务必须有三层超时。** 单 symbol 请求 timeout、baostock login/reconnect timeout、nightly/after-close no-output watchdog。现在 `query_history_k_data_plus()` 有 30 秒 alarm，但 `_login()`/`_reconnect()` 没有硬超时，正好解释了后台卡住的风险。
6. **生产 promote 只看验收，不看下载过程有多努力。** `all.txt >= 4500`、最新交易日覆盖率 >= 95%、OHLCV/factor 字段有限、LGB 最新有限预测 >= 4500，全部通过才 promote。否则只能写 raw cache 和失败报告，不能写生产 cache。

#### 分阶段落地方案

| 阶段 | 目标 | 具体动作 | 验收 |
|---|---|---|---|
| P0.1 止血 | 不再卡死、不再从头重拉 | 给 baostock `_login/_logout/_reconnect` 加硬超时；给 `nightly_train.run_step()` 加 no-output timeout；失败时落 status | 后台任务不会无日志卡 30 分钟以上 |
| P0.2 断点缓存 | 已成功拉到的数据可复用 | 新增 `raw_daily_cache`，每 50/100 只股票 flush；manifest 记录 symbol/date/provider/status | 中断后重跑只补缺口，不重新拉已成功 symbol |
| P0.3 分片拉取 | 全 A 大任务可控 | universe 切 100-200 只 shard；每 shard 独立 status 和临时目录；最后 merge staging | 单 shard 失败不影响其他 shard 成果 |
| P0.4 多源 repair | 提高覆盖 | 对每个 shard 的 missing symbols 用 AKShare hist、baostock、未来 TuShare/JQData 逐层 repair | staging 覆盖 4500+，失败列表可审计 |
| P0.5 复权对账 | 防止混源噪声 | 抽样 50 只股票比较 AKShare qfq、baostock adjustflag=2、TuShare adj_factor 折算后的 close/return | 跨源日收益误差在可解释阈值内 |
| P0.6 生产发布 | 让 LGB 一定用上 | raw cache -> Qlib staging -> health -> train -> smoke -> promote 串行执行 | `latest_finite_prediction_count >= 4500` |

#### 数据源最终口径

| 数据源 | 在当前方案里的位置 | 说明 |
|---|---|---|
| TuShare/JQData/本地 vendor | 最优的批量日线源，但必须先安装、配 token/权限、做复权对账 | 不再写成“默认可用”，而是“配置后优先” |
| AKShare `stock_zh_a_hist` | 免费备选日线源 | 当前网络不稳，需要 shard retry 和低成功率立即 fallback |
| baostock | 慢兜底和缺口 repair | 不能做无限长单进程主链；必须有 timeout、断点、分片 |
| AKShare `stock_zh_a_spot_em` / Tencent spot | 推文/盘中/缓存 | 不进训练日线 |
| Qlib 官方 CN/Yahoo 包 | 研究 bootstrap | 不直接 promote 到生产，除非明确标记 `research_only` 并完成复权口径审计 |

#### 立刻要改的代码点

1. `scripts/update_qlib_data.py:600-616`：给 baostock login/logout/reconnect 加 timeout，避免卡在 reconnect。
2. `scripts/nightly_train.py:55-72`：增加 no-output watchdog，例如 10-15 分钟无日志就 terminate，不能只靠 6 小时总 timeout。
3. `scripts/update_qlib_data.py:484-529`：新增 bootstrap-missing-symbols 模式；本地没有 feature 的全 A 股票不能只拉 365 天。
4. `scripts/update_qlib_data.py:710-779`：TuShare 上线前补 qfq/hfq 折算；不能只写 raw OHLC + `factor`。
5. `scripts/update_qlib_data.py:782-807`：把 provider fallback 从整段切换改成 shard/missing symbol repair。
6. `scripts/install_crontab.py:50-70`：把 17:00 数据、17:35 训练、17:55 smoke 合成 after-close 串行 pipeline。
7. `scripts/smoke_lgb_predict.py` 和 scheduler cache 写入路径：禁止 `min_predictions=100/280` 的结果写 `lgb_latest_predictions.json` 生产缓存。

#### 这次审查后的最终判断

cc 的“全 A 下载慢”诊断应吸收；cc 的“官方数据 + 多线程 baostock 就能解决生产全 A”不应照搬。当前最稳的路线是：**先把下载任务工程化成可断点、可分片、可回放、可对账的数据管线；短期用 AKShare/baostock repair 撑过全 A 建库，中期接入 TuShare/JQData/vendor 这类批量日线源；任何数据都必须经过 raw cache、staging、health、LGB smoke 后才进生产推荐。**

---

## 5. 推荐优先级

| 优先级 | 事项 | 推荐库/数据 | 理由 |
|---|---|---|---|
| P0 | 全 A 数据覆盖和模型预测缓存健康 | Qlib + TuShare/AKShare + baostock repair | 当前 4500+ 有限预测是系统底座 |
| P0 | 全 A 下载加速 | provider auto 批量日线 + 多源 fallback + staging + manifest | 不押单一数据源，解决 baostock 串行全量过慢和中断重跑 |
| P0 | 收盘后串行训练 pipeline | after_close_pipeline + health gate + smoke gate | 防止数据更新失败后仍用旧数据训练并写入生产模型/cache |
| P0 | 生产 LGB 验收统一为 4500+ | train_lgb + smoke_lgb_predict + lgb_cache | 100/280 只能算研究 smoke，不能支撑全 A 推荐 |
| P1 | 信号回测闭环 | vectorbt + QuantStats | 先证明推荐有无价值 |
| P1 | A 股交易规则回测 | RQAlpha | 解决 T+1、涨跌停、手续费、停牌 |
| P1 | 因子诊断 | Alphalens-reloaded | 看 LGB 分数是否有 IC/rank IC |
| P1 | 交易日历统一 | Qlib calendar / exchange calendar | 修正节假日、调休下的明日预测和复盘日期 |
| P2 | 仓位组合 | PyPortfolioOpt / Riskfolio-Lib | 从荐股升级到组合 |
| P2 | 舆情结构化 | FinBERT + SnowNLP重训 + LTP/HanLP + LLM | 让 22:00/09:20 修正可验证 |
| P2 | 细粒度事件冲击 | LLM JSON + LTP/HanLP/PaddleNLP + event_impacts | 把“情绪分”升级为公司/行业/宏观冲击表 |
| P2 | 妖股模型 | 涨停板、龙虎榜、资金流、股吧热度 | 5 倍股/10 倍股必须单独模型 |
| P2 | 真长线模型或改名观察榜 | 财报/估值/机构/行业景气 | 当前长线分数由短线模型派生，不能当长期价值判断 |
| P3 | 多市场扩展 | OpenBB + Lean/NautilusTrader | 美股/港股/加密跨市场时再做 |
| P3 | 研究归因助手 | FinRobot-style workflow / PyABSA research | 只作研究和解释参考，不进近期生产主链 |
| P3 | RL baseline | FinRL/TradeMaster/TensorTrade | 用作实验对照，不先上生产 |

---

## 6. 参考来源

- Qlib: <https://github.com/microsoft/qlib>
- Qlib paper: <https://www.microsoft.com/en-us/research/publication/qlib-an-ai-oriented-quantitative-investment-platform/>
- RD-Agent: <https://github.com/microsoft/RD-Agent>
- TuShare A 股日线行情: <https://www.tushare.pro/document/2?doc_id=27>
- 财联社 TuSharePro 停运报道: <https://www.cls.cn/detail/2125736>
- FinRL: <https://github.com/AI4Finance-Foundation/FinRL>
- FinGPT: <https://github.com/AI4Finance-Foundation/FinGPT>
- FinGPT official site: <https://fingpt.io/>
- FinGPT paper note: <https://ai4finance.org/research/fingpt-open-source-finllm.html>
- FinRobot: <https://github.com/AI4Finance-Foundation/FinRobot>
- TradeMaster: <https://github.com/TradeMaster-NTU/TradeMaster>
- TensorTrade: <https://github.com/tensortrade-org/tensortrade>
- vectorbt: <https://vectorbt.dev/>
- RQAlpha: <https://github.com/ricequant/rqalpha>
- Backtrader: <https://www.backtrader.com/>
- Zipline-reloaded: <https://github.com/stefan-jansen/zipline-reloaded>
- Lean: <https://www.lean.io/>
- Lean docs: <https://www.quantconnect.com/docs/v2/lean-engine/getting-started>
- NautilusTrader: <https://nautilustrader.io/docs/latest/concepts/overview>
- vn.py: <https://github.com/vnpy/vnpy>
- OpenBB: <https://github.com/OpenBB-finance/OpenBB>
- PyPortfolioOpt: <https://github.com/PyPortfolio/PyPortfolioOpt>
- Riskfolio-Lib: <https://pypi.org/project/riskfolio-lib/>
- skfolio: <https://github.com/skfolio/skfolio>
- Alphalens-reloaded: <https://pypi.org/project/alphalens-reloaded/>
- QuantStats: <https://github.com/ranaroussi/quantstats>
- FinBERT: <https://github.com/ProsusAI/finBERT>
- Transformers: <https://huggingface.co/docs/transformers/index>
- Financial PhraseBank: <https://huggingface.co/datasets/takala/financial_phrasebank>
- SnowNLP: <https://github.com/isnowfy/snownlp>
- LTP: <https://ltp.ai/>
- HanLP NER demo/docs: <https://hanlp.hankcs.com/en/demos/ner.html>
- PaddleNLP: <https://paddlenlp.readthedocs.io/en/latest/index.html>
- spaCy: <https://github.com/explosion/spaCy>
- PyABSA: <https://pyabsa.readthedocs.io/en/stable/0_intro/introduction.html>
- VADER: <https://github.com/cjhutto/vaderSentiment>
- GDELT DOC 2.0: <https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/amp/>
- HuggingFace financial datasets search: <https://huggingface.co/datasets?search=financial>
- Twitter Financial News Sentiment: <https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment>
- Advances in Financial Machine Learning: <https://dev.store.wiley.com/en-us/Advances%2Bin%2BFinancial%2BMachine%2BLearning-p-00000140>
- Machine Learning for Algorithmic Trading: <https://books.apple.com/us/book/machine-learning-for-algorithmic-trading/id1525046439>
- Systematic Trading: <https://www.harriman-house.com/systematic-trading>
- Inside the Black Box: <https://www.thequantbook.com/>
- A Man for All Markets: <https://www.edwardothorp.com/books/a-man-for-all-markets/>
- The Man Who Solved the Market: <https://www.randomhousebooks.com/books/557104/>
