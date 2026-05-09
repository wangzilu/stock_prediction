# 量化交易平台、舆情分析库与量化大师深度调研

**日期：** 2026-05-07
**目的：** 系统梳理当前量化交易生态、舆情分析工具链、以及量化领域名人/书籍，为系统 V2 迭代提供决策依据。

---

## 一、量化交易平台与开源库

### 1.1 AI 驱动的量化研究平台（Qlib 级别）

| 平台 | GitHub Stars | 核心能力 | 与本项目关系 |
|------|-------------|---------|-------------|
| **[Qlib](https://github.com/microsoft/qlib)** (微软) | ~39k | Alpha158因子、LightGBM/Transformer模型、滚动训练、组合优化、RD-Agent自动化研发 | **已在用**。当前基础平台，Alpha158 + LightGBM |
| **[FinRL](https://github.com/AI4Finance-Foundation/FinRL)** | ~14k | 深度强化学习交易框架，支持A2C/DDPG/PPO/TD3/SAC，多资产组合优化 | **高优先级**。比当前单票RL更强，Ensemble策略 |
| **[Qbot](https://github.com/UFund-Me/Qbot)** | ~17k | AI量化机器人，本地部署，LSTM/Transformer/LightGBM + FinGPT，自动因子发现 | A股RL替代方案，学术论文模型复现 |
| **[TradeMaster](https://github.com/TradeMaster-NTU/TradeMaster)** | ~1k | 全流程RL交易：数据→环境→智能体→评估 | RL策略基准测试 |
| **[AlphaPy](https://github.com/ScottfreeLLC/AlphaPy)** | ~1k | AutoML框架，集成XGBoost/LightGBM/CatBoost | 快速尝试不同ML模型 |

### 1.2 回测与实盘框架

| 框架 | 特点 | 适用场景 |
|------|------|---------|
| **[VectorBT](https://github.com/polakowo/vectorbt)** | 向量化回测，极快（百万级交易/秒），NumPy/Numba驱动 | 参数优化、大规模策略扫描 |
| **[Backtrader](https://github.com/mementum/backtrader)** | 事件驱动，API优雅，示例丰富 | 原型开发、学习入门 |
| **[NautilusTrader](https://github.com/nautechsystems/nautilus_trader)** | 生产级Rust内核，确定性事件驱动 | 生产环境实盘，最快参数优化 |
| **[QuantConnect/Lean](https://github.com/QuantConnect/Lean)** | C#，多资产，云+本地，30万+用户 | 多资产策略，云端部署 |
| **[Zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded)** | Quantopian维护的Zipline分支，Pipeline API | 因子模型研究 |

### 1.3 A股专用

| 库 | 定位 | 价值 |
|----|------|------|
| **[VNPY](https://github.com/vnpy/vnpy)** | A股/期货全栈量化平台，回测到实盘，CTA为主 | 如果要做实盘交易的首选 |
| **[RQAlpha](https://github.com/ricequant/rqalpha)** | A股回测，插件（Mod）架构，模块化 | A股研究最佳回测工具 |
| **[AKShare](https://github.com/akfamily/akshare)** | A股+全球数据，免费无需认证 | **已在用**，主要数据源 |
| **[TuShare Pro](https://github.com/waditu/tushare)** | A股数据Pro版，涨停板列表/资金流/融资融券/机构持仓 | 涨停板数据、妖股检测必备 |
| **[BaoStock](http://baostock.com/)** | 免费A股数据，API简单 | **已在用**，备份数据源 |

### 1.4 数据平台对比

| 数据源 | 免费 | 涨停板数据 | 龙虎榜 | 融资融券 | 北向资金 | 分钟级 |
|--------|------|-----------|--------|---------|---------|--------|
| AKShare | 是 | `stock_zt_pool_em()` | 有 | 有 | 有 | 部分 |
| TuShare Pro | 积分制 | `limit_list()` | 有 | 有 | 有 | 有 |
| BaoStock | 是 | 无 | 无 | 无 | 无 | 5分钟 |
| JQData | 免费额度 | 有 | 有 | 有 | 有 | 分钟级 |
| Wind API | 付费 | 全 | 全 | 全 | 全 | tick级 |

---

## 二、舆情分析库与工具

### 2.1 金融专用舆情模型

| 模型/库 | 来源 | 特点 | 中文支持 | GitHub |
|---------|------|------|---------|--------|
| **[FinBERT](https://github.com/ProsusAI/finBERT)** | Prosus AI | BERT微调，金融文本情感三分类（正面/负面/中性），Hugging Face可用 | 英文为主 | 5k+ stars |
| **[FinGPT](https://github.com/AI4Finance-Foundation/FinGPT)** | AI4Finance | 开源金融大语言模型，LoRA微调，支持情感分析/摘要/推理 | **中英双语**（v3.1用ChatGLM2-6B） | 14k+ stars |
| **[FinNLP](https://github.com/AI4Finance-Foundation/FinNLP)** | AI4Finance | 金融NLP框架，连接HuggingFace+本地部署LLM | 中英文 | 1k+ stars |
| **[SnowNLP](https://github.com/isnowfy/snownlp)** | 社区 | 轻量级中文NLP，情感分析输出0~1，可自定义训练集 | **中文专用** | 6k+ stars |
| **[stock-sentiment-cn](https://github.com/joinylee/stock-sentiment-cn)** | 社区 | A股市场情绪分析系统V3，AI驱动 | **A股专用** | 新项目 |

### 2.2 通用NLP框架（可用于金融舆情）

| 框架 | 特点 | 金融适用性 |
|------|------|-----------|
| **[Transformers](https://huggingface.co/docs/transformers)** (Hugging Face) | 统一API加载BERT/GPT/LLaMA等所有模型 | 加载FinBERT/FinGPT的标准方式 |
| **[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)** (百度) | 百度自研，中文NLP性能强，ERNIE系列模型 | 中文金融文本处理首选之一 |
| **[spaCy](https://spacy.io/)** | 工业级NLP，快速高效，适合大规模处理 | 批量处理金融新闻 |
| **[VADER](https://github.com/cjhutto/vaderSentiment)** | 基于词典的情感分析，专为社交媒体优化 | 分析股吧/微博等短文本 |
| **[TextBlob](https://textblob.readthedocs.io/)** | 简单易用，入门级 | 快速原型验证 |

### 2.3 中文金融舆情数据源

| 数据源 | 内容 | 获取方式 |
|--------|------|---------|
| **东方财富股吧** | 个股讨论帖，情绪最直观 | 爬虫 / AKShare `stock_comment_em()` |
| **雪球** | 投资者观点，机构/大V分析 | 爬虫 |
| **微博财经** | 热搜话题，散户情绪 | 爬虫 |
| **同花顺新闻** | 财经新闻，公司公告 | 爬虫 / TuShare |
| **新浪财经** | 综合财经新闻 | RSS / 爬虫 |
| **GDELT** | 全球新闻事件数据库 | **已在用**，地缘分析 |

### 2.4 当前系统 vs 理想舆情架构

```
当前：                           理想（V2）：
RSS新闻 → LLM(MiniMax)         东财股吧 → SnowNLP快筛
       → 地缘因子                 ↓ 异常帖 → FinGPT深度分析
                                微博/雪球 → 情绪指数
                                AKShare → 涨停板/龙虎榜情绪
                                RSS新闻 → LLM → 地缘+政策因子
                                        → 多维度情绪融合打分
```

---

## 三、量化大师与必读书籍

### 3.1 量化传奇人物

| 人物 | 公司/成就 | 核心理念 | 代表作/传记 |
|------|---------|---------|------------|
| **Jim Simons（西蒙斯）** | 文艺复兴科技，大奖章基金年化66%（1988-2024） | 纯数学驱动，不招金融人，招数学家/物理学家/CS。"既然我们会做模型，那就跟着模型走" | 《The Man Who Solved the Market》(Zuckerman) / 《解读量化投资》 |
| **Ray Dalio（达里奥）** | 桥水基金，全天候策略 | 宏观+系统化，风险平价（Risk Parity），用原则（Principles）管理一切 | 《原则》《债务危机》 |
| **Ed Thorp（索普）** | 21点计牌发明者→量化对冲基金先驱 | 用数学击败赌场→用同样方法击败市场。统计套利/可转债套利 | 《A Man for All Markets》 |
| **David Shaw（肖）** | D.E.Shaw，统计套利先驱 | 计算力就是alpha，最早用大规模计算做量化 | 无传记，但Jeff Bezos曾是其员工 |
| **Cliff Asness（阿斯内斯）** | AQR Capital，因子投资先驱 | 价值+动量+质量多因子，系统化因子投资推广者 | 大量学术论文 |
| **Marcos Lopez de Prado** | 真实资金管理+学术研究双栖 | 金融ML不是数据科学，是工程学。Triple-barrier、Meta-labeling、HRP | 三部曲（见下方书单） |
| **Ernest Chan（陈欧内斯特）** | 独立量化交易者 | 均值回归+动量策略的实战大师，代码驱动 | 三部曲（见下方书单） |
| **Stefan Jansen** | 《Machine Learning for Algorithmic Trading》作者 | ML在量化中的系统性应用，Zipline-reloaded维护者 | 书同名 |

### 3.2 必读书单（按优先级排序）

#### 第一梯队：改变认知的

| 书名 | 作者 | 核心价值 |
|------|------|---------|
| **《Advances in Financial Machine Learning》** | Marcos Lopez de Prado (2018) | **金融ML圣经**。Triple-barrier标签、Meta-labeling、分数差分、净化K折CV、特征重要性（MDI/MDA/SFI）。不读此书做金融ML等于盲人摸象 |
| **《The Man Who Solved the Market》** | Gregory Zuckerman (2019) | 西蒙斯传记。理解量化投资的哲学：为什么纯数据驱动能打败基本面分析 |
| **《A Man for All Markets》** | Ed Thorp (2017) | 量化投资的源头。从21点到华尔街，概率思维的极致应用 |
| **《Machine Learning for Algorithmic Trading》** | Stefan Jansen (2020) | 最全面的ML量化实操书。因子工程、替代数据、NLP情绪、DRL策略，有完整代码 |

#### 第二梯队：实战技能

| 书名 | 作者 | 核心价值 |
|------|------|---------|
| **《Machine Learning for Asset Managers》** | Lopez de Prado (2020) | 层次风险平价(HRP)、组合构建、协方差矩阵去噪。薄但精 |
| **《Causal Factor Investing》** | Lopez de Prado (2023) | 从相关因子到因果因子，避免"因子动物园"过拟合 |
| **《Quantitative Trading》** | Ernest Chan (2008) | 量化入门第一本。回测陷阱、容量约束、Kelly公式 |
| **《Algorithmic Trading》** | Ernest Chan (2013) | 均值回归、动量、配对交易的实现。有代码 |
| **《Machine Trading》** | Ernest Chan (2017) | 高级策略：组合优化、风控、高频概念 |
| **《Finding Alphas》** | WorldQuant (Kakushadze) (2015) | 101个Alpha公式，直接可实现。因子工程速查手册 |

#### 第三梯队：深度理论

| 书名 | 作者 | 核心价值 |
|------|------|---------|
| **《Active Portfolio Management》** | Grinold & Kahn (1999) | "主动管理基本定律"。信息比率、宽度、技巧。机构量化的理论基石 |
| **《Quantitative Equity Portfolio Management》** | Chincarini & Kim (2006) | Barra风险模型构建。学术严谨 |
| **《原则》** | Ray Dalio (2017) | 不是量化技术书，但对决策系统化思维影响深远 |
| **《The Signal and the Noise》** | Nate Silver (2012) | 预测的艺术与科学，概率思维 |
| **《Fooled by Randomness》** | Nassim Taleb (2001) | 随机性在金融中的角色，对量化结果保持敬畏 |

### 3.3 关键学术论文

| 论文 | 年份 | 核心贡献 |
|------|------|---------|
| **"FinRL: DRL for Automated Stock Trading"** — Liu et al. | 2020 | NeurIPS 2020。FinRL框架，5种DRL基准 |
| **"Deep RL Ensemble Strategy"** — Yang et al. | 2020 | A2C/DDPG/PPO集成，按Sharpe自动切换 |
| **"Temporal Fusion Transformer"** — Lim et al. | 2021 | TFT架构，可解释多步预测。Google Research |
| **"MASTER: Market-Guided Stock Transformer"** — Li et al. | 2024 | AAAI 2024。股内+股间模式 |
| **"TGNS: Transformer-GNN for Stock Forecasting"** | 2025 | GNN+Transformer捕获局部/全局依赖 |
| **"CNN-LSTM-GNN (CLGNN) for A-Share Prediction"** | 2025 | 混合模型，A股实测 |
| **"DoubleAdapt: Meta-Learning for Incremental Learning"** — Qlib团队 | 2023 | 处理概念漂移，Qlib内置 |
| **"Qlib: AI-oriented Quant Platform"** — Yang et al. | 2020 | Qlib平台论文 |
| **"ML in the Chinese Stock Market"** — Leippold et al. | 2022 | JFE。A股ML全面研究，树模型最优 |
| **"AI-Driven Anomaly Detection in Stock Markets"** | 2025 | Isolation Forest + CatBoost异常检测 |
| **"Financial ML: An Engineering Problem"** — Lopez de Prado | 2025 | 最新思考，金融ML是工程问题 |

### 3.4 量化大师经验总结

**西蒙斯的核心经验：**
1. 不招华尔街人，招科学家 — 第一性原理思维
2. 模型说了算，人不干预 — "既然做了模型就跟着走"
3. 短线高频交易为主 — 信号衰减快，快速迭代
4. 数据质量大于模型复杂度 — 垃圾进垃圾出
5. 分散投资大量品种 — 不靠单笔大赚

**Lopez de Prado 的核心经验：**
1. 金融ML是工程，不是实验 — 可复现、可部署、可监控
2. 标签设计比模型选择更重要 — Triple-barrier >> 简单收益率
3. 回测必须净化（Purged CV）— 否则信息泄露
4. 特征重要性要用MDI/MDA/SFI交叉验证 — 单一指标不可靠
5. 层次风险平价(HRP) >> 均值方差优化 — 更稳健

**Ernest Chan 的核心经验：**
1. 策略容量是硬约束 — 再好的策略也有容量上限
2. Kelly公式决定仓位 — 不是拍脑袋
3. 均值回归在短期、动量在中期 — 不同时间尺度用不同策略
4. 交易成本会杀死大多数策略 — 模拟时必须加入滑点和手续费
5. 简单策略 > 复杂策略 — 过拟合是最大敌人

---

## 四、妖股（5x-10x）检测策略

### 4.1 妖股特征画像

- 涨幅 5x-10x，数周到数月完成
- 多个连续涨停板（连板）
- 脱离基本面估值逻辑
- 散户投机性羊群效应 + 题材催化
- 通常小盘、低机构持仓、自由流通市值 < 50亿

### 4.2 五大信号组

**信号组1：量价异常**
- 成交量突增 > 20日均量3倍
- 换手率 > 15%（小盘股）
- 5日收益率占20日收益率80%以上（加速）

**信号组2：涨停板链分析**
- 首板（20+日无涨停后首次涨停）— 最高期望值入场点
- 连板数（2连板是关键确认）
- 板质量（一字板 > T字板 > 烂板）
- 次日溢价率（市场投机情绪温度计）

**信号组3：股本结构**
- 自由流通市值 < 50亿
- 机构持仓 < 10%
- 近期有摘帽/更名/分拆等事件

**信号组4：板块轮动**
- 同概念5日内涨停股 ≥ 3只
- 板块指数突破20日高点
- 政策催化NLP检测

**信号组5：舆情与资金**
- 社交媒体提及量突增 > 7日均值5倍
- 龙虎榜知名游资出现
- 融资买入/融资余额 > 10%

### 4.3 复合妖股评分公式

```
monster_score = (
    0.25 * limit_up_chain_score      # 连板强度 (0-1)
  + 0.20 * volume_anomaly_score      # 量价异常 (0-1)
  + 0.20 * sector_heat_score         # 板块热度 (0-1)
  + 0.15 * float_structure_score     # 股本结构 (0-1)
  + 0.10 * sentiment_spike_score     # 舆情突增 (0-1)
  + 0.10 * technical_breakout_score  # 技术突破 (0-1)
)
```

### 4.4 风控过滤

- 5板以上不追（除非总龙头）
- 仅在投机市场情绪（次日溢价 > 1%）时参与
- 换手率 > 40% 视为出货
- 单只妖股仓位 ≤ 5%

---

## 五、对本项目的映射与 V2 优先级

| 优先级 | 改进项 | 工作量 | 预期收益 |
|--------|--------|--------|---------|
| P0 | **修复Qlib数据兼容性** — 当前Alpha158训练报错 | 低 | 解锁LGB推理 |
| P1 | **涨停板数据接入** — AKShare `stock_zt_pool_em()` | 低 | 妖股检测核心数据 |
| P1 | **SnowNLP快速舆情筛选** — 东财股吧/微博帖子 | 低 | 情绪指数 |
| P2 | **妖股复合评分模块** — 实现4.3打分 | 中 | 直接可用的妖股信号 |
| P2 | **FinGPT中文情感分析** — 替代/增强当前LLM舆情 | 中 | 更精准情感判断 |
| P2 | **Transformer改为序列输入** — 多日特征做attention | 中 | RL模型显著提升 |
| P3 | **板块热度追踪器** — 概念/行业涨停数量追踪 | 中 | 板块轮动信号 |
| P3 | **Triple-barrier标签** — 替代简单前向收益 | 中 | 更好训练信号 |
| P4 | **GNN股票关系图** — 行业/概念/持仓图谱 | 高 | 捕获板块联动 |
| P4 | **FinRL组合级RL** — 替代单票RL | 高 | 更好资金配置 |

---

## Sources

- [Qlib - Microsoft](https://github.com/microsoft/qlib)
- [FinRL - AI4Finance Foundation](https://github.com/AI4Finance-Foundation/FinRL)
- [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT)
- [FinBERT](https://github.com/ProsusAI/finBERT)
- [FinNLP](https://github.com/AI4Finance-Foundation/FinNLP)
- [SnowNLP](https://github.com/isnowfy/snownlp)
- [stock-sentiment-cn](https://github.com/joinylee/stock-sentiment-cn)
- [VectorBT](https://github.com/polakowo/vectorbt)
- [NautilusTrader](https://github.com/nautechsystems/nautilus_trader)
- [VNPY](https://github.com/vnpy/vnpy)
- [RQAlpha](https://github.com/ricequant/rqalpha)
- [AKShare](https://github.com/akfamily/akshare)
- [TuShare](https://github.com/waditu/tushare)
- [Qbot](https://github.com/UFund-Me/Qbot)
- [TradeMaster](https://github.com/TradeMaster-NTU/TradeMaster)
- [QuantConnect/Lean](https://github.com/QuantConnect/Lean)
- [awesome-quant](https://github.com/wilsonfreitas/awesome-quant)
- [awesome-systematic-trading](https://github.com/wangzhe3224/awesome-systematic-trading)
- [Popular open-source quantitative trading projects 2025-2026](https://grokipedia.com/page/Popular_open-source_quantitative_trading_projects_20252026)
- [The Man Who Solved the Market](https://www.amazon.com/Man-Who-Solved-Market-Revolution/dp/073521798X)
- [Jim Simons Trading Strategies](https://www.quantifiedstrategies.com/jim-simons/)
- [Chinese Financial News Sentiment Analysis Framework](https://www.mdpi.com/2504-2289/9/10/263)
- [Factor Investing in Chinese A-Shares](https://alphaarchitect.com/factor-investing-fama-french-5-factor-model-chinese-shares/)
- [ML in the Chinese Stock Market](https://www.sciencedirect.com/science/article/pii/S0304405X21003743)
- [AI-Driven Anomaly Detection in Stock Markets](https://link.springer.com/article/10.1007/s10614-025-11274-8)

---

## 六、适合集成到本项目的筛选与实施方案

> 以下从调研内容中，按"对本项目实际收益 / 集成难度"筛选出值得做的事项，并给出具体实施方案。

### 6.1 筛选原则

**纳入标准：**
- 能直接插入现有 pipeline（data collector → factor → model → scorer → push）
- 不需要重构核心架构
- 对推荐质量或妖股发现有实质提升
- 数据源免费或低成本可获取

**排除项及理由：**

| 排除项 | 理由 |
|--------|------|
| QuantConnect/Lean | C#生态，和本项目Python架构不兼容 |
| NautilusTrader | Rust内核，适合高频/实盘，当前不需要 |
| VectorBT | 回测优化工具，当前系统不做系统化回测 |
| VNPY | CTA/实盘框架，当前系统是信号推送，不做自动下单 |
| 配对交易/统计套利 | A股融券限制大，不适合本项目的多头推荐定位 |
| PaddleNLP | 百度生态，Transformers + FinGPT已经够用 |
| Wind API | 付费商业数据，性价比低 |
| 多智能体RL | 复杂度过高，当前单票RL还没成熟 |

### 6.2 确定集成的 7 个改进（按实施顺序）

---

#### 改进1：涨停板数据采集器 — 妖股检测数据基础

**来源：** 调研 §四 妖股检测 + AKShare数据源
**价值：** 涨停板是A股妖股的核心信号，当前系统完全没有这个数据维度
**工作量：** 1天

**实施方案：**

```
新建文件: data/collectors/limit_up.py

class LimitUpCollector:
    """涨停板数据采集 — 通过 AKShare"""

    def fetch_today_pool(self) -> pd.DataFrame:
        """今日涨停股列表: ak.stock_zt_pool_em()
        返回: 代码, 名称, 涨停时间, 封板资金, 开板次数, 连板数"""

    def fetch_yesterday_pool(self) -> pd.DataFrame:
        """昨日涨停股(用于计算次日溢价): ak.stock_zt_pool_previous_em()"""

    def fetch_strong_stocks(self) -> pd.DataFrame:
        """强势股池(涨幅>5%): ak.stock_zt_pool_strong_em()"""

    def compute_board_premium(self, days=5) -> float:
        """计算滚动5日平均涨停次日溢价率 — 衡量市场投机温度"""

    def get_consecutive_boards(self) -> dict:
        """统计当前连板股: {code: 连板数}"""
```

**接入点：**
- `scheduler/jobs.py` → `run_morning_recommendation()` 中加载涨停数据
- `signals/scorer.py` → `Recommendation` 增加 `limit_up_info` 字段

---

#### 改进2：妖股复合评分模块

**来源：** 调研 §4.3 复合评分公式
**价值：** 将涨停板+量价+板块+舆情融合成一个可排序的妖股分数
**工作量：** 2天
**依赖：** 改进1

**实施方案：**

```
新建文件: factors/monster_stock.py

class MonsterStockScorer:
    """妖股复合评分"""

    def __init__(self, limit_up_collector, sentiment_collector):
        self.limit_up = limit_up_collector
        self.sentiment = sentiment_collector

    def score(self, code, name, price_df, spot_data) -> dict:
        """计算单只股票的妖股评分

        Returns:
            {
                "monster_score": float (0-1),
                "limit_up_chain_score": float,  # 连板强度
                "volume_anomaly_score": float,   # 量价异常
                "sector_heat_score": float,      # 板块热度
                "float_structure_score": float,  # 股本结构
                "sentiment_spike_score": float,  # 舆情突增
                "risk_filter_passed": bool,      # 风控过滤
            }
        """

    def _limit_up_chain_score(self, code) -> float:
        """连板数 / 5，乘以板质量系数"""

    def _volume_anomaly_score(self, price_df) -> float:
        """volume / avg_volume_20d，归一化到0-1"""

    def _sector_heat_score(self, code) -> float:
        """同概念5日涨停数 / 10"""

    def _float_structure_score(self, spot_data) -> float:
        """自由流通市值 < 50亿 → 高分"""

    def _risk_filter(self, code, price_df) -> bool:
        """5板以上不追、换手率>40%排除、营收过低排除"""
```

**接入点：**
- `scheduler/jobs.py` → Stage 1 筛选时额外计算 `monster_score`
- 推文中单独列出妖股候选（区别于LGB推荐）

---

#### 改进3：SnowNLP 快速舆情筛选层

**来源：** 调研 §二 SnowNLP
**价值：** 当前舆情分析靠关键词（`factors/sentiment.py`），SnowNLP 是机器学习模型，准确率更高且依然很快。作为 MiniMax LLM 之前的快筛层，减少 API 调用
**工作量：** 半天

**实施方案：**

```
修改文件: factors/sentiment.py

# 在现有 SentimentScorer 中增加 SnowNLP 路径

from snownlp import SnowNLP

class SentimentScorer:
    def score_text(self, text: str) -> float:
        """单条文本情感评分"""
        # 原有关键词方法作为 fallback
        try:
            s = SnowNLP(text)
            # SnowNLP 输出 0~1, 转换为 -1~1
            return s.sentiments * 2 - 1
        except Exception:
            return self._keyword_score(text)  # fallback

    def score_batch(self, posts: list) -> dict:
        """批量评分 — SnowNLP 快筛 + 异常帖标记"""
        scores = [self.score_text(p.get("text", "")) for p in posts]
        # 对极端负面（< -0.7）的帖子标记，后续可送 LLM 深度分析
        ...
```

**依赖安装：** `pip install snownlp`（纯Python，无外部依赖）

---

#### 改进4：FinGPT 中文金融情感分析（替代/增强 MiniMax 舆情）

**来源：** 调研 §2.1 FinGPT
**价值：** FinGPT v3.1 用 ChatGLM2-6B 底座，专门为金融情感分析微调，比通用 LLM 更准。本地部署可离线运行
**工作量：** 3天（含模型下载和部署）
**前置条件：** 需要 GPU（M1/M2 Mac 的 MPS 也可以跑 6B 模型）

**实施方案：**

```
新建文件: factors/fingpt_sentiment.py

class FinGPTSentimentScorer:
    """基于 FinGPT 的金融情感分析（本地推理）"""

    def __init__(self, model_name="FinGPT/fingpt-sentiment_llama2-13b_lora"):
        # 从 HuggingFace 加载 FinGPT LoRA 模型
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        ...

    def analyze(self, text: str) -> dict:
        """返回 {"sentiment": "positive/negative/neutral", "confidence": 0.95}"""
```

**接入策略：** SnowNLP 快筛 → 异常帖送 FinGPT 深度分析 → 替代当前的 MiniMax 舆情调用
**降级方案：** 如果 GPU 不够，继续用 MiniMax API

---

#### 改进5：板块热度追踪器

**来源：** 调研 §4.2 信号组4 板块轮动
**价值：** A股是板块驱动的市场，当前系统没有板块级信号
**工作量：** 1.5天
**依赖：** 改进1（涨停板数据）

**实施方案：**

```
新建文件: factors/sector_heat.py

class SectorHeatTracker:
    """板块/概念热度追踪"""

    def __init__(self, limit_up_collector):
        self.limit_up = limit_up_collector

    def compute_sector_heat(self) -> pd.DataFrame:
        """计算各板块热度

        数据源: ak.stock_board_concept_name_em() 概念板块列表
               ak.stock_board_concept_cons_em() 概念成分股
               + 涨停板数据交叉

        Returns: DataFrame[concept, heat_score, limit_up_count,
                          avg_change_pct, leader_stock]
        """

    def get_hot_sectors(self, top_n=5) -> list:
        """返回当前最热的N个概念板块"""

    def is_sector_accelerating(self, concept: str) -> bool:
        """板块是否处于加速阶段（指数突破20日高点）"""
```

**接入点：**
- `scheduler/jobs.py` → 推文中增加"板块轮动"板块
- `signals/scorer.py` → `macro_score` 中融入板块热度因子

---

#### 改进6：Triple-Barrier 标签替代简单收益率

**来源：** 调研 §三 Lopez de Prado《AFML》
**价值：** 当前 LGB 用简单5日前向收益率作为标签。Triple-barrier 用止盈/止损/到期三重门槛，产生更实战化的标签（赢/输/平），训练信号质量更高
**工作量：** 1天

**实施方案：**

```
修改文件: scripts/train_lgb.py

def triple_barrier_label(price_series, upper=0.05, lower=-0.03, max_days=5):
    """Triple-barrier labeling (Lopez de Prado)

    Args:
        price_series: 未来N日价格序列
        upper: 止盈线 (+5%)
        lower: 止损线 (-3%)
        max_days: 最大持有天数

    Returns:
        1 (触达止盈), -1 (触达止损), 0 (到期未触发)
    """
    entry_price = price_series.iloc[0]
    for i, price in enumerate(price_series[1:max_days+1]):
        ret = (price - entry_price) / entry_price
        if ret >= upper:
            return 1
        if ret <= lower:
            return -1
    return 0
```

**在 train_lgb.py 中使用：**
- Alpha158 handler 的 `label` 参数改为自定义 triple-barrier 函数
- 或者在 handler 之后对 label 列做后处理

---

#### 改进7：Transformer RL 改为序列输入

**来源：** Code Review §I1 + 调研 §1.4.2
**价值：** 当前 Transformer 处理 seq_len=1 的单步观测，等同于昂贵的 MLP。改为多日序列输入才能发挥 attention 的真正能力
**工作量：** 2天

**实施方案：**

```
修改文件: models/rl_agent.py

class StockTradingEnv(gym.Env):
    def _get_obs(self):
        # 旧: 单步特征
        # feat = self.features[self._step]

        # 新: 20日窗口序列
        start = max(0, self._step - self.window + 1)
        window_feat = self.features[start:self._step + 1]
        # Pad if needed
        if len(window_feat) < self.window:
            pad = np.zeros((self.window - len(window_feat), window_feat.shape[1]))
            window_feat = np.vstack([pad, window_feat])
        # Flatten for gym observation space, or reshape in actor
        obs = window_feat.flatten()
        obs = np.append(obs, [self._position, 0.0])
        return obs.astype(np.float32)

class TransformerActor(nn.Module):
    def forward(self, obs, state=None, info=None):
        # 旧: unsqueeze(1) → seq_len=1
        # 新: reshape to (batch, window, features)
        batch = obs.shape[0] if obs.dim() > 1 else 1
        x = obs.view(batch, self.window, -1)  # (batch, 20, feat_dim)
        x = self.input_proj(x)  # (batch, 20, d_model)
        x = self.transformer(x)
        x = x[:, -1, :]  # 取最后一个时间步
        logits = self.head(x)
        return logits, state
```

---

### 6.3 实施路线图

```
Phase 1 (1周) — 数据层扩展
├─ 改进1: 涨停板数据采集器
├─ 改进3: SnowNLP舆情快筛
└─ 修复: Qlib数据兼容性 (numpy/bin格式)

Phase 2 (1周) — 信号层升级
├─ 改进2: 妖股复合评分
├─ 改进5: 板块热度追踪
└─ 改进6: Triple-barrier标签 + LGB重训

Phase 3 (1周) — 模型层优化
├─ 改进7: Transformer序列输入
├─ 改进4: FinGPT部署（如有GPU）
└─ RL agent评估 + 条件上线

每个Phase结束时:
- 跑一次完整pipeline，对比推文质量
- commit + 记录指标
```

### 6.4 预期收益

---

## 七、原始设计文档问题审查

> 审查 `docs/superpowers/specs/2026-05-05-stock-prediction-system-design.md`（设计文档）
> 和 `docs/superpowers/plans/2026-05-05-stock-prediction-mvp.md`（实施计划）

### 7.1 设计文档问题（design.md）

#### 问题1：FinGPT 情感分析写了但从未实现

设计文档在"舆情因子"和"技术选型"中都写了用 FinGPT，但实际实现用的是**关键词打分**（`factors/sentiment.py`），落差巨大。当前系统的舆情因子本质上是数正面/负面关键词的个数，等同于2010年代的词袋模型水平。

**影响：** 舆情信号质量远低于设计预期，导致推荐的"舆情因子"维度基本无效。

#### 问题2：Scrapy/httpx 爬虫选型不合理

设计里写了用 Scrapy/httpx 做微博/雪球/股吧爬虫。实际问题：
- **雪球** 反爬严格，需要登录 cookie 才能拿到有价值的数据，单纯 httpx 请求拿到的是极少量公开帖
- **微博** 设计里提了但从未实现，实际采集器里根本没有微博模块
- **东方财富股吧** 的爬虫用正则解析 HTML（`re.findall(r'title="([^"]+)"'`），极其脆弱，页面结构一改就全挂

**影响：** 舆情数据覆盖面和质量都远低于预期。

#### 问题3：因子工程层设计和实际脱节

设计文档写的因子工程：
- "大V共识：头部KOL观点方向一致性" — **从未实现**
- "情绪背离：舆情乐观但价格下跌" — **从未实现**
- "舆情和地缘因子处理为日频时序数据，与行情因子对齐存入Qlib本地数据仓库" — **从未实现**，舆情因子是实时计算的临时值，没有持久化到Qlib仓库

**影响：** 设计里最有价值的三个因子（大V共识、情绪背离、舆情时序化）全部是空头支票。

#### 问题4：推送场景设计过于简单

设计写了3种推送（14:00荐股、5日印证、风险警示），实际已经扩展到4个时间槽（9:20/14:30/15:30/22:00），但设计文档没更新。更重要的是：
- 设计里的"每日荐股"没区分短中长线 — 当前已改进为三维度推荐
- 设计里没有"卖出建议"概念 — 只管推荐不管退出，对用户来说是半成品

#### 问题5：宏观模型用"规则引擎+简单ML"太模糊

设计写宏观因子用"规则引擎+简单ML"，但实际实现是用 MiniMax LLM 做地缘分析（`signals/llm_analyst.py`），这比原设计好得多。但设计文档完全没体现这个改进。

### 7.2 实施计划问题（mvp plan）

#### 问题6：model_score 用 change_pct 做代理 — 整个模型层形同虚设

MVP plan 的 `scheduler/jobs.py` 中：
```python
model_score = quote.get("change_pct", 0) / 10  # Normalize to [-1, 1] range
```
这意味着"量化模型预测"实际上就是今天的涨跌幅除以10。这不是预测，这是**追涨杀跌**。LightGBM 模型虽然写了训练代码，但从未被接入推荐 pipeline。

**影响：** 系统的核心卖点（量化模型预测）从 MVP 到最近才尝试修复。

#### 问题7：SentimentScorer 用了一个不存在的模型

MVP 计划写了用 `bardsai/finance-sentiment-zh-base` 模型，但这个模型在 HuggingFace 上并不存在（或已下线）。代码里虽然有 fallback 到关键词，但意味着 FinGPT/FinBERT 的承诺从第一天就没兑现。

#### 问题8：WeChat推送用了企业微信 Webhook 但实际用的是 pushplus

MVP 设计和代码写的都是企业微信 Webhook（`WECHAT_WEBHOOK_URL`，msgtype=markdown），但实际系统用的是 **pushplus.plus**（个人微信推送服务）。API 完全不同，代码已经改成了 pushplus，但文档没更新。

#### 问题9：测试设计依赖网络但没标记

`test_market_collector.py`、`test_sentiment_collector.py` 等测试直接调用 AKShare/雪球 API，需要网络才能跑。但没有 `@pytest.mark.network` 标记，CI 环境必挂。

#### 问题10：记录推荐时用 INSERT OR REPLACE 但 UNIQUE 约束会丢数据

`tracker/verifier.py` 中：
```python
conn.execute(
    """INSERT OR REPLACE INTO recommendations
       (rec_date, code, name, signal, score, price_at_rec)
       VALUES (?, ?, ?, ?, ?, ?)""",
    ...
)
```
如果同一天对同一只股票修改了信号（比如早上看多、下午改看空），`OR REPLACE` 会静默覆盖之前的记录，丢失原始推荐历史。应该用 `INSERT OR IGNORE` 保留第一次推荐。

### 7.3 总结：设计 vs 现实差距

| 设计承诺 | 实际状态 | 差距 |
|---------|---------|------|
| FinGPT中文情感模型 | 关键词词频统计 | 巨大 |
| 微博/雪球/股吧三源舆情 | 雪球勉强、股吧脆弱、微博未实现 | 较大 |
| LightGBM量化预测 | 涨跌幅/10代替 | 巨大（已修复中） |
| 大V共识/情绪背离因子 | 未实现 | 完全缺失 |
| 舆情因子持久化到Qlib | 未实现 | 完全缺失 |
| 企业微信Webhook推送 | pushplus个人微信推送 | 功能ok但文档不一致 |

### 7.4 逐项代码验证发现的具体问题

以下是通过实际运行代码、调用API、grep调用链后发现的问题，每个都可复现。

#### 问题11：雪球爬虫实际已失效 — 反爬拦截返回空

**验证方法：** 直接调用雪球 API
```
GET https://xueqiu.com/query/v1/symbol/search/status.json?q=SH600519&count=3
→ status 200，但 body 为空（非JSON），json() 报 JSONDecodeError
```

**原因：** 雪球对非浏览器请求做了严格反爬，仅靠 `session.get("https://xueqiu.com/")` 拿 cookie 已经不够。需要真实浏览器指纹或登录 token。

**影响：** `SentimentCollector.fetch_xueqiu()` 永远返回空列表。舆情数据来源实际只剩东方财富股吧（正则解析HTML，也不稳定）。整个舆情信号约等于废的。

**解决方案：**
- 短期：改用 AKShare 的 `ak.stock_comment_em()` 获取东财个股评论数据（官方API，不会被反爬）
- 中期：用 SnowNLP + AKShare 舆情数据替代爬虫方案
- 长期：接入 FinGPT 本地部署

#### 问题12：GDELT + GeopoliticalScorer 写了但从未接入 pipeline

**验证方法：** grep 调用链
```bash
grep -r "GDELTCollector\|GeopoliticalScorer" scheduler/jobs.py signals/llm_analyst.py
# 结果：零匹配
```

**事实：**
- `data/collectors/gdelt.py`（249行）— 写了完整的 GDELT API 调用
- `factors/geopolitical.py`（222行）— 写了完整的关键词地缘评分器
- 但 `scheduler/jobs.py` 的 pipeline 中**完全没调用**它们
- 实际地缘分析走的是 `signals/llm_analyst.py` → MiniMax LLM API

**影响：** 471行代码是死代码。设计文档说"GDELT每15分钟更新"、"央行声明关键词检测"全部是空话。

**评估：** MiniMax LLM 做地缘分析其实效果更好（理解上下文，不是简单关键词匹配），所以 GDELT + GeopoliticalScorer 不一定需要接回来。但应该在设计文档中说明这个替换决策，而不是假装两个都在用。

**建议：** 要么正式把 GDELT/GeopoliticalScorer 标记为 deprecated，要么把它作为 LLM 的补充信号源接入（GDELT 的 tone 时序数据做量化指标，LLM 做定性分析，两者互补）。

#### 问题13：东方财富股吧爬虫用 HTML 正则 — 一次页面改版就全挂

**具体代码：** `data/collectors/sentiment.py:83-84`
```python
titles = re.findall(r'title="([^"]+)"[^>]*>([^<]*)</a>', resp.text)
```

**问题：**
- 东方财富股吧是动态渲染页面，直接 GET HTML 拿到的可能是空壳（JS 渲染前）
- 正则 `title="([^"]+)"` 会匹配到大量非帖子标题的 HTML 元素（导航栏、广告链接等）
- 没有任何过滤来区分"用户发帖"和"页面噪声"
- `len(text) < 4` 是唯一的过滤条件，太弱

**解决方案：** 用 AKShare 提供的东财数据接口替代 HTML 爬虫：
```python
import akshare as ak
# 个股评论情绪（官方接口，结构化数据）
df = ak.stock_comment_em()          # 全市场评论情绪
df = ak.stock_comment_detail_zlkp_jgcyd_em(symbol="600519")  # 个股机构参与度
```

#### 问题14：因子工程层 `quant.py` 中的 `get_alpha158_handler` 默认日期写死到 2026-05-01

**代码：** `factors/quant.py:16-20`
```python
def get_alpha158_handler(
    start_time: str = "2020-01-01",
    end_time: str = "2026-05-01",   # ← 写死的日期
    instruments: str = "csi300",
```

**问题：** 到 2026-05-02 以后，调用者如果不显式传 `end_time`，就永远拿不到最新数据。虽然当前没有代码调用这个函数（也是死代码），但如果有人信了设计文档去调用就会踩坑。

**解决方案：** 改成 `end_time: str = None`，函数内部默认取 `datetime.now().strftime("%Y-%m-%d")`。

#### 问题15：MVP plan 的 SentimentScorer 写了 Transformer 模型加载逻辑但实际代码删掉了

**MVP plan 中写的：**
```python
class SentimentScorer:
    def __init__(self, model_name="bardsai/finance-sentiment-zh-base"):
        # 从 HuggingFace 加载模型
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
```

**实际代码：** `factors/sentiment.py` 完全没有 transformers 相关导入，纯关键词实现。

**问题不只是"没实现"：**
- `bardsai/finance-sentiment-zh-base` 这个模型在 HuggingFace 上找不到（可能是虚构的或已下线）
- MVP plan 里 `_score_with_model` 假设模型输出3类（negative/neutral/positive），但没验证 `bardsai` 模型是否真的是3类输出
- Plan 里写了 `if len(probs) == 3` 和 `elif len(probs) == 2` 的分支，说明写 plan 的人自己也不确定模型输出格式

**解决方案：** 替换为经过验证的模型：
- `ProsusAI/finbert` — 确认存在，3类输出，英文金融情感
- `FinGPT/fingpt-sentiment_llama2-13b_lora` — 确认存在，中英文，需要GPU
- 如果不想用大模型，SnowNLP 是最靠谱的中文轻量方案

#### 问题16：backtest 模块存在但完全是空壳

**验证：**
```
backtest/engine.py   — 234行
backtest/optimizer.py — 210行
```

设计文档提到"回测体系完善"在 Phase 5，但实际已经有 444 行回测代码了。问题是 **从未被任何脚本或 pipeline 调用**。没有测试，不知道是否能跑通。

**建议：** 要么删掉（避免维护负担），要么补上测试并在 nightly pipeline 中加入回测验证环节。

#### 问题19：`qlib.run.get_data` 模块不存在 — 数据下载命令是错的

**代码：** `factors/quant.py:36-44`（MVP plan Task 5 也写了同样的命令）
```python
subprocess.run([
    sys.executable, "-m", "qlib.run.get_data",
    "qlib_data",
    "--target_dir", str(data_dir),
    "--region", "cn",
], check=True)
```

**验证：**
```python
>>> import qlib.run
ModuleNotFoundError: No module named 'qlib.run'
```

**事实：** Qlib 0.9.7 中根本没有 `qlib.run` 模块。正确的数据下载方式是：
```python
from qlib.tests.data import GetData
GetData().qlib_data(target_dir="...", region="cn")
```

**影响：** `python main.py --setup` 会直接报错崩溃，首次安装无法完成。设计文档和 MVP plan 都写了这个错误命令。

**解决方案：** 改 `prepare_qlib_data()` 为：
```python
def prepare_qlib_data():
    from qlib.tests.data import GetData
    GetData().qlib_data(target_dir=str(QLIB_DATA_DIR), region="cn")
```

#### 问题20：`region_type="cn"` 参数名错误 — Qlib 报 Unrecognized config

**代码：** `factors/quant.py:13`
```python
qlib.init(provider_uri=QLIB_PROVIDER_URI, region_type="cn")
```

**验证：**
```
WARNING - qlib.Initialization - Unrecognized config region_type
```

Qlib 0.9.7 的正确参数是 `region=REG_CN`（`from qlib.constant import REG_CN`）。`region_type` 被 Qlib 忽略了，意味着 **Qlib 实际上没有被设置为中国区域**，可能影响 instruments 解析和交易日历选择。

有趣的是，`models/short_term.py` 和 `scripts/train_lgb.py` 用了正确的 `region=REG_CN`，只有 `factors/quant.py` 用了错误的 `region_type`。说明是不同时期写的代码，没有统一审查。

**解决方案：** `factors/quant.py:13` 改为：
```python
from qlib.constant import REG_CN
qlib.init(provider_uri=QLIB_PROVIDER_URI, region=REG_CN)
```

#### 问题21：DATA_CUTOFF_TIME 定义了但从未执行

设计文档写了详细的运行时序：
```
13:00  数据采集截止
13:00-13:30  因子计算
13:30-13:50  模型推理
13:50-14:00  生成推荐报告
14:00  推送
```

**验证：** `grep -rn "DATA_CUTOFF_TIME\|cutoff" scheduler/jobs.py main.py` → 零匹配

`config/settings.py` 定义了 `DATA_CUTOFF_TIME = "13:00"`，但**没有任何代码读取或使用它**。pipeline 启动时直接拉数据、跑模型、推送，没有任何时间窗口控制。

**影响：** 如果 9:20 的推荐 pipeline 跑了5分钟（数据采集慢），可能推送延迟但不会被感知。但如果有人把推荐改回 14:00，没有 cutoff 控制意味着 pipeline 可能在 14:30 才推完，超过设计预期。

**建议：** 要么删掉 `DATA_CUTOFF_TIME`（当前不需要），要么加个 deadline 检查：如果 pipeline 耗时超过预期，在推文里标注"数据截止时间"。

#### 问题22：信号融合权重公式 — 设计、计划、实现三个版本互相矛盾

| 来源 | 公式 | 因子数 |
|------|------|--------|
| **设计文档** | `short*w1 + mid*w2 + macro*w3`（没给具体权重） | 3个 |
| **MVP plan** | `model*0.6 + sentiment*0.3 + (heat-0.5)*0.1` | 3个（无mid/macro） |
| **实际代码** | `short*0.4 + mid*0.3 + sentiment*0.2 + macro*0.1` | 4个 |

三个版本的权重、因子数、因子名称全部不同：
- 设计文档提了 short/mid/macro 三维度但没给权重
- MVP plan 砍掉了 mid 和 macro，用 model/sentiment/heat 三因子，model 权重 60%
- 实际代码改回四因子，但 short 权重从 60% 降到 40%

更离谱的是 MVP plan 把 `heat`（讨论热度 0~1）当作独立因子用 `(heat-0.5)*0.1` 参与评分，但实际代码里 heat 不参与打分（只做展示），macro 替代了 heat 的位置。

**影响：** 如果有人按 MVP plan 去理解代码逻辑，会完全搞错。

**建议：** 在实际代码 `signals/scorer.py` 头部加注释说明当前权重设计的理由。

#### 问题23：pushplus API 返回码判断可能有误

**代码：** `push/wechat.py:45`
```python
return data.get("code", -1) == 200
```

**问题：** pushplus.plus 的成功返回码是 `200`，但这个和 HTTP status code 重名了。代码先检查 `resp.status_code == 200`（HTTP层），再检查 `data["code"] == 200`（业务层）。如果 pushplus 改了业务返回码格式（比如用 `0` 表示成功），这里会静默失败。

**更严重的问题：** pushplus 免费版有每日推送次数限制（约200次）。当前系统每天4个时间槽推送 + 风险检查，如果风险检查频繁触发，一天很可能超过200次。**但代码没有任何推送频率统计或限制逻辑。**

设计文档写了 `MAX_PUSH_PER_STOCK_PER_DAY = 2`，但**实际代码从未检查这个限制**。

**解决方案：**
1. 加推送计数器，超过每日限额时降级为日志记录
2. 实现 `MAX_PUSH_PER_STOCK_PER_DAY` 的检查逻辑

#### 问题24：pyproject.toml 依赖声明问题

MVP plan 写的依赖：
```toml
dependencies = [
    "qlib>=0.9.0",
    "httpx>=0.27.0",
    "transformers>=4.40.0",
    "torch>=2.0.0",
    "lightgbm>=4.0.0",
    ...
]
```

实际 pyproject.toml：
```toml
dependencies = [
    "akshare>=1.10.0",
    "ccxt>=4.0.0",
    ...
]
# qlib, lightgbm, torch 都在 optional-dependencies 里
```

**问题：**
1. MVP plan 把 qlib/torch/transformers 放在主依赖里，但实际代码把它们放在 optional。`pip install .` 不会安装 qlib，导致 `import qlib` 在新环境里直接报错
2. MVP plan 列了 `httpx`，但实际代码用的是 `requests`（`SentimentCollector` 里 `import requests`）
3. `baostock` 在哪里都没声明，但 `update_qlib_data.py` 依赖它
4. `snownlp` 如果后续要用也没声明
5. tianshou 的版本约束写的是 `>=0.5.0`，但实际用的是 `2.0.1`，API 完全不同

**解决方案：** 更新 pyproject.toml，把实际运行必需的依赖放到主依赖：
```toml
dependencies = [
    "akshare>=1.10.0",
    "baostock",
    "ccxt>=4.0.0",
    "apscheduler>=3.10.0",
    "requests>=2.31.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0,<2.0",  # Qlib 0.9.7 不兼容 numpy 2.x
    "qlib>=0.9.0",
    "lightgbm>=4.0.0",
    "torch>=2.0.0",
]
```

#### 问题17：设计文档项目结构和实际结构不一致

设计文档写的：
```
factors/
├── quant/           # 量化因子
├── sentiment/       # 舆情因子
└── geopolitical/    # 地缘因子
```

实际结构：
```
factors/
├── quant.py          # 单文件，不是目录
├── sentiment.py      # 单文件，不是目录
└── geopolitical.py   # 单文件，不是目录
```

同样，`models/` 设计写了 `short_term/`、`mid_term/`、`macro/` 子目录，实际都是单文件。这不是大问题，但说明设计文档从未根据实际实现更新过。

#### 问题18：`INSERT OR REPLACE` 在已有推荐的情况下行为不可预期

**代码：** `tracker/verifier.py:52-57`（MVP plan 和实际代码一致）
```python
conn.execute(
    """INSERT OR REPLACE INTO recommendations
       (rec_date, code, name, signal, score, price_at_rec)
       VALUES (?, ?, ?, ?, ?, ?)""",
    ...
)
```

**具体问题：**
- UNIQUE 约束是 `(rec_date, code)`
- 如果 9:20 推了"看多 SH600519"，14:30 sell_check 改成"看空"，再次调用 `record_recommendation` 就会**覆盖**早上的记录
- 5日后验证时，对比的是被覆盖后的信号，**无法追溯原始推荐**
- 当前代码中 sell_check 没有调用 `record_recommendation`，所以这个 bug 暂时不触发，但数据模型设计有缺陷

**解决方案：** 改用 `INSERT OR IGNORE`（保留首次推荐），或者去掉 UNIQUE 约束改用 `(rec_date, code, created_at)` 三列联合主键，保留推荐历史。

**结论：** 原始设计文档有不错的架构思路，但过于理想化，多处承诺未兑现且文档未更新。核心问题是"量化模型"和"舆情分析"这两个最重要的模块在实际实现中都打了大折扣。V2 迭代应优先补齐这两块的短板。

---

## 八、全A数据下载慢问题分析与解决方案

### 8.1 当前瓶颈分析

当前 `update_qlib_data.py` 用 baostock 逐只拉取 CSI300+500（800只）5年日线数据：
- **单只耗时：** ~10秒/只（baostock 单次请求含网络往返）
- **800只总耗时：** ~2.5小时
- **全A股（5800+只）预估：** ~16小时，完全不可接受
- **根因：** baostock 是同步逐只串行请求，没有批量API，且有隐式QPS限制

### 8.2 解决方案对比

| 方案 | 预估耗时 | 成本 | 难度 | 数据质量 |
|------|---------|------|------|---------|
| **方案A：增量更新（推荐）** | 5-10分钟/天 | 免费 | 低 | 高 |
| **方案B：AKShare批量接口** | 30-60分钟全量 | 免费 | 中 | 中 |
| **方案C：多线程并发baostock** | 30-40分钟全量 | 免费 | 中 | 高 |
| **方案D：Qlib官方数据+增量** | 5分钟下载+5分钟增量 | 免费 | 低 | 高 |
| **方案E：TuShare Pro** | 10-20分钟全量 | 积分制 | 低 | 最高 |
| **方案F：本地数据库缓存** | 首次2小时，后续5分钟 | 免费 | 中 | 高 |

### 8.3 推荐方案：D+A组合（Qlib官方基础数据 + baostock增量更新）

**核心思路：** 不要每天拉5年全量，只拉增量。

**Step 1：一次性基础数据（Qlib官方 Yahoo 数据到 2020-09-25）**
```python
from qlib.tests.data import GetData
GetData().qlib_data(target_dir="data/storage/qlib_data", region="cn")
# ~200MB下载，1分钟完成，覆盖 2007-2020，4000+只股票
```

**Step 2：baostock 补齐 2020-10 到今天的数据（一次性，~2小时）**
```python
# 只拉 2020-10-01 ~ today 的数据（~5年）
# 用多线程加速（见方案C）
```

**Step 3：每日增量更新（核心改造，5分钟内完成）**
```python
def update_incremental(self):
    """只拉最近2天的数据，追加到已有bin文件末尾"""

    # 1. 读取 calendar 最新日期
    last_date = self._get_last_calendar_date()

    # 2. 只拉 last_date ~ today 的数据（通常1-2天）
    start = last_date
    end = today

    # 3. 对每只股票，只请求这1-2天的数据
    for code in codes:
        df = fetch_stock_data(code, start, end)  # 1-2行数据，<1秒
        append_to_bin(code, df)  # 追加而非覆盖

    # 4. 更新 calendar
    append_calendar(new_dates)
```

**关键改造点：**

```python
# update_qlib_data.py 改造

def get_last_calendar_date(qlib_dir):
    """获取当前数据的最新日期"""
    cal_file = qlib_dir / "calendars" / "day.txt"
    dates = cal_file.read_text().strip().split("\n")
    return dates[-1]  # e.g. "2026-05-07"

def append_to_bin(inst_dir, df, calendar_dates):
    """将新数据追加到已有bin文件（而非全量覆盖）"""
    for col in df.columns:
        feature_name = col.replace("$", "")
        bin_path = inst_dir / f"{feature_name}.day.bin"

        # 读取已有数据
        existing = np.fromfile(str(bin_path), dtype=np.float32)

        # 新数据对齐到calendar追加
        new_values = df[col].reindex(calendar_dates[-len_new:]).values
        combined = np.append(existing, new_values.astype(np.float32))

        combined.tofile(str(bin_path))

def main():
    last_date = get_last_calendar_date(QLIB_DIR)
    today = datetime.now().strftime("%Y-%m-%d")

    if last_date >= today:
        logger.info("Data already up to date")
        return

    # 只拉增量日期
    logger.info(f"Incremental update: {last_date} → {today}")
    # ... 拉取并追加
```

### 8.4 补充方案C：多线程并发（全量场景加速4-5倍）

首次全量补数据时用多线程，从2.5小时降到30-40分钟：

```python
import concurrent.futures
import baostock as bs

def fetch_one_stock(code, start, end):
    """每个线程独立 login/logout"""
    bs.login()
    df = fetch_stock_data(code, start, end)
    bs.logout()
    return code, df

# 8线程并发
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(fetch_one_stock, c, start, end): c for c in codes}
    for future in concurrent.futures.as_completed(futures):
        code, df = future.result()
        save_to_qlib_format(code, df, QLIB_DIR, calendar_dates)
```

**注意：** baostock 的 `bs.login()` 是全局状态，多线程需要每个线程独立 login/logout。实测8线程稳定，16线程偶尔超时。

### 8.5 补充方案B：AKShare 全市场一次性接口

AKShare 有 `stock_zh_a_spot_em()` 可一次拉全市场实时行情（5800+只，~3秒），但没有批量历史数据接口。变通方案：

```python
# 每日盘后，用 spot 数据作为当天日线（近似）
spot = ak.stock_zh_a_spot_em()
# 取: 代码, 开盘, 最高, 最低, 最新价(收盘), 成交量, 成交额
# 一次获取全市场当天数据，<5秒

# 但缺点：
# - 非精确收盘价（盘中实时价）
# - 缺少前复权处理
# - 只有当天数据，无法补历史
```

**适用场景：** 17:00 盘后快速更新当天数据，精度要求不高时可用。

### 8.6 实施建议

```
立即做（解决当前痛点）：
1. 改 update_qlib_data.py 为增量模式（每日只拉1-2天新数据）
2. 加30秒超时保护（已完成）
3. 日常更新时间 < 10分钟

一次性做（首次数据准备）：
1. 用 Qlib 官方数据作为基础（2007-2020）
2. 用多线程 baostock 补齐 2020-10 到今天（~30分钟）
3. 验证全部 bin 文件对齐 calendar

长期优化（可选）：
1. 接入 TuShare Pro 作为 baostock 备份（更快更稳）
2. 用 parquet/HDF5 替代 Qlib bin 格式做本地缓存
3. AKShare spot 做盘中实时更新，baostock 做盘后精确更新
```

---

---

## 九、CX 文档审查

> 审查三个 cx 前缀文档：
> - `cx-quant-sentiment-deep-research-2026-05-07.md`（调研，341行）
> - `cx-quant-libraries-sentiment-and-masters-research-2026-05-08.md`（补充调研，~400行）
> - `cx-v2-iteration-plan.md`（V2迭代方案，1170行）

### 9.1 总体评价

cx 的工作质量显著高于原始 2026-05-05 设计文档。具体体现在：

**做得好的：**
1. **诊断精准** — cx 独立发现了 Qlib NaN 问题的根因（bin 文件格式、calendar 对齐、instruments 元数据），并实现了 `check_qlib_data_health.py` 和 `smoke_lgb_predict.py` 两个守护脚本，这是本项目最急需的工程保障
2. **staging+promotion 机制** — cx 提出了数据更新先写 staging 再通过健康检查后才 promote 到生产目录，这正是我们 5-07 数据损坏（123/800 只覆盖但 calendar 已推进）的正确修复
3. **舆情定位准确** — "LLM 只做事件修正层，不直接生成 alpha" 和 "结构化事件表 event_impacts" 是非常成熟的工程判断
4. **不替换 Qlib 的决策** — "不换平台，补四个短板"（验证/回测/组合/舆情）是正确的

### 9.2 具体问题

#### CX问题1：TuShare Pro 作为"首选日常数据源"可能不现实

`cx-v2-iteration-plan.md` 第306行：
> "Tushare Pro as the preferred production source. Pull by trade_date to get all-market daily bars in batch."

**问题：**
- TuShare Pro 的积分制在 2025-2026 年多次调整，基础功能需要 2000+ 积分（日线数据需要 120 积分/天的配额）
- TuShare 在 2025-08 曾发生过近一周的服务中断（cx 自己在调研里也提到了）
- 把付费且有中断历史的服务作为生产首选，而把免费稳定的 AKShare/baostock 放 fallback，风险定位反了

**建议：** 日常增量更新首选 AKShare（免费、稳定、有批量接口 `stock_zh_a_spot_em()`），TuShare 作为涨停板/龙虎榜等增值数据的补充源，baostock 做历史回填。

#### CX问题2：MiniQMT/xtquant 列为"serious long-term option"但没评估可行性

第319行：
> "Local terminal/vendor data: Tongdaxin local cache, MiniQMT/QMT, xtquant. Fast and stable."

**问题：** MiniQMT/xtquant 需要券商开户+开通量化交易权限，不是随便就能用的。列在数据源优先级里但完全没讨论准入条件，容易误导。

#### CX问题3：multi-bagger label 设计有严重的前视偏差风险

`cx-v2-iteration-plan.md` 第523-538行：
```
Train labels:
- max_forward_return_6m >= 100%
- max_forward_return_12m >= 300%
- max_forward_return_24m >= 500%
```

**问题：**
- `max_forward_return` 用的是未来最高价而非收盘价。实际交易中你不可能在最高点卖出，这个标签本身就有过度乐观偏差
- 6个月/12个月/24个月的前向窗口在训练时会产生大量重叠样本（2020-01-01 和 2020-01-02 的24个月标签几乎相同），导致特征-标签关系被虚高估计
- 没有提到 Lopez de Prado 的 purged/embargo CV，而这正是处理重叠标签时防止信息泄露的标准方法

**建议：**
- 用 `close_to_close_return` 而非 `max_return`
- 标签窗口不重叠（如按月采样）
- 必须用 purged k-fold CV

#### CX问题4：30 个预测 < 100 最低要求但没给出解决时间表

cx 精确诊断了"30 个有效预测，低于 100 的最低门槛"，但 V2 计划里没有给出明确的修复时间表。Phase 0 说"1-2天"完成，但实际问题是**数据覆盖不够**（121 只股票），而数据覆盖的修复取决于全A数据下载，这在 cx 的方案里被归到"configure a fast source such as Tushare Pro"，但如上所述 TuShare Pro 不一定能快速配置。

**建议：** 明确一个不依赖第三方付费服务的修复路径：用 Qlib 官方 Yahoo 数据做基础，baostock 多线程增量补全，这条路径虽慢但确定能跑通。

#### CX问题5：PyABSA 的推荐缺乏验证

`cx-quant-libraries-sentiment-and-masters-research-2026-05-08.md` 第76行推荐 PyABSA 做"方面级情绪分析"：
> "能把'新能源车利好但锂矿利空'拆成不同 aspect"

**问题：** PyABSA 的训练数据主要来自英文评论（SemEval、Laptop、Restaurant），对中文金融文本没有现成模型。cx 自己也写了"金融中文仍需要自建标注集"，但把它列在推荐里容易让人以为开箱即用。

#### CX问题6：三份文档之间有重复和不一致

- `cx-quant-sentiment-deep-research-2026-05-07.md` 和 `cx-quant-libraries-sentiment-and-masters-research-2026-05-08.md` 大量内容重复（量化库选型、舆情架构、书单几乎相同），第二份说"在第一份基础上补充"但实际上 80% 是复制粘贴
- 两份调研文档的"舆情架构图"结构相同但措辞不同
- 妖股评分公式在 cx-v2 里比两份调研里更详细（多了 label design），但三份文档各自独立，没有交叉引用

**建议：** 合并两份调研为一份，cx-v2 只引用不复制。

#### CX问题7：US/HK 股票扩展设计过早过重

`cx-v2-iteration-plan.md` 第117-236行用了 120 行详细描述 US/HK 股票的数据采集、模型分离、交易日历、汇率处理、scheduler 分时区调度等。

**问题：** 当前 A 股的 Qlib 模型都还没跑通（30/100 预测），在这个阶段花大量篇幅设计美股/港股架构是过早优化。而且 US/HK 扩展的 acceptance criteria（第231行 "US/HK candidates appear in separate report sections"）和当前项目的核心矛盾（"A股模型能不能出有效分数"）完全不相关。

**建议：** US/HK 扩展移到 Phase 3 或更后面，Phase 0-1 专注把 A 股模型跑通。

#### CX问题9：crontab 时间窗口存在竞态 — 数据更新还没完 LGB 就开始训练

**cx 安装的 crontab（实际验证）：**
```
17:00  update_qlib_data.py --universe all --min-health-instruments 4500
17:35  train_lgb.py
17:55  smoke_lgb_predict.py
```

**问题：** 这三个是独立的 cron job，不是串行依赖。17:00 的数据更新用 baostock 拉全A（4500+只），但 baostock 实测 800 只就要 2.5 小时。即使用增量模式（只拉最近1-2天），4500 只 × 每只~2秒 = ~2.5 小时。**17:35 LGB 训练启动时数据更新还在跑。**

cx 的 `nightly_train.py` 是串行的（04:00 跑，内部 data → health → train → smoke），但 17:00-17:55 的三个 cron job 不是串行的。

**实际后果：**
- 17:35 `train_lgb.py` 可能用到半更新状态的 Qlib 数据（部分 bin 已更新、部分还是旧的）
- 17:55 `smoke_lgb_predict.py` 可能测到的不是 17:35 训练的模型（如果训练还没完）
- cx 自己设计的 staging + promotion 机制在 nightly_train 串行流程里有效，但在 crontab 并行场景下被绕过了

**验证：** 比较 cx 的两套调度：
```
nightly_train.py (04:00, 串行):  data → health → train → smoke  ✅ 正确
crontab (17:00-17:55, 并行):     data | train | smoke           ❌ 竞态
```

**解决方案：** 把 17:00-17:55 的三个 cron job 合成一个串行脚本（类似 nightly_train.py 的 after_close_train.py），或者用 flock 加锁：
```bash
# 方案A: 合成一个脚本
17:00  after_close_pipeline.py  # 内部串行: update → health → train → smoke

# 方案B: flock 加锁
17:00  flock /tmp/qlib_update.lock update_qlib_data.py
17:35  flock /tmp/qlib_update.lock train_lgb.py  # 等 update 释放锁才开始
```

#### CX问题10：next_weekday() 不处理中国法定节假日

**代码：** `signals/index_predictor.py:51-55`
```python
def next_weekday(current: date) -> date:
    """Return the next weekday. This is a first pass before exchange calendars."""
    target = current + timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target
```

cx 自己在注释里写了"first pass before exchange calendars"，但这个"first pass"已经进了生产（22:00 推文里展示"明日预测日期"）。

**问题：**
- A 股有大量调休和节假日（如春节、国庆各休7天，但前后周末要补班）
- 周五晚上 22:00 推送"下周一预测"是对的，但国庆前最后一个交易日推送时，`next_weekday` 会返回周一（实际休市），用户看到"预测明日10-08"但10-08不开市
- 这不是小问题 — 用户看到了错误的"明日日期"，会质疑系统专业性

**解决方案：** 从 Qlib calendar 读取交易日历：
```python
def next_trading_day(current: date, calendar_path: Path) -> date:
    cal = [line.strip() for line in calendar_path.read_text().splitlines()]
    current_str = current.strftime("%Y-%m-%d")
    for d in cal:
        if d > current_str:
            return datetime.strptime(d, "%Y-%m-%d").date()
    # fallback
    return next_weekday(current)
```

#### CX问题11：长线推荐用 LGB 5日模型打分 — 概念混淆

**cx-v2-iteration-plan.md 第958行：**
> "Short-term next-day stock change is estimated from LGB's short-term return score"

**实际代码逻辑（从 cx 的 scheduler/jobs.py 提取）：**
```python
mid_score = model_score * 0.70 + (change_pct / 100.0) * 0.20 + liquidity_score * 0.10
long_score = model_score * 0.45 + liquidity_score * 0.35 + max(change_pct, 0.0) / 100.0 * 0.20
```

**问题：** `model_score` 是 LGB 的 **5日前向收益预测**。用它乘以不同权重就叫"中线"和"长线"，这在概念上是错的：
- 5日收益预测高的股票，不代表 1-4 周或 3-24 个月也好
- `long_score` 用 `liquidity_score * 0.35`（流动性占35%权重），但高流动性≠长期好股票（茅台流动性好，但它的长期价值来自 ROE/护城河，不是成交量）
- cx 自己在文档里也写了"Limitation: The long-term stock list is still a longer-horizon observation ranking, not a true fundamental long-term model yet"

**影响：** 用户看到"长线推荐"以为是经过基本面分析的长期价值判断，但实际上是 5 日模型分 × 0.45 + 成交量 × 0.35。这比没有还危险 — 给了不该有的信心。

**解决方案：**
- 短期：长线部分标注"基于短期模型推断，仅供参考"
- 长期：长线需要独立的基本面因子（ROE/营收增速/毛利率/PE 分位数），不能从 5 日模型衍生

#### CX问题15：文档说"baostock 降级为 repair"但 crontab 实际仍在用 baostock 做主源

**cx 文档立场（cx-quant-libraries-sentiment-and-masters-research-2026-05-08.md 第279行）：**
> "baostock 从主链降级成 repair provider"

**实际 crontab（验证于 2026-05-08）：**
```
0 17 * * 1-5 ... update_qlib_data.py --universe all --universe-source baostock ...
```

crontab 用的是 `--universe-source baostock`。虽然代码里写了 `--provider auto`（先试 TuShare，没 token 则 AKShare，再 baostock），但 `--universe-source baostock` 表示股票列表本身还是从 baostock 拉。

**更关键的是：2026-05-08 今日 17:00 的数据更新完全失败。** 日志证据：
```
17:00:01 [INFO] Updating Qlib data: mode=incremental provider=auto universe=csi300
17:00:19 [INFO] Loaded 280 csi300 codes from akshare
17:10:41 ~ 17:11:35 连续 20 条 AKShare failed: Connection aborted
17:11:35 [ERROR] No data fetched; refusing to update Qlib
```

实际流程：baostock 拿 universe → AKShare 拉日线 → AKShare 全部超时 → 没有 baostock fallback → 全部失败。

**讽刺的是：** cx 同一份文档里写了"TuShare Pro 批量日线 > AKShare > baostock repair"的优先级链，但实际 crontab 没有配 TUSHARE_TOKEN，AKShare 连不上时也没 fallback 到 baostock 拉日线。cx 写的三层 fallback 架构在实际运行中只走了一层就放弃了。

#### CX问题16：17:35 LGB 训练在数据更新失败后照样跑了

**日志证据：**
- 17:11:35 数据更新失败：`No data fetched; refusing to update Qlib`
- 17:35:xx LGB 训练照样启动并完成：`Predictions shape: (5040,)` + 模型保存成功

**原因：** crontab 里 17:00/17:35/17:55 是独立 cron job，没有依赖关系。数据更新失败 → LGB 用旧数据训练 → smoke 用旧模型检查 → 全部"通过"。

cx 在 `nightly_train.py`（04:00）里做了正确的串行依赖：
```
if not run_step("Data Update"): return  # 失败则停止
if not run_step("Health Check"): return
if not run_step("LGB Training"): return
```

但这个保护只在 04:00 nightly 流程里有效。17:00-17:55 的 crontab 完全绕过了这些守卫。**cx 精心设计的 staging/health/smoke 防线在日常运行中形同虚设。**

**验证总结：**
```
04:00 nightly_train.py (串行依赖) ← 正确，但每天只跑一次
17:00 update (独立 cron)          ← 今天失败了
17:35 train (独立 cron)           ← 今天用旧数据跑了
17:55 smoke (独立 cron)           ← 今天用旧模型通过了
```

**解决方案：** 把 17:00-17:55 三个 cron 合成一个 `after_close_pipeline.sh`：
```bash
#!/bin/bash
PY=/Users/wangzilu/miniconda3/envs/tianshou/bin/python
CD=/Users/wangzilu/MyProjects/stockPrediction

cd $CD
$PY scripts/update_qlib_data.py ... || exit 1
$PY scripts/check_qlib_data_health.py ... || exit 1
$PY scripts/train_lgb.py || exit 1
$PY scripts/smoke_lgb_predict.py || exit 1
```

#### CX问题17：cx 对 GDELT 的立场前后矛盾

**文档1（05-07）第92行：**
> "RSS / GDELT | 已在用；继续做 22:00 和 09:20 宏观修正"

**文档3（05-08）第338行：**
> "GDELT 不是主舆情链路"

**文档3（05-08）第425行（cx 自我审查时也发现了这个矛盾）：**
> "前面说 GDELT '已在用'，后面又承认 GDELTCollector/GeopoliticalScorer 没接 pipeline"

cx 自己发现了这个矛盾但没有给出最终收敛结论。GDELT 到底用不用？在三份文档里出现了三种立场：
1. "已在用，继续" → 错的，pipeline 没调用
2. "不是主链" → 对的，但那它是什么？
3. "做宏观 tone/time series 补充" → 这是最终立场吗？没有给出接入方案

**收敛建议：** GDELT 的 tone 时序数据适合做地缘风险的量化补充指标（连续值，可回测），但当前 LLM 已经在做定性分析。两者可以共存：GDELT tone 给 LLM 一个数值基线参考，LLM 给出解释和判断。但需要写清接入点和代码。

#### CX问题18：长线推荐的定义在三份文档里不收敛

| 文档 | 长线定义 | 输入 |
|------|---------|------|
| cx-v2 §4 Phase 1 | 3-24 months | revenue/profit/ROE/valuation/institutional |
| cx-v2 §12 实现 | 1-3 月 | `long_score = model_score*0.45 + liquidity*0.35 + change_pct*0.20` |
| cx-08 §4.3 | "longer-horizon observation ranking, not a true fundamental long-term model" | 同上 |

设计说 3-24 个月要用基本面（ROE/营收增速/估值分位数），实现用 5 日模型打分 × 0.45 + 流动性 × 0.35。设计和实现之间的差距没有在推文中告知用户。

**影响：** 用户看到"长线推荐：贵州茅台"以为经过了基本面分析，实际是 5 日 LGB 分 + 成交量。如果用户真的按"长线"持有 3 个月，期间 LGB 分早就变了，但推文暗示这是长期价值判断。

**收敛建议：**
- 短期：推文标注"长线栏目基于短期模型推断，仅供参考"
- 中期：长线部分不展示，直到有基本面因子接入
- 长期：接入营收/利润/ROE/估值等季频数据，训练独立的长期模型

#### CX问题12：Qlib bin 格式理解有误 — cx 说裸数组不对但自己实现也有隐患

cx 在 `cx-quant-libraries-sentiment-and-masters-research-2026-05-08.md` 第330行批评 cc 的 `np.append` 方案：
> "Qlib feature bin 不是裸数组，当前正确格式是 `[start_index, values...]`"

**验证：** 检查 cx 的 `update_qlib_data.py`（1347行），确实实现了 `write_feature_bin()` 写入 `[start_index, values...]` 格式。但问题是：

1. Qlib 官方下载的数据（`GetData().qlib_data()`）其实**就是裸 float32 数组**，没有 start_index 头。cx 的 `check_qlib_data_health.py` 把官方数据判定为"malformed bin"然后做了 repair，但这个"修复"改变了数据格式，和 Qlib 内部读取逻辑是否一致取决于 Qlib 版本
2. Qlib 0.9.x 的 `FileStorageDriver` 读 bin 时，实际上是看 instruments 文件的 start_date 来计算 offset，然后从 bin 里直接按位置读 float32。如果 bin 第一个值是 start_index（整数），Qlib 会把它当成那天的特征值，导致第一天数据出错

**这解释了为什么 cx 修复后虽然 health check 通过了（280 instruments, 100% coverage），但预测值可能有细微偏差。** 不过 cx 的方案确实解决了 NaN 问题，功大于过。

**建议：** 做一次 Qlib 内部读取逻辑的代码 review，确认 `FileStorageDriver` 到底期望裸数组还是带 header 的格式。不同 Qlib 版本可能不同。

#### CX问题13：cx 对 cc 的批评有几处过度

cx 在 `cx-quant-libraries-sentiment-and-masters-research-2026-05-08.md` 第363-476行对 cc 文档做了详细审查。大部分批评有道理，但有几处过度：

**1. "Qlib 官方 Yahoo 数据不能作为生产主链"（P0级否决）**

cx 认为 Yahoo 数据"不一定完美"。但 cc 的建议是用 Yahoo 数据做**基础**（2007-2020），再用 baostock/TuShare 做增量更新到今天。这不是"混拼"，而是分层：Yahoo 提供历史基础（已经对齐了 calendar/instruments），baostock 只补增量。cx 的替代方案（纯 TuShare）需要付费且有宕机风险。

**2. "FinRL/Qbot/AlphaPy 优先级偏高"**

cc 文档把 FinRL 列为调研对象，没有说要立刻接入。cx 批评"不能进入近期主线"是对的，但 cc 也没说要进近期主线。

**3. "论文条目缺少可执行来源"**

cc 的论文列表是调研文档的一部分，不是实施计划。调研文档列论文是为了参考，不需要每篇都有可复现代码。

#### CX问题14：cx 文档自身重复严重 — 第三份文档是对前两份的 patch

`cx-quant-libraries-sentiment-and-masters-research-2026-05-08.md` 的 §4.5-4.8（250行）本质上是对 cc 文档和自己前两份文档的"补丁"。它重复讨论了：
- 数据下载方案（第三次讨论）
- FinGPT/SnowNLP 预期（第三次讨论）
- 妖股评分（第三次讨论）
- 推荐权重（第三次讨论）

三份 cx 文档 + 一份 cc 文档 = ~3500 行，其中约 40% 是重复内容。

**建议：** 合并为一份最终决策文档（~800行），每个决策点只出现一次，标注"采纳/否决/待验证"。

#### CX问题8：FinRobot 推荐但定位模糊

cx 在两个调研文档和 v2 plan 中多次推荐 FinRobot：
> "金融 LLM Agent 平台"、"多 Agent 研究"、"作为解释和归因层"

**问题：** FinRobot 是一个偏研究/demo 的项目（GitHub stars ~1k），其"多 Agent"能力实际上是多个 LLM prompt 串联，不是真正的分布式 agent 系统。把它列在和 FinRL（14k stars）同级别不合适。而且 cx 自己也写了"不应该把 Agent 输出直接当 alpha"，那它在系统里的具体接入点是什么？

**建议：** FinRobot 从推荐列表降级为"参考项目"，不进入任何 Phase 的实施计划。

### 9.3 cx vs cc 文档对比

| 维度 | cx 文档 | cc 文档（本文） |
|------|---------|--------------|
| 诊断深度 | 深，精准定位了 NaN 根因并修复 | 浅，发现问题但修复不彻底 |
| 工程落地 | 强，写了具体脚本和 acceptance criteria | 中，给了代码骨架但没跑通 |
| 范围控制 | 弱，US/HK/FinRobot 等过早铺开 | 强，聚焦 7 个改进项 |
| 重复度 | 高，三份文档大量重复 | 低，一份文档 |
| 数据源评估 | TuShare Pro 过于乐观 | 更务实（AKShare 为主） |
| 舆情架构 | 更成熟（事件冲击表 > 情绪分） | 更粗（SnowNLP + FinGPT） |
| 妖股方案 | 有 label design 但前视偏差 | 有评分公式但无 label design |

**结论：** cx 的 V2 迭代计划整体质量高，尤其是 Phase 0（hotfix/observability）部分已经落地且有效解决了 Qlib NaN 问题。主要问题是范围过大（US/HK 过早）、数据源优先级不够务实（TuShare 依赖）、三份文档冗余。建议合并 cx 和 cc 的优势：用 cx 的 Phase 0 守护脚本 + cc 的涨停板/妖股评分 + cx 的舆情事件冲击表架构。

### 9.4 CX 问题加强证据

#### 证据1：TuShare Pro 2025-08 断网事件全记录

**事实链条（来源：知乎、财联社、证券时报、界面新闻）：**

1. **2025-08-18**：TuShare 突然全线断网，所有数据接口不可用，用户无法登录
2. **断网原因**：数据托管 IDC 机房和运营方发生商业纠纷（非技术故障），是**不可预测的商业风险**
3. **恢复时间**：08-21 开始恢复，08-23 才完全恢复，**宕机 5 天**
4. **行业反应**：财联社标题 ["TusharePro突发停运引量化圈震荡，免费数据源成'避风港'"](https://www.cls.cn/detail/2125736)
5. **核心教训**：证券时报 ["量化'宠儿'突发纠纷断网"](https://www.stcn.com/article/detail/3149814.html)，行业反思单一数据源依赖风险

**cx 的决策问题：** cx 在 v2 plan 中把 TuShare Pro 列为"preferred production source"（第一优先级），但在调研文档中自己也提到了断网事件。这两个判断是矛盾的 — 知道一个服务曾宕机 5 天，还把它放在第一位，说明选型决策没有充分考虑生产可靠性。

**补充验证：** TuShare Pro 积分体系要求 5000+ 积分才能获得"较高频次"权限。新用户从 0 积分开始，获取 5000 积分需要邀请/充值/贡献。日线数据每天 120 积分配额，超额返回 4003 错误码。这意味着**即使 TuShare 没宕机，新用户也可能因为积分不够而无法正常使用**。

**正确优先级应该是：**
```
日常增量：baostock（免费、稳定、无积分限制）
全市场 spot：AKShare stock_zh_a_spot_em()（3秒拿全市场，但连接不稳定需重试）
增值数据：TuShare Pro（涨停板/龙虎榜/融资融券，作为增值补充，不做基础依赖）
历史回填：Qlib 官方 Yahoo 数据（一次性基础）
```

#### 证据2：max_forward_return 标签的前视偏差 — 学术界已有明确警告

**来自学术论文和实践的证据：**

1. **Lopez de Prado《AFML》第3章** 明确指出：用 `max_return` 做标签会导致"你在训练模型预测一个你实际上无法捕获的收益"。因为 max_return 假设你能在最高点卖出，但现实中最高点只有事后才知道

2. **Columbia Data Science 研究** ["Assessing Look-Ahead Bias in Stock Return Predictions"](https://datascience.columbia.edu/wp-content/uploads/2024/01/P021_FB_PosterSession_Fall2023.pdf) 证明：使用前瞻性标签的模型在回测中表现远好于实盘，因为训练信号中包含了未来信息

3. **cx 标签设计的具体问题：**
   ```
   max_forward_return_6m >= 100%  → 股价在6个月内曾翻倍
   max_forward_return_12m >= 300% → 股价在12个月内曾涨3倍
   ```
   - 一只股票 1 月涨 3 倍，3 月跌回原价。用 max_return 标记它为"3倍股"，但实际持有收益为 0
   - 这不是极端情况 — A 股妖股经常暴涨后暴跌，max_return 标签会系统性高估妖股的可投资性

4. **重叠窗口问题：** 2020-01-01 和 2020-01-02 的 `max_forward_return_12m` 几乎完全相同（只差一天），但被当作两个独立样本。这导致模型在交叉验证时看到的"不同"样本实际上高度相关，IC/AUC 被虚高估计

**正确的标签设计：**
- 用 `close_to_close_return`（持有 N 个月后的收盘价收益）替代 `max_return`
- 或用 Lopez de Prado 的 triple-barrier（止盈/止损/到期三重门槛）
- 标签窗口不重叠（按月采样，不是逐日滚动）
- 必须用 purged k-fold CV（去除训练集和测试集之间的重叠窗口样本）

#### 证据3：AKShare 也不稳定 — 实测连接失败

刚才实际测试 AKShare `stock_zh_a_spot_em()` 时也报了连接错误：
```
RemoteDisconnected: Remote end closed connection without response
```

这说明**所有免费数据源都有不稳定时段**。正确的工程方案不是"选一个最好的"，而是多源 fallback + 本地缓存：
1. 首选拉取成功后立刻写入本地 SQLite/parquet 缓存
2. 如果首选源失败，fallback 到下一个
3. 如果所有源都失败，用本地缓存的最近数据（标注 stale）

### 9.5 从 CX 吸收的优秀设计

以下是 cx 文档中值得直接采纳的设计，整合到本项目 V2 路线中。

#### 吸收1：Phase 0 守护脚本体系（已落地，直接使用）

cx 已经实现了以下脚本，经验证有效：

| 脚本 | 功能 | 状态 |
|------|------|------|
| `scripts/check_qlib_data_health.py` | 验证 calendar/bin/instruments 一致性 | 已实现，9320 字节 |
| `scripts/smoke_lgb_predict.py` | 验证 LGB 推理链路能否产出有效预测 | 已实现，5190 字节 |
| `models/lgb_cache.py` | 原子化 JSON 缓存 + 有效性校验 | 已实现 |
| `scheduler/job_status.py` | Job 执行状态持久化 | 已实现 |

**采纳方式：** 直接使用，不重复建设。在 nightly pipeline 中串联：
```
data update → health check → train → smoke predict → promote model
```

#### 吸收2：staging + promotion 机制

cx 提出并实现了数据更新的 staging 机制：
- 新数据先写到 `data/storage/qlib_data_staging/cn_data`
- 通过健康检查（≥95% instruments 覆盖）后才 promote 到生产目录
- 失败时保留昨天的生产数据，不会降级

这正是我们 5-07 数据损坏（123/800 覆盖但已推进 calendar）的正确修复。**直接采纳。**

#### 吸收3：结构化事件冲击表（event_impacts）

cx 的舆情架构比 cc 更成熟。核心设计：

```python
# 每条舆情/新闻产出的结构化记录
{
    "target_type": "market|sector|stock",  # 影响层级
    "impact": -1.0 ~ 1.0,                  # 冲击方向和强度
    "confidence": 0.0 ~ 1.0,               # 置信度
    "decay_hours": 24,                      # 时效衰减
    "source_quality": 0.8,                  # 来源可信度
    "hard_override": "avoid|sell|reduce|none"  # 硬性否决
}
```

**优于 cc 方案的地方：**
- `decay_hours` 解决了"昨天的新闻不应该永久影响今天的分数"
- `hard_override` 解决了"监管处罚/财务造假类硬负面一票否决"
- `source_quality` 解决了"路边社 vs 新华社同等权重"的问题

**采纳方式：** 在 cc 的 Phase 2 舆情升级中实现这个 schema，替代简单的 sentiment_score。

#### 吸收4：推荐降级标识

cx 在推文中加入了模型状态标识：
```
Qlib: OK, 300 predictions, latest 2026-05-08
Qlib: DEGRADED, reason=NaN, fallback=intraday change_pct
```

这解决了"用户以为是模型推荐但实际是涨跌幅排序"的信息不对称。**直接采纳。**

#### 吸收5：三维度推荐引擎的数据契约

cx 定义了结构化信号对象：
```python
{
    "code": "SH600519",
    "horizon": "short|mid|long",
    "score": 0.0,
    "entry_zone": [low, high],
    "stop_loss": 0.0,
    "take_profit": 0.0,
    "holding_days": 5,
    "invalidation_condition": "..."
}
```

比当前的 `Recommendation` dataclass 多了 `entry_zone`（买入区间）、`invalidation_condition`（信号失效条件），这让推荐从"拍脑袋说看多"变成"可执行的交易计划"。**采纳核心字段。**

#### 吸收6：妖股分类四象限

cx 把妖股候选分成四类：
- **潜伏型**：低关注度 + 基本面改善 → 最有价值，早期发现
- **加速型**：突破 + 量/业绩确认 → 确认信号，可入场
- **兑现型**：已拥挤、高风险 → 已晚，不追
- **排除型**：纯炒作、ST/退市风险 → 一票否决

比 cc 的单一 `monster_score` 更细，可以给用户更明确的操作指引。**采纳这个分类框架。**

### 9.6 CC + CX 合并路线图

综合两份文档的优劣，推荐的统一路线：

```
Phase 0（cx 已完成）：守护脚本 + staging + 降级标识
  ✅ check_qlib_data_health.py
  ✅ smoke_lgb_predict.py
  ✅ lgb_cache.py + job_status.py
  ✅ staging + promotion
  ❌ 待修：数据覆盖还是 121 只，需要全A回填

Phase 1（1周）：数据覆盖 + 基础舆情
  - Qlib 官方 Yahoo 数据做基础
  - baostock 多线程增量补全到今天
  - SnowNLP + AKShare 替代失效的雪球爬虫
  - 涨停板数据采集器（AKShare stock_zt_pool_em）

Phase 2（1周）：模型跑通 + 信号验证
  - LGB 重训 + smoke 通过 ≥100 predictions
  - 结构化事件冲击表（吸收 cx event_impacts）
  - Triple-barrier 标签替代简单前向收益
  - vectorbt 回测 14:30/22:00/09:20 信号

Phase 3（1周）：妖股雷达 + 组合层
  - 妖股复合评分模块 + 四象限分类
  - 板块热度追踪器
  - PyPortfolioOpt HRP 做仓位建议
  - 三维度推荐（短/中/长 + entry/stop/take-profit）

Phase 4（后续）：
  - RL Transformer 序列输入修复
  - FinGPT 本地部署（如有GPU）
  - A股真实规则回测（RQAlpha）
  - US/HK 扩展（此时才合适）
```

---

---

## 十、CC vs CX 最终分歧收敛表

经过多轮互相审查后，大部分争议已经收敛。以下是**已收敛共识**和**仍有分歧的 4 个核心议题**的最终判断。

### 10.1 已收敛共识（双方同意）

| 议题 | 共识 |
|------|------|
| 不替换 Qlib | 保留 Qlib 做 A 股日频 alpha 主干 |
| LLM 不直接荐股 | LLM 只做事件修正层，不生成原始 alpha |
| staging + health gate | 数据更新必须先写 staging，通过健康检查后才 promote |
| RL 暂不上生产 | `deployed=false`，先离线验证 |
| 妖股单独建模 | 不和普通推荐混排，单独打分、风控、展示 |
| 多线程 baostock 不做主方案 | session 不稳定，只做缺口修复 |
| VectorBT 是 P1 验证工具 | 用于信号回测，不替代 A 股撮合 |
| 推荐降级标识 | 推文必须显示 Qlib 状态（OK/DEGRADED） |
| entry/stop/take-profit 字段 | 推荐从"看多"升级为可执行交易计划 |
| 妖股四象限分类 | 潜伏/加速/兑现/排除 比单一分数更可操作 |
| next_weekday() 需要交易日历 | 必须处理法定节假日和调休 |
| 17:00 crontab 必须串行化 | 合并为 after_close_pipeline，数据失败不训练 |

### 10.2 仍有分歧的 4 个核心议题

#### 分歧1：数据源第一优先级 — TuShare 还是 AKShare/baostock？

| | cc 立场 | cx 立场 |
|-|---------|---------|
| 主张 | AKShare 免费稳定做主源，TuShare 做增值补充 | TuShare 批量日线做主源，AKShare 备选，baostock 只 repair |
| 论据 | TuShare 2025-08 宕机5天；积分制有门槛；免费优先 | baostock 串行太慢；`stock_zh_a_spot_em()` 不是日线；TuShare 按 trade_date 批量最快 |
| 反驳 | cx 自己的 crontab 用的是 baostock universe + AKShare 日线（今天全失败了） | cc 混淆了 spot 和日线的区别 |

**最终判断：双方都有盲点。**
- cc 对 TuShare 风险的警告有道理（宕机+积分），但把 AKShare spot 当日线主源是错的
- cx 对 TuShare 批量能力的评估有道理，但实际 crontab 没配 TUSHARE_TOKEN，fallback 到 AKShare 后今天全挂了
- **收敛方案：** `provider auto` + 所有源等权 fallback + 本地 manifest 缓存 + 任何一个源跑通就够。不押注单一源。实际跑通的路径比"谁理论上最好"重要。

#### 分歧2：Qlib bin 格式 — 裸数组 vs `[start_index, values...]`

| | cc 立场 | cx 立场 |
|-|---------|---------|
| 主张 | Qlib 官方数据是裸 float32 数组，加 start_index 可能让 Qlib 读错 | 本地 Qlib 0.9.7 的 `FileFeatureStorage` 就是读写 `[start_index, values...]` |
| 论据 | `GetData().qlib_data()` 下载的就是裸数组 | 本地代码验证：`write()` 写 `np.hstack([index, data_array])`，`__getitem__` 跳过 4 字节头 |

**最终判断：cx 在当前环境（Qlib 0.9.7）上是对的。**

cx 做了代码级验证（`FileFeatureStorage.start_index` 读第一个 float，`__getitem__` seek 跳过头），cc 没有做。但 cc 的担忧也不无道理 — Qlib 官方下载的数据确实是裸数组，说明不同版本或不同下载工具产出的格式可能不同。

**收敛方案：** 以本地 Qlib 0.9.7 的 `[start_index, values...]` 为准。cc 的 `np.append` 方案删除。如果未来升级 Qlib 版本，需要先跑 bin 格式兼容性测试。

#### 分歧3：LGB 验收门槛 — 100 还是 4500？

| | cc 立场 | cx 立场 |
|-|---------|---------|
| 主张 | cc §9.6 写 "smoke 通过 ≥100 predictions" | 生产门槛已经是 `LGB_MIN_PREDICTIONS=4500` |
| 论据 | 先让系统跑起来再提高覆盖 | 100 只覆盖全 A 推荐等于盲推 |

**最终判断：cx 对。**

`config/settings.py` 当前配置是 `LGB_MIN_PREDICTIONS=4500`。cc 还在用 100 是因为写文档时参考的是旧版代码。但 cx 自己的 smoke 日志也显示 280 predictions 通过了（应该被 4500 挡住），说明某些路径的门槛配置不一致。

**收敛方案：** 统一所有路径的验收门槛为 4500。不允许 100 或 280 写入生产 cache。研究/调试场景可以用 `--min-predictions 100` 覆盖，但必须标注 `research_only`。

#### 分歧4：Yahoo 基础数据 + baostock 增量 能不能做生产？

| | cc 立场 | cx 立场 |
|-|---------|---------|
| 主张 | Yahoo 做历史基础(2007-2020) + baostock 补增量到今天，这是不依赖付费服务的确定路径 | Yahoo 和 baostock 混用会引入复权口径和字段差异，不适合做生产 |
| 论据 | 不需要 TuShare 积分/付费就能跑通 | 复权方式不同（Yahoo 用后复权，baostock 可选前/后/不复权），calendar 格式有差异 |

**最终判断：两边各有道理，需要加条件。**
- cc 说的"确定能跑通"是对的 — 这是当前唯一不需要任何付费/积分就能获得全A历史+增量的路径
- cx 说的"复权口径差异"是真实风险 — Alpha158 因子对复权方式敏感，混用可能产出看似合理但实际偏差的预测
- **收敛方案：** Yahoo 数据可以做**研究 bootstrap**（快速搭环境、验证 pipeline），但生产模型训练必须用统一口径的数据。如果用 baostock 补增量，必须统一前复权（adjustflag=2）并在 health check 中验证复权一致性。

### 10.3 总结

| 分歧 | 结论 | 行动 |
|------|------|------|
| 数据源优先级 | 不押单一源，provider auto + fallback | 实现真正的多源自动切换，别只在文档里写 |
| Qlib bin 格式 | 按 0.9.7 的 `[start_index, values...]` | 删除 cc 的 np.append 方案 |
| LGB 验收门槛 | 生产 4500，研究可 100 但必须标注 | 统一所有路径的配置 |
| Yahoo + baostock | 研究 bootstrap 可以，生产需要统一复权 | 加复权一致性检查 |

**4 个分歧中 3 个（bin 格式、LGB 门槛、Yahoo 定位）已经有明确收敛方案。唯一真正悬而未决的是"数据源第一优先级"，但这个问题的正确答案不是"选 A 还是 B"，而是"建 fallback 链"。**

### 10.4 cx 具体错误的证据链 — 为什么 cx 的数据源方案在实践中失败了

以下不是"观点分歧"，而是用代码和日志证明的**工程事实**。

#### 事实1：TuShare 根本不可用 — 没装、没配、没 token

cx 在三份文档中反复主张"TuShare Pro 批量日线做首选"，但：

```
证据A: tushare 没有安装
$ python -c "import tushare"
→ ModuleNotFoundError: No module named 'tushare'

证据B: .env 里没有 TUSHARE_TOKEN
$ grep TUSHARE .env
→ 无输出

证据C: settings.py 没有 TUSHARE 配置
$ grep TUSHARE config/settings.py
→ 无输出
```

**论点：** cx 把一个没安装、没配置、没 token 的库写成"首选生产数据源"，这不是技术选型，这是纸上谈兵。cx 的 `provider auto` 逻辑第一步就是 `if os.environ.get("TUSHARE_TOKEN")`，永远跳过 TuShare 直接走 AKShare。cx 写了 700 行 TuShare 相关的代码和文档，**在当前环境一行都没执行过**。

#### 事实2：cx 的 fallback 链有 bug — AKShare 全失败但没触发 baostock

cx 设计的 fallback 链：`TuShare → AKShare → baostock`

2026-05-08 17:00 的实际执行：
```
17:00:01 [INFO] Trying AKShare provider
17:00:19 [INFO] Loaded 280 csi300 codes from akshare  # universe 加载成功
17:10:41 ~ 17:11:35  连续 280 条 AKShare failed: Connection aborted
17:11:35 [ERROR] No data fetched; refusing to update Qlib
```

**关键：日志中没有 "Falling back to baostock" 这一行。**

原因在 cx 的代码 `fetch_with_akshare()` 中：
```python
for i, (code, start_date) in enumerate(sorted(start_by_code.items())):
    try:
        raw = ak.stock_zh_a_hist(...)
    except Exception as exc:
        logger.warning("AKShare failed for %s: %s", code, exc)
    # 只在 i >= 100 且成功率 < 20% 时才 raise 触发 fallback
    if i >= 100 and len(data) / i < 0.20:
        raise RuntimeError("AKShare success rate too low")
return data  # 返回空 dict，不抛异常
```

280 只全部失败时的实际路径：
- 前 100 只：每只都 catch 了异常，`len(data)=0, i=100, 0/100=0 < 0.20` → **应该触发 raise**
- 但这个 raise 在 try/except 外面，只在 for 循环的 progress 检查点触发

等等，让我再看看 —— `i >= 100 and len(data) / i < 0.20`，第 100 只时 `0/100 = 0 < 0.20` 确实满足条件。那为什么没有 fallback？

```
# auto 模式的 fallback 逻辑
try:
    return "akshare", fetch_with_akshare(...)
except Exception as exc:
    logger.warning("AKShare provider unavailable: %s", exc)
# 到这里才 fallback 到 baostock
logger.info("Falling back to baostock provider")
return "baostock", fetch_with_baostock(...)
```

如果 `fetch_with_akshare` 在第 100 只时 raise 了，auto 应该 catch 住并 fallback。但日志没有 "Falling back" 和 "AKShare provider unavailable" 这两行。**说明 RuntimeError 没有被抛出。**

可能原因：280 只中前面几只成功了（`Connection aborted` 是间歇性的），成功率刚好 > 20%，没触发低成功率检查。但最终返回的 data 太少，被下游 `No data fetched` 拒绝了。

**论点：** cx 的 fallback 链设计有逻辑漏洞 — 它只在"连续大量失败"时触发 fallback，但如果 AKShare"偶尔成功几只"（成功率 > 20%），就不会 fallback 到 baostock，最终产出一个"不空但覆盖严重不足"的结果。这比直接失败更危险 — **因为少量成功让系统以为 AKShare 还行，但实际覆盖远不够生产要求。**

#### 事实3：baostock 是目前唯一跑通过全量更新的数据源

所有历史数据更新的实际结果：

| 日期 | 数据源 | 结果 |
|------|--------|------|
| 05-07 00:10-10:07 | baostock（手动 cc 触发） | **798/800 成功** ✅ |
| 05-07 02:00-03:00 | baostock（nightly_train） | 超时（1小时限制太短） ❌ |
| 05-07 17:00 | AKShare（cx crontab auto） | 123/800 成功 ⚠️ |
| 05-08 17:00 | AKShare（cx crontab auto） | **0/280 成功** ❌ |
| TuShare | 从未执行过 | N/A |

**论点：** cx 说"baostock 只做 repair"，但实际上 baostock 是唯一一个曾经成功拉到 798 只数据的源。AKShare 的两次尝试分别是 123/800 和 0/280。TuShare 从未跑过。cx 把唯一跑通的源降级为"repair only"，把从未跑过的源升为"preferred"，这是典型的**理论选型和实际运行脱节**。

#### 事实4：cx 的 4500+ 预测门槛自己也没达到

cx 把 `LGB_MIN_PREDICTIONS` 设为 4500，但：

```
证据: smoke_lgb_predict.py 最近一次结果
- prediction count: 280
- finite prediction count: 280
- result: PASS（写入了生产 cache）
```

280 < 4500，但 smoke 通过了并写入了生产 cache。这说明某些路径的门槛和文档不一致 — cx 文档说 4500 是硬门槛，但实际 smoke 脚本可能用了更低的阈值或被绕过了。

**论点：** cx 批评 cc 用 100 作为门槛，但自己的系统在 280 预测时也通过了。标准不统一，文档和代码不一致。

#### 事实5：cx 的"长线推荐"让用户承担了未标注的风险

cx v2 plan 设计：
> "Long term: 3-24 months, business quality, growth acceleration, valuation"

cx 实际实现：
```python
long_score = model_score * 0.45 + liquidity_score * 0.35 + change_pct * 0.20
```

cx 自己也在文档里写了"not a true fundamental long-term model yet"和"should be replaced"。但推文中仍然展示"长线前五"给用户，**没有任何标注说这不是真正的长线分析**。

**论点：** 如果你知道一个功能不是它声称的样子，你应该要么不展示，要么标注"仅供参考"。cx 选择了既展示又不标注，这对用户不负责任。用 5 日 LGB 分 × 0.45 + 成交量 × 0.35 排出来的"长线推荐"可能让用户长期持有一只短期动量衰减后就下跌的票。

### 10.5 cc 自身论点的修正 — cx 反驳成立的部分

诚实地说，cx 在 §4.9-4.10 对 cc 的回应中有几处反驳是正确的。cc 不应该回避这些。

#### cc 错误1："baostock 应做日常主源" — 逻辑跳跃

cc 的推理链：TuShare 有宕机风险 → 所以 baostock 应做主源。

cx 的反驳（§4.10 第577行）成立：**"TuShare 不能单点依赖"只能推出"不能单押 TuShare"，不能推出"应该单押 baostock"。** baostock 全A串行要 16 小时，日志出现过 `10002007: 网络接收错误`，会话稳定性也不好。cc 批评 TuShare 不可靠，但把同样不可靠的 baostock 抬到主源位置，是双重标准。

**修正：** cc 不应主张 baostock 做主源。正确结论是"所有源都不可靠，必须多源 fallback + 本地缓存"。

#### cc 错误2：混淆了 `--universe-source` 和 `--provider`

cc 在 §9.4 CX问题15 中写"crontab 用 `--universe-source baostock` 说明 baostock 仍是行情主源"。

cx 的反驳（§4.10 第582行）成立：**`--universe-source` 是股票列表来源，`--provider` 是日线行情来源，两个不同参数。** crontab 传了 `--universe-source baostock`（用 baostock 的 CSI300 成分股列表），但没传 `--provider baostock`（日线默认 auto）。cc 混淆概念来证明"baostock 是行情主源"，推理不严谨。

**修正：** 准确表述为"crontab 用 baostock 拉股票列表，用 auto（先试 AKShare）拉日线行情"。"cron 配置与文档不一致"的批评仍然成立，但"baostock 是行情主源"不成立。

#### cc 错误3：AKShare spot 做增量主源 — 自相矛盾

cc §8.5 承认 `stock_zh_a_spot_em()` 缺少前复权、只有当天、非精确收盘价，但 §9.2 又建议做日常增量主源。cx 反驳成立：**spot 快照不是复权日线，不能写入训练数据。**

**修正：** spot 只适用于推文行情展示和动量因子，不做训练数据增量。

#### cc 错误4：Qlib bin 格式 — cx 用源码证明了

cc 质疑 `[start_index, values...]` 会让 Qlib 读错。cx 用 Qlib 0.9.7 源码反驳：
- `FileFeatureStorage.start_index`: 读文件前 4 字节
- `__getitem__`: `seek(... + 4)` 跳过头
- `write()`: `np.hstack([index, data_array])`
- 本地 `close.day.bin` 第一项是 `4944`（start_index），不是价格

**修正：** cc 在这个技术点上错了。`np.append` 方案会破坏 Qlib 格式。

#### cc 仍然正确的论点

以上修正不影响以下论点：
1. **17:00 crontab 竞态** — cx 承认应吸收
2. **fallback 链 05-08 实际失败** — AKShare 280 只全挂没 fallback 到 baostock
3. **TuShare 没安装/没配置** — "首选"从未执行过
4. **长线推荐用 5 日模型** — cx 也承认应改名
5. **next_weekday() 不处理节假日** — cx 也承认
6. **280 预测写入生产 cache** — 低于文档声明的 4500 门槛

### 10.6 最终判决

| 议题 | cc 对/错 | cx 对/错 | 最终结论 |
|------|---------|---------|---------|
| TuShare 做首选 | cc 对（没装/没配） | cx 错（脱离实际） | 不押单一源 |
| baostock 做主源 | **cc 错**（逻辑跳跃） | cx 对（太慢） | 多源 fallback |
| AKShare spot 做训练日线 | **cc 错**（自相矛盾） | cx 对 | spot 只做展示 |
| Qlib bin 格式 | **cc 错**（没看源码） | cx 对 | `[start_index, values...]` |
| fallback 链实际工作 | cc 对（日志铁证） | **cx 错**（代码 bug） | 修 fallback 逻辑 |
| 17:00 crontab 竞态 | cc 对 | **cx 错** | 合并串行 pipeline |
| LGB 门槛 100 vs 4500 | **cc 错**（用旧配置） | cx 对 | 统一 4500 |
| 280 预测写入 cache | cc 对 | **cx 错** | 修 smoke 门槛 |
| 长线推荐误导 | cc 对 | cx 也承认 | 改名"观察榜" |
| 节假日交易日历 | cc 对 | cx 也承认 | 用 Qlib calendar |
| Yahoo 做研究基础 | cc 对 | cx 有道理（复权） | 研究可以，生产统一口径 |

**总比分：cc 拿 6 分（TuShare 没装、fallback bug、crontab 竞态、280 写 cache、长线误导、节假日），cc 犯了 4 个错（baostock 主源、spot 做训练、bin 格式、100 门槛）。cx 拿 4 分（bin 格式、spot≠日线、baostock 不能做主源、门槛已改 4500），cx 犯了 4 个错（TuShare 纸上谈兵、fallback bug、crontab 竞态、280 放行）。**

双方各有 4 个明确错误，说明互相审查有价值 — 单独一个 AI 的盲点可以被另一个发现。但 ~5000 行文档中 40% 是互相辩论，工程决策被淹没了。下一步应提取一份 200 行的执行决策文档。

---

| 改进 | 推荐质量提升 | 妖股发现能力 | 风控改善 |
|------|-------------|-------------|---------|
| 涨停板数据 | - | +++++ | ++ |
| 妖股评分 | + | +++++ | +++ |
| SnowNLP舆情 | ++ | + | + |
| FinGPT | +++ | + | + |
| 板块热度 | ++ | ++++ | ++ |
| Triple-barrier | +++ | + | ++ |
| Transformer序列 | ++ | + | - |
