# Deep Research: Chinese A-Share Quant & Trading Strategies (2024-2026)

*Compiled: 2026-05-08 | Sources: Web search across Chinese and English financial media*

---

## 1. Top Chinese A-Share Quant Fund Strategies

### 1.1 Industry Scale

As of March 2026, the combined AUM of Chinese public and private quant equity funds approached **2 trillion RMB** (~$280B), with private quant funds at approximately **1.46 trillion RMB**. The industry has undergone massive consolidation -- the top 4 firms ("量化四天王") each manage 800-900 billion RMB and are approaching the 1,000-billion threshold.

### 1.2 The Big Players: Detailed Strategy Breakdown

#### Huanfang Quantitative (幻方量化) -- AUM: ~900B RMB, 2025 Return: 56.55%

**Strategy Evolution:** Multi-factor model -> Deep learning -> Integrated framework (3 phases)

**Factor Composition:**
- **Price-Volume factors (40%):** Adaptive momentum, liquidity premium, short-cycle prediction signals
- **Fundamental factors (30%):** Unexpected earnings revisions, cash flow quality, accrual anomalies
- **Alternative data factors (20%):** Satellite image analysis, social media sentiment mining
- **Other/proprietary (10%):** Undisclosed

**Technology Stack:**
- "Firefly 1" cluster: 1,100 GPUs, ~200M RMB investment
- "Firefly 2" cluster (2021): ~10,000 NVIDIA A100 GPUs, ~1B RMB investment, 96% cluster utilization, 8.0 TB/s read bandwidth
- Self-developed FPGA trading cards for nanosecond-level execution
- Order-to-fill latency: **800 nanoseconds**
- First DL-generated trading position deployed: October 21, 2016
- Fully transitioned to deep learning for all trading: 2017

**Key Insight:** Huanfang is the parent company of DeepSeek (the famous LLM). Their quant-to-AI pipeline is bidirectional -- LLM research feeds back into trading models.

**Sources:** [幻方量化 AI 投资方法技术拆解](https://blog.cifangquant.com/post/61.html), [DeepSeek策略曝光](https://deepseek.csdn.net/67ab1dc979aaf67875cb99db.html)

---

#### Jiukun Investment (九坤投资) -- AUM: ~800B RMB

**Founded:** 2012 by Wang Chen (王琛) and Yao Qicong (姚齐聪), both returned from overseas quant roles.

**Strategy Coverage:**
- Stock long-only (指增)
- Market neutral (对冲)
- Managed futures / CTA
- Multi-strategy composite

**Factor Library:**
- Price-volume factors (tick-level and daily)
- Fundamental factors (earnings, balance sheet quality)
- Alternative data factors (web scraping, NLP-based)
- Machine learning-generated factors (auto-mined)

**Technology:** Full-pipeline AI integration -- data analysis, factor mining, strategy construction, trade execution all use ML/AI. Emphasis on **low-correlation factor accumulation** (borrowing from Western hedge fund methodology).

**Source:** [九坤投资：一群用公式和算法战胜市场的"恒温动物"](https://36kr.com/p/1199745566018057)

---

#### Minghe/Minghui Investment (明汯投资) -- AUM: ~800B RMB

**Strategy:** Multi-cycle, multi-frequency, multi-strategy coverage. Known for **full-market stock selection** (全市场选股) rather than index-constrained selection.

**Key differentiator:** Covers multiple time horizons simultaneously -- intraday, short-term (days), medium-term (weeks).

**Source:** [明汯投资](https://www.21jingji.com/article/20260408/herald/97cca928f9ca042393cef00e66cc88cb.html)

---

#### Yanfu Investment (衍复投资) -- AUM: ~800B RMB

**Strategy Focus:** Alpha factor selection with **strict beta exposure control** (market-neutral emphasis).

**Team:** Core investment research team from Peking University, Tsinghua, Ivy League schools with experience at top US hedge funds. Tech team from leading internet companies.

**Key approach:** When new factors have high correlation with existing ones, they **fuse/upgrade** existing factors. Low-correlation new factors are added to the library. This prevents overfitting while expanding the factor zoo.

**Source:** [衍复投资晋升百亿量化私募](https://www.zhihu.com/question/429011298)

---

#### Lingjun Investment (灵均投资) -- 2025 Return: **73.51%** (CHAMPION)

**Strategy:** Quantitative stock selection + index futures hedging for market-neutral exposure.

**Top product:** "灵均信芳量化选股领航2号" returned **79.94%** in 2025.

**Approach:** AI-driven real-time coverage of all ~5,000+ A-share stocks, rapid rebalancing to capture short-term trading opportunities.

**Source:** [灵均投资"夺冠"](https://www.stcn.com/article/detail/3591805.html)

---

#### Chengqi Asset (诚奇资产) -- AUM: ~470B RMB

**Founded:** 2013 in Shenzhen. Two co-founders previously worked at **Millennium Management** and **WorldQuant** (top US quant firms).

**Strategy evolution:**
- Pre-2023: Traditional multi-factor + statistical arbitrage
- Post-2023: Transitioned to **machine learning-based nonlinear modeling frameworks**

**Source:** [诚奇量化总结](https://zhuanlan.zhihu.com/p/1980331980483757087)

---

### 1.3 Aggregate 2025 Performance

| Firm | 2025 Return | Strategy Type |
|------|-------------|---------------|
| 灵均投资 (Lingjun) | 73.51% | Quant stock selection |
| 幻方量化 (Huanfang) | 56.55% | DL multi-factor |
| **Average (45 百亿量化)** | **37.61%** | Various |
| All 45 百亿 quant funds | 100% positive | -- |

### 1.4 Common Strategy Taxonomy

1. **Stock Multi-Factor Selection (股票多因子选股):** The bread-and-butter. Alpha158-style factors + ML.
2. **Market Neutral / Statistical Arbitrage (市场中性):** Long-short with index futures hedging.
3. **High-Frequency / T+0:** Intraday round-trip using ETF or stock positions as base.
4. **Managed Futures / CTA:** Commodity and financial futures momentum/mean-reversion.
5. **Multi-Strategy:** Combining all above with dynamic allocation.
6. **Cross-Market:** A-shares + HK + US + commodities + crypto.

---

## 2. Famous A-Share Stock Pickers / Traders

### 2.1 Xu Xiang (徐翔) -- "The Chinese Soros"

**Background:** Started trading in 1993 with 30,000 RMB. Built it to 4+ billion RMB before arrest in 2015 for market manipulation.

**Daily Routine:** 12+ hours/day studying stocks for 20+ years. Morning meetings, never leaves the trading room during market hours, lunch with sell-side analysts, post-close review and research.

**Core Strategies:**

1. **"一字断魂刀" (One-Stroke Soul-Severing Blade):**
   - On good news, place massive buy orders to seal the limit-up at close
   - Next day, when retail investors rush to buy, sell into their demand ~3% below the bid wall
   - Essentially: manufactured momentum trap for retail

2. **Limit-Up Board (涨停板) Principles:**
   - Best board-sealing time: before 10:30 AM
   - Prefer low-priced stocks (more explosive potential)
   - Time-sharing chart: straight vertical surge = best quality board
   - Stock must have "history" -- prior consecutive limit-ups, strong market popularity
   - Most explosive leaders are dominated by hot money, not institutions

3. **Directed Equity Increase (定增) Arbitrage:**
   - Buy BEFORE the equity increase news announcement (insider info)
   - Sell immediately after announcement of high dividends/splits
   - Post-2010, this became his most reliable profit method

4. **Shareholder Activism (股东积极主义):**
   - Accumulate large positions to become significant shareholders
   - Actively influence company decisions to drive stock price
   - Classic example: bought 30M shares of Chongqing Beer at 17 RMB during scandal crash, sold at 34 RMB for billions in profit

5. **Infrastructure Edge:**
   - Trading servers placed <100 meters from exchange (co-location)
   - Massive commission spending for priority order routing
   - Effectively had "first buyer/seller rights" in China

**Source:** [徐翔的操盘手法和经典案例](https://finance.sina.com.cn/money/fund/fundzmt/2024-12-25/doc-inearqpw7744990.shtml), [一字断魂刀](https://www.xiarj.com/21514.html)

---

### 2.2 Zhao Laoge (赵老哥) -- "8 Years, 10,000x"

**Background:** Born 1987 in Zhejiang. Started with 100,000 RMB in 2007, reached 1 billion by 2015 (10,000x in 8 years).

**Core Strategy: Board-Hitting (打板)**

**Specific Rules:**
- **Entry:** Buy at the limit-up board (涨停板打板), specifically targeting the **first board** (首板)
- **Exit Rule 1:** If stock does NOT gap-up to another board the next day, sell 100% immediately
- **Exit Rule 2:** Losing positions NEVER held overnight -- 100% cut
- **Position Sizing:** Nearly always full position (满仓), but split across 2-3 stocks
- **Holding Period:** Ultra-short, typically 1-2 days (隔日超短)

**Stock Selection:**
- Focus on **main-theme leaders** (主线题材龙头)
- Prefer stocks with strong market popularity and "stock characteristics" (股性好)
- History of consecutive limit-ups is a plus

**Known Seats (龙虎榜 Seats):**
- 银河证券绍兴营业部 (primary)
- 浙商证券绍兴分公司
- 银河证券北京阜成路
- 银泰证券上海嘉善路
- 中信证券淮海中路

**2025 Activity:** Active in AI (computing power, applications) and robotics themes.

**Source:** [赵老哥：从10万到10亿](https://zhuanlan.zhihu.com/p/1944325814708573871), [赵老哥实战交割单分析](https://zhuanlan.zhihu.com/p/26427371194)

---

### 2.3 Chaogu Yangjia (炒股养家) -- "40万 to 10亿+"

**Background:** Former Shanghai SOE employee. Created the famous "养家心法" (Yangjia Heart Method) in 2012.

**Core Philosophy: Emotion Cycle Trading (情绪周期)**

**Key Principles:**
- **Market sentiment is everything:** The core skill is sensing "profit-making effects" (赚钱效应) vs "loss-making effects" (亏钱效应) in the market
- **When profit effect is strong:** Trade hot-topic leading stocks, use relay/follow-through mode
- **When panic effect dominates:** Focus on oversold bounces
- **Dynamic risk/reward assessment:** Continuously weigh risk vs reward based on market emotion phase

**Stock Selection Method:**
- Strong momentum, leadership position, explosive potential
- Uses channel advantages for priority limit-up orders
- Master of "strong stock low-absorption" (强势股低吸) technique
- Deep understanding of market-index-to-individual-stock dynamics

**Trading Framework:**
- Market emotion cycle identification -> sector/theme selection -> individual stock entry
- Position sizing based on emotion cycle phase (high conviction = full position)

**Source:** [养家心法](https://zhuanlan.zhihu.com/p/597850242), [游资之王炒股养家内功心法](https://zhuanlan.zhihu.com/p/362660744)

---

### 2.4 Other Notable Hot Money Players (游资)

| Name | Specialty | Known Seats |
|------|-----------|-------------|
| 方新侠 | High-position relay, large-cap hot money | Various |
| 章盟主 | Large-scale institutional-style hot money | 国泰君安上海江苏路 |
| 陈小群 | Mid-cap board hitting | Various |
| 欢乐海岸 | High-position relay specialist | 东方财富拉萨 |
| 作手新一 | Aggressive small-cap plays | Various |
| 小鳄鱼 | Cross-board relay | Various |

**Specialization Patterns:**
- **3-board specialists (三板接力):** 金田路, 古北路, 赵老哥
- **High-position relay (高位接力):** 欢乐海岸, 章盟主, 古北路, 赵老哥, 溧阳路

**Source:** [游资席位最全名单](https://zhuanlan.zhihu.com/p/605428081), [2025年最新龙虎榜常客](https://m.tgb.cn/a/2hJojLL7wWh)

---

## 3. Top Performing Mutual Funds (2024-2025)

### 3.1 2025 Annual Performance Champions

| Rank | Fund Name | Manager | Return | Theme |
|------|-----------|---------|--------|-------|
| 1 | 永赢科技智选 | 任桀 | **233.29%** | AI/光模块 |
| 2 | 中航机遇领航 | 韩浩 | 168.92% | Technology |
| 3 | 红土创新新兴产业 | 廖星昊 | 148.64% | Tech/Innovation |
| 4 | 恒越优势精选 | -- | >140% | -- |
| 5 | 信澳业绩驱动 | -- | >140% | -- |
| 6 | 中欧数字经济 | -- | >140% | -- |
| 7 | 交银优择回报 | -- | >140% | -- |

**The 233% champion broke a record held since 2007** (Wang Yawei's 226.24% with Huaxia Dapan Jingxuan).

**75 funds doubled in 2025**, with E Fund (易方达) alone accounting for 10 of them.

### 3.2 Champion Fund Holdings: 永赢科技智选

**Concentration:** Top 10 holdings = **73.25%** of NAV (extremely concentrated)

**Sector Breakdown:** 5 Communication + 5 Electronics stocks

**Top Holdings ("易中天" trio):**
1. **新易盛 (Eoptolink)** -- Q4 2025 gain: +187.96%
2. **中际旭创 (Innolight)** -- Q4 2025 gain: +176.76%
3. **天孚通信 (T&S Communications)** -- Q4 2025 gain: +110.76%
4. **深南电路 (Shennan Circuits)** -- Q4 2025 gain: +100.95%

**Theme:** All CPO (Co-Packaged Optics) / optical module / AI computing infrastructure plays.

**Source:** [2025公募业绩放榜](https://www.cls.cn/detail/2246025), [首批基金2025年四季报](https://finance.sina.com.cn/roll/2026-01-14/doc-inhheyvf8792942.shtml)

### 3.3 Star Fund Managers -- Current Status

#### Zhang Kun (张坤) -- 易方达蓝筹精选
- **AUM:** 349.43B RMB (largest active fund)
- **2025 Return:** 12.85% (underperformed)
- **Holdings:** Heavy on baijiu (白酒) -- 五粮液, 泸州老窖, 贵州茅台, 山西汾酒
- **Recent moves:** Added JD Health (京东健康), SF Holding (顺丰控股); removed Meituan
- **Style:** Deep value, concentrated, consumer-heavy

#### Ge Lan (葛兰) -- 中欧医疗健康
- **Focus:** Healthcare/pharma
- **Hidden holdings:** 华东医药, 海思科, 泽璟制药
- **Big addition:** 艾力斯 position increased by **2,627%** (mid-2025 vs end-2024)
- **Thesis:** Innovation drugs, OTC, consumer healthcare

#### Fu Pengbo (傅鹏博) -- 睿远成长价值
- **2025 Return:** 48.50% (strong outperformance)
- **Big adds:** Alibaba (+161%), BYD (+184%)
- **Style:** Growth at reasonable price, willing to shift sectors

### 3.4 Accessing Fund Holdings Data

**Disclosure schedule:** Chinese mutual funds report quarterly. Full holdings in semi-annual and annual reports; top-10 only in Q1/Q3 reports.

**APIs for fund holdings:**
```python
import akshare as ak

# Fund portfolio holdings (per quarter)
df = ak.fund_portfolio_hold_em(symbol="000001", date="2025")

# Fund portfolio changes
df = ak.fund_portfolio_change_em(symbol="000001")

# Fund industry allocation
df = ak.fund_portfolio_industry_allocation_em(symbol="000001")
```

**Source:** [AKShare 公募基金数据](https://akshare-hh.readthedocs.io/en/latest/data/fund/fund_public.html)

---

## 4. Strategies That Could Be Integrated Into This Project

### 4.1 Current Architecture Summary

The project uses:
- **Qlib Alpha158** features (158 price-volume-fundamental factors)
- **LightGBM / XGBoost** for short-term (5-day) return prediction
- **Multi-timeframe scoring** (short + mid + sentiment + macro)
- **AKShare** as primary data source with Tencent fallback
- **Signal scorer** with weighted fusion of model scores

### 4.2 Strategy Integration Evaluation

#### Strategy A: Institutional Holdings Change Factor
| Dimension | Assessment |
|-----------|------------|
| **Quantifiable?** | Yes -- quarterly fund holdings are publicly disclosed |
| **Data Source** | `ak.fund_portfolio_hold_em()` (AKShare), TuShare `fund_portfolio` |
| **Implementation** | Compute quarter-over-quarter holding changes per stock; aggregate across top N funds |
| **Integration Point** | New factor in `factors/` directory, feed into scorer as `institutional_score` |
| **Difficulty** | Medium -- quarterly lag is a challenge |
| **Alpha Potential** | Moderate. Academic evidence: top-10 institutional holdings portfolio yields ~35.68% annualized, ~25.26% excess returns. Net-increase group: ~5% annual alpha, 73% quarterly win rate |
| **Code Sketch** | See Section 5 below |

#### Strategy B: Northbound Capital (北向资金) Flow Factor
| Dimension | Assessment |
|-----------|------------|
| **Quantifiable?** | Yes -- daily flow data available |
| **Data Source** | `ak.stock_hsgt_north_net_flow_in_em()`, `ak.stock_hsgt_hold_stock_em()` |
| **Implementation** | Daily net inflow by stock; relative over/underweight ratio; 2-week observation -> 6-week holding |
| **Integration Point** | Add to `factors/` as northbound factor, update daily in pipeline |
| **Difficulty** | Low-Medium |
| **Alpha Potential** | High. Research shows: multi-long portfolio annualized 28.34%, Sharpe 2.46, max drawdown 6.13%. Best observation period: 2 weeks, holding period: 6 weeks |

#### Strategy C: Main Capital Flow Factor (主力资金流)
| Dimension | Assessment |
|-----------|------------|
| **Quantifiable?** | Yes -- tick-level order flow data |
| **Data Source** | `ak.stock_individual_fund_flow()` or EastMoney fund flow APIs |
| **Implementation** | Compute main-force sell order count (mfd_sellord), net inflow ratio |
| **Integration Point** | Daily factor, add to Alpha158 feature set in `factors/quant.py` |
| **Difficulty** | Low |
| **Alpha Potential** | Very High. mfd_sellord factor: 42.12% annualized return, Sharpe 4.43 (10-day holding). Net inflow factor: 12.25% annualized, Sharpe 2.02 |

#### Strategy D: Limit-Up Board Momentum (涨停板打板 -- Zhao Laoge Style)
| Dimension | Assessment |
|-----------|------------|
| **Quantifiable?** | Partially -- board-hitting itself requires real-time execution, but the screening criteria can be quantified |
| **Data Source** | `ak.stock_zt_pool_em()` (涨停池), `ak.stock_board_concept_name_em()` |
| **Implementation** | Identify first-board (首板) stocks; score by time-of-board-seal, volume ratio, theme alignment |
| **Integration Point** | New module `factors/limit_up.py` (already exists!), enhance with theme scoring |
| **Difficulty** | Medium-High (requires near-real-time data) |
| **Alpha Potential** | High but risky. Works in strong markets; negative in weak markets |

#### Strategy E: Sentiment Cycle Detection (炒股养家 Style)
| Dimension | Assessment |
|-----------|------------|
| **Quantifiable?** | Yes -- market-wide profit/loss effect can be computed |
| **Data Source** | Market breadth data (涨跌家数), limit-up/down counts, sector rotation metrics |
| **Implementation** | Compute "赚钱效应" index from market breadth, consecutive limit-up counts, sector heat |
| **Integration Point** | Enhance `signals/market_judge.py` or `factors/sentiment.py` |
| **Difficulty** | Medium |
| **Alpha Potential** | Moderate. Most useful as a regime filter (trade aggressively in profit-effect markets, defensively in loss-effect markets) |

#### Strategy F: Top Fund Manager Copycat (重仓股跟踪)
| Dimension | Assessment |
|-----------|------------|
| **Quantifiable?** | Yes |
| **Data Source** | AKShare fund portfolio APIs; quarterly reports |
| **Implementation** | Track top-10 holdings of champion funds; compute consensus holdings across top N managers |
| **Integration Point** | New factor `factors/fund_consensus.py` -> feed into scorer |
| **Difficulty** | Low |
| **Alpha Potential** | Moderate with large lag. Better as a universe filter than a timing signal |

#### Strategy G: Deep Learning Factor Generation (幻方 Style)
| Dimension | Assessment |
|-----------|------------|
| **Quantifiable?** | Yes -- the project already has `models/deep_models.py` |
| **Data Source** | Existing Alpha158 features + order book data |
| **Implementation** | Train DNN/Transformer on Alpha158 features to generate nonlinear factor combinations |
| **Integration Point** | Already partially implemented; enhance with attention mechanisms |
| **Difficulty** | High |
| **Alpha Potential** | Very High. Huanfang's DL transition was their key breakthrough |

### 4.3 Priority Ranking for Implementation

1. **Northbound Capital Factor (B)** -- Low difficulty, high alpha, daily frequency
2. **Main Capital Flow Factor (C)** -- Low difficulty, very high alpha, daily frequency
3. **Institutional Holdings Factor (A)** -- Medium difficulty, moderate alpha
4. **Sentiment Cycle Detection (E)** -- Medium difficulty, useful regime filter
5. **Limit-Up Board Enhancement (D)** -- Medium-high difficulty, conditional alpha
6. **Fund Manager Copycat (F)** -- Low difficulty, moderate alpha with lag
7. **Deep Learning Factors (G)** -- High difficulty, high alpha but needs GPU

---

## 5. Fund Holdings as Input Features -- Detailed Research

### 5.1 Academic Evidence for "Smart Money" Following

**Key findings from academic literature:**

1. **Institutional ownership predicts returns:** Top-10 institutional holdings portfolios yield ~35.68% annualized, with ~25.26% excess returns over benchmarks.

2. **Net holding changes are predictive:** Stocks with net institutional buying outperform by ~5% annually (73% quarterly win rate). Stocks with net selling underperform by ~4%.

3. **Enhanced anomaly strategies:** Combining institutional entry/exit signals with factor anomalies adds 19-54 bps/month of Fama-French 5-factor alpha.

4. **Post-publication decay:** Factor premiums decay ~56% after academic publication as institutions crowd in. This means institutional ownership itself becomes a contrarian signal at extremes.

5. **Northbound capital is "smarter":** Research from multiple Chinese brokerages shows northbound (QFII/Stock Connect) capital has superior stock-picking ability vs domestic institutions.

**Sources:**
- [Using Institutional Investor's Trading Data in Factors](https://alphaarchitect.com/using-institutional-investors-trading-data-in-factors/)
- [Institutional ownership data: Quantitative research results](https://cdn.ihsmarkit.com/www/pdf/0621/Institutional_ownership_data_Quantitative_research_results.pdf)
- [基本面量化视角下的北向资金持仓信息](https://bigquant.com/square/paper/9314ee13-e602-4753-a5fa-68465925db03)

### 5.2 Data APIs for Holdings Data

#### AKShare (Free, Open Source)

```python
import akshare as ak

# === Fund Holdings ===
# Get fund portfolio holdings for a specific fund
df = ak.fund_portfolio_hold_em(symbol="110011", date="2025")
# Returns: stock_code, stock_name, holding_ratio, holding_amount, holding_value

# Get top fund holdings changes
df = ak.fund_portfolio_change_em(symbol="110011")

# === Northbound Capital ===
# Daily northbound net flow (aggregate)
df = ak.stock_hsgt_north_net_flow_in_em()
# Returns: date, northbound_net_inflow, shanghai_net, shenzhen_net

# Individual stock northbound holdings
df = ak.stock_hsgt_hold_stock_em(market="北向", indicator="今日排行")
# Returns: stock_code, stock_name, holding_shares, holding_value, change

# Historical northbound holdings for a stock
df = ak.stock_hsgt_individual_em(symbol="600519")  # Moutai example

# === Main Capital Flow ===
# Individual stock fund flow
df = ak.stock_individual_fund_flow(stock="600519", market="sh")
# Returns: date, main_net_inflow, retail_net_inflow, super_large_order, large_order

# Sector fund flow
df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
```

#### TuShare Pro (Freemium, Requires Token)

```python
import tushare as ts
pro = ts.pro_api('YOUR_TOKEN')

# Fund portfolio (quarterly disclosure)
df = pro.fund_portfolio(ts_code='000001.OF', ann_date='20250630')

# Northbound capital holdings
df = pro.hk_hold(trade_date='20250501', exchange='SH')

# Institutional holdings (from quarterly reports)
df = pro.stk_holdernumber(ts_code='600519.SH')

# Money flow
df = pro.moneyflow(ts_code='600519.SH', start_date='20250101')
```

### 5.3 Factor Construction Methods

#### Factor 1: Northbound Net Position Change (Daily)

```python
def northbound_position_change(stock_code, lookback=10):
    """
    Compute relative change in northbound holdings over lookback period.

    Signal: Increasing northbound ownership -> bullish
    Best params from research: 2-week observation, 6-week hold
    Expected: Sharpe 2.46, annual return 28.34%
    """
    df = ak.stock_hsgt_individual_em(symbol=stock_code)
    df['pct_change'] = df['holding_shares'].pct_change(lookback)
    df['z_score'] = (df['pct_change'] - df['pct_change'].rolling(60).mean()) / \
                     df['pct_change'].rolling(60).std()
    return df['z_score'].iloc[-1]
```

#### Factor 2: Institutional Consensus Score (Quarterly)

```python
def institutional_consensus_score(stock_code, top_n_funds=50):
    """
    Compute how many top funds hold this stock and their aggregate change.

    Signal: More top funds adding -> bullish
    Expected: ~5% annual alpha, 73% win rate
    """
    # Get top N funds by AUM
    # For each fund, check if stock_code is in top-10 holdings
    # Compute: count_holding / top_n_funds (breadth)
    # Compute: sum(holding_change) (conviction)
    # Return: breadth * 0.5 + conviction_z_score * 0.5
    pass
```

#### Factor 3: Main Capital Flow Momentum (Daily)

```python
def main_capital_flow_factor(stock_code, lookback=10):
    """
    Main-force net inflow rate and sell order count.

    Signal: Low main-force sell orders -> bullish
    Expected: Sharpe 4.43 (mfd_sellord), 42.12% annualized
    """
    df = ak.stock_individual_fund_flow(stock=stock_code, market="sh")
    # Compute rolling net inflow ratio
    df['net_ratio'] = df['main_net_inflow'] / df['main_net_inflow'].abs().rolling(20).mean()
    # Compute sell pressure indicator
    df['sell_pressure_z'] = -1 * (df['main_sell'] - df['main_sell'].rolling(60).mean()) / \
                             df['main_sell'].rolling(60).std()
    return df[['net_ratio', 'sell_pressure_z']].iloc[-1]
```

### 5.4 Integration Architecture for This Project

```
Current Pipeline:
  Alpha158 factors -> LightGBM -> short_term_score
  News/Sentiment  -> LLM       -> sentiment_score
  Macro/Geo       -> rules     -> macro_score
  All             -> SignalScorer -> final_score

Proposed Enhancement:
  Alpha158 factors ─────────────────┐
  Northbound flow factor ───────────┤
  Main capital flow factor ─────────┼──> LightGBM -> short_term_score (enhanced)
  Institutional holdings factor ────┘

  News/Sentiment  -> LLM       -> sentiment_score
  Macro/Geo       -> rules     -> macro_score
  Emotion cycle   -> rules     -> market_regime (new weight modifier)

  All + regime    -> SignalScorer -> final_score
```

**Implementation Steps:**

1. Create `data/collectors/fund_flow.py` -- collect northbound + main capital flow daily
2. Create `factors/institutional.py` -- compute institutional/northbound/flow factors
3. Modify `factors/quant.py` -- add new factors to Alpha158 feature pipeline
4. Modify `signals/scorer.py` -- add `institutional_score` weight (suggest 0.1, take from macro)
5. Add quarterly job in `scheduler/jobs.py` for fund holdings update

### 5.5 Key Research Reports (券商研报)

- **华泰证券:** "单因子测试之资金流向因子" -- mfd_sellord factor Sharpe 4.43
- **开源证券:** "从涨跌停外溢行为到股票关联网络" -- limit-up spillover effects
- **BigQuant/中金:** "基本面量化视角下的北向资金持仓信息" -- northbound Sharpe 2.46
- **清华五道口:** "中国A股市场量化因子白皮书" -- 56 factors across 6 categories

**Source:** [BigQuant量化因子](https://bigquant.com/wiki/topic/03b4ba0420), [QuantsPlaybook研报复现](https://github.com/hugo2046/QuantsPlaybook)

---

## 6. Regulatory Context (2026)

The CSRC has tightened oversight of quant trading in 2026:
- **High-frequency quant trading** faces deeper supervision
- **Fairness principles** emphasized -- preventing quant firms from having unfair speed advantages
- Legitimate hedging activities still supported
- Several firms faced penalties for excessive order cancellation rates

This means strategies relying on sub-second execution (like Huanfang's 800ns system) face regulatory headwinds. **Mid-frequency strategies (daily/weekly rebalancing) like this project's approach are well-positioned** relative to HFT-dependent competitors.

---

## Summary: Key Takeaways for This Project

1. **Northbound capital flow** is the single highest-ROI factor to add (daily data, high Sharpe, low implementation cost)
2. **Main capital flow** (主力资金流) is the second priority -- mfd_sellord factor has Sharpe 4.43 in backtests
3. **Institutional holdings** are useful but quarterly lag limits alpha; best as a universe filter
4. **Emotion cycle / market regime detection** would improve the SignalScorer's weight adaptation
5. **Deep learning factor generation** (Huanfang-style) is the highest-alpha but highest-cost upgrade
6. **The project's daily Alpha158 + XGB pipeline is already similar to what 百亿 quant funds used 3-5 years ago** -- the main gap is alternative data factors and DL-based nonlinear combinations
7. **Board-hitting (打板) strategies** are difficult to automate but limit-up screening can enhance the existing pipeline

---

*Research compiled from Chinese financial media, academic papers, and broker research reports. All URLs are from publicly accessible sources.*
