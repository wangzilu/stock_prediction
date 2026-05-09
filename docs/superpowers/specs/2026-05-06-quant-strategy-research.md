# Quantitative Trading Strategy Research

**Date:** 2026-05-06
**Purpose:** Comprehensive reference for strategies, libraries, papers, and monster-stock detection methods applicable to this A-share prediction system.

---

## 1. Top Quantitative Trading Strategies

### 1.1 Multi-Factor Models

| Model | Factors | A-Share Relevance |
|-------|---------|-------------------|
| **Fama-French 3-Factor** | Market, Size (SMB), Value (HML) | Strong in A-shares; HML (value) is especially important. The 3-factor model outperforms the 5-factor model in China, opposite to U.S. results. |
| **Fama-French 5-Factor** | + Profitability (RMW), Investment (CMA) | RMW and CMA become redundant when HML is included in A-shares. Lacks robustness for China. |
| **China-Modified 6-Factor** | Market, Size, Value, Profitability, Investment, Momentum | A modified 6-factor and a China-tailored 4-factor model outperform U.S.-developed models on A-shares. |
| **Barra CNE5/CNE6** | ~10 style factors + 30 industry factors | Industry-standard risk model for Chinese equities. Used by institutional quant funds. Factors: Size, Beta, Momentum, Residual Volatility, Non-linear Size, Book-to-Price, Liquidity, Earnings Yield, Growth, Leverage. |
| **Alpha158 (Qlib)** | 158 technical/fundamental factors | Already in use in this project. Covers price, volume, volatility, and derived features. The de facto baseline for Qlib-based research. |

**Key insight for this project:** The current system uses Alpha158 with LightGBM. A natural extension is adding explicit Barra-style risk factor exposure control to avoid unintended factor bets (e.g., accidental all-small-cap portfolios).

### 1.2 Momentum / Reversal Strategies

| Strategy | Description | A-Share Notes |
|----------|-------------|---------------|
| **Cross-sectional momentum** | Long top decile past-12-month winners, short bottom decile | Weak or negative in A-shares historically. Short-term (1-4 week) momentum works better than 12-month. |
| **Short-term reversal** | Buy past-week losers, sell past-week winners | Strong in A-shares due to retail-dominated overreaction. One of the most robust alpha sources. |
| **Industry momentum** | Rotate into sectors with strongest recent performance | Works well in A-shares due to strong sector rotation (板块轮动). The "sector rotation catalyst" is critical for monster stocks. |
| **52-week high momentum** | Buy stocks near 52-week highs | Moderate effectiveness. Combine with volume confirmation for better results. |
| **Earnings momentum (SUE)** | Buy on positive standardized unexpected earnings | Works in A-shares but data is quarterly, limiting frequency. |
| **Limit-up momentum** | Buy stocks with recent limit-up (涨停) events | Very A-share specific. First limit-up after prolonged consolidation is a strong signal. Forms the basis of 打板 (board-hitting) strategies. |

**Key insight:** Short-term reversal + industry momentum is a powerful combination for A-shares. The current system's `change_pct`-based scoring is a crude proxy; explicit factor construction would improve signal quality.

### 1.3 Statistical Arbitrage

| Strategy | Description | Relevance |
|----------|-------------|-----------|
| **Pairs trading** | Cointegrated stock pairs mean-reversion | Limited in A-shares due to short-selling restrictions (融券 is expensive and restricted). Better for ETF pairs. |
| **ETF arbitrage** | Exploit NAV vs. market price deviations | Feasible for LOF/ETF funds. Not relevant to this project's stock-picking focus. |
| **Cross-market arbitrage** | A/H share premium, AH-linked ETFs | Viable but requires Hong Kong market access. |
| **Factor-neutral long-short** | Long high-alpha, short low-alpha within factor-neutral constraints | Difficult in A-shares due to short-selling constraints. Can approximate with index futures hedging. |
| **Intraday mean-reversion** | High-frequency reversion within the day | Requires Level 2 tick data and low latency. Not suitable for daily-frequency system. |

**Key insight:** Pure stat-arb is hard in A-shares due to short-selling constraints. The project should focus on long-only alpha strategies with index futures hedging for risk management.

### 1.4 Machine Learning Approaches

#### 1.4.1 Gradient Boosting (Current: LightGBM)

- **Status:** Already implemented in `scripts/train_lgb.py` with Alpha158 features.
- **Enhancement opportunities:**
  - Add XGBoost and CatBoost as ensemble members for model diversity
  - Implement Qlib's built-in rolling retraining (walk-forward optimization)
  - Add custom features beyond Alpha158: limit-up history, sector momentum, flow data
  - Use Qlib's `TopkDropoutStrategy` for portfolio-level signal generation

#### 1.4.2 Transformer-Based Models

| Model | Description | Reference |
|-------|-------------|-----------|
| **Temporal Fusion Transformer (TFT)** | Attention-based model with variable selection, temporal processing, and interpretable multi-horizon forecasting | Google Research, 2021 |
| **TFT-GNN Hybrid** | TFT + Graph Attention Networks for joint temporal-relational learning across stocks | MDPI 2025 |
| **TGNS** | Transformer-Graph Neural Network for Stock trend forecasting; captures both local/global dependencies | ScienceDirect 2025 |
| **iTransformer** | Inverted transformer treating each feature as a token; better for multivariate time series | ICLR 2024 |
| **PatchTST** | Patch-based transformer for time series; patches reduce computation, improve local pattern capture | ICLR 2023 |
| **Qlib Transformer** | Qlib's built-in `TransformerModel` for stock ranking with Alpha158 input | Microsoft Qlib |

**Key insight:** The current RL agent uses a Transformer encoder but processes only a single time step (obs is unsqueezed to seq_len=1). To truly leverage Transformers, the state should include a lookback window of multiple days' features as a proper sequence.

#### 1.4.3 Graph Neural Networks (GNN)

| Approach | Description |
|----------|-------------|
| **Stock relation graph** | Build graph from industry chains, supply chain, shareholder overlap. GCN/GAT propagates information across related stocks. |
| **CNN-LSTM-GNN (CLGNN)** | Hybrid for A-share prediction: CNN extracts local patterns, LSTM captures temporal, GNN handles relational data. Tested on A-shares 2024. |
| **Temporal dynamic graphs** | Stock relations evolve over time. Use dynamic graph neural networks (DGNN) with temporal attention. |
| **Visibility graph** | Convert price time series into a graph structure; apply GCN/GAT to detect long-memory patterns. |
| **Heterogeneous graph** | Multiple node types (stocks, industries, concepts, news) with different edge types. Multi-agent RL on top for portfolio optimization. |

**Key insight:** GNN is the most promising ML extension for this project because A-share stocks move in sector/concept clusters. Building a stock graph from: (1) industry classification, (2) concept tags (e.g., AI概念, 芯片), (3) fund co-holding, and (4) price correlation would capture the "sector rotation" dynamics critical for monster stock detection.

### 1.5 Reinforcement Learning for Portfolio Management

| Framework | Algorithm | Description |
|-----------|-----------|-------------|
| **Current system** | SAC (Discrete) + Transformer | Per-stock buy/hold/sell timing. Already implemented in `models/rl_agent.py`. |
| **FinRL** | A2C, DDPG, PPO, TD3, SAC | Multi-asset portfolio allocation. Trains 5 agents, uses ensemble. More suitable for portfolio-level decisions. |
| **FinRL-DeepSeek** | RL + LLM | Integrates LLM-generated trading signals from financial news into RL state. Relevant to this project's sentiment pipeline. |
| **TradeMaster** | Multiple RL algos | Covers full pipeline: data, environment, agent, evaluation. Good for benchmarking. |
| **Hierarchical RL** | Macro policy + Micro policy | Top-level policy selects sector allocation, bottom-level policy selects stocks within sector. Natural fit for sector rotation. |
| **Multi-Agent RL** | One agent per stock/sector | Agents communicate through shared market state. Can model competitive dynamics. |

**Key insight:** The current RL agent is per-stock, which ignores portfolio-level considerations (diversification, position sizing, capital allocation). Consider upgrading to a hierarchical approach: (1) RL agent for sector allocation, (2) LightGBM for stock ranking within sector, (3) current RL agent for entry/exit timing.

---

## 2. Best Open-Source Quantitative Libraries

### 2.1 AI/ML Platforms

| Library | GitHub | Stars | Description | Relevance to This Project |
|---------|--------|-------|-------------|--------------------------|
| **Qlib** (Microsoft) | [microsoft/qlib](https://github.com/microsoft/qlib) | ~39k | AI-oriented quant platform. Alpha158, LightGBM, Transformer, rolling retrain, portfolio optimization. | **Already in use.** Core of the prediction pipeline. Expand usage of built-in models and portfolio strategies. |
| **FinRL** | [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL) | ~14k | First open-source DRL framework for finance. Supports A2C/DDPG/PPO/TD3/SAC via Stable-Baselines3. | **High priority.** Better multi-asset RL than current per-stock implementation. FinRL-X is the production-ready successor. |
| **FinRL-X (FinRL-Trading)** | [AI4Finance-Foundation/FinRL-Trading](https://github.com/AI4Finance-Foundation/FinRL-Trading) | ~1k | Next-gen AI-native modular infrastructure. Designed for LLM/agentic AI era. | Worth evaluating as RL upgrade path. |
| **TradeMaster** | [TradeMaster-NTU/TradeMaster](https://github.com/TradeMaster-NTU/TradeMaster) | ~1k | Full RL trading pipeline: data, environment, agent, evaluation. | Good for benchmarking RL strategies. |
| **Qbot (Abu)** | GitHub: UFund-Me/Qbot | ~17k | AI-powered quant robot, fully local deployment, supports RL and DL. Chinese market focused. | Alternative to FinRL for Chinese market RL. |

### 2.2 Backtesting Frameworks

| Library | GitHub | Description | When to Use |
|---------|--------|-------------|-------------|
| **VectorBT** | [polakowo/vectorbt](https://github.com/polakowo/vectorbt) | Vectorized backtesting, extremely fast (millions of trades/sec). NumPy/Numba-powered. | Parameter optimization, bulk strategy scanning. Best speed. |
| **Backtrader** | [mementum/backtrader](https://github.com/mementum/backtrader) | Event-driven, great API, huge example archive. Classic learning tool. | Prototyping, learning. Maintenance risk in 2026. |
| **Zipline-reloaded** | [stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) | Maintained fork of Quantopian's Zipline. Pipeline API for factor research. | Equity factor model research. U.S.-market focus. |
| **NautilusTrader** | [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader) | Production-grade Rust-native engine. Deterministic event-driven architecture. | Production live trading. Fastest parameter optimization. |
| **QuantConnect/Lean** | [QuantConnect/Lean](https://github.com/QuantConnect/Lean) | C#-based, multi-asset, cloud + local. Large community. | Multi-asset strategies, cloud deployment. |

### 2.3 China A-Share Specific

| Library | GitHub | Description | Relevance |
|---------|--------|-------------|-----------|
| **RQAlpha** | [ricequant/rqalpha](https://github.com/ricequant/rqalpha) | A-share backtesting with plugin ("Mod") architecture. Modular and extensible. | Best China-specific backtester for research. |
| **VNPY** | [vnpy/vnpy](https://github.com/vnpy/vnpy) | Full-stack quant trading platform for Chinese markets. Backtest to live. CTA-focused. | **Best choice** if moving to live trading for A-shares/futures. |
| **AKShare** | [akfamily/akshare](https://github.com/akfamily/akshare) | A-share + global market data. Free, no auth required for most endpoints. | **Already in use.** Primary data source. |
| **TuShare** | [waditu/tushare](https://github.com/waditu/tushare) | Pro API for A-share data. Richer institutional/flow data. Requires paid token for full access. | Complement AKShare for limit-up lists, money flow, institutional data. |
| **BaoStock** | [baostock](http://baostock.com/) | Free A-share data. Simpler API than TuShare. | **Already in use** (mentioned in commit history). Backup data source. |
| **PyAlgoTrade** | [gbeced/pyalgotrade](https://github.com/gbeced/pyalgotrade) | Event-driven backtesting. Lightweight. | Simpler alternative to Backtrader. Less maintained. |
| **khQuant** | [khscience/OSkhQuant](https://github.com/khscience/OSkhQuant) | Open-source A-share visual backtesting system. | Niche; useful for visualization only. |

### 2.4 Data Libraries

| Library | Description | Key Data Points |
|---------|-------------|-----------------|
| **AKShare** (in use) | Free A-share + global data | Daily OHLCV, news, macro, sector data |
| **TuShare Pro** | Paid A-share data | **Limit-up/down lists** (`limit_list`), money flow, margin data, institutional holdings, concept stocks |
| **Qlib Data** (in use) | Pre-processed Alpha158 features | Normalized technical factors, aligned and cleaned |
| **CCXT** (in use) | Crypto exchange data | BTC/ETH prices for cross-market correlation |
| **Wind API** | Institutional-grade Chinese financial data | Most complete, but expensive commercial license |
| **JQData (JoinQuant)** | Free tier available, A-share data | Minute-level data, factor library |

---

## 3. Must-Read Books and Papers

### 3.1 Essential Books

| Book | Author | Year | Key Takeaways |
|------|--------|------|---------------|
| **Advances in Financial Machine Learning** | Marcos Lopez de Prado | 2018 | Triple-barrier labeling, meta-labeling, fractional differentiation, purged k-fold CV, feature importance with MDI/MDA/SFI. **The single most important book for ML-based quant.** |
| **Machine Learning for Asset Managers** | Marcos Lopez de Prado | 2020 | Hierarchical Risk Parity (HRP), distance metrics for portfolios, denoising covariance matrices. Practical portfolio construction. |
| **Causal Factor Investing** | Marcos Lopez de Prado | 2023 | Moving from correlative to causal factor investing. Avoids "factor zoo" overfitting. |
| **Quantitative Trading** | Ernest Chan | 2008 | Getting started with systematic trading. Backtesting pitfalls, capacity constraints, Kelly criterion. |
| **Algorithmic Trading** | Ernest Chan | 2013 | Mean reversion, momentum, pairs trading implementation. Practical code examples. |
| **Machine Trading** | Ernest Chan | 2017 | Advanced strategies: portfolio optimization, risk management, HFT concepts. |
| **Quantitative Equity Portfolio Management** | Chincarini & Kim | 2006 | Factor model construction, Barra-style risk models. Academic rigor. |
| **Active Portfolio Management** | Grinold & Kahn | 1999 | The "fundamental law of active management." Information ratio, breadth, skill. Foundational. |
| **Finding Alphas** | WorldQuant (Kakushadze) | 2015 | 101 alpha formulaic expressions. Practical alpha engineering. Many directly implementable. |

### 3.2 Key Papers — ML/DL for Trading

| Paper | Year | Key Contribution |
|-------|------|------------------|
| **"FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading"** — Liu et al. | 2020 | NeurIPS 2020. Established the FinRL framework. Benchmark DRL agents on stock trading. |
| **"Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy"** — Yang et al. | 2020 | Ensemble of A2C/DDPG/PPO with automated switching based on Sharpe ratio. |
| **"Temporal Fusion Transformers for Interpretable Multi-Horizon Forecasting"** — Lim et al. | 2021 | TFT architecture with variable selection, static covariates, interpretable attention. Google Research. |
| **"MASTER: Market-Guided Stock Transformer for Stock Price Forecasting"** — Li et al. | 2024 | AAAI 2024. Market-aware Transformer that captures both intra-stock and inter-stock patterns. |
| **"TGNS: Transformer-Graph Neural Network for Stock Trend Forecasting"** | 2025 | Combines GNN + Transformer to capture local/global dependencies among stocks. |
| **"CNN-LSTM-GNN (CLGNN) for A-Share Stock Prediction"** | 2025 | Hybrid model tested on A-shares. CNN for local patterns, LSTM for temporal, GNN for relational. |
| **"DoubleAdapt: A Meta-Learning Approach for Incremental Learning to Rank"** — Qlib team | 2023 | Handles concept drift in stock prediction. Meta-learning for model adaptation. Built into Qlib. |
| **"Qlib: An AI-oriented Quantitative Investment Platform"** — Yang et al. | 2020 | The Qlib platform paper. Describes Alpha158, data pipeline, model zoo, portfolio optimization. |
| **"Machine Learning in the Chinese Stock Market"** — Leippold et al. | 2022 | Journal of Financial Economics. Comprehensive study of ML methods on A-shares. Tree models dominate. |
| **"Financial Machine Learning: An Engineering Problem"** — Lopez de Prado | 2025 | Latest thinking on treating ML in finance as an engineering discipline rather than a science experiment. |

### 3.3 Key Papers — Monster Stocks / Extreme Movers

| Paper / Resource | Key Insight |
|-----------------|-------------|
| **"Speculative Trading and Stock Prices"** — Xiong et al. (Princeton) | Models speculative overvaluation in Chinese market. Heterogeneous beliefs + short-sale constraints drive extreme prices. |
| **"AI-Driven Anomaly Detection in Stock Markets"** — Springer 2025 | Isolation Forest + CatBoost for detecting exploitable inefficiencies. Directly applicable to monster stock detection. |
| **"Stock Market Manipulation Detection Using Continuous Methods"** | Detects pump-and-dump patterns that often characterize early-stage monster stocks. |

---

## 4. Monster Stock (妖股) Detection Strategies

### 4.1 Definition and Characteristics

A "monster stock" (妖股) in A-shares typically:
- Rises 5x-10x or more within weeks to months
- Has multiple consecutive limit-ups (连板)
- Defies fundamental valuation logic
- Is driven by speculative retail herding + thematic catalysts
- Often has small float (小盘股), low institutional ownership
- Frequently associated with hot concepts/themes (概念股)

### 4.2 Quantitative Signals for Detection

#### Signal Group 1: Volume and Price Anomalies

| Signal | Calculation | Rationale |
|--------|-------------|-----------|
| **Volume spike** | `volume / avg_volume_20d > 3.0` | Sudden interest surge. First sign of speculative attention. |
| **Volume-price divergence** | Price rises but volume contracts on pullbacks | Healthy monster stocks hold volume on advances and dry up on dips — indicates strong holder conviction. |
| **Turnover rate explosion** | `turnover_rate > 15%` for small caps | Extreme turnover in small caps signals speculative frenzy. |
| **Price acceleration** | `return_5d > return_20d * 0.8` | Most of the 20-day return concentrated in the last 5 days. Accelerating momentum. |
| **ATR breakout** | `true_range / atr_20d > 2.5` | Volatility expansion. Stock "waking up" from dormancy. |

#### Signal Group 2: Limit-Up Chain Analysis (连板分析)

| Signal | Calculation | Rationale |
|--------|-------------|-----------|
| **First board (首板)** | Stock hits limit-up after 20+ days without one | First limit-up after consolidation is the highest expected-value entry for monster stocks. |
| **Consecutive boards (连板)** | Count of consecutive limit-up days | 2-board (二连板) is the key confirmation. 3+ boards signal strong speculative momentum. |
| **Board quality** | Sealed time, open count, final volume | "一字板" (opens at limit-up) > "T字板" (opens at limit, pulls back, re-seals) > "烂板" (multiple opens). Earlier seal = stronger. |
| **Next-day premium (溢价)** | Average return on day after limit-up | Market-wide "board premium" varies by market sentiment. Track rolling 5-day average premium to gauge speculative appetite. |
| **Board height rank** | How many boards vs. current market leader (龙头) | The stock with the most consecutive boards is the "market leader." Others follow its rhythm. |

**Implementation with AKShare/TuShare:**
- AKShare: `ak.stock_zt_pool_em()` — daily limit-up pool
- AKShare: `ak.stock_zt_pool_previous_em()` — previous day limit-up stocks (for next-day premium tracking)
- TuShare: `pro.limit_list()` — historical limit-up/down data with seal time and open count

#### Signal Group 3: Float and Structure

| Signal | Calculation | Rationale |
|--------|-------------|-----------|
| **Free float market cap** | `free_float_shares * price < 5B RMB` | Small float = easier to move. Most monster stocks have <5B free float market cap. |
| **Concentration of top holders** | Top 10 holder % increasing | Smart money accumulating before the run. |
| **Low institutional ownership** | `inst_holding_pct < 10%` | Institutions don't drive monster stocks — retail does. Low institutional ownership means less selling pressure during the run. |
| **Recent equity events** | Stock split, rights issue, name change | "摘帽" (ST hat removal), name changes, and splits often trigger speculative interest. |

#### Signal Group 4: Sector Rotation and Thematic Catalysts

| Signal | Calculation | Rationale |
|--------|-------------|-----------|
| **Concept/sector heat** | Count of limit-ups within same concept in 5 days | When 3+ stocks in the same concept (e.g., AI, 算力, 低空经济) hit limit-up, the sector is "in play." |
| **Sector rotation timing** | Sector just entered "acceleration" phase (sector index breaking 20-day high after pullback) | Monster stocks emerge at the beginning of sector rotation, not the end. |
| **Policy catalyst** | NLP detection of policy announcements related to stock's sector | State Council mentions, NDRC plans, etc. can trigger multi-day sector rallies. |
| **Concept tag count** | Number of hot concepts the stock belongs to | Stocks with multiple overlapping hot concepts (e.g., AI + 算力 + 华为) get more speculative attention. |

#### Signal Group 5: Sentiment and Flow

| Signal | Calculation | Rationale |
|--------|-------------|-----------|
| **Social media mention spike** | `mentions_today / avg_mentions_7d > 5.0` | Measured from EastMoney (东方财富) stock bar, Xueqiu (雪球), WeChat public accounts. |
| **News sentiment spike** | Positive sentiment score jumps from neutral to strongly positive | Sudden narrative shift. The project's existing LLM sentiment pipeline can detect this. |
| **Dragon-Tiger list (龙虎榜)** | Presence on institutional buy list | Top 5 buyer/seller disclosure. "知名游资" (famous hot money) appearing on buy side is a strong signal. |
| **Margin trading increase** | `margin_buy / margin_balance > 0.1` | Leveraged buyers entering. Signals confidence but also fragility. |
| **Northbound flow (北向资金)** | Net buy into stock via Stock Connect | Foreign institutional interest. Less relevant for monster stocks (they're usually retail-driven). |

### 4.3 Composite Monster Stock Score

A proposed scoring formula combining the above signals:

```
monster_score = (
    0.25 * limit_up_chain_score      # 连板 strength (0-1)
  + 0.20 * volume_anomaly_score      # Volume spike severity (0-1)
  + 0.20 * sector_heat_score         # Concept/sector activity (0-1)
  + 0.15 * float_structure_score     # Small float + low inst. (0-1)
  + 0.10 * sentiment_spike_score     # Social/news sentiment (0-1)
  + 0.10 * technical_breakout_score  # Chart pattern + ATR (0-1)
)
```

**Where:**
- `limit_up_chain_score` = `min(consecutive_boards / 5, 1.0) * board_quality`
- `volume_anomaly_score` = `min(volume_ratio / 5, 1.0)`
- `sector_heat_score` = `min(sector_limit_up_count / 10, 1.0)`
- `float_structure_score` = `1.0 if free_float_mcap < 3B else (5B - free_float_mcap) / 2B`
- `sentiment_spike_score` = `min(mention_ratio / 10, 1.0)`
- `technical_breakout_score` = breakout pattern detection (cup-and-handle, ascending triangle, flag)

### 4.4 Technical Breakout Patterns for Monster Stocks

| Pattern | Description | Detection Approach |
|---------|-------------|--------------------|
| **Cup and handle** | U-shaped base + small consolidation before breakout | Fit quadratic to 30-60 day price curve; detect handle as 5-10 day pullback <5% from cup rim |
| **Ascending triangle** | Flat resistance + rising lows | Linear regression on lows trending up; horizontal resistance at recent highs |
| **Bull flag** | Sharp rise (pole) + parallel channel pullback (flag) | Detect 20%+ rise in 5 days, followed by 5-10 day consolidation with <8% retracement |
| **Volume dry-up + breakout** | Volume contracts to 20-day low, then explodes 3x+ | `volume < 0.5 * avg_volume_20d` for 3+ days, then `volume > 3 * avg_volume_20d` |
| **Moving average convergence** | 5/10/20/60 MA converge then fan out upward | Standard deviation of MAs reaching historical low, followed by expansion |

### 4.5 Risk Filters (Avoiding Traps)

Monster stock hunting is high-risk. Essential filters:

| Filter | Rule | Purpose |
|--------|------|---------|
| **Avoid late-stage** | Don't enter after 5+ consecutive boards unless it's the undisputed market leader (总龙头) | Late entries have negative expected value |
| **Market regime** | Only hunt in "speculative" regime (market-wide board premium > 1%) | Monster stocks underperform in risk-off markets |
| **Volume ceiling** | Skip if daily turnover > 40% | Extreme turnover = distribution phase |
| **Fundamental floor** | Skip if revenue < 100M RMB AND net income negative for 3+ quarters | Avoid pure shell companies at risk of delisting |
| **Position sizing** | Max 5% of portfolio per monster stock candidate | Asymmetric risk/reward requires strict sizing |

---

## 5. Mapping to Current System Architecture

### What the project already has:
- Alpha158 features via Qlib (`factors/quant.py`)
- LightGBM for short-term prediction (`scripts/train_lgb.py`)
- Transformer + SAC RL agent (`models/rl_agent.py`)
- LLM-based sentiment analysis (`signals/llm_analyst.py`, `data/collectors/sentiment.py`)
- Multi-timeframe signal fusion (`signals/scorer.py`)
- Geopolitical/macro factors (`factors/geopolitical.py`, `data/collectors/macro.py`)
- WeChat push notifications (`push/wechat.py`)

### Highest-impact additions (ordered by effort/impact):

1. **Limit-up chain data collection** — Add `ak.stock_zt_pool_em()` to data collectors. Low effort, high signal for monster stocks.
2. **Monster stock composite score** — New factor module implementing Section 4.3 scoring. Medium effort, directly actionable.
3. **Sector heat tracker** — Track limit-up counts by concept/industry. Feeds sector rotation signals. Medium effort.
4. **Expand RL state to sequence** — Fix the current Transformer to process multi-day sequences instead of single-step. Medium effort, significant model improvement.
5. **GNN for stock relations** — Build industry/concept graph. High effort but captures the sector-rotation dynamics that drive A-share alpha.
6. **FinRL integration** — Replace per-stock RL with portfolio-level RL from FinRL. High effort, better capital allocation.
7. **Triple-barrier labeling** — Replace simple forward-return labels with Lopez de Prado's triple-barrier method (take-profit, stop-loss, time expiry). Medium effort, better training signal.

---

## Sources

- [Qlib - Microsoft](https://github.com/microsoft/qlib)
- [FinRL - AI4Finance Foundation](https://github.com/AI4Finance-Foundation/FinRL)
- [FinRL-X Trading](https://github.com/AI4Finance-Foundation/FinRL-Trading)
- [TradeMaster](https://github.com/TradeMaster-NTU/TradeMaster)
- [NautilusTrader](https://github.com/nautechsystems/nautilus_trader)
- [VectorBT](https://github.com/polakowo/vectorbt)
- [Backtrader](https://github.com/mementum/backtrader)
- [Zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded)
- [QuantConnect/Lean](https://github.com/QuantConnect/Lean)
- [RQAlpha](https://github.com/ricequant/rqalpha)
- [VNPY](https://github.com/vnpy/vnpy)
- [AKShare](https://github.com/akfamily/akshare)
- [TuShare](https://github.com/waditu/tushare)
- [awesome-quant](https://github.com/wilsonfreitas/awesome-quant)
- [awesome-systematic-trading](https://github.com/wangzhe3224/awesome-systematic-trading)
- [Factor Investing: Fama-French 5-Factor on Chinese A-Shares](https://alphaarchitect.com/factor-investing-fama-french-5-factor-model-chinese-shares/)
- [Factor Models for Chinese A-Shares](https://www.sciencedirect.com/science/article/pii/S105752192300491X)
- [Machine Learning in the Chinese Stock Market](https://www.sciencedirect.com/science/article/pii/S0304405X21003743)
- [AI-Driven Anomaly Detection in Stock Markets](https://link.springer.com/article/10.1007/s10614-025-11274-8)
- [Speculative Trading and Stock Prices (Xiong)](https://www.princeton.edu/~wxiong/papers/china.pdf)
- [TGNS: Transformer-Graph Neural Network for Stock Forecasting](https://www.sciencedirect.com/science/article/abs/pii/S0020025525006887)
- [CNN-LSTM-GNN for A-Share Prediction](https://www.mdpi.com/1099-4300/27/8/881)
- [TFT-GNN Hybrid Model](https://www.mdpi.com/2673-9909/5/4/176)
- [Marcos Lopez de Prado Publications](https://www.quantresearch.org/Publications.htm)
- [FinRL Contest 2025](https://open-finance-lab.github.io/FinRL_Contest_2025/)
- [Backtrader vs VnPy vs Qlib Comparison (2026)](https://dev.to/linou518/backtrader-vs-vnpy-vs-qlib-a-deep-comparison-of-python-quant-backtesting-frameworks-2026-3gjl)
