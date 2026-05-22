# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

本项目为 **A股股票多因子量化交易平台**，目标对标百亿级量化私募水平。策略方向为日频调仓的截面多因子选股，技术栈以 Python 为主、Rust（通过 PyO3/Polars）处理性能关键模块。

MVP 聚焦 A 股日频。远期扩展方向：港股/美股、期货/期权/可转债、分钟级调仓、市场中性/指数增强。但 MVP 阶段不做这些。

---

## 一、技术架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                     可视化层 / Dashboard                      │
│                  Streamlit / Dash / Jupyter                   │
├──────────────────────────────────────────────────────────────┤
│                       交易执行层                              │
│         XtQuant/QMT(股票) / VNPY(期货CTA) / XTP              │
├──────────────────────────────────────────────────────────────┤
│                       组合优化层                              │
│            cvxpy (核心) + Riskfolio-Lib (快速原型)             │
├──────────────────────────────────────────────────────────────┤
│                       风险模型层                              │
│            自建 Barra CNE5 模型 (statsmodels + numpy)          │
├──────────────────────────────────────────────────────────────┤
│                    因子研究 & 回测层                           │
│            QLib (核心引擎) + Alphalens (因子评估)              │
├──────────────────────────────────────────────────────────────┤
│                        数据层                                 │
│     AKShare/Tushare(免费) + Wind/聚宽(付费) → 本地数据库       │
│                DuckDB / ClickHouse / Parquet                  │
├──────────────────────────────────────────────────────────────┤
│                      基础设施层                               │
│   Dagster/Prefect(初期) → Airflow(成熟期) + Redis + MinIO     │
└──────────────────────────────────────────────────────────────┘
```

---

## 二、开源框架选型与整合

### 核心框架

| 层次 | 首选 | 备选 | 选型理由 |
|------|------|------|---------|
| 数据获取 | **AKShare** (~9K stars) | Tushare Pro (~13K stars) | AKShare完全免费、覆盖面最广、维护活跃；Tushare财务数据质量更高 |
| 数据存储 | **DuckDB** | ClickHouse (分布式), Parquet文件, **ArcticDB**(版本化时间序列) | DuckDB嵌入式零配置、列式存储、SQL接口，单机开发首选；ArcticDB用于因子/预测/数据快照的immutable版本管理 |
| 因子研究 | **QLib** (~15K stars, 微软) | 自建因子引擎 | 内置Alpha158因子库、ML Pipeline、因子表达式引擎 |
| 因子评估 | **Alphalens-reloaded** | QLib内置 | IC/分组收益/换手率的标准可视化报告 |
| 风险模型 | **自建Barra** | Riskfolio-Lib | 无成熟开源Barra实现，需用statsmodels+numpy自建CNE5 |
| 组合优化 | **cvxpy** (~5.5K stars) | PyPortfolioOpt, Riskfolio-Lib | 最灵活，支持A股特殊约束（行业中性、换手率、涨跌停） |
| 回测引擎 | **QLib** | VectorBT (快速验证) | 日频多因子用向量化回测，QLib链路完整 |
| 实盘交易(股票) | **XtQuant/QMT**（后期预研） | VNPY(期货/CTA)、XTP、恒生 | 非开源，是券商商业通道。MVP不要求开通；模拟盘稳定60天后预研试接 |
| 调度编排 | **Dagster/Prefect**（初期） | Airflow（成熟期） | 初期Dagster更轻量、适合数据资产建模和本地研发；Airflow适合稳定后的大规模日批 |
| 可视化 | **Streamlit** | Dash, Jupyter | 快速构建监控Dashboard |

### 整合设计原则

1. **统一数据格式**：以 DataFrame 为核心，列命名规范：`date`, `asset`, `open`, `high`, `low`, `close`, `volume`, `factor_xxx`
2. **分层解耦**：每层通过 Protocol/ABC 定义接口契约，可独立替换
3. **不让任一开源项目成为全系统中心**：Qlib/Alphalens/cvxpy/DuckDB各有边界，内部接口隔离它们。内部 FactorStore 是数据真值，Qlib只是适配层。禁止把Qlib默认数据格式当唯一存储格式

### 与cx文档的对齐记录（经多轮论证）

以下是cx文档提出的建议中，经过多轮证据化论证后的最终共识。大方向已对齐，仅Polars选型一项保留分歧：

**1. AKShare定位（已接受cx观点）。**
AKShare官方声明"数据仅供学术研究"，底层依赖外部网站。AKShare+Tushare可作为MVP的**研究数据入口**；落入raw layer、通过schema校验、多源抽样校验、生成immutable `data_version`后，可作为"**研究快照真值**"。机构级生产真值需要授权、PIT、退市覆盖、公司行动、修订历史、SLA和可追责供应商。

**2. Dagster引入时机（已接受cx观点）。**
调度器由任务复杂度触发，不按日历硬切。触发条件：(a)日常任务数>10且依赖连续10个交易日稳定；(b)失败重跑超15分钟人工处理；(c)已有data_version/factor_version/backtest_id且CLI已能跑通。Dagster只加可观测性和重试，不定义口径。

**3. Polars vs 自定义Rust（仍维持原判）。**
用Polars替代Pandas不需要profiling——Polars本身是Rust实现，性能优势是已知事实。只有写自定义Rust扩展（因子表达式引擎）时才需要profiling。

**4. AI研发效率（已接受cx修正）。**
AI显著提高候选代码产出速度，但不能说"不会犯错"。Copilot安全研究显示~40%生成代码存在漏洞；EvalPlus扩展测试后pass@k下降19-29%。量化代码风险更隐蔽（未来函数、字段口径错、复权错）。所有AI代码必须通过单元测试、未来函数扫描、样本手算和结果审计。**AI是速度乘数，不是质量豁免。**

```python
class DataLoader(Protocol):
    def load(self, assets, start, end, fields) -> pd.DataFrame: ...

class AlphaModel(Protocol):
    def predict(self, data) -> pd.Series: ...  # asset -> score

class RiskModel(Protocol):
    def get_factor_exposure(self) -> pd.DataFrame: ...
    def get_factor_cov(self) -> pd.DataFrame: ...
    def get_specific_risk(self) -> pd.Series: ...

class Optimizer(Protocol):
    def optimize(self, alpha, risk_model, constraints) -> pd.Series: ...
```

### 各框架详细说明

**QLib (microsoft/qlib)**：AI驱动量化投研平台，内置Alpha158/Alpha360因子库、多种深度学习模型（LSTM/Transformer/TCN）、因子表达式引擎。学习曲线陡峭，回测引擎偏简单，生产级需二次开发交易执行层。原生支持A股。

**VNPY (vnpy/vnpy)**：国内最成熟的交易框架，CTP/恒生/XTP接口，事件驱动。**更适合期货/期权/CTA**，股票多因子实盘还需评估XtQuant/QMT、XTP和券商PB系统。不能假设VNPY直接解决股票多因子执行。

**Backtrader (~14K stars)**：成熟但原作者已停止维护（~2021），性能不佳，不原生支持A股。不推荐作为核心，可参考设计思路。

**Rqalpha (ricequant)**：A股交易规则支持最好（T+1、涨跌停、停牌），mod插件架构灵活。更新频率下降，可作为A股规则参考。

**VectorBT**：向量化回测性能极快（比事件驱动快100-1000倍），适合大规模参数扫描和因子筛选。不支持复杂交易逻辑。Pro版商业化。

---

## 三、因子数据库调研

### 免费数据源

| 数据源 | 覆盖范围 | 特点 | 适用场景 |
|--------|---------|------|---------|
| **AKShare** | 行情/财务/宏观/另类，数据种类最全 | 完全免费开源，**底层爬虫机制，字段稳定性和授权边界不适合直接做生产真值** | 早期原型和探索，非核心数据补充 |
| **Tushare Pro** | 行情/财务/因子/行业分类 | 积分制(捐赠100-500元可解锁日常研究)，数据有专人校验 | 早期主力数据源，生产需本地版本化+多源校验 |
| **BaoStock** | 日/分钟K线/财务/估值 | 完全免费无限制，学术背景，稳定性好 | 入门研究，分钟数据 |
| **JoinQuant (jqdatasdk)** | 行情/财务/因子库(500+)/tick级 | 免费版每日100万条限额，API设计优秀 | 因子研究，财务数据标准化好 |

> **重要**：免费数据源适合早期验证流程，但核心回测必须本地落库并做多源校验。禁止在回测时直接调用远程API。

### 付费专业数据源

| 数据源 | 价格 | 核心优势 | 百亿私募使用率 |
|--------|------|---------|--------------|
| **Wind 万得** | 待供应商确认 | 最全面最权威，PIT数据，一致预期，行业标准 | 机构级主数据源 |
| **朝阳永续** | 待供应商确认 | 分析师一致预期数据业内领先，历史修订完整 | 预期因子研究 |
| **Choice 东方财富** | 待供应商确认 | Wind替代候选，资金流数据独家 | 中小机构 |
| **恒生聚源** | 待供应商确认 | 数据库交付适合企业级，财务数据标准化高 | 公募/大型私募 |
| **同花顺 iFinD** | 待供应商确认 | 产业链数据独家，AI辅助功能，可做Wind交叉验证 | 中小机构 |
| **天软科技** | 待供应商确认 | Tick级高频数据核心优势，L2逐笔数据 | 高频策略 |
| **聚宽 JoinQuant** | 待供应商确认 | 因子库完善、API优秀 | 中小私募 |
| **米筐 RiceQuant** | 待供应商确认 | RQAlpha生态、API设计好 | 量化社区 |
| **通联 DataYes** | 待供应商确认 | 因子库设计专业 | 因子库参考 |

> **注意**：以上价格和"机构使用率"为调研线索，非稳定事实。供应商价格受模块、字段、并发、历史长度、API权限和谈判条款影响。采购前必须逐项向供应商确认字段清单、PIT支持、更新频率、授权范围、API限额和SLA。

### 另类数据源

| 数据类型 | 来源 | 对应因子 |
|---------|------|---------|
| 舆情/新闻 | NLP自建 / Wind资讯 / 同花顺 | 情绪因子、关注度因子、分析师语调因子 |
| 分析师预期 | **朝阳永续**(首选) / Wind一致预期 | SUE、盈利修订因子、预期分歧度因子 |
| 基金持仓 | Tushare/AKShare/Wind (公开数据) | 机构持股因子、公募拥挤度因子 |
| 龙虎榜 | 交易所公开 (免费源可满足) | 机构净买入因子、游资活跃度因子 |
| 大宗交易 | 交易所公开 (免费源可满足) | 折价率因子、大股东减持信号 |

### 数据源组合推荐

| 预算 | 方案 |
|------|------|
| **0元/学习** | BaoStock + AKShare + RQAlpha开源 |
| **500-3000元/年** | Tushare Pro(捐赠) + JoinQuant免费版 + AKShare |
| **1-5万/年** | JoinQuant机构版 + Choice个人版 + Tushare |
| **5-15万/年** | Choice/iFinD机构版 + 朝阳永续基础版 |
| **15-50万/年** | Wind机构版 + 朝阳永续 + 恒生聚源(交叉验证) |
| **50万+/年** | 全部顶级源 + 自建另类数据团队 |

### 数据使用关键原则

1. **PIT（Point-in-Time）**：财务数据必须用实际公告日期而非报告期，否则引入严重前视偏差
2. **多源交叉验证**：核心策略至少两个数据源验证关键数据
3. **分阶段投入**：先用免费源验证策略逻辑，跑通后再投入专业数据
4. **数据合规**：管理规模增大后应转向正规数据源

---

## 四、因子体系

### 因子分类

#### 价值因子（Value）

| 因子 | 公式 | 说明 |
|------|------|------|
| EP | 净利润(TTM) / 总市值 | PE的倒数，数值更稳定 |
| BP | 净资产 / 总市值 | |
| SP | 营业收入(TTM) / 总市值 | |
| CFTP | 经营现金流(TTM) / 总市值 | |
| DP | 每股股息 / 股价 | |

> 实践中取倒数形式（EP而非PE），因为PE在盈利接近0时极端大

#### 动量因子（Momentum）

| 因子 | 公式 | A股特点 |
|------|------|--------|
| 短期反转(STR) | R(t-20, t-1) | **A股短期反转效应非常显著**（与美股相反） |
| 中期动量(MOM) | R(t-252, t-21) | 12月收益扣除最近1月 |
| 残差动量 | Fama-French回归残差累计收益 | 剔除市场和风格影响 |

#### 质量因子（Quality）

ROE、ROA、毛利率、净利率、经营现金流/净利润、应计项目(负向)、Gross Profitability

#### 成长因子（Growth）

营收增速(YoY)、净利润增速、ROE变化、SUE(标准化未预期盈余)

#### 波动率因子（Volatility）

总波动率、特质波动率(IVOL)、下行波动率、Beta
> A股低波动率效应（Low Volatility Anomaly）非常稳健

#### 流动性因子（Liquidity）

换手率(对数)、Amihud非流动性指标 `ILLIQ = mean(|r_i| / Volume_i)`、成交金额

#### 技术因子（Technical）

移动均线偏离度、RSI、MACD、布林带位置、量价背离

#### 另类因子（Alternative）

分析师预期调整、大股东增减持、北向资金持仓变化、新闻情绪NLP、事件驱动(回购/定增)

### 因子方向处理

因子库中统一记录每个因子的"预期方向"，统一为"值越大越看好"：

```python
FACTOR_DIRECTION = {
    'EP': 1,           # 值越大越便宜，正向
    'PE': -1,          # 值越大越贵，反向
    'ROE': 1,          # 值越大盈利越强，正向
    'VOLATILITY': -1,  # 值越大波动越大，反向
    'TURNOVER': -1,    # 值越大流动性越好但可能是投机，视策略定
}

for name, direction in FACTOR_DIRECTION.items():
    factors[name] = factors[name] * direction
```

> 在IC加权合成或ML模型中，方向由权重/模型自动处理，不需要手动翻转

---

## 五、因子预处理（截面处理）

### 什么是截面

"截面"（Cross-Section）指 **同一时间点上所有股票的某个因子值的集合**。例如2024年1月31日A股4000+只股票的EP值构成一个截面。

截面处理的目的：消除量纲差异、去除极端值、剔除市值和行业的系统性影响。

### 完整处理流水线

```
原始因子值 → 缺失值处理 → 去极值(MAD) → Z-Score标准化 → 市值+行业中性化 → 再次标准化
```

**处理顺序的原因**：
- 先去极值：不让极端值干扰后续标准化和回归
- 先标准化再中性化：中性化本质是回归，输入标准化后值更稳定
- 中性化后再标准化：让最终因子在截面上均值为0、标准差为1

### 步骤1：缺失值处理

```python
def fill_missing(factor_df, industry_col='industry'):
    factor_cols = [c for c in factor_df.columns if c != industry_col]
    for col in factor_cols:
        # 用行业中位数填充
        factor_df[col] = factor_df.groupby(industry_col)[col].transform(
            lambda x: x.fillna(x.median())
        )
        # 整个行业缺失则用全市场中位数
        factor_df[col] = factor_df[col].fillna(factor_df[col].median())
    return factor_df
```

> 一般原则：缺失比例低时直接剔除该股票；必须填充时用行业中位数

### 步骤2：去极值（Winsorize）

**MAD法（推荐）** — 基于中位数，对极端值本身更稳健：

```python
def winsorize_mad(series, n=5):
    median = series.median()
    mad = (series - median).abs().median()
    sigma_mad = 1.4826 * mad  # 正态分布下MAD与标准差的等价系数
    upper = median + n * sigma_mad
    lower = median - n * sigma_mad
    return series.clip(lower=lower, upper=upper)
```

**3-sigma法** — 简单但均值和标准差本身受极端值影响：

```python
def winsorize_3sigma(series, n=3):
    mu, sigma = series.mean(), series.std()
    return series.clip(lower=mu - n * sigma, upper=mu + n * sigma)
```

**分位数法** — 无分布假设，固定截断比例：

```python
def winsorize_percentile(series, lower_pct=0.01, upper_pct=0.99):
    return series.clip(lower=series.quantile(lower_pct), upper=series.quantile(upper_pct))
```

| 方法 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| MAD | 稳健，不受极端值影响 | 假设对称分布 | **最推荐** |
| 3-sigma | 简单直观 | 均值和标准差受极值污染 | 一般 |
| 分位数 | 无分布假设 | 固定比例截断不够灵活 | 适用 |

### 步骤3：标准化（Standardize）

**Z-Score标准化**（保留因子值间距信息）：

```python
def standardize_zscore(series):
    return (series - series.mean()) / series.std()
```

**Rank标准化**（映射到正态分布，完全消除极端值）：

```python
from scipy.stats import norm

def standardize_rank(series):
    ranked = series.rank(method='average')
    percentile = ranked / (series.count() + 1)
    return pd.Series(norm.ppf(percentile), index=series.index)
```

> 实践最常用组合：MAD去极值 + Z-Score标准化

### 步骤4：中性化（Neutralize）

剔除行业和市值对因子的系统性影响。数学方法：**截面回归取残差**。

```
f_i = β₀ + β₁·ln(MarketCap_i) + Σγ_k·Industry_k_i + ε_i
```

ε_i 即为中性化后的因子值。

```python
import statsmodels.api as sm

def neutralize(factor_series, market_cap, industry):
    ln_cap = np.log(market_cap)
    industry_dummies = pd.get_dummies(industry, drop_first=True, dtype=float)
    X = pd.concat([ln_cap.rename('ln_cap'), industry_dummies], axis=1)
    X = sm.add_constant(X)
    
    valid = factor_series.notna() & ln_cap.notna()
    model = sm.OLS(factor_series[valid], X.loc[valid]).fit()
    return model.resid  # 残差 = 中性化后的因子
```

### 完整截面处理Pipeline

```python
def process_factor_cross_section(factor_values, market_cap, industry):
    valid_mask = factor_values.notna()
    factor = factor_values[valid_mask].copy()
    factor = winsorize_mad(factor, n=5)        # 去极值
    factor = standardize_zscore(factor)         # 标准化
    factor = neutralize(factor, market_cap[valid_mask], industry[valid_mask])  # 中性化
    factor = standardize_zscore(factor)         # 再标准化
    return factor
```

---

## 六、因子合成

### 等权合成

```python
combined = factor_df.mean(axis=1)  # 简单稳健，无需估计参数
```

### IC加权

按历史平均IC分配权重（IC带符号，自动处理反向因子）：

```python
def ic_weighted_combine(factor_df, ic_series_dict, lookback=12):
    weights = {name: ic_series_dict[name].tail(lookback).mean()
               for name in factor_df.columns}
    total = sum(abs(v) for v in weights.values())
    weights = {k: v / total for k, v in weights.items()}
    return sum(factor_df[name] * w for name, w in weights.items())
```

### ICIR加权

ICIR = IC均值/IC标准差，衡量预测力的稳定性：

```python
def icir_weighted_combine(factor_df, ic_series_dict, lookback=12):
    weights = {}
    for name in factor_df.columns:
        ic = ic_series_dict[name].tail(lookback)
        weights[name] = ic.mean() / ic.std() if ic.std() > 0 else 0
    total = sum(abs(v) for v in weights.values())
    weights = {k: v / total for k, v in weights.items()} if total > 0 else {k: 1/len(weights) for k in weights}
    return sum(factor_df[name] * w for name, w in weights.items())
```

### 最大化IC_IR优化

等价于 Markowitz 均值-方差优化：`w* = Σ_IC⁻¹ · IC_mean`

```python
def max_icir_combine(factor_df, ic_df):
    ic_mean = ic_df.mean().values
    ic_cov = ic_df.cov().values + np.eye(len(ic_mean)) * 1e-6  # 正则化
    w = np.linalg.solve(ic_cov, ic_mean)
    w = w / np.sum(np.abs(w))
    return pd.Series(factor_df.values @ w, index=factor_df.index)
```

### 机器学习合成

特征=各标准化因子值，标签=未来N期收益率：

```python
import lightgbm as lgb

# 关键注意：
# 1. 严格时间序列交叉验证（TimeSeriesSplit），不能随机划分
# 2. 滚动训练：每月重新训练
# 3. 防过拟合：控制树深度、正则化、early_stopping
```

---

## 七、因子评价指标

### IC / Rank IC

因子值与未来收益的截面相关系数。**Rank IC（Spearman秩相关）更稳健，实践中更常用。**

```python
from scipy.stats import spearmanr

def calc_rank_ic(factor_cross_section, return_cross_section):
    common = factor_cross_section.dropna().index.intersection(return_cross_section.dropna().index)
    if len(common) < 30:
        return np.nan
    rank_ic, _ = spearmanr(factor_cross_section[common], return_cross_section[common])
    return rank_ic
```

| 月频 Rank IC 绝对值 | 评价 |
|---------------------|------|
| > 0.05 | 较强 |
| 0.03 ~ 0.05 | 一般 |
| < 0.03 | 较弱 |

> **初筛参考，非机械准入规则**。不同频率、股票池、行业中性方式、成本和持有期下阈值不同。准入应综合：经济逻辑、样本外、walk-forward、分年度稳定性、成本后收益、容量和相关性。

### IC_IR

`IC_IR = mean(IC) / std(IC)`，衡量IC稳定性。IC_IR > 0.5 为不错的因子（初筛参考）。

### 分组收益

按因子值排序分5组或10组，计算每组平均收益。好因子应呈现**清晰单调递增/递减**。

### 换手率分析

`Turnover = (新进入股票数 + 退出股票数) / (2 × 总持仓数)`

换手率过高意味着交易成本侵蚀收益。A股双边成本约 0.3%~0.5%。

### IC衰减分析（IC Decay）

测试因子值与不同滞后期收益率的IC。IC缓慢衰减→适合月度调仓；IC快速衰减→需要周度/日度调仓。

---

## 八、回测体系

### 向量化 vs 事件驱动

| 维度 | 向量化 | 事件驱动 |
|------|--------|---------|
| 速度 | 快（NumPy矩阵运算） | 慢（Python循环） |
| 灵活性 | 低 | 高（复杂订单逻辑） |
| 前视偏差风险 | 高（容易误用未来数据） | 低（按时间推进） |
| 适用 | **日频多因子（主流选择）** | CTA/高频 |

### 日频多因子回测时间轴

```
T-1月末           T月第1个交易日      T月第2个交易日
  │                    │                    │
  │  因子计算日         │  信号生成日         │  交易执行日
  │  (用T-1月末数据)    │  (排序选股)         │  (以开盘价/VWAP买入)
```

> 信号日与交易日之间至少留1个交易日延迟，避免前视偏差

### 回测陷阱

| 陷阱 | 具体表现 | 解决方案 |
|------|---------|---------|
| **前视偏差** | 使用未公布的季报数据 | 严格按公告日期（PIT）使用数据 |
| **前视偏差** | 用当日收盘价做决策并以收盘价成交 | 信号与执行至少间隔一日 |
| **前视偏差** | 使用供应商当前前复权/后复权价格 | **保存未复权价+公司行动+复权因子，按ex-date生成收益** |
| **前视偏差** | 使用当前行业分类/指数成分 | 使用历史时点数据 |
| **幸存者偏差** | 样本只含存活股票 | 使用全市场历史数据（含退市股） |
| **过拟合** | 回测夏普>3但实盘衰减 | 样本外测试、限制参数数量、经济逻辑检验 |

#### 财务数据PIT时间表（A股）

```
一季报：报告期3/31，法定披露截止4/30
半年报：报告期6/30，法定披露截止8/31
三季报：报告期9/30，法定披露截止10/31
年报：  报告期12/31，法定披露截止次年4/30
```

### A股特殊规则

| 规则 | 处理方式 |
|------|---------|
| **T+1交易制度** | 当日买入次日才能卖出 |
| **涨跌停** | 涨停无法买入(跳过)，跌停无法卖出(延迟) |
| **停牌** | 不可买卖（`tradable=false`）。持仓估值：沿用停牌前价格或按基金估值规则调整。复牌后收益集中体现。回测需区分 `tradable`/`price_available`/`valuation_price`/`execution_price` |
| **ST/\*ST** | 多因子选股中直接剔除，涨跌停5% |
| **新股** | 剔除上市不足60~250个交易日的股票 |
| **复权** | 保存未复权价格+公司行动+复权因子，回测时按ex-date/asof_time生成收益。简单使用供应商当前前复权或后复权序列均有未来信息风险 |

### A股交易成本

| 成本项 | 比例 |
|--------|------|
| 券商佣金 | 0.02%~0.03% (机构费率) |
| 印花税 | 0.05% (仅卖出) |
| 冲击成本 | 0.05%~0.2% |
| **保守估计双边总成本** | **0.3%~0.6%** |

### 绩效评估指标

#### 绝对收益

```python
annual_return = (1 + total_return) ** (252 / n_days) - 1
annual_vol = daily_returns.std() * np.sqrt(252)
sharpe = (annual_return - rf) / annual_vol
max_drawdown = ((cummax - nav) / cummax).max()
calmar = annual_return / max_drawdown
```

| 夏普比率 | 评价 |
|---------|------|
| > 2.0 | 优秀（或过拟合警惕） |
| 1.0 ~ 2.0 | 良好 |
| 0.5 ~ 1.0 | 一般 |

#### 相对基准（超额收益）

基准选择：沪深300(大盘)、中证500(中小盘)、中证1000(小盘)

```python
alpha_return = portfolio_return - benchmark_return
information_ratio = alpha_return.mean() / alpha_return.std() * np.sqrt(252)
```

| 信息比率 | 评价 |
|---------|------|
| > 1.5 | 优秀 |
| 1.0 ~ 1.5 | 良好 |
| 0.5 ~ 1.0 | 一般 |

#### 换手率与成本侵蚀

```
年化成本侵蚀 = 年化换手率 × 单次双边成本
```

例：月频调仓，单次换手30%，双边成本0.3% → 年化 3.6×0.3% = 1.08%

---

## 九、百亿私募完整业务流程

### 组织架构

**PM制**（幻方早期、九坤、明汯）：每个PM独立管理子组合，拿超额收益15%-25%分成。创新快但信号可能重叠。

**平台化制**（衍复、灵均）：统一因子库和模型，研究员按因子贡献度考核。因子协同效应强，风控统一。

**混合制**（多数头部趋势）：核心alpha用PM制，基础设施和风控中心化。

### 完整投研Pipeline

#### 1. 数据层

**数据采集**：L1/L2行情(交易所直连/Wind)、财报、分析师预期(朝阳永续)、另类数据(舆情NLP/卫星/电商)

**清洗流程**：
```
原始数据 → 格式标准化 → 缺失值处理 → 异常值检测 → 复权处理 → 时间对齐 → 质量校验 → 入库
```

**存储选型**：

| 数据类型 | 存储方案 |
|---------|---------|
| Tick级行情 | ClickHouse / Arctic / HDF5 |
| 日频因子矩阵 | ClickHouse / Parquet + HDFS |
| 基本面数据 | PostgreSQL |
| 非结构化(新闻/公告) | MongoDB / Elasticsearch |
| 模型文件 | MinIO / NFS |
| 热数据缓存 | Redis |

**数据分层**：ODS(原始) → DWD(清洗明细) → DWS(汇总/因子) → ADS(应用/模型输入)

#### 2. 因子挖掘

**手工因子**：学术论文/直觉 → 假设 → 数学表达 → 编码 → 单因子测试 → 评审入库

**遗传规划(GP)** — 头部私募大规模使用：

```
搜索空间：
  运算符: +, -, *, /, log, abs, rank, ts_mean, ts_std, ts_corr, ts_rank,
          ts_delta, ts_decay_linear, cs_rank, cs_zscore, ...
  操作数: open, high, low, close, volume, vwap, returns, ...
  时间窗口: 5, 10, 20, 60, ...

适应度函数: Rank IC / ICIR / 多空夏普（惩罚过于复杂的表达式）

工具: gplearn(开源基础), 头部私募自研GPU加速GP引擎
```

**深度学习因子**：

| 模型 | 特点 |
|------|------|
| LSTM/GRU | 捕捉时序依赖，最早广泛使用 |
| Transformer | 注意力机制，近年主流 |
| GNN | 捕捉股票间关系(行业链/资金流) |
| TCN | 膨胀因果卷积 |
| AlphaNet | 自动学习量价因子的网络结构 |

#### 3. 因子筛选入库流程

```
Step 1: 单因子IC测试 — Rank IC > 0.02(日频), ICIR > 0.5
Step 2: 分组回测 — 单调性检验, 多空夏普 > 2
Step 3: 相关性检验 — 与现有因子相关系数 < 0.7, 增量IC > 0.01
Step 4: 稳健性检验 — 分年度/分市场环境/不同参数/不同股票池
Step 5: 逻辑审查 — 经济学解释, 数据窥探风险评估
Step 6: 观察期 — 进入"观察池"跟踪实时表现3-6个月后正式入库
```

#### 4. 因子库管理

- **生命周期**：构思 → 研发 → 回测验证 → 入库审批 → 观察期 → 正式上线 → 持续监控 → 衰减/淘汰
- **规模**：手工因子数百~数千个，机器挖掘数万~数十万个，筛选后实际使用数百~数千个
- **监控**：滚动IC/ICIR(20/60/120日窗口)、因子收益率、因子拥挤度、换手率变化
- **衰减处理**：IC持续下降→降权；与新因子高相关→合并/淘汰；定期(季度)全面复审

#### 5. 模型构建

**线性模型**：截面回归 `r_{i,t+1} = α + Σβ_k·f_{k,i,t} + ε`，常用WLS(市值加权)、岭回归(处理共线性)

**机器学习**：XGBoost/LightGBM(最广泛)、随机森林、神经网络。几乎所有头部私募使用模型集成(Ensemble)。

**训练实践**：
- Expanding/Rolling Window时间序列交叉验证
- 标签：未来N日收益率截面排名
- 超参调优：Optuna(贝叶斯优化)

**信号合成**：多模型/因子信号合成→等权/IC加权/IR加权/Stacking meta-model

#### 6. 组合优化

**Barra CNE5 风险模型** — 10个风格因子 + 30个行业因子：

| 风格因子 |
|---------|
| Size(市值)、Beta、Momentum(动量)、Residual Volatility(残差波动率) |
| Non-Linear Size、Book-to-Price(账面市值比)、Liquidity(流动性) |
| Earnings Yield(盈利收益率)、Growth(成长)、Leverage(杠杆) |

**优化目标**：

```
maximize: α'w - λ·w'Σw - tc(w, w₀)

subject to:
  Σw = 0 或 1                    # 市场中性或纯多头
  |w_i| ≤ 1%~3%                  # 个股权重上限
  |w'·X_industry| ≤ 3%~5%        # 行业偏离限制
  |w'·X_style| ≤ 0.2~0.5σ        # 风格暴露限制
  Σ|w - w₀| ≤ turnover_limit     # 换手率限制
```

工具：cvxpy + MOSEK/Gurobi 求解器

#### 7. 风控体系

**事前风控**：个股权重上限、行业偏离限制、风格暴露约束、预期波动率上限(年化8-12%)、禁止池(负面新闻/流动性不足/即将退市)

**事中风控**：实时P&L监控、Beta/行业/风格暴露实时计算、熔断机制(日亏1.5%→减仓/停止)

**事后风控**：收益归因、风险归因、TCA交易成本分析、压力测试(2015股灾/2018贸易战/2020新冠情景)

#### 8. 交易执行

**算法交易**：

| 算法 | 原理 | 适用 |
|------|------|------|
| TWAP | 按时间均匀分拆 | 流动性好的简单场景 |
| VWAP | 按历史成交量分布分拆 | **最常用** |
| IS | 最小化决策价与执行价差 | alpha衰减快时 |
| POV | 按当前成交量固定比例参与 | 控制市场影响 |

**冲击成本模型**：`Impact = σ · √(Q/ADV) · sign(side)`

**柜台系统**：恒生PTFX/O32、迅投QMT、自研OMS

#### 9. 绩效归因

**Brinson归因**：超额收益 = 行业配置收益 + 个股选择收益 + 交互效应

**因子归因**：组合收益 = 市场收益 + Σ(因子暴露×因子收益) + 选股alpha

**收益分解**：
```
总收益
├── 基准收益(Beta)
├── 行业配置收益
├── 风格因子收益(Size/Value/Momentum/...)
├── Alpha收益(选股)
│   ├── 各因子/模型贡献
│   └── 残差
└── 交易成本(佣金+税费+冲击+滑点)
```

### 日度任务调度时间线

```
15:00  收盘
15:05  数据入库（行情、财报）
15:30  数据质量检查
15:35  因子计算（数百因子并行）
16:30  因子计算完成
16:35  模型预测（生成alpha信号）
17:00  组合优化（生成目标持仓）
17:30  交易指令生成
17:45  风控审核
18:00  交易指令就绪
09:15  次日盘前检查
09:30  开盘执行
```

### 技术基础设施

| 资源 | 百亿私募典型配置 |
|------|-----------------|
| GPU集群 | 数十~数百张A100/H100，用于DL训练和GP挖掘 |
| CPU集群 | 数百~上千核 |
| 内存 | 数TB级总内存 |
| 存储 | PB级（Tick数据+因子矩阵） |
| 集群管理 | K8s + Slurm(GPU调度) |
| 分布式文件系统 | Lustre / CephFS / HDFS |
| 代码管理 | GitLab私有部署，严格Code Review |
| CI/CD | GitLab CI / Jenkins |

---

## 关于事件驱动回测引擎（LEAN/NautilusTrader）的立场

cx文档推荐LEAN或NautilusTrader做事件驱动撮合回测。**本项目不采纳此建议作为核心路径**，理由如下：

1. **日频多因子不需要订单簿级撮合**。日频调仓的核心逻辑是：收盘后算因子 → 生成目标持仓 → 次日以VWAP/开盘价执行。不存在限价单排队、部分成交、盘口博弈等高频问题。向量化回测 + 真实的成本/流动性/涨跌停约束完全够用。
2. **LEAN是C#核心**。纯Python团队二次开发LEAN的成本极高，且国内A股交易规则（T+1、涨跌停、分红送转、ST）需要全部自行适配，ROI不合理。
3. **NautilusTrader的Rust核心**虽然性能好，但国内生态和券商适配几乎为零。cx自己也承认"国内生态和券商适配少"。
4. **百亿私募的实际做法**：头部私募（幻方/九坤/衍复）的日频多因子回测都是自研向量化引擎，不用LEAN/Nautilus。事件驱动引擎用于CTA/高频/期权做市等场景。
5. **"回测-实盘一致性"不靠引擎解决**。真正的一致性来自：相同的Universe规则、相同的因子版本、相同的成本模型、相同的约束条件。这些都是业务逻辑层面的，不是撮合引擎层面的。

**本项目的回测路径**：
- 自研向量化日频回测引擎（Polars/NumPy）
- 内置真实的A股约束（T+1、涨跌停、停牌、ST、新股、复权）
- 内置参数化成本模型（固定费率 + 冲击成本 + 流动性约束）
- 当扩展到CTA/期货/日内策略时，再引入事件驱动引擎

## 关于Barra风险模型的实现策略

cx文档建议"不要把商业Barra术语当作已实现能力，先做简化可解释风险模型"。

**我们的立场**：方法论必须完整文档化（已在第十一章完成），但实现分阶段递进。

| 阶段 | 实现内容 | 对应USE4章节 |
|------|---------|------------|
| Phase 3a | 10风格因子暴露计算 + 30行业因子 + WLS截面回归 | Sec 2-3 |
| Phase 3b | 因子协方差（指数衰减 + Newey-West） | Sec 4.1 |
| Phase 3c | 特质风险（时间序列估计 + Bayesian Shrinkage） | Sec 5.1-5.2 |
| Phase 4+ | Eigenfactor Risk Adjustment | Sec 4.2, App B |
| Phase 4+ | Volatility Regime Adjustment | Sec 4.3, 5.3 |

先用简化版（Phase 3a-3c）跑通组合优化流程，再逐步加入高级修正。完整的USE4方法论文档（第十一章）是目标参考，不是第一天就要全部实现。

---

## 十A、数据版本化与可复现性

> 吸收自cx文档的核心工程纪律。回测结果不可复现 = 没有价值。

### 版本化体系

```
每次回测必须绑定：
  data_version    → 原始数据快照（immutable）
  factor_version  → 因子计算输出版本
  model_version   → 模型参数和权重版本
  universe_rule   → 股票池规则版本
  backtest_id     → 回测配置哈希
  code_version    → Git commit hash
```

### 数据时间语义

每条数据必须标注 `asof_time`（该数据在现实中可获得的最早时间）：

```python
# 错误：用报告期作为可用时间
financial_data.loc['2024-03-31']  # 一季报3/31，但实际4月底才公告

# 正确：用公告日期
financial_data_asof = financial_data.set_index('announce_date')
```

### 存储规范

```
data/
  raw/           # 供应商原样，不覆盖，按 vendor/dataset/date 分区
  clean/         # 清洗后，统一schema，按 data_version 快照
  factors/       # 因子输出，按 factor_version 锁定
  predictions/   # 模型预测，按 model_version 锁定
```

- 原始数据落 Parquet，每个 data_version 是 **immutable snapshot**
- 因子版本通过 ArcticDB 或 Parquet + 目录分版本管理
- 回测只读固定版本，**禁止在回测时直接调用远程API**

---

## 十B、Universe 构建

### 分层股票池

```python
UNIVERSE_RULES = {
    'all_a_active':    '当日上市且未退市',
    'tradable_a':      'all_a_active - 停牌 - 上市不足60日 - ST - 日均成交额<500万 - 涨跌停',
    'csi300':          '沪深300历史成分（按当日成分，非当前成分）',
    'zz500':           '中证500历史成分',
    'zz1000':          '中证1000历史成分',
    'zz2000':          '中证2000历史成分',
    'custom_research': 'tradable_a 且满足自定义条件',
}
```

### 关键原则

1. **Universe必须按当日可得信息生成**，不可用未来的指数成分回看
2. **指数成分必须使用历史成分**（不是今天的CSI300成分回溯10年）
3. **退市股票必须保留**，否则收益虚高（幸存者偏差）
4. 每个研究和回测必须**显式声明**使用的universe规则
5. 每个交易日输出 `tradable_mask`，标记可买/可卖/不可交易

---

## 十C、成本与容量压力测试

### 参数化成本模型

```
total_cost = fixed_bps + spread_bps × spread_multiplier + impact_coef × (trade_value / adv)^α
```

其中 `adv` 为过去20日日均成交额，`α` 通常取0.5（Square Root Impact Model）。

### 强制压力测试矩阵

上线前每个策略必须跑完以下组合：

| 维度 | 测试值 |
|------|--------|
| 成本倍数 | 0.5x, **1x**, 2x, 3x |
| 成交量占比上限 | 5%, 10%, **15%**, 20% |
| 成交价假设 | open, **VWAP**, close |
| 调仓延迟 | 0天, **1天**, 2天 |

策略在 2x 成本 + 10%成交量占比 + 延迟1天条件下仍盈利才可考虑上线。

### 标签（Forward Return）精确定义

```
signal_date = t 日收盘后产生信号
entry_price = t+1 日 VWAP（或 open）
exit_price  = t+1+h 日 VWAP（或 close）
forward_return_h = exit_price / entry_price - 1
```

- 单因子IC检验可先不扣成本
- **组合回测必须扣成本**
- 涨跌停买不到/卖不出不能假设成交

---

## 十D、合规与监管

> 2024年证监会发布《证券市场程序化交易管理规定（试行）》，2024-10-08起实施。系统从第一天必须保留合规能力。

### 必须内置的合规能力

| 要求 | 实现方式 |
|------|---------|
| 程序化交易报告信息管理 | 策略、账户、资金、交易权限映射表 |
| 订单和算法参数留痕 | 结构化JSON日志，不可篡改 |
| 高频交易监控指标 | 撤单率、订单频率、成交量占比 |
| 异常交易监控 | 价格偏离、集中度、关联交易 |
| 日志不可篡改存储 | append-only存储，保留至少5年 |
| 数据授权记录 | 每个数据源的商用授权范围 |
| 权限隔离 | 研究员不可直接操作交易系统 |

### 关键监管文件

- 证监会《证券市场程序化交易管理规定（试行）》
- 上交所/深交所程序化交易管理实施细则
- 证监会《期货市场程序化交易管理规定（试行）》（2025年，2025-10-09起实施）

---

## 十E、工程纪律（从第一天执行）

### 版本管理

```
代码版本：Git（建议GitLab私有部署）
数据版本：data_version（immutable Parquet snapshot）
因子版本：factor_version
模型版本：model_version（MLflow或内部注册表）
回测版本：backtest_id = hash(data_version + factor_version + model_version + config)
```

### 配置管理

- 所有配置集中管理：YAML/TOML + schema校验
- 禁止硬编码魔法数字（成本率、阈值、窗口长度等）
- 每个配置项有默认值、类型约束、文档说明

### 测试要求

| 类型 | 范围 |
|------|------|
| 单元测试 | 数据处理函数、因子计算、成本模型、约束构建 |
| 回归测试 | 经典策略在固定data_version上的结果不变 |
| 集成测试 | 端到端pipeline（ingest→factor→backtest→report） |

### 日志

- 结构化JSON格式（非print/logging文本）
- 每条日志包含：timestamp, level, module, event, context
- 交易相关日志不可修改、不可删除

---

## 十F、高风险误区清单

以下错误中的任何一个都可能让整个回测结果不可信：

| # | 误区 | 后果 |
|---|------|------|
| 1 | 财务数据按报告期入库，不用公告日 | 严重前视偏差，回测收益虚高30-50% |
| 2 | 回测用当前指数成分回看历史 | 幸存者偏差 |
| 3 | 不处理退市股票 | 收益虚高 |
| 4 | 忽略涨跌停、停牌和T+1 | 不可执行的交易信号 |
| 5 | 成本模型过低（<0.1%单边） | 高换手策略实盘亏损 |
| 6 | 参数搜索后只展示最好结果 | 过拟合 |
| 7 | 用同一测试集反复调参 | 数据窥探 |
| 8 | 用前复权价格计算历史收益 | 前视偏差（前复权价随时间变化） |
| 9 | 供应商当前复权价直接回测多年 | 公司行动的前视偏差 |
| 10 | 模拟盘没跑稳就上实盘 | 系统性风险 |
| 11 | 收益来自行业/市值暴露，误以为alpha | 伪alpha |
| 12 | 没有kill switch和人工接管 | 极端行情下无法止损 |
| 13 | 先做界面不做数据版本 | 结果不可复现 |
| 14 | 直接信任供应商因子不看口径 | 因子定义不明导致错误结论 |
| 15 | 把妖股候选池当成确定性预测器 | 高风险执行失控 |
| 16 | 用盘后龙虎榜/公告/题材归因预测盘中交易 | 前视偏差 |
| 17 | 低估涨停买不到、跌停卖不出的执行风险 | 回测收益虚高 |

---

## 十、开发路线图

### 五个验收关口

每个Phase必须通过对应的关口才能进入下一阶段：

| 关口 | 验收标准 |
|------|---------|
| **数据关口** | 任意历史日期能重建当日universe、复权收益、行业分类、停牌状态、涨跌停状态和财务可用字段 |
| **因子关口** | 每个因子输出raw/processed/neutralized/rank四种形态，生成覆盖率/缺失率/IC/分层收益/换手/相关性报告 |
| **回测关口** | 同一信号在无成本/有成本/2x成本/延迟1天/限制成交量占比/剔除涨跌停等场景下都能跑，差异可解释 |
| **组合关口** | 同一alpha能生成TopK/score weight/行业中性/市值中性/风险约束优化四类组合，输出暴露和换手差异 |
| **归因关口** | 每次回测可分解为：基准收益 + 行业收益 + 风格收益 + 选股alpha + 交易成本 + 现金拖累 + 未成交影响 |

### 核心模块规范

| 模块 | 关键要求 |
|------|---------|
| `security_master` | 不只是代码表：上市/退市、证券简称历史、ST历史、交易状态、交易所、板块、最小交易单位、涨跌停规则 |
| `calendar` | 区分自然日/交易日/半日交易/交易时段/财报披露可用时间/调仓日 |
| `corporate_action` | 保存分红、送转、配股、拆合股、复权因子和ex-date。**不能只保存复权后价格** |
| `data_quality` | 可失败的硬规则：价格为负、成交额为负、同一主键重复、停牌却有成交量、涨跌停价缺失 |
| `factor_registry` | 因子公式、依赖字段、lookback、方向、处理链、适用universe、作者、创建时间、状态、废弃原因 |
| `factor_store` | 按 `factor_id/date/universe/version` 读取，**禁止同名因子覆盖** |
| `label_store` | entry/exit价格、持有期、交易延迟、是否扣行业/市场收益 |
| `backtest_config` | 完全配置化，任何报告都能从配置+数据版本重放 |
| `optimizer` | 不可交易约束→优化前；交易约束→优化中；成交失败→执行模拟。三者不混在一起 |
| `reporting` | 必须输出数据版本、因子版本、参数、交易假设、成本假设和异常列表，不能只输出净值曲线 |

### Phase 0：边界确认（第0周）

确定MVP边界，**不再扩大调研**：
- 市场：A股。频率：日频。产品：指数增强/多头选股
- 第一版目标：研究和回测平台，**不直接实盘**
- 交付物：数据字典、Universe规则、回测假设文档、因子处理规范

### Phase 1：数据底座（第1-3周）

**第1周：项目骨架和数据契约**

目录结构（第1天就建好）：
```
quant/
├── src/quant_platform/
│   ├── __init__.py
│   ├── config.py              # YAML加载 + schema校验
│   ├── logger.py              # 结构化JSON日志
│   ├── data/
│   │   ├── __init__.py
│   │   ├── calendar.py        # 交易日历
│   │   ├── security_master.py # 证券主数据
│   │   ├── corporate_action.py# 分红送转配股复权因子
│   │   ├── connectors/        # AKShare/Tushare适配器
│   │   │   ├── akshare_connector.py
│   │   │   └── tushare_connector.py
│   │   ├── lake.py            # raw layer落盘
│   │   ├── clean.py           # clean layer生成
│   │   ├── quality.py         # 数据质量检查
│   │   └── loader.py          # DataLoader Protocol
│   ├── factor/
│   │   ├── __init__.py
│   │   ├── registry.py        # 因子注册表
│   │   ├── compute.py         # 因子计算引擎
│   │   ├── processors.py      # 去极值/标准化/中性化
│   │   ├── store.py           # FactorStore（版本化读写）
│   │   └── analysis.py        # IC/分组/换手/相关性
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py          # 向量化日频回测引擎
│   │   ├── cost_model.py      # 成本模型
│   │   └── metrics.py         # 绩效指标
│   ├── portfolio/
│   │   ├── __init__.py
│   │   ├── optimizer.py       # cvxpy组合优化
│   │   ├── constraints.py     # A股约束（行业/市值/换手/涨跌停）
│   │   └── risk_model.py      # Barra-like风险模型
│   └── reports/
│       ├── __init__.py
│       └── generator.py       # 静态HTML/Markdown报告
├── configs/
│   ├── data_sources.yaml
│   ├── universe/tradable_a.yaml
│   ├── factors/base.yaml
│   └── reports/daily.yaml
├── docs/
│   ├── data_dictionary.md
│   ├── backtest_assumptions.md
│   ├── factor_processing_spec.md
│   └── data_quality_rules.md
├── tests/
│   ├── test_calendar.py
│   ├── test_security_master.py
│   ├── test_quality.py
│   └── test_factors.py
├── data/                      # 数据目录（.gitignore）
│   ├── raw/                   # 供应商原样，不覆盖
│   └── clean/                 # 清洗后，按data_version快照
├── pyproject.toml
├── CLAUDE.md
└── Makefile                   # CLI入口
```

第1周逐日任务：
- [ ] Day 1：`pyproject.toml`（Python 3.11+、polars/duckdb/akshare/tushare/statsmodels/cvxpy依赖）、`config.py`、`logger.py`、`Makefile`
- [ ] Day 2：`calendar.py`（交易日历，区分自然日/交易日/半日，验收：能正确返回任意日期范围的交易日列表）
- [ ] Day 3：`security_master.py`（证券主数据：代码/名称/上市日/退市日/交易所/板块/ST历史/涨跌停规则，验收：能查询任意日期某只股票是否ST/停牌/已退市）
- [ ] Day 4：`docs/data_dictionary.md`、`docs/backtest_assumptions.md`（写死成交价假设、成本模型、T+1规则、涨跌停处理、停牌处理、复权方式）
- [ ] Day 5：`loader.py`（DataLoader Protocol）、`lake.py`（raw layer Parquet落盘，按vendor/dataset/date分区）、第一个`data_version`生成逻辑

**第2周：原始数据入库和质量检查**
- [ ] Day 1-2：`akshare_connector.py` + `tushare_connector.py`（日线OHLCV、复权因子、股票列表、行业分类、指数成分）
- [ ] Day 3：`corporate_action.py`（分红/送转/配股，保存未复权价+复权因子+ex-date，验收：能按asof_time正确计算任意历史日期的复权收益率）
- [ ] Day 4：`quality.py`（硬规则：价格≤0、成交额<0、同一主键重复、停牌却有成交量、涨跌停价缺失；验收：故意注入脏数据后检查必须失败）
- [ ] Day 5：`clean.py`（clean layer生成+`data_version`快照）、第一份数据质量报告、每条数据标注`asof_time`

**第3周：股票池和可交易性**
- [ ] Day 1-2：Universe构建（全A/沪深300/中证500/中证1000历史成分，**用历史成分不用当前成分**）
- [ ] Day 3：`tradable_mask`生成（过滤：上市<60日、退市、ST、停牌、日均成交额<500万、涨跌停）
- [ ] Day 4：退市股票保留验证（验收：2015年至今所有退市股在退市前的数据完整存在）
- [ ] Day 5：**数据关口验收** — 任意历史日期能重建当日universe、复权收益、行业分类、停牌/涨跌停状态

### Phase 2：因子平台（第4-5周）

**第4周：因子引擎MVP**
- [ ] 实现10个量价因子：1/5/20日反转、20/60日动量、20日波动率、20日换手、Amihud、量价相关、均线偏离
- [ ] 实现 MAD去极值、行业中位数填充、z-score、rank、行业/市值中性化
- [ ] 建立 `factor_registry` 和 `factor_version`
- [ ] 每个因子保存 raw/processed/neutralized/rank 四种形态

**第5周：因子评价**
- [ ] 实现未来 1/5/10/20 日标签（正确处理entry/exit price）
- [ ] 实现 IC、Rank IC、ICIR、分组收益、单调性、换手、覆盖率、相关性矩阵
- [ ] 输出静态因子分析报告（HTML/Markdown）

### Phase 3：模型与组合（第6-8周）

**第6周：基础回测**
- [ ] 实现 TopK等权、score加权持仓
- [ ] 支持月频/周频/日频调仓
- [ ] 加入交易成本、T+1、停牌、涨跌停不可交易、成交量占比限制
- [ ] 输出净值、超额、回撤、换手和成本报告

**第7周：组合约束和简化风险模型**
- [ ] 实现行业暴露、市值暴露、beta暴露计算
- [ ] cvxpy实现行业偏离、单票上限、换手上限、市值中性约束
- [ ] 简化Barra风险模型：10风格因子暴露 + WLS截面回归 + Ledoit-Wolf协方差
- [ ] 对比 TopK / score weight / 行业中性 / 风险约束 组合的差异

**第8周：第一份端到端研究报告**
- [ ] 固定 data_version + factor_version + 回测配置
- [ ] 跑完整 ingest → factor → analysis → portfolio → backtest → report 链路
- [ ] 成本压力测试（0.5x/1x/2x/3x × 不同成交假设）
- [ ] 写一份策略研究报告：收益来源、风险来源、成本敏感性、容量限制

### Phase 4：回测升级与模拟盘（2-3个月）

- [ ] 风险模型升级：加入Newey-West + Bayesian Shrinkage（USE4 Sec 4.1, 5.2）
- [ ] 因子合成模块（等权/IC加权/ICIR加权/LightGBM）
- [ ] 绩效归因模块（Brinson + 因子归因）
- [ ] Dagster/Prefect 日度调度（Phase 1-3用CLI/Makefile跑通，此阶段任务稳定后引入调度器）
- [ ] 模拟盘：每天生成信号和虚拟订单，连续运行3-6个月

### Phase 5：小资金实盘（2-3个月）

- [ ] 股票券商网关：XtQuant/QMT调研，保留VNPY/XTP适配空间
- [ ] 风控系统（事前/事中/事后）+ kill switch
- [ ] 实盘收益 vs 模拟盘对照
- [ ] 合规日志和审计
- [ ] Streamlit 监控Dashboard

### Phase 6+：持续迭代

- [ ] 风险模型升级：Eigenfactor Risk Adjustment + VRA（USE4 Sec 4.2-4.3）
- [ ] GP因子挖掘引擎
- [ ] 深度学习因子（LSTM/Transformer）
- [ ] 因子拥挤度监控
- [ ] ClickHouse集群（分钟/tick数据）
- [ ] 多策略/多账户/多产品管理

### MVP 阶段明确不做的事

以下看起来"高级"但会拖慢当前目标：

- **不做漂亮Dashboard** → 先保证静态报告指标可信
- **不做实盘** → 先模拟盘稳定运行3-6个月
- **不买昂贵数据** → 先用低成本源跑通数据版本和PIT流程
- **不上ClickHouse/Airflow/K8s** → 先把单机链路做正确
- **不做深度学习/GP** → 先建立线性和树模型baseline
- **不做完整Barra CNE5** → 先做简化版风险模型（方法论已文档化，实现分阶段）
- **不做分钟/tick数据** → 日频MVP先跑通
- **不做LEAN/NautilusTrader深度集成** → 日频多因子用自研向量化回测

---

## 十G、单台 Mac Studio 的极致量化路线

> 你拼不过百亿私募的算力（数百张A100+上千CPU核），但可以在**因子质量、数据纪律和策略独特性**上赢。算力是乘数，alpha才是底数。底数为零时乘数再大也没用。

### 硬件现实

Mac Studio（M2 Ultra/M4 Ultra 典型配置）：
- CPU：24核、统一内存64-192GB
- GPU：不支持CUDA，PyTorch MPS后端可用但生态不成熟
- 存储：1-8TB SSD

**这台机器对日频A股多因子完全够用**。A股约5000只股票 × 252个交易日 × 10年 × 100个因子 ≈ 12.6亿个数据点，float64仅约10GB。Polars/DuckDB在这个规模上秒级响应。

### 个人 vs 百亿私募：扬长避短

| 维度 | 百亿私募优势 | 个人可以赢的地方 |
|------|-----------|---------------|
| 算力 | 数百GPU暴力搜索因子空间 | **不需要暴力搜索：用领域知识+经济逻辑构建因子，比GP挖出的噪声因子更稳健** |
| 数据 | Wind全量+另类数据+专有数据 | **免费数据足够覆盖量价+基础财务因子，这些因子在A股仍然有效** |
| 团队 | 50+研究员并行挖因子 | **聚焦窄问题，但因子相关性必须实测（AI同源因子容易同质化）** |
| 容量 | 管理百亿需要分散到1000+只股票 | **资金量小=可以集中持仓20-50只，捕捉小盘/微盘alpha** |
| 速度 | 策略上线需层层审批 | **发现机会到执行可以当天完成** |
| 执行 | 算法交易降低冲击成本 | **资金量小=大中盘线性冲击低，但小微盘/妖股有严重不可成交和退出风险** |

### 个人的核心策略方向

**不要复制百亿私募的策略（宽基指数增强、市场中性）**。它们的alpha来自大规模因子挖掘+极致风控+低冲击执行，个人无法复制。

**应该做私募做不了或不愿做的事**：

1. **小盘/微盘股alpha**
   - 百亿私募因为容量限制（单票不超过日均成交额的10%），无法重仓市值<30亿的股票
   - 个人资金量小，可以自由交易微盘股
   - A股小盘效应极其显著，且因为机构无法参与而不被套利掉
   - 因子：微盘size、小票反转、小票特质波动率

2. **高换手短周期因子**
   - 百亿私募换手率受限（每天换手5-15%已经很高，交易成本巨大）
   - 个人资金量小，冲击成本≈0，可以做日频甚至更高频的换手
   - 因子：1-5日反转、日内量价模式、盘后信息反应

3. **事件驱动 + 因子叠加**
   - 百亿私募需要统计显著性覆盖全市场，不会为单只股票做特殊判断
   - 个人可以在因子信号基础上叠加事件判断（公告、定增、回购、股权激励、龙虎榜）
   - 结合因子排名 + 事件触发 = 更精确的择时

4. **另类因子（不需要买数据）**
   - 龙虎榜数据（免费，AKShare可获取）→ 机构席位净买入因子
   - 大宗交易数据（免费）→ 折价率因子
   - 北向资金持仓变化（免费）→ 聪明钱因子
   - 股东增减持公告（免费）→ 内部人信号
   - 这些数据百亿私募也用，但你的优势是可以集中在信号最强的少数股票上

### Mac Studio 上的技术栈优化

```
计算：Polars（比pandas快10-50x，原生多核） + NumPy + SciPy
存储：Parquet + DuckDB（零配置，SQL查询）
ML模型：LightGBM/XGBoost（CPU上也很快，不需要GPU）
        PyTorch MPS 可用于小型神经网络实验
优化：cvxpy + OSQP/ECOS（开源求解器在这个规模足够）
回测：自研向量化（Polars/NumPy，秒级完成日频回测）
```

**不需要的**：
- GPU集群（日频多因子不需要大规模DL训练）
- ClickHouse（DuckDB在日频数据量级完全够用）
- Airflow/K8s（cron + Makefile + Python脚本）
- 分布式计算（单机192GB内存可以放下全部A股日频数据）

### 关于"找妖股"

"妖股"（短期暴涨的股票）本质上是**极端动量+情绪驱动+资金合力**的结果。从量化角度：

**可以做的**：
- 构建**异常检测因子**：成交量突变（volume_ratio = 今日成交量/过去20日均量）、换手率飙升、涨停打开次数
- **龙虎榜+资金流信号**：当知名游资席位集中买入时，短期有动量延续
- **情绪因子**：股吧/雪球讨论热度（需要NLP，但小规模可行）
- **技术形态识别**：突破形态、放量突破均线等，可以用量价因子表达

**不建议做的**：
- 试图"预测"哪只股票会成为妖股 — 这是随机性极强的事件，回测容易过拟合
- 追涨已经连续涨停的股票 — 个人不具备游资的盘口控制能力

**务实的做法**：不追求"预测"妖股，而是构建"核心-卫星-现金"三层结构。

### 资金结构：核心-卫星-现金

| 层 | 来源 | 目标 | 标的 | 仓位 | 频率 |
|----|------|------|------|------|------|
| **核心仓** | 稳健多因子 | 稳定、低换手、可解释 | 中证1000/2000/全A tradable | **主要资金** | 周频或月频 |
| **卫星仓** | 妖股/题材候选池 | 捕捉高弹性，严控亏损 | 小盘/微盘/题材强势股 | **小，单票更小** | 日频观察，实盘前长期模拟 |
| **现金/防守** | 风险开关 | 不在情绪退潮时被打穿 | - | 动态 | 触发式 |

**现金触发条件**：全市场炸板率高、跌停数上升、连板高度下降、指数破位、候选池大面率升高、模型连续失效。

**核心原则：核心仓让你活下去，卫星仓给你上限，现金让你不在情绪退潮时被打穿。**

### 主线A：小盘/微盘多因子（核心仓）

在中证2000/国证2000的universe中构建持续超额收益的因子体系。研究压力测试目标年化超额15-20%（非预期收益承诺，真实预期由样本外、模拟盘和小资金验证决定）。

### 主线B：妖股/题材情绪候选池（卫星仓）

不是"预测下一只妖股"的神谕，而是一个情绪监控和候选筛选系统：
- 找出短线情绪正在升温的股票
- 标记首板/二板/三板后的延续概率
- 标记炸板、次日低开、大面风险
- 监控题材扩散和龙头切换
- **输出观察清单，不自动下单**

#### 标签定义（三类）

| 类型 | 标签 |
|------|------|
| **潜力标签** | 未来5日最大涨幅≥20%、未来10日最大涨幅≥30%、未来N日出现3连板+ |
| **延续标签** | 首板后次日是否涨停、二板后是否晋级三板、次日是否高开收正 |
| **风险标签** | 次日最大回撤>8%、炸板后收盘未封住、次日低开低走、未来3日最大跌幅>15% |

标签必须按信号日之后的可交易价格计算。涨停买不到、跌停卖不出不能假设成交。

#### 低算力特征（日频即可，Mac Studio完全胜任）

**量价和涨停结构**：
- 当日是否涨停、首板/二板/三板/高度
- 封板时间、炸板次数、开板成交额、封单占流通市值比例
- 过去1/3/5/10日涨幅、振幅、换手率、成交额放大倍数
- 量比、换手分位数、流通市值、价格、上市天数

**题材和市场情绪**：
- 同概念涨停家数、同概念平均涨幅
- 全市场涨停数/跌停数/炸板率/连板高度
- 昨日连板股次日溢价
- 指数环境：上证/创业板/北证/微盘股指数涨跌

**龙虎榜和资金行为**（免费数据）：
- 买入前五集中度、卖出前五集中度
- 机构席位净买入
- 知名游资席位出现次数
- 营业部历史胜率和偏好题材

**公告和注意力**：
- 重大公告、重组、并购、业绩预告
- 新闻数、互动易问答热度

> 关键：保存 asof_time，避免用盘后才知道的信息预测盘中可交易结果。

#### 模型（从简到繁）

```
第一阶段：规则打分（涨停结构 + 题材扩散 + 流通市值 + 换手 + 市场情绪）
第二阶段：Logistic Regression（延续/炸板二分类baseline）
第三阶段：LightGBM/CatBoost（处理非线性和交互）
```

不要一开始做：深度学习、大模型读新闻自动下单、高频盘口预测、自动追涨实盘。

#### 评估指标

| 指标 | 说明 |
|------|------|
| Top N候选次日平均收益 | 信号质量 |
| Top N候选最大回撤 | 风险控制 |
| 晋级率 | 首板→二板、二板→三板 |
| 炸板率和大面率 | 风险频率 |
| 命中妖股的提前天数 | 信号时效 |
| 候选池覆盖率 | 真正妖股有多少进入过候选池 |
| 误报成本 | 每天候选太多→不可执行 |

#### 候选池输出格式

候选池不输出"买入"，只输出：

```
观察等级：A / B / C / 排除
核心理由：题材 / 涨停结构 / 资金 / 量价 / 市场情绪
最大风险：炸板 / 监管 / 流动性 / 高位 / 题材退潮
相似历史样本：N个，平均次日收益X%
明日验证点：是否高开、是否放量、是否快速封板、是否题材继续扩散
```

每天候选不超过20只。删除无法理解、无法成交、风险标签过高的候选。写下明日验证点，**不临盘改理由**。

#### 交易纪律（写进系统，不靠自觉）

- 候选池只输出观察等级，**不自动下单**
- **严禁用涨停价假设能买入**
- **严禁用跌停价假设能卖出**
- 高位股设置最大亏损、最大仓位、最大持有天数
- 单票仓位远低于多因子组合
- 连续亏损后自动暂停观察池交易
- 被监管关注/严重异动/停牌风险/退市风险直接降级

### Mac Studio 每日运行流程

```
盘后 (15:00-18:00)：
  1. 拉取日线、停牌、涨跌停、ST、指数、行业、公告、龙虎榜
  2. 生成 clean data_version
  3. 运行数据质量检查
  4. 计算多因子 + 妖股事件特征
  5. 生成多因子回测/模拟盘信号
  6. 生成妖股候选池
  7. 输出静态报告

盘前 (08:30-09:15)：
  1. 检查昨日报告和异常
  2. 过滤今日不可交易/停牌/风险警示/涨跌停风险
  3. 人工确认观察清单

盘中：
  1. 初期不做自动交易
  2. 记录候选股表现（涨停打开、封板、炸板、成交额变化）
  3. 用记录反哺模型

盘后复盘：
  1. 候选股分数 vs 实际表现对照
  2. 更新训练样本
  3. 记录错误类型：假题材、弱市场、过高换手、庄股嫌疑、监管风险
```

### 30/90/180天计划

**前30天：只做系统，不判断能力**
- 建data_version、security_master、calendar、corporate_action、tradable_mask
- 建10个量价因子
- 建3个妖股标签：未来5日最大涨幅、次日延续、次日大面
- 建静态报告
- 每天跑通盘后流程

**第31-90天：只做模拟盘，不做实盘**
- 多因子核心仓模拟
- 妖股候选池每日出榜
- 每天记录候选理由和次日验证点
- 每周复盘：命中/误报/大面/错过
- 每月删掉无效特征，不新增复杂模型
- **每周只改一次规则，避免日内情绪化调参**

**第91-180天：小资金验证，只验证流程**
- 先只做多因子核心仓小资金
- 妖股候选池继续人工观察，不自动下单
- 候选池连续60个交易日稳定，才允许极小仓位试验
- 每次实盘交易必须映射到模型分数、候选理由、风险标签和当时报告

**禁止**：看到一次命中就加仓 / 没买到就改规则追高 / AI解释听起来合理就跳过回测 / 小资金就忽略滑点

### CLI 命令结构

```bash
quant ingest --date 2026-05-20           # 数据入库
quant validate --data-version latest      # 质量检查
quant events extract --date 2026-05-20    # AI事件结构化
quant factors run --config configs/factors/base.yaml
quant labels run --horizons 1,5,10,20
quant candidates run --config configs/candidates/anomaly.yaml
quant backtest run --config configs/backtests/core_multifactor.yaml
quant report daily --date 2026-05-20
quant review update --date 2026-05-21     # 复盘更新
```

配置目录：
```
configs/
  data_sources.yaml        universe/tradable_a.yaml
  factors/base.yaml        candidates/anomaly.yaml
  models/continue_1d.yaml  risk/default.yaml   reports/daily.yaml
prompts/
  event_extraction_v1.md   candidate_reason_v1.md   postmortem_v1.md
```

> **所有AI prompt必须版本化。prompt改了→事件标签版本也要变。**

### 底线判断

"靠AI一夜找出明天涨停" → **赌博，不做。**
"用AI把个人研究流程压缩到机构小团队水平" → **可行，Mac Studio足够。**

**绝对不要做**：把AI解释当alpha / 模型说"题材强"就追涨停 / 没有asof_time的回测 / 盘后龙虎榜预测盘中 / 小样本妖股回测当规律 / 候选池误报时加仓赌反转 / 删除失败样本美化曲线 / 把低冲击误解为无流动性风险

**实施顺序**：数据可信 → 因子可信 → 候选池可复盘 → 模拟盘稳定 → 极小资金验证。**顺序乱了，AI只会让错误更快、更像真的。**

### AI 以小搏大：Claude Code 作为你的量化研究团队

你没有50个研究员、没有数百GPU、没有Wind全量数据。但你有一个24小时不累、能读论文能写代码能做回测的AI。以下是**AI在每个环节具体怎么帮你最大化产出**：

#### 1. AI 替代研究员挖因子

百亿私募靠50个研究员各自挖因子。你可以让 Claude Code：
- **批量实现学术因子**：给一篇论文或一个公式，AI直接生成完整的因子计算代码+截面处理+IC评估+分组回测
- **从101 Formulaic Alphas批量生成A股适配版**：逐个公式翻译成Polars代码，自动跑IC筛选
- **从券商研报提取因子逻辑**：读取研报PDF，提取因子定义，实现代码
- **因子变体搜索**：给定一个有效因子，自动生成参数变体（不同窗口、不同权重、加行业中性等），批量评估

**实操流程**：
```
你说："把Novy-Marx的Gross Profitability因子实现一下，用Polars，跑IC和分组收益"
AI：生成完整代码 → 你运行 → 几秒出结果 → 你说"换成TTM口径试试" → AI改代码 → 再跑
一个小时可以迭代10+个因子变体，相当于一个研究员一周的工作量
```

#### 2. AI 替代数据工程师

- **数据入库和清洗代码**：告诉AI数据源和字段，它写完整的ETL pipeline
- **数据质量检查**：AI生成自动化检测脚本（负价格、重复主键、跳变、停牌却有成交等）
- **corporate_action处理**：AI实现分红送转配股的复权因子计算
- **PIT校验**：AI实现公告日期 vs 报告期的asof_time验证逻辑

#### 3. AI 替代风控和归因工程师

- **风险模型代码**：AI实现Barra-like的截面回归+协方差估计+特质风险
- **归因分解**：AI实现Brinson归因+因子归因的完整代码
- **压力测试矩阵**：AI生成成本×成交量占比×延迟的全组合回测脚本

#### 4. AI 做妖股候选池的核心引擎

AI在妖股方向的最大价值是**"把非结构化信息变成可回测事件"**，不是直接判断涨跌。

**每天盘后**，AI把公告/龙虎榜/新闻转成结构化事件表：

```python
event_schema = {
    'event_date': '2026-05-20',
    'symbol': '000001.SZ',
    'event_type': '公告|龙虎榜|题材|政策|业绩|监管|互动易|新闻',
    'event_strength': 3,        # 0-5
    'novelty_score': 4,         # 是否新题材、新催化
    'theme': '机器人|AI|低空经济|算力|并购|重组|国企改革',
    'theme_rank': 1,            # 该股在题材内强度排名
    'risk_flag': '监管关注|ST|退市|高位异动|减持|业绩雷',
    'asof_time': '2026-05-20 16:30:00',
    'source': 'eastmoney_announcement',
    'summary': '...'
}
```

**AI prompt必须固定且版本化**（prompt改了→事件标签版本也要变）：

```
你是A股事件结构化助手。只基于输入文本提取事实，不预测股价。
输出JSON：{event_type, theme, event_strength(0-5), novelty_score(0-5),
          risk_flags[], summary, uncertainty(low|medium|high)}
禁止输出投资建议。无法判断则 uncertainty=high。
```

其他AI用途：
- **复盘分析**：候选vs实际表现给AI，分析错误类型
- **特征工程**：描述一个模式（"首板放量二板缩量"），AI立刻生成量化特征代码
- **公告解读**：公告文本→利好程度+板块关联的结构化判断

#### 5. AI 的不可替代优势：速度

| 任务 | 研究员 | AI + Mac Studio |
|------|--------|----------------|
| 实现一个新因子（含处理+评估） | 1-3天 | **10-30分钟** |
| 批量回测10个参数组合 | 半天 | **5分钟** |
| 写一个数据清洗pipeline | 1-2天 | **30分钟** |
| 读一篇论文并复现核心方法 | 1-2周 | **1-2小时** |
| 生成完整的因子分析报告 | 半天 | **即时** |

**一个人+AI的候选代码产出速度可接近5-10个初级研究员**，但AI代码必须经过测试、审计和Failure Library约束（AI是速度乘数，不是质量豁免）。

#### 6. 具体的每日AI工作流

```
盘后 (15:00-17:00)：
  1. 运行自动化pipeline（数据→因子→候选池→报告）—— AI之前已写好的代码
  2. 检查数据质量报告 —— 有异常时问AI排查

研究时间 (晚上/周末)：
  3. 告诉AI本周要研究的方向（比如"换手率因子在微盘股中的IC衰减"）
  4. AI写代码→你运行→看结果→告诉AI"不对，可能是停牌股没过滤"→AI改→再跑
  5. 一个晚上可以完成3-5个研究迭代

每周复盘：
  6. 把一周的模拟盘数据给AI，让它做归因分析
  7. AI输出：收益来源、因子贡献、暴露变化、异常交易、改进建议
```

#### 7. Mac Studio 资源分配

```
内存分配（假设128GB统一内存）：
  - DuckDB/Polars工作内存：~40GB
  - A股全量日频数据（10年×5000股）：~15GB
  - 因子矩阵（100因子×10年）：~10GB
  - 模型训练（LightGBM/XGBoost）：~10GB
  - 系统+其他：~53GB 富余

存储分配（假设2TB SSD）：
  - raw data（多版本）：~100GB
  - clean data：~50GB
  - 因子store（多版本）：~200GB
  - 回测结果和报告：~50GB
  - 剩余 ~1.5TB 足够

CPU分配（24核）：
  - Polars默认使用全部核心，截面计算秒级
  - LightGBM训练：8-12核，几分钟完成
  - 并行回测多个参数组合：按核数并行
```

**结论：这台机器的算力瓶颈不在硬件，在于你能多快产出有效的研究迭代。AI把迭代速度提升10倍，硬件不是限制。**

### 以小搏大的学术和工程证据

#### 证据1：微盘股因子利差是大盘的2-3倍（OSAM研究）

O'Shaughnessy Asset Management (OSAM) 2017年发表的研究 "[Microcaps — Factor Spreads, Structural Biases, and the Institutional Imperative](https://www.osam.com/Commentary/microcaps-factor-spreads-structural-biases-and-the-institutional-imperative)" 给出了量化证据：

- 微盘股（市值$50M-$200M，对应A股约3-15亿）的多因子选股利差（value + momentum）是大盘股的 **2-3倍**
- 微盘股中最小的因子利差（Earnings Quality, 14.5%）都比大盘股中最大的因子利差（Value, 12.5%）更宽
- 原因：机构因容量限制无法参与 → 分析师覆盖少 → 信息反映慢 → 定价低效持续存在
- 个人投资者在微盘股中有**结构性优势**：不受容量限制、不需要满足基准跟踪误差、不受投资委员会审批

**对A股的映射**：A股小微盘是个人重点研究方向，但股票数量、可交易性和容量必须按每日流通市值、成交额、ST状态和涨跌停动态计算universe（不写固定数量，因为随市场变化）。每次报告必须输出小微盘容量表：样本数量、中位成交额、单票参与率、买入/退出天数。

#### 证据2：AI自动量化研发实现2倍收益（Microsoft RD-Agent）

微软2025年发表的 "[R&D-Agent-Quant](https://arxiv.org/html/2505.15155v2)" 论文证明：

- LLM驱动的多Agent框架自动完成因子挖掘→代码实现→回测验证→分析改进的闭环
- 在CSI300上实现 **2倍于经典因子库的年化收益**，且只用了 **70%更少的因子**
- 整个优化周期的API调用成本 **<$10**（vs 一个quant研究员数百美元/小时）
- 框架开源：[github.com/microsoft/RD-Agent](https://github.com/microsoft/RD-Agent)

**后期可试验**：RD-Agent官方环境偏Linux/Docker，Mac Studio需通过容器或Linux VM运行。初期先自建Alpha Factory Lite（更小闭环），后期接入RD-Agent时应在隔离环境跑`fin_factor`，只允许写候选因子，不改数据层和回测层。

**成本说明**：RD-Agent本身完全免费开源。"<$10"指LLM API调用费用。RD-Agent底层用 **LiteLLM**，支持100+个LLM提供商，任何OpenAI兼容API都能接。

**LLM方案配置**（`.env`文件）：
```bash
CHAT_MODEL=openai/MiniMax-Text-01          # 模型名，openai/前缀走兼容协议
OPENAI_API_KEY=你的API_Key
OPENAI_API_BASE=https://api.minimax.chat/v1  # 或其他提供商端点
```

**各方案性价比对比**：

| 方案 | 成本 | 代码生成质量 | 适合场景 |
|------|------|-----------|---------|
| Claude API | ~$3/M tokens | 最好 | 复杂因子逻辑、论文复现 |
| GPT-4o | ~$2.5/M tokens | 好 | 通用研究 |
| **MiniMax** | ~¥1/M tokens | 良好 | **性价比最优，国内直连无需代理** |
| DeepSeek-R1 API | ~¥2/M tokens | 好（推理强） | 数学/逻辑密集型因子 |
| 本地 Qwen-72B (Ollama) | $0 | 良好 | 完全离线，Mac Studio推理较慢 |

**推荐**：日常研究循环用MiniMax（便宜+国内快），关键因子逻辑和论文复现用Claude/DeepSeek-R1。

#### 证据3：LLM Alpha挖掘框架爆发（2024-2025）

| 项目 | 核心方法 | 来源 |
|------|---------|------|
| **AlphaAgent** | LLM驱动alpha挖掘+正则化探索对抗alpha衰减 | [arXiv 2502.16789](https://arxiv.org/html/2502.16789v2) |
| **QuantaAlpha** | 进化框架+LLM做alpha变异和选择 | [arXiv 2602.07085](https://arxiv.org/html/2602.07085v1) |
| **Alpha-GPT 2.0** | Human-in-the-Loop AI量化投资 | 2024 |
| **AgenticTrading** | 认知AI+量化，Agent自主提出→测试→改进假说 | [GitHub](https://github.com/Open-Finance-Lab/AgenticTrading) |
| **Brainiac** | Agentic AI驱动Alpha构建器 | [GitHub](https://github.com/jdhruv1503/Brainiac) |

综述论文："[From Deep Learning to LLMs: A survey of AI in Quantitative Investment](https://arxiv.org/pdf/2503.21422)" (2025) 系统梳理了这一领域。

论文合集：[Awesome-LLM-Quantitative-Trading-Papers](https://github.com/Tom-roujiang/Awesome-LLM-Quantitative-Trading-Papers)

#### 证据4：个人散户的普遍失败 — 以及为什么你不同

Barber & Odean的经典研究表明散户平均年化跑输市场3-4%。但这是因为：
- 散户凭情绪交易（追涨杀跌）
- 散户不做系统化研究（看新闻/股吧买股票）
- 散户交易成本高（高频买卖+高佣金）

**你和典型散户的区别**：
- 你用的是系统化因子模型，不是情绪驱动
- 你有AI加速的研发Pipeline，而非人肉看盘
- 你有版本化数据+严格回测，而非"感觉这只票不错"
- 你的交易纪律是代码化的，而非靠自律

学术结论：散户**平均**跑输，但**系统化+纪律化+小容量**的个人投资者有结构性优势（微盘+低冲击+无容量限制）。

#### 实操建议：集成RD-Agent

```bash
# 在Mac Studio上安装和运行RD-Agent
pip install rdagent
# 配置Qlib数据源，指向本地FactorStore
# RD-Agent自动：生成因子假说 → 写代码 → 回测 → 评估 → 改进
# 你只需要审核结果和设定搜索方向
```

这相当于拥有了一个**不眠不休的量化研究员**，它的因子挖掘能力已被学术论文验证。

#### 证据5：可借鉴的项目和竞赛模式

| 项目 | 可吸收的精华 | 落地方式 |
|------|-----------|---------|
| **WorldQuant BRAIN** | "Alpha Factory"机制：统一表达式语言+统一回测标准+统一相关性过滤+统一提交门槛 | 建立`alpha_dsl`，支持rank/ts_rank/delay/correlation等算子；AI生成的因子必须通过语法、未来函数、相关性和样本外检查 |
| **Numerai** | 新因子不能只看IC，要看对集成模型的边际贡献(MMC) | 增加`ensemble_contribution`指标：新因子加入现有集合后，组合收益是否提高、回撤是否下降 |
| **FinRL** | 把交易约束写进环境，不靠口头纪律 | 成本/不可成交/仓位/风险开关都在回测环境内强制执行 |

#### 实施改造：6个必须加入的模块

**1. Alpha Factory Lite（AI因子工厂）**

```
idea → AI生成公式/代码 → 静态检查(未来函数/字段依赖) → 因子计算
→ IC/分组/成本/容量/相关性 → ensemble_contribution → 人工审阅 → 入观察库
```

**2. Attention-Lottery Engine（注意力+彩票风险双标签）**

妖股候选池不只给"延续概率"，还必须给"过热/反转风险"：
- `continuation_score`：题材扩散、封板强度、资金流入、量能放大
- `blowup_score`：过去20日最大单日收益(MAX因子)、特质波动率、换手分位数、散户拥挤度、连续涨停高度、炸板率、监管风险

> 学术依据：Bali et al. (2011) 发现MAX因子（极端正收益）与未来收益负相关 — 越像"彩票"的股票未来越差

**3. Capacity Advantage Report（容量优势量化报告）**

每个策略必须输出不同资金规模下的容量分析：

| 资金规模 | 单票参与率 | 预估买入天数 | 预估退出天数 | 涨跌停不可成交天数 | 成本后收益 |
|---------|----------|-----------|-----------|-----------------|----------|
| 10万 | ... | ... | ... | ... | ... |
| 50万 | ... | ... | ... | ... | ... |
| 100万 | ... | ... | ... | ... | ... |
| 500万 | ... | ... | ... | ... | ... |

只有这个表成立，才能说个人小资金有优势。

**4. Factor Contribution Test（因子边际贡献）**

新因子不能只看IC排行榜。必须计算加入现有因子集合后的边际贡献：
- 组合收益是否提高？
- 回撤是否下降？
- 换手是否恶化？
- 行业/市值暴露是否变差？
- 与已有因子的最大相关性？

**5. Failure Library（失败因子库）**

AI生成的失败因子、过拟合因子、未来函数错误、字段误解、复权错误、样本外失效，**全部进入失败库**。以后AI生成新因子前，**先读取失败库避免重复犯错**。这是个人用AI追赶团队的关键：不是只提高产量，而是积累负反馈。

**6. Research Budget（研究预算）**

每周限定：
- 最多3个研究主题
- 每个主题最多50个变体
- 超过预算的结果只入档，不允许用于当周决策

防止AI把你带进无限搜索和回测过拟合。

---

## 十一、Barra 风险模型精确方法论（基于 USE4/CNE5 官方文档）

> 以下内容提取自 MSCI Barra USE4 Methodology Notes (Menchero, Orr & Wang, 2011) 和 Barra Global Equity Risk Model Handbook。CNE5 方法论与 USE4 基本一致，差异标注于相关位置。

### 11.1 多因子模型核心公式

**股票收益分解**（USE4 Eq. 1.1, 3.1）：

```
r_n = f_c + Σ_i X_{ni} f_i + Σ_s X_{ns} f_s + u_n
```

其中：
- `r_n`：股票 n 的超额收益（local excess return）
- `f_c`：Country因子收益（CNE5中为A股市场因子）
- `X_{ni}`：股票n对行业因子i的暴露（0/1或分数行业暴露）
- `f_i`：行业因子收益
- `X_{ns}`：股票n对风格因子s的暴露（连续值，标准化后）
- `f_s`：风格因子收益
- `u_n`：特质收益

**组合风险分解**（USE4 Eq. 1.5）：

```
var(R_P) = Σ_{kl} X_k^P F_{kl} X_l^P + Σ_n w_n² var(u_n)
           ├── 因子风险 ──────────┤   ├── 特质风险 ──┤
```

其中 `F_{kl}` 是因子协方差矩阵，`X_k^P = Σ_n w_n X_{nk}` 是组合的因子暴露。

### 11.2 因子暴露（Factor Exposures）

#### 描述符标准化（USE4 Eq. 2.4）

```
d_{nl} = (d_{nl}^{Raw} - μ_l) / σ_l
```

- `μ_l`：描述符l的 **市值加权** 均值（cap-weighted mean）
- `σ_l`：描述符l的 **等权** 标准差（equal-weighted std）
- 使用市值加权均值 → 让大盘组合的风格暴露近似为0
- 使用等权标准差 → 防止大盘股主导暴露的尺度

#### 风格因子构建（USE4 Eq. 2.5）

每个风格因子由多个描述符加权合成：

```
X_{ns} = Σ_{l∈s} w_l · d_{nl}
```

描述符权重 `w_l` 通过优化算法确定，目标是最大化模型解释力。最终对风格暴露重新标准化：市值加权均值=0，等权标准差=1。

#### 异常值处理（USE4 Section 2.2）

三组分类法：
1. **极端值**（可能数据错误）→ 直接剔除
2. **合法但极端**的值 → **截断到均值±3倍标准差**（trim to 3σ）
3. **正常值**（±3σ以内）→ 不调整

缺失数据处理：对因子暴露缺失的股票，用相同行业/市值分组的其他股票的暴露做**回归替代**。

#### 行业因子暴露

**单行业**：哑变量（0/1）

**多行业暴露**（USE4 Eq. 2.6-2.8）— USE4的创新：

```
M_n = Σ_k A_{nk} β_k^A + ε_n     (用资产分拆行业暴露)
X_{nk}^A = A_{nk} β_k^A / Σ_k A_{nk} β_k^A

X_{nk} = 0.75 · X_{nk}^A + 0.25 · X_{nk}^S   (资产暴露75% + 营收暴露25%)
```

最多允许5个行业暴露，暴露之和归一化到1。

#### CNE5 的 10 个风格因子（各含多个描述符）

| 风格因子 | 描述符 |
|---------|--------|
| **Size** | ln(总市值) |
| **Beta** | 历史Beta（对市场回归）、Historical Sigma |
| **Momentum** | 相对强度（过去504个交易日收益，半衰期126天，排除最近21天） |
| **Residual Volatility** | 日超额收益标准差、累积超额收益的range、CAPM残差波动率 |
| **Non-Linear Size** | ln(市值)的立方，正交化到Size因子 |
| **Book-to-Price** | 最近报告期的每股净资产 / 股价 |
| **Liquidity** | 月换手率（过去1/3/12个月）、年化换手率、季度平均换手率 |
| **Earnings Yield** | 预测EP（分析师预期EPS/价格）、Trailing EP（TTM EPS/价格）、Cash Earnings-to-Price |
| **Growth** | 每股盈利长期预测增速、每股盈利短期预测增速、内部增长率 |
| **Leverage** | 市场杠杆 D/E（市值）、账面杠杆 D/A、负债权益比 |

#### USE4 的 12 个风格因子

USE4多了 **Dividend Yield** 和 **Non-Linear Beta** 两个因子，共12个风格+60个行业。

#### 因子稳定性系数（USE4 Eq. 2.1）

衡量因子暴露在相邻月份间的稳定性：

```
ρ_{st} = Σ_n v_n^t (X_{nk}^t - X̄_k^t)(X_{nk}^{t-1} - X̄_k^{t-1}) /
         [√(Σ_n v_n^t (X_{nk}^t - X̄_k^t)²) · √(Σ_n v_n^t (X_{nk}^{t-1} - X̄_k^{t-1})²)]
```

经验法则：ρ > 0.90 为好，ρ < 0.80 则该因子不够稳定，不宜纳入模型。

### 11.3 因子收益率估计（WLS截面回归）

**回归方程**（USE4 Eq. 3.1）：

```
r_n = f_c + Σ_i X_{ni} f_i + Σ_s X_{ns} f_s + u_n
```

**加权最小二乘（WLS）**：权重与 `√(市值)` 成反比（即假设特质波动率与市值的平方根成反比）。

**约束条件**（USE4 Eq. 3.3）：
```
Σ_i w_i f_i = 0    (市值加权行业收益之和为零)
```

此约束使 Country 因子收益 `f_c` 近似等于市值加权市场组合收益。

**纯因子组合的解释**：
- 行业因子组合 = 100%做多某行业 + 做空Country因子，风格暴露为零
- 风格因子组合 = 单位暴露的$-neutral组合，行业和其他风格暴露为零

### 11.4 因子协方差矩阵估计

#### 步骤1：因子相关矩阵（Newey-West + 指数衰减）

```
1. 对因子收益时间序列使用指数加权，半衰期 τ_ρ^F
2. 使用Newey-West方法处理序列相关，允许 L_ρ^F 个滞后
3. 缺失因子收益使用EM算法（Dempster 1977）迭代估计
```

#### 步骤2：因子波动率（指数衰减 + Newey-West）

```
对因子波动率使用不同（更短）的半衰期 τ_σ^F
F_ij^0 = ρ_ij σ_i σ_j    (USE4 Eq. 4.1)
```

#### USE4S / USE4L 参数（USE4 Table 4.1，所有值为交易日）

| 参数 | USE4S（短期） | USE4L（长期） | 建议A股 |
|------|-------------|-------------|---------|
| 因子波动率半衰期 `τ_σ^F` | **84** | 252 | 84~120 |
| 因子波动率NW滞后 `L_σ^F` | **5** | 5 | 5 |
| 因子相关半衰期 `τ_ρ^F` | **504** | 504 | 504 |
| 因子相关NW滞后 `L_ρ^F` | **2** | 2 | 2 |
| 因子VRA半衰期 `τ_{VRA}^F` | **42** | 168 | 42~84 |

**关键设计原则**：
- 波动率半衰期 << 相关半衰期（波动率变化快，需要快速响应；相关系数相对稳定）
- 短期模型(S)用于月度调仓，长期模型(L)用于更低频调仓

#### 步骤3：Eigenfactor Risk Adjustment（USE4 Section 4.2, Appendix B）

**问题**：采样误差导致风险模型系统性**低估优化组合的风险**。小特征值被低估约40%。

```
σ_{true} ≈ σ_{pred} / (1 - K/T)    (USE4 Eq. 4.2, Shepard 2009)
```

**修正方法**（Monte Carlo模拟法）：

```
Step 1: F_0 = cov(f, f)                    # 样本因子协方差矩阵 (B1)
Step 2: D_0 = U_0' F_0 U_0                 # 特征分解，D_0为对角阵 (B2)
Step 3: f_m = U_0 b_m                      # 模拟因子收益 (B3)
Step 4: F_m = cov(f_m, f_m)                # 模拟的协方差矩阵 (B4)
Step 5: D_m = U_m' F_m U_m                 # 模拟矩阵的特征分解 (B5)
Step 6: D̃_m = U_m' F_0 U_m                # 模拟特征因子的"真实"方差 (B6)
Step 7: 计算偏差比 v_k = √(D̃_m(k)/D_m(k)) # 模拟偏差 (B7)
Step 8: 对 D_0 的对角元素乘以 v_k²           # 修正特征值 (B8/B10)
Step 9: F_adj = U_0 D_adj U_0'              # 旋转回原始基 (B10)
```

模拟次数通常取1000次取平均。修正后bias statistic从1.3~1.6下降到1.0附近。

#### 步骤4：Volatility Regime Adjustment（VRA）（USE4 Section 4.3）

**目的**：用截面信息校准波动率水平，对市场冲击做出更快响应。

**因子截面偏差统计量**（USE4 Eq. 4.3）：
```
B_t^F = √(1/K · Σ_k (f_{kt}/σ_{kt})²)
```
如果 B_t^F > 1，说明当日实际因子波动超过预测，模型低估了风险。

**因子波动率乘子**（USE4 Eq. 4.4）：
```
λ_F = √(Σ_t (B_t^F)² · w_t)    (指数加权，半衰期 τ_{VRA}^F)
```

**调整后的波动率预测**（USE4 Eq. 4.5）：
```
σ̃_k = λ_F · σ_k
```

等价于将整个因子协方差矩阵乘以 `λ_F²`，**不改变相关结构**。

**实证效果**：
- 2008金融危机时 λ_F 升至约1.45（风险被低估45%），VRA快速调增
- 2009年危机后 λ_F 降至约0.7（风险被高估30%），VRA快速调减
- 无VRA时bias statistic在危机期间偏离1.0达30-40%，有VRA后基本维持在1.0附近

### 11.5 特质风险估计

#### 时间序列估计（USE4 Eq. 5.2）

```
σ_n^TS = C_n^NW · [Σ_t w_t (u_{nt} - ū_n)²]^{1/2}
```

- `w_t`：指数权重，半衰期 `τ_σ^S`
- `C_n^NW`：Newey-West序列相关调整系数（滞后 `L_ρ^S`）
- `u_{nt}`：由日度截面回归得到的特质收益

#### 结构化模型（USE4 Eq. 5.3-5.4）

对IPO新股或薄交易股票（时间序列不足），用截面回归估计：

```
ln(σ_n^TS) = Σ_k X_{nk} b_k + ε_n     (5.3)
σ_n^STR = E_0 · exp(Σ_k X_{nk} b_k)    (5.4)
```

混合预测：`σ̂_n = γ_n σ_n^TS + (1-γ_n) σ_n^STR` （5.5），大部分股票 γ_n = 1。

#### Bayesian Shrinkage（USE4 Eq. 5.6-5.9）

**问题**：极低/极高波动率的股票存在均值回归偏差。

```
σ_n^SH = v_n · σ̄(s_n) + (1-v_n) · σ̂_n       (5.6)
σ̄(s_n) = Σ_{n∈s_n} w_n σ̂_n                   (5.7，市值加权分组均值)
v_n = q|σ̂_n - σ̄(s_n)| / (Δ_σ(s_n) + q|σ̂_n - σ̄(s_n)|)   (5.8)
Δ_σ(s_n) = √(1/N(s_n) · Σ_{n∈s_n} (σ̂_n - σ̄(s_n))²)      (5.9)
```

- `s_n`：股票n所在的市值十分位（size decile）
- `q`：收缩参数，USE4中 q = 0.1
- 偏差越大 → `v_n`越大 → 收缩力度越强

#### 特质VRA（USE4 Eq. 5.10-5.12）

与因子VRA相同逻辑：

```
B_t^S = √(Σ_n w_{nt} (u_{nt}/σ_{nt})²)          (5.10)
λ_S = √(Σ_t (B_t^S)² · w_t)                     (5.11)
σ̃_n = λ_S · σ_n^SH                              (5.12)
```

#### 特质风险参数（USE4 Table 5.1）

| 参数 | USE4S | USE4L | 建议A股 |
|------|-------|-------|---------|
| 特质波动率半衰期 `τ_σ^S` | **84** | 252 | 84~120 |
| NW自相关滞后 `L_ρ^S` | **5** | 5 | 5 |
| NW自相关半衰期 | **252** | 252 | 252 |
| Bayesian收缩参数 q | **0.1** | 0.1 | 0.1 |
| 特质VRA半衰期 `τ_{VRA}^S` | **42** | 168 | 42~84 |

### 11.6 风险模型精度评估（Bias Statistic）

**标准化收益**（USE4 Eq. A1）：`b_{nt} = R_{nt} / σ_{nt}`

**Bias Statistic**（USE4 Eq. A2）：
```
B_n = √(1/(T-1) · Σ_{t=1}^T (b_{nt} - b̄_n)²)
```

完美预测时 B_n ≈ 1。95%置信区间为 `[1-√(2/T), 1+√(2/T)]`。

**滚动12月Bias Statistic**（USE4 Eq. A4）：
```
B_n^τ = √(1/11 · Σ_{t=τ}^{τ+12} (b_{nt} - b̄_n)²)
```

**MRAD**（Mean Rolling Absolute Deviation）（USE4 Eq. A6）：
```
MRAD^τ = 1/N · Σ_n |B_n^τ - 1|
```

正态分布下 MRAD理想值约0.17。

---

## 十二、101 Formulaic Alphas 算子体系（基于 Kakushadze 2015）

> 以下内容提取自 Kakushadze (2015) "101 Formulaic Alphas"，WorldQuant 授权发表的101个生产级量价alpha公式。

### 12.1 核心统计特征

- 平均持有期：**0.6~6.4天**（短线量价因子为主）
- 平均pair-wise相关：**15.9%**（14.3%中位数），低相关有利于合成
- 收益与波动率强相关：`R ~ V^X`，X ≈ 0.76
- 收益与换手率无显著关系（t-stat = -0.57，不显著）
- 80个alpha在论文发表时仍在**生产环境使用**

### 12.2 算子定义（Appendix A.1）

#### 基础运算符

```python
abs(x), log(x), sign(x)           # 标准数学函数
+, -, *, /, >, <, ==, ||           # 标准运算符
x ? y : z                          # 三元条件运算
rank(x)                            # 截面排名（cross-sectional rank）
```

#### 时间序列算子（ts_前缀）

```python
delay(x, d)        # x 在 d 天前的值
delta(x, d)        # x 今天的值 - x 在 d 天前的值 = x - delay(x, d)
correlation(x, y, d)   # x 和 y 过去 d 天的时间序列相关系数
covariance(x, y, d)    # x 和 y 过去 d 天的时间序列协方差
ts_min(x, d)       # x 过去 d 天的最小值
ts_max(x, d)       # x 过去 d 天的最大值
ts_argmax(x, d)    # x 过去 d 天取最大值的那天（距今天数）
ts_argmin(x, d)    # x 过去 d 天取最小值的那天
ts_rank(x, d)      # x 在过去 d 天中的时间序列排名
sum(x, d)          # x 过去 d 天的累加和
product(x, d)      # x 过去 d 天的累乘积
stddev(x, d)       # x 过去 d 天的标准差
```

#### 特殊算子

```python
scale(x, a=1)          # 缩放使 sum(abs(x)) = a（默认a=1）
decay_linear(x, d)     # 过去d天的线性衰减加权平均，权重 d, d-1, ..., 1（归一化到和为1）
indneutralize(x, g)    # 行业中性化：在分组g内做截面去均值
SignedPower(x, a)       # x^a（保持符号）
Ts_Rank(x, d)          # 同 ts_rank
```

#### 输入数据

```python
returns    # 日收盘-收盘收益率
open, close, high, low, volume  # 标准OHLCV
vwap       # 日成交量加权平均价
cap        # 流通市值
adv{d}     # 过去d天的日均成交额（如 adv20, adv60, adv120）
IndClass   # 行业分类（GICS/申万等），用于 indneutralize
```

### 12.3 代表性Alpha公式（按策略类型分类）

#### 反转类（Mean-Reversion）

```
# Alpha#101 - 日内反转：如果日内上涨(close>open)，反向做空
Alpha#101: (close - open) / ((high - low) + .001)

# Alpha#42 - delay-0反转：vwap偏离度排名对比
Alpha#42: rank((vwap - close)) / rank((vwap + close))

# Alpha#33 - 隔夜缺口反转
Alpha#33: rank(-1 * ((1 - (open / close))^1))
```

#### 动量类（Momentum）

```
# 动量基本形式：ln(yesterday's close / yesterday's open)
# Alpha#19 - 带delay的动量信号
Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7))))) *
          (1 + rank((1 + sum(returns, 250))))
```

#### 量价关系类

```
# Alpha#2 - 成交量变化与价格位置的负相关
Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))

# Alpha#6 - 开盘价与成交量的负相关
Alpha#6: (-1 * correlation(open, volume, 10))

# Alpha#44 - 最高价与成交量排名的负相关
Alpha#44: (-1 * correlation(high, rank(volume), 5))
```

#### 行业中性化Alpha

```
# Alpha#48 - 行业中性化后的量价因子
Alpha#48: (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1)), 1), 250) *
          delta(close, 1)) / close), IndClass.subindustry) /
          sum(((delta(close, 1) / delay(close, 1))^2), 250))

# Alpha#58 - 行业中性化后的VWAP因子
Alpha#58: (-1 * Ts_Rank(decay_linear(correlation(
          IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))
```

#### 复杂多因素Alpha

```
# Alpha#36 - 融合量价关系、动量、日内模式的复合因子
Alpha#36: (((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) +
           (0.7 * rank((open - close)))) +
           (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) +
           rank(abs(correlation(vwap, adv20, 6)))) +
           (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
```

### 12.4 Alpha合成方法论（论文Section 2-4的关键结论）

1. **Mega-Alpha合成**：数百~数百万个alpha加权合成为统一信号，自动实现交易的内部对冲（internal crossing）
2. **协方差矩阵建模**：alpha相关矩阵用因子模型近似 `Γ_{ij} = ξ_i² δ_{ij} + Σ Ω_{iA} φ_{AB} Ω_{jB}`
3. **换手率无解释力**：换手率对alpha pair-wise相关系数无统计显著的解释力（t-stat < 2）
4. **收益率的缩放律**：`R ~ V^0.76`，alpha收益率与波动率的0.76次方成正比

---

## 十三、学术因子模型演进与关键论文

### 13.1 因子模型族谱

```
CAPM (Sharpe 1964)                     E[r_i] - r_f = β_i (E[r_M] - r_f)
    │
    ├── Fama-French 3因子 (1992, 1993)   + SMB(小市值) + HML(价值)
    │       │
    │       ├── Carhart 4因子 (1997)      + UMD(动量)
    │       │
    │       ├── FF 5因子 (2015)           + RMW(盈利) + CMA(投资)
    │       │
    │       └── FF 6因子 (2018)           5因子 + UMD(动量)
    │
    ├── q-factor (Hou, Xue & Zhang 2015) Market + Size + ROE + Investment
    │       │
    │       └── q5 (2021)                 + Expected Growth
    │
    └── APT (Ross 1976)                  r = Σ β_k f_k + ε (因子未指定)
            │
            └── Barra MFM (Rosenberg 1974) → USE1/2/3/4, CNE5/6 (工程化实现)
```

### 13.2 核心因子的学术来源

| 因子 | 开创论文 | 核心发现 | A股表现 |
|------|---------|---------|---------|
| **市场(Market)** | Sharpe (1964) CAPM | 系统性风险溢价 | 有效 |
| **规模(Size/SMB)** | Banz (1981), FF (1992) | 小盘股长期跑赢大盘股 | A股小盘效应强 |
| **价值(Value/HML)** | FF (1992) | 高B/P跑赢低B/P | A股有效但波动大 |
| **动量(Momentum)** | Jegadeesh & Titman (1993) | 买赢家卖输家获超额收益 | **A股短期反转显著，中期动量弱** |
| **盈利(Profitability)** | Novy-Marx (2013), FF (2015) | 高毛利/总资产公司跑赢 | 有效 |
| **投资(Investment)** | Titman, Wei & Xie (2004), FF (2015) | 低资本支出跑赢高资本支出 | 有效 |
| **质量(Quality)** | Asness et al. (2019) "QMJ" | 高质量（安全+盈利+成长）系统性跑赢 | 有效 |
| **低波动(Low Vol)** | Ang et al. (2006) | 低波动股票收益反而更高，违反CAPM | **A股非常稳健** |
| **特质波动率(IVOL)** | Ang et al. (2006) | 高IVOL股票未来收益低 | A股显著 |
| **流动性(Liquidity)** | Pastor & Stambaugh (2003) | 非流动性溢价 | 有效 |
| **短期反转** | Jegadeesh (1990) | 过去1月输家未来1月跑赢 | **A股最强因子之一** |
| **应计异象(Accruals)** | Sloan (1996) | 高应计项目公司未来收益差 | 有效 |
| **PEAD** | Bernard & Thomas (1989) | 盈余公告后漂移 | 有效 |
| **Betting Against Beta** | Frazzini & Pedersen (2014) | 做多低beta做空高beta | 有效 |

### 13.3 因子动物园与复制危机

| 论文 | 核心结论 | 对实践的启示 |
|------|---------|-----------|
| **Harvey, Liu & Zhu (2016)** | 已发表400+因子，t>3.0才可信（非传统2.0） | 提高统计检验标准 |
| **Hou, Xue & Zhang (2020)** "Replicating Anomalies" | 452个异象复制，**超半数无法通过检验** | 大量"因子"是数据挖掘产物 |
| **McLean & Pontiff (2016)** | 因子发表后收益**衰减约58%** | 样本外效应大幅减弱 |
| **Cochrane (2011)** Presidential Address | 提出"因子动物园"概念 | 因子需理论基础支撑 |

**实践启示**：
- 不能盲目使用学术论文中的因子，需A股样本外严格验证
- 重视因子的经济学逻辑，纯统计挖掘的因子大概率失效
- 发表后的因子收益衰减是常态，需持续挖掘新因子

### 13.4 机器学习在因子投资中的应用（基于 Coqueret & Guida）

**因子预测模型的一般形式**：`r_{t+1,n} = g(x_{t,n}) + ε_{t+1}`

其中 g 是非线性函数（替代传统线性模型），`x_{t,n}` 是t期的因子值（预测因子），`r_{t+1,n}` 是未来收益。

**关键方法论**：
- **LASSO + Fama-MacBeth**（Feng, Giglio & Xiu 2020）：用LASSO从因子动物园中筛选
- **Bootstrap正交化**（Harvey & Liu 2019）：控制多重检验的FDR
- **三遍PCA**（Giglio & Xiu 2019）：降维提取潜在因子
- **贝叶斯模型比较**（Barillas & Shanken 2018）：评估不同因子模型

**Bayesian化的p值调整**：
```
Bpv = e^{-t²/2} × [prior / (1 + e^{-t²/2} × prior)]
```
引入先验概率，避免p值的误用。

### 13.5 必读书单

| # | 书名 | 作者 | 核心价值 |
|---|------|------|---------|
| 1 | **Active Portfolio Management** | Grinold & Kahn | IR = IC × √BR 基本定律，所有多因子框架理论起点 |
| 2 | **因子投资：方法与实践** | 石川、刘洋溢、连祥斌 | A股最佳实操手册，Barra CNE5实现细节 |
| 3 | **Quantitative Equity Portfolio Management** | Qian, Hua & Sorensen | 因子模型→组合优化→交易成本，理论实践最佳结合 |
| 4 | **Expected Returns** | Antti Ilmanen (AQR) | Value/Carry/Momentum/Volatility四大策略系统论述 |
| 5 | **Advances in Financial Machine Learning** | López de Prado | 分数阶差分、Triple Barrier标签、Purged K-Fold CV |
| 6 | **Machine Learning for Factor Investing** | Coqueret & Guida | ML+因子最全教材，90+预测因子数据集，免费在线 |
| 7 | **Empirical Asset Pricing** | Bali, Engle & Murray | Fama-MacBeth回归、截面回归标准教材 |
| 8 | **Asset Management** | Andrew Ang | 因子投资系统框架 |

### 13.6 核心方法论文档

| 文档 | 内容 |
|------|------|
| Barra USE4 Methodology Notes (2011) | 完整风险模型方法论（本文十一章的来源） |
| Barra CNE5 Fact Sheet | A股10风格+32行业+VRA |
| 101 Formulaic Alphas (Kakushadze 2015) | 101个生产级量价alpha公式 |
| Barra Global Equity Risk Model Handbook | 风险模型通用方法论（Size/Success/Value/VIM因子公式） |
| MSCI Foundations of Factor Investing | 因子投资基础白皮书 |

### 13.7 中国券商金工研报

| 团队 | 核心贡献 |
|------|---------|
| **华泰·林晓明** | 人工智能选股系列50+篇：GP因子挖掘→ML合成→可解释性 |
| **国信·多因子Alpha系列** | 因子定义和A股回测方法论 |
| **天风·量化选股系列** | 因子挖掘与组合优化 |
| **浙商·多因子框架梳理** | 多因子投资全流程框架 |
| **GitHub: QuantsPlaybook** | 券商金工研报复现代码合集 |

### 13.8 A股实盘通道参考（后期预研用）

| 通道 | 说明 | 链接 |
|------|------|------|
| 迅投/QMT | 券商股票量化交易终端，非开源，需券商开通权限 | https://www.thinktrader.net/ |
| MiniQMT | 轻量版QMT，开通流程 | https://www.miniqmt.com/pages/quick-open/ |
| QMT Python API | 篮子交易/算法单/账户查询 | https://www.miniqmt.com/qmtapi/QMT_Python_API_Doc.html |

> MVP阶段不要求开通。模拟盘稳定60天后，做券商调研表（资产门槛/软件费用/佣金/L2行情/API权限/Mac兼容性），再决定试接。
