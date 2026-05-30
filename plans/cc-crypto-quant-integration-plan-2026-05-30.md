# cc-crypto-quant-integration-plan-2026-05-30

**Status:** Research synthesis + design — 7 of 8 research agents complete; multi-asset architecture pattern agent still running, final directory tree / interface contracts pending its return. All cross-cutting strategy decisions are settled.

**Author note:** This plan is the deep-research output triggered by 王总 2026-05-30 morning request: "穷尽调研加密货币量化的前沿算法和开源工程 + 仔细思考融合在这个 project 里怎么能够解耦，不影响 A 股，又把代码写优雅". 8 parallel agents dispatched (4 strategy/data + 4 tech stack/architecture). Synthesis below.

---

## 0. 一句话决策

**起步策略 = funding arb（不是 stat arb），起步技术栈 = 库不用框架，起步架构 = `core/ + ashare/ + crypto/` 三 namespace，起步资金 = $5k paper → $50k Phase 1 → $200k Phase 3。** 不做 LP / restaking / RWA / memecoin / MEV / 散户 LOB / 自主 LLM 下单。

---

## 1. 起步策略优先级（按 ROI/风险/学习曲线）

| Phase | 策略 | 资金门槛 | 预期净 carry / Sharpe | 文献依据 | 与 A 股系统耦合 |
|-------|------|----------|----------------------|----------|------------------|
| 0 (1-2 周) | funding arb backtest + paper | $5k | 验证 | He 2024 SSRN 4301150 | 不动 A 股 |
| 1 (1-3 月) | funding arb 实盘 单交易所 | $50k | 7-10% / Sharpe 1.5-2.5 | BitMEX Q3 2025 92% 时间正 funding | 低 |
| 2 (3-6 月) | + 链上因子 overlay | $50k+ | +IC 0.02-0.04 | Chi 2025 arxiv 2411.06327 | 中（共享 factor_store） |
| 3 (6-12 月) | + cross-section LightGBM top-50 universe | $200k+ | Sharpe 1.5-2.0 | Liu-Tsyvinski-Wu 2022 JoF + Fieberg 2024 JFQA | 高（共享 core 模型层） |
| 4 (12 月+) | 双所 + alt + 期权 vol surface | $500k+ | +5%/年 | — | 高 |

**为什么 funding arb 先做（不是 stat arb）**：
- 容量与方向 alpha 无关，$50k-500k 在 BTC/ETH perp 完全无滑点
- 92% 时间正 funding（BitMEX Q3 2025 实测）
- Sharpe 2-3（He 2024 全样本含成本）vs stat arb 期望 1.5-2.0
- 不依赖任何预测能力，纯 carry trade
- ~350 LOC 即可上线，跟你已有 RiskGuard 7 层正交集成

---

## 2. 经济性核心数字

### funding arb (agent A)
- BTC perp funding 历史均值 **~11% 年化**（BitMEX 2025Q3 实测）
- Std (8h): ~0.02-0.03%
- 净 carry ≈ 11% - 3.6% 摩擦（fee + slippage）= **7.4% 年化保守均值**
- 牛市 22-36%（2024Q1 实测）；熊市 -1% 到 -8%
- **最差期间：2022-06-19 至 2022-08-03 连续 46 日负 funding**
- 容量：$10M+ 单交易所滑点 <5bp

### DeFi base layer (agent DeFi)
- Aave V3 USDC 3.45-5.2%
- Sky sUSDS 3.75-4.75%
- Morpho Blue 优质 vault 5-8%
- Curve+Convex 3-7%（需大量 veCRV 锁仓才划算）
- **结论**：base yield 3-5% 当 cash floor，funding arb 才是 alpha layer

### 推荐组合（$50k-500k）
- **60% base yield**（Aave/Sky/Morpho 三家分仓）
- **30% funding arb**（Binance + OKX 双所）
- **10% Hyperliquid HYPE 积分 farm**（轻仓，期权式收益）
- **bridge factor overlay**（接入现有 cron，作为 A 股 risk-on/off 信号）
- 预期组合 net APR **6-10% USD 计价**

---

## 3. 技术栈决策（库 > 框架）

### 不要引入的框架
- **freqtrade / Hummingbot / Jesse / LEAN** — 全部撕裂你现有 alpha → portfolio → RiskGuard 链路
- **NautilusTrader** — 留到 Phase 4 真要 HFT 时再评估
- **FinRL crypto** — 学术沙盒，绝不接钱（189 stars 已说明问题）
- **Backtrader / Zenbot** — 死项目或无 crypto 原生
- **AI Hedge Fund (43k stars)** — 自己明确说 "does not make trades"，demo 不是 alpha

### 起步技术栈（全部库形态）
| 层 | 选型 | 理由 |
|----|------|------|
| 数据接入 | **CCXT Pro** + top-5 交易所 SDK 冗余 | 业界标准，108 交易所统一 API |
| 实时执行 | **自建 executor**（~350 LOC） + 复用 RiskGuard | 不依赖任何框架 |
| 回测 | **vectorbt** + TA-Lib | 向量化，1 年分钟数据百策略秒级 |
| 因子计算 | **Polars + DuckDB**（复用 A 股 stack） | 一套代码两市跑 |
| 组合优化 | **pyportfolioopt** 或复用你的 `optimizer_v2.py` | 后者已支持 reduce_weight |
| 链上数据 | **Alchemy 免费 300M CU/月** + **Dune SQL 2500 credits/月** + **CryptoQuant 免费档** | Glassnode Advanced $49 留到 Phase 2 |
| 状态管理 | SQLite WAL + 复用 `job_status.json` 调度 | 与 A 股一致 |

### 数据吞吐参考
- top-50 现货 1m 全历史 5 年 ≈ 650M 行，parquet 压缩 **8-12 GB**
- DuckDB 全表扫秒级，Mac Studio 32GB 完全够用
- WS 实时 top-50 trade+orderbook L2 单进程 3-8 MB/s
- Polars 5min 滚窗特征 < 200ms

---

## 4. 因子库（已验证文献）

### Phase 2 链上 Top 10（按 IC × 成本性价比）
1. **USDT 净流入交易所**（Chi 2025 最强信号，1-6h IC 显著）
2. **MVRV-Z**（Glassnode 免费）
3. **SOPR**（Glassnode 免费）
4. **NUPL**（Glassnode 免费）
5. **BTC/ETH 交易所净流入**（CryptoQuant 免费）
6. **NVT / Active Addresses**（Coin Metrics 免费）
7. **Hash Ribbons + Puell Multiple**（Glassnode 免费）
8. **稳定币总供应 Δ**（DefiLlama 免费）
9. **CDD / Dormancy**（Glassnode 免费）
10. **DEX:CEX 量比 + L2 桥流量**（DefiLlama + Dune 免费）

### Phase 3 cross-section（文献 OOS 验证）
- **Liu-Tsyvinski-Wu CMOM** 周 L/S ≈ 3%，t > 3
- **Fieberg 2024 CTREND** 28 技术指标聚合，周 3.87%
- **Funding rate basis**（crypto 独有 alpha）
- 短期反转 (Shen 2020)
- **IVOL — 反号抄 A 股**（Zhang-Li 2020 实证正号，与 A 股相反）
- **MAX — 反号抄 A 股**（Li et al. 2021 lottery momentum）
- Amihud → Kyle-Obizhaeva 替代（crypto 上 Amihud 弱）

---

## 5. ML/RL/LLM 集成（保守路径）

**核心原则**：所有新方法的角色应当是 **生成新因子 / 当 meta-feature / 强化 RiskGuard**，**不替换 LGB 主干**。90% 的前沿论文 OOS 失效。

### SOLID（确定可用）
1. **RD-Agent (Microsoft)** — A 股实测 2x ARR with 70% fewer factors，$10/run。直接套你 LGB pipeline。GitHub microsoft/RD-Agent。
2. **Glassnode on-chain factors** 当 KLEN 类因子做

### WORTH TRYING（Phase 2-3 试）
3. **PPO + 离散 action（持仓/减半/平仓）** 替换 RiskGuard L7 固定阈值。reward = log-return - λ·max(0, drawdown-threshold)。参考 FinRL_Crypto anti-overfit 框架（K-fold + PBO < 0.5）
4. **Kronos-Small LoRA fine-tune** on BTC/ETH 4h K-line → next-bar embedding → 作第 5 个 shadow overlay（与 KLEN/ROC5/vol_compression 并列）
5. **GraphSAGE whale wallet clustering** on Ethereum → "smart-money netflow" 因子。**目标不是预测价格，是产生新截面因子**
6. **CryptoTrade-style reflective LLM 仅在 narrative event 触发**（ETF approval / hack / FOMC）作为 risk-off 信号给 RiskGuard

### HYPE（不要做）
- TimeGPT/Chronos 零样本预测价格（打不过 LGB）
- LOB transformer 散户实盘（延迟死亡：你 200ms+ vs alpha 50ms 消失）
- ViT/CNN K-line 图像（4 层 CNN 0.892 AUC > ViT，信息熵损失）
- Twitter sentiment alpha（2025 后被套利）
- Multi-agent debate（TradingAgents demo 价值）
- 自主 LLM 下单（reward hacking）
- GNN 直接预测 BTC/ETH price（论文 OOS 不显著）

---

## 6. Hard Gotchas（避免反向陷阱）

1. **IVOL 在 crypto 是 POSITIVE sign**（vs A 股 negative）— Zhang-Li 2020 实证。直接抄 A 股代码会反向亏钱
2. **MAX 在 crypto 是 POSITIVE sign**（lottery momentum）— Li et al. 2021
3. **传统 Value（B/P）在 crypto 死掉了** — 没现金流
4. **BTC-ETH pairs 2022 Merge 后结构性 broken** — 47-50 天内 ρ 从 0.95 → 0.75，ETH/BTC 单边漂移 47%。不要 pairs cointegration on majors
5. **Survivorship bias 极大**（Ammann 2023：等权偏误 62.19%）— 必须保留 dead coins 最后一根 K 线
6. **Alpha decay 比 A 股快 5-10 倍** — 因子 IC 评估窗口从 60-120D 降到 7-30D；walk-forward 重训从月级降到周级
7. **Funding rate basis 是 crypto 独有 alpha 源**（equity 无对应），必须放进 factor zoo
8. **Uniswap v3 LP 51% 净亏损**（Bancor + IntoTheBlock 研究）— 案例赚 $199.3M fees 但扛 $260.1M IL 净亏 $60.8M。本质 short vol 无定价 vega。**不要做**

---

## 7. 不做清单（明确禁区）

- HFT / market making（资本 + 技术护城河输头部）
- MEV / sandwich attack（90% 利润付 builder bribe + 法律灰）
- 新币上线快投（概率游戏 + 拉黑风险）
- 跨所 HFT 套利（HFT 早吃光价差）
- BTC-ETH pairs cointegration（Merge 后死了）
- 传统 Value/Quality 因子迁移
- Regime switch overlay（A 股已验证 fail，crypto 同理）
- Uniswap v3 LP（51% 亏损 + 无 hedge stack）
- EigenLayer/Symbiotic restaking（Kelp 2026-04 被盗 $3 亿 + slashing cascade）
- Maple/Centrifuge RWA（信用风险定价不擅长）
- Solana memecoin（retail bot 战场，结构性劣势）
- veCRV 锁仓（4 年期限错配）
- Aave liquidation / DEX arb（PGA 战 + $500-2000/月基建）
- Telegram pump signal 系统化（法律灰色）
- LOB 模型实盘（延迟死亡）
- 自主 LLM 下单（reward hacking）

---

## 8. 当前代码 asset-implicit 假设盘点（先行）

### 🟢 Tier 1 — 已是 asset-agnostic（rename 或加 AssetClass 列即可）
- `factor_store` 存储层（parquet + DuckDB）
- LGB/XGB 训练框架
- LLM V2 extractor + EventStore + L0/L1/L2 pipeline
- shadow overlay 引擎
- `run_with_status` + `mark_complete` + `enforce_deps` job wrapper

### 🟡 Tier 2 — 参数化干净（加 protocol 即可）
- `backtest/optimizer_v2.py`（已接受 constraints dict）
- `factors/candidate_sanitizer.py`（规则已 parametric）

### 🔴 Tier 3 — 深度 A 股 implicit（需要明确抽象）
- `scheduler/jobs.py` line 365/477/506/626/902/1262: `code[:2] in ("SH","SZ","BJ")` 共 8 处
- `paper/oms.py` line 82-84: `commission_rate=0.0003 / stamp_tax_rate=0.0005 / slippage_rate=0.001` 硬编码默认
- `paper/oms.py` line 9/15/27/269-284: T+1 reconcile 流程
- `backtest/risk_guard.py`: ST / 涨跌停 / 一字板 层语义不重合
- `models/feature_merger.py`: ST_CLIENT 字段 hardcoded
- `config/watchlist.py`: akshare 调用

### 🔴🔴 永远 A 股专属（不该共享）
- ST list 逻辑
- 涨跌停 / 一字板 逻辑
- akshare / baostock / tushare connectors

---

## 9. 架构 — 10 条原则（实读 Lean / Nautilus / QuantLib 源码总结）

1. **AssetClass 与 InstrumentClass 必须正交**。Lean 一维 12 项 `SecurityType` 在加 `CryptoFuture/CryptoPerpetual` 时已经膨胀。用 Nautilus 双枚举：`(Cryptocurrency, Spot)` ≠ `(Cryptocurrency, Perpetual)`，表达力强 3-4 倍且不破坏 switch
2. **Calendar 是注入物，绝不假设 `trading_day` 存在**。所有"昨天"语义改为 `calendar.previous_session(ts)`。Crypto 的 calendar 是 `Always24x7Calendar`，**实现同一接口**但 `previous_session(ts) == ts - 1day` 恒等 — 下游代码零修改
3. **Settlement 是 Model，不是 OMS 行为**。当前 `paper/oms.py:743` 的 `reconcile()` 把 T+1 写死在主流程里；应抽 `ISettlementModel` 接口，A 股注入 `DelayedFillSettlementModel(days=1, fill_at="open")`，Crypto 注入 `ImmediateSettlementModel()`，**OMS 主循环零行改动**
4. **Cost 拆三层：CommissionModel + TaxModel + ImpactModel**。当前 `cost_model.py:CostModel` 把 commission/stamp/slippage/impact 揉一个 dataclass — Crypto 没印花税不是 0，是**不应该有这个字段**
5. **InstrumentId 必须包含 venue**。`SH600519` 在 A 股唯一，但 `BTC/USDT` 在 Binance vs Coinbase 价格/fee/流动性完全不同。用 `Symbol(asset="BTC/USDT", venue="BINANCE")`
6. **Universe filter 是 Asset-aware Strategy 模式，不是 if-else**。`AShareTradableFilter` 与 `CryptoLiquidityFilter` 都实现同一 `UniverseFilter` Protocol
7. **PricePoint 必须携带 venue + asset_class 元数据**。否则 CostModel 无法多态选税率
8. **Lot size / tick size 是 Instrument 属性，不是 OMS 常数**。`paper/oms.py:401` 的 `int(per_stock_value/buy_price/100)*100` 写死了 100 股一手 — BTC Binance 最小 0.00001 不成立
9. **配置分层：core defaults → asset-class profile → instrument override**。Lean `market-hours-database.json` 三键查 `(market, symbol, type)` 是黄金范式
10. **Account currency 必须显式**。当前代码 PnL 用人民币隐含计价；引入 BTC/USDT 后必须每个 PnL 计算点显式 base_ccy 换算

## 9.1 三 namespace 完整目录结构（file-level granularity）

```
stockPrediction/
├── core/                              # 零 asset 假设，纯抽象
│   ├── asset.py                       # AssetClass enum, InstrumentClass enum, Symbol, Instrument Protocol
│   ├── venue.py                       # Venue dataclass(name, asset_class, fee_schedule_id)
│   ├── calendar.py                    # TimeAxis Protocol; SessionCalendar, Always24x7Calendar
│   ├── price.py                       # PricePoint(ts, symbol, value, ccy, venue)
│   ├── order.py                       # Order, OrderType, TimeInForce, Side enums
│   ├── settlement.py                  # ISettlementModel Protocol + Immediate / DelayedFill
│   ├── cost.py                        # CommissionModel, TaxModel, ImpactModel, CostBundle
│   ├── universe.py                    # UniverseFilter Protocol
│   ├── data_backend.py                # PriceProvider, QuoteProvider Protocols
│   ├── portfolio.py                   # Position, Portfolio (asset-agnostic)
│   ├── risk.py                        # RiskConstraints, IRiskCheck Protocol
│   ├── oms.py                         # OMS framework (策略模式装载 settlement/cost/risk)
│   └── ledger.py                      # PnL, CashBook (multi-ccy)
│
├── ashare/                            # 当前代码 MOVED here
│   ├── calendar.py                    # CsiCalendar (XSHG holiday)
│   ├── codes.py                       # SH/SZ/BJ 规约, to_akshare_code, to_qlib_code
│   ├── instruments/csi300.py          # WATCHLIST_STOCK 移这
│   ├── data/
│   │   ├── collectors/                # market.py, sentiment.py, capital_flow.py
│   │   ├── qlib_backend.py            # 实现 core.PriceProvider
│   │   └── tushare_backend.py
│   ├── cost/
│   │   ├── stamp_tax.py               # 现 cost_model.stamp_tax_rate
│   │   ├── retail_commission.py       # min 5 元
│   │   └── ashare_cost.py             # 组装 CostBundle
│   ├── settlement.py                  # AShareT1Settlement(implements core.ISettlementModel)
│   ├── universe.py                    # AShareTradableFilter (ST/IPO/BSE/涨跌停)
│   ├── risk/
│   │   ├── price_band.py              # 涨跌停 check
│   │   ├── st_check.py
│   │   └── crash_overlay.py
│   ├── factors/                       # 现 factors/* 移这
│   ├── models/                        # 现 models/* 移这
│   ├── backtest/                      # 现 backtest/* 移这，改注入 core 接口
│   └── paper/oms.py                   # 现 paper/oms.py 轻量化为 core.oms 子类
│
├── crypto/                            # 全绿地
│   ├── calendar.py                    # Always24x7Calendar 子类
│   ├── codes.py                       # BASE/QUOTE 解析, venue normalization
│   ├── venues/binance.py + coinbase.py + hyperliquid.py
│   ├── instruments/spot_pairs.py + perpetuals.py
│   ├── data/
│   │   ├── ccxt_backend.py            # 实现 core.PriceProvider
│   │   ├── ws_stream.py
│   │   ├── glassnode_backend.py       # Phase 2
│   │   └── alchemy_backend.py         # Phase 3
│   ├── cost/
│   │   ├── maker_taker.py             # maker/taker fee
│   │   └── crypto_cost.py
│   ├── settlement.py                  # ImmediateSettlement
│   ├── universe.py                    # CryptoLiquidityFilter (24h vol > X USDT)
│   ├── risk/
│   │   ├── leverage.py
│   │   ├── funding_rate.py
│   │   └── depeg.py                   # stablecoin depeg
│   ├── factors/                       # 链上 + cross-section
│   ├── models/
│   ├── backtest/
│   └── paper_oms.py
│
├── scheduler/
│   ├── jobs.py                        # 编排，按 asset_class 分支
│   └── job_registry.py                # 按 AssetClass 注册 pipeline
│
└── tests/
    ├── core/                          # 接口 + 协议测试
    ├── ashare/                        # 回归 (锁定数值)
    └── crypto/                        # 独立测试
```

## 9.2 接口契约（Python Protocol，完整签名）

```python
# core/asset.py
class AssetClass(IntEnum):
    EQUITY_CN = 1; EQUITY_US = 2
    CRYPTO_SPOT = 10; CRYPTO_PERP = 11
    FX = 20; COMMODITY = 30; INDEX = 40

class InstrumentClass(IntEnum):  # 与 AssetClass 正交（Nautilus 模式）
    SPOT = 1; SWAP = 2; FUTURE = 3; CFD = 4
    BOND = 5; OPTION = 6; ...

@runtime_checkable
class Instrument(Protocol):
    symbol: Symbol
    asset_class: AssetClass
    instrument_class: InstrumentClass
    venue: Venue
    price_tick: Decimal      # min price increment
    lot_size: Decimal        # min qty increment (A股 100, BTC 1e-5)
    multiplier: Decimal      # contract multiplier (futures)
    base_ccy: str            # crypto: "BTC"
    quote_ccy: str           # crypto: "USDT"; A股: "CNY"
    def round_qty(self, qty: Decimal) -> Decimal: ...
    def round_price(self, px: Decimal) -> Decimal: ...

# core/calendar.py — 关键: Always24x7Calendar 实现同一接口
class TimeAxis(Protocol):
    def is_session(self, ts: datetime) -> bool: ...
    def next_session(self, ts: datetime) -> datetime: ...
    def previous_session(self, ts: datetime) -> datetime: ...
    def sessions_between(self, a, b) -> list[datetime]: ...

# core/universe.py
class UniverseFilter(Protocol):
    asset_class: AssetClass
    def is_tradable(self, instrument: Instrument, ts: datetime, ctx: dict) -> tuple[bool, str]: ...
    def filter(self, candidates, ts) -> list[Instrument]: ...

# core/cost.py — 三层正交
class CommissionModel(Protocol):
    def commission(self, order, fill_px, fill_qty) -> Decimal: ...
class TaxModel(Protocol):  # A股 StampTax 实现; Crypto 注入 NoTax
    def tax(self, order, fill_px, fill_qty) -> Decimal: ...
class ImpactModel(Protocol):
    def impact(self, order, fill_px, fill_qty, adv=None, vol=None) -> Decimal: ...
@dataclass
class CostBundle:
    commission: CommissionModel
    tax: TaxModel
    impact: ImpactModel
    def total_cost(self, order, fill_px, fill_qty, ctx) -> Decimal: ...

# core/settlement.py — 把 T+1 抽出 OMS
class ISettlementModel(Protocol):
    def next_fill_time(self, order, signal_ts, cal: TimeAxis) -> datetime: ...
    def can_sell(self, position, ts) -> bool: ...

# core/oms.py — 全 DI
class OrderRouter(Protocol):
    instrument_resolver: Callable[[str], Instrument]
    price_provider: PriceProvider
    cost: CostBundle
    settlement: ISettlementModel
    universe: UniverseFilter
    calendar: TimeAxis
    risk_checks: list[IRiskCheck]
    def submit(self, order: Order) -> Fill | Pending: ...
```

## 9.3 5 个关键设计抉择 + 理由

**抉择 1：Nautilus 双枚举 `(AssetClass × InstrumentClass)`，而非 Lean 的一维 12 项 SecurityType**
- 理由：未来必加 `Cryptocurrency × Perpetual` 与 `Cryptocurrency × Option`。一维 enum 在第 N 次扩张时会变 `CryptoFuturePerpetualLongDated`。Lean 已膨胀到 12 项。正交二维表达力强 3-4 倍且不破坏 switch

**抉择 2：Settlement 用 Strategy Pattern 注入，绝不放 OMS 主循环里 `if asset_class == ...`**
- 理由：现 `paper/oms.py:743` 的 `reconcile()` 长达 200 行就是因为 T+1 嵌进了主流程。Crypto 即时成交时这条路径完全跳过 = 注入 `ImmediateSettlement()`，其 `next_fill_time` 直接返回 `signal_ts`，OMS 一行代码不改

**抉择 3：Calendar 接口实现 `Always24x7Calendar` 单独存在，不让 crypto 路径绕过 calendar**
- 理由：Zipline issue #2798 卡 4 年的根本原因 — 他们曾尝试"crypto 不走 calendar"，结果 `auto_close_date`/`first_traded` 等所有依赖 calendar 的下游代码都炸。正确解法是 crypto **也走 calendar 接口**，只是实例的 `is_session()` 恒返 `True`。下游代码零修改

**抉择 4：`ashare/` 是物理目录、有自己的 namespace，而非 `core/strategies/ashare/`**
- 理由：A 股代码 ~8000 行，远超 core 抽象层。当 first-class 子产品（像 Lean 的 `QuantConnect.Brokerages.GDAX`）而非 core 的子目录，意味着 **crypto 可以完全不 import ashare** — 污染隔离的物理保证

**抉择 5：重构按"先建抽象 → facade 桥接 → 物理迁移"三段式，中间夹回归 snapshot 测试**
- 理由：你 RiskGuard 7 层 + 4 个 shadow + 35 个 cron job 在跑，任何破坏性变更都丢实盘数据。step 11-13 的"等价 facade"+ trades.jsonl 逐条 diff 是质量门；step 22 一次性 `git mv` 必须在前 21 步全绿且至少 1 天 cron 跑通后才能扣扳机。**不要倒序，不要省略 facade**

## 9.4 CryptoSanitizer 14 条规则草案（替换 A 股 CandidateSanitizer）
1. `listed_days < 30` → 排除（替代 IPO<60）
2. `daily_volume_usd < 5M` → 排除（替代低成交）
3. `is_stablecoin and abs(price-1) > 0.005` → depeg 警报（替代 ST）
4. `funding_rate > 0.1%/8h or < -0.1%/8h` → 极端拥挤，冷却
5. `exchange_withdrawal_halted` → 排除
6. `unlock_event_in_next_7d and unlock_pct > 5%` → 排除（替代解禁/减持）
7. `is_in_known_scam_list` → 排除（替代 ST）
8. `wick_ratio_15m > 3` → 冷却（替代一字板）
9. `cex_inflow_24h > avg×3` → 抛压预警
10. `orderbook_depth_1pct < 100k USD` → 排除
11. `contract_audit_status != verified` → 排除
12. `cooldown_after_circuit_break(symbol, 4h)` → 替代 A 股个股冷却
13. `cross_exchange_premium > 2%` → 套利异常，冷却
14. `chain_congestion (gas>100gwei) and trade_size>X` → 滑点风险冷却

---

## 10. 4 周里程碑

| 周 | 交付物 | 涉及研究 |
|---|--------|---------|
| **Week 1** | core/ 抽离 + ashare/ 迁入回归测试通过 + crypto/data/ 拉 2 年 BTC/ETH OHLCV + funding rate history 入 parquet + funding arb backtest pipeline（vectorbt + CCXT） | 架构 + funding arb |
| **Week 2** | crypto universe（top-50 by 90d MCap + volume + listed_days）+ CryptoSanitizer 14 条规则 + Top 10 链上因子接入 factor_store（USDT 净流入 / MVRV / SOPR / NUPL / NVT / Puell / Hash Ribbons / 稳定币供应 / CDD / DEX:CEX） | 架构 + 链上 alpha |
| **Week 3** | funding arb $5k paper 7 天 + 复现 Liu-Tsyvinski-Wu CMOM + Fieberg CTREND（验证基础 alpha） + LightGBM 4h bar 训练框架（含 IVOL/MAX **反号**）+ RD-Agent 接入 A 股因子库验证 | funding arb + stat arb + ML 集成 |
| **Week 4** | RiskGuard 三新层（funding 极值 / 清算预警 / depeg） + UTC 调度（00/08/16 对齐 funding 结算）+ funding arb 实盘 $5k testnet/小额 + cross-section 模型 walk-forward backtest 报告 + DeFi base yield 自动化（Aave V3 USDC + Sky sUSDS）| 全维度 |

---

## 11. 月度成本结构

| 项 | 月成本 | 必要性 |
|----|--------|--------|
| CryptoQuant + DefiLlama + Dune 免费档 | $0 | 必备 |
| Alchemy 免费档 300M CU/月 | $0 | 必备（链上节点） |
| Glassnode Advanced | $49 | Phase 2 起（Phase 0/1 免费档够） |
| 交易所手续费 | 0.3%/round-trip | 必备 |
| RD-Agent LLM API（factor mining） | ~$10/run | 周/月跑 |
| Mac Studio 自建节点 | $400 一次性（2TB NVMe） | Phase 3 可选 |
| **Phase 0/1 起步月成本** | **$0** | |
| **Phase 2 起月成本** | **~$49 + LLM API** | |

---

## 12. 关键决策点（待用户拍）

1. 何时开 `core/` 抽离工作（不破坏 A 股 production）
2. 何时充值 Glassnode Advanced（$49/m，Phase 2 启动条件）
3. $5k Phase 0 paper 用哪家交易所开户（Binance / OKX / Bybit）
4. 是否周末做 funding arb backtest（A 股周末无 cron，是 crypto 调研黄金时段）
5. DeFi yield base layer 用哪家钱包托管（Fireblocks SaaS vs Ledger 硬件 + 多签 vs Coinbase Custody）

---

## 13. 论文 / GitHub anchor URL（实施时查阅）

### funding arb
- He, Manela, Ross & von Wachter "Fundamentals of Perpetual Futures" SSRN 4301150: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4301150
- BitMEX Funding Rates Structure Q3 2025: https://www.bitmex.com/blog/2025q3-derivatives-report
- ScienceDirect Risk and Return Funding Rate Arbitrage 2025: https://www.sciencedirect.com/science/article/pii/S2096720925000818

### 链上 alpha
- Chi, Chu & Hao 2025 arxiv 2411.06327: https://arxiv.org/abs/2411.06327
- Mahmudov & Puell MVRV 2018: https://medium.com/@adamtaché/an-analysis-of-the-mvrv-ratio-7c6cf7e6c92e
- checkonchain: https://charts.checkonchain.com/

### cross-section
- Liu & Tsyvinski 2021 RFS: https://academic.oup.com/rfs/article/34/6/2689/5912024
- Liu, Tsyvinski & Wu 2022 JoF: https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13119
- Fieberg 2024 JFQA CTREND: https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/trend-factor-for-the-cross-section-of-cryptocurrency-returns/4C1509ACBA33D5DCAF0AC24379148178
- Ammann 2023 Survivorship: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4287573
- Bianchi & Babiak IPCA SSRN 3935934: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3935934
- Zhang-Li 2020 IVOL POSITIVE sign: https://www.sciencedirect.com/science/article/abs/pii/S0275531920301926
- Li et al. 2021 MAX POSITIVE sign: https://jfin-swufe.springeropen.com/articles/10.1186/s40854-021-00291-9

### ML/RL frontier
- RD-Agent (Microsoft): https://github.com/microsoft/RD-Agent
- Kronos (AAAI 2026) arxiv 2508.02739: https://arxiv.org/abs/2508.02739
- CryptoTrade EMNLP 2024 arxiv 2407.09546: https://arxiv.org/abs/2407.09546
- FinRL_Crypto arxiv 2209.05559: https://arxiv.org/pdf/2209.05559
- TLOB arxiv 2502.15757: https://arxiv.org/abs/2502.15757
- ChronoWave-GNN Nature SciRep: https://www.nature.com/articles/s41598-025-23901-3
- TSFM Benchmark Leakage arxiv 2510.13654: https://arxiv.org/html/2510.13654v2

### DeFi
- DefiLlama Yields & Bridges: https://defillama.com/yields
- Aave V3: https://aave.com/
- Sky sUSDS: https://sky.money/
- Flashbots Protect: https://docs.flashbots.net/flashbots-protect/overview
- Uniswap v3 LP loss analysis HAL: https://hal.science/hal-04214315v3/document

### 开源框架
- CCXT: https://github.com/ccxt/ccxt
- vectorbt: https://github.com/polakowo/vectorbt
- NautilusTrader (Phase 4 备选): https://github.com/nautechsystems/nautilus_trader
- TA-Lib Python: https://github.com/TA-Lib/ta-lib-python

---

## 14. 25 步重构序列（按风险递增，含 diff size + A 股回归测试策略）

| # | 步骤 | diff | A 股风险 | 测试策略 | 备注 |
|---|---|---|---|---|---|
| 1 | 建 `core/` 空目录 + `core/__init__.py` | XS | 无 | n/a | 准备工作 |
| 2 | 新建 `core/asset.py`: `AssetClass + InstrumentClass` enum + `Symbol` dataclass | S | 无 | unit 测试 enum 值 | 抄 Nautilus |
| 3 | 新建 `core/calendar.py`: `TimeAxis` Protocol + `Always24x7Calendar` + `AShareCalendar` (wrap Qlib) | M | 无 | 对比 Qlib `D.calendar()` 输出 | 不动旧代码 |
| 4 | 新建 `core/settlement.py`: `ISettlementModel` + `Immediate` + `DelayedFill` | S | 无 | unit | 仅定义，无 wire |
| 5 | 新建 `core/cost.py`: `CommissionModel/TaxModel/ImpactModel/CostBundle` | S | 无 | unit | 仅定义 |
| 6 | 新建 `core/universe.py`: `UniverseFilter` Protocol | XS | 无 | n/a | |
| 7 | 新建 `core/data_backend.py`: `PriceProvider/QuoteProvider` Protocol | XS | 无 | n/a | |
| 8 | 新建 `ashare/` 包 + `__init__.py`(re-export 旧路径，先 alias 不移动) | S | 无 (alias only) | `import ashare.backtest.cost_model` 通 | 兼容层 |
| 9 | 把 `factors/quant.py` 里 `code[:2] in ("SH","SZ","BJ")` 抽到 `ashare/codes.py:AShareCode.is_ashare()` | S | 低 | snapshot 测试 quant.py 输出 | 纯函数提取 |
| 10 | 把 `config/watchlist.py:load_csi300` 移到 `ashare/instruments/csi300.py`，旧路径 re-export | S | 低 | import 路径不变 | |
| 11 | 在 `core/cost.py` 实现一个**等价于现 CostModel** 的 `CostBundle`，A 股侧实例化时填 `commission=AShareRetailCommission(), tax=StampTax(0.0005), impact=SqrtAdvImpact(coeff=0.1)` | M | 中 | `backtest/cost_model.py:summary()` 数值逐字段对齐 | 关键 |
| 12 | `backtest/cost_model.py:CostModel` 改为 **deprecated facade** 调 `CostBundle` | M | 中 | 现有所有调用点保留接口签名；回归跑 `tests/test_backtest_*` 全过 | |
| 13 | 把 `paper/oms.py:execute_orders` 内联的 `commission + stamp_tax + slippage` 替换为 `cost_bundle.total_buy(order)/total_sell(order)` | M | **高** (paper 跑 live) | 把 1 个历史回放日 (e.g. 2026-05-23) 的 `trades.jsonl` 在新旧两套代码上跑，逐条 diff，允许 0.01 元 round-trip 误差 | 仅替换计算 |
| 14 | 新增 `core/oms.py:BaseOMS`，迁入现 `paper/oms.py` 中**非 A 股 specific** 部分 (cash 簿、Position dataclass、daily PnL 累计、状态持久化) | L | **高** | 现 `PaperOMS` 继承 `BaseOMS`，所有 public API 不变；跑 7 天回放对齐 PnL 至 0.01 元 | **最大且最危险**，必须配对 step 15 |
| 15 | 把 `_load_real_prices` 改为通过 `self.price_provider` (DI)，A 股 default 注入 `QlibCnPriceProvider` | M | 高 | 同 14 共测 | step 14 的姐妹步 |
| 16 | `paper/oms.py:401,846` 的 `int(.../100)*100` 改 `self.instrument.round_qty(qty)`，A 股 instrument 返回 100 倍数 | S | 中 | snapshot trades:shares 字段不变 | 为 crypto 0.00001 lot 铺路 |
| 17 | 抽 `core/settlement.py` 在 `BaseOMS.execute_orders` 中调用：`fill_time = settlement.next_fill(order, now)`；A 股注入 `DelayedFillSettlement(days=1, at="open")` | L | **高** | T+1 reconcile 逻辑等价；7 天回放 PnL 不变 | 把 `reconcile()` 改为 settlement 驱动 |
| 18 | `models/universe_filter.py` 实现 `core.UniverseFilter` Protocol；`UniverseFilter` 改名 `AShareTradableFilter` 移入 `ashare/universe.py`，旧路径 re-export | M | 中 | inference 路径 `_load_and_filter_predictions` snapshot 不变 | |
| 19 | `backtest/risk_guard.py` 的 `_check_st_stocks` `_check_limit_down` 抽到 `ashare/risk/`，`RiskGuard` 改用插件注册：`risk_guard.register(PriceBandCheck())` | L | 中 | 现 OMS 调用点不动；`force_sell`/`cannot_buy` 输出逐 stock diff 为空 | |
| 20 | `scheduler/jobs.py` 顶部 import 改 DI：`pipeline_registry.get(AssetClass.EQUITY_CN)` 返回 `AShareDailyPipeline` | M | 中 | cron 任务列表不变，但每个 job 改读 registry | |
| 21 | `config/settings.py:TAKE_PROFIT_PCT/STOP_LOSS_PCT/QLIB_PROVIDER_URI` 移到 `ashare/config.py`，`core/settings.py` 只留 `BASE_CCY/DATA_ROOT/LOG_LEVEL` | M | 中 | 所有引用一并改 import | grep+sed 机械迁移 |
| 22 | 物理移动文件: `factors/*` → `ashare/factors/*`，`models/*` → `ashare/models/*`，`backtest/*` → `ashare/backtest/*` (`git mv`，旧路径留 re-export shim) | L | 中 | `python -m pytest tests/` 全过；cron 至少跑通 1 天 | 一次性、可回滚 |
| 23 | 新建 `crypto/` 骨架 + `crypto/calendar.py:Always24x7Calendar` + `crypto/cost/maker_taker.py` + `crypto/codes.py` + `crypto/data/ccxt_backend.py` | M | **无** | 独立 unit 测试，不接 cron | A 股代码不动 |
| 24 | 新建 `crypto/paper_oms.py = BaseOMS(settlement=ImmediateSettlement, cost=CryptoCost, instrument=CryptoSpot(lot=1e-5))` + 一个 paper backtest job | M | 无 | 独立 7 天 BTC/USDT 回放 | 首个完整端到端 crypto 路径 |
| 25 | 删除 `core/__init__.py` 中旧路径的兼容 shim；固化新结构 | S | 低 (此时 cron 已稳定) | 完整集成测试套件 | 收尾 |

**配对依赖**：step 14↔15 必同 PR；step 17↔step 24 测同一接口；step 22 必须在 step 9-21 全绿后才执行。

## 14.1 当前代码 20 处 asset-implicit 假设（file:line）

| # | 文件:行 | 假设 | 解耦方案 |
|---|---|---|---|
| 1 | `backtest/cost_model.py:44` | `stamp_tax_rate=0.0005` 字段存在 | 拆 `TaxModel`，A 股 `StampTaxModel`，Crypto `NoTaxModel` |
| 2 | `backtest/cost_model.py:39-40` | `min_commission=5.0` 元 | Commission 改 model 注入 |
| 3 | `paper/oms.py:401,846` | `int(.../100)*100` 100 股板块 | `instrument.round_quantity(qty)`；A 股 lot=100，Crypto lot=1e-5 |
| 4 | `paper/oms.py:743,981` | T+1 reconcile 写死在 `run_daily`/`reconcile` | OMS 调 `settlement_model.next_fill_time(order)` |
| 5 | `paper/oms.py:288,317` | `D.features` + `Ref($open,-1)` Qlib API | `PriceProvider` 接口；A 股 `QlibCnDataProvider`，Crypto `CcxtProvider` |
| 6 | `models/universe_filter.py:55,95` | `exclude_bse` + `startswith("bj")` | `AShareCodePolicy` vs `CryptoCodePolicy` |
| 7 | `models/universe_filter.py:43` | `min_listing_days=60` 默认 | `AShareUniverseConfig.min_listing_days=60`, `CryptoUniverseConfig.min_listing_days=14` |
| 8 | `models/universe_filter.py:127` | ST 检查走 ST_CLIENT (TuShare A 股) | ST 移入 `AShareTradableFilter` 独享 |
| 9 | `backtest/risk_guard.py:110,143` | `_check_st_stocks` + `_check_limit_down` | 涨跌停 = A 股 `PriceBandPolicy`；Crypto 无 |
| 10 | `config/settings.py:21` | `QLIB_PROVIDER_URI` 默认 `cn_data` | 在 `ashare.config` 下，`core.config` 不知 Qlib |
| 11 | `config/settings.py:59-60` | `TAKE_PROFIT_PCT=8.0` `STOP_LOSS_PCT=5.0` 全局 | 移入 `AShareRiskPolicy`；Crypto 波动率 5×，需独立默认 |
| 12 | `config/watchlist.py:55` | `load_csi300()` import 时执行 (副作用) | 改 `WatchlistProvider.get(asset_class)` lazy |
| 13 | `config/watchlist.py:32-35` | `code.startswith("6")` → `SH`，else `SZ` | `AShareCodeNormalizer` 移出 core |
| 14 | `factors/quant.py:19` | `instruments="csi300"` Qlib 字符串 | `AShareInstrumentSet.csi300` |
| 15 | `scheduler/jobs.py:1-50` | 直接 import `MarketCollector/CryptoCollector/GoldCollector` | DI 容器，jobs 只依赖 `AssetPipeline` 协议 |
| 16 | `paper/oms.py:42-43` | `PAPER_DIR = data/storage/paper` 单一目录 | 路径包含 asset_class：`paper/ashare/...`、`paper/crypto/...` |
| 17 | `paper/oms.py:402-410` | `commission + stamp_tax + slippage` 公式串联 | OMS 仅调 `cost_model.total_cost(order, fill)`，不知费率拆解 |
| 18 | `backtest/risk_guard.py:148` | `st_stock_list.json` 单文件 | per-asset state dir |
| 19 | `paper/oms.py:294` | `init_qlib(.../cn_data)` 全局副作用 | 移入 `AShareDataBackend.__enter__` |
| 20 | `scheduler/jobs.py:25-28` | `MARKET_STOCK/CRYPTO/GOLD` 字符串常量当类型 | 升级为 `AssetClass` enum |

---

## 15. 与现有 production 系统的硬约束

1. **A 股 production cron 不能被中断**：周一 18:25 的 enforce_deps 闸门 + paper OMS reconcile + LLM pipeline 必须照常跑。任何 core/ 抽离必须先迁移 + 回归测试 + 灰度，再切换
2. **不引入 multi-asset 框架的副作用**：选 NautilusTrader 意味着抛弃现有调度/状态/RiskGuard。**明确不做**
3. **数据隔离**：A 股 storage 与 crypto storage 分目录，DuckDB schema 加 asset_class 字段而不是混库
4. **配置隔离**：`configs/ashare.yaml` / `configs/crypto.yaml` 独立，禁止 cross-import
5. **每个新 crypto 因子必须先在 A 股 stack 复现验证可用**（RD-Agent 跑 A 股 → 验证 → 再 port 到 crypto）

---

## 16. Memory 链接

- [[crypto-quant-research-20260530]] — 第一轮 4 agent 研究记录
- [[project_phases]] — 当前 A 股 production 状态（4J~4X+CX30, 35 cron jobs, RiskGuard 7 层）
- [[llm-pipeline-architecture]] — L0/L1/L2 spec 模式可直接复用到 crypto

---

**Last updated:** 2026-05-30
**Status:** 7/8 agents complete; multi-asset architecture pattern agent pending — final architecture section (§9 接口契约 + §14 待补充) will be updated upon return.
