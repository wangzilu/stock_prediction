# Portfolio Execution 研究：从 alpha 到可交易的组合策略

**日期：** 2026-05-17
**作者：** CC
**状态：** 调研中

---

## 一、当前问题

### Track A（模型选股能力）✅ 稳定通过

| 指标 | 15/24 splits 中间结果 |
|------|:---:|
| avg RankIC | +0.058（gate ≥0.04 ✅） |
| RankIC>0 | 87%（gate ≥65% ✅） |
| avg Spread | +2.79%（gate ≥1.2% ✅） |
| Spread>0 | 87%（gate ≥65% ✅） |

**结论：模型有真实 alpha，且在牛市/熊市/高波/低波都有效。**

### Track B'（交易策略）❌ 极不稳定

| Split | 测试区间 | daily | weekly+bonus | biweekly+dropout |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 03/16~05/15 | +8.6% | -43.8% | -19.5% |
| 2 | 01/09~03/16 | -40.5% | -65.6% | -60.0% |
| 3 | 11/12~01/09 | -15.2% | +57.3% | +20.8% |

**结论：同一个模型，不同的组合执行策略，在不同市场环境下结果天差地别。**

### 核心矛盾

```
模型说"这 20 只票未来 5 天会涨" → 模型大部分时间是对的（RankIC +0.058）

但是：
- daily 换仓：知道谁涨，但换太频繁，成本吃光
- biweekly 不动：牛市拿住赚大钱，熊市拿住亏死
- 没有一个固定参数能适应所有市场环境
```

---

## 二、问题本质

这不是"模型不好"的问题，而是**组合执行层的问题**。学术界和工业界称之为：

1. **Portfolio Turnover-Performance Tradeoff** — 换手越低成本越少，但信号越陈旧
2. **Dynamic Portfolio Rebalancing** — 什么时候该换、换多少
3. **Transaction Cost-Aware Portfolio Optimization** — 把成本纳入优化目标
4. **Regime-Adaptive Execution** — 根据市场状态动态调策略参数

---

## 三、待调研方向

### 3.1 Qlib 内置模块

需要调研 Qlib 是否已有：
- TopkDropoutStrategy 的参数如何动态调
- NestedExecutor / TWAP / VWAP 执行器
- Portfolio optimization with turnover constraint
- Cost-aware signal decay / signal half-life

### 3.2 经典学术方案

| 方向 | 核心思路 |
|------|------|
| Grinold & Kahn "Breadth" | Alpha 衰减速度决定最优换仓频率 |
| Almgren-Chriss | 最优执行：最小化 impact + timing risk |
| Garleanu & Pedersen 2013 | "Dynamic Trading with Predictable Returns" — 含交易成本的动态组合 |
| DeMiguel et al. | Turnover-constrained portfolio — l1 正则化换手 |
| Boyd et al. (cvxpy) | Multi-period portfolio optimization |

### 3.3 开源工程

| 项目 | 是否相关 |
|------|------|
| Qlib TopkDropoutStrategy | 最接近，但 cvxpy 依赖坏了 |
| Zipline rebalance API | 有 order_target_percent + commission model |
| vectorbt Portfolio | 支持 size/direction/frequency 参数化回测 |
| FinRL | RL-based portfolio，但偏端到端 |
| Riskfolio-Lib | 含 turnover constraint 的均值-方差优化 |
| PyPortfolioOpt | 有 transaction_cost 参数，但 cvxpy 依赖 |

### 3.4 工业界常见做法

需要调研：
- 百亿私募如何处理换手控制
- 做市商的 inventory management 是否有借鉴
- 量化基金的 signal decay 研究
- 动态止损/移动止盈的经典实现

---

## 四、理想的解决方案特征

一个好的组合执行方案应该具备：

1. **成本感知** — 不是事后扣成本，而是在选股时就考虑"换不换这只票值不值"
2. **Regime 自适应** — 牛市少动拿住，熊市快跑止损
3. **信号衰减建模** — 知道 alpha 的半衰期，到了该换就换
4. **可回测可验证** — rolling 多窗口稳定
5. **简单可解释** — 不能是黑箱

---

## 五、请 CX 补充

1. Qlib 里 `TopkDropoutStrategy` 的 dropout 参数是怎么工作的？跟我们的 `dropout_k` 是一个意思吗？
2. Qlib 的 `NestedExecutor` 和 `VWAPExecutor` 是否能直接接入？
3. Garleanu-Pedersen 2013 的 "aim portfolio" 方法是否适合我们的日频场景？
4. 有没有开源的 regime-switching portfolio 实现？
5. 工业界对 5 天 alpha 信号的最优换仓频率有没有经验值？
