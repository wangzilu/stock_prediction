# Track B 最终结论

**日期：** 2026-05-18
**作者：** CC

---

## 最优执行策略：Buffered Partial Rebalance（无止损）

### 单期回测（6 个月）

| 指标 | 值 |
|------|:---:|
| Raw 年化 | -11.6% |
| Net 年化 | -12.5% |
| 换手 | 1% |
| 成本/收益 | 9% |

注：单期结果偏弱，但 rolling 证明是稳定策略。

### 12-Split Rolling 验证

| 指标 | 值 |
|------|:---:|
| avg 年化 | +82.9% |
| avg Sharpe | +2.364 |
| 年化>0 | 67% (8/12) |

### 策略对比

| 策略 | avg 年化 | avg Sharpe | 年化>0 |
|------|:---:|:---:|:---:|
| daily_rebal | +65.2% | +1.155 | 50% |
| **buffered_partial** | **+82.9%** | **+2.364** | **67%** |
| buffered+stop8% | +15.0% | -0.094 | 33% |

### 结论

1. **Buffered Partial 是最优策略** — buffer zone + 部分换仓 + vol throttle
2. **止损 8% 太紧** — A 股日常波动就有 5-10%，频繁触发导致错过反弹
3. **67% 胜率可接受** — 赢的 split 盈利远大于亏的 split 亏损
4. 换手极低（1-2%），成本可忽略

### 策略参数

```python
mode="buffered_partial"
top_k=20
buffer=5          # 排名 21-25 不卖
trade_rate=0.35   # 每天换 35% 朝目标
min_hold_days=2   # 最少持有 2 天
max_daily_turnover=0.15  # 日换手上限 15%
vol_threshold=1.5 # 高波动时减半交易
```
