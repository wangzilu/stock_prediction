# Track B 结论

**日期：** 2026-05-18（05-19 修正）
**作者：** CC

---

## 执行策略候选：Buffered Partial Rebalance（无止损）

**定位：shadow/paper 候选，非"已确认最优"。**

### 12-Split Rolling 验证

| 指标 | 值 | 备注 |
|------|:---:|------|
| avg 年化 | +82.9% | ⚠️ 被 +280%/+323%/+426% 极端 split 拉高 |
| **中位年化** | **+8.7%** | 更真实的水平 |
| avg Sharpe | +2.364 | 同样被极端值拉高 |
| 年化>0 | 67% (8/12) | 仍有 4/12 亏损 |

### 策略对比

| 策略 | avg 年化 | avg Sharpe | 年化>0 |
|------|:---:|:---:|:---:|
| daily_rebal | +65.2% | +1.155 | 50% |
| buffered_partial | +82.9% | +2.364 | 67% |
| buffered+stop8% | +15.0% | -0.094 | 33% |

### 结论（修正）

1. **Buffered Partial 优于 daily rebal** — 67% 胜率 vs 50%，成本更低
2. **但不能说"最优策略已定"** — 中位年化仅 +8.7%，极端 split 贡献了大部分收益
3. **止损 8% 太紧** — A 股日常波动就有 5-10%，频繁触发导致错过反弹
4. **正确定位：进入 shadow/paper 候选**，在 paper trading 中继续验证

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
