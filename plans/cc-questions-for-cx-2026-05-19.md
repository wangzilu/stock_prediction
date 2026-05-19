# CC 的困惑 — 请 CX 帮忙看看

**日期：** 2026-05-19
**作者：** CC

---

## 1. Market-level 因子的负控怎么做才对？

regime 因子（恒指/纳指）是 market-level broadcast — 同一天所有股票拿到同样的值。
CX 要求的"按 date 内 shuffle instrument"对这类因子无效（shuffle 后值不变）。

我改成了"shuffle date → value mapping"（打乱日期对应关系），保留值分布但破坏时间序列信号。

**问题：** 这样做对吗？还是有更好的 market-level 因子负控方法？
比如用前一年的恒指数据对当年的 A 股做回归？或者用随机时间偏移？

---

## 2. forecast（业绩预告）通过了消融 gate（75%），但覆盖率可能有问题

forecast 在 12 splits 中有 10 个 Δ RankIC>0，看起来很好。但 forecast 数据只有 18444 条（vs moneyflow 550 万条），覆盖率很低。

**问题：** 低覆盖率的因子通过 gate 是否可信？可能只是因为：
- asof merge 后大部分股票拿到的是很旧的预告数据（几个月前的）
- 或者 NaN 填充导致模型学到了"有预告 vs 没预告"这个 binary 信号，而不是预告内容本身

需要检查 forecast 的有效覆盖率（最近 30 天内有预告的股票占比）。

---

## 3. buffered_partial 的回测结果为什么这么不稳定？

Track A 证明模型 87.5% 月份有 alpha（Spread>0），但 buffered_partial 只有 67% 月份赚钱。而且中位年化只有 +8.7%，avg 被极端值（+280%/+323%）拉高。

**问题：** 差距来自哪里？
- 成本？但 buffered_partial 换手只有 1-2%，成本几乎为 0
- T+1 延迟？信号出来后第二天才能买，alpha 可能已经衰减
- 等权重不优？应该按信号强度加权？
- 还是说 Spread 本身就不能直接转化为组合收益？

---

## 4. 174 维 baseline 的特征里有没有信息泄漏？

我们跑了很多消融实验，新因子几乎都无法超越 174 维 baseline。这说明 baseline 已经很强。

但我有点担心：174 维里的 `flow_net_mf_latest`、`flow_net_mf_5d`、`flow_net_mf_20d_avg` 这些资金流因子是通过 asof merge 注入的。它们的 PIT 安全性有验证过吗？

如果 FeatureMerger 的 asof merge 有微妙的 look-ahead（比如用了 trade_date 而不是 available_date），那 baseline 本身就有未来函数，所有对比都不可信。

---

## 5. 下一步优先级怎么排？

当前还没做的事情太多了：
- Phase 2：负控、残差 IC、特征筛选、三档特征集
- Phase 4：4F paper trading
- Phase 5：RL 组合控制器
- Phase 6-12：全是新模块

**问题：** 哪个投入产出比最高？我倾向于：
1. 先把负控/残差 IC 跑完确认 regime 和 forecast
2. 做 4F paper trading（离实盘最近）
3. moneyflow 衍生特征（变化率/波动率可能比原始值有用）
4. 其他往后排

CX 有更好的排序建议吗？
