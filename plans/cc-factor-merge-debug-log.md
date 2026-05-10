# CC 因子合并 Debug 日志 — 请 CX 审查

**日期：** 2026-05-10
**问题：** Alpha158 + PE/PB/Turn 自定义因子合并后，XGB IC 从 +0.024 变成 -0.015，加因子反而变差。

---

## 做了什么

### 第一步：发现 Qlib bin 里已经有 PE/PB/Turn

baostock 更新数据时就写入了 `pe.day.bin`、`pb.day.bin`、`turn.day.bin`、`amount.day.bin`。Alpha158 没有用这些字段（只用 `$open/$high/$low/$close/$volume`）。

验证：
```
$ ls data/storage/qlib_data/cn_data/features/sz002790/
amount.day.bin  close.day.bin  high.day.bin  low.day.bin
open.day.bin    pb.day.bin     pe.day.bin    turn.day.bin  volume.day.bin
```

### 第二步：用 Qlib 表达式引擎构造 20 个候选因子

```python
FACTORS = {
    "$pe": "PE",
    "$pb": "PB",
    "1.0 / $pe": "EP(盈利收益率)",
    "$pe / Ref($pe, 20) - 1": "PE动量20日",
    "$turn / Mean($turn, 20)": "换手异常",
    "($close - Min($close, 60)) / (Max($close, 60) - Min($close, 60))": "价格位置60日",
    # ... 共 20 个
}
```

### 第三步：单因子 IC 测试（scripts/evaluate_factor_ic.py）

用 `config/qlib_runtime.py` 解决了 macOS multiprocessing spawn 问题。结果：

| 因子 | IC | 判定 |
|------|-----|------|
| 价格位置60日 | +0.028 | STRONG ✅ |
| EP(1/PE) | +0.021 | STRONG ✅ |
| 价格位置20日 | +0.016 | OK |
| PB动量5日 | +0.011 | OK |
| PE动量5日 | +0.006 | OK |
| PE原始 | +0.002 | WEAK |
| PB原始 | +0.004 | WEAK |
| BP(1/PB) | **-0.052** | NEGATIVE ❌ |
| 换手异常20日 | **-0.012** | NEGATIVE ❌ |
| 换手异常60日 | **-0.019** | NEGATIVE ❌ |
| 换手动量 | **-0.010** | NEGATIVE ❌ |
| ... 共 8 个 NEGATIVE | | |

**20 个候选中只有 2 个 STRONG，8 个 NEGATIVE。**

### 第四步：只加 STRONG 因子重训 — 仍然变差

只加了 4 个因子（2 个 STRONG + 2 个 OK），158 → 162 维：

| 指标 | Alpha158 (158维) | +4 STRONG (162维) | 变化 |
|------|:---:|:---:|:---:|
| IC | +0.024 | **-0.015** | ❌ 大幅恶化 |
| RankIC | +0.015 | -0.005 | ❌ |
| Spread | +7.1% | -1.4% | ❌ |

### 第五步：全部堆进去（171维）— 更差

| 指标 | Alpha158 | +13 ALL (171维) |
|------|:---:|:---:|
| IC | +0.024 | **-0.004** |
| Spread | +7.1% | -0.4% |

---

## 我的 Debug 假设

### 假设1：尺度不匹配（最可能）

Alpha158 经过 `CSZScoreNorm` 处理后，所有特征是截面标准化的（均值~0，标准差~1）。

但自定义因子是**原始值**：
- Alpha158 特征范围：[-3, 3]
- PE 原始值范围：[-1000, +5000]
- Turn 原始值范围：[0, 50]

XGB 虽然理论上不受尺度影响（基于分裂点），但实际上：
1. `CSZScoreNorm` 改变了 Alpha158 特征的分布形状
2. 原始值因子的极端值可能干扰树的分裂策略
3. NaN 的分布也不同（Alpha158 的 NaN 被 CSZScoreNorm 处理过，自定义因子没有）

**待验证：** 对自定义因子也做 CSZScoreNorm 后再合并。

### 假设2：信息泄露/标签错位

D.features 返回的 index 是 `(instrument, datetime)`，Alpha158 的 index 是 `(datetime, instrument)`。虽然用了 `swaplevel()` 修复，NaN 降到了 0.6%，但有没有可能少量行的对齐是**错位**的？

如果第 N 只股票的 PE 值被对齐到第 M 只股票的 Alpha158 特征上，XGB 学到的就是噪声。

**待验证：** 抽样检查几只股票的合并结果是否正确。

### 假设3：价格位置因子与 Alpha158 高度冗余

Alpha158 已经有 `($close - Min($close, N)) / (Max($close, N) - Min($close, N))` 类似的因子（RSV 等）。加入一个高度相关但尺度不同的版本可能引入多重共线性。

**待验证：** 计算自定义因子和 Alpha158 特征的相关性矩阵。

### 假设4：测试窗口太短

只用了 29 天测试（test_start = today-29），IC 的方差很大。可能纯粹是运气不好。

**待验证：** 用 60-120 天测试窗口重跑。

---

## 请 CX 看的问题

1. **合并自定义因子时，应该在 CSZScoreNorm 之前还是之后加？**
   - 如果之前：自定义因子也会被归一化，但 Alpha158 handler 的 data_loader 不容易扩展
   - 如果之后：需要自己做归一化，保证尺度一致

2. **Qlib 有没有更优雅的方式给 Alpha158 追加自定义字段？**
   - 当前方案是 Alpha158 handler 出特征 → D.features 出自定义因子 → swaplevel → hstack
   - 有没有办法直接在 handler config 里加自定义表达式？

3. **D.features 和 dataset.prepare 的 index 格式不同（instrument,datetime vs datetime,instrument）是设计如此还是 bug？**
   - 如果是设计如此，应该有官方的 merge 方式

4. **增量因子的正确验证流程是什么？**
   - 单因子 IC 通过 ≠ 模型提升（已证明）
   - 应该用什么方法验证"加入这个因子后模型整体提升"？

5. **20 个因子中 8 个是 NEGATIVE，是不是测试窗口太短导致的偶然？**
   - 应该用 rolling 多窗口测试因子稳定性
   - 一个因子在 12 个窗口中 8 个 IC>0 vs 当前窗口 IC<0，哪个更可信？

---

## 下一步计划

1. 验证假设1：对自定义因子做独立 CSZScoreNorm 后再合并
2. 验证假设2：抽样 10 只股票检查合并后的值是否正确
3. 验证假设3：计算 correlation matrix
4. 验证假设4：用 120 天测试窗口重跑
5. 如果以上都不行，考虑 **two-stage model**：Alpha158→XGB 出一个分数，自定义因子→第二个模型出一个分数，两个分数再 ensemble

---

## 已解决的技术问题

- ✅ Qlib multiprocessing spawn 问题 → 用 cx 的 `config/qlib_runtime.py`
- ✅ D.features index 顺序 `(instrument, datetime)` vs Alpha158 `(datetime, instrument)` → `swaplevel()`
- ✅ PE/PB/Turn 数据在 Qlib bin 里已有 → 不需要外部 API
- ✅ 单因子 IC 测试框架 → `scripts/evaluate_factor_ic.py`
