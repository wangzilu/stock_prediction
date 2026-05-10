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

## CC 的详细解决方案

### 方案 A：在 CSZScoreNorm 之前合并（最正统）

**思路：** 问题的根因是 Alpha158 handler 内部做了 CSZScoreNorm，输出的特征是标准化的。而自定义因子在 handler 外部加入，跳过了归一化。正确做法是让自定义因子**也经过同一个归一化流程**。

**实现：** 继承 Alpha158，覆盖 `get_feature_config` 追加自定义字段：

```python
from qlib.contrib.data.handler import Alpha158

class Alpha158Enhanced(Alpha158):
    """Alpha158 + PE/PB/Turn custom factors, all processed together."""

    CUSTOM_FIELDS = [
        "($close - Min($close, 60)) / (Max($close, 60) - Min($close, 60) + 1e-8)",
        "1.0 / If(Abs($pe) > 0.01, $pe, 1.0)",
        "($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20) + 1e-8)",
        "$pb / Ref($pb, 5) - 1",
    ]
    CUSTOM_NAMES = ["price_pos60", "ep", "price_pos20", "pb_mom5"]

    def get_feature_config(self):
        fields, names = super().get_feature_config()
        # Append custom group
        return (list(fields) + [self.CUSTOM_FIELDS],
                list(names) + [self.CUSTOM_NAMES])
```

然后在 dataset config 中用 `Alpha158Enhanced` 替代 `Alpha158`。这样自定义因子和 Alpha158 原始因子一起经过 `CSZScoreNorm`，尺度完全一致。

**优点：** 最干净，一套 handler 出所有特征，归一化统一
**风险：** `get_feature_config` 返回的 list 结构需要兼容 QlibDataLoader 的解析逻辑，可能有嵌套格式问题（之前碰过 `'list' object has no attribute 'get_extended_window_size'`）
**验证方法：** 先用 CSI300 小数据跑通，确认 162 列全部非 NaN 且分布在 [-3, 3]

### 方案 B：手动对自定义因子做截面标准化（最稳妥）

**思路：** 不改 handler，在合并后手动对新因子做 CSZScoreNorm：

```python
# Alpha158 出来的特征已经是标准化的
X_alpha = dataset.prepare("train", col_set="feature")  # 158 列, ~N(0,1)

# 自定义因子是原始值
custom = D.features(instruments, exprs).swaplevel().reindex(X_alpha.index)

# 手动做截面标准化（和 CSZScoreNorm 相同逻辑）
for col in custom.columns:
    grouped = custom[col].groupby(level=0)  # 按日期分组
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    custom[col] = (custom[col] - mean) / (std + 1e-8)
    custom[col] = custom[col].clip(-3, 3)  # 截尾

# 现在两者尺度一致，可以合并
X_merged = pd.concat([X_alpha, custom], axis=1)
```

**优点：** 不改 Qlib 内部，最安全；归一化逻辑透明可控
**风险：** 需要确保分组方式（按日期截面）和 CSZScoreNorm 一致
**验证方法：** 合并后检查所有列的均值和标准差是否接近

### 方案 C：Two-Stage Model（避开合并问题）

**思路：** 不合并特征矩阵，而是训练两个独立模型再融合：

```
Stage 1: Alpha158 → XGB_alpha → score_alpha (当前 IC=0.024)
Stage 2: 自定义因子 → XGB_custom → score_custom
Final:   rank(score_alpha) * 0.7 + rank(score_custom) * 0.3 → ensemble
```

**优点：** 完全避开尺度/归一化问题；两个模型独立训练，互不干扰
**风险：** 自定义因子只有 4 个维度，单独训练 XGB 可能过拟合
**变种：** 用自定义因子做 `score_alpha` 的修正而非独立模型

```python
# Stage 2 变种：线性修正
final_score = score_alpha + 0.1 * zscore(ep) + 0.1 * zscore(price_pos60)
```

### 方案 D：用 Qlib 自定义 DataLoader（最优雅但需要调试）

**思路：** 用 `QlibDataLoader` 直接定义包含 Alpha158 + 自定义字段的完整特征集，让 Qlib 内部统一处理：

```python
handler_config = {
    "class": "DataHandlerLP",
    "module_path": "qlib.data.dataset.handler",
    "kwargs": {
        "data_loader": {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (
                        alpha158_fields + [CUSTOM_EXPRS],  # 所有字段
                        alpha158_names + [CUSTOM_NAMES],
                    ),
                    "label": ([[LABEL_EXPR]], [["LABEL0"]]),
                },
            },
        },
        "infer_processors": [{"class": "CSZScoreNorm", ...}],
    },
}
```

**优点：** 最 Qlib-native，所有特征同等处理
**风险：** 之前碰过 `'list' object has no attribute 'get_extended_window_size'` 错误，是 Alpha158 的 feature config 返回格式和 QlibDataLoader 的期望格式不兼容。需要把 Alpha158 的嵌套 list 展平。
**关键：** Alpha158 的 `get_feature_config()` 返回的是 `([group1, group2, ...], [names1, names2, ...])`，每个 group 是一个 list of expressions。QlibDataLoader 期望的是 `([[expr1, expr2, ...]], [[name1, name2, ...]])`。需要展平。

### 方案推荐顺序

| 优先级 | 方案 | 成功概率 | 复杂度 |
|:---:|:---:|:---:|:---:|
| 1 | **B（手动截面标准化）** | 高 | 低 |
| 2 | A（继承 Alpha158） | 中 | 中 |
| 3 | C（Two-Stage） | 高 | 低 |
| 4 | D（QlibDataLoader） | 中 | 高 |

**建议先试 B** — 只需要加 10 行代码，最快验证"尺度统一后是否能提升 IC"。如果 B 有效，说明确实是尺度问题，后续再改成 A 或 D 做长期方案。如果 B 无效，说明问题不在尺度，应该试 C（two-stage）。

### 方案 B 的具体实施代码

```python
def prepare_merged_normalized(dataset, segment, custom_exprs, custom_names):
    """Merge Alpha158 + custom factors with consistent normalization."""
    X = dataset.prepare(segment, col_set="feature")
    y = dataset.prepare(segment, col_set="label")
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # Fetch custom features
    instruments = list(set(str(c) for c in X.index.get_level_values(1)))
    dates = sorted(X.index.get_level_values(0).unique())
    custom = D.features(instruments, custom_exprs,
                        start_time=str(min(dates))[:10],
                        end_time=str(max(dates))[:10])
    custom.columns = custom_names
    custom = custom.swaplevel().sort_index().reindex(X.index)

    # KEY FIX: Cross-sectional z-score normalize each custom factor
    # This matches what CSZScoreNorm does to Alpha158 features
    for col in custom.columns:
        grouped = custom[col].groupby(level=0)
        mean = grouped.transform("mean")
        std = grouped.transform("std")
        custom[col] = (custom[col] - mean) / (std + 1e-8)
        custom[col] = custom[col].clip(-3, 3)

    X_merged = pd.concat([X, custom], axis=1)
    return X_merged, y
```

### 额外建议：因子稳定性 Rolling 测试

不要用单个 60 天窗口判断因子好坏。应该：

```python
# 12 个滚动窗口，每个 20 天
for split in range(12):
    test_end = today - split * 20 days
    test_start = test_end - 20 days
    ic = calc_single_factor_ic(factor, label, test_start, test_end)
    ic_history.append(ic)

# 判断标准：
# - IC > 0 的窗口占比 > 60% → 稳定有效
# - IC 均值 > 0.01 → 足够强
# - IC 标准差 < IC 均值 → 信号稳定
stable = (positive_ratio > 0.6) and (ic_mean > 0.01) and (ic_std < ic_mean)
```

一个因子在 12 个窗口中有 8 个 IC > 0，比当前窗口 IC = +0.028 但下个窗口可能 -0.03 更可信。

---

## CX 回应的关键纠正 — cc 必须接受的

### 纠正1：Alpha158 默认不对 feature 做 CSZScoreNorm

cc 假设 "Alpha158 特征经过 CSZScoreNorm，范围约 [-3, 3]"。

cx 查了 Qlib 0.9.7 源码，指出：
> Alpha158 默认 `learn_processors` 只对 **label** 做 `CSZScoreNorm`，不对 feature 做。Alpha158 的 feature 看起来稳定是因为表达式本身做了归一化（如 `Mean($close, 20)/$close`），不是因为有统一的 feature zscore。

**cc 接受这个修正。** 这意味着"尺度不匹配"的假设需要重新表述：不是"归一化 vs 未归一化"的冲突，而是"自归一化的技术因子 vs 原始值的基本面因子"的分布形态差异。

### 纠正2：单因子筛选标准偏了

cc 用 Pearson IC > 0.02 判定 STRONG。cx 指出：

> 价格位置60日 IC=+0.028 但 RankIC=-0.010，EP IC=+0.021 但 RankIC=-0.015。RankIC 反向意味着"按排序选 TopK"时这个因子可能是负贡献。

**cc 接受。** 生产目标是 TopK 选股，应该用 RankIC + TopK spread 联合判定，不能只看 Pearson IC。

修正后的 STRONG 判定：
```
STRONG = RankIC > 0.01 AND TopK_spread > 0 AND IC > 0
```

按这个标准重新看 20 个因子：**没有一个同时满足所有三个条件。** 这解释了为什么加进去都变差。

### 纠正3：对照实验不干净

cc 把增强版结果和"旧 baseline IC=0.024"对比。cx 指出两者可能来自不同日期/标签/实现。

**cc 接受。** 正确做法是在同一个脚本里即时训练 base 和 enhanced，用同一个 random seed、同一个 data split。

### 纠正4：residual IC 是更正确的验证方法

cx 提出的验证流程：
1. 先用 base model 预测 → `pred_base`
2. 计算残差 → `residual = label - pred_base`
3. 看新因子对 residual 是否有 IC
4. 如果没有 → 因子只是重复 Alpha158 已有信息

**cc 完全同意。** 这比直接 hstack 后训练更严谨。

### 纠正5：这次失败是"因子工厂第一堂工程课"

cx 总结：
> 因子宽度路线不能靠"看到几个字段就堆进去"。要建立：因子审计 → 稳定性监控 → 边际贡献实验 → rolling 验证 → 成本后回测 → 灰度回滚。

**cc 完全同意。** 不急着堆因子，先把"证明一个因子真的增量有效"的流程搭起来。

## 收敛后的下一步

根据 cc 和 cx 共同判断，下一步应该是：

1. **`scripts/audit_factor_merge.py`** — 合并审计（cx 提议 P1）
2. **`scripts/train_factor_ablation.py`** — 增量消融实验（cx 提议 P2）
   - base, base+factor_i, base+shuffled_factor（负控）
   - 在同一脚本同一 split 同一 seed 下对比
3. **改 `evaluate_factor_ic.py`** — 用 RankIC + TopK spread 联合判定（cx 提议 P3）
4. **residual IC 测试** — 新因子对 base model 残差是否有预测力

**只有通过以上全部验证的因子，才能进入生产模型。**

## CX 对 A/B/C/D 方案的精准纠正 — cc 再次接受

### 方案 A 的代码有结构错误

cc 写的：
```python
return (list(fields) + [self.CUSTOM_FIELDS], list(names) + [self.CUSTOM_NAMES])
```

cx 指出 Alpha158 的 `get_feature_config()` 返回的 `fields` 和 `names` **已经是平铺 list**（158 个 string），不是嵌套 list。cc 加了 `[self.CUSTOM_FIELDS]`（list of list），导致最后一个元素是 list 而非 string，这就是 `'list' object has no attribute get_extended_window_size'` 报错的根因。

**正确写法：**
```python
return fields + self.CUSTOM_FIELDS, names + self.CUSTOM_NAMES  # 平铺拼接
```

**cc 接受。** 这个 bug 确认了之前 QlibDataLoader 报错的原因。

### 方案 B 的前提不严谨

cx 指出：
1. Alpha158 features 不是 `~N(0,1)` — 只是表达式自身做了归一化
2. 手动 zscore custom 不等于"和 Alpha158 尺度一致"
3. 必须加 winsorize（PE 极值主导均值/方差）和 shuffled 负控

**cc 接受。** B 的正确实验名应该是 `base + custom_winsor_zscore`，而非 "尺度统一后的合并"。

### 方案 C 不能先拍 0.7/0.3 权重

cx 指出正确顺序：先训 base → 算 residual → 测 custom 对 residual 的 IC → 再决定是否 two-stage。如果 custom 对 residual 没有解释力，two-stage 只是包装噪声。

**cc 接受。** residual IC 是更严谨的验证方法。

### cx 修正后的执行顺序

| 优先级 | 动作 | cc 态度 |
|:---:|------|:---:|
| 1 | merge audit（审计对齐/缺失/极值）| 接受 ✅ |
| 2 | 同脚本 baseline 重训（消除口径差异）| 接受 ✅ |
| 3 | B'：winsorize + rank/zscore + shuffled 负控 | 接受 ✅ |
| 4 | 单因子 ablation + shuffled control | 接受 ✅ |
| 5 | residual IC → 再决定 two-stage | 接受 ✅ |
| 6 | Alpha158Enhanced handler（只在证明有效后）| 接受 ✅ |

### cx 定义的最小可执行实验

必须同时跑这 6 组：
```
base_raw
base_raw + custom_raw
base_raw + custom_winsor_zscore
base_raw + custom_rank
base_raw + shuffled_custom_winsor_zscore  ← 负控
base_raw + each_one_factor               ← 单因子消融
```

只有 `custom_winsor_zscore` 明显优于 `base_raw`，且 `shuffled_custom` 不优于 `base_raw`，才能说因子有真实边际贡献。

**cc 完全同意这个实验设计。** 比 cc 之前的"train → 看 IC → 结论"严谨得多。

---

## 已解决的技术问题

- ✅ Qlib multiprocessing spawn 问题 → 用 cx 的 `config/qlib_runtime.py`
- ✅ D.features index 顺序 `(instrument, datetime)` vs Alpha158 `(datetime, instrument)` → `swaplevel()`
- ✅ PE/PB/Turn 数据在 Qlib bin 里已有 → 不需要外部 API
- ✅ 单因子 IC 测试框架 → `scripts/evaluate_factor_ic.py`
