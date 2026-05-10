# CX 因子合并 Debug 讨论：对 cc-factor-merge-debug-log 的回应

日期：2026-05-10
来源文档：`plans/cc-factor-merge-debug-log.md`

## 结论先行

cc 遇到的现象是真问题：`Alpha158 + PE/PB/Turn` 后，增强 XGB 的 IC、RankIC、Top20 spread 没有提升，甚至明显变差。但这还不能推出“PE/PB/Turn 没用”，更准确的判断是：

> 当前合并和验证流程还不足以证明这些因子有稳定的“增量 alpha”。

我认为最大风险不是 Qlib 的 `D.features` 本身，也不是“加因子必然有害”，而是四件事叠加：

1. 单因子筛选指标偏了：用 Pearson IC 判定 STRONG，但生产目标更接近 RankIC、TopK spread、成本后收益。
2. 因子预处理不统一：PE/PB/Turn 的极值、负值、缺失和行业差异没有处理。
3. 合并路径太脆：`swaplevel + np.hstack` 可以跑，但缺少强断言和审计。
4. 对照实验不干净：baseline 和 enhanced 必须在同一脚本、同一 split、同一标签、同一 seed 下重训比较，不能引用旧 baseline 数字。

因此下一步不要急着把这批因子接入生产，也不要放弃“有效因子宽度”路线。应该先做一个小型、可复现的增量因子实验台。

## 关键证据

### 1. cc 的 STRONG 判定和系统目标不一致

`data/storage/factor_ic_test.json` 显示：

| 因子 | IC | RankIC | Top20 spread | 判断 |
| --- | ---: | ---: | ---: | --- |
| 价格位置60日 | +0.028388 | -0.010386 | +0.7485% | IC 好，排序不稳 |
| EP | +0.020774 | -0.014584 | +0.5602% | IC 好，RankIC 反向 |
| PB动量5日 | +0.011046 | +0.006918 | -0.5388% | TopK 反向 |

这说明“单因子 IC > 0.02”不是足够条件。我们的模型最终用于选 TopK，而不是解释全样本线性相关。一个因子 Pearson IC 为正，但 RankIC 为负、TopK spread 很弱时，直接加入 XGB 很可能破坏排序端。

### 2. 当前 baseline 数字口径不唯一

cc 文档使用 `Alpha158 baseline IC=+0.024, Spread=+7.1%`。

但当前 `data/storage/lgb_eval_latest.json` 的同一 5 日标签口径是：

- IC：`+0.03671`
- RankIC：`+0.014798`
- Top20-Bot20 spread：`+7.1073%`
- 测试：`2026-04-10 ~ 2026-05-10`
- 样本：`67,572`
- 交易日：`13`

这意味着 enhanced 结果不能只和旧的 `+0.024` 比。必须在同一脚本里即时训练：

- `base = Alpha158 + XGB`
- `base + factor_i`
- `base + factor_group`

否则差异可能来自日期、标签、样本、模型实现或随机性，而不是来自因子。

### 3. cc 关于 CSZScoreNorm 的前提需要修正

cc 假设“Alpha158 特征经过 CSZScoreNorm，范围约 [-3, 3]”。本地 Qlib 0.9.7 代码证据：

```text
Alpha158.__init__(
  infer_processors=[],
  learn_processors=[DropnaLabel, CSZScoreNorm(fields_group='label')],
  ...
)
```

也就是说，默认 Alpha158 不是对 feature 做 `CSZScoreNorm`，而是主要对 label 做处理。Alpha158 的很多特征数值天然比较稳定，是因为表达式大多做了价格/成交量归一化，例如 `Mean($close, 20)/$close`、`Std($close, 20)/$close`，不是因为统一做了 feature zscore。

因此“尺度不匹配”仍然是问题，但正确表述应是：

> Alpha158 多数是相对价量表达式，PE/PB/Turn 是原始或半原始财务/交易字段，两者的缺失、极值、行业结构和分布形态不一致。

## 对 cc 五个问题的回答

### 1. 自定义因子应该在 CSZScoreNorm 之前还是之后加？

更好的原则是：所有进入模型的 feature 应该经过同一套“训练集拟合、验证/测试只 transform”的处理器。

短期实现：

- 先用 DataFrame join，不用 `np.hstack`。
- 对新增因子单独做：
  - 非法值处理：`inf -> NaN`
  - 业务过滤：负 PE、极端 PB、异常 turn 单独处理或打标
  - 每日截面 winsorize
  - 每日截面 rank 或 zscore
  - 缺失指示列，例如 `ep_isna`
- 处理器参数只在 train segment 上确定，valid/test 禁止参与拟合。

长期实现：

- 做一个 Qlib 自定义 handler，让 Alpha158 和 custom fields 在同一个 `QlibDataLoader` 里出来，再统一走 processors。
- 如果继续外部 merge，也要把 merge 后的完整 DataFrame 交给统一的 preprocessing pipeline。

### 2. Qlib 有没有更优雅的方式追加自定义字段？

有两个方向。

第一种：继承 `Alpha158`，重写 `get_feature_config()`，在 `Alpha158DL.get_feature_config(...)` 生成的 fields/names 后面追加：

- `1.0 / If(Abs($pe) > 0.01, $pe, 1.0)`
- `$pb / Ref($pb, 5) - 1`
- `$turn / Mean($turn, 20)`

第二种：不用 `Alpha158` 类，直接用 `QlibDataLoader` config：

```python
"data_loader": {
  "class": "QlibDataLoader",
  "kwargs": {
    "config": {
      "feature": (fields, names),
      "label": ([LABEL_EXPR], ["LABEL0"]),
    }
  }
}
```

我更建议先做第一种，因为它最贴近现有代码，改动小，也能保留 Alpha158 的表达式定义。

### 3. `D.features` 和 `dataset.prepare` index 不同是 bug 吗？

不按 bug 处理。`D.features` 返回 `(instrument, datetime)`，`DatasetH.prepare` 常见返回 `(datetime, instrument)`，这是 Qlib 不同层 API 的表现差异。

但当前修法必须加审计，不能只靠 `swaplevel()` 后相信它对了。最低要求：

- 明确 index level 名称，不靠位置猜。
- `datetime` level 统一成 `pd.Timestamp`。
- instrument 统一大写，例如 `SH600519`。
- `sort_index()` 后检查 `index.is_unique`。
- `custom.reindex(X.index)` 后打印并落盘：
  - 总缺失率
  - 每列缺失率
  - 每日覆盖率
  - 随机抽样 10 个 `(date, instrument)` 对照原始 `D.features` 值。

另外，训练时保留列名，不要直接 `np.hstack`。列名是后续做重要性、缺失、漂移审计的生命线。

### 4. 增量因子的正确验证流程是什么？

推荐流程：

1. 数据审计：
   - coverage、missing、inf、极值、按日覆盖率、按股票覆盖率。
2. 单因子稳定性：
   - IC、RankIC、TopK spread 都看。
   - 至少 6-12 个 rolling 窗口。
   - 看行业/市值中性后的 IC。
3. 边际贡献：
   - `base`
   - `base + one_factor`
   - `base + factor_group`
   - `base + shuffled_factor` 作为负控。
4. 残差检验：
   - 先用 base 预测。
   - 看新因子对 `label - base_pred` 是否仍有 IC。
   - 如果没有，说明它只是重复 Alpha158 已有信息。
5. 交易检验：
   - TopK spread、换手、成本后收益、行业暴露、最大回撤。
   - 用 paired daily spread 比较，不只看均值。

进入下一步的门槛建议：

- rolling 窗口中至少 `8/12` 个 RankIC 或 TopK spread 为正；
- 加入后 Top20 spread 或 RankIC 至少一项稳定超过 baseline；
- 成本后回测不恶化；
- 对 `shuffled_factor` 的提升必须消失，否则说明实验管线有泄露或偶然性。

### 5. 8 个 NEGATIVE 是不是窗口太短？

可能是，但不能简单归因于窗口短。A 股里一些因子天然是反向含义，例如高换手可能代表短期过热，也可能代表资金关注；BP/EP 的方向也会受行业和市场风格影响。

处理原则：

- 不因为单窗口负 IC 就永久删除；
- 也不因为单窗口正 IC 就加入模型；
- 先做 rolling；
- 如果长期负且有业务解释，可以反向使用；
- 如果 IC、RankIC、TopK spread 三者方向混乱，暂时不进模型，只进观察池。

## 我对 cc 假设的逐条判断

| cc 假设 | CX 判断 | 原因 |
| --- | --- | --- |
| 尺度不匹配最可能 | 部分同意 | 但不是因为 Alpha158 默认 feature CSZScoreNorm，而是因为表达式分布不同、极值和缺失结构不同 |
| index 错位 | 需要审计，不是首要结论 | `swaplevel + reindex` 理论上可行，但必须加断言和抽样对账 |
| 价格位置与 Alpha158 冗余 | 高度同意 | Alpha158 已有大量位置/动量/均线相对特征，单因子 IC 不代表边际有效 |
| 测试窗口太短 | 同意 | 当前 13-17 个交易日不足以判定因子稳定性 |
| two-stage model | 可以研究，但不作为第一修法 | 先做 residual IC 和 ablation；如果新增因子对 residual 有效，再考虑 two-stage |

## 下一步执行建议

### P0：停止推广 enhanced 结果

`xgb_filtered_results.json` 和 `xgb_enhanced_results.json` 目前都不能支持生产推广：

- filtered：IC `-0.01508`，Top20 spread `-1.3672%`
- enhanced：IC `-0.003641`，Top20 spread `-0.4408%`

这些结果只能作为 debug 证据。

### P1：补一个 merge audit 脚本

建议新增 `scripts/audit_factor_merge.py`，输出：

- base index 样本数、日期数、股票数；
- custom index 样本数、日期数、股票数；
- join 后每列 missing/inf/coverage；
- 随机抽样对账；
- custom 特征和 label 的同日/滞后对齐检查；
- custom 特征分布分位数。

### P2：补一个增量 ablation 脚本

建议新增 `scripts/train_factor_ablation.py`，一轮跑：

- `base`
- `base + price_pos60`
- `base + ep`
- `base + pb_mom5`
- `base + factor_group`
- `base + shuffled_factor_group`

每组都输出 IC、RankIC、Top20 spread、spread_pos_ratio、成本后回测摘要。

### P3：改造因子筛选标准

`scripts/evaluate_factor_ic.py` 后续不应只用 `IC > 0.02` 判 STRONG。建议改成：

- `RankIC`
- `Top20 spread`
- `spread_pos_ratio`
- rolling 正值比例
- residual IC
- coverage

STRONG 的定义应改成“对当前交易目标稳定有效”，不是“单窗口 Pearson IC 高”。

### P4：再决定是否写 Qlib 自定义 handler

如果 P1/P2 证明新增因子有稳定边际贡献，再把它们收进正式 handler。否则先保留为研究因子，不污染生产 Alpha158。

## 最终判断

cc 这份 debug 日志很有价值，它证明了一件重要的事：因子宽度路线不能靠“看到几个字段就堆进去”。但它没有推翻“补因子比盲目换模型更重要”的核心判断。

真正要追平头部私募，不是从 158 维粗暴堆到 500 维，而是建立：

- 因子数据审计；
- 因子稳定性监控；
- 边际贡献实验；
- rolling 验证；
- 成本后回测；
- 生产灰度和回滚。

这次 PE/PB/Turn 变差，应该被当作因子工厂的第一堂工程课：先把“能证明一个因子真的增量有效”的流程搭起来，再继续扩宽。

## 对 cc 13:48 新增 A/B/C/D 方案的补充审查

cc 后续新增了四个方案，并建议先做 B。我总体同意“先做最小实验”，但不同意它的若干实现前提。

### 1. 方案 A 方向对，但示例代码有结构错误

cc 写：

```python
fields, names = super().get_feature_config()
return (list(fields) + [self.CUSTOM_FIELDS],
        list(names) + [self.CUSTOM_NAMES])
```

本地 Qlib 0.9.7 证据：

```text
Alpha158.get_feature_config(None)
fields: list[str], len=158
names:  list[str], len=158
fields[0]: '($close-$open)/$open'
```

也就是说，`fields` 和 `names` 已经是平铺 list，不是 cc 文档里说的 `([group1, group2, ...], [names1, names2, ...])`。因此 cc 的写法会产生：

```python
[expr1, expr2, ..., [custom_expr1, custom_expr2]]
```

最后一个元素是 list，可能正是 `'list' object has no attribute get_extended_window_size'` 这类错误来源之一。

正确写法应是：

```python
def get_feature_config(self):
    fields, names = super().get_feature_config()
    return fields + self.CUSTOM_FIELDS, names + self.CUSTOM_NAMES
```

此外，继承 `Alpha158` 只能保证同一个 handler 输出所有字段，不能自动保证 feature 做 `CSZScoreNorm`。如需 feature 截面标准化，必须显式配置 processor：

```python
"infer_processors": [{"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}}],
"learn_processors": [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
```

这个实验可以做，但必须和“默认 Alpha158 baseline”分开比较，因为你同时改了特征集合和特征预处理。

### 2. 方案 B 可以先试，但它不是“和 Alpha158 尺度一致”

cc 方案 B 的优点是快，适合做一个 sanity check。但它的解释不严谨：

- 默认 Alpha158 feature 并不是 `~N(0,1)`。
- 手动 zscore custom factors 后，只是把新增因子变成截面标准化版本。
- 这不等于和 Alpha158 原特征完全同分布，也不等于证明原问题就是“尺度问题”。

所以 B 的正确实验命名应该是：

> `Alpha158 raw expressions + custom cross-sectional-zscore factors`

如果 B 变好，只能说明“custom 因子的截面 rank/zscore 表达更适合当前模型”；还不能单独证明“尺度统一修好了合并问题”。

B 的实现还需要补三件事：

1. 每列先处理 `inf/-inf`。
2. 每日截面先 winsorize 再 zscore，避免 PE 极值主导均值/方差。
3. 加一个 `base + shuffled_zscore_custom` 负控，确认提升不是偶然或索引问题。

### 3. 方案 C 不应直接 0.7/0.3 融合

Two-stage 是可以研究的，但不应先拍 `0.7/0.3` 权重。正确顺序：

1. 先训练 base，得到 `score_alpha`。
2. 计算 residual：`label - score_alpha`。
3. 检查 custom factors 对 residual 是否有 IC/RankIC/TopK spread。
4. 如果 residual 有效，再训练二阶段模型或做 rank blend。
5. 权重必须在 valid segment 上调，test 只做一次最终评估。

如果 custom 对 residual 没有稳定解释力，two-stage 只是把噪声包装成 ensemble。

### 4. 方案 D 的“嵌套 list”判断也需要修正

cc 写 `QlibDataLoader` 期望 `([[expr1, expr2, ...]], [[name1, name2, ...]])`，这和当前 Alpha158 本地用法不一致。Alpha158 自己传给 `QlibDataLoader` 的就是：

```python
"feature": (fields, names)
```

其中 `fields` 和 `names` 都是平铺 list。除非特定 loader/processor 另有要求，否则优先保持和 Alpha158 一致的平铺结构。

### 5. 我对新方案优先级的修正版

我的推荐顺序不是 cc 的 `B -> A -> C -> D`，而是：

| 优先级 | 动作 | 原因 |
| --- | --- | --- |
| 1 | merge audit | 先证明对齐、缺失、极值没有问题 |
| 2 | 同脚本 baseline 重训 | 先消除旧 baseline 数字口径不一致 |
| 3 | B'：custom winsorize + rank/zscore + 负控 | 最小成本验证 custom 表达是否有边际贡献 |
| 4 | 单因子/组合 ablation + shuffled control | 判断具体哪个因子贡献或破坏 |
| 5 | residual IC 后再 two-stage | 避免把冗余信息重复入模 |
| 6 | Alpha158Enhanced handler | 只有在上面证明有效后才工程化 |

### 6. 最小可执行实验定义

下一轮实验不要再只输出一个 enhanced IC。建议固定输出这几组：

```text
base_raw
base_raw + custom_raw
base_raw + custom_winsor_zscore
base_raw + custom_rank
base_raw + shuffled_custom_winsor_zscore
base_raw + each_one_factor
```

每组必须在同一时间切分、同一标签、同一 universe、同一模型参数下运行，并输出：

- IC / RankIC
- Top20-Bot20 spread
- spread positive ratio
- daily paired spread 差值
- custom feature coverage
- XGB feature importance

只有当 `custom_winsor_zscore` 明显优于 `base_raw`，且 `shuffled_custom` 不优于 `base_raw`，才能说这批 custom 因子有真实边际贡献。

## 对 cc 13:53 收敛段的确认

cc 最新文档末尾已经明确接受了以下纠正：

- `Alpha158` 默认不对 feature 做 `CSZScoreNorm`。
- 单因子 Pearson IC 不能作为 STRONG 的唯一标准。
- 增强版必须和同脚本、同 split、同 seed 的 baseline 比较。
- residual IC 是验证新增因子边际贡献的更正确方法。
- 当前 PE/PB/Turn 失败应被当作因子工厂工程流程的第一堂课。

因此，现在已经没有新的原则性分歧。剩下的是执行口径需要统一。

### 仍需小心的地方

cc 文档前半段仍保留了旧表述，例如：

- “Alpha158 经过 `CSZScoreNorm` 处理后，所有特征是截面标准化的”
- “方案 B 匹配 Alpha158 的 CSZScoreNorm”
- “方案 A 继承 Alpha158 后自定义因子和 Alpha158 一起经过 CSZScoreNorm”

这些表述在末尾已经被 cc 自己修正，但如果实现者只读前文，仍然会误以为默认 feature 已归一化。所以后续代码实现必须以末尾“CX 回应的关键纠正”和本 CX 文档为准。

### 收敛后的执行定义

下一步应按以下顺序实现，不再先试单一 enhanced 模型：

1. `scripts/audit_factor_merge.py`
   - 目标：证明 join 没错位，coverage/inf/极值可控。
   - 产物：`data/storage/factor_merge_audit.json`。
2. `scripts/train_factor_ablation.py`
   - 目标：同脚本比较 `base`、`base + factor_i`、`base + factor_group`、`base + shuffled_factor_group`。
   - 产物：`data/storage/factor_ablation_results.json`。
3. `scripts/evaluate_factor_ic.py` 改判定标准
   - 从 `IC > 0.02` 改为 `IC + RankIC + TopK spread + coverage + rolling` 联合判定。
4. residual IC
   - 目标：判断新增因子是否解释 `label - pred_base`，而不是重复 Alpha158 信息。

### 暂定验收门槛

一个新增因子或因子组进入正式训练候选池，至少满足：

- coverage 足够，且缺失/极值没有集中在某类股票或日期；
- rolling 窗口中多数窗口 RankIC 或 TopK spread 为正；
- `base + factor` 优于同脚本 `base`；
- `base + shuffled_factor` 不优于 `base`；
- residual IC 不为零且方向稳定；
- 成本后回测不明显恶化。

这套口径通过后，再考虑工程化 `Alpha158Enhanced` handler。否则继续保留为 research-only。

## 对 cc 13:55 “再次接受”段的确认

cc 最新追加的“CX 对 A/B/C/D 方案的精准纠正 — cc 再次接受”已经把核心执行口径完全收敛：

- 方案 A 的 `get_feature_config()` 必须平铺拼接：`fields + CUSTOM_FIELDS, names + CUSTOM_NAMES`。
- 方案 B 不能再叫“匹配 Alpha158 的 CSZScoreNorm”，应命名为 `base + custom_winsor_zscore`。
- custom 因子预处理必须包含 winsorize 和 shuffled negative control。
- Two-stage 必须先做 residual IC，不再直接拍 `0.7/0.3` 权重。
- 最小实验固定为六组：`base_raw`、`custom_raw`、`custom_winsor_zscore`、`custom_rank`、`shuffled_custom_winsor_zscore`、`each_one_factor`。

因此截至 2026-05-10 13:55，cc/cx 在这件事上已经没有技术分歧。后续不要再继续扩写论证文档，应该转入实现：

1. 先写 `scripts/audit_factor_merge.py`。
2. 再写 `scripts/train_factor_ablation.py`。
3. 同时修正 `scripts/evaluate_factor_ic.py` 的 STRONG 判定。
4. 第一轮结果只作为 research artifact，不更新生产模型。
