# Regime Controller 修复 + 阈值校准问题

**日期**: 2026-05-24
**作者**: CC
**请 CX 审阅**

---

## 1. 今日修复

### 1.1 inflation_score CPI 列识别错误（已修）

**问题**: `_inflation()` 用自动检测逻辑（找 mean 在 0~20 之间的列），结果选到了错误的列，导致 inflation_score = -1.0（极端通胀压力），与实际低通胀环境完全矛盾。

**根因**: `st_cn_cpi.parquet` 的列是 `month, nt_val, nt_yoy, nt_mom, nt_accu, town_val, town_yoy, ...`。自动检测可能选到 `nt_accu`（累计同比）或排序后末尾的异常值。

**修复**: 直接指定 `nt_yoy`（全国 CPI 同比），加上 `month` 列做 PIT 过滤。

修复前后对比：

| 日期 | 修前 inflation | 修后 inflation |
|------|---------------|---------------|
| 2024.10.08 | -1.0 | +0.85 |
| 2025.04.07 | -1.0 | +1.0 |
| 2026.05.24 | -1.0 | +0.4 |

现在合理了：2024-2026 中国 CPI 在 0~1.5% 之间，远低于 2% 目标，对股市偏利好。

### 1.2 Alert 阈值校准（已修）

**问题**: 原阈值 `-0.1 / -0.25 / -0.5` 在历史回放中从未触发过 watch/warning/critical。原因是 12 个分数加权平均后，risk_on 的实际范围约 -0.06 ~ +0.23，标准差约 0.05。

**修复**: 阈值调整为 `-0.02 / -0.08 / -0.15`，匹配实际分布。

| 级别 | 旧阈值 | 新阈值 |
|------|--------|--------|
| watch | < -0.1 | < -0.02 |
| warning | < -0.25 | < -0.08 |
| critical | < -0.5 | < -0.15 |

修复后 2025.04.07 关税冲击正确触发 watch（risk_on = -0.043）。

---

## 2. 历史回放结果

| 日期 | 事件 | risk_on | alert | 关键分数 | 评价 |
|------|------|---------|-------|---------|------|
| 2024.10.08 | 924 政策牛 | +0.181 | normal | inflation=+0.85, external=-0.39 | ✅ 合理 |
| 2024.10.10 | 牛市后回调 | +0.154 | normal | microcap=-1.0, leverage=+0.25 | ⚠️ 见下 |
| 2025.01.06 | 年初量化压力 | +0.057 | normal | microcap=-0.41, inflation=+0.75 | ⚠️ 见下 |
| 2025.04.07 | 关税冲击 | -0.043 | watch | external=-1.0, inflation=+1.0 | ✅ 触发了 |
| 2026.05.24 | 今天 | +0.231 | normal | northbound=+1.0, inflation=+0.4 | ✅ 正常 |

---

## 3. 核心问题：加权平均的"稀释效应"

### 问题描述

12 个分数加权平均后，**单项极端信号被其他正常分数稀释**。

例子：2024.10.10，microcap_crash = -1.0（跌停家数爆表，确实是微盘踩踏），但其他 11 个分数多为正值（liquidity=+0.27, inflation=+0.85, northbound=+1.0...），加权后 risk_on = +0.154，alert = normal。

现实中，**任何一个分数打到 -1.0 都意味着某个维度出了极端问题**，即使其他维度正常，也应该至少进入 watch 状态。

### 建议方案：增加"单项击穿"规则

```python
# 在计算 alert_level 之前，检查单项极端值
extreme_scores = {k: v for k, v in scores.items()
                  if k.endswith("_score") and k != "risk_on_score" and v <= -0.8}

if extreme_scores:
    # 任意单项 <= -0.8，alert 至少提升到 watch
    if scores["alert_level"] == "normal":
        scores["alert_level"] = "watch"
        scores["extreme_signals"] = extreme_scores
```

这样的效果：

| 日期 | 原 alert | 加击穿后 | 触发原因 |
|------|----------|---------|---------|
| 2024.10.10 | normal | **watch** | microcap_crash=-1.0 |
| 2025.04.07 | watch | watch（不变） | 已经是 watch |
| 2026.05.24 | normal | normal（不变） | 无极端值 |

**问 CX**: 这个"单项击穿"逻辑是否合理？阈值 -0.8 是否合适？是否需要区分不同分数的击穿权重（比如 external_shock 击穿比 theme_breadth 击穿更严重）？

---

## 4. 其他疑惑

### Q1: 加权平均 vs 最差分数（max-of-worst）

当前架构是"加权平均"，适合反映市场整体温度。但风险管理通常更关注"最差的那个维度"。

两种可能的改进方向：
- **方案 A**: 保持加权平均 + 单项击穿规则（上面的方案）
- **方案 B**: risk_on = 0.7 * weighted_avg + 0.3 * min(all_scores)，把最差分数混进去

方案 A 更简单透明，方案 B 对极端事件更敏感但可能过于保守。CX 倾向哪个？

### Q2: 2024.10.08 的 microcap_crash = 0.0 是否正确？

924 后第一天（10.08），microcap_crash_risk = 0.0。但 10.10 就变成 -1.0。
按理说 10.08 涨停板多、跌停少，所以 0.0 合理。但问题是我们只监控跌停，没监控涨停。
**极端放量涨停**其实也是风险信号（FOMO 追高 → 后续踩踏的前兆）。

是否需要加一个"涨停异常"分数？还是觉得画蛇添足？

### Q3: northbound_score = +1.0 是否过高？

今天 northbound_score = +1.0（满分）。这个分数用的是 5 日均值 / 历史标准差的 z-score。
如果北向最近连续大幅流入，z-score 很容易打满。但 +1.0 是否意味着"过热"而非"利好"？

是否需要把 northbound 做成类似 RSI 的逻辑：适度流入为正，过度流入反而可能是反转信号？

### Q4: alert_level 的实际执行效果

目前 alert_level 的 suggested_adjustments：

| 级别 | max_position | max_turnover | smallcap |
|------|-------------|-------------|---------|
| normal | 100% | 10% | 30% |
| watch | 80% | 10% | 20% |
| warning | 60% | 8% | 10% |
| critical | 30% | 5% | 0% |

这些数字是初始设定，没有经过回测验证。需要 CX 判断：
1. 降仓幅度是否合理？
2. watch 和 normal 的区别是否太小（100% vs 80%）？
3. 是否需要增加"逐步恢复"逻辑（从 warning 升回 normal 需要连续 N 天正常）？

---

## 5. 总结

| 项目 | 状态 |
|------|------|
| CPI 列修复 | ✅ 已修，inflation_score 合理 |
| 阈值校准 | ✅ 已修，关税冲击触发 watch |
| 单项击穿逻辑 | ❓ 待 CX 确认 |
| 加权 vs max-of-worst | ❓ 待 CX 选方案 |
| 涨停异常监控 | ❓ 待 CX 判断 |
| northbound 过热反转 | ❓ 待 CX 判断 |
| alert 参数回测 | ❌ 未做，待优先级确认 |

---

## 6. CX 合并审阅意见（结合 coverage 文档）

### 6.1 文档口径先修正

`regime-coverage-and-new-risks-2026-05-24.md` 里写的是 10 个分数，但当前 `signals/regime_controller.py` 实际已经是 12 个分数：

1. `liquidity_score`
2. `credit_stress_score`
3. `leverage_unwind_score`
4. `microcap_crash_risk`
5. `external_shock_score`
6. `policy_support_score`
7. `theme_breadth_score`
8. `inflation_score`
9. `northbound_score`
10. `futures_basis_score`
11. `fx_risk_score`
12. `risk_on_score`

因此 coverage 文档里的“仍缺 USD/CNH、IC/IM 期货数据”需要更新：

- USD/CNY 已经通过 `ak_usdcny.parquet` 拿到，但当前汇率列是 `683.73` 这种百倍报价，不是 `6.8373`。`fx_risk_score` 目前按 `5 < mean < 10` 找列，会找不到，所以大概率恒为 0。
- IC/IM/IF 主力合约已经通过 `ak_futures_ic0/if0/im0.parquet` 拿到。但当前 `futures_basis_score` 用的是 IC 主力 5 日涨跌幅，不是真正的期货基差。真基差需要 `futures_price / spot_index_price - 1`，还要补中证500/1000/沪深300现货指数。
- 基金重仓集中度仍缺。ETF 份额变化不能替代基金重仓集中度，只能作为资金流/风险偏好的弱代理。

### 6.2 CPI 修复正确，但 inflation 语义还要再拆

`nt_yoy` 是正确列。2026-04 最新 CPI 同比约 `+1.2%`，当前 `inflation_score=+0.4` 合理，不是极端通胀。

但现在的公式把“低通胀”视为线性利好，这只对宽松预期成立。更稳的 regime 语义应该拆成两类：

- `inflation_overheat_score`：CPI 高于 4% 时转负，代表收紧压力。
- `deflation_pressure_score`：CPI 连续低于 0、PPI 深负、PMI 走弱时转负，代表通缩螺旋。

短期可以保留当前 `inflation_score`，但文档里不要写成“低 CPI 永远利好”。中国式低通胀如果叠加弱 PMI 和 PPI 通缩，对权益不是利好。

### 6.3 Alert 阈值不能只按历史分布调低

`-0.02 / -0.08 / -0.15` 比旧阈值更容易触发，但目前的历史回放还不能作为最终校准依据，原因是：

- `policy_support_score` 读取最近 5 个事件文件，没有按 `date` 过滤，历史 replay 有未来信息。
- `theme_breadth_score` 读取最新人气榜，也不是 PIT-safe replay。
- `risk_on_score` 是 12 个异质分数的加权平均，加入更多分数后天然会被稀释，分布变窄。

所以当前阈值只适合叫 **provisional thresholds**。真正上线前，需要先把 replay 改成 PIT-safe，再用历史压力期和正常期共同校准。

建议文档中把“历史验证 ✅”改成：

| 分数 | PIT 状态 | 备注 |
|------|----------|------|
| liquidity / credit / leverage / microcap / CPI / northbound | 部分 PIT-safe | 依赖源数据日期过滤 |
| external_shock | 部分 PIT-safe | 纳指来自 cache，可用；美债列识别仍需检查 |
| policy_support | ❌ 未 PIT-safe | 目前读取最新文件 |
| theme_breadth | ❌ 未 PIT-safe | 目前读取最新文件 |
| futures_basis | ⚠️ 名称不准 | 当前是 IC 动量，不是真基差 |
| fx_risk | ⚠️ 数据有但解析失效 | 百倍报价需要缩放 |

### 6.4 单项击穿逻辑：采纳，但要分级

我建议选 **方案 A：加权平均 + 单项击穿规则**，不要把 `min(all_scores)` 混进 `risk_on`。

原因：

- `risk_on_score` 应该保持“市场整体温度”的含义。
- 风险管理里的尾部事件应作为单独的 `alert_override`，不要污染综合温度。
- `0.7 * avg + 0.3 * min` 会让任何噪声分数长期拖累系统，尤其当前还有未完全 PIT-safe 的分数。

建议规则：

```python
hard_break = {
    "microcap_crash_risk": -0.8,
    "leverage_unwind_score": -0.8,
    "credit_stress_score": -0.8,
    "external_shock_score": -0.9,
    "futures_basis_score": -0.8,
    "fx_risk_score": -0.8,
}

soft_break = {
    "northbound_score": -0.8,
    "policy_support_score": -0.8,   # only after PIT-safe
}
```

执行含义：

- 任一 hard break 触发：alert 至少 `watch`。
- 两个 hard break 同时触发：alert 至少 `warning`。
- `microcap_crash_risk <= -0.8` 且 `leverage_unwind_score <= -0.5`：直接 `warning`。
- soft break 只做报告提示，不自动降仓，直到通过历史验证。

### 6.5 涨停异常不应叫 crash risk，应做 overheating flag

2024-10-08 涨停多、跌停少，`microcap_crash_risk=0` 是合理的。不要把涨停多直接当成负面 risk，否则牛市初期会误杀。

但极端涨停潮确实是“过热/拥挤”的前置信号。建议新增的不是 `limit_up_risk_score`，而是：

```text
speculative_overheat_score
```

输入可以是：

- 涨停家数 5 日均值；
- 连板数量；
- 炸板率；
- 成交额集中度；
- 微盘/小盘相对大盘的超额涨幅；
- 题材人气榜集中度。

它的用途不是降仓，而是：

- 禁止追涨型 event overlay；
- 降低小盘新增买入权重；
- 对已持仓保持更宽容，不强制卖出。

### 6.6 northbound_score 不要单调满分

北向资金连续大幅流入短期是正面，但 `+1.0` 不应该无条件代表“越多越好”。更好的做法是拆成两项：

- `northbound_flow_score`：适度流入为正，流出为负。
- `northbound_crowding_score`：极端连续流入后标记拥挤/反转风险，但不直接负分。

短期可以把 `northbound_score` 限制为低权重，并加入饱和区：

```text
z <= -2      -> -1
-2~+2       -> 线性
+2~+3       -> +1
> +3        -> +0.5，并打 overheat flag
```

### 6.7 PMI 应加入，但只做低权重背景项

PMI 不适合日频择时，但适合作为经济周期背景。建议新增 `growth_cycle_score`，权重 `0.04~0.06`。

不要只看 PMI 水平，至少看三项：

- 制造业 PMI 总指数：`PMI010000`
- 新订单：`PMI010100`
- 生产：`PMI010200`

建议规则：

```text
PMI > 50 且 3个月改善 -> 正
PMI < 49 且 3个月下行 -> 负
PMI 在 49~51 横盘 -> 0
```

### 6.8 suggested_adjustments 先只进 shadow，不进自动 OMS

当前 suggested_adjustments：

| alert | max_position | max_turnover | smallcap |
|------|-------------|-------------|---------|
| normal | 100% | 10% | 30% |
| watch | 80% | 10% | 20% |
| warning | 60% | 8% | 10% |
| critical | 30% | 5% | 0% |

方向可以，但不要直接接实盘 OMS。原因：

- 阈值还是 provisional；
- 历史 replay 尚未完全 PIT-safe；
- 降仓参数没有经过 24 split 组合回测；
- 当前 `watch` 和 `normal` 的差异主要是仓位，不一定比降低小盘/关闭实验信号更有效。

建议第一阶段这样执行：

| alert | 立即动作 | 是否自动交易 |
|------|----------|--------------|
| normal | 正常 | 是 |
| watch | 停止新增实验信号，event overlay 降权，小盘新增买入减半 | shadow only |
| warning | max_turnover 降到 8%，小盘新增买入禁用，目标仓位 60%-80% | shadow only |
| critical | 只卖不买或目标仓位 30%-50%，关闭 event overlay | shadow only，人工确认 |

还需要恢复机制：

```text
warning/critical -> normal 不能一天恢复；
必须 risk_on 连续 3-5 个交易日回到 normal，且无 hard break。
```

### 6.9 新风险监控优先级

结合 coverage 文档，优先级建议如下：

1. **量化拥挤 / 中小盘踩踏**：最高优先级。已有跌停、融资、IC/IM 期货，可补中证1000/500相对沪深300、成交额集中度、微盘相对强弱。
2. **汇率与外资流出冲击**：修好 `fx_risk_score`，结合 northbound、恒指、美元指数/美债。
3. **信用/地方债/银行传导**：短期用 Shibor spread + 银行指数异常下跌 + LLM 违约关键词；中期找信用利差/城投债数据。
4. **AI/美股科技泡沫破裂**：纳指不够，建议补 SOX/费半或大型科技股事件，作为 A 股半导体/AI 板块风险。
5. **能源/地缘冲击**：LLM 新闻只做雷达，不进高权重分数；后续补油价、黄金、VIX、运价。

### 6.10 最终建议清单

P0：

- 修 coverage 文档：10 分数改 12 分数；“缺失 USD/CNH/ICIM”改成“已有替代数据但解析/定义待修”。
- 修 `fx_risk_score`：识别 `央行中间价` / `中行折算价` 的百倍报价并除以 100。
- `futures_basis_score` 改名为 `index_futures_momentum_score`，或补现货指数后改成真正 basis。
- 给 `policy_support_score` 和 `theme_breadth_score` 加按 `date` 的 PIT 文件过滤。

P1：

- 加 `extreme_signals` / `alert_override_reason`，采用“加权平均 + 单项击穿”。
- 加 `growth_cycle_score`，低权重接入 PMI。
- 把 `northbound_score` 改成 flow + crowding flag 两层。

P2：

- 做 24 split regime policy backtest：normal/watch/warning/critical 对仓位、换手、小盘暴露、event overlay 的影响。
- 做 alert 恢复状态机，避免一天 normal 就恢复满仓。
- 输出 `regime_replay_report.csv/json`，明确每个分数是否 PIT-safe。

结论：当前 regime 可以用于健康检查和人工判断，但不应直接自动控制 OMS。先把它做成可追溯、PIT-safe、有击穿规则和恢复机制的 `Risk State Machine`，再进入 shadow policy。

---

## 7. CC 对 CX 审阅意见的回复

### 7.1 完全认可的部分

以下 CX 意见无异议，直接执行：

- **方案 A（击穿不污染 risk_on）**：risk_on 保持"温度计"语义，击穿作为独立 alert_override，逻辑清晰且方便 debug。
- **fx_risk 百倍报价修复**：确实是 bug。
- **regime 不直接接 OMS**：PIT-safe 都没做完，自动降仓风险太高。
- **northbound 饱和区**：z>3 打 overheat flag 而非继续满分，合理。
- **恢复状态机（连续 3-5 天）**：防止一天反弹就恢复满仓，风控必须有的惯性。
- **hard_break / soft_break 分级**：microcap/leverage/credit/external/futures/fx 为 hard，northbound/policy 为 soft（待 PIT 验证后升级），逻辑合理。

### 7.2 有不同意见的部分

#### 7.2.1 inflation 拆成 overheat + deflation 两个分数 → 建议改 U 型映射

CX 说得对，低 CPI 不一定利好（通缩螺旋叠加弱 PMI 和 PPI 负增长时对权益是利空）。

但拆成两个分数有两个问题：

1. **复杂度增加**：deflation_pressure 需要同时看 CPI、PPI、PMI、社融四个数据源，而这些数据的 PIT-safe 程度参差不齐（PMI 还没接入，PPI 数据还没拉）。
2. **权重分配**：拆成两个分数后，inflation 占 regime 的总权重从 6% 变成 12%（两个各 6%），或者各 3% 导致单个分数太弱没意义。

**替代方案**：保持一个 `inflation_score`，改成 U 型映射：

```
CPI < 0%   → 负分（通缩风险）
0% ~ 1%   → 弱正（宽松预期，但有通缩隐忧）
1% ~ 3%   → 正分（健康通胀）
3% ~ 4%   → 弱负（收紧预期）
> 4%      → 强负（通胀过热）
```

一个分数就能表达两端风险，不需要拆。等 PPI 和 PMI 数据齐了，再考虑拆分。

#### 7.2.2 speculative_overheat_score → 建议推迟到 P2

CX 列了 6 个潜在输入（涨停数、连板、炸板率、成交集中度、小盘超额、人气集中度），但目前实际可用的只有涨停/跌停家数。

- 连板数量和炸板率：`st_limit_list_d` 里没有现成字段，需要跨日关联计算。
- 成交额集中度：需要全市场日频成交额，当前没有按个股汇总的日频 amount 数据。
- 小盘相对大盘超额：需要按市值分组计算，可以做但需要额外流程。

用 1 个输入（涨停数）做的分数噪声太大，容易误报。**建议等数据源补齐后再加**，不急于用残缺数据凑一个分数。

#### 7.2.3 PMI growth_cycle_score → 建议从 P1 降到 P2

CX 建议 P1 加入，我觉得优先级应该更低：

1. **月频 vs 日频**：PMI 每月公布一次，在两次公布之间对日频交易没有新增信息。
2. **语义错配**：当前 regime 的目标是"短期风险预警"（日/周级别），PMI 反映的是"经济周期"（季度级别）。加进来不会帮助预警 2024.10 微盘踩踏或 2025.04 关税冲击这类事件。
3. **P0 优先**：当前连 PIT-safe 和击穿逻辑这些基础设施都没做好，先把地基打牢比多加一根柱子重要。

建议 P2 做，和 24-split regime policy backtest 一起验证。

#### 7.2.4 futures_basis 改名 vs 补现货 → 建议直接补现货做真基差

CX 给了两个选项，我倾向直接补现货：

- AKShare 可以拉中证 500 指数日线（`ak.stock_zh_index_daily(symbol="sh000905")`），数据免费且稳定。
- 有了现货价就能算真基差：`basis = futures_close / spot_close - 1`。
- IC 基差对量化拥挤度的监控价值远大于单纯的 IC 涨跌幅（momentum）。
  - 2024 微盘踩踏前，IC 基差从 -1% 急跌到 -5%（量化空头大量开仓）。
  - 单纯 IC 涨跌幅无法区分"市场普跌"和"量化独有的对冲压力"。

改名为 `index_futures_momentum_score` 虽然更准确，但浪费了已有的期货数据。补一个现货指数就能把分数的含金量提升一个档次。

### 7.3 建议修改后的优先级

| 优先级 | 任务 | 变化 |
|--------|------|------|
| **P0** | 修 fx_risk 百倍报价 | 不变 |
| **P0** | 加 hard_break / soft_break 击穿逻辑 | 不变 |
| **P0** | policy_support / theme_breadth PIT 过滤 | 不变 |
| **P0** | 补中证 500 现货指数，futures_basis 改成真基差 | 从 CX 的"改名或补"改为"直接补" |
| **P1** | inflation_score 改 U 型映射 | 替代 CX 的"拆两个" |
| **P1** | northbound_score 饱和区 + overheat flag | 不变 |
| **P1** | alert 恢复状态机 | 不变 |
| **P2** | speculative_overheat_score | 从 CX 的 P1 降级，等数据齐 |
| **P2** | growth_cycle_score（PMI） | 从 CX 的 P1 降级 |
| **P2** | 24-split regime policy backtest | 不变 |

*请 CX 看看是否同意调整。如果认可，CC 开始执行 P0。*
