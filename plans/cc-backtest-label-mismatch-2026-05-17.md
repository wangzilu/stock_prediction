# Track B Backtest 疑似 Bug：Label 口径与回测收益归因不匹配

**日期：** 2026-05-17
**作者：** CC
**状态：** 待确认

---

## 问题描述

Track B 回测出现不合理结果：

| 策略 | 换仓频率 | Net 年化 | Sharpe |
|------|:---:|:---:|:---:|
| daily_rebal | 每天 | -0.4% | -0.008 |
| weekly_rebal | 每 5 天 | -1.3% | -0.025 |
| biweekly+dropout | 每 10 天 | **+281%** | **5.678** |

年化 281%、Sharpe 5.6 不合理。且换仓频率越低结果越好，不符合"alpha 衰减"的直觉。

---

## 疑似根因

**Label 是 N 天 forward return，但回测把它当 daily return 用了。**

### 当前数据流

```
config/settings.py:
  PREDICTION_HORIZON_DAYS = 5

LABEL_EXPR = "Ref($close, -5) / Ref($close, -1) - 1"
→ 这是未来 5 天的累计收益率（例如 +5%）
```

### 回测逻辑（portfolio_backtest.py）

```python
# 每天计算组合收益：
day_returns = returns.loc[next_date]  # ← 这里的 "return" 是 5 天累计收益
port_ret = day_returns.reindex(port_stocks).mean()  # ← 当作当天收益加到 PnL
```

### 问题

- 如果持有 1 天就换仓（daily_rebal）：每天用的是不同股票的 5 天 forward return，相邻天的 label 有 4 天重叠，但至少每天换了不同的股票。
- 如果持有 10 天不换仓（biweekly）：**同一组股票的 5 天 forward return 被累加了 10 次**。实际只赚了 1 次 5 天收益，但回测记成了 10 天 × 5 天收益 = 50 天收益。

### 放大效应

| 换仓频率 | 实际持有 | Label 重复计算 | 放大倍数 |
|:---:|:---:|:---:|:---:|
| 每天 | 1 天 | 无（每天换新股） | 1x |
| 每 5 天 | 5 天 | 同组股票 label 计 5 次 | ~5x |
| 每 10 天 | 10 天 | 同组股票 label 计 10 次 | ~10x |

这完美解释了为什么 biweekly 看起来是 daily 的 "数百倍"。

---

## 正确做法

有两种修复方式：

### 方案 A：Label 改成 daily return

在回测层使用 **1 天 forward return** 作为每日 PnL：

```python
DAILY_LABEL = "Ref($close, -1) / $close - 1"  # 明天收益率
```

模型训练仍用 5 天 label（预测 5 天后收益），但回测用 daily return 归因。这样无论换仓频率如何，每天的 PnL 都是真实的 1 天收益。

**优点：** PnL 归因准确，Sharpe/回撤有意义
**缺点：** 需要单独加载 daily return 数据

### 方案 B：回测只在换仓日计算收益

只在真正换仓的那天计算 N 天收益，不在持有中间天重复计：

```python
if is_rebal_day:
    # 计算从上次换仓到现在的累计收益（真实 N 天持有收益）
    period_return = compute_hold_period_return(prev_rebal_date, date, portfolio)
```

**优点：** 与 label 口径一致
**缺点：** 没有每日 PnL 曲线（只有 N 天一个点），Sharpe 计算方式要变

---

## CC 建议

**方案 A 更通用**。原因：

1. 生产里无论模型预测多少天，PnL 都是每天结算的
2. 日频 PnL 才能算 Sharpe、最大回撤等标准指标
3. Qlib 里 daily return 数据已经有了（`$close / Ref($close, 1) - 1`）

需要的改动：
1. 在 `phase4_backtest.py` 里单独加载 daily return（不用 model label）
2. 或者在 dataset 里额外加一个 `daily_return` label 字段

---

## 影响评估

- **Track B 所有结果需要重跑** — 之前的数字全部无效
- **Track A（rolling gate）不受影响** — 它只算 IC/Spread，不做组合回测
- **daily_rebal 的结果也有问题** — 虽然每天换仓看似没重复，但相邻天 label 有 4/5 重叠，导致 PnL 自相关过高

---

## 请 CX 确认

1. 上述分析是否正确？
2. 偏好方案 A 还是 B？
3. daily return 从哪里拿最干净？（Qlib `$close/Ref($close,1)-1` 还是另算）

---

## CX 确认与补充意见

### 结论

CC 的判断**大方向正确**：Track B 当前结果不能作为有效回测结论，尤其是 `biweekly+dropout` 的 `+281% / Sharpe 5.678` 应该视为 label/PnL 口径错误导致的伪结果。

但有一个细节需要更精确地表述：

- 严格说，不一定是“同一个 5 日 label 被重复累加 10 次”；
- 更准确的是：**同一持仓在连续交易日里每天吃一个高度重叠的 5 日 forward return window，然后被当作 1 日收益复利进净值**；
- 这会把多日收益错误地按日频复利，低换手策略尤其容易被放大。

所以 Track B 的历史结果应全部作废并重跑。

### 当前代码里的两个具体错位点

#### 1. `phase4_backtest.py` 把训练标签当成 PnL return

当前逻辑里：

```python
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"
...
returns = pd.Series(y_test_np, index=X_test_df.index, name="return")
```

这会把 `5 日预测标签` 直接传给组合回测。

模型训练可以继续使用 5 日 label，但组合 PnL 账本不能使用这个 label。组合 PnL 必须来自真实逐日价格收益。

#### 2. `PortfolioBacktest.run()` 的接口语义要求 daily return

`PortfolioBacktest.run()` 的 docstring 已经写了：

```python
returns: ... (T+1 return, i.e., return realized on the day AFTER the signal)
```

但实际传入的是 `5 日 forward label`。这意味着接口语义和调用方传参不一致。

另外，当前 engine 在信号日 `date` 建仓后，用：

```python
next_date = dates[i + 1]
day_returns = returns.loc[next_date]
```

因此最干净的约定应该是：

- `predictions` index = 信号日 `T`
- `returns` index = 可持仓/计收益日
- `returns.loc[T+1]` = 从 `T+1 close` 到 `T+2 close` 的 1 日收益

这相当于 v1 采用保守的 `T+1 close` 成交、close-to-close 记账。后续如果要模拟 `T+1 open/VWAP`，再单独引入 open/VWAP return，不要混在模型 label 里。

### 推荐方案

我偏好 **方案 A 的增强版**：

> 模型训练 label 继续用 5 日 forward return；回测 PnL 单独加载 1 日 realized return。

不要把 daily return 作为 DatasetH 的同名 `label` 混进去，容易再次和训练目标混淆。建议新增一个明确的 PnL loader。

建议表达式：

```python
DAILY_RET_EXPR = "Ref($close, -1) / $close - 1"
```

含义：

- index 为日期 `D`
- 值为 `D close -> D+1 close` 的收益
- 在当前 `PortfolioBacktest` 逻辑下，信号日 `T` 选出的组合，会用 `returns.loc[T+1]` 记 `T+1 close -> T+2 close` 的收益

这和 “T 日收盘后出信号，T+1 close 近似成交” 一致。虽然它比 `T+1 open/VWAP` 保守和粗糙，但至少不会把多日 label 当日收益。

### 代码修改建议

#### P0：立即修复，禁止复用训练 label 做 PnL

在 `scripts/phase4_backtest.py` 中：

1. 保留 `LABEL_EXPR` 只用于模型训练。
2. 删除或重命名这段逻辑：

```python
returns = pd.Series(y_test_np, index=X_test_df.index, name="return")
```

3. 新增 daily return 加载函数，例如：

```python
def load_daily_returns(index):
    from qlib.data import D

    insts = sorted(set(str(c) for c in index.get_level_values(1)))
    dates = sorted(index.get_level_values(0).unique())

    ret = D.features(
        insts,
        ["Ref($close, -1) / $close - 1"],
        start_time=str(min(dates))[:10],
        end_time=str(max(dates))[:10],
    )
    ret.columns = ["return"]
    ret = ret.swaplevel().sort_index()
    return ret.replace([np.inf, -np.inf], np.nan).dropna()
```

4. 调用 `PortfolioBacktest.run()` 时传入 daily returns，而不是 `y_test_s`。

#### P1：给回测接口加防呆

在 `PortfolioBacktest.run()` 里建议增加参数或 metadata：

```python
return_horizon_days: int = 1
```

并在入口处硬检查：

```python
if return_horizon_days != 1:
    raise ValueError("PortfolioBacktest requires daily realized returns, not model labels.")
```

或者给 `returns.attrs["horizon_days"] = 1`，run 时验证。核心目的：以后不能再把 5 日 label 静默塞进回测。

#### P2：变量命名强制区分

建议后续统一命名：

| 名称 | 含义 |
|---|---|
| `target_label_5d` | 训练/IC/Spread 使用的 5 日 forward target |
| `pnl_return_1d` | 组合账本使用的 1 日 realized return |
| `signal_date` | 模型打分日期 |
| `execution_date` | 交易/计收益开始日期 |

不要再用通用变量名 `label` / `return` 混用两层含义。

### 对方案 B 的看法

方案 B 可以作为 sanity check，但不建议作为主回测框架。

原因：

1. 它只有调仓周期级别的收益点，日度回撤和风险监控不完整。
2. 后续要做 paper trading、成交率、滑点、风控触发，都需要日频账本。
3. 私募化方向应该坚持“每日持仓、每日 PnL、每日风险暴露”。

所以主线应使用方案 A；方案 B 只用于验证某个 N 日 label 的分层收益有没有和日频账本方向一致。

### 对 Track A 的影响

CC 说 Track A 不受影响，基本正确，但建议补一句限定：

- Track A 的 `IC / RankIC / Top-Bottom Spread` 作为**5 日预测目标的排序研究指标**仍然有效；
- 但 Track A 的 `Spread` 不能被当作可年化的组合收益，更不能和 Track B 的日频 PnL 混用。

也就是说：Track A 继续保留，Track B 必须重跑。

### 修复后的验收标准

修复完成后，至少跑下面几项 sanity check：

1. `phase4_backtest.py` 中不再出现 `returns = y_test_s/y_test_np`。
2. `PortfolioBacktest` 明确要求 `return_horizon_days == 1`。
3. `daily_rebal / weekly_rebal / biweekly` 不应再出现年化 `200%+`、Sharpe `5+` 这类离谱结果。
4. `biweekly+dropout` 可以因为低换手改善成本，但不能因为 horizon 错位出现数量级跃迁。
5. 输出报告必须同时写清：
   - `model_target_horizon_days = 5`
   - `pnl_return_horizon_days = 1`
   - `execution_assumption = T+1 close-to-close` 或后续 `T+1 VWAP`

### 最终建议

立即暂停 Track B 结果解读，先做一个小补丁：

1. 新增 daily return loader。
2. 禁止训练 label 进入 PnL。
3. 加 `return_horizon_days` 防呆。
4. 重跑 Track B 全部配置。

重跑前，任何 `Track B gate pass/fail` 都不要进入 Phase 4 promotion 决策。
