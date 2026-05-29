# Qlib 高级功能后续实施建议

日期：2026-05-09

主执行入口：`plans/cx-v2-iteration-plan.md`。本文件是 Qlib 高级功能的详细附录，保存指标定义、API 校验和 cc/cx 分歧证据；具体排期以 V2 计划的 Phase 3 和“Current Implementation Order”为准。

## 结论

当前项目已经把 Qlib 的数据、Alpha158、LightGBM 全 A 训练、生产预测缓存跑通了，但仍然主要停留在“能训练、能预测”的阶段。Qlib 更高级、也更应该优先补上的能力不是先换 Transformer，而是先补三件事：

1. 用 IC、RankIC、分层收益、回测收益证明预测分数是否真的有交易价值。
2. 用 Qlib 回测和组合策略把“个股分数”变成可执行的持仓组合。
3. 用 Recorder、Rolling、模型对照实验让模型训练可追溯、可比较、可迭代。

今晚全 A LGB 的复查结果说明了更细的事实：工程链路健康，模型分数不是随机数；但交易有效性还没有通过正式回测、成本和执行约束证明。

| 评估口径 | 样本/日期 | IC | ICIR | RankIC | Top/Bottom 分层 |
|---|---:|---:|---:|---:|---:|
| `ret1 = Ref($close, -2)/Ref($close, -1)-1` | 83,180 / 16 日 | 0.0452 | 0.8105 | 0.0184 | Top20-Bot20 +2.740% |
| `ret5 = Ref($close, -5)/Ref($close, -1)-1` | 67,572 / 13 日 | 0.0329 | 0.7676 | 0.0023 | Top20-Bot20 +6.659% |

这意味着：当前模型有 IC 信号，Top20/Bottom20 极端分层在最近样本上很强；但广义截面排序仍弱，因为 RankIC 明显低于 0.03。下一步要先建设评估和回测闭环，把这个 TopK 信号放进成本、换手、涨跌停、停牌、T+1/settlement 和基准收益框架里验证，再谈更复杂模型。

## 当前已用和未用的 Qlib 能力

| 层级 | 当前已用 | 还没用上 | 优先级 |
|---|---|---|---|
| 数据 | Qlib bin、calendar、instruments、全 A daily bars | Qlib Dataset Cache、自动任务缓存、更多字段/行业特征 | 中 |
| 特征 | Alpha158 | Alpha360、自定义表达式、行业/市值/流动性中性化特征 | 中 |
| 模型 | LGBModel | XGBModel、CatBoostModel、DoubleEnsemble、LSTM、GRU、ALSTM、Transformer、HIST、TabNet | 中高 |
| 评估 | 自写健康检查、方向命中率/IC 手算 | Qlib Recorder、SignalRecord、PortAnaRecord、`calc_ic`、`calc_long_short_return`、risk_analysis、indicator_analysis | 最高 |
| 回测 | 暂无正式 Qlib 回测 | Qlib backtest core、Exchange、Account、手续费/滑点/涨跌停约束、TopK/TopK-dropout 组合策略 | 最高 |
| 训练方式 | 单次滑窗训练 | RollingGen、滚动训练、模型版本对照 | 高 |
| 组合 | 推送 top candidates | TopK 持仓、换手约束、组合收益/回撤/换手统计 | 高 |
| 实验管理 | 覆盖写 `lgb_model.pkl` | MLflow Recorder、artifact、参数/指标留痕 | 高 |
| 线上治理 | smoke + cache | OnlineManager、模型注册、灰度/回滚 | 中 |

## 实施原则

1. 先评估，后升级模型。  
   如果没有 IC、RankIC、回测收益和交易成本，换再高级的模型也只是更贵的随机数。

2. 先横截面排序，再方向准确率。  
   A 股日频选股更关心“谁比谁更好”，不是每只股票涨跌都猜对。方向命中率可以保留，但不能作为唯一门槛。

3. 每个模型必须经过同一套 test/rolling/backtest。  
   LightGBM、CatBoost、XGBoost、DoubleEnsemble、Transformer 都必须输出同格式预测，用同一套脚本比较。

4. 生产推送只用通过门禁的模型。  
   如果 latest model 的 RankIC、回测 Sharpe、覆盖率不达标，继续使用上一版模型或降级到缓存。

## Phase 0：把评估指标固化为脚本

目标：每天训练后自动输出真实测试指标，避免只看“5204 finite predictions”，也避免混用 1 日标签和 5 日产品口径。

新增脚本：

- `scripts/evaluate_lgb_test.py`

输入：

- `data/storage/lgb_model.pkl`
- `data/storage/qlib_data/cn_data`
- universe=`all`
- 与 `scripts/train_lgb.py` 相同的 train/valid/test 日期切分

输出：

- `data/storage/lgb_eval_latest.json`
- `data/storage/lgb_eval_history.json`

至少包含：

| 指标 | 说明 | 建议门槛 |
|---|---|---:|
| finite_prediction_count | 最新交易日有效预测数 | >= 4500 |
| label_expression | 本次评估标签表达式 | 必填 |
| direction_accuracy | 真实涨跌方向命中率 | 仅观察，不作为强门槛 |
| daily_ic_mean | 日均 IC | > 0.03 |
| daily_rank_ic_mean | 日均 RankIC | > 0.03 |
| top20_bottom20_spread | Top20 - Bottom20 真实收益差 | > 0 |
| broad_quantile_spread | Top10% - Bottom10% 真实收益差 | > 0 |
| top10_positive_rate | Top10% 上涨比例 | > 全市场上涨比例 |
| n_test_days | 测试交易日数 | `ret1 >= 15`；`ret5` 可因未来 5 日标签变成 13 |

建议命令：

```bash
/Users/wangzilu/miniconda3/envs/tianshou/bin/python scripts/evaluate_lgb_test.py \
  --model-path data/storage/lgb_model.pkl \
  --universe all \
  --min-predictions 4500
```

接入点：

- `scripts/train_lgb.py` 训练完成、prediction health 通过后调用。
- `scripts/smoke_lgb_predict.py` 只负责生产推理健康，不负责模型质量。
- `scripts/nightly_train.py` 在 LGB smoke 后追加 evaluation step。

验收：

- 每次训练都有一条 JSON 记录。
- 如果 RankIC 或 top-bottom spread 连续 3 次低于门槛，在 `job_status.json` 标记为 `degraded_quality`，但不阻止数据更新。

## Phase 1：接入 Qlib 正式回测

目标：把模型预测从“分数”变成“组合收益曲线”。

新增脚本：

- `scripts/backtest_qlib_signal.py`

核心功能：

1. 加载模型在 test/rolling 区间的预测分数。
2. 使用本地 TopK/TopK-dropout 策略或修复后的 `qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy` 构建组合。
3. 使用 Qlib backtest 模拟手续费、滑点、持仓、换手。
4. 输出组合收益、基准收益、超额收益、最大回撤、Sharpe、换手率。

默认参数建议：

| 参数 | 建议值 |
|---|---:|
| topk | 20 |
| n_drop | 5 |
| account | 1,000,000 |
| benchmark | SH000300 或全 A 等权自建基准 |
| open_cost | 0.0005 |
| close_cost | 0.0015 |
| min_cost | 5 |
| limit_threshold | 0.095 |

输出：

- `data/storage/lgb_backtest_latest.json`
- `data/storage/lgb_backtest_report.csv`
- `data/storage/lgb_backtest_curve.csv`

门槛建议：

| 指标 | 最低门槛 | 理想门槛 |
|---|---:|---:|
| excess_return | > 0 | > 5%/year |
| information_ratio | > 0.3 | > 0.8 |
| max_drawdown | < 20% | < 12% |
| average_turnover | < 30%/day | < 15%/day |
| win_days_ratio | > 50% | > 53% |

接入推送：

- 22:00 晚间展望不要只写“模型分最高前五”。
- 增加一行模型状态：

```text
短线模型：覆盖5204只，近16日RankIC=0.018，Top-Bottom=-0.008%，质量偏弱，今日建议降低仓位。
```

如果回测质量达标，再展示：

```text
短线模型：近3个月年化超额+X%，IR=Y，最大回撤-Z%，质量正常。
```

## Phase 2：使用 Recorder 管理实验

目标：不再只覆盖 `lgb_model.pkl`，而是保留每次训练的参数、数据范围、指标和模型。

改造文件：

- `scripts/train_lgb.py`
- `scripts/evaluate_lgb_test.py`
- `scripts/backtest_qlib_signal.py`

新增目录：

- `data/storage/model_registry/`

建议结构：

```text
data/storage/model_registry/
  2026-05-09_202036_lgb_all/
    model.pkl
    dataset_config.json
    train_config.json
    eval.json
    backtest.json
    predictions.parquet
```

Qlib Recorder 用法：

```python
from qlib.workflow import R

with R.start(experiment_name="lgb_all_alpha158"):
    R.log_params(**model_config["kwargs"])
    R.log_params(
        universe="all",
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        test_start=test_start,
        test_end=test_end,
    )
    model.fit(dataset)
    R.save_objects(**{"model.pkl": model})
    R.log_metrics(
        daily_ic_mean=daily_ic_mean,
        daily_rank_ic_mean=daily_rank_ic_mean,
        top_bottom_spread=top_bottom_spread,
    )
```

生产模型选择规则：

1. 新模型训练成功。
2. smoke 通过。
3. evaluation 通过最低门槛。
4. backtest 不显著劣于当前 production model。
5. 满足以上条件才更新 `data/storage/lgb_model.pkl` 和 `lgb_latest_predictions.json`。

## Phase 3：模型对照和 Ensemble

目标：建立模型竞赛，而不是凭感觉换模型。

新增脚本：

- `scripts/train_model_suite.py`

首批模型：

| 模型 | Qlib class | 用途 |
|---|---|---|
| LightGBM | `qlib.contrib.model.gbdt.LGBModel` | 当前基线 |
| XGBoost | `qlib.contrib.model.xgboost.XGBModel` | 树模型对照 |
| CatBoost | `qlib.contrib.model.catboost_model.CatBoostModel` | 类别/非线性鲁棒性对照 |
| DoubleEnsemble | `qlib.contrib.model.double_ensemble.DEnsembleModel` | Qlib 内置增强基线 |

暂不优先上：

| 模型 | 原因 |
|---|---|
| Transformer / Localformer | 更适合 Alpha360/序列输入，先补回测再做 |
| HIST / IGMTF | 需要行业/图结构和更细的调参，工程成本高 |
| TabNet | 可试，但优先级低于树模型 ensemble |

Ensemble 方案：

```python
ensemble_pred = (
    0.50 * lgb_pred.rank(pct=True)
    + 0.25 * xgb_pred.rank(pct=True)
    + 0.25 * cat_pred.rank(pct=True)
)
```

注意：不要直接平均 raw prediction，因为不同模型输出尺度不同。先做截面 rank 或 z-score，再 ensemble。

验收：

- `ensemble_rank_ic_mean` 比 LGB 单模型提升至少 10%。
- Top-Bottom spread 必须为正。
- 回测 IR 不低于 LGB。

## Phase 4：Rolling 训练和走步回测

目标：避免固定窗口测试太短、过于偶然。

新增脚本：

- `scripts/rolling_train_qlib.py`

建议配置：

| 参数 | 建议值 |
|---|---:|
| train_window | 3 年 |
| valid_window | 60 交易日 |
| test_window | 20 交易日 |
| step | 20 交易日 |
| rolling_start | 2023-01-01 |
| rolling_end | 最新交易日 |

产物：

- `data/storage/rolling/lgb_alpha158_all/predictions.parquet`
- `data/storage/rolling/lgb_alpha158_all/eval.json`
- `data/storage/rolling/lgb_alpha158_all/backtest.json`

验收：

- 至少 24 个 rolling test window。
- rolling RankIC 均值 > 0.03。
- RankIC 正值窗口比例 > 55%。
- Top-Bottom spread 正值窗口比例 > 55%。
- 回测净值曲线不能只靠少数几天贡献。

接入生产：

- 当前每日训练仍保留。
- 每周末或每月运行 rolling 评估。
- 如果 rolling 评估连续恶化，降低推送里的模型权重。

## Phase 5：Alpha360 和深度模型试验

目标：在已经有评估/回测标尺后，并行试更复杂模型。cc 对 MPS 的新论据已在本机复现，深度模型不再因为“只能 CPU 跑”而推迟；但仍不能绕过统一评估、回测、registry 和生产门禁。

新增脚本：

- `scripts/train_alpha360_baseline.py`
- `scripts/train_deep_model_suite.py`

试验顺序：

1. `Alpha360 + LightGBM`
2. `Alpha360 + GRU`
3. `Alpha360 + ALSTM`
4. `Alpha360 + Transformer`

验收方式：

- 全部走 Phase 0 到 Phase 1 的同一评估和回测。
- 不以训练 loss 判断好坏。
- 如果深度模型 RankIC 或回测不如树模型 ensemble，不进入生产。

注意事项：

- PyTorch 模型优先使用 MPS，并在日志中记录 device、wall time、随机种子、数据窗口和内存压力。
- 深度模型可以进入 after-close 后的研究/对照任务，但不作为生产关键路径阻塞当天推送。
- 必须保存训练日志和随机种子，避免结果不可复现。

## Phase 6：组合和风险约束

目标：让推荐从“买哪些”升级为“买多少、持多久、何时换”。

新增脚本：

- `scripts/build_qlib_portfolio.py`

推荐组合逻辑：

1. 使用模型预测分数生成候选池。
2. 用 TopK + Dropout 控制换手。
3. 过滤停牌、涨停不可买、跌停不可卖、成交额不足。
4. 加入行业集中度限制。
5. 输出目标权重和换仓清单。

输出：

- `data/storage/portfolio/latest_target_weights.csv`
- `data/storage/portfolio/latest_rebalance.json`

示例推送：

```text
短线组合：目标持仓20只，今日换入5只、换出4只，预计换手25%。
模型质量：RankIC=0.041，近3月IR=0.72，状态正常。
风险约束：单票上限8%，单行业上限25%，现金保留10%。
```

## Phase 7：重构每日管线

当前 crontab 中 17:00 数据更新、17:35 LGB 训练、17:55 smoke 是独立任务。真实运行中全 A 数据更新可能超过 35 分钟，存在训练用旧数据或半更新数据的风险。

建议新增：

- `scripts/after_close_pipeline.py`

串行顺序：

```text
1. update_qlib_data.py --end-date 最新交易日
2. check_qlib_data_health.py
3. train_lgb.py
4. smoke_lgb_predict.py
5. evaluate_lgb_test.py
6. backtest_qlib_signal.py
7. 若质量达标，promote model registry -> production model
8. refresh evening/morning prediction cache
```

替代 crontab：

```text
17:00 after_close_pipeline.py
22:00 main.py --evening-outlook
04:00 nightly_train.py 只跑研究型 RL/rolling，不覆盖生产模型
```

额外修正：

- 手动补跑时默认使用 Qlib calendar 的最新交易日，不使用自然日。
- 周六、周日、节假日运行时必须自动回退到最新交易日。
- 如果数据更新失败，不训练新模型。
- 如果模型质量失败，不覆盖生产模型。

## 推荐优先级

### 第 1 周

1. 新增 `evaluate_lgb_test.py`。
2. 在 `train_lgb.py` 后接 evaluation。
3. 新增 `backtest_qlib_signal.py`。
4. 推送内容增加模型质量状态。

验收：能回答“这个模型最近 16 天 ret1、13 天 ret5、以及 60 天滚动窗口有没有有效排序/分层能力”，而不是只回答“能不能输出 5204 个分数”。

### 第 2 周

1. 接入 Recorder/model registry。
2. 训练 LGB/XGB/CatBoost/DoubleEnsemble 对照。
3. 同步启动 Alpha360/GRU/ALSTM/Transformer 的 MPS 对照实验。
4. 用 rank ensemble 生成候选预测。

验收：至少有一组模型或 ensemble 的 RankIC、Top-Bottom spread、回测 IR 明显优于当前 LGB。

### 第 3 周

1. 做 rolling train/backtest。
2. 用 rolling 结果决定是否进入生产。
3. 改 after-close pipeline 为串行依赖。

验收：生产模型不再只由单次训练决定，而由 rolling 质量和最新训练质量共同决定。

### 第 4 周以后

1. rolling train/backtest。
2. 行业/市值/流动性中性化。
3. 组合优化和风险预算。
4. 深度模型若已在第 2-3 周跑赢树模型 ensemble，再进入生产候选。

验收：复杂模型必须在统一回测框架下超过树模型 ensemble，否则只保留为研究结果。

## 不建议马上做的事

1. 不建议现在直接把 Transformer 接进生产。  
   当前瓶颈是评估/回测，不是模型复杂度。

2. 不建议只看方向准确率。  
   今晚结果显示方向命中率 51.06%，但全市场上涨比例 52.84%。这个指标单独看会误导。

3. 不建议继续把“短线模型分”包装成中线/长线结论。  
   当前 LGB 预测的是短期 forward return，长线推荐需要单独标签和单独回测。

4. 不建议模型训练成功就覆盖生产。  
   必须通过 smoke、evaluation、backtest 三道门。

## 对 cc roadmap 的采纳、修正和证据

项目里已有一份 `plans/cc-qlib-advanced-features-roadmap.md`。我重新按本机 Qlib 0.9.7、当前数据目录和依赖环境逐项校验后，结论是：cc 文档的路线方向大体正确，尤其是“先补评估和回测，再谈更复杂模型”。cc 后续追加的 2026-05-09 TopK/IC 实测也已在本地复现，应作为有效证据纳入计划。分歧只保留在能用本地 API 签名、源码、实际导入结果或可复现实测证明的地方。

### 可以采纳的判断

1. 当前最大缺口是评估和回测，而不是模型复杂度。
2. `Alpha158 + LGBModel + DatasetH` 只是 Qlib 的基础用法。
3. TopK/TopK-dropout 思路应该成为从“个股推荐”升级到“组合持仓”的核心策略；当前环境暂不能直接依赖 Qlib contrib 的 `TopkDropoutStrategy` 导入路径。
4. `Recorder`/experiment tracking 必须补上，否则模型训练不可追溯。
5. Rolling 训练是从研究脚本走向生产系统的关键步骤。
6. CatBoost、XGBoost、DoubleEnsemble、Alpha360、GRU、ALSTM、Transformer 都可以进入同一轮模型对照；MPS 可用后，不必因为速度假设把深度模型推到很后面。
7. 深度模型不应绕过评价、回测、rolling 和模型注册；只有跑赢树模型 ensemble 后才进入生产候选。

### 逐项校验结论

| cc 文档位置 | cc 主张 | 本地校验结论 | 执行口径 |
|---|---|---|---|
| lines 27-54 | 用 IC/RankIC 做因子诊断 | 方向正确，但代码错。`risk_analysis(r)` 的参数是收益序列，不是预测分数；`qlib.contrib.eva.alpha.alpha_analysis` 在本地不存在。 | 用 `qlib.contrib.eva.alpha.calc_ic(pred, label)`、`calc_long_short_return(pred, label)`，或用 `SignalRecord` 生成 IC/RankIC。 |
| lines 60-99 | 用 Qlib backtest 替代 `change_pct` 验证 | 方向正确，但示例缺少必填 `executor`；`benchmark="SH000300"` 当前也不可直接用。 | `scripts/backtest_qlib_signal.py` 必须显式配置 `SimulatorExecutor(time_per_step="day")`；基准先用全 A 等权或补指数数据后再用沪深 300。 |
| lines 103-132 | LGB/CatBoost/XGBoost ensemble | 方向正确，本地 `xgboost`、`catboost` 和 Qlib 模型模块可导入。 | 不直接平均 raw prediction；先做截面 rank 或 z-score，再加权合成。 |
| lines 138-171 | Rolling 训练 | 方向正确，但示例里 `risk_analysis(pd.concat(all_preds))` 仍然把预测分数当收益。 | rolling 输出的 out-of-sample prediction 进入 `calc_ic` 和 backtest，而不是直接进 `risk_analysis`。 |
| lines 175-196 | TopkDropoutStrategy | 策略思想可以采纳，但 cc 对当前环境的导入风险判断成立：`signal_strategy.py` 顶部导入 `EnhancedIndexingOptimizer`，进而导入 `cvxpy`，当前会因 `numpy.lib.array_utils` 缺失失败。 | 初版用本地 TopK/TopK-dropout 等价实现接 Qlib backtest core；修复 cvxpy/numpy 后再切回 Qlib contrib 类。 |
| lines 200-214 | Alpha360 | 可以采纳为实验。`Alpha360` 类本地存在。 | 先用 Alpha158 评估/回测闭环；再做 Alpha360 对照，防止把特征、模型、回测同时改动导致无法归因。 |
| lines 220-238 | Recorder | 可以采纳，但 `R.log_metrics(**risk_analysis(pred))` 仍然是预测分数误用。 | 用 `SignalRecord` 记录 `pred.pkl`/`label.pkl`，用 `SigAnaRecord`/`PortAnaRecord` 或自写 evaluate 脚本记录指标。 |
| lines 242-260 | PortfolioOptimizer | cc 示例在本地不可运行。真实签名没有 `max_weight`/`min_weight`，调用方式是 `optimizer(S, r, w0)` 而不是 `.optimize(...)`。当前导入还会被 `cvxpy`/`numpy` 冲突挡住。 | 组合优化不进第 1 周。先用等权 TopK/本地 TopK-dropout 建回测闭环，再单独修依赖和约束优化。 |
| lines 264-293 | Transformer/ALSTM/TRA/HIST | 模块方向可以，本地 Transformer 模块可导入，且 MPS 已复现可用。 | 评估/回测脚本完成后可与树模型并行对照；但不先接生产，必须跑赢基线且通过成本回测。 |
| lines 351-394 | 外部库大多可由 Qlib 内置替代 | 结论需要更细。Qlib 的确优先，但“Qlib PortfolioOptimizer 避免 cvxpy 冲突”在当前环境为假；“Qlib 不支持停牌跳过”也不准确。 | 外部库暂不扩张。Qlib first，但 RQAlpha/QuantStats/vectorbt 保留为后续可选工具，不在第一阶段引入。 |
| lines 682-718 | cc 实测 IC/RankIC/Top20 | 本地已复现。`ret5` 对齐后得到 `67,572` 样本、13 日、`IC=0.0329`、`ICIR=0.7676`、`RankIC=0.0023`、`Top20-Bot20=+6.659%`。 | 接受“Top20 极端分层有效”的证据；同时保留“广义排序弱、未经过成本/执行/回测”的生产门槛。 |

### 本地证据记录

1. `risk_analysis` 真实签名：

```python
risk_analysis(r, N=None, freq="day", mode="sum")
```

它分析的是收益序列。IC/RankIC 的正确本地 Qlib API 是：

```python
from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return

ic, rank_ic = calc_ic(pred, label)
long_short_r, long_avg_r = calc_long_short_return(pred, label)
```

本地 `qlib.contrib.eva.alpha` 可用函数包括 `calc_ic`、`calc_long_short_return`、`calc_long_short_prec`；没有 `alpha_analysis`。

2. `backtest` 真实签名：

```python
backtest(start_time, end_time, strategy, executor, benchmark="SH000300", ...)
```

因此 cc 示例的 `qlib_backtest(strategy=strategy_config, **backtest_config)` 少了 `executor`。本地日频执行器可用：

```python
SimulatorExecutor(time_per_step="day", trade_type="serial")
```

3. 当前 Qlib 数据目录没有 `SH000300`：

```text
data/storage/qlib_data/cn_data/features/sh000300  不存在
data/storage/qlib_data/cn_data/features/SH000300  不存在
instruments 下也没有 sh000300 记录
```

4. `PortfolioOptimizer` 真实源码支持 `gmv`、`mvo`、`rp`、`inv`，构造签名是：

```python
PortfolioOptimizer(method="inv", lamb=0, delta=0, alpha=0.0, scale_return=True, tol=1e-8)
```

它没有 `max_weight`、`min_weight` 参数，也没有 `.optimize(...)` 方法；主调用入口是 `__call__(S, r=None, w0=None)`。当前环境导入会报：

```text
ModuleNotFoundError: No module named 'numpy.lib.array_utils'
```

依赖证据：当前 `numpy==1.26.4`，`cvxpy==1.8.2` 的 metadata 要求 `numpy>=2.0.0`。Qlib 的 `optimizer/__init__.py` 会同时导入 `EnhancedIndexingOptimizer`，而 `enhanced_indexing.py` 顶部 `import cvxpy as cp`，所以“改用 Qlib PortfolioOptimizer 就能避开 cvxpy/numpy 冲突”在当前环境不成立。

补充校验：`from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy` 在当前环境也会走到同一个 `EnhancedIndexingOptimizer`/`cvxpy` 导入链并失败。因此 Phase 1 不能直接照搬 Qlib contrib 的 `TopkDropoutStrategy`；应该先实现一个本地 TopK/TopK-dropout 策略接 Qlib backtest core，或者先修复依赖环境。

5. Qlib 回测确实支持不少 A 股交易约束。`Exchange` 源码里有：

```python
limit_threshold
volume_threshold
open_cost
close_cost
min_cost
impact_cost
trade_unit
check_stock_suspended(...)
is_stock_tradable(...)
```

并且 `$close` 为 NaN 时会被视为停牌/不可交易。因此 cc 文档说 Qlib 已支持涨跌停、手续费、滑点这部分是对的；说“不支持停牌跳过”不准确。T+1 是否完整模拟仍需要单独验证，不能在第一阶段承诺。

6. 模型和特征可用性：

```text
Alpha360 class exists
XGBModel import OK
CatBoostModel import OK
DEnsembleModel import OK
TransformerModel import OK
```

所以模型升级路线可以保留，但排序应服从评估和回测闭环。

7. cc TopK/IC 实测复现：

本地用当前 `data/storage/lgb_model.pkl`、`data/storage/qlib_data/cn_data`，重建 `Alpha158` test dataset（不能直接依赖 `lgb_dataset.pkl`，因为 pickle handler reload 后缺 `_infer/_learn`），得到：

```text
prediction rows: 93,587 finite rows, 18 trading dates
ret5 aligned rows: 67,572, 13 dates
ret5 IC mean: 0.032923
ret5 ICIR: 0.767551
ret5 RankIC mean: 0.002334
ret5 RankIC > 0 ratio: 53.85%
ret5 Top20 avg: +4.145%
ret5 Bot20 avg: -2.513%
ret5 Top20-Bot20 spread: +6.659%
ret5 universe avg: +1.300%
ret5 spread > 0 ratio: 84.6%
```

同时，`ret1 = Ref($close, -2)/Ref($close, -1)-1` 是 `scripts/train_lgb.py` 当前未显式覆盖标签时的 Alpha158 默认口径，得到 `83,180` 样本、16 日、`IC=0.045188`、`RankIC=0.018402`、`Top20-Bot20=+2.740%`。因此 `evaluate_lgb_test.py` 必须把 label expression 写进 JSON 和推送摘要，否则“16 日/83,180 样本”和“13 日/67,572 样本”会被混为一谈。

8. cc MPS 实测复现：

本地按 cc 提供的 `tianshou` Python 路径复查，结论是 cc 对 MPS 的反驳成立：

```text
python: /Users/wangzilu/miniconda3/envs/tianshou/bin/python
macOS: 26.3
torch: 2.11.0
mps_available: True
mps_built: True
MPS tensor creation: SUCCESS
TransformerEncoder batch benchmark:
  CPU: 564.0 ms/batch
  MPS: 43.9 ms/batch
  speedup: 12.8x
```

因此原先“当前环境 MPS 不可用，所以深度模型只能 smoke/small-sample”的论据撤销。新的执行口径是：评估/回测门禁仍先做；门禁完成后，深度模型可和树模型并行对照，是否进入生产只看统一评估、成本回测、rolling 和 registry 结果。

### 外部库口径修正

| 需求 | cc 倾向 | 本文修正后的口径 |
|---|---|---|
| 因子诊断 / IC | Qlib 内置替代 Alphalens | 同意第一阶段不用 Alphalens，但不是用 `risk_analysis(pred)`，而是用 `calc_ic`、`calc_long_short_return`、`SignalRecord`。 |
| 回测 | Qlib 内置为主，vectorbt/RQAlpha 辅助 | 同意 Qlib first。vectorbt 只在大规模参数扫描时考虑；RQAlpha 只在确认 T+1/交易规则成为瓶颈时考虑。 |
| 组合优化 | Qlib PortfolioOptimizer 替代 PyPortfolioOpt | 方向上同意暂不引入 PyPortfolioOpt，但当前 Qlib PortfolioOptimizer 也被 cvxpy/numpy 冲突影响，不能放进短期计划。 |
| 绩效报告 | Qlib report + 可选 QuantStats | 同意。先用 `risk_analysis`、`PortAnaRecord` 和本地评价脚本产出核心指标，QuantStats 只作为展示层，不作为核心依赖。 |
| 风险模型 | Qlib riskmodel 替代 Riskfolio-Lib | 基本同意第一阶段不用 Riskfolio-Lib。Qlib 有 `ShrinkCovEstimator`、`StructuredCovEstimator`、`POETCovEstimator`，但 CVaR/复杂约束不是当前优先级。 |
| A 股交易规则 | Qlib 不完全支持，RQAlpha 有价值 | 同意 RQAlpha 有未来价值；但 Qlib 已支持停牌、涨跌停、费用、冲击成本、成交量约束和交易单位。T+1 单独验证后再决定。 |

### 执行口径

后续实施时：

- 以本文件作为执行计划。
- 以 `cc-qlib-advanced-features-roadmap.md` 作为功能地图和灵感来源。
- cc 文档里的代码块必须先在本地 Qlib 0.9.7 环境验证 API 后再写入生产脚本。
- 第一阶段仍然只做 `evaluate_lgb_test.py`、`backtest_qlib_signal.py`、`after_close_pipeline.py` 三个最小闭环脚本。
- 所有模型升级都必须复用同一套 evaluation、backtest、rolling、registry 标准。

## 最小可执行下一步

最小闭环只需要三个脚本：

```text
scripts/evaluate_lgb_test.py
scripts/backtest_qlib_signal.py
scripts/after_close_pipeline.py
```

完成这三个以后，项目会从：

```text
数据健康 -> 训练 -> 能预测
```

升级为：

```text
数据健康 -> 训练 -> 能预测 -> 有统计信号 -> 有回测收益 -> 才推广生产
```

这才是后续所有高级模型和高级 Qlib 功能值得接入的地基。
