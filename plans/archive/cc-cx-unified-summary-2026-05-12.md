# CC/CX 开发计划综合总结

**日期：** 2026-05-12
**范围：** 综合 `plans/` 下全部 cc 和 cx 文档（共 13 份），提取共识、分歧解决、已完成工作和下一步路线。

---

## 一、项目定位

A 股量化预测系统，从数据采集 → Alpha158 + 多源因子 → LGB/XGB/深度模型 → 日频推荐 → WeChat 推送的全链路。目标不是复制百亿私募，而是建立**机构级纪律的小型量化研究与推荐平台**。

---

## 二、CC 和 CX 的角色分工

| 角色 | 偏好 | 核心贡献 |
|------|------|---------|
| **CC** | "先跑通再优化"，偏实战快速验证 | 识别资金流/北向高 alpha 因子、MPS GPU 验证、TopK 回测实测、快速 prototype |
| **CX** | "先设计再实现"，偏工程严谨性 | 全 A 数据覆盖、PIT 对齐审计、因子消融实验设计、Qlib API 校验、cx-v2 主执行计划 |

两者经过多轮辩论后在所有技术分歧上达成共识（见第五节）。

---

## 三、当前系统能力全景

### 已完成（截至 2026-05-12）

| 模块 | 状态 | 关键文件 |
|------|------|---------|
| **Qlib 数据链** | 全 A 5411 只，staging/promotion/health gate | `update_qlib_data.py`, `check_qlib_data_health.py` |
| **LGB 主模型** | 5204 预测，IC=0.052, Top20 spread +8.28% | `train_lgb.py`, `lgb_model.pkl` |
| **评估闭环** | IC/RankIC/Spread/方向命中率 | `evaluate_lgb_test.py` |
| **简化回测** | TopK + dropout + 成本 | `backtest_qlib_signal.py` |
| **Rolling 验证** | 6 split, 平均 IC=0.067 | `rolling_train.py` |
| **因子评估** | 单因子 IC + ablation | `evaluate_factor_ic.py`, `train_factor_ablation.py` |
| **盘后 Pipeline** | 串行: 数据→健康→训练→smoke→评估→归因→衰减→promotion | `after_close_pipeline.py` |
| **模型治理** | registry + promotion + rollback | `model_registry.py`, `promote_model.py` |
| **因子衰减监控** | IC 时序追踪 + 预警 | `monitor_factor_decay.py` |
| **事件冲击** | 结构化事件表 + 衰减 + 时间加权 | `event_impacts` 模块 |
| **多模型对照** | LGB/XGB/CatBoost/DoubleEnsemble/ALSTM/Transformer | `train_model_suite.py` |
| **生产调度** | 5 时段 crontab + job_status + WeChat 告警 | `install_crontab.py`, `job_status.py` |
| **估值因子** | PE/PB/PS/PCF 采集 + 注入 | `fetch_fundamental_valuation.py` |
| **资金流因子** | 主力净流入 + 北向 | `fund_flow`, `northbound` collectors |
| **FeatureMerger** | 多源特征合并框架 | `models/feature_merger.py` |
| **RL 训练** | DQN 雏形，标记 experimental | `train_rl.py`, `rl_model.pt` |
| **MPS GPU** | M4 Max 验证可用，Transformer 13x 加速 | 双方确认 |

### 正在进行（未提交）

| 文件 | 内容 |
|------|------|
| `models/feature_merger.py` | 新因子合并逻辑扩展（可能含 PIT 修复） |
| `scripts/fetch_fundamental_quality.py` | **新** 质量因子采集 (ROE/ROA/毛利率) |
| `scripts/fetch_shareholder_data.py` | **新** 股东数据采集 |
| `scripts/run_brinson_attribution.py` | **新** Brinson 归因脚本 |
| `scripts/evaluate_factor_ic.py` | 因子评估增强 |
| `scripts/train_factor_ablation.py` | 消融实验改进 |
| `scripts/monitor_factor_decay.py` | 衰减监控扩展 |

---

## 四、核心共识（CC + CX 完全一致）

### 4.1 战略方向
1. **先加数据（乘法），再加工程（除法），最后加行为（加法）**
2. **先评估后升级模型** — 没有 IC/RankIC/回测收益，换模型只是更贵的随机数
3. **差距最大的不是模型，是有效因子宽度** — Alpha158 只有价量维度，缺 6 大维度约 120+ 因子
4. **深度模型不急上生产** — MPS 可用消除速度顾虑，但须通过同一评估/回测门禁
5. **长线标注为"观察榜"** — 没有真正基本面模型前不冒充生产推荐
6. **不押单一数据源** — provider=auto + staging + health gate
7. **Ensemble 用 rank 加权** — 不直接平均 raw prediction

### 4.2 因子工厂路线
1. 估值因子（PE/PB/PS/PCF）→ 质量因子（ROE/ROA/毛利率）→ 资金流 → 北向 → 股东 → 行业 → 事件
2. 每个因子必须通过：**RankIC > 0.01 + TopK spread > 0 + IC > 0** 联合判定
3. 增量因子必须通过 **ablation + shuffled negative control + residual IC** 验证
4. 所有因子需要 **winsorize + rank/zscore** 预处理后才能合并
5. 测试窗口不能少于 60 天，应该用 12+ 个 rolling 窗口验证稳定性

### 4.3 工程架构
1. 数据管线：`provider → raw_daily_cache → normalize → qlib_staging → health+smoke → promote`
2. 复权口径必须统一，混合数据源前做 adjustment reconciliation
3. 首次 bootstrap 和日常增量更新必须明确分开
4. 串行 after_close_pipeline，任一环节失败即停止
5. 生产 LGB 门槛：4500+ 最新有效预测 + 4500+ 数据 instruments

### 4.4 技术选型
| 需求 | 统一结论 |
|------|---------|
| 因子 IC | Qlib `calc_ic` 做快速检查 + Alphalens 做深度诊断（待装） |
| 回测 | Qlib backtest 为主 + RQAlpha 补 T+1（待装） |
| 组合优化 | cvxpy/numpy 冲突未解决，暂用 scipy 或自写约束 |
| 风险模型 | Qlib `ShrinkCovEstimator` 可用 |
| 实验管理 | Qlib Recorder + SignalRecord（初版已接入） |
| 深度模型 | MPS 可用，Week 3 和树模型并行对照 |
| RL | 实验性质，不进生产；等 PIT + 组合回测 + paper OMS 稳定后再启动 |

---

## 五、已解决的技术分歧

| 分歧 | CC 立场 | CX 立场 | 最终裁决 |
|------|--------|---------|---------|
| **MPS GPU 可用性** | 可用，13x 加速 | 不可用(False) | **CC 对** — CX 独立复现后撤销，确认 MPS 可用 |
| **模型排序能力** | TopK 分层有效 | 排序能力弱 | **收敛** — 广义 RankIC 弱，但 Top/Bottom 极端分层有效 |
| **深度模型时序** | Week 3 并行对照 | Week 5+ | **收敛** — MPS 消除速度问题，Week 3 并行对照 |
| **baostock 定位** | 非日常主源 | 先跑通当主源 | **收敛** — bootstrap/repair 用，日常用更快源 |
| **外部库必要性** | 4/6 不需要 | 6 个都有价值 | **CC 修正** — 3 个有独立价值(Alphalens/RQAlpha/QuantStats)，1 个暂不可用(cvxpy) |
| **CSZScoreNorm 前提** | Alpha158 feature 做了 zscore | 只对 label 做 | **CX 对** — Alpha158 默认不对 feature 做 CSZScoreNorm |
| **因子 STRONG 判定** | Pearson IC > 0.02 | 需要 RankIC + TopK spread 联合 | **CX 对** — 生产目标是 TopK 选股 |
| **Ensemble 平均方式** | raw 平均 | rank 加权 | **CX 对** — 不同模型输出尺度不同 |
| **Qlib API 用法** | risk_analysis(pred) | 参数类型错 | **CX 对** — 应用 calc_ic(pred, label) |
| **13 天测试窗口** | 模型已验证 | 太短不够 | **CX 对** — 需要 60-120 天 + rolling |

---

## 六、与百亿量化私募的差距评分

| 模块 | 当前 | 目标 | 最大差距 |
|------|:---:|:---:|---------|
| 日线数据主链 | 55 | 95 | 数据源稳定性 |
| **PIT 多源数据** | **20** | **95** | **最新值广播有未来函数泄漏** |
| **因子工厂** | **25** | **95** | **无 registry、衰减监控、分层上线** |
| 模型训练 | 45 | 90 | 实验追踪不完整 |
| Rolling 验证 | 35 | 90 | 只有 6 split，需 24+ |
| **回测系统** | **25** | **95** | **简化 TopK，非订单/成交级** |
| **风险模型** | **15** | **95** | **无行业/风格/容量/成本控制** |
| 交易执行 | 5 | 95 | 无 OMS/券商接口/合规 |
| 生产治理 | 35 | 90 | 有雏形，缺 artifact lineage |
| 算力平台 | 20 | 90 | 单机为主 |

**关键洞察：** 差距最大的不是模型（XGB 已接近 3-5 年前百亿水平），而是 **PIT 数据安全** 和 **因子工厂** 和 **风险模型**。

---

## 七、统一追赶路线图（2026-05-13 更新，整合 cx PIT-safe 计划）

### 已验证的 PIT-safe 基线（2026-05-13 实测）

| Model | Dims | IC | ICIR | RankIC | Spread | RankIC>0 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| LGB | 158 | -0.003 | -0.07 | +0.005 | +0.65% | 46% |
| XGB | 158 | -0.007 | -0.16 | +0.006 | -0.38% | 46% |
| **CatBoost** | **158** | **+0.002** | **+0.03** | **+0.010** | **+2.44%** | **54%** |
| LGB | 202 | -0.011 | -0.26 | +0.020 | -1.65% | 77% |
| **XGB** | **202** | **+0.005** | **+0.16** | **+0.030** | -0.21% | **77%** |
| CatBoost | 202 | -0.011 | -0.28 | +0.018 | -0.04% | 69% |

**关键结论：**
- 补充因子有效：所有模型 202 维 RankIC 都高于 158 维（3-6 倍提升）
- CatBoost 158 维 Spread 最高（+2.44%），XGB 202 维 RankIC 最高（+0.030）
- IC 和 RankIC 方向不一致，需要排序优化（LGBMRanker/LambdaRank）
- 旧 LGB 的 IC=0.072 / ICIR=2.09 已确认为 PIT 泄漏虚高

### Phase 0：PIT 修复 + 可信基线 ✅ 已完成

1. ~~FeatureMerger asof join~~ ✅
2. ~~PIT-safe LGB/XGB/CatBoost × 158/202 六档基线~~ ✅
3. ~~旧泄漏结果从候选中剔除~~ ✅

### Phase 1：因子工厂 v1 + ST_CLIENT 数据扩展

**目标：从 202 维扩展到 280+ 维 PIT-safe 有效因子。**

**1a. ST_CLIENT 批量日频因子（正在拉取中）：**

| 接口 | 因子 | 维度 | 状态 |
|------|------|:---:|:---:|
| `margin_detail(trade_date=)` | 融资融券余额/买入额 | +7 | 正在拉 5 年历史 |
| `top_list(trade_date=)` | 龙虎榜 | +7 | 正在拉 |
| `limit_list_d(trade_date=)` | 涨跌停 | +10 | 正在拉 |
| `moneyflow_hsgt(trade_date=)` | 北向资金汇总 | +4 | 正在拉 |
| `fina_indicator(ts_code=)` | 100+ 财务指标 | +10 | 待 API 限额重置 |
| `stk_holdernumber(ts_code=)` | 股东户数 | +2 | 待 API 限额重置 |

**1b. 截面预处理（cx Phase 2，必做）：**
1. 非法值处理：`inf → NaN`，极端 PE/PB 做业务 guard
2. 每日截面 winsorize：1%-99%
3. 每日截面 zscore 和 rank percentile
4. 缺失指示列（missing flags）
5. 行业中性化 + 市值中性化
6. 时序衍生：5/20/60 日均值、变化率、波动率、历史分位

**1c. 因子注册与健康监控：**
- 每个因子有数据字典（来源/公式/频率/可用时间/缺失处理/覆盖率阈值）
- 每日输出 coverage、NaN、inf、极值、重复键、最新日期报告
- 覆盖率低于阈值的因子只进观察池

### Phase 2：严格消融 + 负控 + 特征集定稿

**目标：证明每个因子包真的有边际 alpha，确定最终上线特征集。**

**2a. 因子包消融实验（cx Iteration 2）：**

每个因子包（valuation / moneyflow / northbound / quality / shareholder / margin / industry）分别跑：
1. `base` (Alpha158)
2. `base + factor_group`
3. `base + shuffled_factor_group` ← 负控
4. `base + selected_factor_group`

通过门槛：
- rolling 窗口中 ≥70% 的 RankIC/spread 为正
- 负控不提升
- 对 baseline residual 有 RankIC
- coverage ≥ 80%

**2b. 六档特征集对照（cx Iteration 1）：**

| 版本 | 特征 | 回答 |
|------|------|------|
| FS-158 | Alpha158 | 价量基线 |
| FS-206 | Alpha158 + 补充因子 | 非价量是否有增量 |
| FS-360 | Alpha360 | 原始价量序列是否更强 |
| FS-408 | Alpha360 + 补充因子 | 序列 + 非价量是否互补 |
| FS-518 | Alpha158 + Alpha360 | 两套价量是否冗余 |
| FS-566 | 全量 | 能否被一个模型吸收 |

**2c. 特征筛选（cx Iteration 3）：**
1. 数据质量过滤 → 2. 单因子 rolling → 3. 残差 IC → 4. 相关性聚类 → 5. 重要性稳定性 → 6. ablation

产出三套特征集：compact(250-300) / main(300-450) / full(566)

### Phase 3：模型优化 + Ensemble

**目标：在可信因子基础上榨出排序能力。**

**3a. 模型分工（cx Iteration 4）：**

| 模型 | 输入 | 角色 |
|------|------|------|
| LGB-compact | 250-300 selected | 稳定主力 |
| XGB-compact | 250-300 selected | 多样性 |
| CatBoost-compact | 250-300 selected | 非线性+缺失处理 |
| LGBMRanker | selected features | 直接优化排序目标 |
| ALSTM/Transformer | Alpha360 序列 | 价格路径模型 |

**3b. Rank Ensemble：**
- 每日 rank 加权融合，不直接平均 raw score
- 权重由 rolling 表现决定
- 单模型权重上限 50-60%

**3c. 判定标准（cx Iteration 5）：**
- rolling OOS RankIC 提升
- Top20 spread 成本后提升
- 换手没有显著恶化
- 行业/风格暴露可解释
- 负控不提升
- IC 提升但 TopK spread 不提升 ≠ 更好

### Phase 4：Rolling + 成本 + 可交易验证

**目标：从"预测强"变成"可交易、成本后、风险可控"。**

1. Rolling 24+ split，覆盖 3 年+，市场状态分层
2. 训练窗口对比：2 年 / 3 年 / 5 年 / expanding
3. TopK 回测加交易成本、滑点、停牌、涨跌停
4. 流动性过滤、行业权重上限、单票上限
5. 输出换手率、最大回撤、胜率、容量粗估
6. Qlib Recorder 全程记录 artifact

### Phase 5：组合优化 + 风险模型

1. 建行业映射表（申万/中信）
2. 风险暴露矩阵：行业 + size + beta + momentum + value + volatility + liquidity
3. 交易成本模型：固定成本 + 滑点 + 冲击成本
4. 组合归因：行业配置、风格暴露、个股选择、交易成本
5. 约束优化（scipy，绕过 cvxpy）

### Phase 6：Paper Trading + 执行系统

1. Paper OMS：target position → order → simulated fill → position → pnl
2. T+1、涨跌停不可买卖、停牌、最小交易单位
3. TWAP/VWAP 执行算法模拟
4. 合规日志 + 审计
5. shadow 2-4 周 → promotion → 一键 rollback

### Phase 7：生产治理闭环（cx Phase 6）

每日固定报告：
1. 数据健康：覆盖率、缺失、异常值、延迟
2. 因子健康：IC、RankIC、衰减、分布漂移
3. 模型健康：预测覆盖、score 分布、TopK 变化
4. 组合健康：行业暴露、风格暴露、换手、预估成本
5. 归因报告：收益来源分解
6. promotion gate：候选模型是否允许 shadow

### Phase 8：算力 + 研究平台

1. Qlib cache + factor store parquet 分区
2. 并行因子计算
3. Optuna/Ray Tune 超参搜索
4. MPS 深度模型 pipeline
5. 研究看板

### Phase 9：RL 策略控制器（最后）

**定位：** 组合层动态控制（仓位/换手/TopK/风险降级），不做个股预测。
- 先做 rule-based baseline → 再训 RL → 24+ rolling 对照
- 必须在 Phase 4-5 稳定后才启动

### 不做什么

1. 不把旧 LGB 泄漏结果当目标
2. 不因单次 IC 提高就上线
3. 不把低覆盖因子直接喂模型
4. 不把最新快照广播到历史样本
5. 不先押 Transformer/RL 来掩盖数据问题
6. 不用单窗口 Top20 spread 证明模型变强

---

## 八、因子合并 Debug 教训（重要经验）

Alpha158 + PE/PB/Turn 合并后 IC 从 +0.024 变成 -0.015 的事件揭示了关键教训：

| 教训 | 说明 |
|------|------|
| **单因子 IC > 0 ≠ 模型提升** | Pearson IC 正但 RankIC 负的因子加入后破坏排序 |
| **尺度不匹配是真问题** | Alpha158 是相对价量表达式，PE/PB 是原始值，分布形态差异大 |
| **对照实验必须干净** | 同一脚本、同一 split、同一 seed、即时训练 base 和 enhanced |
| **负控必须做** | shuffled 因子不应优于 base，否则说明改善来自噪声 |
| **residual IC 更严谨** | 新因子应对 base model 残差有解释力，而非只有独立 IC |
| **预处理不能跳过** | winsorize + rank/zscore 是合并前的必要步骤 |

**最小可执行实验设计（6 组）：**
```
base_raw
base_raw + custom_raw
base_raw + custom_winsor_zscore
base_raw + custom_rank
base_raw + shuffled_custom_winsor_zscore  ← 负控
base_raw + each_one_factor               ← 单因子消融
```

---

## 九、技术选型统一表

### 当前使用
| 工具 | 角色 |
|------|------|
| **Qlib** | A 股日频 alpha、数据、模型训练、评估、回测主干 |
| **LightGBM** | 生产主模型 |
| **XGBoost/CatBoost** | 对照模型 |
| **AKShare** | 免费数据采集主源 |
| **baostock** | 历史数据 bootstrap/repair |
| **MiniMax LLM** | 晚间宏观分析 |
| **SnowNLP** | 中文舆情 baseline |
| **PyTorch + MPS** | 深度模型训练（M4 Max 32 核 GPU） |
| **Tianshou** | RL 训练框架 |

### 待引入（按优先级）
| 工具 | 角色 | 优先级 |
|------|------|:---:|
| **statsmodels** | Qlib report 可视化依赖 | P0 |
| **Alphalens** | 深度因子诊断（分位收益/换手/衰减） | P1 |
| **QuantStats** | 绩效报告 HTML tear sheet | P1 |
| **RQAlpha** | A 股 T+1/涨跌停/停牌完整回测 | P2 |
| **TuShare Pro** | 批量日线数据（可选） | P2 |
| **scipy.optimize** | 简化约束组合优化（绕过 cvxpy） | P2 |
| **vectorbt** | 快速参数扫描回测 | P3 |

### 暂不引入
| 工具 | 原因 |
|------|------|
| PyPortfolioOpt | cvxpy/numpy 冲突未解决 |
| Qlib TopkDropoutStrategy | 同上 |
| Qlib PortfolioOptimizer | 同上 |
| vn.py/Lean/NautilusTrader | 还没到实盘执行阶段 |
| FinRL | RL 不是当前优先级 |

---

## 十、3 个月里程碑

| 月份 | 目标 | 验收指标 |
|------|------|---------|
| **5 月** | Phase 0 (PIT 修复) + Phase 1 开始 (估值+质量因子) | FeatureMerger 无未来函数；因子 ablation 有 1+ 个因子通过联合判定 |
| **6 月** | Phase 1 完成 + Phase 2 (Recorder + Rolling 24+) | IC 从 0.05 → 0.06+；有 Brinson 归因；推送含因子健康和模型状态 |
| **7 月** | Phase 3 开始 (风险模型 + 组合约束) | 回测有行业约束、换手上限、成本后收益；推文变成"组合持仓" |

---

## 十一、小资金差异化优势

百亿私募追不上，但小资金有三个可行突破方向：

1. **小盘灵活性** — 不需要承载百亿容量，可做机构因容量限制做不了的小盘/事件/拥挤反转策略
2. **AI 解释层** — 用 LLM 做公告/新闻/舆情/政策事件结构化，形成比 Alpha158 更低相关的信息源
3. **研究纪律** — 严格执行 PIT、negative control、rolling、成本后组合和灰度上线，避免"看起来很强、实盘失效"的陷阱

**真正的目标：** 数据不污染 → 因子有证据 → 模型有 rolling → 组合有约束 → 交易有成本 → 生产能回滚 → 每天知道收益和亏损来自哪里。

---

## 十二、文档索引

| 文档 | 作者 | 核心内容 |
|------|:---:|---------|
| `cx-v2-iteration-plan.md` | CX | **主执行计划**，Phase 0-1 实施记录、cc/cx 分歧裁决、Qlib 工作流 |
| `cx-billion-quant-gap-research-2026-05-12.md` | CX | 与百亿私募差距分析、6 Phase 追赶路线图、RL 策略控制器设计 |
| `cx-qlib-advanced-implementation-plan-2026-05-09.md` | CX | Qlib API 校验、评估/回测脚本详细规格 |
| `cx-qlib-next-version-iteration-plan-2026-05-10.md` | CX | 影响因子层 schema、多源因子统一架构 |
| `cc-next-phase-roadmap.md` | CC | 三条线并行路线图、因子 ROI 排序、3 个月里程碑 |
| `cc-factor-gap-analysis.md` | CC | 因子缺口分析 158→280 维、FeatureMerger 架构、百亿对照 |
| `cc-qlib-advanced-features-roadmap.md` | CC | Qlib 功能全景、Tier 排序、GPU 验证、cc 自我修正 |
| `cc-fund-strategy-integration.md` | CC | 头部私募策略、A 股大神量化、5 个高 ROI 因子 |
| `cc-factor-merge-debug-log.md` | CC | 因子合并 IC 恶化 debug、4 方案、cx 纠正接受 |
| `cx-factor-merge-debug-discussion-2026-05-10.md` | CX | 对 cc 因子合并的回应、正确验证流程 |
| `cx-quant-sentiment-deep-research-2026-05-07.md` | CX | 量化库全景、舆情 NLP 选型 |
| `cx-quant-libraries-sentiment-and-masters-research-2026-05-08.md` | CX | 量化库/舆情/大师经验深度调研 |
| `cc-qlib-advanced-features-roadmap.md` 附录 | CC | 自定义因子表达式示例、Qlib 未用模块清单 |
