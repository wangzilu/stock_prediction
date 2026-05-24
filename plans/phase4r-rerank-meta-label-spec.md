# Phase 4R: XGB Candidate Rerank / Meta-Selector 设计规范

**日期**: 2026-05-24
**来源**: CX 深度调研 + 学术文献综述
**状态**: 设计阶段，待 PIT/T+1/LLM overlay 稳定后开始
**优先级**: Phase 4N 之后，深度模型/RL 之前

---

## 1. 核心架构

```
XGB174 (主alpha召回器)
  → 全A 5000只 → Top100/Top200 候选池

二阶段模型 (精排/过滤/加权)
  → Top100/200 → final Top20/Top100 weights

optimizer_v2 (最终组合执行器)
  → weights → orders → paper/live trading
```

**关键原则**：
- 二阶段不重新发现 alpha，只判断"XGB 这次要不要信"
- 训练必须用 OOF XGB 候选池（walk-forward，不能回头看）
- 不让二阶段完全重排，初版只过滤/微调

---

## 2. 三条实验路线（CX 推荐，按风险从低到高）

### 2.1 Meta-filter（最保守）

```python
# 主模型: XGB174 选 Top100
# 副模型: 判断"这只 XGB 推荐的票，未来5日是否跑赢候选池中位数+成本"
# 上线: 只过滤 meta_prob 最低的 20-30%，不大幅重排

label = (future_return > median_return + cost_threshold)  # 二分类
model = LightGBM/XGBoost binary classifier
output = meta_prob ∈ [0, 1]
action = filter out bottom 20-30% by meta_prob
```

### 2.2 LTR Reranker（中等风险）

```python
# 用 XGBRanker / LGBMRanker，每天 = 一个 query group
# 只在 XGB OOF Top100 内训练，优化 NDCG@20

qid = trade_date
candidates = XGB_OOF_Top100[date]
label = rank_in_pool(future_return)  # 0-4 分位等级
objective = rank:ndcg / lambdarank
final_score = 0.75 * rank(xgb_score) + 0.25 * rank(rerank_score)
```

### 2.3 Risk-aware Reranker（最完整）

```python
# 二阶段加入风险/成本/流动性惩罚
rerank_score = expected_alpha
             - λ1 * predicted_vol
             - λ2 * liquidity_cost
             - λ3 * drawdown_risk
```

---

## 3. 二阶段特征（不重复 174 维）

```python
RERANK_FEATURES = {
    # XGB 信号特征
    "xgb_score": "XGB174 原始分数",
    "xgb_rank_pct": "XGB 分数在当天的百分位",
    "rank_gap_to_cutoff": "距离 Top100 边界的排名差",

    # 流动性/可交易性
    "adv_20d": "20日平均日成交额",
    "turnover_ratio": "当日换手率",
    "is_limit_up": "是否涨停（买不到）",
    "is_limit_down": "是否跌停（卖不出）",
    "impact_cost_est": "预估冲击成本",

    # 风格/暴露
    "industry_code": "申万行业",
    "log_market_cap": "对数市值",
    "volatility_20d": "20日波动率",

    # 事件/新闻 (LLM overlay)
    "llm_event_alpha": "LLM 事件 gated overlay 分数",
    "has_official_event": "是否有交易所公告",
    "event_novelty": "事件新颖度",

    # Regime 信号
    "regime_risk_on": "regime controller 综合分数",

    # Entry timing
    "ret_1d": "昨日收益率",
    "ret_3d": "3日收益率",
    "gap_from_ma5": "距5日均线偏离",
}
```

---

## 4. 验收门槛（CX 定义）

- 24 split 里至少 16/24 超额为正
- 成本后年化/IR 相对 opt_top100_to10 提升 >= 10%
- 平均换手恶化不超过 15%
- 最大回撤不恶化
- 行业/市值暴露不明显偏离
- Top100 内 RankIC 或 Precision@20 有稳定增量
- Shadow 至少 20 个交易日，不直接进 champion

---

## 5. 实施纪律（CX 强调）

1. **OOF 候选池**：历史每天的 Top100 必须来自只用过去数据训练的 XGB 预测
2. **不要 100% 用 ranker**：初版 0.75*xgb + 0.25*rerank，或只过滤
3. **不要重复 naive ensemble**：之前 XGB+Ranker 简单加权已经失败了
4. **先稳住 PIT/T+1/overlay**，再开二阶段

---

## 6. 学术参考

| 方法 | 论文/来源 | 启发 |
|------|---------|------|
| Cascade Reranking | Google Stochastic Retrieval-Conditioned Reranking | 二阶段按一阶段分布训练 |
| LTR for stocks | Stock portfolio selection using LTR with news sentiment | RankNet/ListNet 按新闻情绪排序 |
| STHAN-SR | AAAI 2021 | 超图建模股票关系做排序 |
| Meta-labeling | López de Prado / Joubert | 主模型给机会，副模型判断下注 |
| Multi-task ranking | Quantitative stock portfolio optimization | 同时学 return + volatility |
| Two-stage selection | Enhancing Portfolio Optimization | 先预选后组合优化 |
| TRA (Qlib) | Temporal Routing Adaptor | 样本按 regime 分配到不同预测器 |
| DoubleEnsemble (Qlib) | 样本重加权 + 特征选择 | 缓解低信噪比过拟合 |

---

## 7. 前置依赖

- [x] XGB174 champion 稳定（24-split 验证通过）
- [x] opt_top100_to10 执行层（Sharpe 4.5+）
- [ ] PIT publish_time 时间对齐（已修，验证中）
- [ ] T+1 open 执行价格（已修）
- [ ] LLM 事件 overlay 进 shadow（待 CX 确认）
- [ ] 60 天 paper trading 数据积累
