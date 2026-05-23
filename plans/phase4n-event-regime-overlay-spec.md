# Phase 4N: Event + Regime Overlay 设计规范

**日期**: 2026-05-23
**来源**: CX 审阅 + CC 实施计划
**状态**: 设计阶段

---

## 1. 核心原则

- **LLM 抽事实，历史数据校准影响** — 不让 LLM 直接给 impact 数值
- **事件做 overlay，不塞进 XGB** — `final_score = zscore(xgb) + alpha * zscore(event_alpha)`
- **regime 控仓位和风险预算，不选股**
- **事件分三层，不混合** — official > media > social

---

## 2. 事件侧设计

### 2.1 数据源优先级（CX 修正版）

| 优先级 | 来源 | 用途 | 状态 |
|:---:|------|------|:---:|
| 1 | 交易所/巨潮公告 | 大合同、业绩预告、回购、减持、诉讼、重组 | ✅ 东财公告 API |
| 2 | 年报/半年报/调研纪要/互动易 | 产品、客户、订单、市占率 | ST_CLIENT irm_qa |
| 3 | 财经媒体（东财/财联社/证券时报） | 事件传播强度、二次解读 | ✅ 东财搜索 API |
| 4 | 雪球/股吧 | 情绪/关注度/分歧度（不是事实源） | ✅ 雪球 API 可用 |

### 2.2 LLM 抽取字段（升级版 v2）

当前 v1 只抽：event_type, impact_1d, impact_5d, confidence, relevance, novelty

CX 要求的 v2 字段：
```python
EVENT_SCHEMA_V2 = {
    # 事件事实
    "event_type": str,           # order_win, buyback, insider_buy, earnings, lawsuit, ...
    "event_time": str,           # ISO datetime
    "is_official_disclosure": bool,  # 是否交易所公告
    "is_new_information": bool,     # 是否首次披露
    "is_repeated_news": bool,       # 是否重复报道

    # 量级
    "event_magnitude": float,       # 金额（万元）
    "magnitude_to_revenue": float,  # 金额/TTM营收
    "magnitude_to_mcap": float,     # 金额/市值

    # 质量
    "is_framework_agreement": bool,  # 是否框架协议（vs 真实订单）
    "customer_quality": str,         # "top_tier", "mid", "unknown"
    "delivery_horizon": str,         # "short", "medium", "long"

    # 方向和置信度
    "direction": int,               # +1, 0, -1
    "confidence": float,            # 0-1
    "is_price_sensitive": bool,     # 可能引起显著股价反应

    # 来源
    "source": str,
    "source_quality": float,        # 0-1
}
```

### 2.3 事件因子三层架构

```
Layer 1: official_event_alpha
  来源: 交易所公告
  因子: direction × magnitude_to_revenue × decay(half_life=2d)
  最硬，PIT 最清楚

Layer 2: media_dissemination_alpha
  来源: 财经媒体
  因子: direction × novelty × source_quality × decay(half_life=1d)
  适合事件传播强度

Layer 3: social_attention_alpha
  来源: 雪球/股吧
  因子: attention_zscore × sentiment_direction × decay(half_life=0.5d)
  只做情绪扩散，不做事实
```

### 2.4 校准流程

不用 LLM 的 impact 数值。改为：
1. 按 event_type × source_quality × novelty 分桶
2. 统计每个桶的历史未来 1d/3d/5d 真实收益
3. 低样本桶强 shrink 到 0
4. 生成 `calibrated_event_alpha = bucket_mean_return × decay`

### 2.5 验收门槛（CX 定义）

- 事件子集 RankIC >= 0.08
- 控制 XGB 后 residual IC > 0.02
- Top20 成本后收益提升 >= 8%
- 换手增加 <= 15%
- 最大回撤不恶化
- 负面事件过滤能减少踩雷

### 2.6 必须补的验证

- [ ] 事件子集 RankIC（已做：+0.093）
- [ ] 控制 XGB score 后的 residual IC
- [ ] 按事件类型分桶消融
- [ ] publish_time 级别 PIT 验证
- [ ] Placebo：事件日期后移 5 天 / 随机换股票
- [ ] Event study：T-3 到 T+10 收益路径

---

## 3. Regime 侧设计

### 3.1 核心分数

```python
REGIME_SCORES = {
    "policy_support_score": "政策支持力度",
    "liquidity_score": "流动性充裕度",
    "credit_stress_score": "信用压力",
    "leverage_unwind_score": "杠杆解除风险",
    "microcap_crash_risk": "小微盘踩踏风险",
    "theme_breadth_score": "题材扩散宽度",
    "overseas_shock_score": "海外冲击",
}
```

### 3.2 控制目标（不选股）

```python
# 当 risk 高时
if credit_stress_score > threshold:
    max_turnover *= 0.5
    smallcap_exposure = 0
    cash_buffer = 0.20

# 当 policy_support 强时
if policy_support_score > threshold:
    max_industry_deviation *= 1.5  # 允许行业偏移
```

### 3.3 验收门槛

- 2015/2018/2022/2024 压力窗口回撤下降
- 正常市场收益不明显牺牲
- 小微盘踩踏期自动降暴露

---

## 4. 实施文件

```
scripts/
  collect_announcements.py          ✅ 已写
  collect_company_events.py         待写（合并新闻+公告+互动易）
  backfill_llm_events.py            ✅ 已写

factors/
  llm_event_extractor.py            ✅ v1 已写
  llm_event_extractor_v2.py         待写（v2 schema）

scripts/
  build_event_factor_v2.py          待写（校准+三层架构）
  build_market_regime_features.py   待写
  phase4n_event_overlay_ablation.py 待写
  phase4n_regime_controller.py      待写
```

---

## 5. 当前进展

| 项目 | 状态 |
|------|:---:|
| 东财新闻采集（5000只/天） | ✅ |
| 东财公告采集（2000+/天） | ✅ |
| LLM 事件提取 v1 | ✅ |
| 并发+流式写入+重试 | ✅ |
| 历史回填 18 天 | ✅ |
| 子集 RankIC +0.093 | ✅ |
| 全A RankIC +0.018 (RICIR 0.94) | ✅ |
| v2 schema 抽取 | 待做 |
| 历史收益校准 | 待做 |
| 事件分层 (official/media/social) | 待做 |
| Residual IC 验证 | 待做 |
| Regime controller | 待做 |
