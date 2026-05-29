# LLM Event Pipeline — L0/L1/L2 Tiered Routing Design

> Draft skeleton — 2026-05-29 night. Content/TBD markers to fill 5/30.

## Context

2026-05-29 evening incident: LLM pipeline produced **0 / 426** events because
MiniMax-M2.5-highspeed spent 489 / 512 completion_tokens on `<think>` blocks,
truncating JSON output. Hotfix shipped: swap to `MiniMax-Text-01` (commit
`07a5fb6`) + per-extractor token stats. Rerun recovered 152 events but
exposed **MiniMax RPM rate limit** as new bottleneck (273 / 425 calls hit
HTTP 429 `rate_limit_error 1002`).

Hotfix solved the WHAT (model swap) but not the HOW (architecture). Need
tiered routing so we (a) don't burn rate budget on low-value items, (b)
have escalation path for genuinely complex events, (c) survive RPM
ceilings via retry queue.

See [[llm-pipeline-architecture]] memory + [[project_audit_20260529]] for
full incident timeline.

## Current Architecture (post-hotfix)

```
collect_daily_news  → daily_news/{date}.jsonl  (raw, 15000+ items)
   ↓
event_filter        → daily_news_filtered/{date}.jsonl  (filtered, ~500 items)
   ↓
LLMEventExtractorV2 → MiniMax-Text-01 → llm_events_v2/{date}.jsonl
   ↓
_write_to_unified_store → EventStore (partition by signal_date)
   ↓
build_llm_event_factors → llm_event_factors.parquet
```

**Known problems:**
- ✅ Reasoning waste — fixed today
- 🔴 RPM ceiling — 35.8% of calls fail with 429 (no retry, lost forever)
- 🔴 No L0 rule filter beyond `event_filter.py` (which is keyword pre-rank,
  not classification) — sends ~500 items to LLM regardless of how many are
  trivially classifiable
- 🔴 No L2 escalation — complex multi-entity / supply-chain news gets the
  same shallow extraction as routine announcements
- 🔴 No dedup cache — same title repeated across days re-extracts
- 🔴 No daily cost / token dashboard surface in `daily_health_check`

## Proposed Architecture

```
                              ┌──────────────┐
                              │ daily_news + │
                              │ announcements│
                              └──────┬───────┘
                                     ▼
                      ┌─────────────────────────────┐
              L0 ───▶ │ Rule classifier + dedup    │
                      │ (extend event_filter.py)    │
                      └──────┬──────────────────────┘
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐  ┌────────────┐  ┌──────────┐
        │ drop /   │  │ direct     │  │ → L1     │
        │ trivial  │  │ classify   │  │ extract  │
        └──────────┘  │ (no LLM)   │  └─────┬────┘
                      └────────────┘        │
                                            ▼
                            ┌──────────────────────┐
                       L1 ─▶│ MiniMax-Text-01      │
                            │ extraction prompt    │
                            └──┬──────────────┬────┘
                               │              │
                  confident ───┘     not confident
                                     OR multi-entity
                                     OR long content
                                            │
                                            ▼
                            ┌──────────────────────┐
                       L2 ─▶│ MiniMax-M2.5 reason  │
                            │ deep analysis        │
                            └──────┬───────────────┘
                                   ▼
                          (write to EventStore)
```

**Target volume ratio:** L0 70% / L1 25% / L2 5%

## Components

### L0 — Rule classifier + dedup (extends `factors/event_filter.py`)
- **Input:** raw merged news + announcements (~16500 items/day)
- **Output:** `{drop, direct_classify, route_to_l1, route_to_l2}` per item
- **Rules** (TBD priorities):
  - Drop: pure社交闲聊、与 A 股无关、太短(<10 字)
  - Direct classify: 标题含强关键词 → 直接打标，跳过 LLM
    - "撤销退市风险警示" → `routine_announcement, direction=0`
    - "停牌" → `routine_announcement, direction=0`
    - "中标" + 金额 → `order_win, direction=+1` (金额从 title 抽)
    - "回购" + "已实施" → `share_buyback, direction=+1`
    - …enum TBD: 完整列表见 [[TBD-l0-rules]]
  - Route to L2 hint: 多主体 (≥3 公司名)、供应链/产业链关键词、长公告 (>500 字)
  - 否则 → L1
- **Dedup hash:** `sha1(stock_code + title_normalized + source + publish_date)`
- **Cache:** SQLite or parquet keyed by hash → existing extraction reused
- **Implementation TBD:** classifier 是 hardcoded rules vs 小模型？hardcoded 起步、覆盖 ≥70% drop/direct

### L1 — Cheap non-reasoning extraction (`factors/llm_event_extractor_v2.py`)
- **Model:** `MiniMax-Text-01` (already wired today)
- **Prompt:** strict extraction (already rewritten today)
- **Concurrency:** TBD — 当前 16 workers + 60 RPM 触发 429。建议 8 workers + 实测 RPM 上限
- **Per-call max_tokens:** 512 default, 1024 only for items L0 标记为 "long content"
- **Output:** structured event JSON → EventStore

### L2 — Reasoning extraction (NEW: `factors/llm_event_extractor_l2.py`)
- **Model:** `minimax-m2.5-highspeed` 或 `MiniMax-M1` (reasoning, deeper)
- **Triggers:**
  - L0 hint: multi-entity / supply-chain
  - L1 confidence < 0.4 (uncertain) — TBD threshold
  - L1 detected event_type ∈ {`tech_breakthrough`, `restructuring`, `lawsuit_filed`} — categories where impact direction needs context
- **Prompt:** allows reasoning, asks for evidence + sub-events + impacted parties
- **Output schema:** extends V1 with `related_stocks: list`, `transmission_path: list`, `evidence_spans: list[str]`
- **Rate budget:** stricter cap (10 RPM) since reasoning calls expensive

### Rate limiter + retry queue
- **Per-tier RPM cap:** L1 30 RPM, L2 10 RPM (TBD — measure first)
- **Jitter:** `time.sleep(random.uniform(0, base_interval))` between calls
- **Retry on 429:**
  - Exponential backoff: 5s → 15s → 45s
  - After 3 attempts, write to **persistent retry queue** `data/storage/llm_retry_queue/{date}.jsonl`
  - Retry queue drain job runs at 22:30 (after main pipeline + cron quiet hours)
- **Circuit breaker:** if 5 consecutive 429 across all workers, pause 60s

### Daily cost / health dashboard
Added to `daily_health_check.py`:
```
✅ LLM Pipeline: L0 dropped 12000/16500, L1 ok 380/400 (429: 0), L2 ok 12/15
   Cost: $0.32 (Text-01) + $0.45 (M2.5) = $0.77
   Avg tokens: L1=210, L2=890
   Cache hit rate: 23% (1100 / 4800 unique items)
   Retry queue: 3 pending
```

## Migration Plan (按风险递增)

1. **L0 rule classifier (week 1)** — extend `event_filter.py` with classify-not-just-rank
   - Don't disable existing L1 path; L0 first, anything not classified falls through
   - Add per-rule hit count log, verify rules work as expected
2. **Dedup cache (week 1)** — hash-based skip in L1 entry
3. **Daily cost dashboard (week 1)** — add health_check fields, no behavior change
4. **Persistent retry queue (week 2)** — 429s no longer dropped
5. **L2 escalation (week 2)** — new extractor module, route confidence < 0.4 + supply-chain hints
6. **Tune L0 rules** based on production traffic — adjust drop / direct / L1 / L2 ratios

## Open Questions / TBD

- [ ] **L0 rules — hardcoded vs learned**: 起步全 hardcoded，但后期是否值得训练一个轻量 classifier？取决于 L0 hit rate 真实分布。
- [ ] **L2 trigger threshold for L1 confidence**：0.4 是直觉值，需要 backtest 看 confidence 与 event 真实价值（5d forward return）的关系。
- [ ] **MiniMax actual RPM**：今晚 429 是 60 RPM × 16 workers bursty pattern 触发。需要单线程 sweep 测真实账户 RPM 上限。
- [ ] **Cache TTL**：相同 title 间隔多久还算"重复"？一天内必然 dedup，跨周可能内容已过期。
- [ ] **L2 模型选**：M2.5-highspeed vs M1 vs 其他厂商。M1 更深推理但贵。
- [ ] **Retry queue 持久化格式**：JSONL 还是 SQLite？规模 < 1000 items/day 倾向 JSONL 简单。
- [ ] **冲突解决**：L1 和 L2 对同一 news 给出不同 event_type 时，谁覆盖谁？默认 L2 覆盖。
- [ ] **Schema 升级**：events 表加 `tier_used` 字段以便 audit + 计费？建议加。

## Out of Scope (defer)

- 多 API key 轮换（如果有多个 MiniMax 账号）
- 跨 provider failover (OpenAI / DeepSeek 备份)
- 真正的 streaming pipeline（当前是日批就够）
- LLM fine-tuning（成本不值，extraction 任务通用模型够用）

## References

- Memory: [[llm-pipeline-architecture]] — 王总的架构指引 + 今晚验证数据
- Memory: [[project_audit_20260529]] — 数据卫生审计发现
- Code: `factors/event_filter.py` (待扩展为 L0)
- Code: `factors/llm_event_extractor_v2.py` (已是 L1 wire-ready, commit 07a5fb6)
- Code: `scripts/run_llm_event_pipeline.py` (orchestrator)
- Incident logs: `logs/llm_event_pipeline_manual_rerun.log` (today's rerun w/ 273 / 425 429s)
