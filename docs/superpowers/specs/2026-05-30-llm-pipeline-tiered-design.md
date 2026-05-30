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
- **Input:** raw merged news + announcements (yesterday: 16509 items, of which 15000 news + 1509 announcements)
- **Output:** `{drop, direct_classify, route_to_l1, route_to_l2}` per item
- **Direct classify rules** (already implemented 5/29, conservative high-confidence patterns):
  - `撤销退市风险警示|撤销.*风险警示|摘帽` → `routine_announcement, direction=+1, conf=0.92`
  - `关于.*问询函的回复|问询函回复` → `routine_announcement, direction=0, conf=0.90`
  - `召开.*股东大会.*通知|股东大会决议公告` → `routine_announcement, direction=0, conf=0.88`
  - `独立董事(关于|的)|独董(声明|任职)` → `routine_announcement, direction=0, conf=0.88`
  - `日常关联交易|常规关联交易` → `routine_announcement, direction=0, conf=0.88`
  - `更正公告|补充公告` → `routine_announcement, direction=0, conf=0.85`
  - `回购.*(进展|结果)公告|关于.*累计回购` → `share_buyback, direction=+1, conf=0.88`
  - `限售股.*(解除限售|上市流通)` → `share_unlock, direction=0, conf=0.85`
- **Drop rules:**
  - title 长度 ≤ 7 字符 → `drop:title_too_short`
  - 标题正则 `^(今日|盘中|尾盘|早盘)[价涨跌幅\d\s\.\%]+$` → `drop:pure_price_chatter`
- **L2 hint patterns** (mark only, still flow through L1):
  - `产业链|供应链|上下游|卡脖子|关键材料|关键技术`
  - `出口管制|反制裁|进口替代|国产替代|断供`
  - 标题含 ≥2 个 6 位股票代码（多主体）
- **Dedup hash:** `sha1(stock_code[-6:] + title[:60] + source + publish_date[:10])` — implemented as `_content_hash()` in event_filter.py, persisted to `data/storage/llm_event_cache/seen.jsonl`
- **Implementation decision:** hardcoded rules (5/29 smoke test on 5 samples: 2 direct/2 L1/1 drop with 1 L2 hint). Coverage measurement defers to first production run (Monday 16:30 cron after install_crontab.py applied today)

### L1 — Cheap non-reasoning extraction (`factors/llm_event_extractor_v2.py`)
- **Model:** `MiniMax-Text-01` (wired commit 07a5fb6)
- **Prompt:** strict extraction (wired commit 07a5fb6, 0 reasoning tokens verified)
- **Measured per-call cost (5/29 rerun on 425 calls):**
  - avg total tokens: **208** (prompt 168 + completion 40)
  - prompt total: 71545 tokens
  - completion total: 16732 tokens (of which reasoning_tokens=0)
  - cost: ~$0.034 (Text-01 @ $0.20/$1.10 per M tokens)
- **Measured 429 rate:** 273 of 425 calls = **64% rate-limited** with 16 workers × 60 RPM rate limiter. Single-threaded probe in earlier ping confirmed Text-01 endpoint healthy.
- **Concurrency recommendation:** drop to 4 workers + jittered backoff (5/15/45s, already implemented commit 34bf8ef wave 3). Measure RPM ceiling on next live run.
- **Per-call max_tokens:** 512 default (sufficient — completion topped 107 tokens in probe samples). Bump to 1024 only for items L0 marked as long content (`>500字 announcement`).
- **Output:** structured event JSON → EventStore (commit 0d705d8)

### L2 — Reasoning extraction (NEW: `factors/llm_event_extractor_l2.py`)
- **Model:** `minimax-m2.5-highspeed` (5/29 verified: 952 total tokens/call, 489 reasoning. 5x cost of L1).
- **Triggers:**
  - L0 hint set: multi-entity / supply-chain / 卡脖子 — already marked by L0
  - L1 confidence < 0.5 (conservative initial threshold; tune after first 30 days of L1 production with confidence/IC backtest)
  - L1 detected event_type ∈ {`tech_breakthrough`, `restructuring`, `lawsuit_filed`, `strategic_cooperation`, `joint_venture`} — categories where impact direction needs context
- **Prompt:** allows reasoning, asks for evidence + sub-events + impacted parties
- **Output schema:** extends V1 with `related_stocks: list`, `transmission_path: list`, `evidence_spans: list[str]`
- **Rate budget:** stricter cap (~10 RPM) since reasoning calls 5x more expensive AND M2.5 specifically hits the same MiniMax account quota as L1

### Rate limiter + retry queue
- **Per-tier RPM cap (initial, refine after Monday production):**
  - L1: 30 RPM (4 workers × 7.5 calls/min each, conservative vs measured 64% 429 at 16×60)
  - L2: 10 RPM (single worker, since reasoning calls are 5x expensive and same quota)
- **Jitter:** ±30% multiplier on backoff (implemented in V2 commit 34bf8ef wave 3)
- **Retry on 429:**
  - Exponential backoff: 5s → 15s → 45s with ±30% jitter (implemented)
  - After 4 attempts, **write to persistent retry queue** `data/storage/llm_retry_queue/{date}.jsonl` (not yet implemented — week 2 work)
  - Retry queue drain job: cron at 22:30 (after main pipeline + cron quiet hours)
- **Circuit breaker:** if 5 consecutive 429 across all workers within 30s, pause 60s (not yet implemented)

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

- [x] **L0 rules — hardcoded vs learned**: hardcoded 起步，8 direct rules 已实施。learned classifier 推到 4 周后看真实 production hit-rate 决定。
- [ ] **L2 trigger threshold for L1 confidence**：暂用 0.5（保守），需要 30 天 L1 生产 confidence × 5d forward return backtest 后调整。
- [x] **MiniMax actual RPM**：5/29 实测 16 workers × 60 RPM 命中 64% 429。下调到 4 workers + 30 RPM 后下次 production 重测。
- [ ] **Cache TTL**：相同 title 间隔多久还算"重复"？一天内必然 dedup，跨周可能内容已过期。当前实现：永久去重，写到 `seen.jsonl` 不过期。1 个月后看 cache 大小决定是否加 TTL。
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
