# Weekly Report: 2026-05-24 ~ 2026-05-26

39 commits across a 2-day session. Focus: supply chain pipeline, production hardening, experiment validation.

---

## 1. Infrastructure Built

| Module | Description |
|--------|-------------|
| `scripts/run_network_job.py` | Network profile wrapper (domestic/global/llm/push) with proxy + timeout |
| `scripts/run_with_status.py` | Job status tracker — writes `job_status.json` for health checks |
| `factors/global_supply_chain_extractor.py` | LLM-free extraction of supply chain events from global news |
| `scripts/collect_global_industry_news.py` | GDELT + RSS global industry news collector (7 topics) |
| `scripts/extract_global_supply_chain_events.py` | Batch extractor: news JSONL -> chain events JSONL |
| `scripts/build_global_chain_factors.py` | Propagate chain events through edge table -> per-stock alpha |
| `factors/supply_chain_mapper.py` | Industry-level mapper: company 84 + industry 175 = 259 stock coverage |
| `data/config/supply_chain_edges.yaml` | Edge table: 130 edges, 78 stocks (expanded from 56/36) |
| `scripts/validate_global_chain_overlay.py` | Structural validation: overlay math works end-to-end |
| `scripts/shadow_supply_chain_overlay.py` | Daily shadow comparison: Top20 with/without chain overlay |
| `factors/event_filter.py` | Phase 4T-3: event surprise filter (dedup, novelty scoring) |
| `factors/freshness_gate.py` | Phase 4X: data freshness gate — block stale data from pipeline |
| `scripts/daily_health_check.py` | Reads `job_status.json` as primary source (fixed) |
| `scripts/predict_crash_daily.py` | Phase 4O: daily crash probability prediction |
| `factors/moneyflow_v2.py` | Phase 4L: moneyflow neutral factor (sector-demeaned) |
| `factors/event_store.py` | Phase 4N: 5 time fields for events + surprise factors |
| Phase 4E ensemble | DoubleEnsemble + 24-split ensemble training script (resumable) |
| Phase 4G factor inventory | Official feature path + factor inventory from CX review |

## 2. Experiments and Conclusions

| Experiment | Result | Action |
|------------|--------|--------|
| holder_decrease 24-split training | PASS — consistent improvement | Keep in ensemble |
| holder_decrease regime-weighted | FAIL — no incremental value over base | Dropped |
| DoubleEnsemble gate evaluation | Marginal — not worth complexity | Shelved |
| Supply chain overlay structural test | Plumbing OK — ranks shift as expected | Proceed to shadow |
| Global news via ShadowsocksX proxy | Works on HTTP port 10818 | Deployed to cron |
| MacroCollector RSS in domestic mode | Blocked cron (timeout) | Fixed: skip all RSS |

## 3. Production Fixes

| Bug | Fix | Impact |
|-----|-----|--------|
| Pending OMS crash on missing `prev_close` | Guard with `.get()` fallback | P0 — paper trading was broken |
| RiskGuard cross-contamination between runs | Isolated state per call | P0 — wrong risk signals |
| Morning cron timeout (840s > 300s limit) | Increased to 900s | P1 — missed morning push |
| MacroCollector hanging on RSS fetch | Total timeout 60s + skip in domestic | P1 — blocked downstream jobs |
| Global news cron used wrong network profile | `network=none` (direct works) | P1 — news not collected |
| Proxy port 10808 vs 10818 | Corrected to 10818 (HTTP bridge) | P1 — global news proxy |
| Health check reading logs instead of status | Primary: `job_status.json` | P2 — false alerts |
| Job display names missing in failure alerts | Added all 24 names | P2 — unclear alerts |
| Feature column selection bug (CX review) | Fixed naming + promote guard | P2 — wrong features |
| Global news partial save on crash | Streaming write: save after each topic | P2 — data loss on timeout |

## 4. Supply Chain Pipeline Status

**Status: Shadow-ready**

- Edge table: 130 edges covering 78 A-share stocks (semiconductor, EV, Apple chain, strategic materials)
- Industry mapper adds 175 more stocks via CSRC industry classification
- Strategic material chain added: rare earth, Ge, Ga, W, graphite (China chokepoints)
- 3 cron jobs deployed: news collect (16:25), event extract (16:50), factor build (17:10)
- Shadow overlay comparison runs daily at 18:45 (added this session)
- NOT in production path yet — requires multi-week shadow validation before promotion

## 5. Phase Status Update

| Phase | Status | Notes |
|-------|--------|-------|
| 4E Ensemble | Done | 24-split training, DoubleEnsemble evaluated |
| 4G Factor Inventory | Done | Official feature path defined |
| 4L Moneyflow | Done | V2 neutral factor, sector-demeaned |
| 4N Event Store | Done | 5 time fields + surprise factors |
| 4O Crash Model | Done | crash_prob in OMS + daily prediction |
| 4R Meta-filter | Proto | Multi-signal meta-filter prototype |
| 4T LLM Pipeline | Done | Event filter, V2 default, sector heat |
| 4U Supply Chain | Shadow | Pipeline complete, shadow comparison live |
| 4X Reliability | Done | Network wrapper, freshness gate, health tracking |

## 6. Next Priorities

1. **Shadow validation**: accumulate 2+ weeks of chain overlay data before considering promotion
2. **Factor ablation**: 229 factors fetched but rolling ablation not yet run (Phase 2 debt)
3. **Meta-filter 4R**: multi-signal filter needs real data validation
4. **Feature cache 175**: `build_feature_cache_175.py` script pending
5. **Weekly retrain**: confirm Saturday auto-retrain is stable
