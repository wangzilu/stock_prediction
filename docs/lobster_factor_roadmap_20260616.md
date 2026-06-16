# ST_CLIENT Lobster Tier — Factor Roadmap

**Date:** 2026-06-16
**Status:** lobster token verified, 26/33 priority endpoints work via the new client. Integration deferred to dual-client setup (#195).

## Sweep results — 33 factor-candidate endpoints

Probed with the lobster client (`proxy/调用方式1：ST_CLIENT(14行替换自己的token).py`) + lobster token. All ✅ entries return real data (verified on 2026-06-15 / 2026-06-16).

### Tier 1 — academic first-class (highest priority for production wiring)

| Endpoint | Factor use | Status |
|---|---|---|
| `forecast` | **SUE** (Standardized Unexpected Earnings) / PEAD — Bernard & Thomas 1989, Hou-Xue-Zhang q-factor | ✅ |
| `forecast_vip` | Same, VIP bulk endpoint | ✅ |
| `express` | 业绩快报 — early earnings signal between forecast and full report | ✅ |
| `dividend` | DP (dividend yield) factor / 红利策略 baseline | ✅ |

### Tier 2 — corporate-action signals

| Endpoint | Factor use | Status |
|---|---|---|
| `repurchase` | 回购公告 — insider conviction signal | ✅ |
| `share_float` | 限售解禁 — selling-pressure / liquidity factor | ✅ |
| `stk_holdertrade` | 大股东增减持 — same scope as **LLM-Ext #3** but structured | ✅ |
| `pledge_stat` / `pledge_detail` | 股权质押 — tail-risk factor for distressed names | ✅ |

### Tier 3 — flow / sentiment / attention

| Endpoint | Factor use | Status |
|---|---|---|
| `top_list` | 龙虎榜个股 — concentration / "smart money" signal | ✅ |
| `top_inst` | 龙虎榜机构 — institutional net buying | ✅ |
| `hm_detail` | 游资明细 — hot-money tracking | ✅ |
| `hm_list` | 游资列表 — hot-money universe | ✅ |
| `stk_premarket` | 集合竞价 — pre-open sentiment | ✅ |
| `kpl_concept` / `kpl_list` | 开盘啦概念 / 榜单 — theme attention | ✅ |
| `limit_cpt_list` | 涨停连板 — momentum extreme | ✅ |
| `stk_surv` | 调研 — institutional interest tracker | ✅ |

### Tier 4 — research / analyst

| Endpoint | Factor use | Status |
|---|---|---|
| `report_rc` | 研报评级 — same scope as **LLM-Ext #1** but structured | ✅ |
| `broker_recommend` | 券商推荐月度 | ✅ |
| `cctv_news` | 央视新闻每日 — same scope as XWLB but ST mirror | ✅ |
| `major_news` | 主要新闻 — overlap with global_industry_news | ✅ |

### Tier 5 — fundamentals deepening

| Endpoint | Factor use | Status |
|---|---|---|
| `fina_indicator` | 财务指标 — augments current fundamental_update | ✅ |
| `fina_mainbz` | 主营业务构成 — segment-level revenue mix | ✅ |
| `anns_d` | 公告 — coverage of disclosure date | ✅ |
| `stk_managers` | 管理层人员 — for stk_surv linkage | ✅ |
| `index_dailybasic` | 指数日基础 — for universe construction | ✅ |

### Excluded — broken or out of scope (7 endpoints)

| Endpoint | Why | Likely fix |
|---|---|---|
| `news` | HTTP 400 with current params | Probe kwarg shape |
| `index_member_all` | HTTP 400 | Wrong kwarg name (try `l1_code`/`l2_code`/`l3_code`?) |
| `index_weight` | NO_VALID_KW | Need date or index_code combo |
| `cyq_chips` | HTTP 400 | Probably needs `trade_date` only OR `ts_code` only, not both |
| `fina_indicator_vip` | NO_VALID_KW | Period format wrong? |
| `stk_rewards` | code=None empty | Try without args / different period format |

## Integration blockers (NOT solvable tonight)

The lobster client returns **bare lists** from gateway-wrapped endpoints (vs the old client's `{code, data: {items}}` envelope). Replacing `ST_CLIENT.py` wholesale breaks every existing consumer (`update_regime_daily.py`, `fetch_st_daily_factors.py`, etc.). The safe path is **dual-client routing** (task #195):

1. Keep `ST_CLIENT.py` (old form-encoded client) as production primary.
2. Add `ST_CLIENT_LOBSTER.py` (new JSON-bodied client) only for lobster-only endpoints.
3. Per-endpoint router decides which client to invoke based on a registry.
4. Estimate: 3-4 hours for setup, then ~1 hour per factor wired in.

## Recommended production rollout (post-dual-client)

| Phase | Endpoints | Factor delivered | Lead time |
|---|---|---|---|
| **A** | `forecast` + `forecast_vip` + `express` | SUE / PEAD score | 1 week (need 30+ days of data for cross-sectional rank) |
| **B** | `dividend` | DP factor (cross-section + 12m rolling) | 3 days |
| **C** | `stk_holdertrade` | Insider net change (5-day rolling) — **replaces LLM-Ext #3 production path** | 5 days |
| **D** | `report_rc` | Analyst revision intensity — **replaces LLM-Ext #1 production path** | 5 days |
| **E** | `repurchase` + `share_float` | Corporate-action overlay | 1 week |
| **F** | `top_list` + `top_inst` + `hm_detail` | 龙虎榜 / 游资 sentiment | 2 weeks (need rolling baseline) |
| **G** | `pledge_*` | Tail-risk score | 1 week |
| **H** | `kpl_*` + `limit_cpt_list` + `stk_premarket` | Attention / momentum | 2 weeks |

## Replaces LLM-Ext effort

Two LLM extractor jobs become redundant once their structured equivalents are wired in:

- **LLM-Ext #1 (研报评级抽取)** → `report_rc` structured pull. Cost: 0 LLM tokens vs ~$2-5/day in LLM API spend.
- **LLM-Ext #3 (高管/大股东增减持)** → `stk_holdertrade` structured pull. Same savings.

Recommend keeping LLM-Ext #1/#3 in shadow mode for 2 weeks after the structured factor lands, to validate consistency. Then deprecate.

## Open questions

1. **realtime_list 参数契约** — the lobster client passes `{}` and server returns "参数不能为空". Need to look at the lobster client's full set of realtime_* methods + try `src=em`/`src=sina`/`src=ths` body params with the JSON-body endpoint, not the legacy form-encoded one.
2. **Storage layout** — each new factor needs a parquet (`data/storage/st_forecast.parquet` etc.) and a FeatureMerger group. Decide if these go into the production profile or a candidate profile first.
3. **PIT discipline** — `forecast` and `express` have an `ann_date` field. Must use that as `asof_time`, not the report period end. Existing `models/feature_merger.py` PIT pattern for `fundamental_update` is the reference.
