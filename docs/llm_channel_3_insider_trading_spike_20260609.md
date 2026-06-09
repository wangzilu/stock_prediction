# LLM Channel 3 — Insider Trading Announcements — Spike

**Date**: 2026-06-09
**Status**: SCAFFOLD ONLY (no LLM calls executed, no parquet written)
**Author**: Channel 3 spike

## 1. Why a dedicated channel

The generic V2 event extractor (`factors/llm_event_extractor_v2.py`) carries
an `insider_buy` / `insider_sell` enum out of 29 event types. In production
this yields ~5–10 insider events per day across the whole A-share universe
because:

1. The single `direction` enum buries holder type. A `控股股东 减持` (very
   strong negative signal per Lakonishok-Lee 2001) and a `董秘 减持` (weak,
   often tax-driven) collapse to the same `insider_sell` row.
2. The `magnitude_value_wan` field is one float — it cannot represent both
   "shares changed" and "% of own holding sold", and the literature shows
   the latter is the cleaner alpha (a 控股股东 减持 1% of company that
   represents 30% of own position is much stronger than a 控股股东 减持
   1% of company that is 1/100 of own position).
3. Insider buys get diluted in the LOO ablation because they share a
   factor channel with 27 unrelated event types.

Lifting insider events into Channel 3 lets us emit holder-type-decomposed
factors at full resolution and run an isolated LOO test against the
209-cache.

## 2. Data source feasibility — VERDICT: GO

### 2.1 Primary path (chosen)

**Filter from the existing Eastmoney 公告 stream**
(`scripts/collect_announcements.py`).

Measurements (single-day sample, 2026-06-09):

| Metric | Value |
|---|---|
| Total announcements collected | 1,924 |
| Insider-like (V0 keyword filter) | 46 |
| Filter precision (eyeballed top-20) | ≥ 90 % |
| Stocks impacted | ~ 35 unique |

Extrapolated yearly volume: 250 × ~50 = **~12,500 insider events/yr**.
Plenty for the LOO and cross-sectional tests, well above the ~500-event
floor where IC stabilises.

**Reuses zero new infrastructure**: the upstream announcement collector
already runs on the daily cron with manifest + atomic-replace + ChunkedEncodingError
recovery, and the SLA gate is already registered for it in `config/data_sla.py`.

### 2.2 Secondary / cross-check sources (AKShare, NOT used in this spike)

Probed:

- `stock_ggcg_em(symbol="股东减持")` — Eastmoney 高管持股 endpoint.
  Confirmed accessible (~223 pages of detail) but flaky:
  `ChunkedEncodingError: Response ended prematurely` on the test run.
- `stock_hold_management_detail_em()` — 董监高及相关人员持股变动明细.
  Also confirmed accessible (~338 pages of detail), same chunking
  fragility.
- `stock_share_hold_change_sse / _szse / _bse` — 沪深北 disclosures.
  Confirmed per-stock API; not adopted because querying for the
  universe of ~5300 stocks would crush the rate limiter.
- `stock_shareholder_change_ths` — 同花顺. Available but unverified.

**Decision**: defer AKShare to a Phase 2 cross-check (back-fill numerics
where the LLM extracted 0.0 due to ambiguous announcement text).

### 2.3 Why NOT 巨潮资讯网 directly

巨潮 (`cninfo.com.cn`) is the canonical primary source but has no public
JSON API; scraping the announcement detail PDFs would add ~3–5s per
announcement, blow the daily budget, and reintroduce the parsing problems
the LLM channel is supposed to abstract over. Eastmoney's 公告 API
already mirrors 巨潮 with a ~3-second lag.

## 3. Schema (decided)

Per-announcement output (after LLM extraction):

```jsonc
{
  "ts_code": "SZ002444",
  "announce_date": "2026-06-09",
  "action": "减持",           // 增持 | 减持 | 新进 | 协议转让出让 | 协议转让受让 | 被动稀释 | 其他
  "holder_type": "控股股东",  // 实控人 | 控股股东 | 董监高 | 持股5%以上股东 | 战略投资者 | 普通股东 | 外部机构 | 其他
  "holder_name": "巨星科技集团有限公司",
  "shares_changed": -1234.5,             // 万股, signed
  "pct_of_company": 0.5,                 // % (0.5 = 0.5 %)
  "pct_of_holder_position_change": 12.0, // % of holder's own holding sold
  "is_committed_no_sell": false,
  "reason_disclosed": "个人资金需求",
  "price_band": "≥¥18.50",
  "summary_sentence": "控股股东计划集中竞价减持不超过1234.5万股，占总股本0.5%。",
  "confidence": 0.92,
  "is_official_disclosure": true
}
```

PIT contract: `announce_date` is the signal day. Execution day is
`announce_date + 1 BDay` (matches the convention already used in
`scripts/build_event_factors.py` for the 业绩预告 / 龙虎榜 streams). Post-15:00
announcements (the majority — exchange disclosures batched after market
close) get the +1 BDay shift by default; the rare pre-09:30 disclosure
also gets +1 BDay so the convention is uniform and audit-checkable.

## 4. Factor surface (decided)

Built by `scripts/build_insider_factors.py`. Per (qlib_code, signal_date):

| Factor | Window | Aggregation | Lit. justification |
|---|---|---|---|
| `insider_net_buy_5d_pct` | 5d | Σ signed pct_of_company | Seyhun 1986 short-window |
| `insider_net_buy_20d_pct` | 20d | Σ signed pct_of_company | Lakonishok-Lee 2001 |
| `insider_buy_count_20d` | 20d | count(action ∈ buy) where conf ≥ 0.7 | event frequency channel |
| `insider_sell_count_20d` | 20d | count(action ∈ sell) where conf ≥ 0.7 | event frequency channel |
| `has_controlling_holder_sell_20d` | 20d | bool(实控人/控股股东 减持) | Cohen-Malloy-Pomorski 2012 quality channel |
| `has_strategic_buy_20d` | 20d | bool(战略投资者 增持/新进) | Brav-Jiang-Kim 2008 (activist signals) |
| `has_committed_no_sell_20d` | 20d | bool(commitment present) | management confidence weak proxy |
| `insider_event_count_5d` | 5d | unfiltered count | attention channel |

## 5. Expected IC reasoning

Per the academic literature on insider trading + the Channel-1/2 LOO
deltas we have already measured in this codebase:

- **Seyhun (1986)**: insider purchases predict 4–5 % abnormal returns
  over 12 months (US sample). Same direction holds in Lakonishok-Lee
  (2001), which extends to controlling-shareholder transactions.
- **A-share specifics**: Liu, Zhang & Zhao (2017) on 高管 trading shows
  IC ≈ 0.03–0.05 on 20-day forward returns, **larger for sell-side
  controlling-shareholder events** (~0.06) and **near-zero for low-level
  董秘 / 副总 events** (~0.01) — exactly the holder-type decomposition
  Channel 3 surfaces.
- Likely IC on this codebase: **0.02–0.04 RankIC on 5-day forward**, with
  the bulk of signal in `has_controlling_holder_sell_20d` and
  `insider_net_buy_20d_pct`. The flat-attention `insider_event_count_5d`
  is expected to be a wash but is included as the regime-aware overlay
  uses event-count channels for the "uncertain" regime.

**Bar for shipping**: LOO RankIC delta on 209-cache ablation
≥ **+0.005** vs the current Channel-1/2 stack, with the per-regime
breakdown showing ≥ +0.01 in `trend_down` regime (the regime where
controlling-holder sells are most predictive — see
`memory/feedback_regime_first_architecture.md`).

## 6. Files created (this spike)

- `scripts/collect_insider_announcements.py` — V0 keyword filter on top of
  the existing announcement stream. Fully runnable; no LLM dep.
- `factors/insider_trading_extractor.py` — Channel-3 LLM extractor.
  Reuses `LLMEventExtractorV2._call_llm` for transport (rate-limit, 429
  backoff, retry queue). NOT yet exercised.
- `scripts/build_insider_factors.py` — Per (stock, date) factor builder
  with PIT cutoff. Wired but produces empty parquet until the extractor
  has been run.
- `docs/llm_channel_3_insider_trading_spike_20260609.md` (this doc).

## 7. Risks

1. **Stale planned-but-unexecuted disclosures.** Under 减持新规 (2023)
   controlling holders publish a 减持预披露 6 months before any
   actual sale. The "predictive" content of the announcement is largely
   exhausted on day 1 of the plan, but the actual reduction can come
   over 90 days. The V0 filter excludes "计划期满未实施" titles, but
   the 减持新规 still leaks signal across the window. Plan-vs-execution
   separation is a Phase 2 work item.
2. **ETF / passive issuer dumps mis-tagged as insider.** When a 持股5%
   以上股东 is an ETF issuer rebalancing the basket, the `action="减持"`
   is mechanical, not informed. The LLM is asked to tag these as
   `holder_type=外部机构` and downstream factors exclude `外部机构` from
   `has_controlling_holder_sell_20d`, but this depends on the LLM
   correctly identifying issuer names. Audit needed in Phase 1.
3. **被动稀释 noise.** Recent (2026) announcements include a flood of
   "被动稀释" rows where the holder's % dropped because the company
   issued new shares — the holder did not sell. Direction sign is 0 in
   `DIRECTION_BY_ACTION`, but if the LLM mis-tags this as `减持` the
   `insider_net_buy_20d_pct` factor is contaminated. Mitigation: keyword
   gate on title ("被动稀释" forces action="被动稀释" even if LLM emits
   otherwise — TODO for Phase 1).
4. **Coverage gap on 北交所 (BJ) names.** Eastmoney's 公告 API has
   thinner BJ coverage than SH/SZ. The Phase 1 audit should compare
   counts against `stock_share_hold_change_bse` to size the gap.
5. **Holder name normalisation.** Same controlling holder appears as
   "巨星科技集团有限公司" / "巨星集团" / "巨星控股" across announcements.
   Without name dedup, multi-tranche reduction shows up as N independent
   events and inflates `insider_sell_count_20d`. The current SPIKE does
   NOT dedup. Name normalisation = Phase 2.
6. **Confidence floor is a hyperparameter.** `CONFIDENCE_FLOOR_COUNT = 0.7`
   was picked by reading the V2 confidence distribution; not yet validated
   on Channel-3-specific data. Sweep in Phase 1 IC test.

## 8. Go / No-Go

**GO**, conditional on Phase 1 audit:

1. Hand-label 100 random insider announcements over 5 distinct trading
   days. Measure LLM precision per (action, holder_type) cell.
   **Bar**: ≥ 85 % macro-precision on action, ≥ 75 % on holder_type.
2. Run the extractor on the same 5 days, produce factor parquet,
   reindex onto a small (3-month) slice of the 209-cache, run a
   single-factor IC test against forward 5d return.
   **Bar**: at least one of `insider_net_buy_20d_pct` or
   `has_controlling_holder_sell_20d` shows RankIC ≥ |0.02| on the
   3-month slice.
3. If both bars pass: wire into daily cron, register in
   `config/data_sla.py` (SLA: 24h after market close), add to the 209
   feature-cache joiner per
   `memory/feedback_offline_join_case_bug.md` (i.e. use
   `feature_cache_utils.normalize_instrument_index` + `assert_join_coverage`).
4. If either bar fails: park the SPIKE. Cost so far: ~¥15 (250-day
   backfill at worst case). Cheap.

## 9. Blockers

None. All upstream dependencies exist:

- announcement collector — already in cron.
- LLM transport — `LLMEventExtractorV2._call_llm` already exposes the
  `system_prompt` kwarg (cx review P1 #2 fix from 2026-06-07).
- feature-cache safety primitives — `normalize_instrument_index` and
  `assert_join_coverage` already in production (see
  `memory/feedback_offline_join_case_bug.md`).

## 10. Out of scope for this spike

- Hand-labelled ground truth (Phase 1).
- Holder-name normalisation / fuzzy dedup (Phase 2).
- AKShare cross-check on numerics (Phase 2).
- Plan-vs-execution disclosure separation under 减持新规 (Phase 2).
- 北交所 coverage parity (Phase 2).
- IC backtest + LOO delta on 209-cache (Phase 1 bar).
