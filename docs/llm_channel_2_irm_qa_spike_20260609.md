# LLM Channel 2 — Investor-Interaction Q&A (IRM cninfo + SSE e互动) SPIKE

**Date:** 2026-06-09
**Author:** spike scaffold
**Status:** DESIGN + SCAFFOLD, no LLM calls yet, no cron wire-up
**Files staged:**
- `scripts/collect_irm_qa.py` (fetcher)
- `factors/irm_qa_extractor.py` (LLM extractor class)
- `scripts/build_irm_qa_factors.py` (factor builder)

## 1. Why this channel

Investor Q&A on 深交所互动易 (`irm.cninfo.com.cn`) and 上证 e 互动
(`sns.sseinfo.com`) is the closest A-share analogue to US earnings-call Q&A
that is **(a) public, (b) free, and (c) daily**. Retail investors post
questions; management or IR officers reply. Unlike scheduled earnings calls,
this channel runs every business day, and management responses sometimes
contain real forward-looking content (guidance revisions, order updates,
capex timing) before they appear in periodic reports.

Existing pipeline already has 8 LLM extractors covering announcements, news,
guba sentiment, and 4 policy verticals. **None** target firm-management
narratives. This is a structurally different signal — neither news (which is
about the firm by external parties) nor announcements (which are
disclosure-mandated). Adding it diversifies the LLM factor stack into
firm-side communication.

## 2. Data source feasibility — VERDICT: GREEN

AKShare has first-class helpers for both venues — confirmed by introspection
+ live probe on 2026-06-09:

```
['stock_irm_ans_cninfo', 'stock_irm_cninfo', 'stock_sns_sseinfo']
```

### 2.1 `stock_irm_cninfo(symbol=<6-digit-code>)`

- Returns a **single DataFrame containing BOTH question and answer** for all
  recent Q&A on a stock. Probe on `002594` (BYD) → **320 rows × 14 columns**
  in 4.5s.
- Schema columns: `['股票代码', '公司简称', '行业', '行业代码', '问题',
  '提问者', '来源', '提问时间', '更新时间', '提问者编号', '问题编号',
  '回答ID', '回答内容', '回答者']`.
- `提问时间` = ask_time, `更新时间` ≈ answer_time, `回答内容` = answer text.
- No paging — the API returns recent history (~hundreds of rows) on each
  call. We filter on `answer_date == target_date` client-side.
- Conclusion: **the bulk-listing call is enough**. The
  `stock_irm_ans_cninfo` helper is for the single-question detail view; we
  do NOT need it for daily factor build.

### 2.2 `stock_sns_sseinfo(symbol=<6-digit-code>)`

- Same semantic surface for SSE stocks.
- Probe returned a `ReadTimeout` on `603119` (the AKShare default symbol)
  on 2026-06-09. The endpoint is slower / more flaky than cninfo — our
  scaffold treats it as a soft fallback per stock (try, fail silently,
  fall back to cninfo). No retry-loop within a single tick.
- **Risk:** SSE coverage is shakier. Empirical reliability TBD during the
  go-live soak (see §6).

### 2.3 Scraping fallback (not needed)

If AKShare ever drops support, the URL pattern is:
`https://irm.cninfo.com.cn/ssessgs/S?stockCode=<code>` (HTML) and
`https://sns.sseinfo.com/company.do?uid=<sse_uid>` (HTML). We do NOT
implement this in the spike; AKShare coverage is sufficient.

## 3. LLM extraction schema (decided)

Per-Q&A row (output of `factors.irm_qa_extractor.IRMQAExtractor.extract_single`):

| field                       | type            | notes                                                     |
|---                          |---              |---                                                         |
| `question_topic`            | enum 12 values  | guidance / orders / capex / product / management / regulation / ma_event / dividend / shareholding / operations / esg / other |
| `is_substantive`            | bool            | LLM judges if answer contains specific facts vs deflection |
| `information_value_score`   | int 1-5         | 5=hard number + dated commitment; 1=no reply / template    |
| `forward_signal_direction`  | int 1/-1/0      | sign of management's communication, NOT expected return    |
| `contains_guidance_change`  | bool            | explicit revision of earlier guidance                      |
| `contains_specific_number`  | bool            | number / date / percentage in answer                       |
| `summary_sentence`          | str ≤80 chars   | one-line factual summary                                   |
| `confidence`                | float [0,1]     | LLM self-assessment                                        |

**Joined with row metadata** when persisted: `stock_code, stock_name,
qlib_code, venue, question_id, industry, ask_date, answer_date,
is_answered, extract_date, extractor_version`.

**Post-LLM topic gate.** Mirrors `factors.event_schema_validator` — when
the LLM tags `orders` but neither question nor answer contains any
order-related keyword, downgrade to `other` with a reason string. Same
pattern that fixed the over-classification problem in the V2 event
extractor (project_audit_full_review_20260529).

## 4. Per-stock-per-day factors (decided)

Window: **5 calendar days** trailing the signal date. Reactive channel,
fast half-life.

| factor                              | definition                                                   |
|---                                   |---                                                            |
| `irm_qa_count_5d`                   | total Q&A in window                                           |
| `irm_qa_substantive_count_5d`       | only `is_substantive=True`                                    |
| `irm_qa_mean_info_value_5d`         | mean information_value_score (1-5)                            |
| `irm_qa_net_forward_signal_5d`      | Σ(forward_signal_direction × confidence)                       |
| `irm_qa_guidance_change_count_5d`   | count of `contains_guidance_change=True`                       |
| `irm_qa_dodge_rate_5d`              | (count - substantive_count) / count — "stonewall" share        |
| `irm_qa_topic_concentration_5d`     | Herfindahl on topic share — "everyone asking same thing"       |

Output parquet: `data/storage/irm_qa_factors.parquet` keyed by
`(datetime, instrument)` where `instrument` is **lowercase qlib_code**
(mandatory per `factors/feature_cache_utils.py` 2026-06-08 case-bug note).

## 5. PIT discipline

**Signal date = answer_date + 1 BDay**, NOT ask_date. Justification:
- The question is retail-side noise, not new information.
- The answer is when public information enters the market.
- The +1 BDay shift mirrors the LLM event V2 post-15:00 convention. The
  current scaffold applies it unconditionally (we don't preserve
  hour-of-day from `answer_time` in the collector). Conservative-late
  bias is fine; spurious-early would not be.

A question asked 2 weeks ago but answered today is a TODAY signal.

## 6. Expected daily volume + backfill cost

- Probe: BYD (top-1 retail attention stock) → 320 rows visible. Average
  liquid stock on these platforms answers 1-5 Q/day; small caps see
  0-1/week.
- Default universe = 300 most liquid stocks.
- **Expected daily volume**: ~600-1500 Q&A rows / day → after substantive
  filter, ~200-500 useful rows.
- Wall-clock fetch: 300 stocks × 1-2 calls × 0.5s pacing × 4 workers ≈
  **25 min/day**.
- LLM cost: 200-500 substantive rows/day × ~400 tokens/row ≈ 80k-200k
  tokens/day. MiniMax-Text-01 pricing puts that at well under $1/day.
- Backfill: AKShare returns ~recent history per call (not a date-range
  API). We can NOT cleanly backfill — bulk listing currently shows
  ~recent N hundred Q&A regardless of historical window. Backfill
  strategy: collect daily from go-live forward. **No useful 3-month
  pre-history available**; first IC test will need a 30-day soak.

## 7. Expected IC reasoning

No published research on A-share IRM/SNS Q&A specifically — this is
greenfield. Three indirect literature priors support a real but small
effect:

1. **Earnings call Q&A linguistic signal** (Larcker & Zakolyukina 2012,
   Hassan et al. 2019 on conference call deception/uncertainty): the
   linguistic register of management Q&A predicts subsequent returns and
   volatility at the firm level. The mechanism (information revelation
   under non-rehearsed pressure) carries over.
2. **A-share retail-investor sentiment as proxy** (Da, Engelberg, Gao 2011
   on Google search; Chinese guba studies replicating it 2018-2021):
   retail attention itself predicts short-horizon returns but flips sign at
   longer horizons. The Q&A *volume* factor (`count_5d`) likely has the
   same shape — useful at 1-3 day horizons.
3. **Disclosure quality & cross-section of returns** (Botosan 1997; Chinese
   replication Lin & Wei 2014): IR responsiveness correlates with
   subsequent earnings surprise — `dodge_rate_5d` should negatively
   predict surprise direction.

**My priors for IC vs T+5 next-trading-day returns:**

- `irm_qa_substantive_count_5d`: |IC| ~ 0.005-0.015 (small, attention proxy)
- `irm_qa_dodge_rate_5d`: |IC| ~ 0.01-0.02 (medium — disclosure-quality
  literature priors)
- `irm_qa_guidance_change_count_5d`: |IC| ~ 0.02-0.04 conditional on
  count>0 (large but **sparse** — most days have zero)
- `irm_qa_net_forward_signal_5d`: |IC| ~ 0.005-0.020 (depends heavily on
  the LLM's direction-call quality)

These are **per-channel** ICs. The realistic add to the existing factor
bank after orthogonalization is roughly half of the raw |IC| above. Even
a +0.005 add to the regularized model is worth the daily $1 + 25 min
compute on a long-run paper-trading basis.

## 8. Risks

1. **Sparse small-cap coverage.** Below the top-500 universe, weekly Q&A
   counts collapse to 0-1. Universe pre-filter (top-300 by liquidity)
   exists; reviewers may want a "no-zeros" backfill from the larger
   set the FeatureMerger will fill missing keys with 0 anyway, so this is
   not a hard correctness issue, but it is a factor-design issue (most
   stocks always at the mean → low cross-sectional spread).
2. **IR template dilution.** A material fraction of replies (~30-50%
   estimated from manual sampling of 002594) are pure boilerplate. The
   `is_substantive` filter and the topic gate should handle this, but
   they only work if the LLM accurately distinguishes. The
   `dodge_rate_5d` factor turns this risk into a feature — but only at
   the per-stock relative level.
3. **SSE e-互动 endpoint flakiness.** The probe timed out. Per-symbol
   failures are silently dropped; if a structural outage hits, the
   factor would silently lose SSE coverage for half the universe. The
   health gate (`irm_qa` source) catches volume drops; an explicit
   per-venue counter belongs in v2.
4. **Lookahead from `更新时间`.** The `更新时间` field in the cninfo
   response is "last update time" — if a Q&A is edited later, the
   timestamp shifts. We treat it as answer_date; the +1 BDay shift gives
   us buffer, but a `merge` of edited rows could re-emit signals for
   past dates. Mitigation: dedupe by `(stock_code, question_id)` is
   already in place; first-write-wins downstream of the JSONL.
5. **Lookahead from collector run time.** If the collector runs at 09:00
   it sees only some of "today's" answers. Cron must run after market
   close (≥ 16:00 CST) and downstream consumes `answer_date +1 BDay`.
6. **Greenfield IC.** Channel has zero historical backfill. First go/no-
   go after 30d soak + ablation against the existing LLM event factors.

## 9. Go / No-go

**GO** for the SPIKE→soak transition with the following gates before
production wire-in:

- [ ] 1-day pilot collection on top-300 universe → confirm ≥ 500 Q&A
      rows collected, ≥ 80% with non-empty answer field.
- [ ] LLM extraction on 200-row sample → manual review of 30 rows,
      ≥ 25/30 topic+substantive correct.
- [ ] 30-day daily collection → first IC test against
      `realized_return_5d`, single-factor and incremental over
      `llm_event_factors`.
- [ ] Add `irm_qa` source to `scheduler/cron_critical_sources.json`.
- [ ] Add freshness gate entry to `scheduler/data_health` profile
      `ashare`.

**NO-GO triggers** during soak:
- Substantive row count < 50/day after week 2.
- Single-factor |IC| < 0.005 on any of the 7 factors after 30d.
- SSE e-互动 endpoint outage > 5 consecutive business days.

## 10. Files & line pointers

| concern               | file                                            |
|---                    |---                                              |
| fetcher               | `scripts/collect_irm_qa.py`                     |
| LLM extractor         | `factors/irm_qa_extractor.py`                   |
| factor builder        | `scripts/build_irm_qa_factors.py`               |
| spike doc (this)      | `docs/llm_channel_2_irm_qa_spike_20260609.md`   |

Reference / pattern sources used:
- `scripts/collect_daily_news.py` — universe resolution, ST_CLIENT pattern.
- `scripts/collect_announcements.py` — qlib_code lowercasing convention.
- `factors/llm_event_extractor_v2.py` — LLM rate-limit + retry-queue plumbing.
- `factors/event_schema_validator.py` — post-LLM keyword gate pattern.
- `scripts/build_policy_factors.py` — per-instrument factor parquet pattern,
  atomic write, health sidecar.
- `scripts/build_llm_event_factors.py` — PIT post-15:00 BDay shift convention.
- `factors/feature_cache_utils.py` — lowercase canonical instrument case.
- `scheduler/data_health.py` — `HealthStatus` + `write_health` API.
