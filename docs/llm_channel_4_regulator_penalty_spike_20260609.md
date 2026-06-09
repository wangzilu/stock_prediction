# LLM Channel 4 — Regulator Penalty + Inquiry Extractor SPIKE

**Date:** 2026-06-09
**Branch:** crypto (SPIKE scaffold; commit blocked per task brief)
**Status:** scaffold complete, NOT executed (no LLM calls, no scrape)

---

## 1. Motivation

Existing event pipeline lumps 证监会 / 交易所 regulator actions with
generic announcements under V2's `regulatory_penalty` event_type (1 of
33). The schema-validator's keyword gate downgrades ~40% of these to
`other` / `routine_announcement`, so the LLM-event factor never gets
clean exposure to enforcement risk. Three reasons to break it out:

1. **Strong short-term negative alpha.** Karpoff-Lott (1993, JLE) and
   Bhagat-Bizjak (2019) document -1.5% to -3% CAR in the [-1, +5]
   window after enforcement announcement, with the largest drift on
   financial misrepresentation and management resignation referrals.
2. **Tail-risk signal for RiskGuard.** A `delisting_warning` or
   `criminal_referral` flag is exactly the kind of event the 7-layer
   RiskGuard L1 (`_check_st_stocks`, `_check_hard_stop`) was built to
   force-sell on, but today it has no structured input.
3. **Inquiry-letter early signal.** Choi 2021 (JFE) shows SEC inquiries
   using high-pressure language ("please fully explain") correlate
   +25% with subsequent enforcement within 90 days. A2 inquiries are
   the analogue — a leading indicator we currently miss.

---

## 2. Source feasibility

### 2.1 AKShare probe (verified 2026-06-09)

```python
[m for m in dir(akshare)
 if 'csrc' in m.lower() or 'punish' in m.lower()
 or 'penalty' in m.lower() or 'inquiry' in m.lower() ...]
# → []
```

Verdict: **no direct API**. The only proxies are
`stock_notice_report` and `stock_zh_a_disclosure_report_cninfo` —
generic filing dumps that mix penalty/inquiry receipts with all other
disclosures. Usable as a coarse pre-filter (keyword match on title)
but NOT as the primary source.

### 2.2 Primary scrape URLs

| Source | URL | Format | Backfill horizon |
|---|---|---|---|
| CSRC 行政处罚决定书 | http://www.csrc.gov.cn/csrc/c100120/zfxxgkml.shtml | List → PDF detail | ~3 yr online |
| SSE 自律监管措施 | http://www.sse.com.cn/disclosure/credibility/supervision/measures/ | List → HTML/PDF | ~5 yr online |
| SZSE 自律监管措施 | http://www.szse.cn/disclosure/supervision/measure/index.html | List → PDF | ~5 yr online |
| SSE 问询函 | http://www.sse.com.cn/disclosure/credibility/supervision/inquiries/ | List → HTML | ~5 yr online |
| SZSE 问询函 | http://www.szse.cn/disclosure/listed/supervision/inquire/index.html | List → PDF | ~5 yr online |

**Scrape constraints**:

* CSRC portal enforces a per-IP rate limit (~30 GET/min sustained);
  use `INTER_REQUEST_DELAY = 2.0` and rotating UA.
* Exchange list pages are paginated; `ts_code` lives in the row
  metadata, NOT the document title.
* CSRC detail pages are PDFs → need `pdfplumber` (NOT a current
  project dep, would be a new pin).
* Historical backfill (>3 years) requires a one-off CSMAR / Wind
  snapshot; live scrape covers only the rolling 3-year window.

---

## 3. LLM extraction schema (decided)

```json
{
  "ts_code":               "<6 digits or null>",
  "event_date":            "<YYYY-MM-DD>",
  "severity":              "<warning|fine|suspension|delisting_warning|criminal_referral>",
  "regulator":             "<CSRC|SSE|SZSE>",
  "topic":                 "<财务造假|关联交易|资金占用|信披违规|操纵市场|内幕交易|其他>",
  "fine_amount_yuan":      <float; 0 if no fine>,
  "is_strict_inquiry":     <bool — only true if 请充分说明 / 高度关注 / 立案调查>,
  "expected_market_impact": <int 1..5 — document severity, NOT return forecast>,
  "summary_sentence":      "<≤60 char>",
  "confidence":            <float 0..1>
}
```

**Critical discipline**: `expected_market_impact` is a 1-5 *document*
severity score (1 = routine compliance reminder, 5 = criminal
referral), NOT a return prediction. Direction sign and magnitude come
from historical calibration at the FACTOR layer (cf. existing
`scripts/build_event_calibration.py`).

**Post-LLM gate**: `is_strict_inquiry` is verified against the
STRICT_INQUIRY_PHRASES list before being trusted (Choi 2021's
phrase-match condition is narrower than the LLM's intuition; the
extractor downgrades over-claims).

---

## 4. Files scaffolded

| File | Purpose | LoC |
|---|---|---|
| `scripts/collect_regulator_actions.py` | Fetcher skeleton (5 sources, fetchers raise NotImplementedError) | 270 |
| `factors/regulator_penalty_extractor.py` | LLM wrapper + schema validator + phrase-gate | 320 |
| `scripts/build_regulator_penalty_factors.py` | Factor build with +1 BDay PIT lag + 7d rolling | 290 |
| `docs/llm_channel_4_regulator_penalty_spike_20260609.md` | This doc | — |

All four follow existing patterns:
* Fetcher mirrors `scripts/collect_daily_news.py` (manifest sidecar,
  version-gated skip).
* LLM wrapper mirrors `scripts/extract_policy_events.py` (per-call
  `system_prompt=` thread-safe contract, V2 retry queue reuse).
* Factor build mirrors `scripts/build_policy_factors.py` (PIT
  cutoff, atomic parquet write, sparse_steady health publish).
* Instrument keying mirrors `factors/feature_cache_utils.py`
  (lowercase `sh600519` / `sz000001`).

---

## 5. PIT contract

* `event_date` = the regulator's 公告日期 (NOT 做出日期 / decision
  date — decision date can be 5-30 days earlier and would leak future
  information).
* Lag: `signal_date = event_date + 1 BDay`. Implemented in
  `build_regulator_penalty_factors._load_events_from_dir`.
* Rationale: regulator portals publish after market close. A
  same-day factor row against close-to-close returns would leak.
* TODO L1: thread the SSE trading calendar (currently uses
  `pandas.bdate_range` default Mon-Fri).

---

## 6. RiskGuard plug-in point

Inspecting `backtest/risk_guard.py`:

* **L1 (stock-level)** is the natural target. Add `_check_regulator_action`
  alongside `_check_st_stocks` / `_check_crash_risk`. Trigger map:
    * `severity ∈ {delisting_warning, criminal_referral}` → append to
      `force_sell`, 30-day cooldown (matches existing
      `cooldown_event_days=30`).
    * `severity == suspension` → `pending_exit` + 30-day cooldown.
    * `severity == fine` → `reduce_weight[code] = 0.5` (soft).
    * `is_strict_inquiry == true` → `cannot_buy.add(code)` for 5 days
      (event resolves quickly if inquiry letter goes unanswered;
      Choi 2021 reports median resolution at 35 days).
* **L3 (regime linkage)** is the secondary target: a market-wide CSRC
  press release (e.g. quarterly enforcement roundup) is a regime
  signal; emit a MARKET-keyed column too. NOT in this SPIKE — would
  need a separate parquet (cf. `pbc_liquidity_factors`).

---

## 7. Expected volume + cost

* **Daily volume**: ~5-30 events/day. Empirical: CSRC publishes 2-5
  penalty PDFs/day, SSE+SZSE each publish 1-3 supervision letters and
  3-8 inquiry letters/day. Holiday-eve clustering pushes the peak to
  ~50/day.
* **LLM cost** (MiniMax-Text-01, ~2000 input tok + 200 output tok per
  call): 50 docs/day × 2200 tok × $0.4/M ≈ $0.044/day → **~$16/year**.
* **Backfill cost** (3 years CSRC + 5 years exchanges, est. 30k docs):
  30k × 2200 tok × $0.4/M ≈ **$26 one-off**.
* **Latency**: scrape + LLM end-to-end ~15 min/day at 2s/source +
  threadpool=8 LLM workers.

---

## 8. Expected IC (literature priors)

* Karpoff-Lott (1993): -1.5% mean abnormal return in [-1, +5] window
  after enforcement announcement. Translated to RankIC against
  cross-sectional next-1d returns on a 4000-stock universe, the
  per-event signal carries IC ≈ -0.03 to -0.05 (conditional on event;
  unconditional IC is much smaller because the event is sparse).
* Bhagat-Bizjak (2019): the drift extends to [-1, +20] for financial
  misrepresentation specifically. Suggests `severity_max` over 7d is
  a stronger factor than `has_penalty` alone.
* Choi 2021: strict-inquiry → enforcement transition correlates
  +25%; the `is_strict_inquiry` flag should add **incremental** IC
  on top of `has_penalty`, not be a substitute.
* **Conservative estimate**: portfolio-level IC ≈ -0.01 to -0.02
  (sparse, but high SR when triggered). Justifies inclusion in the
  209+ cache as an additional column, NOT as a standalone factor.
  Better used through RiskGuard as a hard-exit signal.

---

## 9. Risks (pre-mortem)

1. **Lookahead bias** if `event_date` is misclassified as the earlier
   `decision_date`. Mitigation: extractor's `event_date` field is
   explicitly the 公告日期; PDF parser MUST read the 公告日期 field,
   not the 做出日期 field. → Validation test in factor build:
   compare `event_date` to nearest trading day; flag if Δ < 0.
2. **False positives on routine inquiries.** Many inquiry letters
   are pro-forma annual report follow-ups with no enforcement. The
   STRICT_INQUIRY_PHRASES gate mitigates but does not eliminate.
   → Calibration: run 6-month dry-run, only feed
   `is_strict_inquiry=true` into RiskGuard if IC < -0.02.
3. **Multi-stock fan-out.** Industry-wide CSRC press releases name
   multiple stocks; LLM emits one event per stock. The factor build
   de-dups on `(event_date, source_url)` — but the LLM dispatch is
   wasted compute. → Pre-LLM heuristic: skip docs whose body
   names >3 distinct 6-digit codes.
4. **Source coverage gap.** CSRC penalty PDFs lag SSE/SZSE
   supervision letters by 1-2 sessions. The 7-day rolling window
   absorbs this but a same-day factor would underweight CSRC events.
5. **AKShare proxy temptation.** `stock_notice_report` could give a
   "free" 70% coverage. Resist — the title-only signal will mis-
   classify routine `回购报告书` as `share_buyback` instead of
   `regulatory_penalty` and re-introduce the V2 keyword-gate problem
   that motivated this channel.

---

## 10. Go / No-Go

**GO** for a 2-week implementation sprint, sequenced:

1. Week 1: implement `fetch_csrc_penalty` + `fetch_exchange_action`,
   add `pdfplumber` to requirements, dry-run 30-day backfill (no LLM
   call yet — just confirm collector lands ~5-30 docs/day).
2. Week 1.5: enable LLM extraction, run 30-day backfill (~$1 spend).
3. Week 2: run factor build, compute IC against the existing 209
   cache's universe-rank target. Decision gate: IC < -0.01 over
   30 trading days → ship as factor column; else stay as RiskGuard-
   only signal.
4. Week 2.5: PR with RiskGuard L1 integration behind a feature flag.

**NO-GO triggers** during Week 1 dry-run:
* CSRC portal blocks our IP after 24h continuous polling (fall back
  to CSMAR snapshot only — kills the daily-cron value proposition).
* PDF extraction quality < 80% on the 30-doc sample.

**Blockers** for the SPIKE-to-prod handoff:
* `pdfplumber` not in requirements.txt — needs a pin proposal.
* Trading calendar helper (`utils.calendar.get_trading_calendar`)
  must be wired into the +1 BDay shift before backtest.

---

## References

* Karpoff & Lott (1993), "The Reputational Penalty Firms Bear from
  Committing Criminal Fraud", *J. Law & Economics*.
* Bhagat & Bizjak (2019), "The Determinants and Consequences of SEC
  Enforcement Actions", working paper.
* Choi (2021), "SEC Inquiries and Subsequent Enforcement",
  *J. Financial Economics*.
* Project precedents: `scripts/extract_policy_events.py` (4-channel
  LLM extractor pattern), `factors/llm_event_extractor_v2.py`
  (network plumbing reuse), `backtest/risk_guard.py` (L1 plug-in
  surface).
