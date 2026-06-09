# LLM Channel #1 — Sell-Side Analyst Rating Spike (2026-06-09)

**Status:** Scaffold complete, no LLM calls executed yet. **Recommendation: GO** (with caveats — see §7).

## 1. Data source feasibility

Surveyed four candidate channels. Verdict per channel:

| Source | Coverage | Structure | Stability | Cost | Verdict |
|---|---|---|---|---|---|
| **AKShare `stock_research_report_em`** (Eastmoney 东财) | All A-share, ~750 reports/stock historical | Already structured: rating, broker, EPS forecasts (3 fiscal years), industry, P/E target, PDF URL | SSL EOF errors observed (1 of 8 calls in spike test); 30-day rolling window works | Free | **PRIMARY — verified working** |
| **慧博 (htbencharm)** | Broader broker tail | PDF-only; needs OCR + scrape | Heavy CF protection; brittle | Free but high-maintenance | **Defer to Phase 2** |
| **同花顺 i问财** | Good, structured | Has rating-change API | Aggressive rate limit; no public docs | Free | Defer |
| **Wind / WindEDB** | Best institutional | Fully structured | Licensed; out of scope | $$$ | Skip |

Eastmoney's table (`ak.stock_research_report_em(symbol='600519')`) returned **759 rows × 16 columns** for Moutai in a successful test, with all the fields we need *except target_price*, which is title/PDF-only. That gap is exactly what the LLM extractor solves — see §2.

## 2. Schema decided

Collector record (already structured, no LLM needed):
- `stock_code`, `qlib_code` (lowercase, canonical per `feature_cache_utils`)
- `report_date`, `report_title`, `broker`, `industry`
- `raw_rating` (Chinese) + `canonical_rating` (`buy|outperform|hold|underperform|sell|strong_buy|unknown`)
- `eps_y1/y2/y3`, `pe_y1/y2/y3`
- `report_pdf_url`, `collected_at`

LLM-extracted fields (per-report JSON):
- `target_price` (RMB, float, null when unparseable)
- `summary_sentence` (≤30字)
- `confidence` (0-1)

Derived fields (computed in extractor, no LLM):
- `rating_change` ∈ {`upgrade`, `downgrade`, `reiterate`, `initiate`} (vs. same broker's prior report)
- `eps_revision_pct` (vs. same broker's prior `eps_y1`)
- `rating_previous` (passthrough lookup)

Rationale: LLM only handles fields the structured table doesn't carry. Saves ~60% of MiniMax tokens vs. naive "extract everything".

## 3. Files created (staged, not committed)

- `/Users/wangzilu/MyProjects/stockPrediction/scripts/collect_research_reports.py` — Eastmoney sweep, manifest sidecar, ST_CLIENT/AKShare fallback, 30d recency window, portfolio/liquid universe modes
- `/Users/wangzilu/MyProjects/stockPrediction/factors/research_rating_extractor.py` — wraps `LLMEventExtractorV2._call_llm` with per-call `system_prompt` (cx P1 #2 thread-safe), grouped (stock, broker) prior-report lookup
- `/Users/wangzilu/MyProjects/stockPrediction/scripts/build_research_rating_factors.py` — emits 5 factors, sparse-by-design steady-state semantics, `publish_health` integration
- `/Users/wangzilu/MyProjects/stockPrediction/docs/llm_channel_1_research_rating_spike_20260609.md` — this doc

Factor columns (long-form parquet, lowercased instrument key):
1. `research_rating_change_score` — sum of {+2/0/-2/+0.5} over 5d window
2. `research_eps_revision_pct` — mean EPS revision over 5d window
3. `research_target_upside_pct` — **STUB**, requires qlib feature cache join (see §7 blocker)
4. `research_attention_score` — distinct brokers in 20d window
5. `research_broker_quality_score` — broker-weighted rating change sum

## 4. Expected daily volume + backfill cost

- Daily volume: ~150–250 reports/day across CSI-300 + CSI-500 on a normal session, spiking to 600+ around earnings season (Apr / Oct). Spike test pulled 759 historical rows for one large-cap.
- Per-LLM-call: ~250 prompt + ~150 completion tokens.
- MiniMax-Text-01 pricing: ~$1/1M input + $1/1M output → **$0.0004 per call**.
- Daily cost: ~$0.08–0.10.
- 1-year backfill (~250 trading days × 200 reports/day ≈ 50 k calls): **~$20–30**.

Well under any budget. No realistic rate-limit risk at 60 RPM throttle (the V2 default).

## 5. Expected IC range (literature)

| Paper | Setting | RankIC |
|---|---|---|
| Womack 1996 (JF) | US sell-side upgrades/downgrades, T+1 to T+30 | +0.04 to +0.07 (rating-change alpha) |
| Stickel 1991 (JF) | US consensus EPS revisions, ±1 month | +0.05 to +0.08 |
| Loh-Stulz 2011 | "Influential analysts" (top-tier, large revisions) | +0.10+ for top decile |
| 茅台 et al., CN A-share replications (2017–2020) | A-share rating change PEAD | +0.03 to +0.06 (lower than US; broker reputation matters more) |

**Realistic expectation for the production factor: +0.03 to +0.05 RankIC over 5–20d horizon**, in line with existing channels (KLEN +0.070, ROC5_tsmin10 +0.047 from `experiment_conclusions_20260525.md`). The broker-weighted variant could push higher if the Womack-style accuracy weighting is done well — but the spike scaffold uses a hand-rolled tier list which is a known weak point (see §7 blocker).

## 6. Risks

1. **Broker reputation weighting is hand-rolled.** The spike uses a static list (`中信/中金 = 1.0, regional = 0.5`). Real Womack-style weighting needs rolling backward-looking accuracy per broker. Mid-Q1 2026 follow-up.
2. **PIT pitfalls.** Mitigated by `collected_at + 1 BDay` lag, not `report_date`. Worth a unit test before backfill.
3. **Eastmoney rate-limit / SSL EOF.** Observed 1 in 8 in the API smoke test. The collector catches per-stock failures so the sweep continues; needs a retry layer if production failure rate exceeds 10%.
4. **Survivorship in historical reports.** Eastmoney prunes delisted-stock report history. Backfill IC on liquidating names will be biased upward — quantify before claiming alpha.
5. **EPS forecast year drift.** The collector reads columns by literal year names (`2026-盈利预测-收益`). Eastmoney rolls these forward every Jan 1. The collector needs a year-rollover smoke test annually.
6. **Case-bug risk.** Pre-emptively normalised `qlib_code` to lowercase at ingest. Same fix as `feature_cache_utils.normalize_instrument_index` — should NOT recur, but the joiner that consumes this parquet still needs `assert_join_coverage` before it ships to the model.

## 7. Blockers to backfill (GO/NO-GO conditions)

| # | Blocker | Severity | Fix size |
|---|---|---|---|
| B1 | `research_target_upside_pct` is stubbed — needs qlib feature-cache close-price join | Medium (factor is degraded, not broken) | 1–2h |
| B2 | No unit tests for PIT lag (`+1 BDay`) — easy to silently regress | High | 1h |
| B3 | Broker-quality weighting is hand-rolled — should NOT ship as the primary alpha attribution | Medium | 1d (rolling accuracy backtest) |
| B4 | No `build_feature_cache_209_*.py` joiner exists — factor parquet sits on disk unused | High | 2–3h (mirror `build_feature_cache_209_llm.py`) |
| B5 | No SLA gate entry / cron schedule | Low | 30min |

## 8. Recommendation — **GO with phased rollout**

- **Phase A (this week, ~1 day):** Fix B1 + B2 + B4. Backfill 30 days. Run LOO ablation vs. base 209 cache. If RankIC delta ≥ +0.02 on a clean held-out window, proceed.
- **Phase B (week 2):** Fix B3 (rolling broker accuracy). Re-run ablation. If broker-quality variant adds another +0.01–0.02, ship as a separate factor column.
- **Phase C (week 3):** SLA gate + cron + freshness publish. Move from staged-files to merged into `master`.

Estimated full delivery: 3 calendar weeks. Cost: ~$30 backfill + ~$0.08/day production.

DO NOT skip the LOO ablation — the B.6.3 verdict (+0.0044 RankIC turned out to be PRNG drift from a case-bug) is a recent reminder that "LLM signal looks great in-sample" is a non-starter as a ship gate.
