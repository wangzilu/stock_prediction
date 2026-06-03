# fix/llm-prefilter-dedup — PR Self-Review

Date: 2026-06-03
Branch: `fix/llm-prefilter-dedup` (3 commits ahead of `master` post crypto merge)
Status: ready for external review (cx + user)
Tests: 95/95 pass across prefilter + integration + L0-cache + quarantine
+ capital-flow + backtest-compile + retry-queue + scheduler suites

## Three commits

```
3a38604 fix: L0 cache no longer suppresses retryable L1 items (cx review #3)
c0b5cec fix: prefilter actually reaches LLM (cx code review P1/P2/P3)
81f4212 feat: LLM event pipeline pre-filter (Layer 1 stale + Layer 3 generic)
```

## What this PR does

Adds a cost-reduction filter in front of the MiniMax LLM event extractor
pipeline. Drops items that have negative or zero expected LLM value
before they spend RPM / tokens.

| Layer | Drop rule | Real-data impact (06-01 sample, 15,253 items) |
|---|---|---|
| **1 stale** | `publish_time < target_date - 7 days` | -9,708 (63.6%) |
| **3 generic** | regex blacklist on dashboard/template titles | -341 (2.2%) |
| **Layer 2 (dedup with fan-out)** | **NOT IN THIS PR** | follow-up |
| **net** | | **-65.9%** (kept 5,204 → LLM) |

By-source breakdown showed 界面新闻 dropping from 6,848 → 159 (97.7%
drop) — overwhelmingly stale leakage, NOT a content quality problem.
交易所公告 100% kept (zero false positives on official disclosures).

## What this PR explicitly does NOT do (Layer 2 fan-out concern)

The user's specific review concern was: "Layer 2 fan-out 不丢 stock
association". This PR does not implement Layer 2 dedup at all, by
deliberate scope choice. Reasoning:

- Minute-level dedup with stock fan-out preservation requires either
  extractor surgery (modify `extract_from_news_file`) or a post-process
  expansion step (read events output, fan out by stock list).
- Either path expands the blast radius significantly. The Layer 1+3
  savings alone (~66%) already solve the immediate $1/day + 429 +
  timeout pain.
- Splitting keeps each PR independently reviewable.

**Confirmation that Layer 1 + Layer 3 do not lose stock association**
(static review of `factors/llm_event_prefilter.py`):

- Layer 1 operates per-item via `publish_time` parsing + stale check.
  Items are kept or dropped whole. No cross-item logic.
  See `prefilter_news` loop body, file `factors/llm_event_prefilter.py`
  around the `for item in news_items:` loop — each branch produces
  either `kept.append(item)` or a `continue`. No dedup map.
- Layer 3 operates per-title via `GENERIC_TEMPLATE_PATTERNS` regex.
  Same per-item kept-or-dropped semantics.
- The existing L0 `classify_l0()` does have cross-day dedup
  (`_content_hash` includes `stock_code`), but this is the PRE-EXISTING
  behavior and unchanged by this PR. Same hash before / after.

So whatever stock-association behavior shipped 5 days ago is preserved
exactly. The remaining (across-stock, same-title-same-minute) dedup
opportunity is the Layer 2 follow-up.

## Three cx review rounds absorbed

| cx round | finding | fix | commit |
|---|---|---|---|
| #1 (P1) | DATA_DIR NameError swallowed L0 silently → prefilter never wrote filtered file → extractor ate raw | (a) Add `DATA_DIR` at module top of `scripts/run_llm_event_pipeline.py`; (b) Restructure Step 1.5: prefilter writes filtered file UNCONDITIONALLY before L0 attempts enrichment | `c0b5cec` |
| #1 (P2) | `_parse_publish_date` only handled `%Y-%m-%d`; YYYYMMDD announcements fell to missing-ts fallback, then official source rule kept them — bypassing 7d window | Add `%Y/%m/%d` and `%Y%m%d` parsers; regression tests for both | `c0b5cec` |
| #1 (P3) | Pure-function tests miss the integration P1 | New `tests/test_run_llm_event_pipeline_prefilter_integration.py` with 3 tests pinning: filtered path reaches extractor; L0 raise still writes prefilter; direct-event contract | `c0b5cec` + `3a38604` |
| #1 (bonus) | `ann_items` UnboundLocalError when Step 0 empty | `ann_items = []` initialized before try | `c0b5cec` |
| #2 (P1) | classify_l0 cached L1 hashes BEFORE LLM ran → 429/timeout/crash silenced items forever (retry queue broken) | classify_l0 caches ONLY direct hashes; 6 unit tests in `tests/test_event_filter_cache_semantics.py` pin the retry contract | `3a38604` |
| #2 (P2) | Integration test fixture avoided DIRECT_CLASSIFY_RULES titles, didn't lock direct-event contract | New `test_direct_classified_item_bypasses_llm_but_lands_in_events` | `3a38604` |
| #3 | All fixes accepted; review confirmed filtered path reaches extractor, L0 failure falls back to prefiltered not raw, direct events tested, L1 cache no longer suppresses retries | — | merge gate |

## Test counts

- `tests/test_llm_event_prefilter.py` — 20 unit tests
- `tests/test_run_llm_event_pipeline_prefilter_integration.py` — 3 integration tests
- `tests/test_event_filter_cache_semantics.py` — 6 retry-safety tests
- Plus adjacent suites pass clean (quarantine 24, capital flow 5, backtest compile 28, retry queue 7, scheduler 14)
- **Total: 95 passed in 195s on `fix/llm-prefilter-dedup` branch**

## File-level summary

```
factors/event_filter.py                            |  15 +-   ← L0 cache semantic fix
factors/llm_event_prefilter.py                     | 204 ++   ← NEW pure module
scripts/run_llm_event_pipeline.py                  | 139 +--   ← injection point + restructure
tests/test_event_filter_cache_semantics.py         | 184 ++   ← NEW retry test
tests/test_llm_event_prefilter.py                  | 251 ++   ← NEW unit test
tests/test_run_llm_event_pipeline_prefilter_integration.py | 399 ++   ← NEW integration test
```

## Known follow-ups (NOT in this PR)

1. **Task #69 (done) / no follow-up**: DATA_DIR NameError — fixed in #c0b5cec
2. **Task #71 (done) / no follow-up**: ann_items unbound — fixed in #c0b5cec
3. **Layer 2 dedup with fan-out** — independent PR after this lands
4. **Task #62 / Batch 3**: macro PIT drop, train_lgb FeatureMerger
5. **Task #63 / Batch 4**: sqrt_adv to OMS, multi-comparison, hold-out

## Suggested merge call

Soak passed. PR has absorbed 3 review rounds. Tests cover the failure
modes that would matter (extraction path, retry safety, direct-event,
boundary date formats, no-timestamp policy).

Risk of merging now: low. Risk of holding: prefilter PR drifts behind
master as Batch 3/4 lands, harder rebase later.

But final call is the user's per sequence
`cron apply ✓ → review prefilter PR → A股 Batch 3 → sqrt_adv → Crypto-0`.

End of review note.
