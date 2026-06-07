# LLM L6 — Historical Event Backfill + Ablation Re-run Plan (2026-06-07)

> Task #136. The B.6 verdict on 2026-06-07
> (`docs/phase_b6_v4_llm_12col_verdict_20260607.md`) showed
> `xgb_209_llm` at +6.81 bps Spread20 over `xgb_209` but only
> −0.0002 ΔRankIC — below the +0.005 promotion gate. The hypothesis
> behind L6: current LLM event coverage is too thin for the factor
> to express its full value, and a denser backfill could either
> crystallise the promotion or rule the factor out for good.

## 1. Inventory of current coverage

Counted lines per `data/storage/llm_events_v2/<date>.jsonl` (the raw
LLM extractor output — the unified `events/<date>.jsonl` store is
keyed by `signal_date`, not extract date, so it is NOT a clean
measure of per-day extraction effort).

| date       | n_news (cache) | n_v2_events | yield | status       |
|------------|---------------:|------------:|------:|--------------|
| 2026-04-27 | (legacy)       |         649 |   n/a | OK           |
| 2026-04-28 |                |         874 |       | OK           |
| 2026-04-29 |                |        1116 |       | OK           |
| 2026-04-30 |                |        1058 |       | OK           |
| 2026-05-01 |                |         687 |       | OK (holiday) |
| 2026-05-06 |                |         651 |       | OK           |
| 2026-05-07 |                |        1378 |       | OK           |
| 2026-05-08 |                |        1345 |       | OK           |
| 2026-05-11 |                |         884 |       | OK           |
| 2026-05-12 |                |        1220 |       | OK           |
| 2026-05-13 |                |        1252 |       | OK           |
| 2026-05-14 |                |        1259 |       | OK           |
| 2026-05-15 |                |        1359 |       | OK           |
| 2026-05-18 |                |         978 |       | OK           |
| 2026-05-19 |                |        1287 |       | OK           |
| 2026-05-20 |                |        1384 |       | OK           |
| 2026-05-21 |                |        1018 |       | OK           |
| 2026-05-22 |        7,938   |        1828 |  23%  | OK           |
| **2026-05-25** |     500    |     **256** |  51%  | **THIN**     |
| **2026-05-26** |     500    |     **260** |  52%  | **THIN**     |
| **2026-05-27** |     500    |     **254** |  51%  | **THIN**     |
| **2026-05-28** |     500    |       **0** |   0%  | **FAILED**   |
| **2026-05-29** |     500    |     **152** |  30%  | **THIN**     |
| 2026-06-01 |       15,253   |        2289 |  15%  | OK           |
| **2026-06-02** |  16,712    |     **487** |   3%  | **THIN**     |
| **2026-06-03** |  16,578    |     **492** |   3%  | **THIN**     |
| 2026-06-04 |       16,150   |        2348 |  15%  | OK           |
| **2026-06-05** |   7,665    |     **320** |   4%  | **THIN**     |

`THIN` = re-extract candidates. `FAILED` = 2026-05-28's run produced
59 events per the pipeline log (HTTP 429 storm dropped 368 of 427
items), then the file was later overwritten to 0 bytes by a
subsequent unknown rerun. Re-extraction will regenerate it.

## 2. Backfill scope

### What we CAN backfill

8 dates × cached news files (already in `data/storage/daily_news/`):

```
2026-05-25, 2026-05-26, 2026-05-27, 2026-05-28, 2026-05-29,
2026-06-02, 2026-06-03, 2026-06-05
```

The driver `scripts/run_l6_backfill.py` (PID-managed, sequential,
30s sleep between dates) calls
`LLMEventExtractorV2.extract_from_news_file(news_path, target_date=date)`
directly on cached news, removes the existing thin v2 file first so
the extractor's own `≥500 skip gate` doesn't no-op, and pushes the
resulting events through `EventStore.add_events()` so `signal_date`
routing matches today's production pipeline.

### What we CANNOT backfill

The task spec asked for 60-90 trading days. Two hard constraints
prevent that:

1. **News source is not retroactively queryable.**
   `scripts/collect_daily_news.py` enforces `_NEWS_RECENCY_DAYS=7`
   on both ST_CLIENT `anns_d` and the Eastmoney search API. Items
   older than 7 days from the per-call reference date are dropped
   at the connector layer (cx round 4 P0-2 fix from 2026-06-04).
   We cannot collect 2026-04 news today.
2. **The news cache only covers 30 trading days** (2026-04-27 →
   2026-06-05). Dates before 2026-04-27 have no
   `daily_news/<date>.jsonl` and no v2 events to re-extract from.

The combined result: 8 dates of re-extraction is the realistic L6
ceiling, not 60-90. Going forward, the daily pipeline's existing
500-news-filter cap also means each new trading day will contribute
≤500 LLM-extracted events, so total coverage uplift from L6 over
the lookback window the smoke test uses is bounded.

## 3. LLM budget estimate

Per-date extraction tasks (`max_news_per_stock=1`, dedup'd
within-stock):

| date       | tasks (est.) | calls         | tokens (est., ~440/call) |
|------------|-------------:|--------------:|-------------------------:|
| 2026-05-25 |          431 |           431 |                  ~190k   |
| 2026-05-26 |          431 |           431 |                  ~190k   |
| 2026-05-27 |          431 |           431 |                  ~190k   |
| 2026-05-28 |          427 |           427 |                  ~190k   |
| 2026-05-29 |          431 |           431 |                  ~190k   |
| 2026-06-02 |       ~5,160 |        ~5,160 |                 ~2,270k  |
| 2026-06-03 |       ~5,000 |        ~5,000 |                 ~2,200k  |
| 2026-06-05 |       ~5,000 |        ~5,000 |                 ~2,200k  |
| **total**  |   **~17,310** |   **~17,310** |              **~7.6M**   |

At MiniMax-Text-01 list pricing (~¥1 / 1M tokens for input + ~¥4 /
1M for output, mostly input here), this is roughly ¥10 (~$1.40)
total. Negligible budget.

The hard limit is wall-clock, not budget: MiniMax RPM=60 per single
session (extractor's `max_calls_per_minute=60` default), so the big
dates take ~85 min each.

## 4. Expected timeline

| step                                  | est. duration |
|---------------------------------------|--------------:|
| Re-extract 5 thin dates (500 news ea) |       ~40 min |
| Re-extract 3 thin dates (5k tasks ea) |     ~250 min  |
| Inter-date sleeps (7 × 30s)           |         ~4 min|
| `build_llm_event_factors`             |        ~1 min |
| `build_feature_cache_209_llm`         |        ~3 min |
| `phase4e_24split_ensemble --preset 6split` | ~30 min  |
| **total**                             |    **~5.5 h** |

Conservative ETA: backfill alone done at **+5 hours** from launch
(2026-06-07 22:50 launch → ETA ~04:00 next morning). The followup
script (`scripts/run_l6_followup.sh`) polls the backfill log and
auto-triggers the 3 rebuild + smoke steps; total wall clock until
the v4 ablation summary lands is ~6 hours.

## 5. Ablation comparison target

The smoke run writes to
`data/storage/phase4e_xgb_l6_after_backfill_6split/summary.json`
(checkpoint tag `l6_after_backfill`). Compare to:

| Run                                | RankIC  | Sp20 (bps) | Source                                  |
|------------------------------------|---------|-----------:|------------------------------------------|
| **xgb_209 baseline** (no LLM)      | 0.0339  |        ~73 | docs/phase_b6_llm_verdict_20260606.md   |
| xgb_209_llm pre-L6 (12-col, current coverage) | 0.0371  | 78.76 | docs/phase_b6_v4_llm_12col_verdict_20260607.md |
| **xgb_209_llm post-L6 (target)**   |     ≥0.0389 | ≥80 | this run (`l6_after_backfill`)          |

### Promotion gate (re-applied with denser data)

* ΔRankIC vs `xgb_209` ≥ **+0.005** AND
* ΔSpread20 vs `xgb_209` strictly tighter (negative-direction movement allowed only if ICIR improves)

If post-L6 still fails ΔRankIC ≥ +0.005 → conclude the LLM event
factor's marginal value is structurally bounded at this model
family / horizon, document, deprioritise further LLM polish, and
move the engineering budget to the regime-architecture migration
already queued in `memory/feedback_regime_first_architecture.md`.

If post-L6 crosses +0.005 → schedule a 24-split LOO confirmation
(`--preset 24split`) before promoting to production.

## 6. Operational artefacts

| artefact                                       | path                                                     |
|------------------------------------------------|----------------------------------------------------------|
| Backfill driver                                | `scripts/run_l6_backfill.py`                             |
| Backfill log (live)                            | `logs/llm_l6_backfill.log`                               |
| Completion marker (sentinel for followup)      | `data/storage/llm_l6_backfill_done.json`                 |
| Followup automation                            | `scripts/run_l6_followup.sh`                             |
| Smoke output dir                               | `data/storage/phase4e_xgb_l6_after_backfill_6split/`     |
| Smoke log                                      | `logs/l6_smoke.log`                                      |
| One-line append log                            | `docs/llm_l6_backfill_log.md`                            |
| This plan doc                                  | `docs/llm_l6_backfill_plan_20260607.md`                  |

## 7. Failure modes & rollback

* **MiniMax 429 storm** during a big date → extractor's existing
  exponential backoff (4 attempts, 5/15/45s jittered) handles it.
  If a date ends with high `http_fail` count, manually re-run with
  `python scripts/run_l6_backfill.py --dates <date>` after the
  storm passes.
* **Pipeline schema drift** (cache rebuild fails the LLM-col count
  gate) → `build_feature_cache_209_llm --allow-schema-drift` is the
  documented escape hatch (P1 #1 fix from cx audit). Do NOT use it
  silently; bump `PROFILE_EXPECTED_COUNTS` in the same change.
* **Backfill events poison the cache** (e.g. signal_date routing
  bug) → revert by deleting the new rows from the EventStore
  signal_date partitions and rerun the factor builder. The legacy
  v2 jsonls live alongside the unified store so we keep the audit
  trail.
* **Smoke shows regression vs pre-L6** → most likely the new events
  introduce noise. Tag the result in `docs/llm_l6_backfill_log.md`,
  do NOT silently roll back — file a follow-up issue to investigate
  the source of the regression (probably 2026-05-29's 30% yield is
  systematically lower-quality, e.g. boilerplate news).

## 8. Honest scope statement

The smoke results from this L6 run will measure the value of:
- Filling 8 thin dates within a 16-day window (2026-05-25 → 2026-06-05)
- Going from ~250-500 events on those dates to a target of ~1,000-3,000

It will NOT measure 60-90 days of additional history (impossible
given the news recency constraint). If the answer is "still no
ΔRankIC ≥ +0.005 even with the gap filled", the project lead's
original concern that the LLM event factor is structurally
inadequate is confirmed and we move on.

## 9. Launch postscript — 2026-06-07 22:50 / 22:55 BOTH 429-stormed

Two launch attempts tonight, both killed within 5 minutes after
hitting the MiniMax account-level RPM cap on every call.

| attempt | PID    | rpm | result                                |
|---------|--------|-----|---------------------------------------|
| 22:50   | 22496  |  60 | 0 successes, 12+ 4-attempt 429 fails  |
| 22:55   | 47511  |  30 | 0 successes, ~10 4-attempt 429 fails  |

Both runs hit ``HTTP 429 rate_limit_error: rate limit exceeded(RPM)
(1002)``. The `(1002)` is MiniMax's **account-shared RPM counter**,
not our per-instance rate. We were sending ≤30 RPM in the second
attempt; the account meter is still saturated from today's 16:30
daily cron (which extracted ~22k events that built `llm_event_factors`
end at 17:29 per `logs/llm_event_pipeline.log`).

A standalone single-call probe at 22:55 succeeded in 6.7s (token
usage 406+96=502, no 429), confirming MiniMax itself is alive — the
quota is intermittently saturated when the daily cron's tail
activity overlaps with our backfill bursts.

Side effect to flag: the first launch deleted
`llm_events_v2/2026-05-25.jsonl` (256 events) before stalling, then
the second launch wrote 95 events into it before being killed. So
the v2 file for 2026-05-25 is now thinner than it was pre-L6. The
unified `events/2026-05-25.jsonl` retains the original 256 events
(EventStore dedups by `_hash` in `add_events`), so the production
factor build is unaffected.

### Operator next step

Wait until off-peak (e.g. early next morning, before the 16:30 daily
cron) and relaunch:

```bash
cd /Users/wangzilu/MyProjects/stockPrediction
nohup python -u scripts/run_l6_backfill.py --sleep-secs 60 --rpm 20 \
  > logs/llm_l6_backfill.log 2>&1 &
echo "BACKFILL_PID=$!" > logs/llm_l6_backfill.pid
```

In a separate shell, register the followup (will block until the
backfill log shows `[L6 backfill done]`):

```bash
nohup scripts/run_l6_followup.sh > logs/llm_l6_followup.log 2>&1 &
```

If 429s persist even at off-peak hours with rpm=20, escalate to
MiniMax for an account RPM cap raise, or split the L6 backfill
across multiple accounts via env-variable swap. The driver script
is idempotent — re-running it picks up where it left off based on
the per-date `≥--min-events` skip rule.
