# Source-Specific SLA Gate Audit 2026-06-07

## Scope

Phase A.7 from `plans/ashare-phases-2026-06.md`. Replaces the
"latest_date == latest trading day" gate with a per-source SLA budget,
so weekly / quarterly disclosure sources don't either falsely-fail the
gate forever or get silently exempted.

This audit follows the same shape as the A.5 / A.6 audits and serves
as the sign-off gate for Phase B (LOO ablation).

## Verdict

Phase A.7 SLA infrastructure is implemented and regression-tested.

- `config/data_sla.py` defines the SLA contract per source.
- `scheduler/data_health.is_fresh_sla` / `sla_verdict` apply per-source
  trading-day budgets instead of a single global rule.
- `tests/test_data_sla.py` (13 tests) pin the contract:
  every `PRODUCTION_GROUP_TO_HEALTH_SOURCE` entry has an SLA, every
  quarterly source has ≥ 60 trading days of headroom, the unregistered
  policy defaults to fail-closed.

Phase B can now start: ablation runs that consume the SLA gate will
treat weekly / quarterly sources honestly.

## SLA Table

Source — frequency / budget / rationale.

| Health source | Frequency | Budget (trading days) | Rationale |
|---|---|---|---|
| `qlib_data_update` | daily | 1 | After-close Alpha158 update batch. |
| `fund_flow_update` | daily | 1 | Daily fund-flow rows. |
| `northbound_update` | daily | 1 | Split from fund-flow after A6-5. |
| `regime_daily_update` | daily | 1 | Regime collector (critical sub-sources per A6-3). |
| `valuation_update` | daily | 1 | Baostock PE/PB/PS daily. |
| `st_daily_basic_update` | daily | 1 | ST_CLIENT daily_basic factors. |
| `st_moneyflow_update` | daily | 1 | ST_CLIENT moneyflow factors. |
| `fundamental_update` | weekly | 7 | Saturday refresh of `fundamental_features.parquet`. |
| `quality_update` | weekly | 7 | Weekly fundamental quality factors. |
| `shareholder_update` | quarterly | 65 | Shareholder count → quarterly filings. |
| `st_holder_number_update` | quarterly | 65 | Holder-number disclosure → quarterly. |
| `llm_event_pipeline` | daily | 2 | Daily overlay; 2-day budget so one failed run doesn't kill the overlay. |
| `global_chain_factors` | daily | 2 | Daily supply-chain overlay; same 2-day softness. |
| `weekly_mask_rebuild` | weekly | 7 | Saturday mask rebuild; consumed through the week. |

## Why this matters

Before A.7 there was effectively one gate rule everywhere:

```python
fresh = success and not partial and latest_date == today
```

That works for daily sources. It does not work for:

- `fundamental_update` (weekly). Monday morning would always show
  `latest_date = Saturday`, which fails `== Monday`. The Phase A.6
  refactor gave this its own health row but the gate would have rejected
  every weekday read.
- `shareholder_update` (quarterly). Between disclosure windows the
  recorded `latest_date` lags by 30-60 trading days. A strict equality
  check would reject the entire period.

So the historical choice was either:

- weaken the gate to "success only, no latest_date check" — back to the
  A.6 silent-stale class; or
- mark these sources non-critical — which lets them rot.

A.7 lets the gate say "fresh enough for this source's cadence":

```python
fresh = success and not partial and (
    trading_day_age(latest_date, today) <= SLA[source].max_age
)
```

## Migration path (recommended order)

1. `scheduler/jobs.py` freshness gate for the morning_recommendation
   pipeline switches from `check_freshness(require_latest_date=True)`
   to `sla_verdict(sources)`. Surface the per-source bucket in the
   log line so the operator can see which sources are blocking.
2. `models/lgb_cache.py` consumes `sla_verdict` for its overlay
   freshness check (currently uses a strict daily expectation).
3. Cron-side health rows that are non-production-critical (`llm_event_pipeline`,
   `global_chain_factors`) can opt-in to the SLA path; the daily
   refresh keeps the budget tight (2 days) but tolerates a single
   failure without silencing the overlay.
4. Add coverage tests asserting (a) every production gate caller goes
   through `sla_verdict`, (b) the result is logged with the same
   ``fresh / stale / exempt`` shape.

## Operator workflow

When a downstream gate reports `stale`:

```
sla_verdict result: stale=[shareholder_update]
  shareholder_update: status=stale reason=exceeds_budget age=72d budget=65d frequency=quarterly
```

The operator:

1. Reads the per-source detail; sees the budget the source is held to.
2. If the budget is wrong (e.g. the cadence changed), updates
   `config/data_sla.py` — single source of truth.
3. If the source legitimately broke, fixes the collector. The SLA
   does not auto-tolerate breakage — it only tolerates lag within
   the declared cadence.

## Tests

`tests/test_data_sla.py`:

1. `test_sla_dataclass_rejects_bad_frequency` — frequency typos surface
   at SLA-table edit time.
2. `test_sla_dataclass_rejects_negative_budget` — negative budgets fail.
3. `test_sla_map_covers_every_production_group_source` — every entry in
   `PRODUCTION_GROUP_TO_HEALTH_SOURCE` has an SLA.
4. `test_quarterly_sources_have_at_least_one_quarter_budget` — guards
   against typos that would lock a quarterly source out.
5. `test_daily_source_within_budget_is_fresh` — happy path.
6. `test_daily_source_one_day_late_is_stale` — strict daily window.
7. `test_weekly_source_3_day_lag_still_fresh` — Mon morning reading Fri.
8. `test_quarterly_source_30_day_lag_still_fresh` — mid-quarter staleness
   is acceptable.
9. `test_failed_health_is_stale` — success=False bypasses budget check.
10. `test_partial_health_is_stale` — partial=True still rejected.
11. `test_unregistered_source_fail_closed_default` — typo-friendly.
12. `test_unregistered_source_exempt_policy` — explicit opt-in for legacy.
13. `test_sla_verdict_buckets` — multi-source verdict gives
    `fresh / stale / exempt` partitions + per-source detail.

## Sign-off prerequisites

- `tests/test_data_sla.py` passes (✅ 13/13 as of 2026-06-06 11:00).
- `docs/health_truthfulness_audit_20260606.md` references this audit
  as the next-step from its "Important residual" section.
- Project lead sign-off before Phase B (LOO ablation) consumes the
  new gate semantics.
