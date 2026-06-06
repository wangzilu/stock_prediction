# Data Health Truthfulness Audit 2026-06-06

## Scope

This audit covers Phase A.6 from `plans/ashare-phases-2026-06.md`.
The goal is to stop scheduled jobs from reporting green status while the
actual feature parquet is stale, empty, partially fetched, or sourced
from a different collector than the health row claims.

## Verdict

The top-priority truthfulness fixes are implemented and tested:

- chain jobs enforce upstream dependencies and fail red on empty output;
- `regime_daily_update` publishes sub-source health;
- Qlib no-op updates publish a real health row;
- fund flow and northbound have separate health rows;
- production supplementary groups no longer piggyback on unrelated
  health rows;
- LLM event pipeline publishes health.

Important residual:

The current training gate still has only a daily critical-source list.
Slow-frequency sources such as fundamental, quality, shareholder, and
holder-number now have real health rows, but they need source-specific
SLA enforcement before they should become hard training blockers. Do not
model them as `latest_date == latest trading day`; that would be wrong
for quarterly disclosure data.

## Production Group Health Map

Current `scheduler/data_health.py:400` mapping:

| Production group | Health source | Frequency / note |
| --- | --- | --- |
| `fundamental` | `fundamental_update` | weekly collector for `fundamental_features.parquet` |
| `capital_flow` | `fund_flow_update` | daily fund-flow rows |
| `macro_zero_baseline` | `qlib_data_update` | intentional zero baseline, no independent source |
| `shareholder` | `shareholder_update` | shareholder_features collector |
| `valuation` | `valuation_update` | daily valuation parquet |
| `northbound` | `northbound_update` | split from fund-flow job |
| `quality` | `quality_update` | weekly / quarterly quality parquet |
| `st_daily_basic` | `st_daily_basic_update` | daily ST basic factors |
| `st_moneyflow` | `st_moneyflow_update` | daily ST moneyflow factors |
| `st_holder_number` | `st_holder_number_update` | quarterly holder-number parquet |
| `cross_market_regime` | `regime_daily_update` | regime collector with sub-source health |

Regression guard:

- `tests/test_production_feature_contract.py:104` asserts active
  production groups map to the expected real health source.

## Fixes

### A6-1 / A6-2: global chain jobs fail closed

Pre-fix behavior:

- Cron registered `global_chain_extract` and `global_chain_factors`
  without enforcing the upstream DAG.
- Empty chain extraction / factor build could exit 0 and leave the old
  parquet on disk.

Post-fix behavior:

- `scripts/install_crontab.py:107` and `scripts/install_crontab.py:112`
  render both chain jobs with `--enforce-deps --dep-wait-seconds 3600`.
- `scripts/extract_global_supply_chain_events.py:57` writes red health
  and exits 1 for missing input.
- `scripts/extract_global_supply_chain_events.py:77` writes red health
  and exits 1 for empty input.
- `scripts/extract_global_supply_chain_events.py:90` writes red health
  and exits 1 for zero extracted events.
- `scripts/build_global_chain_factors.py:512` writes red health and
  exits 1 when no factors are produced.

### A6-3: regime daily sub-source health

Implemented in the existing `scripts/update_regime_daily.py` changes:

- critical sub-sources are `margin_detail`, `limit_list_d`,
  `moneyflow_hsgt`;
- aggregate success requires all critical sub-sources to be ok;
- non-critical failures are marked partial;
- details are stored in `HealthStatus.extra`.

Regression evidence:

- `tests/test_update_regime_daily_health.py` passed in targeted tests.

### A6-4: Qlib no-op update writes health

Pre-fix behavior:

- If every symbol was already up to date, `update_qlib_data.py` could
  return 0 without writing today's `qlib_data_update.json`.

Post-fix behavior:

- `scripts/update_qlib_data.py:1386` computes the real latest Qlib
  calendar date.
- Health-check failures on the no-op path write red health before
  returning 1.
- Successful no-op writes green health with `extra.noop=True`.

Regression evidence:

- `tests/test_qlib_data_health.py:93` verifies no-op health is written.

### A6-5: fund flow and northbound split

Pre-fix behavior:

- `fund_flow_update` could go green even if northbound returned empty.

Post-fix behavior:

- `scripts/fetch_fund_flow_history.py:687` writes separate
  `fund_flow_update` and `northbound_update` rows.
- Each row records its own latest date, item count, failure type, and
  partial status.

### A6-6: production health source mapping cleaned up

Pre-fix behavior:

- `shareholder`, `northbound`, `st_daily_basic`, `st_moneyflow`,
  `st_holder_number`, `fundamental`, and `quality` could piggyback on
  unrelated health rows.

Post-fix behavior:

- `scripts/fetch_shareholder_data.py` writes `shareholder_update`.
- `scripts/fetch_st_daily_factors.py` writes `st_daily_basic_update`
  and `st_moneyflow_update`.
- `scripts/fetch_st_round3.py:238` writes `st_holder_number_update`.
- `scripts/fetch_fundamental_features.py:27` writes
  `fundamental_update`.
- `scripts/fetch_fundamental_quality.py:165` writes `quality_update`.
- `scheduler/data_health.py:400` maps production groups to those real
  sources.

Scheduling notes:

- Daily sources are scheduled after Qlib update:
  `st_daily_factors_update`, `valuation_update`, `shareholder_update`,
  and `regime_daily_update`.
- Slow-frequency sources are scheduled weekly:
  `fundamental_update`, `quality_update`, `st_holder_number_update`.
- They are intentionally not same-day dependencies of
  `feature_cache_rebuild`, because the current `run_with_status` DAG
  only understands same-date completion. A weekly source in the daily
  dependency list would block Monday to Friday cache rebuilds.

### A6-8: LLM event pipeline writes health

Post-fix behavior:

- `scripts/run_llm_event_pipeline.py` writes `llm_event_pipeline`
  health after the pipeline returns.
- Failure writes `success=False`, `partial=True`, and `PipelineFailed`.

## Verification

Commands run:

```bash
python -m py_compile factors/candidate_sanitizer.py scheduler/jobs.py signals/market_judge.py scripts/install_crontab.py scripts/build_global_chain_factors.py scripts/extract_global_supply_chain_events.py scripts/fetch_fund_flow_history.py scripts/fetch_st_daily_factors.py scripts/fetch_st_round3.py scripts/fetch_shareholder_data.py scripts/fetch_fundamental_features.py scripts/fetch_fundamental_quality.py scripts/run_llm_event_pipeline.py scheduler/data_health.py scheduler/job_deps.py scripts/update_regime_daily.py scripts/update_qlib_data.py
```

```bash
pytest tests/test_phase_a5_shadow_containment.py tests/test_market_judge_weights.py tests/test_schedule_status.py tests/test_production_feature_contract.py tests/test_qlib_data_health.py tests/test_update_regime_daily_health.py -q
```

Result:

```text
36 passed in 40.71s
```

## Remaining Work

### R1: source-specific SLA gate

Add a policy layer above `PRODUCTION_GROUP_TO_HEALTH_SOURCE`:

- daily sources require `latest_date >= expected latest trading day`;
- weekly collection-date sources require `finished_at` or `latest_date`
  within an allowed age window;
- quarterly disclosure sources require coverage and an acceptable
  reporting-period age, not latest trading day equality.

Only after that should `check_training_gate()` hard-block on every
production supplementary source.

### R2: first-run health backfill

The new health sources will be missing until their first cron run:

- `fundamental_update`;
- `quality_update`;
- `st_holder_number_update`.

Before using those rows as hard gates, run the collectors once or write
an explicit migration/backfill health record after verifying the parquet.

### R3: stale-parquet deletion policy

For several collectors, failure now writes red health but intentionally
does not delete old parquet files. That is safer operationally, but
feature loaders can still read old data if a caller bypasses the health
gate. The follow-up should make loaders/gates refuse stale production
groups by SLA instead of relying on operator discipline.
