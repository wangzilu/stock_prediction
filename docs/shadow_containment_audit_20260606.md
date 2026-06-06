# Shadow Containment Audit 2026-06-06

## Scope

This audit covers Phase A.5 from `plans/ashare-phases-2026-06.md`.
The goal is to stop shadow / experimental signals from silently
changing production buy decisions before they have promotion evidence.

## Verdict

Phase A.5 core containment is implemented and regression-tested.

Production recommendation paths can still read shadow signals for
annotation / reranking, but the uncalibrated global supply-chain alpha
no longer hard-rejects candidates by default. Global chain data is also
loaded as-of the target date instead of falling back to future rows.

## Fixes

### A5-1: chain alpha no longer hard-blocks by default

Pre-fix behavior:

- `CandidateSanitizer` rejected any candidate with
  `global_chain_alpha < -2.0`.
- The raw global-chain distribution was broad and uncalibrated, so this
  could silently turn a shadow factor into a production buy gate.

Post-fix behavior:

- `factors/candidate_sanitizer.py:118` records chain alpha as a shadow
  / rerank signal by default.
- `factors/candidate_sanitizer.py:331` adds `chain_negative` to
  `last_soft_tags` and records `last_chain_alpha`.
- Only `enable_chain_hard_block=True` turns the soft tag into a reject.

Regression evidence:

- `tests/test_phase_a5_shadow_containment.py:21` verifies negative chain
  alpha passes by default and is tagged.
- `tests/test_phase_a5_shadow_containment.py:40` verifies an explicit
  hard-block flag still rejects.

### A5-2: chain alpha fallback is as-of only

Pre-fix behavior:

- If the exact target date was absent, the loader could use the parquet's
  latest date even when that date was after the target date.
- Backtests / replays could therefore consume future supply-chain rows.

Post-fix behavior:

- `scheduler/jobs.py:1241` filters candidate dates to `dates <= target`.
- `scheduler/jobs.py:1250` rejects stale chain data by trading-day age.
- Future-only rows return `None` and the sanitizer skips chain tags.

Regression evidence:

- `tests/test_phase_a5_shadow_containment.py:58` verifies future-only
  rows are refused.
- `tests/test_phase_a5_shadow_containment.py:77` verifies a valid past
  row is used.

### A5-3: geo / LLM remain report-only for market direction

Pre-fix risk:

- Early-session market scoring could be influenced by geo / LLM factors
  even though the intended contract said they were report-only.

Current behavior:

- Existing `signals/market_judge.py` behavior is pinned by
  `tests/test_market_judge_weights.py`.
- The tests assert that score sign and magnitude follow index movement,
  not geo / LLM disagreement.

Regression evidence:

- `tests/test_market_judge_weights.py` passed in the targeted test run.

### A5-4: geo fallback no longer reads supply-chain news

Pre-fix risk:

- When `MacroCollector` returned no headlines, geo analysis could fall
  back to `global_industry_news`.
- That mixed supply-chain research material into macro / geo reporting.

Post-fix behavior:

- `scheduler/jobs.py:2306` falls back only to `macro_policy_news`.
- `scheduler/jobs.py:2314` rejects files after the target date.
- `scheduler/jobs.py:2316` rejects files older than one trading day.

## Production Impact

The production buy list should no longer be pruned by unpromoted
global-chain alpha. Any reduction in candidate count from
`chain_negative` after this change requires an explicit
`enable_chain_hard_block=True` caller.

## Verification

Command run:

```bash
pytest tests/test_phase_a5_shadow_containment.py tests/test_market_judge_weights.py -q
```

Included in a broader targeted run:

```text
36 passed in 40.71s
```

## Residual Risk

Global-chain alpha can still influence production if a caller explicitly
opts into `enable_chain_hard_block=True`. That is intentional, but such
a caller must provide promotion evidence and should be reviewed as a
production gate change.
