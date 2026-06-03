# fix/macro-pit-drop — PR Self-Review

Date: 2026-06-03
Branch: `fix/macro-pit-drop` (1 commit ahead of `master`)
Commit: `233fcd2 fix: drop macro features from training (cx review round 3 P1)`
Tests: 5/5 new + 42/42 adjacent suites pass

## What this PR does

Drops the look-ahead-biased macro features from the training pipeline
until daily as-of macro data is available.

| File | Change |
|---|---|
| `models/feature_merger.py` | `_load_macro` returns None unconditionally with class-level warn-once log |
| `config/data_availability.py` | `DATA_REGISTRY['macro_features'].allowed_usage`: `["training"]` → `[]` |
| `tests/test_macro_pit_drop.py` (new, 5 tests) | Pin the contract |

## The bug being fixed (cx review round 3 P1)

Previous `_load_macro` implementation:

1. Read `macro_features.parquet` (single-row latest snapshot)
2. `df.iloc[-1]` → take that latest row
3. Broadcast the row to EVERY (date, stock) row of the training index

Every training row therefore saw the LATEST macro values, not the macro
values known at that row's prediction time. The "impact is small because
macro changes slowly" rationale that previously dismissed this is not
acceptable for a training input: any consistent broadcast of future
values can be learned by the model as a spurious shortcut.

`config/data_availability.py` had already tagged the source
`pit_safe_level="unsafe"` but kept `allowed_usage=["training"]`, which
let the leak persist.

## Why drop instead of fix

A correct fix would replace the broadcast with an asof merge against
an `available_date` column on a daily time series. We don't have that
data yet — `macro_features.parquet` is still a single-row snapshot.

Building the daily as-of macro data is a separate work item, not gated
by this PR. Dropping now stops the leak; we re-enable when the data
foundation is in place.

## Re-enable contract (pinned in docstring + tests)

A future PR that flips this back must satisfy ALL of:

1. `macro_features.parquet` is a daily time series with an explicit
   `available_date` column (T+1 publication conservatism).
2. `_load_macro` joins via asof on `available_date <= trade_date`.
3. A new test asserts each training row's `macro_*` value is drawn
   from a row with `available_date <= trade_date`.

The 5 tests in `tests/test_macro_pit_drop.py` will fail loudly if a
future PR silently re-enables without satisfying (1)-(3).

## Downstream audit (no breakage expected)

Audited every reference to `macro_` columns in the codebase:

- `scripts/generate_factor_inventory.py:115-117,199` — inventory
  documentation, not a model whitelist. Harmless.
- `factors/geopolitical.py` — references the variable name `macro_news`
  (a list of news dicts), not column prefix. Unrelated.
- No `models/`, `scripts/train_*.py`, `scripts/smoke_*.py`, or any other
  training/inference script hardcodes a `macro_*` column whitelist.

So removing macro from the training frame is a no-op for everything
except the model's input dim — which goes down by whatever number of
`macro_*` columns were previously being silently broadcast.

## Test coverage

```
tests/test_macro_pit_drop.py
├── test_load_macro_returns_none_even_with_valid_parquet     ← core contract
├── test_load_macro_returns_none_with_missing_parquet        ← legacy safety
├── test_warn_once_per_session                               ← log hygiene
├── test_load_supplementary_has_no_macro_columns             ← integration
└── test_data_availability_registry_has_empty_allowed_usage_for_macro  ← registry policy
```

Plus 42/42 across adjacent suites (capital flow target-date, backtest
compile, crypto quarantine) confirms no collateral breakage.

## Suggested merge call

Risk of merging now: low. The diff is minimal, contracts are tested,
the previous behaviour was actively unsafe.

Risk of holding: every additional day, morning_recommendation +
midweek_train continue training on leaked future macro state.

Final call is the user's; this PR is on its own branch and ready.

## Follow-ups (NOT in this PR)

- **Batch 3 step B** (Task #78): `scripts/train_lgb.py:205` directly
  calls `_load_supplementary()` and injects raw columns into Qlib
  internals, skipping the `merge_for_training` preprocessing. This
  changes the production champion's training distribution and warrants
  its own backtest validation.
- **Batch 3 step C**: ST mask historisation (P2, requires ST_CLIENT
  historical API).
- **Future re-enable**: daily as-of macro data pipeline (out of band).
