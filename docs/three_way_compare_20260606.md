# Three-way head-to-head: 174 / 175 / 242

- Generated: 2026-06-06T08:49:43
- Split config: `24split`
- Pinned data_end: `2026-05-19`
- Rows are from `data/storage/experiments_ledger.jsonl`.

## Headline metrics

| Model | Commit | Cache | n_feat | RankIC | ICIR | Spread20 | Spread100 | Days |
|---|---|---|---|---|---|---|---|---|
| xgb_205_legacy_174_runner | `72aa580` | `feature_cache_174_holder_regime_ma.parquet` | 205 | +0.0419 | +0.35 | 25 | — | 777 |
| xgb_242 | `1fe9138` | `feature_cache_242_production.parquet` | 242 | +0.0273 | +0.25 | 35 | 23 | 820 |

## Verdict prerequisites

Before reading anything into RankIC differences, verify:
1. All rows share the same `Commit` and `data_end` columns.
2. All rows used the same `split_config`.
3. No row is missing metrics (`—` in a cell means metrics were absent or NaN).

If any of those fails, the comparison is NOT apples-to-apples and
the rankings can be misleading.

## Provenance

- `xgb_205_24split_20260605_221630` → `feature_cache_174_holder_regime_ma.parquet` (commit 72aa580)
- `xgb_242_24split_20260605_235458` → `feature_cache_242_production.parquet` (commit 1fe9138)