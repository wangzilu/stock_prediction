# Data Pipeline Redundancy Audit — 2026-06-17

**Triggered by**: tonight's observation that valuation_update + shareholder_update
both take ~1.5–2 hr via baostock per-stock and block the entire downstream chain.

## Headline finding

ST_CLIENT **`bak_basic`** is a mega-endpoint that returns the full A-share universe
(5,525 stocks) in **one** call with all of:

| Domain | Fields covered |
|---|---|
| Valuation | `pe`, `pb` |
| Share structure | `total_share`, `float_share` |
| Balance sheet | `total_assets`, `liquid_assets`, `fixed_assets`, `reserved` |
| Per-share metrics | `eps`, `bvps`, `per_undp` |
| Growth | `rev_yoy`, `profit_yoy` |
| Profitability | `gpr` (gross margin), `npr` (net margin) |
| Holders | `holder_num` |
| Listing meta | `industry`, `area`, `list_date` |

A single 1-second bulk call replaces what currently takes 3+ hours of per-stock
baostock loops across **four** independent cron jobs.

## Confirmed redundancies

### 1. Valuation (PE/PB/PS) — TRIPLE source

| Source | File | Path | Cost | Status |
|---|---|---|---|---|
| baostock `valuation_update` | `fundamental_valuation.parquet` | `scripts/fetch_fundamental_valuation.py` | ~1.5 hr/day per-stock | currently RUNNING |
| ST `daily_basic` | `st_daily_basic.parquet` | `scripts/fetch_st_daily_factors.py` | ~1 s/day full-universe | currently RUNNING |
| ST `bak_basic` | — not yet ingested — | (not in cron) | ~1 s/day | unused |

PE/PB are in all three. baostock path is the bottleneck. Drop baostock, lean on ST.

### 2. Share structure — DOUBLE source

| Source | Fields | Cost |
|---|---|---|
| baostock `shareholder_update` | `total_share`, `liquid_share`, `liquid_ratio` | ~2 hr/day per-stock |
| ST `bak_basic` | `total_share`, `float_share` | ~1 s/day full-universe |

baostock is the bottleneck blocking feature_cache_rebuild for 9 trading days mid-June.
`bak_basic.float_share` is the equivalent of `liquid_share`. `liquid_ratio` is
trivially derivable as `float_share / total_share`.

### 3. Moneyflow (capital flow) — DOUBLE source ~1.4 GB redundant

| Source | File | Size | Schema |
|---|---|---|---|
| AKShare `fund_flow_history` | `fund_flow_history.parquet` | **1,012 MB** | Chinese cols `主力净流入-净额` etc. |
| ST `moneyflow` | `st_moneyflow.parquet` | **450 MB** | English cols `net_mf_amount` etc. |

Same domain (大单 / 中单 / 小单 / 主力 / 净流入). Two different conventions
mixed in the same FeatureMerger. **1.4 GB of duplicated daily writes**.

### 4. Northbound flow — DOUBLE source

| Source | File | Source convention |
|---|---|---|
| AKShare `fetch_fund_holdings` | `northbound_history.parquet` | 持股数量 / 持股市值 (HKD-side derived) |
| ST `moneyflow_hsgt` | `st_moneyflow_hsgt.parquet` | `hgt`, `sgt`, `north_money`, `south_money` |

Different sides of the same flow. Probably both needed for cross-check, but the
audit confirms ST is enough for production-time signal.

### 5. Holders — different facts, NOT redundant

| Source | Fact |
|---|---|
| baostock `shareholder_update` | share STRUCTURE (totalShare / liqaShare) |
| ST `st_stk_holdernumber.parquet` | holder COUNT (holder_num) |
| ST `bak_basic` | both, in one call |

## Storage bloat

Beyond redundant input sources, downstream cache parquets have accumulated:

| File | Size |
|---|---|
| `factor_bank.parquet` | 9.4 GB |
| `feature_cache_174_holder_regime_ma.parquet` | 4.0 GB |
| `feature_cache_175.parquet` | 4.0 GB |
| `feature_cache_209_chain_llm.parquet` | 3.9 GB |
| `feature_cache_209_chain.parquet` | 3.9 GB |
| `feature_cache_209_guba.parquet` | 3.9 GB |
| `feature_cache_209_latest.parquet` | 3.6 GB |
| `feature_cache_209_llm_latest.parquet` | 3.6 GB |
| `feature_cache_209_llm.parquet` | 3.9 GB |
| `feature_cache_209_pbc.parquet` | 3.9 GB |
| `feature_cache_209_production.parquet` | 3.9 GB |
| `feature_cache_242_latest.parquet` | 3.7 GB |
| `feature_cache_242_production.parquet` | 3.7 GB |
| `feature_cache_alpha360.parquet` | 6.3 GB |
| `derived_moneyflow_cyq.parquet` | 285 MB |
| `fund_flow_history.parquet` | 1.0 GB |

Sum: **~62 GB**. The 8 `feature_cache_209_*` variants alone are ~30 GB. Each is a
near-clone of every other (small extra column groups, e.g., `_chain` vs `_llm` vs
`_guba` vs `_pbc` vs `_production`). Active production reads `feature_cache_209_latest`
+ `_llm_latest` + `_production`. The rest are research/historical and could be
moved out of `data/storage/` (which is on the SSD critical path) into archived
storage.

## Data cleaning status

- **No formal schema validator** before parquet writes. We've seen the impact:
  - `fund_flow_history.日期` accumulated "None" strings until 2026-06-16
  - `moneyflow_hsgt.hgt` mixed str/float concat broke today's regime cron
  - 30 corrupt rows survived in fund_flow_history until tonight's cleanup
- **No formal dedup pass** on append-only parquets. Drop-duplicates only happens
  inline when the writer remembers to call it.
- **No null-rate or freshness gate** beyond the per-source SLA health.

## Recommended cleanup (phased)

### Phase 1 — quick wins (2-3 hrs, tonight or this week)

1. **Wire `fetch_st_bak_basic.py`** — daily 1-second pull, write
   `data/storage/st_bak_basic.parquet`. Becomes the new source of truth for
   PE/PB/share structure/EPS/BVPS/growth/margins.
2. **Disable** baostock `valuation_update` cron after FeatureMerger migration.
3. **Disable** baostock `shareholder_update` cron — derive `liquid_ratio` from
   `float_share / total_share`.
4. Migrate FeatureMerger consumers to read from `st_bak_basic.parquet`.
5. End state: 3+ hrs of nightly cron disappear; `lgb_after_close_smoke` no longer
   waits on baostock; the entire downstream chain has 3-hr more headroom.

### Phase 2 — moneyflow consolidation (1 day)

1. Decide: AKShare schema vs ST schema as the canonical source.
2. Migrate downstream consumers in `models/feature_merger.py`.
3. Drop `fund_flow_history.parquet` (1 GB).
4. Estimated downstream win: ~1.4 GB disk, no new factor information.

### Phase 3 — feature_cache spring cleaning (half a day)

1. Inventory which `feature_cache_*` parquets are still referenced by any code
   path. Grep + AST scan of `models/` + `scripts/`.
2. Move research-only caches out of `data/storage/` to an archived location.
3. Estimated win: ~30 GB freed.

### Phase 4 — schema validator + dedup framework (2 days)

1. Adopt a tiny schema-validator (e.g. `pandera`) at every parquet write site.
2. Add a daily `data_quality_check` cron that runs basic null-rate / type /
   freshness asserts and writes its own health.
3. Estimated win: prevents future silent-corruption incidents.

## Tonight's immediate action

Build `fetch_st_bak_basic.py` and validate the field mapping against the
existing baostock fields. **Do not yet disable the baostock crons** — keep them
running in shadow until FeatureMerger has been validated against the ST data.

## Risks

- ST `bak_basic` may have a stricter SLA than baostock (i.e. miss recent days);
  need to verify by sampling `trade_date='YYYYMMDD'` for the last 5 trading days.
- ST `float_share` vs baostock `liqaShare` may not be exact equivalents
  (different float definition windows). Cross-check on the sampling.
- `pe` from `bak_basic` is a simple PE; `daily_basic.pe_ttm` is TTM-adjusted —
  prefer the TTM version for factor work. So `bak_basic` is *additive*, not a
  replacement for `daily_basic`.
