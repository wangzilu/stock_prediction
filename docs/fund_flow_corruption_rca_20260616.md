# fund_flow_history.parquet — corruption RCA (2026-06-16)

## TL;DR
Not corruption. The audit checker pointed at the wrong column. The parquet has **no `date` column** — it has two date columns: `日期` (AKShare format) and `trade_date` (Tushare/ST format). All 13,587,829 rows have a valid `trade_date` (20070104 → 20260616, 5,419 stocks × 4,702 dates). The `日期` column is the one that is 99% null-stringified, and downstream code (`models/feature_merger.py:902`) already reads `trade_date` exclusively. The data is usable as-is; the wasted column is a hygiene issue, not a corruption bug.

## 1. Parquet inspection
- 13,587,829 rows, 35 cols, 1.06 GB.
- `日期` value_counts: `"None"` = 13,354,693 · `"nan"` = 233,106 · valid (`2026-05-11`) = 30.
- `trade_date`: 13,587,799 valid `YYYYMMDD` strings (30 = literal `"None"`).
- The 30 `日期`-valid rows are exactly the 30 `trade_date`-`"None"` rows. They are mutually exclusive sources, not bogus rows. Every row has valid `qlib_code`, `code`, and numeric columns (e.g. only 615k nulls in `net_mf_amount`).

So the rows are **valid data with a redundant null column**, not "rows with no date".

## 2. Writer paths
Single writer: `scripts/fetch_fund_flow_history.py`.
- `fetch_fund_flow_batch` (line 476–511) — ST batch-by-date, populates `trade_date` only (line 491). No `日期`. This is the cron path (`scheduler/jobs.py:517`, `scripts/install_crontab.py:391`).
- `_fetch_one_flow_ak` / per-stock ST (`_to_standard_flow_columns`, line 361–372) — AKShare returns `日期`, gets back-filled to `trade_date` (line 365). Both columns present.

## 3. Root cause
Both `save_checkpoint` (line 212–234) and `_save_with_merge` (line 650–664) do:

```
for _col in result.select_dtypes(include="object").columns:
    result[_col] = result[_col].astype(str)
result.to_parquet(...)
```

When the ST batch frame (no `日期`) is `concat`-ed with the existing parquet (which has `日期`), pandas fills the missing column with `NaN`. The unconditional `astype(str)` on every object column then converts `NaN` → `"nan"` and `None` → `"None"`. Each daily run re-reads the now-stringified parquet, re-concats, and the `"None"` strings persist forever. The 233k `"nan"` rows are leftover from earlier runs before a None vs NaN code drift; the 13.35M `"None"` rows are the current steady-state output.

The same defensive cast was added on 2026-06-08 to fix a ggt_ss mixed-type ArrowTypeError (line 224–230 comment). The fix worked for that bug but stringified all NaNs in the process.

## 4. Minimal fix
Two-line change in `scripts/fetch_fund_flow_history.py`, in both `save_checkpoint` (line 231–232) and `_save_with_merge` (line 661–662): mask NaN/None before casting. Replace `result[_col].astype(str)` with something equivalent to "convert non-null values to str, leave null as NaN" (e.g. `.where(result[_col].notna(), other=np.nan).astype("string")` or `.map(lambda x: str(x) if pd.notna(x) else None)`). The downstream `to_parquet` then writes a real null. Add a one-time recovery: read the file, replace `"None"`/`"nan"` in `日期` with NaN, rewrite. No re-fetch needed — `trade_date` is intact and is what the consumer uses anyway.

## 5. Scope
**30-minute one-line fix + 5-minute one-shot recovery script.** Two writer sites + one cleanup pass. No schema migration, no re-fetch, no consumer changes.
