# CX Stock Prediction V2 Iteration Plan

Date: 2026-05-07
Owner: Codex
Last updated: 2026-05-10

This is the primary execution plan for the project. Newer Qlib-specific research and cc/cx reconciliation notes are folded back into this file; detailed evidence remains in `plans/cx-qlib-advanced-implementation-plan-2026-05-09.md`, `plans/cx-qlib-next-version-iteration-plan-2026-05-10.md`, and `plans/cc-qlib-advanced-features-roadmap.md`.

2026-05-09 cc unified-plan review:

- Adopt cc/cx converged decisions where they are backed by local code or run logs: serial after-close pipeline, production LGB threshold 4500+, Qlib bin format `[start_index, values...]`, multi-source data fallback, research-only Yahoo bootstrap, and explicit model degradation labels.
- Correct this V2 plan where it was too optimistic: TuShare is not a hard production dependency unless installed and tokened; long-horizon lists must be marked as observation/research until a true fundamentals model exists; multi-bagger labels must avoid `max_forward_return` look-ahead bias.
- Do not adopt cc's older `>=100 predictions` smoke threshold for production, direct `max_forward_return` labels, or raw prediction ensemble averaging.

## 1. Current Diagnosis

### Qlib/LightGBM

Historical 2026-05-07 status: the LightGBM model artifact existed, but the production recommendation path could not reliably produce usable Qlib scores.

Current 2026-05-09 status: the all-universe Qlib data/update/train/smoke chain has since been repaired enough to produce production-scale latest predictions above the 4500 threshold. The remaining blocker is no longer "can it infer"; it is "does the signal have proven trading value after evaluation and backtest." Keep the historical failure record below because it explains the guardrails, but do not treat the 121/280 prediction state as current.

Evidence from local runtime:

- `data/storage/lgb_model.pkl` and `data/storage/lgb_dataset.pkl` exist.
- `logs/cron.log` on 2026-05-07 shows recommendation runs at 09:50 and 13:50 both failed to load/use LGB predictions:
  - `Failed to load LGB model: nan`
  - `Failed to load LGB model: cannot convert float NaN to integer`
  - `Screened 300 A-shares (0 with LGB scores)`
- `logs/train.log` shows the nightly LightGBM job failed:
  - data update timed out after 1 hour
  - LightGBM training raised `ValueError: Empty data from dataset`
- A direct stdin-based probe of `ShortTermModel.load_from_pickle().predict_batch()` did not complete because Qlib multiprocessing tried to spawn from `<stdin>`, so future model smoke tests should live in a real script file.
- 2026-05-10 fix: Qlib initialization is now centralized in `config/qlib_runtime.py`. Normal production runs keep Qlib defaults; debugging can set `QLIB_DEBUG_SAFE=1`, or pass `--joblib-backend threading --kernels 1` to `scripts/smoke_qlib_dataset.py`, to avoid macOS/joblib multiprocessing spawn from `<stdin>`.

Likely root causes:

- Qlib calendar/instrument metadata and local bin data are not fully aligned.
- The dynamic training window can land on sparse or invalid segments after Qlib data update failures.
- The inference dataset is rebuilt without labels, over a short recent window, and still hits Qlib handler/calendar NaN conversion.
- The app silently falls back to intraday `change_pct / 10`, so recommendations still push but are not actually Qlib-driven.

### Qlib NaN Failure Record

Current status: Qlib/LightGBM is producing or surfacing NaN-related failures during both training and inference.

Observed errors:

- 2026-05-07 morning recommendation: `Failed to load LGB model: nan`.
- 2026-05-07 afternoon recommendation: `Failed to load LGB model: cannot convert float NaN to integer`.
- 2026-05-07 manual training: `ValueError: cannot convert float NaN to integer` during Qlib `DatasetH` / `Alpha158` initialization.
- Nightly training: `ValueError: Empty data from dataset`.

Why this happens:

- The data updater can partially succeed and still advance global Qlib metadata. For example, one update completed with only 123/800 stocks but still wrote calendar/instrument changes.
- Qlib expects feature `.bin` files, calendar, and instrument date ranges to be aligned. If the calendar includes a new trading date but many symbols do not have aligned feature values, Alpha158 expressions can produce large NaN regions.
- The current writer rewrites each feature file aligned to the full calendar. A failed or partial fetch can leave different symbols with inconsistent coverage.
- The LightGBM training window is dynamic. If the selected train/valid/test segments overlap a damaged or sparse period, Qlib can return an empty dataset after processors such as `DropnaLabel` or fail while converting malformed index/date metadata.
- The inference path rebuilds a recent Alpha158 dataset against the same Qlib store, so it fails even if an old `lgb_model.pkl` exists.

Why this matters:

- A model artifact existing on disk does not mean the model is usable today.
- If Qlib inference fails, the recommendation pipeline currently falls back to price-change screening, so a push can still happen while containing zero Qlib scores.
- NaN failures can silently downgrade signal quality unless the report exposes model/data health.

Required guardrails:

- Add `scripts/check_qlib_data_health.py`.
  - Check calendar latest date.
  - Check expected instrument count.
  - Check each required field exists: `open`, `high`, `low`, `close`, `volume`, `amount`.
  - Check feature array length matches calendar length.
  - Check latest N trading days have acceptable NaN coverage.
  - Check train/valid/test windows produce non-empty Qlib datasets before training.
- Add `scripts/smoke_lgb_predict.py`.
  - Load the saved model.
  - Build the exact inference dataset used by production.
  - Run `predict_batch()`.
  - Fail if predictions are empty, all NaN, contain too few symbols, or raise Qlib conversion errors.
- Add promotion gating.
  - Do not promote staged Qlib data when health checks fail.
  - Do not start LightGBM/RL training when update or health checks fail.
  - Do not use LGB predictions in recommendation scoring unless smoke prediction passes.
- Add explicit report status.
  - `Qlib: OK, N predictions, latest date YYYY-MM-DD`
  - `Qlib: DEGRADED, reason=<error>, fallback=intraday change_pct`

Acceptance criteria:

- Qlib NaN or empty-dataset errors fail fast before model training and recommendation scoring.
- A partial daily data update cannot create a new production calendar date unless most instruments have valid aligned features for that date.
- The user can tell from the push whether the recommendation used real Qlib predictions or degraded fallback screening.

### Recommendation Schedule

Current status: the repo defines a 4-slot scheduler, but the installed crontab only calls two one-shot recommendation runs.

Current crontab:

```text
50 9 * * 1-5 python main.py --run-now
50 13 * * 1-5 python main.py --run-now
0 4 * * * python scripts/nightly_train.py
0 17 * * 1-5 python scripts/update_qlib_data.py
```

`main.py` defines:

- 09:20 morning recommendation
- 14:30 sell check
- 15:30 daily summary
- 22:00 evening outlook
- hourly risk check during trading hours

But these only run when `python main.py` is started as a long-lived APScheduler process. Process inspection showed no running `main.py` scheduler. This explains why there was no 15:30 close push on 2026-05-07.

### Recommendation Content

Current status: recommendation scoring is still a single mixed score. It has short score, optional mid score, macro score, sentiment, and heat, but the pushed result is not explicitly split into short/mid/long horizons.

V2 target:

- Short term: 1-5 trading days, timing and momentum.
- Mid term: 20-60 trading days, trend continuation and fundamental/sector confirmation.
- Long term: 3-24 months, business quality, growth acceleration, valuation, and multi-bagger optionality.

Long-term and mid-term inputs must not be derived only from the current short-term LGB score. Add slow-moving fundamental and ownership factors such as valuation, profit growth, cash-flow quality, industry prosperity, and lagged institutional/fund holdings. Public fund heavy-position data is usable only after its public disclosure date, with a trading-day lag, so it should be treated as a quarterly "smart money / crowding / theme confirmation" factor rather than an intraday or next-day timing signal.

Do not conflate quarterly fund holdings with daily capital-flow signals. Daily main-force fund flow, sector fund flow, and northbound/HK Stock Connect holdings are separate ownership/flow factors. They can update daily and are better candidates for the short/mid-horizon model than public fund quarterlies, but they still need the same IC, RankIC, turnover, capacity, and cost validation before production scoring.

### US/HK Stock Recommendation Expansion

Current status: US and Hong Kong markets are not first-class recommendation markets yet.

Current code shape:

- `config/watchlist.py` only defines `MARKET_STOCK`, `MARKET_CRYPTO`, and `MARKET_GOLD`.
- `MARKET_STOCK` currently means A-share, not a generic global stock market.
- `MarketCollector` is A-share specific: AKShare A-share, baostock, Tencent A-share quote format.
- `scheduler/jobs.py` screens A-shares from the A-share spot cache, then manually adds crypto and gold.
- Qlib/LightGBM currently trains and infers on China instruments such as CSI300/CSI500.
- US/HK data is present only as global market context in `GlobalIndicesCollector`, not as individual stock recommendations.

Required product decision:

- Decide whether US/HK stocks are:
  - full recommendation markets with buy/sell/verification, or
  - informational watchlists shown next to A-share recommendations.
- V2 recommendation should not mix A-share, US stock, and HK stock scores blindly unless each market has calibrated scoring and risk rules.

Required market taxonomy:

- Add explicit market constants:
  - `MARKET_A_STOCK`
  - `MARKET_US_STOCK`
  - `MARKET_HK_STOCK`
  - keep `MARKET_CRYPTO`
  - keep `MARKET_GOLD`
- Rename or alias current `MARKET_STOCK` carefully so existing A-share tests and records do not break.
- Store market, exchange, currency, timezone, and quote source on every candidate.

Required data collectors:

- Add `USStockCollector`.
  - Realtime/near-real-time quote.
  - Daily OHLCV.
  - Corporate actions/splits/dividends.
  - Optional sources: yfinance, Stooq/Nasdaq/Alpha Vantage/Polygon/IEX depending on budget and reliability.
- Add `HKStockCollector`.
  - Realtime/near-real-time quote.
  - Daily OHLCV.
  - Corporate actions.
  - Optional sources: AKShare HK Eastmoney/Tencent, yfinance, vendor data.
- Normalize both collectors to the same schema:
  - `date`, `open`, `high`, `low`, `close`, `volume`, `amount`, `currency`, `timezone`, `source`.
- Add quote freshness checks because US market data during China daytime is usually previous close/pre-market, while HK trades during China daytime.

Required model/data storage:

- Do not reuse the current A-share Qlib model for US/HK stocks.
- Maintain separate data roots or namespaces:
  - `data/storage/qlib_data/cn_data`
  - `data/storage/qlib_data/us_data`
  - `data/storage/qlib_data/hk_data`
- Maintain separate calendars:
  - A-share trading calendar
  - NYSE/Nasdaq calendar
  - HKEX calendar
- Train separate short-term models by market, or start US/HK with rule-based momentum until enough clean data exists.
- Add separate smoke tests:
  - `smoke_lgb_predict_cn.py` or current `smoke_lgb_predict.py`
  - `smoke_lgb_predict_us.py`
  - `smoke_lgb_predict_hk.py`

Required scoring changes:

- Add market-aware scoring weights.
  - A-share: policy, liquidity, limit-up/down, retail sentiment.
  - US stock: earnings, guidance, rates, sector ETF momentum, USD liquidity, overnight news.
  - HK stock: China macro, HKD/USD peg/rates, southbound flow, ADR/A-share linkage.
- Add currency-aware return and risk handling.
  - USD for US stocks.
  - HKD for HK stocks.
  - CNY for A-shares.
  - Optional FX-adjusted return if the portfolio base currency is CNY.
- Add market-specific filters:
  - US: earnings date, extreme gap risk, ADR risk, pre/post-market availability.
  - HK: low liquidity, wide spread, southbound eligibility, half-day trading, typhoon/holiday closures.
  - A-share: ST, suspension, limit-up/down, T+1.
- Calibrate recommendation thresholds per market instead of using one global `HIGH_THRESHOLD`/`MID_THRESHOLD`.

Required scheduler changes:

- Split jobs by market session:
  - A-share morning and post-close.
  - HK pre-open/post-close around HKEX hours.
  - US evening preview before US open, and morning China-time wrap after US close.
- Avoid pushing stale US recommendations at A-share close when the US market has not opened yet.
- Daily report can have sections:
  - `A股`
  - `港股`
  - `美股`
  - `加密/黄金`

Required verification changes:

- Verification must use the correct trading calendar per market.
- Holding horizon should be trading days in that market, not China calendar days.
- Price-at-recommendation must respect the market session:
  - US recommendation made at China morning may refer to previous US close unless explicitly marked as pre-market.
  - HK recommendation can use same-day HK quote during China daytime.
- Store currency and market in the tracker table so historical verification is unambiguous.

Recommended implementation order:

1. Add market taxonomy and watchlists for US/HK without changing scoring.
2. Add collectors and unit tests for daily/realtime normalized schema.
3. Add market sections to the push report using rule-based momentum only.
4. Add market-specific verification and tracker fields.
5. Add separate Qlib/data roots and model smoke tests for US/HK.
6. Train calibrated US/HK models only after enough clean data and backtests exist.

Acceptance criteria:

- US/HK candidates appear in separate report sections, not mixed into A-share ranking without labels.
- Each US/HK item shows quote date/time, currency, and data source.
- Verification uses NYSE/Nasdaq or HKEX trading days.
- A failed US/HK data/model path degrades only that market section, not the whole recommendation report.
- A-share Qlib model failures do not block US/HK rule-based sections, and vice versa.

### Daily Data Update

Current status: the daily Qlib update path is slow and can corrupt downstream model readiness.

Evidence from local runtime:

- `scripts/update_qlib_data.py` defaults to downloading roughly 5 years of history every run, from `today - 365 * 5` to today.
- The script fetches each stock serially through baostock, so 800 stocks means 800 slow historical API calls before writing data.
- `logs/data_update_now.log` on 2026-05-07 shows one manual update ran from 07:37 to 10:07, about 2.5 hours.
- `logs/train.log` shows the nightly update timed out after 1 hour, so the following LightGBM training received stale or incomplete data.
- `logs/data_update.log` shows the 17:00 update completed with only 123 successful stocks out of 800, but still updated the Qlib calendar/instrument metadata. This can create calendar/data alignment gaps and trigger Qlib `NaN`/empty-dataset errors.

Root causes:

- The current daily job is really a full historical rebuild, not an incremental update.
- baostock is being used as the primary daily data source even though it is better suited for slow backfill and gap repair.
- There is no staging area or success-rate gate, so partial updates can be promoted into the production Qlib directory.
- Calendar construction is tied to the full stock loop, so a partial run can advance the global calendar without complete feature data.
- The downstream training job starts even when data update failed or produced too few valid instruments.

V2 target:

- Full historical rebuild should be a manual/weekend operation only.
- Daily update should pull only the latest 1-5 trading days, or from each stock's `last_success_date + 1`.
- Use a faster primary source for daily bars:
  - preferred: Tushare Pro batch daily data by `trade_date`
  - free fallback: AKShare Eastmoney/Tencent daily history with limited parallelism
  - final fallback: baostock only for gap repair
- Write daily updates to a staging directory first, run data health checks, then atomically promote to the main Qlib directory.
- Require a minimum success threshold, for example 95% valid instruments, before training or inference can use the new data.

### Data Update Decision Record

This section records the follow-up diagnosis and recommended replacement path for the current baostock-based updater.

#### Why The Current Daily Download Is Too Slow

The current implementation is slow because the daily task is doing the wrong shape of work:

- It performs a full 5-year historical download every day.
- It loops through roughly 800 stocks serially.
- It uses baostock as the primary source for every historical query.
- It builds the global calendar from the full loop every run.
- It writes directly into the production Qlib directory without a staging/promotion gate.

The important point is that the issue is not only "baostock is slow". Even with a faster API, daily production updates should not re-download 5 years of history for every stock.

#### Recommended Architecture

Split the data pipeline into two modes:

1. Full bootstrap/rebuild.
   - Used only for first-time setup, weekend repair, or data corruption recovery.
   - Pulls multi-year history.
   - Can be slow, but should run outside the daily trading workflow.

2. Daily incremental update.
   - Default production mode.
   - Pulls only the latest 1-5 trading days.
   - Uses per-symbol state from `data/storage/update_manifest.json`.
   - Starts from each stock's `last_success_date + 1`.
   - Writes to staging first, then promotes only after health checks pass.

Target runtime: reduce daily updates from 1-2.5 hours to roughly 5-15 minutes under normal conditions.

#### Source Policy

Do not make production depend on one theoretical "best" source. The production policy is `provider=auto` plus local cache, manifest resume, staging, and hard coverage gates. A provider should be accepted only if it produces enough aligned, adjusted daily bars for the requested universe; a small partial result must trigger fallback or fail before promotion.

Provider roles:

1. Tushare Pro as an optional batch source when installed and tokened.
   - Pull by `trade_date` to get all-market daily bars in batch.
   - Combine `daily`, `adj_factor`, and `daily_basic` for price, adjustment, turnover, valuation, and liquidity fields.
   - Do not assume it exists in the runtime. If `tushare` is not installed or no token is configured, skip it without treating the plan as blocked.
   - Before production use, convert OHLC to the same adjusted-price convention as the rest of the Qlib store and add cross-source reconciliation checks.

2. AKShare daily history as a free fallback.
   - Use Eastmoney/Tencent daily history endpoints for recent bars.
   - Still often per-symbol, but usually faster than baostock.
   - Run with limited parallelism, such as 4-8 workers.
   - Good fallback for missing symbols or when Tushare is unavailable.
   - Do not use `stock_zh_a_spot_em()` as training daily bars. Spot snapshots are only for push display, intraday momentum, liquidity proxy, and emergency fallback.

3. Local terminal/vendor data as the serious long-term option.
   - Examples: Tongdaxin local cache, MiniQMT/QMT, `xtquant`.
   - Fast and stable when the local environment is available.
   - Strong fit for A-share production use, but more environment-dependent.

4. baostock for deterministic backfill, gap repair, or full rebuild when runtime is allowed to be slow.
   - It is free and has already been the most reliable historical backfill path in local runs.
   - It is not the preferred daily low-latency source because full A-share serial fetch can be too slow.
   - Use `adjustflag=2` or the project-standard adjusted convention, and validate adjustment consistency before mixing with other sources.

5. Qlib official Yahoo collector as research fallback only.
   - Useful for quick experiments and official Qlib workflows.
   - Can bootstrap a research environment, but do not mix Yahoo base data and baostock/AKShare production increments unless adjustment, calendar, and field-scale consistency have been explicitly validated.

#### Promotion And Health Gate

Daily update must use staging:

1. Download to a temporary/staging directory or write a staged delta.
2. Validate row count, instrument count, latest trading date, NaN ratio, and OHLCV sanity.
3. Validate coverage, for example at least 95% of expected instruments.
4. Promote to the main Qlib directory only if the check passes.
5. If the check fails, keep yesterday's production data and alert.

This prevents the 2026-05-07 failure mode where only 123/800 stocks updated but the Qlib calendar/instrument metadata was still advanced.

#### Implementation Tasks

- Add `--incremental` and `--full` modes to `scripts/update_qlib_data.py`.
- Make `--incremental` the default.
- Add `data/storage/update_manifest.json`.
- Add a staged output path, for example `data/storage/qlib_data_staging/cn_data`.
- Add `scripts/check_qlib_data_health.py`.
- Add a minimum promotion threshold, initially 95% valid instruments.
- Stop `scripts/nightly_train.py` from launching LightGBM/RL training when data update fails or health checks fail.
- Add source adapters:
  - `TushareDailyProvider`
  - `AkshareDailyProvider`
  - `BaostockBackfillProvider`
- Log source-level success rates and update duration.
- In `provider=auto`, fallback must be based on usable coverage, not only exceptions. If a provider returns a non-empty but insufficient result, continue to the next source or fail before promotion.
- Production promotion requires `LGB_MIN_DATA_INSTRUMENTS` and downstream smoke coverage to pass. Research/test runs with lower thresholds must be labeled `research_only` and must not write production cache.

#### Acceptance Criteria

- Normal daily update completes in 5-15 minutes.
- A partial failed update cannot overwrite or advance production Qlib data.
- LightGBM training starts only after data health checks pass.
- The recommendation report can show the data source and data freshness date.
- baostock is no longer used for daily full-market historical refresh.

## 2. V2 Product Goals

1. Make the model status observable: never confuse fallback screening with a working Qlib signal.
2. Restore a reliable daily push rhythm: pre-market, intraday/sell, post-close, evening outlook.
3. Split recommendations into short/mid/long views instead of one blended list.
4. Add a "multi-bagger radar" for 5x/10x candidates as a research/watchlist product, not as a daily buy signal.
5. Add reproducible backtests, model diagnostics, and data health gates before a signal can enter production scoring.

## 3. Phase 0: Hotfix And Observability

Target: 1-2 days.

### Tasks

- Add `scripts/smoke_lgb_predict.py`.
  - Load `ShortTermModel` from `data/storage/lgb_model.pkl`.
  - Run `predict_batch()`.
  - Print prediction count, date span, NaN count, top/bottom examples.
  - Exit nonzero if predictions are empty, all NaN, or below a minimum symbol count.
- Add Qlib data health check.
  - Validate calendar length, instrument count, close/high/low/volume availability.
  - Validate latest trading date is within 2 trading days of today.
  - Validate CSI300/CSI500/all-share universe resolves to non-empty instruments.
- Split data update modes.
  - `--incremental` should be the default daily mode.
  - `--full` should be reserved for initial build or weekend repair.
  - Store per-symbol update state in `data/storage/update_manifest.json`.
  - Never promote a staging update when valid-instrument coverage is below threshold.
- Make fallback explicit in WeChat report.
  - If `len(lgb_preds) == 0`, mark report as `数据降级: LGB unavailable, using intraday fallback`.
  - Do not label fallback as model recommendation.
- Fix scheduling.
  - Either install crontab entries for each one-shot command:
    - 09:20 `main.py --morning`
    - 14:30 `main.py --sell-check`
    - 15:30 `main.py --daily-summary`
    - 22:00 `main.py --evening-outlook`
    - hourly `main.py --risk-check`
  - Or run `python main.py` under `launchd`, `systemd`, or `supervisord` as the single source of schedule truth.
- Add log heartbeat.
  - Every scheduled job should log start/end/status/duration/recommendation count.
  - Write latest job status to `data/storage/job_status.json`.

### Acceptance Criteria

- `scripts/smoke_lgb_predict.py` returns nonzero when the current Qlib model cannot infer.
- The push report clearly shows whether Qlib, mid-term, RL, and LLM signals are live or degraded.
- 15:30 and 22:00 pushes are triggered without a manually running shell session.

## 4. Phase 1: Three-Horizon Recommendation Engine

Target: 1 week.

### Data Contract

Create a structured signal object:

```python
{
    "code": "SH600519",
    "name": "贵州茅台",
    "horizon": "short|mid|long",
    "score": 0.0,
    "confidence": 0.0,
    "entry_zone": [0.0, 0.0],
    "stop_loss": 0.0,
    "take_profit": 0.0,
    "holding_days": 5,
    "drivers": [],
    "risks": [],
    "data_status": {}
}
```

### Short-Term Module

Purpose: 1-5 trading days.

Inputs:

- Qlib LightGBM 5-day forward return prediction
- intraday change/volume/turnover
- limit-up/limit-down status
- short-term sentiment heat
- market regime filter

Rules:

- Only use Qlib score when smoke test passes.
- Exclude stocks with stale quote, suspended state, ST flags, or impossible price/volume.
- Use a ranking model first; absolute return prediction is secondary.

### Mid-Term Module

Purpose: 20-60 trading days.

Inputs:

- trend strength: 20/60-day moving average, new high, volatility contraction
- sector momentum
- fund flow and turnover persistence
- optional trained `MidTermModel` checkpoint
- earnings/fundamental acceleration where available

Rules:

- Keep `mid_score = 0` unless a validated checkpoint exists.
- Validate model out-of-sample IC/RankIC and drawdown before enabling production weight.

### Long-Term Module

Purpose: 3-24 months.

Inputs:

- revenue/profit acceleration
- ROE/ROIC and margin trend
- valuation percentile
- industry tailwind
- market cap and free float
- institutional ownership change
- policy/news catalyst score

Rules:

- Long-term picks should be watchlist candidates with staged buy zones, not same-day momentum calls.
- Until a real fundamentals/industry/valuation model exists, user-facing output must call this section `长线观察榜` or `中长期观察`, not a production-grade long-term recommendation.
- The current short-term LGB score, same-day momentum, and liquidity proxy may support a longer-horizon observation rank, but they must not be presented as validated 3-24 month alpha.
- Use monthly/quarterly refresh plus event-triggered updates.

### Push Format

Daily recommendation report should have three explicit sections:

```text
短线 1-5日: top 3
中线 1-3月: top 3
长线观察榜 3-24月: top 3
```

Each item must include:

- reason
- data/model status
- entry/stop/take-profit
- expected holding period
- confidence
- invalidation condition

## 5. Phase 2: Multi-Bagger Radar

Target: 2-3 weeks.

Goal: find 5x/10x candidates early enough to track, not to chase after the move.

### Label Design

Primary train labels:

- fixed-horizon close-to-close returns, sampled monthly to reduce overlapping-window leakage.
- triple-barrier labels: profit target, stop-loss, and max holding period.
- sector/size-relative forward returns, so a candidate is compared with a realistic peer group.

Secondary labels:

- maximum favorable excursion, used only as a diagnostic feature/analysis target, not as the primary investable label.
- max drawdown before target
- time to target
- limit-up cluster count
- survival after first 100% move

Use ranking labels as the primary formulation:

- Rank candidates by future multi-bagger potential within the same month/sector/market-cap bucket.
- Treat 5x/10x as rare-event classification only after class imbalance handling.
- Do not use raw `max_forward_return_6m/12m/24m` as the primary training label; it assumes hindsight exits at the future high and can materially inflate backtest quality.
- Use purged or embargoed time-series validation when labels have overlapping future windows.

### Feature Families

Growth and quality:

- revenue/profit growth acceleration
- gross margin expansion
- ROE/ROIC improvement
- operating cash flow quality
- R&D intensity

Valuation and size:

- market cap percentile
- free float market cap
- PS/PE/PB percentile within industry
- valuation compression plus earnings acceleration

Price and volume:

- 52-week high breakout
- volatility contraction pattern
- turnover regime shift
- limit-up clusters
- relative strength vs sector and market

Capital and attention:

- northbound/fund flow
- institutional ownership change
- shareholder count change
- margin financing activity
- news/social heat acceleration

Theme and catalyst:

- industry policy tailwind
- supply-demand inflection
- new product cycle
- export substitution/import substitution
- AI/robotics/semiconductor/biotech/new energy theme exposure when backed by real fundamentals

### Modeling

Baseline:

- LightGBM/XGBoost/CatBoost ranker
- monthly rebalance top-N watchlist
- sector/size neutral evaluation

Advanced:

- survival/hazard model for "time to multi-bagger"
- event model for catalyst arrival
- graph features for supply chain/theme relationships
- LLM-assisted catalyst extraction with strict source attribution

### Output

Create "妖股雷达" as a separate weekly report:

- 潜伏型: low attention, improving fundamentals
- 加速型: breakout plus volume/earnings confirmation
- 兑现型: already crowded, high risk
- 排除型: hype without fundamentals, ST/delist/suspension/liquidity risk

## 6. Phase 3: Research, Backtest, And Governance

Target: next implementation phase after the Phase 0/1 reliability work.

2026-05-09 integration decision:

- Keep this V2 plan as the main execution document.
- Treat `plans/cx-qlib-advanced-implementation-plan-2026-05-09.md` as the detailed Qlib implementation appendix and API evidence record.
- Treat `plans/cc-qlib-advanced-features-roadmap.md` as a feature map and independent review source, not as copy-paste-ready production code.
- First close the evaluation/backtest/registry loop for the current `Alpha158 + LightGBM` model. Do not make Transformer/ALSTM/RL production-critical until simpler models pass the same gates.
- The cc unified plan was reviewed again. Adopted additions: 4500+ production LGB threshold, coverage-based provider fallback, research-only Yahoo bootstrap, long-term observation labeling, anti-look-ahead multi-bagger labels, and the warning that Qlib's contrib strategy package is currently blocked by the cvxpy/numpy issue. Not adopted as first-line: VectorBT before a Qlib-core/custom backtest attempt, PyPortfolioOpt/PortfolioOptimizer before the cvxpy/numpy issue is fixed, and direct triple-barrier replacement of the main LGB label before an experiment branch exists.

2026-05-09 final cc review delta:

- Evidence standard for any remaining disagreement:
  - A disagreement is only actionable when backed by at least one of: local source line, local command output, run log, dependency/import check, or a reproducible Qlib API signature check.
  - Separate facts from policy. For example, "vectorbt is not installed" is a fact; "therefore do not make it the first production gate" is an execution policy derived from that fact plus the current Qlib artifact layout.
  - If cc is correct on the fact but the execution priority differs, record both. Do not hide the agreed risk behind a priority disagreement.
- cc is right that several current implementation details still contradict the desired production policy:
  - `signals/index_predictor.py` still uses `next_weekday()`, which only skips weekends and does not handle A-share exchange holidays or adjusted trading days. Production target dates must come from the Qlib calendar.
  - `scheduler/jobs.py` still renders user-facing `长线（1-3月）` and `长线前五`. Until a real fundamentals/valuation long-horizon model exists, these must be renamed to `长线观察榜` or explicitly marked as observation/research.
  - `fetch_data(provider="auto")` still returns AKShare data immediately if the provider call returns a non-empty dict. It falls back on thrown exceptions, but not yet on insufficient usable coverage after a partial provider success. This is the same failure class cc described for 05-08.
  - The managed schedule still has separate after-close steps in the historical notes. The intended production target is one serial `after_close_pipeline.py` that stops on data, health, train, smoke, evaluation, or backtest failure.
- cc is also right that current environment constraints change the immediate Qlib plan:
  - `TopkDropoutStrategy` cannot be used directly in this environment because importing it reaches `EnhancedIndexingOptimizer`, then `cvxpy`, then fails on `numpy.lib.array_utils` with current `numpy==1.26.4` and `cvxpy==1.8.2`.
  - Therefore Phase 1 backtest should use Qlib core exchange/executor plus a local TopK/TopK-dropout strategy equivalent, or fix the Python dependency set first.
- Remaining evidence-based disagreements with cc:
  - Do not make vectorbt the first-line P1 backtest gate. It is not installed locally, and Qlib core is closer to the data/model artifacts we already have. Keep vectorbt for later parameter scans.
  - Do not make RQAlpha mandatory until a Qlib baseline proves a measured T+1/A-share execution gap. RQAlpha is not installed locally.
  - Do not make SnowNLP a near-term production dependency. It is not installed locally, and sentiment should enter through structured `event_impacts` with source quality, decay, confidence, and hard overrides.
  - Do not promote PyPortfolioOpt or Qlib `PortfolioOptimizer` into the next sprint while the cvxpy/numpy import chain is broken.
  - Do not replace the main short-horizon LGB label with triple-barrier immediately. Use triple-barrier for a separate experiment or multi-bagger/risk label after purged/embargoed validation exists.
- Local evidence behind the remaining disagreements:
  - Package availability check in the `tianshou` environment returned `{'vectorbt': False, 'rqalpha': False, 'snownlp': False, 'tushare': False, 'xgboost': True, 'catboost': True}`. This supports using installed tree baselines next, while treating vectorbt/RQAlpha/SnowNLP/TuShare as optional setup work rather than already-available production components.
  - `from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy` fails through `qlib.contrib.strategy.optimizer.enhanced_indexing -> import cvxpy as cp -> ModuleNotFoundError: No module named 'numpy.lib.array_utils'`.
  - The same environment reports `numpy 1.26.4` and `cvxpy 1.8.2`, so optimizer-dependent strategy/portfolio code is a dependency repair task before it is an implementation task.
  - Qlib evaluation API checks show `risk_analysis(r, N=None, freq='day', mode='sum')`; it takes returns, not raw predictions. `qlib.contrib.eva.alpha` exposes `calc_ic`, `calc_long_short_return`, and `calc_long_short_prec`, but no local `alpha_analysis`. This is why the evaluation gate uses IC/long-short functions and reserves `risk_analysis` for portfolio returns.
  - The current code confirms cc's unresolved production-policy findings: `signals/index_predictor.py` has weekend-only `next_weekday()`, `scheduler/jobs.py` still renders `长线（1-3月）`/`长线前五`, and `scripts/update_qlib_data.py` returns AKShare immediately from `provider="auto"` instead of checking final usable coverage before provider acceptance.

2026-05-09 cc new-argument review:

- Accepted:
  - cc's corrected position on `PortfolioOptimizer` and `TopkDropoutStrategy` is now aligned with local evidence: both are blocked by the cvxpy/numpy import chain in this environment.
  - cc is right that `ShrinkRiskModel` was the wrong class name. Local import check confirms `from qlib.model.riskmodel import ShrinkCovEstimator` works.
  - cc is right that Alphalens and QuantStats have value as deeper reporting/presentation tools. However, local package check shows `alphalens=False` and `quantstats=False`, so they are optional setup work, not first-gate dependencies.
  - cc is right that once the evaluation/backtest scripts are generic, model comparison should not be artificially limited to tree models. Week 3 should include deep-model comparison jobs under the same evaluation interface, not only tree baselines.
  - cc's MPS counterexample is reproducible in the actual project Python. On 2026-05-09 22:35, `/Users/wangzilu/miniconda3/envs/tianshou/bin/python` reported `macOS 26.3`, `torch 2.11.0`, `mps_available=True`, `mps_built=True`, and MPS tensor creation succeeded. A local TransformerEncoder benchmark for `batch=2048, seq=20, d_model=158, layers=2` measured `CPU ~= 564.0 ms/batch`, `MPS ~= 43.9 ms/batch`, about `12.8x` speedup. This supersedes the earlier MPS-false note.
- Still disagreed with cc:
  - Do not make deep models production-critical before the evaluation/backtest/registry gates exist. MPS removes the strongest runtime objection, but it does not remove the need to beat LGB/XGB/CatBoost/DoubleEnsemble out-of-sample after costs.
  - cc's line saying Qlib does not support suspended-stock skipping is too broad. Local Qlib `exchange.py` explicitly documents `$close is None` as suspended, builds `limit_buy`/`limit_sell` from suspended flags, and exposes `check_stock_suspended()` plus `is_stock_tradable()`. The narrower concern that Qlib may not fully model A-share T+1 position lock is still valid and keeps RQAlpha as a later validation tool.

2026-05-09 cc hard-argument review:

- Accepted hard arguments:
  - The strongest cc/cx convergence is the data architecture: `provider -> raw_daily_cache -> normalize -> qlib_staging -> health+smoke -> promote`. Current code has `manifest` and Qlib staging/promotion, but it does not yet have a durable raw daily cache before bin writing. Add this before further provider complexity so successful per-symbol fetches survive a failed promotion run.
  - Provider fallback should operate on usable gaps, not just whole-provider exceptions. Current `fetch_data(provider="auto")` still returns immediately from AKShare/TuShare if a provider returns a non-empty dict. This must become per-symbol/per-shard gap filling, or at least coverage-gated fallback before provider acceptance.
  - Bootstrap/backfill and daily incremental update must remain separate modes. This is already in the plan conceptually, but the implementation should make it explicit in script behavior and logs.
  - Adjustment consistency is a production gate. Current sources use different adjustment conventions (`baostock adjustflag=2`, `AKShare adjust="qfq"`, TuShare raw daily plus `adj_factor`). Before mixing providers in one production store, add adjustment reconciliation checks or store source-specific raw data and normalize through one project-standard convention.
  - cc's 2026-05-09 empirical TopK/IC run is reproducible locally and should be accepted as evidence, not treated as speculation. Rebuilding the `Alpha158` test dataset for `all` and evaluating the current `data/storage/lgb_model.pkl` over `2026-04-10 ~ 2026-05-09` produced `93,587` finite prediction rows across 18 trading dates. Against the 5-day forward return expression `Ref($close, -5)/Ref($close, -1)-1`, the aligned set was `67,572` samples over 13 dates with `IC mean=0.032923`, `ICIR=0.767551`, `RankIC mean=0.002334`, `RankIC>0=53.85%`, `Top20 avg=+4.145%`, `Bot20 avg=-2.513%`, `Top-Bottom spread=+6.659%`, `Universe avg=+1.300%`, and `Spread>0=84.6%`. These numbers match cc's document.
- Partially accepted hard arguments:
  - "先跑通再优化" is valid only as a controlled bootstrap/repair tactic: a manually triggered baostock backfill with timeout, manifest resume, and health gates can be used to fill coverage faster. It must not bypass staging, health checks, smoke checks, or the 4500 production prediction/data thresholds.
  - cc is right that the current LGB score has a real extreme-bucket signal. The conclusion should be narrowed: this proves Top20/Bottom20 stratification on the recent 13-day 5-day-return sample, but it does not yet prove a production-ready strategy. Reasons: RankIC remains near zero, the window is very short, the simulation is costless, and it does not yet include tradability, limit-up/down, suspension, T+1/settlement, turnover, slippage, or benchmark-relative portfolio accounting.
  - The earlier "model sorting ability is weak" wording must be corrected. A more precise statement is: broad cross-sectional monotonic ranking is weak (`RankIC` low), while the top/bottom extremes currently show economically meaningful separation. The next gate should preserve both facts instead of choosing only one.
  - The `83,180` sample / 16-day test number corresponds to the shorter Alpha158 default label `Ref($close, -2)/Ref($close, -1)-1`, which locally gives `IC mean=0.045188`, `ICIR=0.810490`, `RankIC mean=0.018402`, and `Top20 spread=+2.740%`. The cc Top20 argument uses the 5-day forward return expression and therefore has 13 valid dates. Evaluation output must print the label expression and horizon beside every metric to avoid mixing these two tests.
  - cc's MPS/deep-model argument is now accepted on runtime: the `tianshou` environment reproduces MPS availability and roughly `44 ms/batch` Transformer inference. Full deep training is still gated on measured end-to-end train/eval/backtest wall time and out-of-sample quality, not on GPU availability.
- Rejected or narrowed hard arguments:
  - Do not treat baostock as the daily production source merely because it was the only source that previously completed a large backfill. That proves it is valuable for bootstrap/repair; it does not prove it is the best low-latency daily source.
  - Do not treat Qlib as unable to skip suspended stocks. Source inspection shows it handles suspended stocks through NaN `$close` and tradability checks. Keep the narrower T+1 position-lock concern for later RQAlpha validation.
  - Do not convert cc's costless Top20 result directly into a production buy list. It is strong enough to prioritize the evaluation/backtest script immediately, but not strong enough to skip the portfolio backtest, execution constraints, and model registry gates.
  - Do not let the current code keep an implicit label mismatch. `models/short_term.py` uses `PREDICTION_HORIZON_DAYS=5` when training through `ShortTermModel.train()`, but `scripts/train_lgb.py` currently instantiates `Alpha158` without a `label` override, so Qlib's default label is `Ref($close, -2)/Ref($close, -1)-1`. The next LGB training change must make the label expression explicit and store it in artifacts.

### Concrete Qlib Workstream

1. Add `scripts/evaluate_lgb_test.py`.
   - Load the latest trained LGB model and test segment predictions.
   - Rebuild the test `DatasetH` instead of trusting `lgb_dataset.pkl`; local evidence shows the pickled `Alpha158` handler can reload without `_infer/_learn` and then fail on `predict()`/`prepare()`.
   - Make the evaluated label expression explicit. At minimum report both Alpha158 default `ret1 = Ref($close, -2)/Ref($close, -1)-1` and product-horizon `ret5 = Ref($close, -5)/Ref($close, -1)-1` until training and product copy are aligned.
   - Use `qlib.contrib.eva.alpha.calc_ic` and `calc_long_short_return` where they fit, not `risk_analysis(pred)`.
   - Report direction accuracy, IC, ICIR, RankIC, RankIC positive ratio, Top20/Bot20 return, broad quantile return, long-short return, coverage, finite prediction count, latest prediction date, test date count, sample count, and label expression.
   - Exit nonzero if coverage or finite latest-day predictions are below production thresholds.
   - Keep cc's reproduced numbers as the regression target for the first version: `ret5` should reproduce roughly `67,572` aligned samples over 13 valid dates, `IC mean ~= 0.0329`, `RankIC mean ~= 0.0023`, and `Top20 spread ~= +6.66%` on the current artifact/data snapshot.
2. Add `scripts/backtest_qlib_signal.py`.
   - Use Qlib core backtest plus an explicit `SimulatorExecutor(time_per_step="day")`.
   - Do not rely on `qlib.contrib.strategy.signal_strategy.TopkDropoutStrategy` until the current import path is fixed. In this environment, `signal_strategy.py` imports `EnhancedIndexingOptimizer`, which imports `cvxpy`, which fails under the current numpy/cvxpy versions.
   - First implementation should use a small local TopK/TopK-dropout strategy equivalent, or fix the cvxpy/numpy environment before using Qlib's contrib strategy class.
   - Include transaction costs, slippage/impact cost, trade unit, tradability filtering, and limit-up/down settings.
   - Do not assume `benchmark="SH000300"` until index data exists in the local Qlib store; initially use absolute portfolio metrics or a locally built all-A equal-weight benchmark.
3. Add `scripts/after_close_pipeline.py`.
   - Serial order: update Qlib data -> health check -> train LGB -> smoke prediction -> evaluate -> backtest -> refresh cache -> update job status.
   - Production mode must use the latest Qlib trading day, not natural calendar day.
   - Test mode may allow a manual `--as-of` date and non-trading-day dry runs, but must clearly label them as tests.
4. Add durable raw daily cache and provider-gap repair before additional provider expansion.
   - Store raw per-source daily bars before writing Qlib bins.
   - Normalize from raw cache into the project-standard adjusted convention.
   - Allow provider fallback to fill only failed symbols/shards.
   - Add adjustment reconciliation checks before mixing providers in one production Qlib store.
5. Add Qlib Recorder/model registry.
   - Save model config, dataset config, data date range, metrics, prediction artifact, and backtest artifact for every accepted run.
   - Keep the previous production model when evaluation/backtest gates fail.
6. Add model comparison only after the above scripts exist.
   - Compare LGB, XGBoost, CatBoost, DoubleEnsemble, GRU, ALSTM, Transformer, and Alpha360 variants under the same evaluation/backtest scripts. The jobs can run in parallel once the gates exist; do not wait for tree models to fail before starting deep-model experiments.
   - Ensemble predictions must use cross-sectional rank or z-score normalization before averaging.
   - Use MPS for PyTorch models when available, and log device, wall time, memory pressure, data window, and random seed.
   - Full deep-model training is not production-critical until the model beats tree baselines out-of-sample after costs and has registry/rollback artifacts.

### Strategy Families To Study

- Momentum and trend following: cross-sectional momentum, time-series momentum, relative strength, volatility breakout.
- Quality/growth: earnings acceleration, profitability, margin expansion, reinvestment runway.
- Value and mean reversion: valuation spread, residual reversal, sector-neutral value.
- Event-driven: earnings surprise, policy catalyst, industry news, limit-up continuation.
- Multi-factor ranking: IC/RankIC, factor decay, factor crowding, sector/size neutralization.
- Portfolio construction: risk parity, mean-variance, hierarchical risk parity, turnover constraints.
- Reinforcement learning: keep experimental until it beats simple baselines after costs.

### Open-Source Stack Policy

- Qlib is the first-line stack for A-share daily alpha, model training, signal records, and initial portfolio backtesting.
- Use Qlib `calc_ic`, `calc_long_short_return`, `SignalRecord`, `SigAnaRecord`, `PortAnaRecord`, and `risk_analysis` in their correct roles.
- Alphalens is optional later for deeper factor tearsheets such as quantile returns, turnover, and factor decay. It is not required for the first production gate.
- vectorbt is optional later for fast parameter scans. It is not the first portfolio backtest engine.
- RQAlpha is optional later if T+1 or richer A-share execution rules become a measured Qlib bottleneck.
- QuantStats is optional later for polished reporting, not core signal acceptance.
- PyPortfolioOpt is not a near-term dependency. Current `cvxpy==1.8.2` requires `numpy>=2.0.0`, while the Qlib runtime is on `numpy==1.26.4`.
- Qlib `PortfolioOptimizer` is also deferred in the current environment because importing the optimizer package reaches `cvxpy` through `EnhancedIndexingOptimizer`.
- Qlib `TopkDropoutStrategy` is conceptually the right strategy family, but the normal import path currently reaches the same optimizer package through `signal_strategy.py`; use a local strategy implementation until fixed.
- Riskfolio-Lib and FinRL remain research references, not production dependencies.
- Tianshou remains limited to the experimental RL timing layer until it beats simple baselines after transaction costs.

### Book List

Core quant:

- Grinold & Kahn, `Active Portfolio Management`
- Marcos Lopez de Prado, `Advances in Financial Machine Learning`
- Marcos Lopez de Prado, `Machine Learning for Asset Managers`
- Stefan Jansen, `Machine Learning for Algorithmic Trading`
- Ernest Chan, `Quantitative Trading`
- Ernest Chan, `Algorithmic Trading`
- Perry Kaufman, `Trading Systems and Methods`
- David Aronson, `Evidence-Based Technical Analysis`

Growth and multi-bagger:

- William O'Neil, `How to Make Money in Stocks`
- Mark Minervini, `Trade Like a Stock Market Wizard`
- Nicolas Darvas, `How I Made $2,000,000 in the Stock Market`
- Philip Fisher, `Common Stocks and Uncommon Profits`
- Peter Lynch, `One Up On Wall Street`

### Reference Links

- Microsoft Qlib: https://github.com/microsoft/qlib
- Zipline Reloaded: https://zipline.ml4trading.io/
- Backtrader: https://www.backtrader.com/
- vectorbt: https://vectorbt.dev/
- Alphalens: https://github.com/quantopian/alphalens
- PyPortfolioOpt: https://pyportfolioopt.readthedocs.io/
- FinRL: https://github.com/AI4Finance-Foundation/FinRL
- Tianshou: https://tianshou.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- MLflow: https://mlflow.org/docs/latest/
- QuantConnect Lean: https://github.com/QuantConnect/Lean

## 7. Current Implementation Order

As of 2026-05-09, the original Phase 0 guardrails, scheduler visibility, LGB cache, and daily job wrappers are mostly in place. The next work should proceed in this order:

1. Keep daily operations stable.
   - Continue using `scripts/check_qlib_data_health.py`, `scripts/smoke_lgb_predict.py`, `scripts/run_with_status.py`, and the managed crontab.
   - Daily jobs should run against Qlib trading days in production; non-trading-day runs are test mode only.
   - Production LGB acceptance is 4500+ latest finite predictions and 4500+ data instruments. Lower thresholds are research-only and must not write production cache.
   - `provider=auto` must fallback on insufficient usable coverage, not only on thrown exceptions.
2. Build the LGB evaluation gate.
   - Implement `scripts/evaluate_lgb_test.py`.
   - Promote no model without IC/RankIC, bucket return, latest coverage, and finite prediction checks.
3. Build the Qlib backtest gate.
   - Implement `scripts/backtest_qlib_signal.py`.
   - Use Qlib's backtest/exchange/executor core, but implement the initial TopK/TopK-dropout logic locally unless the contrib strategy import is repaired.
   - Include costs, tradability, limit-up/down handling, turnover, drawdown, and benchmark policy.
4. Merge the after-close workflow into one serial pipeline.
   - Implement `scripts/after_close_pipeline.py`.
   - Stop the chain on data, health, train, smoke, evaluation, or backtest failure.
5. Add Recorder/model registry.
   - Every accepted model must have reproducible parameters, data windows, metrics, artifacts, and rollback information.
6. Compare model families.
   - Run LGB/XGB/CatBoost/DoubleEnsemble under the same evaluation and backtest scripts.
   - Run Alpha360 and deep models in the same comparison phase once the gates exist; MPS is available locally, so runtime is not a reason to postpone the experiments.
7. Continue product expansion after the signal gate is trustworthy.
   - Multi-horizon recommendations, weekly multi-bagger radar, US/HK sections, and RL timing should not weaken the Qlib data/evaluation gates.

## 8. V2 Done Criteria

- A failed Qlib model can no longer masquerade as a valid model-driven recommendation.
- Daily jobs push at the expected times without manual intervention.
- The main recommendation report is explicitly split into short/mid/long sections.
- The system has a separate multi-bagger radar with labels, features, and backtest metrics.
- Every production model has a smoke test, last-trained timestamp, prediction count, and fallback state.
- Backtests include transaction costs, suspension/limit-up constraints, and A-share trading-calendar handling.
- Every accepted LGB or successor model has stored evaluation metrics: direction accuracy, IC, RankIC, long-short return, bucket returns, coverage, and latest finite prediction count.
- Every accepted trading signal has a cost-aware Qlib backtest artifact or is explicitly marked research-only.
- External research libraries are introduced only after a local Qlib-first baseline exposes a measured gap.
- User-facing long-horizon sections are labeled as observation/watchlist until supported by a dedicated long-horizon model and backtest.
- Multi-bagger research uses investable labels and purged validation; raw future maximum return is not an acceptance label.

## 9. Phase 0 Implementation Notes

Implemented on 2026-05-07:

- Added `scripts/check_qlib_data_health.py`.
  - Validates calendar presence and freshness.
  - Validates Qlib feature bin format.
  - Rejects malformed bins where the first value is `NaN` instead of a valid `start_index`.
  - Checks required OHLCV/amount fields and latest close coverage.
- Added `scripts/smoke_lgb_predict.py`.
  - Runs the exact production `ShortTermModel.load_from_pickle().predict_batch()` path from a real script file.
  - Fails clearly when Qlib raises `cannot convert float NaN to integer`.
- Reworked `scripts/update_qlib_data.py`.
  - Default mode is now incremental.
  - Added `--full` for historical rebuilds.
  - Added source selection: `auto`, `tushare`, `akshare`, `baostock`.
  - Added `data/storage/update_manifest.json` support.
  - Added staging directory promotion.
  - Added health check gate before promotion.
  - Added legacy bin repair to convert old full-calendar arrays into Qlib `[start_index, values...]` format.
- Updated `scripts/nightly_train.py`.
  - Stops if data update fails.
  - Stops if Qlib data health fails.
  - Stops if LightGBM training fails.
  - Runs LGB smoke prediction before RL training.
- Updated `scheduler/jobs.py`.
  - Recommendation pushes now prepend model/data status.
  - If LGB inference fails, the report shows `Qlib: DEGRADED` and the fallback reason.
- Added `tests/test_qlib_data_health.py`.
  - Covers rejection of legacy malformed bins.
  - Covers successful repair into Qlib header format.

Current observed state after implementation:

- `scripts/check_qlib_data_health.py --json` correctly fails on the existing local Qlib data because current bins start with `NaN`.
- `scripts/smoke_lgb_predict.py --json` in the `tianshou` conda env correctly fails with `ValueError: cannot convert float NaN to integer`.
- This confirms the new guardrails detect the current broken data before training or recommendation scoring can treat it as healthy.

Follow-up repair result on 2026-05-07:

- Added and ran `scripts/update_qlib_data.py --repair-only`.
- The command repaired 1107 legacy Qlib feature bins.
- Qlib health check now passes on the local `cn_data` directory:
  - calendar count: 6154
  - latest calendar date: 2026-05-07
  - instruments checked: 121
  - latest close coverage: 100%
  - malformed bins: none
- LGB smoke prediction no longer fails with `NaN`/`cannot convert float NaN to integer`.
- The current saved LGB model produced 30 finite predictions and 0 NaN predictions.
- Smoke still fails the production threshold because 30 finite predictions is below the required 100.

Interpretation:

- The NaN-format failure is fixed locally.
- Existing data coverage is still too narrow for production-grade recommendations.
- Do not retrain the production model on the current 121-instrument/30-prediction state unless this is explicitly a small-universe experiment.
- For production, first rebuild or supplement clean market data to restore broad CSI300/CSI500 or all-share coverage, then retrain LightGBM, then run smoke prediction again.

Retraining decision:

- New data is not required to fix the NaN exception itself; the local format repair fixed that.
- New or refreshed data is required before trusting the model in production, because the current universe coverage is too small.
- Preferred path:
  1. Configure a fast source such as Tushare Pro, or use AKShare with refreshed universe as fallback.
  2. Rebuild/supplement Qlib data until instrument coverage is acceptable.
  3. Run `scripts/check_qlib_data_health.py`.
  4. Run `scripts/train_lgb.py`.
  5. Run `scripts/smoke_lgb_predict.py`.
  6. Enable Qlib scoring only if prediction count and finite coverage pass thresholds.

Hard NaN guard implementation on 2026-05-07:

- Updated `models/short_term.py`.
  - Normalizes Qlib prediction output from either `Series` or `DataFrame`.
  - Rejects empty prediction output.
  - Converts scores to numeric and drops only non-finite values from the latest prediction date.
  - Does not silently fall back to an older date if the newest date has only `NaN`.
  - Normalizes stock codes to `SH600000` / `SZ000000` style before returning batch scores.
- Updated `signals/scorer.py`.
  - Coerces `NaN`, `inf`, and invalid upstream scores to neutral `0.0`.
  - Clamps sentiment heat to `[0, 1]` and all signal scores to `[-1, 1]`.
  - This prevents non-finite values from leaking into final recommendation objects, reports, or verification storage.
- Updated `scheduler/jobs.py`.
  - Filters LGB predictions again at the production boundary.
  - Requires at least `LGB_MIN_PREDICTIONS` finite predictions before using Qlib scores.
  - If the count is too low, the recommendation report is explicitly marked `Qlib: DEGRADED` and uses intraday fallback screening.
  - Sanitizes spot quotes, crypto/gold change percentages, macro scores, and mid-term scores before scoring.
- Updated `scripts/train_lgb.py`.
  - Runs Qlib data health check before training.
  - Validates test-segment predictions before saving any model artifact.
  - Requires the latest prediction date to have at least `LGB_MIN_PREDICTIONS` finite predictions.
  - Saves model and dataset through temporary files and `os.replace()` only after validation passes.
  - If validation fails, the previous production `lgb_model.pkl` is left untouched.
- Updated `scripts/smoke_lgb_predict.py`.
  - Uses `LGB_MIN_PREDICTIONS` from config by default.
  - Reports finite and non-finite prediction counts through the exact production inference path.
- Added `tests/test_signal_scorer.py` coverage for non-finite scoring inputs.
- Added `tests/test_scheduler.py` coverage for low-count LGB degradation.

Observed verification after the hard guard:

- `scripts/check_qlib_data_health.py --json` passes:
  - latest calendar date: 2026-05-07
  - instruments checked: 121
  - latest close coverage: 100%
  - malformed bins: none
- `/Users/wangzilu/miniconda3/envs/tianshou/bin/python scripts/smoke_lgb_predict.py --json` no longer reports NaN:
  - prediction count: 30
  - finite prediction count: 30
  - NaN prediction count: 0
  - result: failed because `30 < LGB_MIN_PREDICTIONS=100`
- `/Users/wangzilu/miniconda3/envs/tianshou/bin/python scripts/train_lgb.py` now trains, validates, and refuses to save:
  - prediction count: 1776
  - finite prediction count: 1776
  - latest finite prediction count: 30
  - latest date: 2026-05-07
  - result: failed because `30 < LGB_MIN_PREDICTIONS=100`
  - existing `data/storage/lgb_model.pkl` timestamp remained `May 6 23:08:27 2026`, so the failed run did not overwrite the previous artifact.
- Local regression checks passed:
  - Python compile check for the changed model, scheduler, scorer, smoke, and training scripts.
  - `tests/test_signal_scorer.py tests/test_qlib_data_health.py tests/test_scheduler.py tests/test_verifier.py tests/test_multi_timeframe.py`
  - result: 29 passed, 1 environment warning from `urllib3` / LibreSSL.

Conclusion:

- The original Qlib `NaN` / `cannot convert float NaN to integer` failure is fixed locally by repairing legacy bin format and adding data health checks.
- The production path now treats NaN, empty output, stale latest-date output, and low prediction coverage as hard failures.
- The current blocker is not NaN anymore. It is insufficient latest-date universe coverage: only 30 usable latest-day predictions, below the configured minimum of 100.

## 10. Phase 0.5 Implementation Notes: Evening Index Forecast

Implemented on 2026-05-07:

- Added `signals/index_predictor.py`.
  - Creates a structured next-trading-day `沪深300` forecast for the 22:00 evening outlook.
  - Outputs:
    - target date
    - direction: `看涨` / `看跌` / `震荡`
    - expected change percentage
    - lower/upper change range
    - up probability
    - confidence
    - drivers, risks, and data status
  - Uses a transparent baseline rather than pretending to have a trained index model.
  - Inputs include A-share index momentum, US/HK index lead, crypto/gold risk appetite, geo/policy factors, and Qlib breadth when available.
  - If Qlib is degraded, the forecast still runs but marks the risk explicitly.
- Updated `scheduler/jobs.py`.
  - 22:00 `run_evening_outlook()` now generates the structured index forecast.
  - The forecast block is prepended to the pushed evening report.
  - The forecast is recorded to SQLite through `Verifier.record_market_prediction()`.
  - If the LLM outlook text is unavailable, the structured forecast still gets pushed.
  - 15:30 `run_daily_summary()` now verifies due market forecasts using current `沪深300` change percentage and prepends a compact verification block to the close summary.
- Updated `tracker/verifier.py`.
  - Added `market_predictions` table.
  - Added record, due-query, verify, cumulative stats, and report generation methods for index forecasts.
  - Tracks direction correctness and interval hit rate separately.
- Updated `signals/llm_analyst.py`.
  - The evening prompt now includes the structured index forecast.
  - The first paragraph of the LLM outlook is required to cite direction, range, up probability, and confidence.
- Updated `data/collectors/global_indices.py`.
  - `format_for_report()` can now format already-fetched index data to avoid double fetching.
- Updated `main.py`.
  - Added one-shot command entry points:
    - `--morning`
    - `--sell-check`
    - `--daily-summary`
    - `--evening-outlook`
  - This allows crontab/launchd to call each scheduled job directly without requiring a long-running scheduler process.
- Added tests:
  - `tests/test_index_predictor.py`
  - market prediction record/verify coverage in `tests/test_verifier.py`
  - 22:00 structured forecast integration coverage in `tests/test_scheduler.py`

Verification after implementation:

- Python compile check passed for the changed predictor, scheduler, LLM, verifier, collector, main entrypoint, and tests.
- Regression subset passed:
  - `tests/test_index_predictor.py`
  - `tests/test_signal_scorer.py`
  - `tests/test_qlib_data_health.py`
  - `tests/test_scheduler.py`
  - `tests/test_verifier.py`
  - `tests/test_multi_timeframe.py`
  - result: 34 passed, 1 environment warning from `urllib3` / LibreSSL.
- `git diff --check` passed.

Important limitation:

- This is a verifiable baseline forecast, not a trained index alpha model.
- The next step is to collect enough historical prediction/actual pairs, then replace or augment the baseline with a trained index model and evaluate MAE, direction hit rate, interval coverage, and calibration.

## 11. Phase 0.6 Implementation Notes: Restore LGB Coverage From Data

Implemented on 2026-05-07:

- Root cause:
  - Scheduler was correctly refusing to use LGB when finite predictions were below `LGB_MIN_PREDICTIONS=100`.
  - The failure was not a scoring threshold problem. The production Qlib `cn_data` universe had shrunk to a small/damaged local instrument set.
  - Before the data repair, smoke inference only produced 30 finite latest-day predictions, so scheduler marked Qlib as degraded.
- Data repair:
  - Refreshed the LGB inference universe from AKShare index constituents.
  - Updated `data/storage/qlib_data/cn_data` for current `csi300` coverage.
  - Synced Qlib instrument files from actual feature coverage instead of keeping stale local files.
- Code hardening:
  - Added `LGB_INFERENCE_UNIVERSE=csi300` and `LGB_MIN_DATA_INSTRUMENTS=250` config knobs.
  - `scripts/check_qlib_data_health.py` now supports `--min-instruments`.
  - `scripts/update_qlib_data.py` now:
    - forces universe refresh when the local LGB inference universe is too small;
    - validates both the requested update universe and the LGB inference universe before promotion;
    - can run a staged LGB smoke check with `--lgb-smoke-check`;
    - refuses promotion if the staged data cannot support enough finite LGB predictions.
  - `scripts/nightly_train.py` now refreshes the LGB inference universe every night and runs the staged LGB smoke gate before training.
  - `models/short_term.py` and `scripts/train_lgb.py` now use `LGB_INFERENCE_UNIVERSE` instead of hard-coded `csi300`.
  - Added `scripts/check_scheduler_lgb_status.py` to verify scheduler LGB status from a real script file, avoiding Qlib/joblib stdin spawn failures on macOS.
  - Added `config/qlib_runtime.py` so all Qlib entry points share the same init path and can opt into `QLIB_DEBUG_SAFE=1` for single-kernel/threading diagnostics.
- Current verification:
  - `scripts/check_qlib_data_health.py --qlib-dir data/storage/qlib_data/cn_data --universe csi300 --min-instruments 250 --json`
    - ok: true
    - latest calendar date: 2026-05-07
    - instruments checked: 280
    - latest close coverage: 100%
    - malformed bins: none
  - `/Users/wangzilu/miniconda3/envs/tianshou/bin/python scripts/smoke_lgb_predict.py --json`
    - ok: true
    - prediction count: 280
    - finite prediction count: 280
    - NaN prediction count: 0
    - min predictions: 100
  - `/Users/wangzilu/miniconda3/envs/tianshou/bin/python scripts/check_scheduler_lgb_status.py --json`
    - status: ok
    - count: 280
    - min required: 100
    - used by scheduler: true

Conclusion:

- LGB is no longer being forced in by lowering scheduler safeguards.
- The scheduler LGB gate remains strict.
- The current production data now gives enough finite latest-day predictions for scheduler to use LGB.
- Future nightly data updates should fail before promotion if data coverage regresses below the LGB requirement.

## 12. Phase 0.7 Implementation Notes: Next-Day Market Forecast and Multi-Horizon Recommendations

Implemented on 2026-05-07:

- 22:00 evening module:
  - Keeps the structured `明日大盘量化预测` block.
  - The target line now explicitly says it is forecasting `A股大盘（沪深300）`.
  - The block includes:
    - next trading date
    - direction
    - expected next-day change percentage
    - lower/upper expected range
    - up probability
    - confidence
  - Adds a deterministic `明日短线候选` block based on LGB scores.
  - Each short candidate shows:
    - stock code
    - raw LGB score
    - estimated next-day change percentage
- Daily recommendation module:
  - Recommendations are now grouped into:
    - `短线（明日）`
    - `中线（1-4周）`
    - `长线（1-3月）`
  - Short-term recommendations carry `next_day_change_pct`.
  - The pushed report prepends a deterministic `长中短线分类推荐` block before the LLM narrative.
  - The LLM prompt receives the grouped recommendation text and is required to cite the short-term next-day change estimate.
- Scoring data model:
  - `signals.scorer.Recommendation` now includes:
    - `horizon`
    - `horizon_score`
    - `next_day_change_pct`

Current implementation detail:

- Short-term next-day stock change is estimated from LGB's short-term return score:
  - when LGB is available: `LGB score * 100 / PREDICTION_HORIZON_DAYS`, blended with a small intraday momentum term;
  - when LGB is not available: conservative intraday momentum fallback.
- This is a practical production estimate, not a separately trained one-day model yet.
- A later version should train a dedicated next-day return model and verify MAE / direction hit rate separately from the current 5-day recommendation verifier.

Verification:

- Python compile check passed for:
  - `scheduler/jobs.py`
  - `signals/scorer.py`
  - `signals/llm_analyst.py`
  - `signals/index_predictor.py`
- Focused tests passed:
  - `tests/test_signal_scorer.py`
  - `tests/test_scheduler.py`
  - `tests/test_index_predictor.py`
  - result: 13 passed, 1 environment warning from `urllib3` / LibreSSL.

## 13. Phase 0.8 Implementation Notes: Reliable Production Scheduling

Implemented on 2026-05-07:

- Added `scheduler/job_status.py`.
  - Persists per-job status to `data/storage/job_status.json`.
  - Records:
    - `status`: running / success / failed
    - `started_at`
    - `finished_at`
    - `duration_seconds`
    - `run_count`
    - `error`
    - compact traceback on failure
- Added `scripts/run_with_status.py`.
  - Wraps any scheduled command and updates `job_status.json`.
  - Cron jobs use this wrapper so failures are observable outside log files.
- Added `scripts/install_crontab.py`.
  - Generates an idempotent managed crontab block.
  - Removes old project-specific `main.py --run-now` style entries.
  - Keeps unrelated user crontab lines.
  - Uses the conda `tianshou` Python path by default:
    - `/Users/wangzilu/miniconda3/envs/tianshou/bin/python`
- Installed the managed production crontab.
  - 09:20 `main.py --morning`
  - 14:30 `main.py --sell-check`
  - 15:30 `main.py --daily-summary`
  - 22:00 `main.py --evening-outlook`
  - 09:35-15:35 hourly `main.py --risk-check`
  - 17:00 Qlib data update with LGB data coverage gate and smoke check
  - 17:35 after-close LGB retraining
  - 17:55 after-close LGB smoke check
  - 04:00 nightly train pipeline
- Updated `main.py`.
  - Long-running APScheduler jobs are now wrapped with the same job status persistence.
- Updated `scripts/nightly_train.py`.
  - RL training remains logged, but is non-blocking after data and LGB steps succeed.
  - This prevents an experimental RL failure from marking the whole nightly LGB/data pipeline failed.

Operational effect:

- The old 09:50 and 13:50 `--run-now` cron entries were replaced.
- 15:30 and 22:00 pushes no longer depend on a manually running `python main.py` process.
- The 22:00 recommendation should use a same-day post-close refreshed LGB model when 17:00 data update and 17:35 retraining pass.

Verification:

- `scripts/install_crontab.py --dry-run` produced the expected managed block.
- `scripts/install_crontab.py --apply` installed the block successfully.
- `crontab -l` shows only the managed stock prediction schedule for this project.
- `scripts/run_with_status.py` smoke run wrote a successful status record.
- Focused tests passed:
  - `tests/test_schedule_status.py`
  - `tests/test_scheduler.py`
  - result: 8 passed, 1 environment warning from `urllib3` / LibreSSL.

## 14. Phase 0.9 Implementation Notes: LGB Runtime Cache and RL Training

Implemented on 2026-05-07:

- Diagnosed the runtime warning:
  - User-observed warning: `Failed to load LGB model: No module named 'qlib'`.
  - The `tianshou` conda environment itself can import Qlib from:
    `/Users/wangzilu/miniconda3/envs/tianshou/lib/python3.11/site-packages/qlib`.
  - A shell prompt showing `(tianshou)` is not enough proof that `python3` resolves to the conda interpreter. One observed manual run loaded packages from `/Users/wangzilu/Library/Python/3.9`, which indicates a system/user Python path was involved.
  - `main.py` now logs the exact `sys.executable`, Python version, `CONDA_PREFIX`, Qlib module path, and CLI args at startup.
  - Production cron remains pinned to:
    `/Users/wangzilu/miniconda3/envs/tianshou/bin/python`.

- Added validated LGB prediction cache:
  - New helper: `models/lgb_cache.py`.
  - New config:
    - `LGB_PREDICTION_CACHE_PATH=data/storage/lgb_latest_predictions.json`
    - `LGB_CACHE_MAX_AGE_DAYS=7`
  - `scripts/smoke_lgb_predict.py` now writes a validated cache after a successful smoke run.
  - `scheduler/jobs.py` now writes the cache after live LGB inference succeeds.
  - If live LGB inference fails, scheduler can still use a fresh validated cache instead of dropping directly to intraday fallback.
  - Reports now distinguish `Qlib: OK/live` from `Qlib: OK/cache`.

- Current LGB verification:
  - `/Users/wangzilu/miniconda3/envs/tianshou/bin/python scripts/smoke_lgb_predict.py --json`
  - Result:
    - ok: true
    - prediction count: 280
    - finite prediction count: 280
    - NaN prediction count: 0
    - cache latest date: 2026-05-07
    - cache path: `data/storage/lgb_latest_predictions.json`
  - A manual recommendation run at 21:48 also logged:
    - Qlib initialized successfully
    - `Loaded LGB predictions for 280 stocks`

- Updated RL training path:
  - `scripts/train_rl.py --lgb-score-mode latest` now reads `data/storage/lgb_latest_predictions.json` instead of calling Qlib/LGB inference inside the RL process.
  - This avoids the prior macOS/Qlib/joblib crash path during RL training.
  - RL state includes:
    - Alpha158 features
    - cached Qlib score
    - sentiment score placeholder
    - sentiment heat placeholder
    - market regime placeholder
    - position
    - unrealized return
  - Historical sentiment is still not available, so the current RL run is explicitly marked:
    `sentiment_mode=neutral_placeholder_until_historical_store_exists`.

- Added RL numerical stability guards:
  - Sanitize `NaN`, `inf`, and `-inf` in Alpha158 features and environment observations.
  - Clip Alpha158 features to a bounded range before feeding the network.
  - Clip rewards to a bounded range.
  - Skip DQN updates when Q values, targets, loss, or gradient norm are non-finite.
  - Refuse to save a new RL model if metrics or model weights are non-finite.
  - Metrics JSON is written with `allow_nan=False`, so invalid JSON `NaN` cannot be persisted again.

- RL training result:
  - First 120-stock run exposed RL `mean_loss=NaN`; that run is retained in `rl_metrics.json` as `valid=false`.
  - After stability guards, a smoke run passed:
    - max envs: 8
    - epochs: 2
    - `skipped_nonfinite_updates=0`
  - Stable manual background training then completed:
    - job id: `rl_manual_train_stable`
    - log: `logs/rl_manual_train_20260507_2232.log`
    - max envs: 120
    - epochs: 10
    - step per epoch: 2000
    - finite loss through all 10 epochs
    - `skipped_nonfinite_updates=0`
    - status: success
    - saved model: `data/storage/rl_model.pt`
    - saved metrics: `data/storage/rl_metrics.json`

Important limitation:

- The RL model is trained and saved, but it remains `deployed=false`.
- It should be evaluated as an experimental execution/timing layer, not yet as a production scoring source.
- To make the original "Qlib score + sentiment" idea real, the next phase must persist historical daily sentiment snapshots per stock and backfill enough history to train against non-neutral sentiment states.

## 15. Phase 1.0 Implementation Notes: Evening Push Format

Implemented on 2026-05-07:

- Reworked the 22:00 evening push from a stacked information dump into a fixed narrative structure.
- The report now follows this order:
  1. 世界大事
  2. 对世界格局的影响
  3. 对投资的影响
  4. 明日A股大盘预测
  5. 个股预测
  6. 黄金预测
  7. 加密货币预测
- The LLM only writes the first three macro sections.
- Deterministic code now writes the numeric forecast sections, so the same LGB/index/gold/crypto information is not repeated by the LLM.
- A股大盘预测 now splits:
  - 上证
  - 深证
  - 北证
- Follow-up update:
  - Added 科创 to the user-facing A-share forecast.
  - Added `科创50` to global index collection.
  - The 22:00 report now shows:
    - 上证
    - 深证
    - 北证
    - 科创
- Added data collection entries for:
  - 深证成指
  - 北证50
- Added deterministic stock forecast blocks:
  - 短线前五
  - 中线前五
  - 长线前五
  - 综合前五
- Current stock horizon ranking inputs:
  - LGB latest prediction score
  - same-day spot momentum from the A-share snapshot cache
  - liquidity proxy from spot volume
- Limitation:
  - The long-term stock list is still a "longer-horizon observation" ranking, not a true fundamental long-term model yet.
  - It should be replaced by a fundamentals/industry/valuation model when that data is available.
- The report still records/verifies the broad `沪深300` forecast internally, but the user-facing push presents the clearer 上证/深证/北证 view.

Design intent:

- The push should read like a concise evening strategy note:
  - first identify the world event;
  - then explain how it changes the global pattern;
  - then map that into risk appetite, liquidity, and tomorrow's trading choices.
- Numeric predictions are kept in one place only.
- If the LLM is unavailable, the system falls back to a short structured macro section rather than sending a broken or empty report.
