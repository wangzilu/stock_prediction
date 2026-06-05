# A-Share Quant Development Phases — June 2026

**Single-source roadmap** for A-share work as of 2026-06-05.
Consolidates the project lead's 2026-06-05 framing — "now do not ask
who is strongest, first ask who wins on the same exam" — plus the LLM
event critique, the supply-chain critique, and the perf/data fixes
that unlocked the framework tonight.

| Doc | Use when |
|---|---|
| **`ashare-phases-2026-06.md`** (this file) | **Start here.** Roadmap + gates + work order. |
| `cx-institutional-quant-development-roadmap-2026-05-20.md` | Long-range R&D / institutional architecture |
| `cx-phase4-private-fund-roadmap-2026-05-17.md` | Pre-2026-06 phase 4 roadmap (background) |
| `cx-phase5-rl-control-roadmap-2026-05-17.md` | RL control roadmap (deferred — gate behind champion decision) |
| `cc-phase4-final-summary-2026-05-19.md` | 5-19 phase 4 final summary |
| `crypto-dev-phases.md` | Crypto track (separate roadmap) |

Memory anchors:
- [[project_phases]] — overall phase progress
- [[regime-first-architecture]] — "market state recognition > single-strategy optimisation" design principle
- [[ashare-tech-debt-followups]] — open A-share tech debt
- [[experiment-conclusions-20260525]] — pre-2026-06 experiment scoreboard

---

## Constraints That Override Everything

1. **A-share production must not break.** Every change in this doc goes
   through shadow-first when applicable; production champion only
   changes after a documented head-to-head with same code commit +
   same data window + same split + same metrics.
2. **Crypto isolation invariant**: no crypto bug, slow import, network
   call, or cron may slow / break A-share production. Same rule
   applies in reverse — A-share work must not block crypto track.
3. **Honest naming.** The 174-family cache is actually 205 features.
   The xgb_242 production profile has zero 24-split evidence as of
   2026-06-05 22:55. Do not use stale code-vs-data labels.
4. **PIT or shadow, never silent.** Any feature that does not pass a
   strict point-in-time check stays in shadow until it does.

---

## Phase A — Same-Exam Framework (this week)

**Goal:** answer "who actually wins on the same exam?" for xgb_174 /
xgb175 / xgb_242 under one code commit, one data window, one split
config, one metric set.

**Gate to exit Phase A:** `docs/three_way_compare_20260606.md` exists,
all three rows share the same `code_commit` + `data_end` columns in
`data/storage/experiments_ledger.jsonl`, and the project lead has
signed off on the verdict.

### A.0 — Same-exam infrastructure (done 2026-06-05)

- [x] `tracker/experiment_ledger.py` — append-only ledger with
  `model_profile / code_commit / data_end / split_config / cache_path /
  feature_groups / dropped_groups / metrics`.
- [x] `scripts/phase4e_24split_ensemble.py` — `--cache-path / --models /
  --preset / --early-stopping-rounds / --checkpoint-tag` flags + ledger
  hook at the end of every run.
- [x] `scripts/build_feature_cache_242.py` — one-shot builder for the
  production xgb_242 cache (158 Alpha158 + 84 PRODUCTION_SUPPLEMENTARY_GROUPS).
- [x] `scripts/three_way_compare.py` — same-exam comparator.

### A.1 — Run xgb_205-family 24-split (done 2026-06-05 22:53)

Run on commit `72aa580`, end_date `2026-05-19`, 205-feature cache.
Full OOF RankIC **+0.0419**, posRatio 81.82%, Spread20 25.58 bps.
Mean per-split RankIC +0.0301, median +0.0428, std 0.0467.
Recorded as `xgb_205_legacy_174_runner` in the ledger because the
runner label "xgb_174" does not match the 205-col cache it consumed.

### A.2 — Run xgb_242 24-split (in flight 2026-06-05)

- [x] Build `feature_cache_242_production.parquet` via
  `scripts/build_feature_cache_242.py --end 2026-05-19`.
- [ ] Run `scripts/phase4e_24split_ensemble.py --preset 24split
  --models xgb --end-date 2026-05-19 --cache-path
  data/storage/feature_cache_242_production.parquet --checkpoint-tag
  24split_xgb_242`.
- [ ] Ledger row auto-recorded by the runner.

### A.3 — Re-run xgb175 24-split (current code, same end_date)

The 2026-05-26 +0.0785 RankIC is stale. Re-run with the current tree.
Same 205-cache (Alpha158 158 + holder + regime + ma overlay), same
`--end-date 2026-05-19`, fresh code.

### A.4 — Three-way compare + verdict (sign-off needed)

- [ ] `python scripts/three_way_compare.py --split-config 24split
  --data-end 2026-05-19 --markdown docs/three_way_compare_20260606.md`
- [ ] Markdown report committed.
- [ ] Project lead sign-off on champion choice (or "rollback to
  175 / stay on 242 / promote 174").

---

## Phase A.5 — Shadow Containment Hardfix (this week, before Phase B)

**Goal:** stop half-baked shadow factors and explanatory LLM signals
from leaking into the production buy/sell decision path. The four
bugs below all share one shape — a feature labelled "shadow" or
"radar" in comments / docs but actually used as a hard gate in the
runtime code path.

**Gate to exit Phase A.5:** all four hardfixes in production AND
`docs/shadow_containment_audit_20260606.md` documents the verified
old-vs-new behaviour per fix.

These are correctness bugs, not optimisations. They block Phase B
because any ablation result is meaningless while shadow factors can
silently reject buy candidates or read future-dated files.

### A5-1 (P0) — `chain_alpha` hard block is reading uncalibrated raw scores

`factors/candidate_sanitizer.py:322-327` rejects a candidate when
`global_chain_alpha < min_chain_alpha (default -2.0)`. But
`global_chain_alpha` raw distribution is mean ≈ -14.5, min ≈ -47.8 —
the threshold of `-2.0` essentially fires for any name caught in
an industry-level negative broadcast. Companies in 有色 / 半导体 /
战略材料 lose buy eligibility because of policy / commodity events
that hit the whole industry.

Fix:
- Demote `chain_negative` from hard reject to a soft tag (rerank only).
- Add `enable_chain_hard_block` param, default `False`.
- Hard block only ever fires on `company_level_chain_alpha`;
  industry / policy / commodity levels can rerank but not reject.
- Industry-level alpha must be `zscore_by_date * shrink * clip([-3, 3])`
  before any consumer touches it.
- Threshold becomes a quantile or calibrated probability (e.g.
  `chain_risk_z < -2.5`), never a raw score.

### A5-2 (P0) — Chain fallback reads future rows on backfill / replay

`scheduler/jobs.py:1226-1234` falls back to `dates.max()` when the
target date is not in the parquet. For a historical replay the
parquet's `max` IS the future relative to `target_date`, and the
existing `age > 2` check fails silently when `age` is negative.
Result: backfill silently consumes the latest chain factor row even
when it post-dates the signal date — straight look-ahead bias.

Fix:
```python
valid_dates = dates[dates <= dt]
if len(valid_dates) == 0:
    return None
latest = valid_dates.max()
age = trading_day_gap(latest, dt)   # CN calendar, not naive .days
if age > 2:
    return None
```

Plus a unit test that calls `_load_chain_alpha(target=<earlier date>,
parquet_max=<later date>)` and asserts the function returns `None`
without ever calling `df.xs(latest, ...)`.

### A5-3 (P1) — `market_judge` weights LLM despite "radar only" comment

`signals/market_judge.py:36-58` comment: "Verified: LLM direction
accuracy = 33% (worse than random). Index price action is the only
reliable short-term signal. Geo/LLM only used for report text, NOT
for scoring." Code: `final_score = index*w_index + geo*w_geo + llm*w_llm`
with `(w_index, w_geo, w_llm) = (0.60, 0.25, 0.15)` early session.
The comment is a lie at runtime — 40% of the early-session market
judgment IS geo + LLM.

Fix (default):
```python
w_index, w_geo, w_llm = 1.0, 0.0, 0.0
```

Geo + LLM stay in the report text fields (`reason` / `key_events` /
etc) but contribute zero to `final_score`. If a future backtest proves
either deserves weight, raise it explicitly with the backtest commit
linked in the comment.

### A5-4 (P1) — Geo fallback grabs latest-by-filename `global_industry_news`

`scheduler/jobs.py:2274-2280` falls back to
`sorted(gn_dir.glob("*.jsonl"), reverse=True)[:3]` when the RSS
collector returns empty. Replay / backfill reading a date earlier than
the latest news file picks up FUTURE news. On top of that
`global_industry_news` is supply-chain news, not macro / geo — using
it as a macro fallback is also categorically wrong.

Fix:
- Filter `gn_dir.glob("*.jsonl")` by `file_date <= target_date`
  before picking the top 3.
- Freshness cap: skip files older than 1 trading day (CN calendar).
  Beyond that, return `geo_factors = None` rather than silently
  inserting zero.
- Separate source paths: supply-chain news stays at
  `data/storage/global_industry_news/`. Macro / geopolitical news
  goes to the NEW `data/storage/macro_policy_news/<YYYY-MM-DD>.jsonl`
  produced by Phase E.1 → E.4 collectors. The fallback only reads
  the macro-policy directory.

### A5-5 — Audit report `docs/shadow_containment_audit_20260606.md`

Document for each of A5-1 … A5-4:
- The exact pre-fix line range that leaked.
- A replay screenshot or assertion showing the leak.
- The post-fix replay with the same input showing the leak closed.
- Whether any current production recommendation could have been
  affected — and if so, how the recommendation log was retroactively
  annotated.

This is the file the project lead has to sign off on before Phase B
can start; it prevents the same leakage class from being reintroduced
when Phases C/D/E land new shadow factors.

---

## Phase B — Ablation (after Phase A + A.5 verdict)

**Goal:** know which of the 11 PRODUCTION_SUPPLEMENTARY_GROUPS actually
contribute. Two-stage to avoid the 11×24-split cost.

**Gate to enter Phase B:** Phase A has decided which profile is the
champion. Ablation only makes sense relative to one baseline.

### B.1 — 6-split LOO fast screen

`phase4e_24split_ensemble.py --preset 6split --models xgb --drop-group <X>`
for each of:

```
fundamental / capital_flow / macro_zero_baseline / shareholder /
valuation / northbound / quality / st_daily_basic / st_moneyflow /
st_holder_number / cross_market_regime
```

Run 11 jobs × ~7 min ≈ 80 min total wall time. Rank by Δ RankIC
relative to the full baseline. Carry forward the 2–4 groups that
either (a) hurt the baseline when present, or (b) contribute almost
nothing.

### B.2 — 24-split confirmation on suspicious groups

For each carried-forward group, run the full 24-split with that group
dropped, confirm or reject the 6-split signal.

### B.3 — Final group set decision

Update `config/production_features.SUPPLEMENTARY_GROUPS_BY_PROFILE`
to reflect the post-ablation contract. Bump
`PROFILE_EXPECTED_COUNTS` so the gate refuses the old shape.

### B.4 — Re-run training with new group set + roll champion forward

Train + 24-split the post-ablation profile. Promote only if the
24-split RankIC improves AND `docs/three_way_compare_*` rerun shows
the new profile still wins.

---

## Phase C — LLM Event Quality (next 2-3 weeks)

**Goal:** stop pretending LLM-extracted events can predict A-share
returns. Refactor the pipeline into a clean, PIT-safe event database
that future overlays / features can rely on.

**Gate to enter Phase C:** Phase A verdict landed; nothing in Phase C
is allowed to touch production champion features until Phase D.

### C.1 — Expand event_filter generic blacklist (L4)

`factors/event_filter.py` add drop patterns:

```
资金流向日报 / 概念上涨/下跌 / 主力资金净流入/流出这些股 /
即将分红 / 抢权行情 / 股东户数降幅榜 / 突破均线 / 突破年线 /
融资客名单 / 机构调研名单 / 涨停跌停名单
```

Cheap win — drops noise before the LLM ever runs.

### C.2 — Event-type schema validator (L3)

Post-process every LLM event with field-level keyword gates:

- `earnings_*` → require `年报|季报|业绩预告|业绩快报|净利润|营收`
- `share_buyback` → require `回购`
- `dividend_increase` → require `分红|派息|送转`
- `regulatory_penalty` → require `处罚|立案|警示函|监管`

Anything that fails the gate downgrades to `other` or
`routine_announcement`. Keeps the type distribution honest.

### C.3 — Replace synthesized `impact_*` with fact-count factors (L1)

`scripts/build_llm_event_factors.py` deletes the `direction * 0.05`
synthesis. Output columns instead:

```
llm_positive_event_count_3d
llm_negative_event_count_3d
llm_price_sensitive_count_3d
llm_official_event_count_3d
llm_repeated_ratio_3d
llm_event_intensity
```

Old `impact_1d/5d` columns survive one release as a deprecation
window.

### C.4 — Unify on EventStore PIT (L2)

`build_llm_event_factors.py` default `source` flips to `eventstore`;
JSONL stays raw / debug only. **Warning:** distribution will shift —
record a same-day side-by-side under both sources for one week before
the flip so consumers can compare.

### C.5 — Daily LLM factor quality report (L5)

`scripts/llm_factor_quality_report.py` runs after
`build_llm_event_factors` and writes
`data/storage/llm_factor_quality/<YYYY-MM-DD>.json`:

```
events_count / stock_coverage / event_type_distribution /
direction_distribution / repeated_ratio / generic_drop_count /
top_duplicate_titles / source_distribution / PIT_invalid_count
```

A `prefilter_stats` cron entry can read this to alert on drift.

### C.6 — 60-90 day backfill + ablation (L6)

Only AFTER C.1 → C.5 land. Backfill EventStore so we have ≥60 trading
days of clean events, THEN ablation: baseline / baseline +
llm_fact_overlay / baseline + llm_fact_features.

Roll forward only on RankIC AND posRatio improvement — LLM noise is
high enough that one window of luck can fool a single-window test.

---

## Phase D — Supply-Chain Quality (next 4-6 weeks)

**Goal:** turn the supply-chain extractor into a real relation
extractor instead of a "global news theme classifier." Project lead's
critique 2026-06-05 frames this as a 3-layer refactor.

**Gate to enter Phase D:** Phase C.1 → C.5 in production (so the
schema validator + black-list improvements can be reused), and Phase A
champion is set so the supply-chain overlay knows which features it
augments.

### D.1 — Split `global_chain_alpha` into 4 sub-alphas (SC-A1)

`factors/global_supply_chain_extractor.py` and downstream code emit:

```
company_level_alpha
industry_level_alpha
policy_risk_alpha
commodity_alpha
```

Industry-level alpha **must** be shrunk:
`industry_score = zscore_by_date(score) * 0.1 * clip([-1, 1])`.
The current `-47 → +15` raw scale is not usable.

Same-day signal count cap: refuse to broadcast a single shock to
>500 instruments before normalisation; if a shock hits 1470 names the
broadcaster is wrong, not the alpha.

### D.2 — LLM extracts RELATIONS, not direction (SC-A2)

`factors/global_chain_llm_extractor.py` schema becomes:

```json
{
  "is_supply_chain_event": true,
  "source_entity": "Nvidia",
  "affected_entity": "optical-module suppliers",
  "affected_product": "800G optical module",
  "relation": "customer_supplier",
  "shock_type": "demand_increase",
  "direction_for_supplier": +1,
  "direction_for_customer": -1,
  "evidence": "Blackwell demand quote",
  "is_new_information": true,
  "confidence": 0.62
}
```

The LLM never outputs an "A-share goes up / down" call. Direction is
computed from `shock_type × relation` at the consumer layer.

Production cron flips to the LLM extractor (currently the rule
extractor at `scripts/extract_global_supply_chain_events.py:52` is
production).

### D.3 — Edge YAML gets A/B/C/D grading (SC-A3)

`data/config/supply_chain_edges.yaml` (851 rows) annotated per edge:

```yaml
A: 年报 / 公告 confirmed customer-supplier relationship
B: 公司互动 / 订单 / public certification
C: research report inference
D: pure theme mapping
```

Production overlay only ingests A and B. C/D rows feed shadow
overlays with weight 0.1–0.3.

---

## Phase E — Policy Event Factors (independent track, 4-6 weeks)

**Goal:** policy / central-bank / state-media texts are more
authoritative, sparser, and have more persistent industry / style
impact than ordinary news. Build a PIT-safe policy event database
and turn it into low-frequency alpha + regime overlay — NOT into "let
the LLM predict A-share returns."

**Why a separate Phase, not part of Phase C:**
The legacy `_load_macro` was disabled because its parquet was a
single-row latest snapshot broadcast back into history (the look-ahead
bias incident, cx round 3 P1). Policy events MUST be PIT-safe from day
one, on an independent pipeline. Reusing the macro loader would
re-import the same incident class.

**Gate to enter Phase E:** Phase C.4 (EventStore PIT canonical) live —
this reuses that infrastructure rather than rebuilding it.

### E.1 — PBOC liquidity factor (PE-1)

```
scripts/collect_policy_texts.py   --source pbc
  -> data/storage/policy_texts/<YYYY-MM-DD>.jsonl
scripts/extract_policy_events.py  --source pbc
  -> data/storage/policy_events/<YYYY-MM-DD>.jsonl  +  EventStore
scripts/build_policy_factors.py   --source pbc
  -> data/storage/pbc_liquidity_factors.parquet
```

Extracted fields (LLM extracts FACTS, not direction):
`policy_stance` | `liquidity_injection_amount` | `net_injection` |
`repo_rate_change` | `tool_type` | `duration_days` | `unexpectedness`.

Factors produced: `pbc_liquidity_zscore_20d`,
`pbc_easing_dummy`, `pbc_tightening_dummy`, `short_rate_pressure`.

Use as **regime / position sizing input**, not direct stock alpha.

### E.2 — Industry policy support (PE-2)

State Council and ministry policy documents from `gov.cn`.

Extracted fields: `target_industries`, `policy_direction`,
`policy_strength`, `fiscal_support`, `subsidy_or_tax`,
`regulatory_tightening`, `implementation_deadline`.

Factors: `industry_policy_support_5d`, `industry_policy_support_20d`,
`industry_policy_novelty`. Mapped to per-stock via the industry
classification at execution time (not retroactively).

### E.3 — Macro surprise from statistics interpretation (PE-3)

NBS data and the official data portal interpretations (CPI / PPI /
PMI / 社零 / 工业增加值 / etc).

Extracted fields: `macro_surprise`, `inflation_pressure`,
`ppi_upstream_pressure`, `consumption_recovery`,
`manufacturing_momentum`, `real_estate_pressure`, `export_pressure`.

Schema retains `actual` vs `consensus` diff so a downstream surprise
calc does not have to re-parse the headline number.

### E.4 — Xinwen Lianbo theme attention (PE-4)

CCTV Xinwen Lianbo program page + transcript-style summaries.

Extracted fields: `state_media_attention`, `industry_mentions`,
`policy_narrative_score`, `geopolitical_tone`,
`technology_self_reliance_score`, `consumption_stimulus_score`.

Themes (initial 9): 科技自立 / 扩大内需 / 房地产 / 民营经济 /
资本市场 / 机器人 AI / 新能源 / 军工安全 / 一带一路.

Factors: `xinwenlianbo_theme_attention_{theme}_1d`,
`{theme}_5d`, `{theme}_acceleration`.

Use as **industry / theme rotation overlay**, never as a short-term
stock signal.

### E.5 — Strict PIT timing chain (PE-5)

Every policy_factors.parquet row carries:
`publish_time` / `available_time` / `signal_date` / `execution_date`.

PIT rules:
- PBOC 09:20 publish → intraday usable, training uses T+1 open.
- Xinwen Lianbo 19:00-19:30 → 22:00 visible, next-day open execution.
- State Council / ministry intraday publishes → use real publish_time;
  publishes after 15:00 only act on next trading day.
- NBS 09:30 / 10:00 publishes → intraday usable, training uses next_open.

**Backtests must never use `filename date`.** Validator in
`scripts/build_policy_factors.py` refuses to save a row whose
`signal_date <= publish_time` would be physically impossible.

### E.6 — Event study validation (PE-6)

**No training before event study.** For each factor:

- Supportive policy vs target industry T+1 / T+5 excess return.
- Restrictive policy vs target industry T+1 / T+5 excess return.
- Xinwen Lianbo theme intensity vs basket return on T+1 / T+5.
- PBOC net injection vs small-cap / growth / high-beta excess.

Only factors that pass event study graduate to 6-split ablation
(Phase C.6 pattern: baseline / baseline + policy_overlay / baseline +
policy_features). LLM noise + low base rate means single-window IC is
not enough.

### E.7 — Production integration (gated)

If E.6 lands a real edge:
- Policy overlay enters as a shadow scorer first (`scheduler/jobs.py`
  scorer slot, weight 0).
- After one month of shadow IC + drawdown that beats the static
  baseline, weight rises in 0.05 increments per week.
- Never enters PRODUCTION_SUPPLEMENTARY_GROUPS for the trained
  model until 90 days of clean event-study evidence.

---

## Cross-Cutting

### Experiment ledger discipline

Every training / backtest run MUST write a ledger row before claiming
a number. The three_way_compare REFUSES to compare rows from
different `code_commit` or `data_end` unless the operator pins them
explicitly — exactly the cross-time confusion this whole framework
exists to prevent.

### Performance fix bank (done 2026-06-05)

- `models/feature_merger.py::_load_capital_flow_from_history` —
  groupby.rolling instead of `groupby.transform(lambda x: x)` /
  `rolling(...).sum()`. 550s → 22s.
- `models/feature_merger.py::_asof_merge_timeseries` — pre-built
  `{stock: (dates, values)}` dict instead of O(stock × ts_rows) bool
  scan. 208s → 10s.
- `models/feature_merger.py::inject_supplementary_into_handler` and
  `inject_qlib_custom_factors_into_handler` — single `pd.concat` per
  frame instead of 252-op nested loop. T3 inject 890s → 36s.
- `scripts/fetch_fundamental_valuation.py` — `--incremental` lifts
  `start_date` to the parquet's recorded latest_date instead of
  silently skipping stocks. Plus `write_health()` so the freshness
  gate can see successful runs.
- `scheduler/jobs.py` — Recommendation `replace(...)` uses
  `horizon_dailyized_return_pct` field (the property setter was not
  a dataclass field, which crashed the morning horizon grouping).

End-to-end validation: morning_recommendation goes from 1800s SIGTERM
hang to **11:14 wall, 5 recommendations produced**
(2026-06-05 22:00 manual run).

### Deferred / parked work

| Item | Reason for deferral |
|---|---|
| ST mask 历史化 (#91) | needs ST_CLIENT historical namechange API + one-shot backfill; defer until ST_CLIENT vendor responds |
| LLM 6-4 RPM 429 timeout (#104) | superseded by Phase C — the right fix is to drop noise (C.1) and rate-aware schedule, not patch on top of the current pipeline |
| cx rounds 16-24 follow-ups (#117) | mostly absorbed by tonight's perf fixes + ledger; remaining items folded into Phase B |
| 22:30 24-split full run (#120) | done 2026-06-05 22:16-22:53 |
| RL control roadmap | gate behind Phase A champion decision; no rollout while production model itself is unverified |

---

## Status As Of 2026-06-05 23:00

- ✅ Phase A.0 — same-exam infrastructure landed and pushed.
- ✅ Phase A.1 — xgb_205 24-split landed (RankIC +0.0419).
- 🔄 Phase A.2 — xgb_242 cache build in flight; 24-split next.
- ⏸ Phase A.3 — xgb175 24-split next.
- ⏸ Phase A.4 — three-way compare + sign-off.
- ⏸ Phase B / C / D — gated behind Phase A.

Read this doc — not the chat history — when you come back to A-share
work. Update it as phases land.
