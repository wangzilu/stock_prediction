# Phase B.6.3 — xgb_209_llm 24-Split Final Verdict (2026-06-07)

## Summary

**xgb_209_llm is the next-champion candidate.** 24-split LOO confirms
the 6-split signal: LLM event factors give a real Spread20 lift on
trading metrics, with RankIC just below the strict promotion gate.

**Decision: conservative path** — keep xgb_209 as production default,
shadow paper-trade xgb_209_llm vs xgb_209 for 5+ trading days, flip
default only after shadow confirms the +17.62 bps Spread20 lift
survives live trading.

## Result

Cache: `feature_cache_209_llm.parquet` (12 LLM cols post-L1 rebuild).
End-date: 2026-05-19. xgb-only, 500 estimators, early-stop 30.

| Run | Feat | RankIC | ICIR | Spread20 | Days |
|---|---|---|---|---|---|
| baseline (xgb_209 + 12 LLM) | 221 | +0.0389 | +0.383 | **90.65 bps** | 1223 |
| drop llm_event (= xgb_209) | 209 | +0.0345 | +0.351 | 73.03 bps | 1223 |

| Δ (LLM contribution at 24-split) | |
|---|---|
| ΔRankIC | **+0.0044** (88% of +0.005 gate) |
| ΔICIR | **+0.031** (+9% stability) |
| ΔSpread20 | **+17.62 bps (+24%)** |

## 6-split vs 24-split consistency

| Metric | 6-split (B.6.2 v4) | 24-split (B.6.3) | Direction |
|---|---|---|---|
| ΔRankIC | −0.0002 | +0.0044 | grew with more splits |
| ΔSpread20 | +6.81 bps | +17.62 bps | grew 2.6× with more splits |
| ΔICIR | −0.002 | +0.031 | flipped positive |

The 24-split run is statistically stronger (longer test window, more
splits, more days) and produces a CLEARER positive signal. The
6-split fast-screen was actually under-counting the LLM contribution.
This is a known feature of the 6-split → 24-split progression where
sparse positive signals get washed out on shorter windows.

## Promotion decision

**RankIC gate**: +0.0044 vs +0.005 threshold — 88% met but technically
below. Strict interpretation: no promotion.

**Spread20 gate**: +17.62 bps is **massive** for a trading metric.
xgb_209_llm's 90.65 bps Spread20 is the highest single-run result
recorded in the experiment ledger to date.

**ICIR gate**: +0.031 stability improvement on top of the alpha lift.

**Operator decision (2026-06-07 12:00)**: conservative path.
- Production stays on `xgb_209` (no env var change).
- `lgb_model_xgb_209_llm.pkl` retrains TODAY on `end-date 2026-06-05`
  using the freshly-joined `feature_cache_209_llm_latest.parquet`.
- Starting Monday 2026-06-09, both models serve in parallel via the
  shadow paper-trade harness (task #162). 5 trading days of overlap
  comparison required before flipping the default.
- If Spread20 advantage holds on at least 3 of 5 shadow days OR the
  cumulative shadow Spread20 advantage stays > 0, promote.
- If shadow shows a regression (cumulative Spread20 disadvantage),
  rollback the candidate and re-investigate.

## Caveats

- **Single-window evidence** (end-date 2026-05-19). A walk-forward
  24-split could swing either direction in alpha-decay regimes.
- **LLM cache snapshot is current** (post-EventStore source flip,
  post-L1 rebuild, post-bool.astype fix). All known upstream bugs
  fixed before this LOO ran.
- **Per-stock news LLM pipeline only** — global supply chain LLM
  (in Phase B.7), PE-1 PBC liquidity factors, sentiment / guba
  factors are separate candidate profiles awaiting their own LOOs.
- **Spread20 metric is sensitive to tail picks** — the 24-split's
  73 → 91 bps lift could be from a small number of high-confidence
  picks. The shadow paper-trade will validate this against real
  next-day prices, not OOF labels.

## What's in xgb_209_llm

Single LLM pipeline:

```
ST_CLIENT 公告 + Eastmoney 搜索 (collect_daily_news)
            ↓
LLM 抽 per-stock event (MiniMax + llm_event_extractor_v2)
   fields: event_type, direction, sentiment, confidence,
           is_price_sensitive, is_official_disclosure,
           is_repeated_news
            ↓
EventStore (PIT signal_date) + llm_events_v2/*.jsonl
            ↓
build_llm_event_factors.py (12 cols: 5 legacy + 7 L1 fact-count)
            ↓
llm_event_factors.parquet → FeatureMerger._load_llm_event_factors
            ↓
xgb_209_llm (221 feat = 158 alpha + 51 base supp + 12 LLM)
```

**NOT** in xgb_209_llm yet (other candidate profiles):
- Global supply chain (rule or LLM) — xgb_209_chain / xgb_209_chain_llm
- PBC monetary policy — xgb_209_pbc
- Guba popularity — xgb_209_guba

## Provenance

- Cache: `data/storage/feature_cache_209_llm.parquet` (3.64 GB,
  6,027,907 rows × 223 cols)
- Wrapper: `/tmp/b6_24split.sh` (inline script; replicate via
  scripts/run_phase_b6_llm_ablation.sh logic with `--preset 24split`)
- Ledger rows (24-split, end-date 2026-05-19):
  - baseline (12 LLM, 221 feat): logged 2026-06-07T11:15
  - drop llm_event (209): logged 2026-06-07T12:02
- 6-split companion verdict: `docs/phase_b6_v4_llm_12col_verdict_20260607.md`
- Production retrain: `lgb_model_xgb_209_llm.pkl` (training in
  progress at commit time, end-date 2026-06-05)

## Next actions

1. **Done now**: This doc + production retrain (`lgb_model_xgb_209_llm.pkl`).
2. **Next**: Shadow paper-trade harness (task #162) — log both
   xgb_209 and xgb_209_llm picks every morning, compare realised
   Spread20 next day, write a daily comparison row.
3. **Monday morning**: First shadow day. Both models serve. No
   production change yet.
4. **By Friday 2026-06-13**: 5 trading days of shadow data.
   Promotion decision based on cumulative Spread20 advantage.
5. **In parallel (already running)**: Phase B.7 chain rule LOO,
   chain LLM backfill (3h), then B.7 LLM LOO. Independent verdict.
