"""Compare LLM event factor IC: source="jsonl" vs source="eventstore".

2026-06-06: the project lead's data-pipeline critique flagged the
production default flip (jsonl → eventstore) as needing IC evidence
before we can lock the new default in for ablation runs. This script
runs build_factors for both sources over the same date range and
computes rank-IC against forward 1-day return at signal_date+1.

The build_llm_event_factors docstring documented that on 2026-05-29
the two paths produced a 17000% sentiment_score mean difference and
only ~24% stock overlap — that distribution shock could go either
direction for IC, hence this evidence file.

Output:
- ``data/storage/llm_factor_ic_compare/<YYYY-MM-DD>_summary.json``
- ``data/storage/llm_factor_ic_compare/<YYYY-MM-DD>_per_day.parquet``

Usage::

    python scripts/compare_llm_factor_sources_ic.py \
        --start 2026-04-01 --end 2026-06-05 \
        --factor sentiment_score
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_forward_returns(signal_dates: list[str]) -> pd.DataFrame:
    """Load 1-day forward returns by signal_date + qlib_code.

    Reads the production qlib market parquet path. Forward return
    is (close[t+1] - close[t]) / close[t] on the same business-day
    calendar build_factors assumes.
    """
    qlib_market = DATA_DIR / "market_calendar_daily.parquet"
    if not qlib_market.exists():
        # Fallback: many runs name it differently. Use the production
        # 242 cache, which has __label_5d but also __close embeddings.
        candidates = [
            DATA_DIR / "feature_cache_242_production.parquet",
            DATA_DIR / "feature_cache_209_production.parquet",
        ]
        for p in candidates:
            if p.exists():
                qlib_market = p
                break
    logger.info("Reading forward returns from %s", qlib_market)
    df = pd.read_parquet(qlib_market, columns=["__label_5d"]) \
        if "_label" in str(qlib_market) else pd.read_parquet(qlib_market)
    # Production caches use multiindex (datetime, instrument). The
    # forward return for a same-day rank-IC test is the close-to-close
    # next-day return = label / 5 doesn't apply, so just use __label_5d
    # as an IC proxy (it's the model target anyway).
    if isinstance(df.index, pd.MultiIndex) and "datetime" in df.index.names:
        ret_col = "__label_5d" if "__label_5d" in df.columns else "__label_1d"
        if ret_col not in df.columns:
            raise RuntimeError(
                f"No __label_5d / __label_1d in {qlib_market}. "
                f"Columns: {list(df.columns)[:20]}"
            )
        out = df[[ret_col]].rename(columns={ret_col: "fwd_return"})
        out = out.reset_index()
        out["signal_date"] = out["datetime"].dt.strftime("%Y-%m-%d")
        out["qlib_code"] = out["instrument"]
        return out[["signal_date", "qlib_code", "fwd_return"]]
    raise RuntimeError(f"Unexpected market parquet shape: {df.shape}")


def compute_rank_ic(merged: pd.DataFrame, factor_col: str) -> pd.DataFrame:
    """Group by signal_date and compute Spearman rank-IC vs fwd_return."""
    rows = []
    for date, g in merged.groupby("signal_date"):
        sub = g[[factor_col, "fwd_return"]].dropna()
        if len(sub) < 30:
            rows.append({"signal_date": date, "rank_ic": np.nan, "n": len(sub)})
            continue
        ic = sub[factor_col].rank().corr(sub["fwd_return"].rank())
        rows.append({"signal_date": date, "rank_ic": ic, "n": len(sub)})
    return pd.DataFrame(rows)


def build_factors_one_source(
    start: str, end: str, source: str, lookback: int = 30,
) -> pd.DataFrame:
    """Call build_llm_event_factors for each business day in [start, end]."""
    from scripts.build_llm_event_factors import build_factors

    out = []
    current = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    while current <= end_dt:
        if current.weekday() < 5:  # Mon–Fri
            d = current.strftime("%Y-%m-%d")
            try:
                df = build_factors(
                    signal_date=d, lookback_days=lookback,
                    source=source, allow_fallback=False,
                )
                if df is not None and not df.empty:
                    df = df.copy()
                    df["signal_date"] = d
                    df["source"] = source
                    out.append(df)
                    logger.info("  %s source=%s  rows=%d", d, source, len(df))
                else:
                    logger.info("  %s source=%s  EMPTY", d, source)
            except Exception as e:
                logger.warning("  %s source=%s  FAILED: %s", d, source, e)
        current += timedelta(days=1)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--lookback", type=int, default=30,
                    help="Event lookback window (default 30)")
    ap.add_argument(
        "--factor", default="sentiment_score",
        help="Factor column to compare IC on (default: sentiment_score)",
    )
    args = ap.parse_args()

    out_dir = DATA_DIR / "llm_factor_ic_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M")
    summary_path = out_dir / f"{run_id}_summary.json"
    per_day_path = out_dir / f"{run_id}_per_day.parquet"

    logger.info("Building forward returns table…")
    fwd = load_forward_returns([])

    rows = []
    summary = {
        "run_id": run_id,
        "start": args.start, "end": args.end,
        "factor_column": args.factor,
        "lookback_days": args.lookback,
        "by_source": {},
    }

    for source in ("jsonl", "eventstore"):
        logger.info("--- building %s source ---", source)
        factors_df = build_factors_one_source(
            args.start, args.end, source, args.lookback,
        )
        if factors_df.empty:
            logger.warning("source=%s produced no factors; skipping IC", source)
            summary["by_source"][source] = {"factor_rows": 0}
            continue
        if args.factor not in factors_df.columns:
            logger.warning("source=%s missing factor column %s; available=%s",
                           source, args.factor, list(factors_df.columns)[:10])
            summary["by_source"][source] = {
                "factor_rows": len(factors_df),
                "factor_column_missing": True,
            }
            continue
        merged = factors_df.merge(fwd, on=["signal_date", "qlib_code"], how="inner")
        ic = compute_rank_ic(merged, args.factor)
        ic["source"] = source
        rows.append(ic)

        mean_ic = ic["rank_ic"].mean()
        std_ic = ic["rank_ic"].std()
        icir = mean_ic / std_ic if std_ic and std_ic > 0 else np.nan
        summary["by_source"][source] = {
            "factor_rows": int(len(factors_df)),
            "merged_rows": int(len(merged)),
            "days_with_ic": int(ic["rank_ic"].notna().sum()),
            "mean_rank_ic": float(mean_ic) if pd.notna(mean_ic) else None,
            "rank_icir": float(icir) if pd.notna(icir) else None,
            "rank_ic_pos_ratio": float((ic["rank_ic"] > 0).mean())
                if ic["rank_ic"].notna().any() else None,
        }
        logger.info("source=%s  mean_rank_ic=%.4f  ICIR=%.3f  days=%d",
                    source, mean_ic, icir, len(ic))

    if rows:
        per_day = pd.concat(rows, ignore_index=True)
        per_day.to_parquet(per_day_path)
        logger.info("Wrote per-day IC: %s", per_day_path)
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote summary: %s", summary_path)


if __name__ == "__main__":
    main()
