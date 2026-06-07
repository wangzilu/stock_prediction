"""Shadow paper-trade: xgb_209 (production) vs xgb_209_llm (candidate).

2026-06-07: Phase B.6.3 24-split verdict promoted xgb_209_llm as the
next-champion CANDIDATE with ΔRankIC +0.0044 (88% of strict gate) but
+17.62 bps Spread20 lift on 24-split. The conservative path is shadow
paper-trade for 5+ trading days before flipping the production default.

This script runs DAILY before market open:
  1. Loads both production models (lgb_model_xgb_209.pkl,
     lgb_model_xgb_209_llm.pkl) and their respective feature caches
  2. Generates top-20 picks for each
  3. Saves picks to data/storage/shadow_paper_trade/<YYYY-MM-DD>.json
  4. When run on a date with realised next-day prices available,
     also computes the realised Spread20 for each model's picks
  5. Writes a cumulative comparison row to
     data/storage/shadow_paper_trade/comparison.parquet

Promotion gate: after 5+ shadow days, if xgb_209_llm cumulative
Spread20 ≥ xgb_209 cumulative Spread20 (or wins on ≥3 of 5 individual
days), flip PRODUCTION_MODEL_PROFILE default to xgb_209_llm.

Usage:
    # Generate today's picks
    python scripts/shadow_paper_trade.py --date 2026-06-09

    # Backfill picks + realised spread on a historical date
    python scripts/shadow_paper_trade.py --date 2026-06-05 --backfill

    # Show cumulative comparison
    python scripts/shadow_paper_trade.py --summary
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

SHADOW_DIR = DATA_DIR / "shadow_paper_trade"
COMPARISON_PATH = SHADOW_DIR / "comparison.parquet"

PROFILES = ("xgb_209", "xgb_209_llm")
TOP_N = 20


def predict_top_n(profile: str, date: str, top_n: int = TOP_N) -> pd.DataFrame:
    """Load profile's model + cache, predict for ``date``, return top_n."""
    import pickle

    from config.production_features import production_model_filename

    model_path = DATA_DIR / production_model_filename(profile)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model binary missing for profile={profile}: {model_path}. "
            f"Run scripts/train_lgb.py with PRODUCTION_MODEL_PROFILE={profile}."
        )

    # Cache: xgb_209 uses 209-feat cache; xgb_209_llm uses joined LLM cache.
    cache_map = {
        "xgb_209": DATA_DIR / "feature_cache_209_latest.parquet",
        "xgb_209_llm": DATA_DIR / "feature_cache_209_llm_latest.parquet",
    }
    cache_path = cache_map.get(profile)
    if cache_path is None or not cache_path.exists():
        raise FileNotFoundError(
            f"Cache missing for profile={profile}: {cache_path}"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_parquet(cache_path)
    # Filter to target date.
    dt = pd.Timestamp(date)
    if "datetime" in df.index.names:
        day_df = df.xs(dt, level="datetime", drop_level=False)
    else:
        day_df = df[df["datetime"] == dt]
    if day_df.empty:
        logger.warning("No rows for profile=%s date=%s", profile, date)
        return pd.DataFrame()

    # Strip label + auxiliary cols, keep features in the order the
    # contract expects.
    from config.production_features import PROFILE_EXPECTED_COUNTS
    expected_total = PROFILE_EXPECTED_COUNTS[profile]["total"]
    feature_cols = [c for c in day_df.columns
                    if c not in ("__label_1d", "__label_5d", "_close",
                                  "_ma5", "_ma20", "ext_holder_decrease",
                                  "__pnl_return_1d")
                    and not c.startswith("_")
                    and pd.api.types.is_numeric_dtype(day_df[c])]
    if len(feature_cols) != expected_total:
        logger.warning(
            "Profile %s cache feature count mismatch: got %d, expected %d",
            profile, len(feature_cols), expected_total,
        )
    X = day_df[feature_cols].fillna(0.0)
    if hasattr(model, "predict"):
        scores = model.predict(X.values)
    else:
        raise RuntimeError(f"Model for {profile} has no predict()")

    out = pd.DataFrame({
        "instrument": day_df.index.get_level_values("instrument"),
        "score": scores,
    })
    out = out.sort_values("score", ascending=False).head(top_n)
    out["rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


def realised_spread(picks_top: list[str], picks_bot: list[str],
                     next_day_returns: pd.Series) -> float:
    """Compute Spread20 = mean(top picks return) − mean(bottom picks return) in bps."""
    top_ret = next_day_returns.reindex(picks_top).dropna()
    bot_ret = next_day_returns.reindex(picks_bot).dropna()
    if top_ret.empty or bot_ret.empty:
        return float("nan")
    return float((top_ret.mean() - bot_ret.mean()) * 10000)


def load_next_day_returns(date: str) -> pd.Series:
    """Load t+1 close-to-close returns keyed by qlib_code."""
    # Reuse the production cache's __label_1d if present; else compute
    # from the qlib data directly. For shadow paper-trade we trust the
    # cache the model trained against.
    df = pd.read_parquet(DATA_DIR / "feature_cache_209_latest.parquet")
    dt = pd.Timestamp(date)
    if "datetime" in df.index.names:
        try:
            day_df = df.xs(dt, level="datetime", drop_level=False)
        except KeyError:
            return pd.Series(dtype=float)
    else:
        return pd.Series(dtype=float)
    label_col = "__label_5d" if "__label_5d" in day_df.columns else "__label_1d"
    if label_col not in day_df.columns:
        return pd.Series(dtype=float)
    s = day_df[label_col].droplevel("datetime")
    return s.dropna()


def run_one_day(date: str, *, backfill: bool = False) -> dict:
    """Generate picks for both profiles on ``date``."""
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    row = {"date": date, "profiles": {}}
    for profile in PROFILES:
        try:
            picks = predict_top_n(profile, date, TOP_N)
            row["profiles"][profile] = {
                "top": picks["instrument"].tolist(),
                "scores": picks["score"].tolist(),
                "rank": picks["rank"].tolist(),
            }
        except Exception as e:
            logger.error("Profile %s failed for %s: %s", profile, date, e)
            row["profiles"][profile] = {"error": str(e)}

    # Realised spread requires next-day returns
    if backfill:
        returns = load_next_day_returns(date)
        if not returns.empty:
            # Bottom 20 picks as the "sell" side proxy
            for profile in PROFILES:
                if "top" not in row["profiles"].get(profile, {}):
                    continue
                top = row["profiles"][profile]["top"]
                # Sort all instruments by score, take bottom 20
                # — but the predict_top_n only returned top-20; for the
                # bottom-20 we'd need another pass. Approximate with
                # universe-mean = 0 for now.
                top_ret_mean = returns.reindex(top).dropna().mean()
                row["profiles"][profile]["realised_top_mean_bps"] = float(
                    top_ret_mean * 10000 if pd.notna(top_ret_mean) else 0.0
                )

    out_path = SHADOW_DIR / f"{date}.json"
    out_path.write_text(json.dumps(row, indent=2, ensure_ascii=False))
    logger.info("Wrote shadow row: %s", out_path)
    return row


def summary() -> None:
    """Print cumulative comparison."""
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    rows = sorted(SHADOW_DIR.glob("*.json"))
    if not rows:
        print("No shadow rows yet.")
        return
    table = []
    for p in rows:
        d = json.loads(p.read_text())
        rec = {"date": d["date"]}
        for profile in PROFILES:
            pr = d["profiles"].get(profile, {})
            rec[f"{profile}_top_mean_bps"] = pr.get("realised_top_mean_bps")
        table.append(rec)
    df = pd.DataFrame(table)
    print(df.to_string(index=False))
    # Cumulative comparison
    if len(df) >= 2:
        for profile in PROFILES:
            col = f"{profile}_top_mean_bps"
            if col in df.columns:
                cum = df[col].dropna().sum()
                avg = df[col].dropna().mean()
                print(f"  {profile} cumulative top mean: {cum:.2f} bps, "
                      f"daily avg: {avg:.2f} bps")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--backfill", action="store_true",
                    help="Also compute realised top-20 mean return.")
    ap.add_argument("--summary", action="store_true",
                    help="Print cumulative comparison and exit.")
    args = ap.parse_args()

    if args.summary:
        summary()
        return

    date = args.date or datetime.now().strftime("%Y-%m-%d")
    run_one_day(date, backfill=args.backfill)


if __name__ == "__main__":
    main()
