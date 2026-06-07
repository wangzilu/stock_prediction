"""Shadow paper-trade: xgb_209 (production) vs xgb_209_llm (candidate).

2026-06-07: Phase B.6.3 24-split verdict promoted xgb_209_llm as the
next-champion CANDIDATE with ΔRankIC +0.0044 (88% of strict gate) but
+17.62 bps Spread20 lift on 24-split. The conservative path is shadow
paper-trade for 5+ trading days before flipping the production default.

This script runs DAILY before market open:
  1. Loads both production models (lgb_model_xgb_209.pkl,
     lgb_model_xgb_209_llm.pkl) and their respective feature caches
  2. Generates top-20 AND bottom-20 picks for each (for true Spread20)
  3. Saves picks to data/storage/shadow_paper_trade/<YYYY-MM-DD>.json
  4. When run on a date with realised next-day prices available,
     also computes the realised Spread20 (top mean − bottom mean
     using __label_1d) for each model's picks
  5. The ``--summary`` mode aggregates daily JSONs into a printed
     cumulative comparison; the historical claim of writing a
     comparison.parquet was removed because the JSONs already serve
     as the canonical per-day records.

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

PROFILES = ("xgb_209", "xgb_209_llm")
TOP_N = 20


def predict_top_n(profile: str, date: str, top_n: int = TOP_N) -> pd.DataFrame:
    """Load profile's model + cache, predict for ``date``.

    2026-06-07: returns the FULL ranked universe (sorted by score desc),
    NOT just top_n. ``predict_top_and_bottom`` slices the head + tail
    for a true Spread20 metric. The ``top_n`` argument is unused after
    the cx P1 #2 refactor; left in the signature only for callers that
    still pass it positionally.
    """
    _ = top_n  # explicitly unused — see docstring
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

    # 2026-06-07 fix: production .pkl is a qlib XGBModel wrapper, not a
    # raw sklearn predictor. Its .predict() expects a Qlib DatasetH.
    # For the shadow harness we bypass that machinery and use the
    # inner xgb.Booster directly via DMatrix.
    import xgboost as xgb
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "predict"):
        raise RuntimeError(
            f"Profile {profile} model has no inner xgb.Booster — "
            f"unexpected pickle structure: {type(model)}"
        )

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

    # Strip label + auxiliary cols. cx P2 #4 fix: feature-count
    # mismatch is now fail-loud, not warning, since the shadow gate
    # decides production promotion.
    from config.production_features import PROFILE_EXPECTED_COUNTS
    expected_total = PROFILE_EXPECTED_COUNTS[profile]["total"]
    feature_cols = [c for c in day_df.columns
                    if c not in ("__label_1d", "__label_5d", "_close",
                                  "_ma5", "_ma20", "ext_holder_decrease",
                                  "__pnl_return_1d")
                    and not c.startswith("_")
                    and pd.api.types.is_numeric_dtype(day_df[c])]
    if len(feature_cols) != expected_total:
        raise RuntimeError(
            f"Profile {profile} cache feature count mismatch: got "
            f"{len(feature_cols)}, expected {expected_total}. "
            f"Refusing to predict so the shadow gate does not consume "
            f"an off-contract result."
        )
    X = day_df[feature_cols].fillna(0.0)
    # cx P1 #1 fix: production .pkl is qlib XGBModel wrapper whose
    # .predict() expects DatasetH. Use the inner xgb.Booster directly.
    # Validate the booster's feature count matches our column set.
    try:
        booster_n = inner.num_features()
        if booster_n != len(feature_cols):
            raise RuntimeError(
                f"Profile {profile} booster expects {booster_n} feats "
                f"but cache supplies {len(feature_cols)}. Cache and "
                f".pkl are out of sync."
            )
    except AttributeError:
        # Older xgb versions expose attributes differently; tolerate
        # but log so the operator notices.
        logger.warning("inner.num_features() unavailable; relying on "
                       "shape match alone.")
    scores = inner.predict(xgb.DMatrix(X.values))

    out = pd.DataFrame({
        "instrument": day_df.index.get_level_values("instrument"),
        "score": scores,
    })
    out = out.sort_values("score", ascending=False)
    return out.reset_index(drop=True)


def predict_top_and_bottom(profile: str, date: str,
                            top_n: int = TOP_N) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (top_n, bottom_n) scored picks for a true Spread20 gate.

    cx P1 #2 fix: the shadow gate is called "Spread20" but the
    pre-fix harness only computed top-mean. A real Spread20 needs
    BOTH ends — top mean − bottom mean — so the metric is
    direction-neutral (beta / market drift cancel out).
    """
    full = predict_top_n(profile, date, top_n=10**9)  # full ranked frame
    if full.empty:
        return pd.DataFrame(), pd.DataFrame()
    top = full.head(top_n).copy()
    top["rank"] = range(1, len(top) + 1)
    bot = full.tail(top_n).copy()
    bot["rank"] = range(len(full) - len(bot) + 1, len(full) + 1)
    return top, bot


def realised_spread(picks_top: list[str], picks_bot: list[str],
                     next_day_returns: pd.Series) -> float:
    """Compute Spread20 = mean(top picks return) − mean(bottom picks return) in bps."""
    top_ret = next_day_returns.reindex(picks_top).dropna()
    bot_ret = next_day_returns.reindex(picks_bot).dropna()
    if top_ret.empty or bot_ret.empty:
        return float("nan")
    return float((top_ret.mean() - bot_ret.mean()) * 10000)


def load_next_day_returns(date: str) -> pd.Series:
    """Load t+1 close-to-close returns keyed by qlib_code.

    cx P1 #3 fix: ALWAYS prefer ``__label_1d`` over ``__label_5d``.
    The shadow gate is a daily Spread20, not a 5-day return; using
    the 5-day label would mis-attribute multi-day trends to the
    daily decision.
    """
    df = pd.read_parquet(DATA_DIR / "feature_cache_209_latest.parquet")
    dt = pd.Timestamp(date)
    if "datetime" in df.index.names:
        try:
            day_df = df.xs(dt, level="datetime", drop_level=False)
        except KeyError:
            return pd.Series(dtype=float)
    else:
        return pd.Series(dtype=float)
    # 2026-06-07: 1-day strictly preferred. If absent, return empty
    # (don't silently substitute 5-day; the caller's gate is a daily
    # metric).
    if "__label_1d" in day_df.columns:
        label_col = "__label_1d"
    else:
        logger.error("__label_1d missing from cache for %s; refusing to "
                     "use __label_5d as a daily proxy.", date)
        return pd.Series(dtype=float)
    if label_col not in day_df.columns:
        return pd.Series(dtype=float)
    s = day_df[label_col].droplevel("datetime")
    return s.dropna()


def run_one_day(date: str, *, backfill: bool = False) -> dict:
    """Generate picks for both profiles on ``date``.

    cx P1 #2 fix: now records BOTH top20 and bottom20 picks so the
    realised spread is a true Spread20 (top mean − bottom mean) rather
    than top-mean alone. The latter inherits market beta and would
    promote a candidate that merely happened to overlap with a
    rising tail.
    """
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    row = {"date": date, "profiles": {}}
    for profile in PROFILES:
        try:
            top, bot = predict_top_and_bottom(profile, date, TOP_N)
            row["profiles"][profile] = {
                "top": top["instrument"].tolist(),
                "top_scores": top["score"].tolist(),
                "bottom": bot["instrument"].tolist(),
                "bottom_scores": bot["score"].tolist(),
            }
        except Exception as e:
            logger.error("Profile %s failed for %s: %s", profile, date, e)
            row["profiles"][profile] = {"error": str(e)}

    # Realised Spread20 requires next-day (1-day) returns
    if backfill:
        returns = load_next_day_returns(date)
        if not returns.empty:
            for profile in PROFILES:
                pr = row["profiles"].get(profile, {})
                if "top" not in pr or "bottom" not in pr:
                    continue
                top_ret = returns.reindex(pr["top"]).dropna()
                bot_ret = returns.reindex(pr["bottom"]).dropna()
                if top_ret.empty or bot_ret.empty:
                    pr["realised_spread20_bps"] = None
                    pr["realised_top_mean_bps"] = (float(top_ret.mean() * 10000)
                                                     if not top_ret.empty else None)
                    pr["realised_bottom_mean_bps"] = (float(bot_ret.mean() * 10000)
                                                       if not bot_ret.empty else None)
                    continue
                pr["realised_top_mean_bps"] = float(top_ret.mean() * 10000)
                pr["realised_bottom_mean_bps"] = float(bot_ret.mean() * 10000)
                pr["realised_spread20_bps"] = float(
                    (top_ret.mean() - bot_ret.mean()) * 10000
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
            rec[f"{profile}_spread20_bps"] = pr.get("realised_spread20_bps")
            rec[f"{profile}_top_mean_bps"] = pr.get("realised_top_mean_bps")
            rec[f"{profile}_bot_mean_bps"] = pr.get("realised_bottom_mean_bps")
        table.append(rec)
    df = pd.DataFrame(table)
    print(df.to_string(index=False))
    # Cumulative comparison — Spread20 is the canonical gate metric.
    if len(df) >= 2:
        print("\n== Cumulative Spread20 (top mean − bottom mean, bps) ==")
        for profile in PROFILES:
            col = f"{profile}_spread20_bps"
            if col in df.columns:
                cum = df[col].dropna().sum()
                avg = df[col].dropna().mean()
                print(f"  {profile}: cumulative {cum:+.2f} bps, "
                      f"daily avg {avg:+.2f} bps over {df[col].notna().sum()} days")
        # Head-to-head days won
        if all(f"{p}_spread20_bps" in df.columns for p in PROFILES):
            both = df.dropna(subset=[f"{p}_spread20_bps" for p in PROFILES])
            if not both.empty:
                wins = (both[f"{PROFILES[1]}_spread20_bps"]
                        > both[f"{PROFILES[0]}_spread20_bps"]).sum()
                print(f"  {PROFILES[1]} won Spread20 on {wins}/{len(both)} days")


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
