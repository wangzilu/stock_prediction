"""Shadow comparison: vol_compression_20 overlay vs pure XGB Top20.

Daily script that compares Top20 picks with and without the
vol_compression_20 factor overlay across a weight grid.

Factor: cs_rank(-ts_std(ROC20, 20))
Meaning: low volatility compression (declining rolling std of 20-day
returns) predicts upcoming breakout.

Usage:
    python scripts/shadow_vol_compression.py [--date YYYY-MM-DD]
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
OUTPUT_DIR = DATA_DIR / "shadow_vol_compression"
PREDICTIONS_PATH = DATA_DIR / "lgb_latest_predictions.json"

WEIGHT_GRID = [0.0, 0.05, 0.10, 0.15]
TOP_N = 20


def safe_zscore(s: pd.Series) -> pd.Series:
    """Compute zscore, handling NaN and constant series."""
    finite = s[np.isfinite(s)]
    if len(finite) < 2 or finite.std() == 0:
        return pd.Series(0.0, index=s.index)
    z = (s - finite.mean()) / finite.std()
    return z.fillna(0.0)


def compute_vol_compression_factor(date: str) -> pd.Series:
    """Compute cs_rank(-ts_std(ROC20, 20)) for the given date.

    Loads ROC20 from the feature cache, computes rolling 20-day std,
    then cross-sectional rank and negate (low vol = high rank).
    """
    try:
        cache = pd.read_parquet(
            DATA_DIR / "feature_cache_174_holder_regime_ma.parquet",
            columns=["ROC20"],
        )
        target = pd.Timestamp(date)
        dates = sorted(cache.index.get_level_values(0).unique())
        avail = [d for d in dates if d <= target]
        if not avail:
            return pd.Series(dtype=float)

        # Need ~25 recent days for rolling(20) to have enough data
        recent = avail[-30:]
        subset = cache.loc[cache.index.get_level_values(0).isin(recent)]
        roc20 = subset["ROC20"]

        # ts_std(ROC20, 20): rolling std per stock over 20 days
        ts_std = roc20.groupby(level=1).transform(
            lambda x: x.rolling(20, min_periods=10).std()
        )

        # Get the latest date's values
        use_date = avail[-1]
        if use_date not in ts_std.index.get_level_values(0):
            return pd.Series(dtype=float)

        latest_std = ts_std.loc[use_date]

        # cs_rank(-ts_std): negate then rank (low vol → high score)
        factor = (-latest_std).rank(pct=True)
        factor.name = "vol_compression_20"
        return factor

    except Exception as e:
        logger.warning(f"Failed to compute vol_compression factor: {e}")
        return pd.Series(dtype=float)


def load_xgb_predictions() -> pd.Series:
    """Load XGB predictions through validated loader (cx round 3 P1-8)."""
    from models.lgb_cache import load_prediction_cache
    from models.prediction_health import PredictionDistributionRed
    try:
        preds, _payload = load_prediction_cache(PREDICTIONS_PATH)
    except FileNotFoundError:
        logger.error("lgb_latest_predictions.json not found at %s", PREDICTIONS_PATH)
        return pd.Series(dtype=float)
    except PredictionDistributionRed as exc:
        logger.error("shadow_vol_compression refusing RED cache: %s — skip.", exc)
        return pd.Series(dtype=float)
    except RuntimeError as exc:
        logger.error("shadow_vol_compression cache load failed: %s", exc)
        return pd.Series(dtype=float)
    s = pd.Series(preds, dtype=float)
    s.index.name = "instrument"
    logger.info("Loaded XGB predictions: %d stocks", len(s))
    return s


def compare_overlay(xgb_preds: pd.Series, factor: pd.Series,
                    weight: float) -> dict:
    """Compare Top-N with and without vol_compression overlay at given weight."""
    baseline_top = set(xgb_preds.nlargest(TOP_N).index)

    z_xgb = safe_zscore(xgb_preds)
    final_score = z_xgb.copy()

    common = xgb_preds.index.intersection(factor.dropna().index)
    if len(common) > 0:
        z_factor = safe_zscore(factor.reindex(xgb_preds.index).fillna(0.0))
        final_score = z_xgb + weight * z_factor

    overlay_top = set(final_score.nlargest(TOP_N).index)

    added = overlay_top - baseline_top
    removed = baseline_top - overlay_top

    def stock_detail(code: str) -> dict:
        return {
            "code": code,
            "xgb_pred": round(float(xgb_preds.get(code, 0)), 6),
            "xgb_zscore": round(float(z_xgb.get(code, 0)), 4),
            "final_score": round(float(final_score.get(code, 0)), 4),
            "vol_compression": round(float(factor.get(code, 0)), 4) if code in factor.index else None,
            "xgb_rank": int(xgb_preds.rank(ascending=False).get(code, 0)),
            "overlay_rank": int(final_score.rank(ascending=False).get(code, 0)),
        }

    top20_changed = []
    for c in sorted(added):
        d = stock_detail(c)
        d["change"] = "added"
        top20_changed.append(d)
    for c in sorted(removed):
        d = stock_detail(c)
        d["change"] = "removed"
        top20_changed.append(d)

    return {
        "top20_changed": top20_changed,
        "n_affected": len(added) + len(removed),
        "overlay_top20": sorted(overlay_top),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=str, default=None,
                        help="Target date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    date = args.date or datetime.now().strftime("%Y-%m-%d")
    logger.info(f"=== Vol Compression Shadow: {date} ===")

    # 1. Load XGB predictions
    xgb_preds = load_xgb_predictions()
    if xgb_preds.empty:
        logger.error("No XGB predictions -- cannot compare")
        sys.exit(1)

    # Normalize index
    xgb_preds.index = xgb_preds.index.str.lower()

    # 2. Compute vol_compression_20 factor
    factor = compute_vol_compression_factor(date)
    if factor.empty:
        logger.warning("Vol compression factor empty -- no overlay today")
        return

    factor.index = factor.index.str.lower() if hasattr(factor.index, 'str') else factor.index
    common = xgb_preds.index.intersection(factor.dropna().index)
    logger.info(f"Vol compression factor: {len(factor)} stocks, {len(common)} common with XGB")

    # 3. Compare across weight grid
    baseline_top = sorted(xgb_preds.nlargest(TOP_N).index)
    weight_results = {}
    for w in WEIGHT_GRID:
        w_key = f"w_{w:.2f}"
        res = compare_overlay(xgb_preds, factor, weight=w)
        weight_results[w_key] = res
        logger.info(f"  weight={w:.2f}: {res['n_affected']} stocks changed in Top20")

    # 4. Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "date": date,
        "weight_grid": WEIGHT_GRID,
        "xgb_top20": baseline_top,
        "variants": weight_results,
        "top_n": TOP_N,
        "n_xgb_stocks": len(xgb_preds),
        "n_factor_stocks": len(factor),
        "n_common": len(common),
    }

    out_path = OUTPUT_DIR / f"{date}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {out_path}")

    # 5. Print summary
    print("\n" + "=" * 70)
    print(f"Shadow Vol Compression Overlay (weight grid) -- {date}")
    print("=" * 70)
    print(f"XGB stocks: {len(xgb_preds)}  |  Factor stocks: {len(factor)}  |  Common: {len(common)}")
    print(f"Weight grid: {WEIGHT_GRID}")
    print(f"XGB Top-{TOP_N}: {', '.join(baseline_top[:5])} ...")

    for w_key, w_entry in weight_results.items():
        n_aff = w_entry["n_affected"]
        print(f"\n--- {w_key}: n_affected={n_aff} ---")
        if w_entry["top20_changed"]:
            for s in w_entry["top20_changed"]:
                vc_str = (f"vol_comp={s['vol_compression']:+.4f}"
                          if s["vol_compression"] is not None else "vol_comp=N/A")
                print(f"  {s['change']:>7s}  {s['code']:12s}  "
                      f"xgb_rank={s['xgb_rank']:>5d} -> overlay_rank={s['overlay_rank']:>5d}  "
                      f"{vc_str}")
        else:
            print("  No changes -- overlay had no effect on Top-20.")
    print("=" * 70)


if __name__ == "__main__":
    main()
