"""Shadow comparison: KLEN reversal overlay vs pure XGB Top20.

Daily script that compares Top20 picks with and without the KLEN
reversal factor overlay across a weight grid.

Factor: -ts_max(rank(KLEN), 5)
Meaning: stocks with large K-line bodies in recent 5 days get penalized
(short-term reversal after extreme price movements)

24-split validated: Raw IC +0.070 (100% positive), Residual IC +0.045,
Top20 spread +11.4% improvement.

Usage:
    python scripts/shadow_klen_overlay.py [--date YYYY-MM-DD]
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
OUTPUT_DIR = DATA_DIR / "shadow_klen_overlay"

WEIGHT_GRID = [0.0, 0.05, 0.10, 0.15, 0.20]
TOP_N = 20


def safe_zscore(s: pd.Series) -> pd.Series:
    """Compute zscore, handling NaN and constant series."""
    finite = s[np.isfinite(s)]
    if len(finite) < 2 or finite.std() == 0:
        return pd.Series(0.0, index=s.index)
    z = (s - finite.mean()) / finite.std()
    return z.fillna(0.0)


def compute_klen_factor(date: str) -> pd.Series:
    """Compute -ts_max(rank(KLEN), 5) for the given date.

    Needs recent 5 trading days of KLEN from the feature cache.
    """
    try:
        from config.qlib_runtime import init_qlib
        from qlib.data import D

        init_qlib(str(DATA_DIR / "qlib_data" / "cn_data"))

        # Get KLEN for recent 20 days (need 5 for rolling)
        df = D.features(
            D.instruments("all"),
            ["$close/$open - 1"],  # KLEN proxy: close/open ratio - 1
            start_time=pd.Timestamp(date) - pd.Timedelta(days=30),
            end_time=date,
        )
        if df is None or df.empty:
            return pd.Series(dtype=float)

        df.columns = ["klen"]
        klen = df["klen"].abs()  # absolute body length

        # rank per date, then rolling max over 5 days
        klen_rank = klen.groupby(level=0).rank(pct=True)
        ts_max_rank = klen_rank.groupby(level=1).transform(
            lambda x: x.rolling(5, min_periods=3).max()
        )

        # Get the latest date's values
        target = pd.Timestamp(date)
        dates = sorted(ts_max_rank.index.get_level_values(0).unique())
        avail = [d for d in dates if d <= target]
        if not avail:
            return pd.Series(dtype=float)

        use_date = avail[-1]
        factor = -ts_max_rank.loc[use_date]  # negate: high KLEN rank = bad
        factor.name = "klen_reversal"
        return factor

    except Exception as e:
        logger.warning(f"Failed to compute KLEN factor from Qlib: {e}")

        # Fallback: try from feature cache
        try:
            cache = pd.read_parquet(
                DATA_DIR / "feature_cache_174_holder_regime_ma.parquet",
                columns=["KLEN"],
            )
            target = pd.Timestamp(date)
            dates = sorted(cache.index.get_level_values(0).unique())
            avail = [d for d in dates if d <= target]
            if not avail:
                return pd.Series(dtype=float)

            # Use last 10 days
            recent = avail[-10:]
            subset = cache.loc[cache.index.get_level_values(0).isin(recent)]
            klen = subset["KLEN"]
            klen_rank = klen.groupby(level=0).rank(pct=True)
            ts_max_rank = klen_rank.groupby(level=1).transform(
                lambda x: x.rolling(5, min_periods=3).max()
            )

            use_date = avail[-1]
            factor = -ts_max_rank.loc[use_date]
            factor.name = "klen_reversal"
            return factor
        except Exception as e2:
            logger.warning(f"Fallback also failed: {e2}")
            return pd.Series(dtype=float)


def compare_overlay(xgb_preds: pd.Series, factor: pd.Series,
                    weight: float) -> dict:
    """Compare Top-N with and without KLEN overlay at given weight."""
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
            "klen_reversal": round(float(factor.get(code, 0)), 4) if code in factor.index else None,
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
    logger.info(f"=== KLEN Reversal Shadow (weight grid): {date} ===")

    # Load XGB predictions
    pred_path = DATA_DIR / "lgb_latest_predictions.json"
    if not pred_path.exists():
        logger.error("No predictions file")
        return

    preds_raw = json.load(open(pred_path))
    pred_date = preds_raw.get("latest_date", "?")
    predictions = preds_raw.get("predictions", {})
    logger.info(f"XGB predictions: {len(predictions)} stocks (date={pred_date})")

    xgb_preds = pd.Series(predictions, dtype=float)
    xgb_preds.index = xgb_preds.index.str.lower()
    xgb_preds.index.name = "instrument"

    # Compute KLEN factor
    klen_factor = compute_klen_factor(date)
    if klen_factor.empty:
        logger.warning("KLEN factor empty -- no overlay today")
        return

    klen_factor.index = klen_factor.index.str.lower() if hasattr(klen_factor.index, 'str') else klen_factor.index
    common = xgb_preds.index.intersection(klen_factor.dropna().index)
    logger.info(f"KLEN factor: {len(klen_factor)} stocks, {len(common)} common with XGB")

    # Compare across weight grid
    baseline_top = sorted(xgb_preds.nlargest(TOP_N).index)
    weight_results = {}
    for w in WEIGHT_GRID:
        w_key = f"w_{w:.2f}"
        res = compare_overlay(xgb_preds, klen_factor, weight=w)
        weight_results[w_key] = res
        logger.info(f"  weight={w:.2f}: {res['n_affected']} stocks changed in Top20")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "date": date,
        "pred_date": pred_date,
        "weight_grid": WEIGHT_GRID,
        "xgb_top20": baseline_top,
        "variants": weight_results,
        "top_n": TOP_N,
        "n_xgb_stocks": len(xgb_preds),
        "n_factor_stocks": len(klen_factor),
        "n_common": len(common),
    }

    out_path = OUTPUT_DIR / f"{date}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {out_path}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"Shadow KLEN Reversal Overlay (weight grid) -- {date}")
    print("=" * 70)
    print(f"XGB stocks: {len(xgb_preds)}  |  KLEN stocks: {len(klen_factor)}  |  Common: {len(common)}")
    print(f"Weight grid: {WEIGHT_GRID}")
    print(f"XGB Top-{TOP_N}: {', '.join(baseline_top[:5])} ...")

    for w_key, w_entry in weight_results.items():
        n_aff = w_entry["n_affected"]
        print(f"\n--- {w_key}: n_affected={n_aff} ---")
        if w_entry["top20_changed"]:
            for s in w_entry["top20_changed"]:
                klen_str = (f"klen={s['klen_reversal']:+.4f}"
                            if s["klen_reversal"] is not None else "klen=N/A")
                print(f"  {s['change']:>7s}  {s['code']:12s}  "
                      f"xgb_rank={s['xgb_rank']:>5d} -> overlay_rank={s['overlay_rank']:>5d}  "
                      f"{klen_str}")
        else:
            print("  No changes -- overlay had no effect on Top-20.")
    print("=" * 70)


if __name__ == "__main__":
    main()
