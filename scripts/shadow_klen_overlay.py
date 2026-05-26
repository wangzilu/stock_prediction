"""Shadow comparison: KLEN reversal overlay vs pure XGB Top20.

Daily script that compares Top20 picks with and without the KLEN
reversal factor overlay (alpha=0.10).

Factor: -ts_max(rank(KLEN), 5)
Meaning: stocks with large K-line bodies in recent 5 days get penalized
(short-term reversal after extreme price movements)

24-split validated: Raw IC +0.070 (100% positive), Residual IC +0.045,
Top20 spread +11.4% improvement.

Usage:
    python scripts/shadow_klen_overlay.py [--date YYYY-MM-DD] [--alpha 0.10]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()
    date = args.date
    alpha = args.alpha

    logger.info(f"=== KLEN Reversal Shadow: {date}, alpha={alpha} ===")

    # Load XGB predictions
    pred_path = DATA_DIR / "lgb_latest_predictions.json"
    if not pred_path.exists():
        logger.error("No predictions file")
        return

    preds_raw = json.load(open(pred_path))
    pred_date = preds_raw.get("latest_date", "?")
    predictions = preds_raw.get("predictions", {})
    logger.info(f"XGB predictions: {len(predictions)} stocks (date={pred_date})")

    # Compute KLEN factor
    klen_factor = compute_klen_factor(date)
    if klen_factor.empty:
        logger.warning("KLEN factor empty — no overlay today")
        return

    logger.info(f"KLEN factor: {len(klen_factor)} stocks")

    # Align
    xgb_series = pd.Series(predictions)
    xgb_series.index = xgb_series.index.str.lower()
    klen_factor.index = klen_factor.index.str.lower() if hasattr(klen_factor.index, 'str') else klen_factor.index

    common = xgb_series.index.intersection(klen_factor.dropna().index)
    xgb_a = xgb_series.reindex(common)
    klen_a = klen_factor.reindex(common)

    logger.info(f"Common stocks: {len(common)}")

    # XGB only: Top20
    xgb_top20 = set(xgb_a.nlargest(20).index)

    # XGB + KLEN overlay: Top20
    xgb_rank = xgb_a.rank(pct=True)
    klen_rank = klen_a.rank(pct=True)
    combined = xgb_rank + alpha * klen_rank
    overlay_top20 = set(combined.nlargest(20).index)

    # Compare
    added = overlay_top20 - xgb_top20
    removed = xgb_top20 - overlay_top20
    kept = xgb_top20 & overlay_top20

    logger.info(f"\nTop20 comparison:")
    logger.info(f"  Kept:    {len(kept)}")
    logger.info(f"  Added:   {len(added)}")
    logger.info(f"  Removed: {len(removed)}")

    for stock in sorted(added):
        xgb_r = int(xgb_a.rank(ascending=False).get(stock, 0))
        ovl_r = int(combined.rank(ascending=False).get(stock, 0))
        klen_v = klen_a.get(stock, 0)
        logger.info(f"    + {stock.upper():12s} xgb_rank={xgb_r:4d} → overlay_rank={ovl_r:4d}  klen={klen_v:+.3f}")

    for stock in sorted(removed):
        xgb_r = int(xgb_a.rank(ascending=False).get(stock, 0))
        ovl_r = int(combined.rank(ascending=False).get(stock, 0))
        logger.info(f"    - {stock.upper():12s} xgb_rank={xgb_r:4d} → overlay_rank={ovl_r:4d}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "date": date,
        "alpha": alpha,
        "pred_date": pred_date,
        "n_common": len(common),
        "xgb_top20": sorted(xgb_top20),
        "overlay_top20": sorted(overlay_top20),
        "added": sorted(added),
        "removed": sorted(removed),
        "n_added": len(added),
        "n_removed": len(removed),
    }

    out_path = OUTPUT_DIR / f"{date}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
