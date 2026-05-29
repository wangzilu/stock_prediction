"""Shadow comparison: ROC5_tsmin10 overlay vs pure XGB Top20.

Factor: rank(ROC5_tsmin10) — momentum bottom reversal
Meaning: stocks whose 5-day momentum hit the lowest point in past 10 days
tend to rebound (contrarian momentum reversal).

24-split validated: IC +0.047, 100% positive (24/24).

Usage:
    python scripts/shadow_roc5_tsmin10.py [--date YYYY-MM-DD]
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
OUTPUT_DIR = DATA_DIR / "shadow_roc5_tsmin10"

WEIGHT_GRID = [0.0, 0.05, 0.10, 0.15]
TOP_N = 20


def safe_zscore(s):
    finite = s[np.isfinite(s)]
    if len(finite) < 2 or finite.std() == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - finite.mean()) / finite.std()).fillna(0.0)


def compute_factor(date: str) -> pd.Series:
    """Compute rank(ROC5_tsmin10) from factor bank or feature cache."""
    # Try factor bank first
    bank_path = DATA_DIR / "factor_bank.parquet"
    if bank_path.exists():
        try:
            bank = pd.read_parquet(bank_path, columns=["ROC5_tsmin10"])
            target = pd.Timestamp(date)
            dates = sorted(bank.index.get_level_values(0).unique())
            avail = [d for d in dates if d <= target]
            if avail:
                use_date = avail[-1]
                factor = bank.loc[use_date, "ROC5_tsmin10"]
                factor = factor.groupby(level=0).rank(pct=True) if factor.index.nlevels > 1 else factor.rank(pct=True)
                factor.name = "roc5_tsmin10"
                logger.info(f"Factor from bank: {len(factor)} stocks, date={use_date.strftime('%Y-%m-%d')}")
                return factor
        except Exception as e:
            logger.warning(f"Factor bank failed: {e}")

    # Fallback: compute from feature cache
    try:
        cache = pd.read_parquet(
            DATA_DIR / "feature_cache_174_holder_regime_ma.parquet",
            columns=["ROC5"],
        )
        target = pd.Timestamp(date)
        dates = sorted(cache.index.get_level_values(0).unique())
        avail = [d for d in dates if d <= target]
        if not avail:
            return pd.Series(dtype=float)

        recent = avail[-15:]
        subset = cache.loc[cache.index.get_level_values(0).isin(recent)]
        roc5 = subset["ROC5"]
        tsmin10 = roc5.groupby(level=1).transform(
            lambda x: x.rolling(10, min_periods=5).min()
        )
        use_date = avail[-1]
        factor = tsmin10.loc[use_date].rank(pct=True)
        factor.name = "roc5_tsmin10"
        return factor
    except Exception as e:
        logger.warning(f"Fallback failed: {e}")
        return pd.Series(dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()
    date = args.date

    logger.info(f"=== ROC5_tsmin10 Shadow: {date} ===")

    # Load XGB predictions
    pred_path = DATA_DIR / "lgb_latest_predictions.json"
    if not pred_path.exists():
        logger.error("No predictions")
        return
    preds_raw = json.load(open(pred_path))
    xgb = pd.Series(preds_raw["predictions"], dtype=float)
    xgb.index = xgb.index.str.lower()
    logger.info(f"XGB: {len(xgb)} stocks")

    # Factor
    factor = compute_factor(date)
    if factor.empty:
        logger.warning("Factor empty")
        return
    factor.index = factor.index.str.lower() if hasattr(factor.index, "str") else factor.index
    logger.info(f"Factor: {len(factor)} stocks")

    # Compare across weight grid
    baseline_top = sorted(xgb.nlargest(TOP_N).index)
    variants = {}
    for w in WEIGHT_GRID:
        z_xgb = safe_zscore(xgb)
        z_fac = safe_zscore(factor.reindex(xgb.index).fillna(0.0))
        combined = z_xgb + w * z_fac
        overlay_top = set(combined.nlargest(TOP_N).index)
        added = overlay_top - set(baseline_top)
        removed = set(baseline_top) - overlay_top
        variants[f"w_{w:.2f}"] = {
            "n_affected": len(added) + len(removed),
            "overlay_top20": sorted(overlay_top),
        }
        logger.info(f"  w={w:.2f}: {len(added)+len(removed)} changed")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "date": date,
        "weight_grid": WEIGHT_GRID,
        "xgb_top20": baseline_top,
        "variants": variants,
    }
    out_path = OUTPUT_DIR / f"{date}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
