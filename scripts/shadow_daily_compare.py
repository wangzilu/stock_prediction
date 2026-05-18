"""Phase 4D: Shadow daily comparison — run champion and shadow side by side.

Every day after close:
1. Both models predict on latest data
2. Compare Top20 picks, overlap, score correlation
3. Track cumulative shadow vs champion divergence
4. Log to shadow_compare.jsonl (append-only)

Designed to be called from crontab after lgb_after_close_train.

Usage:
    python scripts/shadow_daily_compare.py
    python scripts/shadow_daily_compare.py --date 2026-05-19
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config.settings import PREDICTION_HORIZON_DAYS
from config.qlib_runtime import init_qlib
from models.feature_merger import FeatureMerger
from models.feature_pipeline import prepare_features_174, train_xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"
SHADOW_LOG = DATA_DIR / "phase4" / "shadow_compare.jsonl"


def predict_with_model(dataset, merger, segment="test", include_regime=False):
    """Train XGB and predict on segment. Returns (predictions Series, n_features)."""
    import xgboost as xgb
    from datetime import timedelta

    X, y = prepare_features_174(dataset, "train", merger)

    # Add regime features if shadow model
    if include_regime:
        regime = merger._load_cross_market_regime(X.index)
        if regime is not None and not regime.empty:
            X = X.join(regime, how="left")

    Xn = X.values.astype(np.float32)
    yn = y.values.astype(np.float32)
    mask = np.isfinite(yn)

    # Valid
    X_val, y_val = prepare_features_174(dataset, "valid", merger)
    if include_regime:
        regime_v = merger._load_cross_market_regime(X_val.index)
        if regime_v is not None and not regime_v.empty:
            X_val = X_val.join(regime_v, how="left")

    Xv = X_val.values.astype(np.float32)
    yv = y_val.values.astype(np.float32)
    mask_v = np.isfinite(yv)

    model = train_xgb(Xn[mask], yn[mask], Xv[mask_v], yv[mask_v])

    # Test
    X_test, y_test = prepare_features_174(dataset, segment, merger)
    if include_regime:
        regime_t = merger._load_cross_market_regime(X_test.index)
        if regime_t is not None and not regime_t.empty:
            X_test = X_test.join(regime_t, how="left")

    pred = model.predict(xgb.DMatrix(X_test.values.astype(np.float32)))
    predictions = pd.Series(pred, index=X_test.index, name="score")
    predictions = predictions[np.isfinite(predictions)]

    return predictions, X_test.shape[1]


def compare_predictions(champion_pred, shadow_pred, date, top_k=20):
    """Compare champion and shadow predictions for a single date."""
    if date not in champion_pred.index.get_level_values(0):
        return None
    if date not in shadow_pred.index.get_level_values(0):
        return None

    ch = champion_pred.loc[date].sort_values(ascending=False)
    sh = shadow_pred.loc[date].sort_values(ascending=False)

    ch_top = set(ch.head(top_k).index)
    sh_top = set(sh.head(top_k).index)

    overlap = ch_top & sh_top
    ch_only = ch_top - sh_top
    sh_only = sh_top - ch_top

    # Rank correlation on common stocks
    common = ch.index.intersection(sh.index)
    if len(common) > 50:
        from scipy.stats import spearmanr
        rank_corr, _ = spearmanr(ch.loc[common], sh.loc[common])
    else:
        rank_corr = float("nan")

    return {
        "date": str(date)[:10],
        "top_k": top_k,
        "overlap": len(overlap),
        "overlap_pct": round(len(overlap) / top_k, 4),
        "champion_only": len(ch_only),
        "shadow_only": len(sh_only),
        "rank_correlation": round(float(rank_corr), 4) if not np.isnan(rank_corr) else None,
        "champion_top5": [str(s) for s in ch.head(5).index],
        "shadow_top5": [str(s) for s in sh.head(5).index],
    }


def main():
    from qlib.utils import init_instance_by_config
    from datetime import timedelta

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="Compare date (default: latest)")
    args = parser.parse_args()

    init_qlib(QLIB_DATA)
    merger = FeatureMerger(DATA_DIR)

    today = datetime.now()
    # Recent window for quick comparison
    test_end = today.strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=31)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=91)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=92)).strftime("%Y-%m-%d")
    train_start = (today - timedelta(days=365 * 3 + 92)).strftime("%Y-%m-%d")

    logger.info(f"=== Shadow Daily Compare ===")
    logger.info(f"Test: {test_start}~{test_end}")

    dataset = init_instance_by_config({
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
                "kwargs": {"start_time": train_start, "end_time": test_end,
                           "instruments": "all", "label": [LABEL_EXPR]},
            },
            "segments": {
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (test_start, test_end),
            },
        },
    })

    # Champion: XGB 174 (no regime)
    logger.info("Training champion (XGB 174)...")
    t0 = time.time()
    ch_pred, ch_feat = predict_with_model(dataset, merger, include_regime=False)
    logger.info(f"  Champion: {ch_feat} features, {len(ch_pred)} predictions, {time.time()-t0:.1f}s")

    # Shadow: XGB 205 (with regime)
    logger.info("Training shadow (XGB 205)...")
    t1 = time.time()
    sh_pred, sh_feat = predict_with_model(dataset, merger, include_regime=True)
    logger.info(f"  Shadow: {sh_feat} features, {len(sh_pred)} predictions, {time.time()-t1:.1f}s")

    # Compare on latest date (or specified date)
    test_dates = sorted(ch_pred.index.get_level_values(0).unique())
    if args.date:
        compare_date = pd.Timestamp(args.date)
    else:
        compare_date = test_dates[-1]

    logger.info(f"\nComparing on {str(compare_date)[:10]}:")
    result = compare_predictions(ch_pred, sh_pred, compare_date)

    if result:
        result["champion_model"] = f"xgb_{ch_feat}"
        result["shadow_model"] = f"xgb_{sh_feat}"
        result["compared_at"] = datetime.now().isoformat(timespec="seconds")

        logger.info(f"  Overlap: {result['overlap']}/{result['top_k']} ({result['overlap_pct']:.0%})")
        logger.info(f"  Rank corr: {result['rank_correlation']}")
        logger.info(f"  Champion top5: {result['champion_top5']}")
        logger.info(f"  Shadow top5:   {result['shadow_top5']}")

        # Append to shadow log
        SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(str(SHADOW_LOG), "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        logger.info(f"  Logged to: {SHADOW_LOG}")

        # Also compare across all test dates
        logger.info(f"\nFull test period comparison ({len(test_dates)} dates):")
        overlaps = []
        for d in test_dates:
            r = compare_predictions(ch_pred, sh_pred, d)
            if r:
                overlaps.append(r["overlap_pct"])

        if overlaps:
            logger.info(f"  Avg overlap: {np.mean(overlaps):.0%}")
            logger.info(f"  Min overlap: {min(overlaps):.0%}")
            logger.info(f"  Max overlap: {max(overlaps):.0%}")
    else:
        logger.warning(f"  No predictions for {compare_date}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
