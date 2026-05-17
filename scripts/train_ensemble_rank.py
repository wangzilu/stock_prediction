"""Rank-weighted Ensemble: XGB + LGBMRanker on 174-dim.

Fuses XGB (regression, good spread) and LGBMRanker (ranking, good RankIC)
via per-date rank normalization + weighted average.

Usage:
    python scripts/train_ensemble_rank.py
    python scripts/train_ensemble_rank.py --weights 0.5,0.5
    python scripts/train_ensemble_rank.py --search   # grid search weights
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config.settings import PREDICTION_HORIZON_DAYS
from config.qlib_runtime import init_qlib
from models.feature_merger import FeatureMerger
from models.feature_pipeline import (
    prepare_segment_numpy, train_xgb as _train_xgb,
    evaluate_predictions, XGB_PARAMS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"
SEED = 42


def returns_to_rank_label(y, index, n_bins=5):
    """Discretize returns to int labels for Ranker."""
    s = pd.Series(y, index=index)
    labels = s.groupby(level=0).transform(
        lambda x: pd.qcut(x, n_bins, labels=False, duplicates="drop")
    )
    return labels.fillna(0).astype(np.int32).values


def train_xgb(X_train, y_train, X_valid, y_valid):
    return _train_xgb(X_train, y_train, X_valid, y_valid)


def train_ranker(X_train, y_train, X_valid, y_valid, train_idx, valid_idx):
    import lightgbm as lgb

    y_train_int = returns_to_rank_label(y_train, train_idx)
    y_valid_int = returns_to_rank_label(y_valid, valid_idx)

    train_dates = train_idx.get_level_values(0)
    train_groups = train_dates.value_counts().sort_index().values.tolist()
    valid_dates = valid_idx.get_level_values(0)
    valid_groups = valid_dates.value_counts().sort_index().values.tolist()

    dtrain = lgb.Dataset(X_train, label=y_train_int, group=train_groups)
    dvalid = lgb.Dataset(X_valid, label=y_valid_int, group=valid_groups, reference=dtrain)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [20, 50],
        "num_leaves": 128,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 200,
        "reg_lambda": 500,
        "verbose": -1,
        "seed": SEED,
    }
    model = lgb.train(params, dtrain, num_boost_round=500,
                      valid_sets=[dvalid], callbacks=[lgb.early_stopping(50)])
    return model


def rank_normalize(pred, index):
    """Per-date rank percentile normalization (0~1)."""
    s = pd.Series(pred, index=index)
    return s.groupby(level=0).rank(pct=True).values


def ensemble_predictions(pred_xgb, pred_ranker, index, w_xgb=0.5, w_ranker=0.5):
    """Rank-weighted ensemble: normalize each model's predictions to rank percentile,
    then weighted average."""
    rank_xgb = rank_normalize(pred_xgb, index)
    rank_ranker = rank_normalize(pred_ranker, index)
    return w_xgb * rank_xgb + w_ranker * rank_ranker


def evaluate(pred, label, index):
    return evaluate_predictions(pred, label, index)


def main():
    import xgboost as xgb

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="0.6,0.4",
                        help="XGB,Ranker weights (default: 0.6,0.4)")
    parser.add_argument("--search", action="store_true",
                        help="Grid search optimal weights on validation set")
    args = parser.parse_args()

    from qlib.utils import init_instance_by_config
    init_qlib(QLIB_DATA)

    today = datetime.now()
    train_start = (today - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=89)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")

    logger.info("=== Rank-Weighted Ensemble: XGB + LGBMRanker (174-dim) ===")

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

    merger = FeatureMerger(DATA_DIR)

    # Prepare features
    segs = {}
    for seg in ["train", "valid", "test"]:
        logger.info(f"Preparing {seg}...")
        Xn, yn, idx, _ = prepare_segment_numpy(dataset, seg, merger)
        segs[seg] = (Xn, yn, idx)
        logger.info(f"  {seg}: {Xn.shape}")

    X_train, y_train, train_idx = segs["train"]
    X_valid, y_valid, valid_idx = segs["valid"]
    X_test, y_test, test_idx = segs["test"]
    n_feat = X_train.shape[1]
    logger.info(f"Features: {n_feat}")

    # Train XGB
    logger.info("Training XGB...")
    t0 = time.time()
    model_xgb = train_xgb(X_train, y_train, X_valid, y_valid)
    logger.info(f"  XGB done: {time.time()-t0:.1f}s")

    # Train Ranker
    logger.info("Training LGBMRanker...")
    t0 = time.time()
    model_ranker = train_ranker(X_train, y_train, X_valid, y_valid, train_idx, valid_idx)
    logger.info(f"  Ranker done: {time.time()-t0:.1f}s")

    # Get predictions on test
    pred_xgb = model_xgb.predict(xgb.DMatrix(X_test))
    pred_ranker = model_ranker.predict(X_test)

    # Also get valid predictions for weight search
    pred_xgb_val = model_xgb.predict(xgb.DMatrix(X_valid))
    pred_ranker_val = model_ranker.predict(X_valid)

    # Evaluate individual models
    m_xgb = evaluate(pred_xgb, y_test, test_idx)
    m_ranker = evaluate(pred_ranker, y_test, test_idx)

    logger.info(f"\nIndividual models (test):")
    logger.info(f"  XGB:    IC={m_xgb['ic_mean']:+.4f} ICIR={m_xgb['icir']:+.3f} "
                f"RankIC={m_xgb['rank_ic_mean']:+.4f} Spread={m_xgb['top20_spread']*100:+.3f}%")
    logger.info(f"  Ranker: IC={m_ranker['ic_mean']:+.4f} ICIR={m_ranker['icir']:+.3f} "
                f"RankIC={m_ranker['rank_ic_mean']:+.4f} Spread={m_ranker['top20_spread']*100:+.3f}%")

    # Weight search on validation set
    if args.search:
        logger.info("\nGrid searching weights on validation set...")
        best_w = None
        best_score = -999
        weight_results = []

        for w_xgb_pct in range(20, 85, 5):
            w_xgb = w_xgb_pct / 100.0
            w_ranker = 1.0 - w_xgb
            ens_val = ensemble_predictions(pred_xgb_val, pred_ranker_val, valid_idx,
                                           w_xgb, w_ranker)
            m_val = evaluate(ens_val, y_valid, valid_idx)
            # Score = RankIC + Spread (both matter)
            score = m_val["rank_ic_mean"] + m_val["top20_spread"]
            weight_results.append({
                "w_xgb": w_xgb, "w_ranker": w_ranker,
                "rank_ic": m_val["rank_ic_mean"],
                "spread": m_val["top20_spread"],
                "score": score,
            })
            if score > best_score:
                best_score = score
                best_w = (w_xgb, w_ranker)

        logger.info(f"  Best weights (valid): XGB={best_w[0]:.2f}, Ranker={best_w[1]:.2f}")
        for wr in weight_results:
            logger.info(f"    XGB={wr['w_xgb']:.2f} Ranker={wr['w_ranker']:.2f} "
                        f"RankIC={wr['rank_ic']:+.4f} Spread={wr['spread']*100:+.3f}%")

        w_xgb, w_ranker = best_w
    else:
        weights = [float(w) for w in args.weights.split(",")]
        w_xgb, w_ranker = weights[0], weights[1]

    # Final ensemble on test
    logger.info(f"\nEnsemble weights: XGB={w_xgb:.2f}, Ranker={w_ranker:.2f}")
    pred_ensemble = ensemble_predictions(pred_xgb, pred_ranker, test_idx, w_xgb, w_ranker)
    m_ens = evaluate(pred_ensemble, y_test, test_idx)

    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS (test: {test_start}~{test_end})")
    logger.info(f"{'='*70}")
    logger.info(f"{'Model':<20} {'IC':>8} {'ICIR':>8} {'RankIC':>8} {'Spread':>10} {'RIC>0':>8}")
    logger.info("-" * 65)
    logger.info(f"{'XGB':<20} {m_xgb['ic_mean']:+.4f}  {m_xgb['icir']:+.3f}  "
                f"{m_xgb['rank_ic_mean']:+.4f}  {m_xgb['top20_spread']*100:+.3f}%  "
                f"{m_xgb['rank_ic_pos']:.0%}")
    logger.info(f"{'Ranker':<20} {m_ranker['ic_mean']:+.4f}  {m_ranker['icir']:+.3f}  "
                f"{m_ranker['rank_ic_mean']:+.4f}  {m_ranker['top20_spread']*100:+.3f}%  "
                f"{m_ranker['rank_ic_pos']:.0%}")
    logger.info(f"{'Ensemble':<20} {m_ens['ic_mean']:+.4f}  {m_ens['icir']:+.3f}  "
                f"{m_ens['rank_ic_mean']:+.4f}  {m_ens['top20_spread']*100:+.3f}%  "
                f"{m_ens['rank_ic_pos']:.0%}")

    # Check if ensemble beats both
    beats_xgb_ric = m_ens["rank_ic_mean"] > m_xgb["rank_ic_mean"]
    beats_ranker_ric = m_ens["rank_ic_mean"] > m_ranker["rank_ic_mean"]
    beats_xgb_spr = m_ens["top20_spread"] > m_xgb["top20_spread"]
    beats_ranker_spr = m_ens["top20_spread"] > m_ranker["top20_spread"]

    logger.info(f"\n  Ensemble vs XGB:    RankIC {'↑' if beats_xgb_ric else '↓'}  "
                f"Spread {'↑' if beats_xgb_spr else '↓'}")
    logger.info(f"  Ensemble vs Ranker: RankIC {'↑' if beats_ranker_ric else '↓'}  "
                f"Spread {'↑' if beats_ranker_spr else '↓'}")

    # Save
    results = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "features": n_feat,
        "weights": {"xgb": w_xgb, "ranker": w_ranker},
        "test_period": f"{test_start}~{test_end}",
        "xgb": m_xgb,
        "ranker": m_ranker,
        "ensemble": m_ens,
    }
    out_path = DATA_DIR / "ensemble_rank_results.json"
    with open(str(out_path), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved: {out_path}")

    # Push
    try:
        from push.wechat import WeChatPusher
        lines = [
            f"🔀 Ensemble: XGB({w_xgb:.0%}) + Ranker({w_ranker:.0%})",
            "",
            f"XGB:      RankIC={m_xgb['rank_ic_mean']:+.4f}  Spread={m_xgb['top20_spread']*100:+.3f}%",
            f"Ranker:   RankIC={m_ranker['rank_ic_mean']:+.4f}  Spread={m_ranker['top20_spread']*100:+.3f}%",
            f"Ensemble: RankIC={m_ens['rank_ic_mean']:+.4f}  Spread={m_ens['top20_spread']*100:+.3f}%",
        ]
        WeChatPusher().send("\n".join(lines), title="Ensemble 结果")
        logger.info("✅ 推送成功")
    except Exception as e:
        logger.warning(f"Push failed: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
