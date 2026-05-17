"""Train 175-dim (174 base + holder_num) XGB and push Top20 recommendation.

Best model from v2 enhanced comparison:
  174+holder: ICIR=0.727, Spread=+3.334%, RankIC=+0.0381

Usage:
    python scripts/train_175_and_recommend.py
"""
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from models.feature_pipeline import prepare_features_174, train_xgb, XGB_PARAMS

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"
SEED = 42


def prepare_segment(dataset, seg, merger):
    """Prepare 175-dim features using shared pipeline."""
    return prepare_features_174(dataset, seg, merger, include_holder=True)


def main():
    import xgboost as xgb
    from qlib.utils import init_instance_by_config
    from qlib.contrib.eva.alpha import calc_ic

    init_qlib(QLIB_DATA)

    today = datetime.now()
    train_start = (today - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=89)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")

    logger.info("=== 175-dim (174+holder) Training & Recommendation ===")
    logger.info(f"Train: {train_start}~{train_end}, Test: {test_start}~{test_end}")

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

    # Prepare all segments
    segs = {}
    for seg in ["train", "valid", "test"]:
        logger.info(f"Preparing {seg}...")
        X, y = prepare_segment(dataset, seg, merger)
        Xn = X.values.astype(np.float32)
        yn = y.values.astype(np.float32)
        mask = np.isfinite(yn)
        segs[seg] = (Xn[mask], yn[mask], X.index[mask], list(X.columns))
        logger.info(f"  {seg}: {Xn[mask].shape}")

    X_train, y_train, _, fnames = segs["train"]
    X_valid, y_valid, _, _ = segs["valid"]
    X_test, y_test, test_idx, _ = segs["test"]
    n_feat = X_train.shape[1]
    logger.info(f"Features: {n_feat}")

    # Train XGB
    logger.info("Training XGB...")
    t0 = time.time()
    fn = fnames if len(fnames) == n_feat else None
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=fn)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, feature_names=fn)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=fn)

    params = {
        "max_depth": 8, "learning_rate": 0.05,
        "subsample": 0.8789, "colsample_bytree": 0.8879,
        "reg_alpha": 205.6999, "reg_lambda": 580.9768,
        "objective": "reg:squarederror", "nthread": 4,
        "verbosity": 0, "seed": SEED,
    }
    model = xgb.train(params, dtrain, num_boost_round=500,
                      evals=[(dvalid, "valid")], early_stopping_rounds=50,
                      verbose_eval=20)
    logger.info(f"Trained in {time.time()-t0:.1f}s")

    # Evaluate
    pred = model.predict(dtest)
    mask = np.isfinite(pred) & np.isfinite(y_test)
    ps = pd.Series(pred[mask], index=test_idx[mask])
    ls = pd.Series(y_test[mask], index=test_idx[mask])
    ic, ric = calc_ic(ps, ls)

    logger.info(f"IC={float(ic.mean()):+.4f}  ICIR={float(ic.mean())/(float(ic.std())+1e-8):+.3f}")
    logger.info(f"RankIC={float(ric.mean()):+.4f}  RIC>0={float((ric>0).mean()):.0%}")

    # Get latest-date predictions for recommendation
    latest_date = test_idx.get_level_values(0).max()
    latest_mask = test_idx.get_level_values(0) == latest_date
    latest_pred = pd.Series(pred[mask][latest_mask[mask]],
                            index=test_idx[mask][latest_mask[mask]])

    if latest_pred.empty:
        # Fallback: use last available date
        all_dates = sorted(test_idx.get_level_values(0).unique())
        for d in reversed(all_dates):
            m = test_idx.get_level_values(0) == d
            sub = pd.Series(pred[mask][m[mask]], index=test_idx[mask][m[mask]])
            if len(sub) > 100:
                latest_pred = sub
                latest_date = d
                break

    logger.info(f"Latest date: {str(latest_date)[:10]}, {len(latest_pred)} stocks")

    # Top 20 recommendation
    top20 = latest_pred.sort_values(ascending=False).head(20)
    bottom20 = latest_pred.sort_values(ascending=True).head(20)

    logger.info(f"\n{'='*60}")
    logger.info(f"TOP 20 推荐 ({str(latest_date)[:10]})")
    logger.info(f"{'='*60}")
    for i, (idx, score) in enumerate(top20.items(), 1):
        code = str(idx[1]) if isinstance(idx, tuple) else str(idx)
        logger.info(f"  {i:2d}. {code}  score={score:+.6f}")

    # Save model
    model_path = DATA_DIR / "xgb_175_holder_model.json"
    model.save_model(str(model_path))
    logger.info(f"Model saved: {model_path}")

    # Write prediction cache (compatible with production pipeline)
    from models.lgb_cache import write_prediction_cache
    pred_map = {}
    for idx, score in latest_pred.items():
        code = str(idx[1]) if isinstance(idx, tuple) else str(idx)
        pred_map[code.upper()] = float(score)

    write_prediction_cache(
        pred_map,
        latest_date=str(latest_date)[:10],
        model_path=str(model_path),
        source="xgb_175_holder",
    )
    logger.info(f"Prediction cache updated: {len(pred_map)} stocks")

    # Push recommendation via WeChat
    try:
        from push.wechat import WeChatPusher
        pusher = WeChatPusher()

        lines = [
            f"🏆 175维最优模型荐股 ({str(latest_date)[:10]})",
            f"模型: XGB 174+holder (ICIR=0.727, Spread=+3.3%)",
            "",
            "📈 Top 20 看多:",
            "-" * 40,
        ]
        for i, (idx, score) in enumerate(top20.items(), 1):
            code = str(idx[1]) if isinstance(idx, tuple) else str(idx)
            lines.append(f"  {i:2d}. {code}  ({score:+.4f})")

        lines.extend(["", "📉 Bottom 5 看空:"])
        for i, (idx, score) in enumerate(bottom20.head(5).items(), 1):
            code = str(idx[1]) if isinstance(idx, tuple) else str(idx)
            lines.append(f"  {i:2d}. {code}  ({score:+.4f})")

        lines.extend([
            "",
            f"覆盖: {len(latest_pred)} 只",
            f"IC={float(ic.mean()):+.4f} RankIC={float(ric.mean()):+.4f}",
            "",
            "⚠️ 实验模型，仅供参考",
        ])

        msg = "\n".join(lines)
        print(msg)
        if pusher.send(msg, title="175维最优模型荐股"):
            logger.info("✅ 推送成功")
        else:
            logger.info("❌ 推送失败")
    except Exception as e:
        logger.warning(f"Push failed: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
