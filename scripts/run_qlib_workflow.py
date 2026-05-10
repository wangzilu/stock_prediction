"""Train XGB with Qlib Recorder for experiment tracking.

Every training run is recorded with:
- Parameters (model config, data window, universe)
- Predictions (SignalRecord)
- Metrics (IC, RankIC, Spread)
- Artifacts (model path)

Usage:
    python scripts/run_qlib_workflow.py
    python scripts/run_qlib_workflow.py --experiment my_experiment
"""
import argparse
import json
import logging
import os
import pickle
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DATA = str(DATA_DIR / "qlib_data" / "cn_data")
LABEL_EXPR = f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"


def main():
    parser = argparse.ArgumentParser(description="Train with Qlib Recorder")
    parser.add_argument("--experiment", default="xgb_alpha158_all")
    parser.add_argument("--universe", default="all")
    parser.add_argument("--model-type", default="xgb", choices=["xgb", "lgb", "catboost"])
    args = parser.parse_args()

    init_qlib(QLIB_DATA)

    from qlib.workflow import R
    from qlib.utils import init_instance_by_config
    from qlib.contrib.eva.alpha import calc_ic

    today = datetime.now()
    train_start = (today - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=89)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")

    # Model configs
    model_configs = {
        "xgb": {
            "class": "XGBModel", "module_path": "qlib.contrib.model.xgboost",
            "kwargs": {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.05,
                       "subsample": 0.8789, "colsample_bytree": 0.8879,
                       "reg_alpha": 205.7, "reg_lambda": 580.98, "n_jobs": 4},
        },
        "lgb": {
            "class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {"loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.05,
                       "subsample": 0.8789, "lambda_l1": 205.7, "lambda_l2": 580.98,
                       "max_depth": 8, "num_leaves": 210, "num_threads": 4},
        },
        "catboost": {
            "class": "CatBoostModel", "module_path": "qlib.contrib.model.catboost_model",
            "kwargs": {"loss_function": "RMSE", "iterations": 500, "depth": 8, "learning_rate": 0.05},
        },
    }

    model_config = model_configs[args.model_type]
    recorder_name = f"{args.model_type}_{today.strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"Experiment: {args.experiment}, Recorder: {recorder_name}")

    # Start Qlib Recorder
    with R.start(experiment_name=args.experiment, recorder_name=recorder_name):
        # Log parameters
        R.log_params(**{
            "model_type": args.model_type,
            "universe": args.universe,
            "train_start": train_start,
            "train_end": train_end,
            "valid_start": valid_start,
            "valid_end": valid_end,
            "test_start": test_start,
            "test_end": test_end,
            "label": LABEL_EXPR,
        })
        R.log_params(**model_config.get("kwargs", {}))

        # Build dataset
        logger.info("Loading dataset...")
        t0 = time.time()
        handler_config = {
            "class": "Alpha158", "module_path": "qlib.contrib.data.handler",
            "kwargs": {"start_time": train_start, "end_time": test_end,
                       "instruments": args.universe, "label": [LABEL_EXPR]},
        }
        dataset_config = {
            "class": "DatasetH", "module_path": "qlib.data.dataset",
            "kwargs": {"handler": handler_config, "segments": {
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (test_start, test_end),
            }},
        }
        dataset = init_instance_by_config(dataset_config)
        logger.info(f"Dataset loaded: {time.time()-t0:.0f}s")

        # Train
        logger.info(f"Training {args.model_type}...")
        t1 = time.time()
        model = init_instance_by_config(model_config)
        model.fit(dataset)
        train_time = time.time() - t1
        logger.info(f"Trained: {train_time:.0f}s")
        R.log_metrics(train_time_s=round(train_time, 1))

        # Predict
        pred = model.predict(dataset=dataset)
        if isinstance(pred, pd.Series):
            pred = pred.to_frame("score")

        # Save prediction as artifact
        R.save_objects(**{"pred.pkl": pred, "model.pkl": model})

        # Evaluate
        label = dataset.prepare("test", col_set="label")
        if isinstance(label, pd.DataFrame):
            label = label.iloc[:, 0]

        common = pred.index.intersection(label.index)
        pred_s = pred.loc[common, "score"] if "score" in pred.columns else pred.loc[common].iloc[:, 0]
        label_s = label.loc[common]

        mask = pred_s.notna() & label_s.notna() & np.isfinite(pred_s) & np.isfinite(label_s)
        p, l = pred_s[mask], label_s[mask]

        if len(p) > 100:
            ic, ric = calc_ic(p, l)
            ic_mean = float(ic.mean())
            ric_mean = float(ric.mean())
            icir = ic_mean / (float(ic.std()) + 1e-8)

            df = pd.DataFrame({"pred": p, "label": l})
            spreads = []
            for d, g in df.groupby(level=0):
                if len(g) >= 40:
                    s = g.sort_values("pred", ascending=False)
                    spreads.append(s.head(20)["label"].mean() - s.tail(20)["label"].mean())

            spread = float(np.mean(spreads)) if spreads else 0

            R.log_metrics(
                ic_mean=round(ic_mean, 6),
                icir=round(icir, 4),
                rank_ic_mean=round(ric_mean, 6),
                top20_spread=round(spread, 6),
                n_predictions=len(p),
            )

            logger.info(f"IC={ic_mean:+.4f} ICIR={icir:.3f} RankIC={ric_mean:+.4f} Spread={spread*100:+.3f}%")
        else:
            logger.warning(f"Too few predictions: {len(p)}")

        logger.info(f"Recorder: {recorder_name} saved to experiment {args.experiment}")


if __name__ == "__main__":
    main()
