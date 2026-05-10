"""Train LightGBM model using Qlib Alpha158 factors.

Usage: python scripts/train_lgb.py
"""
import os
import sys
import pickle
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from qlib.utils import init_instance_by_config

from config.qlib_runtime import init_qlib
from config.settings import (
    LGB_INFERENCE_UNIVERSE,
    LGB_MIN_DATA_INSTRUMENTS,
    LGB_MIN_PREDICTIONS,
    QLIB_PROVIDER_URI,
)
from scripts.check_qlib_data_health import check_qlib_dir

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "storage")
QLIB_DATA = QLIB_PROVIDER_URI
MODEL_PATH = os.path.join(DATA_DIR, "lgb_model.pkl")
DATASET_PATH = os.path.join(DATA_DIR, "lgb_dataset.pkl")


def _prediction_score_series(predictions) -> pd.Series:
    if isinstance(predictions, pd.Series):
        series = predictions
    elif isinstance(predictions, pd.DataFrame):
        if "score" in predictions.columns:
            series = predictions["score"]
        elif len(predictions.columns) == 1:
            series = predictions.iloc[:, 0]
        else:
            numeric_cols = [
                col for col in predictions.columns
                if pd.api.types.is_numeric_dtype(predictions[col])
            ]
            if len(numeric_cols) != 1:
                raise RuntimeError("prediction output does not contain a single score column")
            series = predictions[numeric_cols[0]]
    else:
        raise RuntimeError(
            f"prediction output must be a Series or DataFrame, got {type(predictions).__name__}"
        )
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _datetime_level(index: pd.MultiIndex) -> int:
    for i, name in enumerate(index.names):
        if name and str(name).lower() in ("datetime", "date"):
            return i
    for i in range(index.nlevels):
        values = index.get_level_values(i)
        if pd.api.types.is_datetime64_any_dtype(values):
            return i
    return 0


def _instrument_level(index: pd.MultiIndex, date_level: int) -> int:
    for i, name in enumerate(index.names):
        if name and str(name).lower() in ("instrument", "code", "symbol"):
            return i
    return 1 if date_level == 0 and index.nlevels > 1 else 0


def _prediction_health(predictions, min_predictions: int) -> dict:
    scores = _prediction_score_series(predictions)
    values = scores.to_numpy()
    finite_mask = np.isfinite(values)
    finite_scores = scores.loc[finite_mask]

    latest_finite_count = len(finite_scores)
    latest_date = None
    stale_prediction_count = 0
    if isinstance(scores.index, pd.MultiIndex) and not finite_scores.empty:
        date_level = _datetime_level(scores.index)
        instrument_level = _instrument_level(scores.index, date_level)
        latest_date = scores.index.get_level_values(date_level).max()
        finite_frame = finite_scores.to_frame("score")
        finite_frame["_datetime"] = pd.to_datetime(
            finite_frame.index.get_level_values(date_level),
            errors="coerce",
        )
        finite_frame["_instrument"] = [
            str(code).upper()
            for code in finite_frame.index.get_level_values(instrument_level)
        ]
        finite_frame = finite_frame.dropna(subset=["_datetime"])
        latest_per_instrument = finite_frame.sort_values(
            ["_instrument", "_datetime"]
        ).groupby("_instrument", sort=False).tail(1)
        latest_finite_count = int(len(latest_per_instrument))
        stale_prediction_count = int(
            (latest_per_instrument["_datetime"] < pd.Timestamp(latest_date)).sum()
        )

    stats = {
        "prediction_count": int(len(scores)),
        "finite_prediction_count": int(finite_mask.sum()),
        "non_finite_prediction_count": int((~finite_mask).sum()),
        "latest_finite_prediction_count": latest_finite_count,
        "stale_prediction_count": stale_prediction_count,
        "latest_date": str(latest_date) if latest_date is not None else "",
        "min_predictions": min_predictions,
    }
    if stats["finite_prediction_count"] == 0:
        raise RuntimeError(f"model produced no finite predictions: {stats}")
    if latest_finite_count < min_predictions:
        raise RuntimeError(
            f"latest finite predictions {latest_finite_count} < required {min_predictions}: {stats}"
        )
    return stats


def _predict_test_segment(model, dataset):
    try:
        return model.predict(dataset, segment="test")
    except TypeError:
        return model.predict(dataset)


def _save_artifacts_atomically(model, dataset):
    os.makedirs(DATA_DIR, exist_ok=True)
    model_path = Path(MODEL_PATH)
    dataset_path = Path(DATASET_PATH)
    tmp_model_path = model_path.with_name(f"{model_path.name}.tmp")
    tmp_dataset_path = dataset_path.with_name(f"{dataset_path.name}.tmp")

    try:
        with tmp_model_path.open("wb") as f:
            pickle.dump(model, f)
        with tmp_dataset_path.open("wb") as f:
            pickle.dump(dataset, f)
        os.replace(tmp_model_path, model_path)
        os.replace(tmp_dataset_path, dataset_path)
    finally:
        for path in (tmp_model_path, tmp_dataset_path):
            if path.exists():
                path.unlink()


def main():
    print("Checking Qlib data health...")
    health = check_qlib_dir(
        Path(QLIB_DATA),
        universe=LGB_INFERENCE_UNIVERSE,
        min_instruments=LGB_MIN_DATA_INSTRUMENTS,
    )
    if not health.ok:
        print("Qlib data health check failed; refusing to train.")
        for error in health.errors:
            print(f"- {error}")
        return 1

    print("Initializing Qlib...")
    init_qlib(QLIB_DATA)

    # Dynamic date ranges
    today = datetime.now()
    train_start = (today - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=89)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")

    print(f"Train: {train_start} ~ {train_end}")
    print(f"Valid: {valid_start} ~ {valid_end}")
    print(f"Test:  {test_start} ~ {test_end}")

    handler_config = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": train_start,
            "end_time": test_end,
            "instruments": LGB_INFERENCE_UNIVERSE,
        },
    }
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (test_start, test_end),
            },
        },
    }

    print(f"Loading dataset (Alpha158 x {LGB_INFERENCE_UNIVERSE} x 7 years)...")
    dataset = init_instance_by_config(dataset_config)
    print("Dataset ready.")

    # Model selection: XGB (better IC) or LGB (fallback)
    model_type = os.environ.get("TRAIN_MODEL_TYPE", "xgb").lower()

    if model_type == "xgb":
        model_config = {
            "class": "XGBModel",
            "module_path": "qlib.contrib.model.xgboost",
            "kwargs": {
                "n_estimators": 500,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.8789,
                "colsample_bytree": 0.8879,
                "reg_alpha": 205.6999,
                "reg_lambda": 580.9768,
                "n_jobs": 4,
            },
        }
        print(f"Training XGBoost (IC=0.024 > LGB IC=0.008)...")
    else:
        model_config = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.05,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 4,
            },
        }
        print("Training LightGBM (fallback)...")
    model = init_instance_by_config(model_config)
    model.fit(dataset)
    print("Training complete!")

    # Validate before touching the production model artifact.
    pred = _predict_test_segment(model, dataset)
    try:
        stats = _prediction_health(pred, LGB_MIN_PREDICTIONS)
    except RuntimeError as exc:
        print(f"Prediction health failed; refusing to save model: {exc}")
        return 1
    print("Prediction health passed:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save model + dataset only after finite prediction validation passes.
    _save_artifacts_atomically(model, dataset)
    print(f"Model saved to {MODEL_PATH}")

    print(f"\nPredictions shape: {pred.shape}")
    print(f"Last 5 predictions:")
    print(pred.tail(5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
