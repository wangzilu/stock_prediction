"""Train LightGBM model using Qlib Alpha158 factors.

Usage: python scripts/train_lgb.py
"""
import os
import sys
import pickle

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "storage")
QLIB_DATA = os.path.join(DATA_DIR, "qlib_data", "cn_data")
MODEL_PATH = os.path.join(DATA_DIR, "lgb_model.pkl")
DATASET_PATH = os.path.join(DATA_DIR, "lgb_dataset.pkl")


def main():
    print("Initializing Qlib...")
    qlib.init(provider_uri=QLIB_DATA, region=REG_CN)

    # Note: Qlib Yahoo data ends ~2020-09-25. Adjust date ranges accordingly.
    # For production, update data with AKShare/baostock daily.
    handler_config = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": "2010-01-01",
            "end_time": "2020-09-25",
            "instruments": "csi300",
        },
    }
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {
                "train": ("2010-01-01", "2019-06-30"),
                "valid": ("2019-07-01", "2020-03-31"),
                "test": ("2020-04-01", "2020-09-25"),
            },
        },
    }

    print("Loading dataset (Alpha158 x csi300 x 7 years)...")
    dataset = init_instance_by_config(dataset_config)
    print("Dataset ready.")

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

    print("Training LightGBM...")
    model = init_instance_by_config(model_config)
    model.fit(dataset)
    print("Training complete!")

    # Save model + dataset
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(DATASET_PATH, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Model saved to {MODEL_PATH}")

    # Quick evaluation
    pred = model.predict(dataset)
    print(f"\nPredictions shape: {pred.shape}")
    print(f"Last 5 predictions:")
    print(pred.tail(5))


if __name__ == "__main__":
    main()
