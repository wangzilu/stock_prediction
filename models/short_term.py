import pandas as pd
from datetime import datetime

import qlib
from qlib.data.dataset import DatasetH
from qlib.contrib.model.gbdt import LGBModel
from qlib.utils import init_instance_by_config

from config.settings import QLIB_PROVIDER_URI, PREDICTION_HORIZON_DAYS, TOP_K_STOCKS
from config.watchlist import WATCHLIST


class ShortTermModel:
    """LightGBM-based short-term stock prediction model using Qlib.

    Predicts 5-day forward returns using Alpha158 factors.
    """

    def __init__(self):
        self._model = None
        self._dataset = None
        self._initialized = False

    def initialize(self):
        """Initialize Qlib and prepare model."""
        if self._initialized:
            return
        qlib.init(provider_uri=QLIB_PROVIDER_URI, region_type="cn")
        self._initialized = True

    def train(
        self,
        train_start: str = "2020-01-01",
        train_end: str = "2025-12-31",
        valid_start: str = "2026-01-01",
        valid_end: str = "2026-03-31",
    ):
        """Train the LightGBM model."""
        self.initialize()

        handler_config = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": train_start,
                "end_time": valid_end,
                "instruments": "csi300",
                "label": [
                    f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"
                ],
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
                },
            },
        }

        self._dataset = init_instance_by_config(dataset_config)

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

        self._model = init_instance_by_config(model_config)
        self._model.fit(self._dataset)

    def predict(self, date: str = None) -> pd.DataFrame:
        """Generate predictions for the watchlist stocks.

        Args:
            date: Prediction date (defaults to today)

        Returns:
            DataFrame with columns [code, name, score, rank]
            sorted by score descending.
        """
        self.initialize()

        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        predictions = self._model.predict(dataset=self._dataset)

        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame("score")

        latest_date = predictions.index.get_level_values(0).max()
        latest_preds = predictions.loc[latest_date]

        watchlist_codes = {code for code, _ in WATCHLIST}
        watchlist_map = {code: name for code, name in WATCHLIST}

        results = []
        for code in latest_preds.index:
            if code in watchlist_codes:
                results.append({
                    "code": code,
                    "name": watchlist_map.get(code, ""),
                    "score": float(latest_preds.loc[code, "score"]),
                })

        df = pd.DataFrame(results)
        if df.empty:
            return df

        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df.head(TOP_K_STOCKS)
