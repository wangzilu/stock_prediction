import pandas as pd
import numpy as np
from datetime import datetime

from qlib.data.dataset import DatasetH
from qlib.contrib.model.gbdt import LGBModel
from qlib.utils import init_instance_by_config

from config.qlib_runtime import init_qlib
from config.settings import (
    QLIB_PROVIDER_URI,
    PREDICTION_HORIZON_DAYS,
    TOP_K_STOCKS,
    LGB_INFERENCE_UNIVERSE,
)
from config.watchlist import WATCHLIST_STOCK


class ShortTermModel:
    """LightGBM-based short-term stock prediction model using Qlib.

    Predicts 5-day forward returns using Alpha158 factors.
    """

    def __init__(self):
        self._model = None
        self._dataset = None
        self._initialized = False
        self.latest_prediction_date = ""

    @staticmethod
    def _normalize_code(code) -> str:
        text = str(code).upper()
        if "." in text:
            left, right = text.split(".", 1)
            if right in ("SH", "SS"):
                return f"SH{left.zfill(6)}"
            if right == "SZ":
                return f"SZ{left.zfill(6)}"
        return text

    @staticmethod
    def _score_column(predictions: pd.DataFrame):
        if "score" in predictions.columns:
            return "score"
        if len(predictions.columns) == 1:
            return predictions.columns[0]
        numeric_cols = [
            col for col in predictions.columns
            if pd.api.types.is_numeric_dtype(predictions[col])
        ]
        if len(numeric_cols) == 1:
            return numeric_cols[0]
        raise RuntimeError("Qlib prediction output does not contain a score column")

    @staticmethod
    def _datetime_level(index: pd.MultiIndex) -> int:
        for i, name in enumerate(index.names):
            if name and str(name).lower() in ("datetime", "date"):
                return i
        for i in range(index.nlevels):
            values = index.get_level_values(i)
            if pd.api.types.is_datetime64_any_dtype(values):
                return i
        return 0

    @staticmethod
    def _instrument_level(index: pd.MultiIndex, date_level: int) -> int:
        for i, name in enumerate(index.names):
            if name and str(name).lower() in ("instrument", "code", "symbol"):
                return i
        return 1 if date_level == 0 and index.nlevels > 1 else 0

    @classmethod
    def _prediction_frame(cls, predictions) -> pd.DataFrame:
        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame("score")
        if not isinstance(predictions, pd.DataFrame):
            raise RuntimeError(
                f"Qlib prediction output must be a DataFrame or Series, got {type(predictions).__name__}"
            )
        if predictions.empty:
            raise RuntimeError("Qlib model produced empty predictions")

        score_col = cls._score_column(predictions)
        scores = pd.to_numeric(predictions[score_col], errors="coerce").astype("float64")
        normalized = predictions.copy()
        normalized["score"] = scores
        return normalized[["score"]]

    @classmethod
    def _normalize_prediction_frame(cls, predictions) -> pd.DataFrame:
        normalized = cls._prediction_frame(predictions)
        scores = normalized["score"]
        finite_mask = np.isfinite(scores.to_numpy())

        normalized = normalized.loc[finite_mask].copy()
        if normalized.empty:
            raise RuntimeError("Qlib model produced no finite predictions")
        return normalized[["score"]]

    @classmethod
    def _latest_finite_predictions(cls, predictions) -> pd.DataFrame:
        normalized = cls._prediction_frame(predictions)
        if not isinstance(normalized.index, pd.MultiIndex):
            finite_mask = np.isfinite(normalized["score"].to_numpy())
            latest = normalized.loc[finite_mask].copy()
            if latest.empty:
                raise RuntimeError("Qlib model produced no finite predictions")
            latest.index = [cls._normalize_code(code) for code in latest.index]
            return latest

        index = normalized.index
        date_level = cls._datetime_level(index)
        instrument_level = cls._instrument_level(index, date_level)
        latest_date = index.get_level_values(date_level).max()

        finite_mask = np.isfinite(normalized["score"].to_numpy())
        finite = normalized.loc[finite_mask].copy()
        if finite.empty:
            raise RuntimeError("Qlib model produced no finite predictions")

        finite["_datetime"] = pd.to_datetime(
            finite.index.get_level_values(date_level),
            errors="coerce",
        )
        finite["_instrument"] = [
            cls._normalize_code(code)
            for code in finite.index.get_level_values(instrument_level)
        ]
        finite = finite.dropna(subset=["_datetime"])
        if finite.empty:
            raise RuntimeError("Qlib model produced no finite dated predictions")

        finite = finite.sort_values(["_instrument", "_datetime"])
        latest = finite.groupby("_instrument", sort=False).tail(1).copy()
        latest.index = latest["_instrument"]
        latest = latest[~latest.index.duplicated(keep="last")]
        selected_dates = latest["_datetime"].copy()

        # cx code review 2026-06-04 P1 #7: do NOT mix per-stock stale
        # predictions into the production output. Previously the
        # groupby+tail(1) silently used an OLDER day's prediction
        # for any stock whose latest date had no finite prediction,
        # contaminating the production score set with stale signal.
        # Drop stale rows here; record the count in attrs so the cron
        # health gate can surface the gap explicitly.
        latest_ts = pd.Timestamp(latest_date)
        stale_mask = selected_dates < latest_ts
        n_stale = int(stale_mask.sum())
        if n_stale:
            latest = latest.loc[~stale_mask]
            selected_dates = selected_dates.loc[~stale_mask]

        latest = latest[["score"]]
        if latest.empty:
            raise RuntimeError("Qlib model produced no finite latest-date predictions")
        try:
            latest.attrs["latest_date"] = pd.Timestamp(latest_date).strftime("%Y-%m-%d")
        except Exception:
            latest.attrs["latest_date"] = str(latest_date)
        try:
            latest.attrs["stale_prediction_count"] = int(
                (selected_dates < pd.Timestamp(latest_date)).sum()
            )
        except Exception:
            latest.attrs["stale_prediction_count"] = 0
        return latest

    def initialize(self):
        """Initialize Qlib and prepare model."""
        if self._initialized:
            return
        init_qlib(QLIB_PROVIDER_URI)
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
                "instruments": LGB_INFERENCE_UNIVERSE,
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

        latest_preds = self._latest_finite_predictions(
            self._model.predict(dataset=self._dataset)
        )

        watchlist_codes = {self._normalize_code(code) for code, _ in WATCHLIST_STOCK}
        watchlist_map = {code: name for code, name in WATCHLIST_STOCK}

        results = []
        for raw_code in latest_preds.index:
            code = self._normalize_code(raw_code)
            if code in watchlist_codes:
                results.append({
                    "code": code,
                    "name": watchlist_map.get(code, ""),
                    "score": float(latest_preds.loc[raw_code, "score"]),
                })

        df = pd.DataFrame(results)
        if df.empty:
            return df

        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df.head(TOP_K_STOCKS)

    @classmethod
    def load_from_pickle(cls, model_path: str = None, dataset_path: str = None):
        """Load pre-trained model and rebuild dataset for inference.

        The dataset is rebuilt fresh from Qlib to avoid pickle
        incompatibility with Alpha158 handler across Qlib versions.
        """
        import pickle
        from datetime import datetime, timedelta
        from config.settings import DATA_DIR

        model_path = model_path or str(DATA_DIR / "lgb_model.pkl")

        instance = cls()
        instance.initialize()

        with open(model_path, "rb") as f:
            instance._model = pickle.load(f)

        # Rebuild dataset fresh — avoids Alpha158 pickle issues
        today = datetime.now()
        test_start = (today - timedelta(days=29)).strftime("%Y-%m-%d")
        test_end = today.strftime("%Y-%m-%d")

        handler_config = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": test_start,
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
                    "test": (test_start, test_end),
                },
            },
        }
        instance._dataset = init_instance_by_config(dataset_config)

        # 2026-06-03 P0 fix (cx code review of all-negative incident):
        # The booster was trained INCLUDING supplementary features
        # (capital flow / valuation / quality / ST / cross-market / etc.
        # — ~84 columns) via train_lgb.py's manual injection into the
        # handler. Without this same injection at inference, the
        # rebuilt dataset only contains Alpha158 (158 cols) and the
        # booster's predict() walks default-direction branches on
        # the missing 84 cols → predictions cluster around a single
        # leaf and on bearish days every stock ends up negative.
        try:
            from models.feature_merger import FeatureMerger
            handler = instance._dataset.handler
            merger = FeatureMerger()
            n_supp = merger.inject_supplementary_into_handler(
                handler, preprocess=False,
            )
            print(f"[short_term] injected {n_supp} supplementary cols at inference")
        except Exception as e:  # noqa: BLE001
            print(f"[short_term] supplementary injection failed (non-fatal): {e}")

        # Sanity gate: booster feature count must match the prepared
        # feature matrix width, or predictions are silent garbage.
        try:
            from qlib.data.dataset.handler import DataHandlerLP as _DK
            _xtest = instance._dataset.prepare(
                "test", col_set="feature", data_key=_DK.DK_I,
            )
            _booster = getattr(instance._model, "model", None)
            if _booster is not None and hasattr(_booster, "num_features"):
                booster_n = int(_booster.num_features())
                dataset_n = int(_xtest.shape[1])
                if booster_n != dataset_n:
                    raise RuntimeError(
                        f"[short_term] FATAL: trained model expects "
                        f"{booster_n} features but inference dataset has "
                        f"{dataset_n}. Predictions would be silent "
                        f"default-leaf garbage (cx review 2026-06-03 "
                        f"incident). Retrain or wire supplementary "
                        f"injection."
                    )
        except RuntimeError:
            raise
        except Exception as _e:  # noqa: BLE001
            print(f"[short_term] dim sanity gate skipped: {_e}")

        return instance

    def predict_batch(self) -> dict:
        """Generate predictions for ALL stocks in the dataset.

        Returns:
            Dict mapping qlib_code (e.g. 'SH600519') to predicted score (float).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_from_pickle() first.")

        latest_preds = self._latest_finite_predictions(
            self._model.predict(dataset=self._dataset)
        )
        self.latest_prediction_date = str(latest_preds.attrs.get("latest_date", ""))
        result = {
            self._normalize_code(code): float(row["score"])
            for code, row in latest_preds.iterrows()
        }
        if not result:
            raise RuntimeError("Qlib model produced no finite latest-date predictions")
        return result
