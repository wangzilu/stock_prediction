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


# Re-export so existing callers keep working.
# Canonical definition lives in models.feature_contract because
# cx round 2 P1-3 makes the artifact a real gate (not just a report).
from models.feature_contract import (
    FeatureContractViolation,
    load_contract,
    verify_inference_dataset,
)


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

    def train(self, *args, **kwargs):
        """LEGACY — DO NOT USE (cx round 8 P1-4).

        The previous body trained an Alpha158-only (158-dim) LGBModel
        with NO supplementary feature injection, NO PRODUCTION_
        SUPPLEMENTARY_GROUPS gate, NO feature contract write, and NO
        tradable mask. Any model saved by this path would be 158-dim
        while inference expects 242 — the exact 6-3 22:00 incident
        mechanism.

        The production training path is ``scripts/train_lgb.py``.
        That script enforces the supplementary contract, writes the
        feature contract artifact, gates on prediction health, and
        atomic-saves the model only after the contract write succeeds.

        Raises:
            RuntimeError: always. To re-enable for an explicit research
                purpose (NOT production), set environment variable
                ``LEGACY_SHORT_TERM_TRAIN_OVERRIDE=acknowledge_158_dim``
                — the resulting model still cannot be loaded by
                ``load_from_pickle`` because the contract gate will
                refuse a 158-dim artifact.
        """
        import os as _os
        if _os.environ.get("LEGACY_SHORT_TERM_TRAIN_OVERRIDE") != "acknowledge_158_dim":
            raise RuntimeError(
                "ShortTermModel.train() is DISABLED (cx round 8 P1-4 — "
                "2026-06-04). It would train an Alpha158-only 158-dim "
                "model that immediately fails the production 242-dim "
                "feature contract. Use scripts/train_lgb.py instead."
            )
        # If you really want the legacy Alpha158-only path (e.g. for a
        # diagnostic 158-vs-242 comparison), the code lives in the git
        # history of this file pre-2026-06-04. We do not reproduce it
        # in the live tree because copy-paste recipients would forget
        # to set the override.
        raise RuntimeError(
            "Legacy ShortTermModel.train body was removed; consult "
            "git history if you really need a 158-only training path."
        )

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
    def load_from_pickle(cls, model_path: str = None):
        """Load pre-trained model and rebuild dataset for inference.

        2026-06-04 cx round 8 P2-5: removed the ``dataset_path``
        parameter. It was advertised in the signature but never
        consulted — the method ALWAYS rebuilt the inference dataset
        from Qlib over the last 29 days. Callers who thought they
        could load a configured dataset artifact got a silent
        substitution. Removing the param means a caller who needs
        to evaluate against a specific historical window should use
        ``models.production_inference.load_production_model`` (which
        takes explicit (test_start, test_end) ranges).

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
        #
        # 2026-06-04 cx round 2 P0-2 follow-up: previously this block
        # was wrapped in try/except Exception with a "non-fatal" print,
        # i.e. inference would proceed with whatever partial supp was
        # injected (or none) and the next dim check would catch — or
        # not catch — the resulting skew. The "non-fatal" framing was
        # also factually wrong: missing supp on a 242-trained model is
        # the precise mechanism of the 22:00 0-rec incident. Failure
        # MUST propagate so the outer scheduler hard-fails.
        from models.feature_merger import FeatureMerger
        from config.production_features import (
            PRODUCTION_SUPPLEMENTARY_GROUPS,
        )
        handler = instance._dataset.handler
        merger = FeatureMerger()
        n_supp = merger.inject_supplementary_into_handler(
            handler, preprocess=False,
            groups=PRODUCTION_SUPPLEMENTARY_GROUPS,
        )
        print(f"[short_term] injected {n_supp} supplementary cols at inference")
        if n_supp == 0:
            raise FeatureContractViolation(
                "[short_term] inject_supplementary_into_handler returned 0 "
                "columns at inference. The 242-dim production champion "
                "would silently degrade to 158-dim default-leaf "
                "predictions. Refusing to serve."
            )

        # Sanity gate: booster feature count must match the prepared
        # feature matrix width, or predictions are silent garbage.
        # The outer try is now ONLY around qlib's dataset.prepare —
        # if that itself blows up we want a clear contract violation,
        # not a silent skip.
        from qlib.data.dataset.handler import DataHandlerLP as _DK
        _xtest = instance._dataset.prepare(
            "test", col_set="feature", data_key=_DK.DK_I,
        )
        _booster = getattr(instance._model, "model", None)
        if _booster is None or not hasattr(_booster, "num_features"):
            raise FeatureContractViolation(
                "[short_term] loaded model has no inspectable booster "
                "(num_features missing). Cannot verify the 242-dim "
                "contract; refusing to serve."
            )
        booster_n = int(_booster.num_features())
        dataset_n = int(_xtest.shape[1])
        if booster_n != dataset_n:
            raise FeatureContractViolation(
                f"[short_term] FATAL: trained model expects "
                f"{booster_n} features but inference dataset has "
                f"{dataset_n}. Predictions would be silent "
                f"default-leaf garbage (cx review 2026-06-03 "
                f"incident). Retrain or wire supplementary "
                f"injection."
            )

        # cx round 2 P1-3 / P1-4 + round 8 P1-1: real-name gate
        # against the production contract artifact. The artifact is
        # MANDATORY at inference time — there is no "bootstrap" window
        # where a count-only gate is acceptable for production
        # recommendations. (Pre-fix this section accepted missing
        # contract with a warning, which is exactly the silent-fallback
        # pattern we've been closing across the codebase.)
        # If the artifact really is missing (fresh deploy), the operator
        # must run train_lgb.py once to populate it — that's the only
        # path that produces a contract aligned with the trained model.
        # ``scripts/export_feature_contract.py`` is a diagnostic tool
        # and does NOT count for this gate (cx round 8 P1-2).
        from pathlib import Path as _P
        _data_dir = _P(__file__).resolve().parents[1] / "data" / "storage"
        contract = load_contract(_data_dir)
        if contract is None:
            raise FeatureContractViolation(
                "[short_term] production feature contract artifact "
                "MISSING at data/storage/production_feature_contract.json. "
                "Inference refuses to serve without it — a count-only "
                "gate cannot catch loader reorder / silent column swap "
                "(the exact failure mode this artifact exists to pin). "
                "Re-run scripts/train_lgb.py to produce a contract "
                "aligned with the current lgb_model.pkl."
            )
        actual_names = [
            col[1] if isinstance(col, tuple) else str(col)
            for col in _xtest.columns.tolist()
        ]
        verify_inference_dataset(contract, actual_names)
        print(f"[short_term] feature contract verified "
              f"({len(actual_names)} cols, schema v"
              f"{contract.get('schema_version', 1)})")

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
