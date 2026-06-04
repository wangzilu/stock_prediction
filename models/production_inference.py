"""Production-safe LGB model loader for monitoring / evaluation scripts.

Why this exists (cx round 8 P1-3, 2026-06-04):
    ``scripts/evaluate_lgb_test.py``,
    ``scripts/run_brinson_attribution.py``,
    ``scripts/attribution.py`` and
    ``scripts/backtest_qlib_signal.py`` all loaded
    ``data/storage/lgb_model.pkl`` with raw ``pickle.load`` and then
    fed the model a bare Alpha158 dataset. The production champion
    is 242-dim (Alpha158 + 84 supplementary cols), so evaluating
    it against a 158-dim dataset walked XGB default-direction
    branches and produced silent-garbage IC / spread / Brinson
    numbers that misled "is 242 OK?" decisions — the exact failure
    mode the 6-3 22:00 incident showed in production.

This module provides ``load_production_model(...)`` which mirrors
``ShortTermModel.load_from_pickle`` but with a configurable
date range, so evaluation scripts can use ARBITRARY test windows
while still enforcing:

* Supplementary feature injection from
  ``PRODUCTION_SUPPLEMENTARY_GROUPS``
* booster.num_features vs dataset.shape[1] dim sanity gate
* ``production_feature_contract.json`` NAME+ORDER verify
  (raises ``FeatureContractViolation`` if drift)

Any monitoring script that uses raw ``pickle.load`` and a bare
Alpha158 dataset is producing numbers that do not represent the
production model's actual behavior — switch it to this loader.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from config.qlib_runtime import init_qlib
from config.settings import (
    LGB_INFERENCE_UNIVERSE,
    LGB_MODEL_PATH,
    QLIB_PROVIDER_URI,
)
from models.feature_contract import (
    FeatureContractViolation,
    load_contract,
    verify_inference_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "storage"


def load_production_model(
    test_start: str,
    test_end: str,
    *,
    model_path: str | Path = LGB_MODEL_PATH,
    instruments: str = LGB_INFERENCE_UNIVERSE,
    label_expr: str | None = None,
):
    """Load the production LGB model + a 242-dim aligned dataset.

    Args:
        test_start, test_end: ISO date strings bounding the test
            segment. Used to build a ``DatasetH(segments={"test":
            (test_start, test_end)})`` configuration.
        model_path: path to ``lgb_model.pkl``. Defaults to the
            production champion.
        instruments: Qlib instrument set name. Defaults to the
            production inference universe.
        label_expr: optional Qlib label expression. None uses the
            default 5-day forward-return expression.

    Returns:
        Tuple ``(model, dataset)`` where:
          * ``model`` is the unpickled XGB/LGB model object
          * ``dataset`` is a Qlib ``DatasetH`` with the production
            242-dim feature shape (Alpha158 + PRODUCTION_SUPPLEMENTARY_GROUPS)
            on the test segment

    Raises:
        FeatureContractViolation: when the trained model and the
            assembled inference dataset disagree on column count,
            supp column names, or the production_feature_contract.json
            artifact is missing.
    """
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset.handler import DataHandlerLP

    init_qlib(QLIB_PROVIDER_URI)

    if label_expr is None:
        from config.settings import PREDICTION_HORIZON_DAYS
        label_expr = (
            f"Ref($close, -{PREDICTION_HORIZON_DAYS}) / Ref($close, -1) - 1"
        )

    handler_config = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": test_start,
            "end_time": test_end,
            "instruments": instruments,
            "label": [label_expr],
        },
    }
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {"test": (test_start, test_end)},
        },
    }
    dataset = init_instance_by_config(dataset_config)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Same supp injection as ShortTermModel.load_from_pickle.
    from models.feature_merger import FeatureMerger
    from config.production_features import PRODUCTION_SUPPLEMENTARY_GROUPS
    merger = FeatureMerger()
    n_supp = merger.inject_supplementary_into_handler(
        dataset.handler, preprocess=False,
        groups=PRODUCTION_SUPPLEMENTARY_GROUPS,
    )
    if n_supp == 0:
        raise FeatureContractViolation(
            "load_production_model: inject_supplementary_into_handler "
            "returned 0 supp columns; refusing to evaluate against the "
            "242-dim production model."
        )

    # Sanity gate identical to ShortTermModel.load_from_pickle.
    _xtest = dataset.prepare("test", col_set="feature",
                              data_key=DataHandlerLP.DK_I)
    _booster = getattr(model, "model", None)
    if _booster is None or not hasattr(_booster, "num_features"):
        raise FeatureContractViolation(
            "load_production_model: loaded model has no inspectable "
            "booster.num_features — cannot verify the 242-dim contract."
        )
    booster_n = int(_booster.num_features())
    dataset_n = int(_xtest.shape[1])
    if booster_n != dataset_n:
        raise FeatureContractViolation(
            f"load_production_model: model expects {booster_n} features "
            f"but dataset has {dataset_n}. Evaluation would be silent "
            f"default-leaf garbage."
        )

    contract = load_contract(DATA_DIR)
    if contract is None:
        raise FeatureContractViolation(
            "load_production_model: production_feature_contract.json "
            "missing. Run scripts/train_lgb.py to populate."
        )
    actual_names = [
        col[1] if isinstance(col, tuple) else str(col)
        for col in _xtest.columns.tolist()
    ]
    verify_inference_dataset(contract, actual_names)

    return model, dataset
