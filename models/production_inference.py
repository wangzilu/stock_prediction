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

import logging
import pickle
import re
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

logger = logging.getLogger(__name__)


def _infer_profile_from_model_path(model_path: str | Path) -> str:
    """Best-effort: pull a profile name out of a model binary basename.

    Used by ``load_production_model`` when the caller passes a custom
    ``model_path`` without an explicit ``profile=`` — falling back to the
    live ``PRODUCTION_MODEL_PROFILE`` global would silently apply the
    champion's supplementary groups / contract to a candidate-profile
    binary, which is exactly the cx round 23 E.P1 #3 hazard.

    Example: ``data/storage/lgb_model_xgb_242.pkl`` → ``"xgb_242"``.
    """
    from config.production_features import SUPPLEMENTARY_GROUPS_BY_PROFILE

    basename = Path(model_path).name
    # Match ``lgb_model_<profile>.pkl`` where <profile> is a known key.
    match = re.match(r"lgb_model_(?P<profile>[A-Za-z0-9_]+)\.pkl$", basename)
    if match:
        candidate = match.group("profile").strip().lower()
        if candidate in SUPPLEMENTARY_GROUPS_BY_PROFILE:
            return candidate
    raise FeatureContractViolation(
        f"load_production_model: cannot infer profile from model_path "
        f"basename {basename!r}. Pass profile= explicitly. Refusing to "
        f"fall back to the live PRODUCTION_MODEL_PROFILE — that would "
        f"apply the champion's supplementary groups + contract to a "
        f"binary that may be a different profile."
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
    profile: str | None = None,
):
    """Load the production LGB model + a profile-aligned dataset.

    Args:
        test_start, test_end: ISO date strings bounding the test
            segment. Used to build a ``DatasetH(segments={"test":
            (test_start, test_end)})`` configuration.
        model_path: path to ``lgb_model_<profile>.pkl``. Defaults to
            the active profile's binary.
        instruments: Qlib instrument set name. Defaults to the
            production inference universe.
        label_expr: optional Qlib label expression. None uses the
            default 5-day forward-return expression.
        profile: optional profile name (e.g. ``"xgb_242"``). When None
            we fall back to ``PRODUCTION_MODEL_PROFILE`` ONLY if the
            caller is using the default model_path; if model_path was
            overridden to a custom binary, we try to infer the profile
            from the basename and raise rather than silently use the
            live champion's contract (cx round 23 E.P1 #3).

    Returns:
        Tuple ``(model, dataset)`` where:
          * ``model`` is the unpickled XGB/LGB model object
          * ``dataset`` is a Qlib ``DatasetH`` whose feature shape
            matches the resolved profile's contract on the test segment

    Raises:
        FeatureContractViolation: when the trained model and the
            assembled inference dataset disagree on column count,
            supp column names, the contract artifact is missing, or
            the profile cannot be resolved.
    """
    from qlib.utils import init_instance_by_config
    from qlib.data.dataset.handler import DataHandlerLP
    from config.production_features import (
        PRODUCTION_MODEL_PROFILE,
        SUPPLEMENTARY_GROUPS_BY_PROFILE,
    )

    # cx round 23 E.P1 #3: resolve profile explicitly. The old code
    # always pulled PRODUCTION_SUPPLEMENTARY_GROUPS / the live contract
    # from the active champion profile — so pointing model_path at a
    # candidate binary would still verify against the champion's
    # contract and produce silent-garbage numbers.
    if profile is not None:
        resolved_profile = profile.strip().lower()
        if resolved_profile not in SUPPLEMENTARY_GROUPS_BY_PROFILE:
            raise FeatureContractViolation(
                f"load_production_model: unknown profile {resolved_profile!r}. "
                f"Allowed: {list(SUPPLEMENTARY_GROUPS_BY_PROFILE)}"
            )
    else:
        default_path = Path(LGB_MODEL_PATH)
        if Path(model_path) == default_path:
            resolved_profile = PRODUCTION_MODEL_PROFILE.strip().lower()
        else:
            logger.warning(
                "load_production_model: custom model_path=%s passed without "
                "profile=; attempting to infer profile from basename to "
                "avoid silently using the live champion's contract.",
                model_path,
            )
            resolved_profile = _infer_profile_from_model_path(model_path)

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

    # Same supp injection as ShortTermModel.load_from_pickle, but keyed
    # on the resolved profile (NOT the live champion). cx round 23 E.P1 #3.
    from models.feature_merger import FeatureMerger
    from config.production_features import QLIB_CUSTOM_FACTORS_BY_PROFILE
    merger = FeatureMerger()
    n_supp = merger.inject_supplementary_into_handler(
        dataset.handler, preprocess=False,
        groups=SUPPLEMENTARY_GROUPS_BY_PROFILE[resolved_profile],
    )
    # cx round 10 follow-up: some profiles (e.g. xgb_174) inject qlib-custom
    # expression factors; xgb_242 has none, so this is a no-op there. We
    # pass the resolved profile's custom factors explicitly so a candidate
    # binary isn't silently fed the champion's custom-factor set.
    # NOTE: pass the resolved profile's tuple ALWAYS, even when empty —
    # passing None would make the helper fall back to the live profile's
    # factors via ``current_profile_qlib_custom_factors()``, which is the
    # exact silent-champion-fallback we are eliminating here.
    custom_factors = QLIB_CUSTOM_FACTORS_BY_PROFILE.get(resolved_profile, ())
    n_custom = merger.inject_qlib_custom_factors_into_handler(
        dataset.handler, factor_specs=custom_factors,
    )
    # cx round 16 P1-3: strict profile-aware dim assertion, pinned to the
    # resolved profile rather than PRODUCTION_MODEL_PROFILE.
    from config.production_features import assert_profile_dimensions
    try:
        assert_profile_dimensions(
            alpha_count=158,
            supp_count=int(n_supp or 0),
            custom_count=int(n_custom or 0),
            profile=resolved_profile,
        )
    except RuntimeError as exc:
        raise FeatureContractViolation(str(exc)) from exc

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

    # cx round 23 E.P1 #3: load_contract takes profile=. Pin to the
    # resolved profile so a candidate model isn't verified against the
    # champion contract.
    contract = load_contract(DATA_DIR, profile=resolved_profile)
    if contract is None:
        raise FeatureContractViolation(
            f"load_production_model: contract artifact for profile "
            f"{resolved_profile!r} missing. Run scripts/train_lgb.py to "
            f"populate."
        )
    actual_names = [
        col[1] if isinstance(col, tuple) else str(col)
        for col in _xtest.columns.tolist()
    ]
    verify_inference_dataset(contract, actual_names)

    return model, dataset
