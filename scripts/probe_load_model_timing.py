"""Time each phase of ShortTermModel.load_from_pickle() to find the
25-min silent hang in morning_recommendation.

Phases timed (in order):
  T0  Module import + qlib init
  T1  Pickle load (model only, no dataset)
  T2  Alpha158 handler creation + raw data load
  T3  Supplementary feature injection (~84 cols, parquet I/O)
  T4  Qlib custom factor injection (no-op for xgb_242)
  T5  Profile dim assertion
  T6  dataset.prepare("test")  ← the contract-gate materialization
  T7  Booster num_features() + count gate
  T8  load_contract + verify_inference_dataset (name-order gate)
  T9  predict_batch (actual inference)

Output: one phase-timing line per phase, plus a final summary.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# unbuffered stdout so we see each phase as it completes
sys.stdout.reconfigure(line_buffering=True)


def main():
    overall_t0 = time.time()
    phases: list[tuple[str, float]] = []

    def stamp(name: str, t0: float):
        dt = time.time() - t0
        phases.append((name, dt))
        print(f"[{time.strftime('%H:%M:%S')}] {name}: {dt:.2f}s "
              f"(cum {time.time()-overall_t0:.1f}s)", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] === probe start ===", flush=True)

    t = time.time()
    from config.qlib_runtime import init_qlib
    from config.settings import DATA_DIR, LGB_MODEL_PATH, LGB_INFERENCE_UNIVERSE, QLIB_PROVIDER_URI
    from config.production_features import (
        production_model_filename, PRODUCTION_SUPPLEMENTARY_GROUPS,
        PRODUCTION_MODEL_PROFILE, assert_profile_dimensions,
    )
    from models.feature_merger import FeatureMerger
    from models.feature_contract import load_contract, verify_inference_dataset
    init_qlib(QLIB_PROVIDER_URI)
    stamp("T0 imports + init_qlib", t)

    profile_filename = production_model_filename()
    preferred = DATA_DIR / profile_filename
    model_path = str(preferred) if preferred.exists() else str(DATA_DIR / "lgb_model.pkl")
    print(f"  model_path: {model_path}, profile: {PRODUCTION_MODEL_PROFILE}", flush=True)

    t = time.time()
    with open(model_path, "rb") as f:
        model_obj = pickle.load(f)
    stamp("T1 pickle.load(model)", t)

    from datetime import datetime, timedelta
    from qlib.utils import init_instance_by_config
    today = datetime.now()
    test_start = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")
    print(f"  test window: {test_start} ~ {test_end}", flush=True)

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
            "segments": {"test": (test_start, test_end)},
        },
    }
    t = time.time()
    dataset = init_instance_by_config(dataset_config)
    stamp("T2 init_instance_by_config (Alpha158 raw load)", t)

    handler = dataset.handler
    merger = FeatureMerger()

    t = time.time()
    n_supp = merger.inject_supplementary_into_handler(
        handler, preprocess=False,
        groups=PRODUCTION_SUPPLEMENTARY_GROUPS,
    )
    stamp(f"T3 inject_supplementary ({n_supp} cols)", t)

    t = time.time()
    n_custom = merger.inject_qlib_custom_factors_into_handler(handler)
    stamp(f"T4 inject_qlib_custom ({n_custom} cols)", t)

    t = time.time()
    assert_profile_dimensions(
        alpha_count=158,
        supp_count=int(n_supp or 0),
        custom_count=int(n_custom or 0),
    )
    stamp("T5 assert_profile_dimensions", t)

    t = time.time()
    from qlib.data.dataset.handler import DataHandlerLP as _DK
    xtest = dataset.prepare("test", col_set="feature", data_key=_DK.DK_I)
    stamp(f"T6 dataset.prepare(test) shape={xtest.shape}", t)

    t = time.time()
    booster = getattr(model_obj, "model", None)
    booster_n = int(booster.num_features()) if booster is not None and hasattr(booster, "num_features") else -1
    stamp(f"T7 booster.num_features()={booster_n} vs dataset {xtest.shape[1]}", t)

    t = time.time()
    contract = load_contract(DATA_DIR)
    actual_names = [
        col[1] if isinstance(col, tuple) else str(col)
        for col in xtest.columns.tolist()
    ]
    verify_inference_dataset(contract, actual_names)
    stamp(f"T8 load_contract + verify_inference_dataset (profile={contract.get('profile', '?')})", t)

    t = time.time()
    preds = model_obj.predict(dataset=dataset)
    stamp(f"T9 model.predict() len={len(preds)}", t)

    print(f"\n[{time.strftime('%H:%M:%S')}] === Summary ===", flush=True)
    for name, dt in phases:
        bar = "█" * min(80, int(dt))
        print(f"  {dt:8.2f}s  {name}  {bar}")
    total = sum(dt for _, dt in phases)
    print(f"  --------")
    print(f"  {total:8.2f}s  TOTAL", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
