"""Daily crash probability prediction — inference only.

Loads the latest trained LightGBM crash model from
data/storage/experiments/crash_crash_5d_*/, runs inference on today's
features from the feature cache, and writes crash_prob predictions to
data/storage/crash_predictions_latest.json.

If no trained crash model exists, exits gracefully with rc=0.

Usage:
    python scripts/predict_crash_daily.py
"""
from __future__ import annotations

import glob
import json
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "storage"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
FEATURE_CACHE = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
OUTPUT_PATH = DATA_DIR / "crash_predictions_latest.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_latest_crash_model() -> Path | None:
    """Find the most recent crash_crash_5d experiment with a model.pkl.

    Falls back to data/storage/crash_model/model.pkl if no experiment artifact
    contains a model.
    """
    pattern = str(EXPERIMENTS_DIR / "crash_crash_5d_*")
    candidates = sorted(glob.glob(pattern), reverse=True)
    for exp_dir in candidates:
        model_path = Path(exp_dir) / "model.pkl"
        if model_path.exists():
            return model_path
    # Fallback: standalone crash model directory
    fallback = DATA_DIR / "crash_model" / "model.pkl"
    if fallback.exists():
        return fallback
    return None


def load_today_features():
    """Load the latest date's features from the feature cache.

    Returns (X, instruments, date_str, feat_cols).
    """
    import pandas as pd

    if not FEATURE_CACHE.exists():
        raise FileNotFoundError(f"Feature cache not found: {FEATURE_CACHE}")

    df = pd.read_parquet(FEATURE_CACHE)
    dates = df.index.get_level_values("datetime")
    latest_date = dates.max()
    latest_slice = df.loc[latest_date]

    # Feature columns: exclude labels/internal columns
    feat_cols = [c for c in latest_slice.columns
                 if not c.startswith("__") and not c.startswith("_")]

    X = latest_slice[feat_cols].replace([np.inf, -np.inf], np.nan)
    instruments = latest_slice.index.tolist()
    date_str = str(latest_date.date()) if hasattr(latest_date, "date") else str(latest_date)

    return X, instruments, date_str, feat_cols


def predict(model, X) -> np.ndarray:
    """Run crash probability inference."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X.values)[:, 1]
    # Fallback for raw booster
    return model.predict(X.values)


def run() -> dict:
    """Main prediction pipeline. Returns result dict."""
    t0 = time.time()

    # 1. Find model
    model_path = find_latest_crash_model()
    if model_path is None:
        logger.info("No trained crash model found — skipping.")
        return {"ok": True, "skipped": True, "reason": "no_model"}

    logger.info("Loading crash model: %s", model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. Load features
    logger.info("Loading today's features from %s", FEATURE_CACHE)
    X, instruments, date_str, feat_cols = load_today_features()
    logger.info("Date: %s, instruments: %d, features: %d",
                date_str, len(instruments), len(feat_cols))

    # 3. Predict
    proba = predict(model, X)
    predictions = {
        inst: round(float(p), 6)
        for inst, p in zip(instruments, proba)
        if np.isfinite(p)
    }
    logger.info("Generated %d crash_prob predictions (%.1fs)",
                len(predictions), time.time() - t0)

    # 4. Save
    payload = {
        "date": date_str,
        "model_path": str(model_path),
        "n_predictions": len(predictions),
        "generated_at": datetime.now().isoformat(),
        "predictions": predictions,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp.replace(OUTPUT_PATH)
    logger.info("Saved to %s", OUTPUT_PATH)

    # Summary stats
    vals = list(predictions.values())
    high_risk = [v for v in vals if v > 0.3]
    return {
        "ok": True,
        "skipped": False,
        "date": date_str,
        "model_path": str(model_path),
        "n_predictions": len(predictions),
        "n_high_risk": len(high_risk),
        "mean_prob": round(float(np.mean(vals)), 4) if vals else 0.0,
        "max_prob": round(float(np.max(vals)), 4) if vals else 0.0,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def main(argv: Iterable[str] | None = None) -> int:
    from scheduler.data_health import write_health, HealthStatus

    try:
        result = run()
    except Exception as exc:
        logger.error("Crash prediction failed: %s", exc, exc_info=True)
        write_health("crash_predict", HealthStatus(
            success=False,
            error_type=type(exc).__name__,
            error_message=str(exc)[:200],
        ))
        return 1

    if result.get("skipped"):
        logger.info("Skipped — no crash model available.")
        write_health("crash_predict", HealthStatus(
            success=True,
            error_message="skipped: no model",
        ))
        return 0

    if result["ok"]:
        logger.info(
            "Crash predict OK: %d predictions, %d high-risk (>0.3), "
            "mean=%.4f, max=%.4f, %.1fs",
            result["n_predictions"], result["n_high_risk"],
            result["mean_prob"], result["max_prob"],
            result["elapsed_sec"],
        )
        write_health("crash_predict", HealthStatus(
            success=True,
            n_items=result["n_predictions"],
            latest_date=result["date"],
        ))
    else:
        write_health("crash_predict", HealthStatus(
            success=False,
            error_type="PredictionFailure",
            error_message=str(result.get("error", ""))[:200],
        ))

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
