"""Phase 4D: Shadow daily inference — load saved models, predict, compare.

NO retraining. Models are loaded from saved artifacts.
Retrain happens separately on weekly/monthly cadence.

Daily flow: load models → predict latest → compare → log

Usage:
    python scripts/shadow_daily_inference.py
"""
import json
import logging
import os
import sys
import time
from datetime import datetime
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
SHADOW_LOG = DATA_DIR / "phase4" / "shadow_ledger.jsonl"

# Model paths — loaded from registry, not hardcoded
from models.registry import ModelRegistry


def load_latest_predictions():
    """Load latest predictions through validated loader (cx round 3 P1-8).

    Returns (predictions_dict, full_payload) or (None, {}) when the
    cache is missing, RED, or stale. The shadow run skips when None
    rather than fabricating stats off a poisoned cache."""
    from models.lgb_cache import load_prediction_cache
    from models.prediction_health import PredictionDistributionRed
    cache_path = DATA_DIR / "lgb_latest_predictions.json"
    try:
        preds, payload = load_prediction_cache(cache_path)
    except FileNotFoundError:
        return None, {}
    except PredictionDistributionRed as exc:
        logger.error("shadow_daily_inference refusing RED cache: %s — skip.", exc)
        return None, {}
    except RuntimeError as exc:
        logger.error("shadow_daily_inference cache load failed: %s", exc)
        return None, {}
    return preds, payload


def predict_with_xgb_model(model_path: str, feature_cache_path: str) -> dict:
    """Load XGB model and predict on latest date from cache."""
    import xgboost as xgb

    if not Path(model_path).exists():
        logger.warning(f"Model not found: {model_path}")
        return {}

    if not Path(feature_cache_path).exists():
        logger.warning(f"Cache not found: {feature_cache_path}")
        return {}

    model = xgb.Booster()
    model.load_model(str(model_path))

    # Load cache, get latest date only
    cache = pd.read_parquet(str(feature_cache_path))
    latest_date = cache.index.get_level_values(0).max()
    latest = cache.loc[latest_date]

    feature_cols = [c for c in cache.columns if not c.startswith("__") and not c.startswith("_")]
    X = latest[feature_cols].values.astype(np.float32)

    pred = model.predict(xgb.DMatrix(X))
    instruments = latest.index if isinstance(latest.index, pd.Index) else latest.index.get_level_values(0)

    result = {}
    for inst, score in zip(instruments, pred):
        if np.isfinite(score):
            result[str(inst).upper()] = float(score)

    return result


def compare_predictions(champion_preds: dict, shadow_preds: dict, top_k: int = 20) -> dict:
    """Compare champion vs shadow Top20."""
    if not champion_preds or not shadow_preds:
        return {"error": "missing predictions"}

    ch_sorted = sorted(champion_preds.items(), key=lambda x: -x[1])
    sh_sorted = sorted(shadow_preds.items(), key=lambda x: -x[1])

    ch_top = set(k for k, _ in ch_sorted[:top_k])
    sh_top = set(k for k, _ in sh_sorted[:top_k])

    overlap = ch_top & sh_top
    ch_only = ch_top - sh_top
    sh_only = sh_top - ch_top

    # Rank correlation on common stocks
    common = set(champion_preds.keys()) & set(shadow_preds.keys())
    if len(common) > 50:
        from scipy.stats import spearmanr
        common_list = sorted(common)
        ch_vals = [champion_preds[c] for c in common_list]
        sh_vals = [shadow_preds[c] for c in common_list]
        rank_corr, _ = spearmanr(ch_vals, sh_vals)
    else:
        rank_corr = float("nan")

    return {
        "top_k": top_k,
        "champion_count": len(champion_preds),
        "shadow_count": len(shadow_preds),
        "overlap": len(overlap),
        "overlap_pct": round(len(overlap) / top_k, 4),
        "champion_only": sorted(ch_only)[:5],
        "shadow_only": sorted(sh_only)[:5],
        "rank_correlation": round(float(rank_corr), 4) if np.isfinite(rank_corr) else None,
        "champion_top5": [k for k, _ in ch_sorted[:5]],
        "shadow_top5": [k for k, _ in sh_sorted[:5]],
    }


def main():
    init_qlib(QLIB_DATA)

    logger.info("=== Shadow Daily Inference ===")
    t0 = time.time()

    # Load model paths from registry
    reg = ModelRegistry()
    champion_info = reg.get_champion()
    shadow_info = reg.get_shadow()

    logger.info(f"  Registry: champion={reg.status()['champion']}, shadow={reg.status()['shadow']}")

    # Champion: load from prediction cache (already computed by after_close_pipeline)
    logger.info("Loading champion predictions from cache...")
    ch_preds, ch_meta = load_latest_predictions()
    if ch_preds:
        logger.info(f"  Champion: {len(ch_preds)} predictions, "
                    f"source={ch_meta.get('source', '?')}, "
                    f"date={ch_meta.get('latest_date', '?')}")
    else:
        logger.warning("  Champion predictions not available!")

    # Shadow: load model from registry path
    shadow_model_path = shadow_info.get("model_path", "") if shadow_info else ""
    shadow_cache_path = DATA_DIR / "feature_cache_174_holder_regime_ma.parquet"
    if not shadow_model_path:
        # Fallback to known paths
        for fallback in ["xgb_205_regime_model.json", "xgb_175_holder_model.json"]:
            if (DATA_DIR / fallback).exists():
                shadow_model_path = str(DATA_DIR / fallback)
                break

    logger.info(f"Loading shadow model: {shadow_model_path}")
    sh_preds = predict_with_xgb_model(shadow_model_path, str(shadow_cache_path))
    if sh_preds:
        logger.info(f"  Shadow: {len(sh_preds)} predictions")
    else:
        logger.warning("  Shadow predictions not available!")

    # Compare
    result = compare_predictions(ch_preds or {}, sh_preds)
    result["date"] = datetime.now().strftime("%Y-%m-%d")
    result["compared_at"] = datetime.now().isoformat(timespec="seconds")
    result["champion_model"] = reg.status()["champion"] or "xgb_174"
    result["shadow_model"] = reg.status()["shadow"] or "xgb_205"
    result["inference_time_s"] = round(time.time() - t0, 1)

    logger.info(f"\n  Overlap: {result.get('overlap', 0)}/{result.get('top_k', 20)} "
                f"({result.get('overlap_pct', 0):.0%})")
    logger.info(f"  Rank corr: {result.get('rank_correlation', 'N/A')}")
    logger.info(f"  Champion top5: {result.get('champion_top5', [])}")
    logger.info(f"  Shadow top5:   {result.get('shadow_top5', [])}")
    logger.info(f"  Time: {result['inference_time_s']:.1f}s")

    # Append to ledger
    SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(str(SHADOW_LOG), "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    logger.info(f"  Logged to: {SHADOW_LOG}")

    # Check shadow ledger length for promotion readiness
    if SHADOW_LOG.exists():
        n_days = sum(1 for _ in open(str(SHADOW_LOG)))
        logger.info(f"  Shadow ledger: {n_days} days (need 20 for promotion)")

    logger.info("Done!")


if __name__ == "__main__":
    main()
