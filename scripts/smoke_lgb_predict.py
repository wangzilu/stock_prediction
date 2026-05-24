"""Smoke-test the production LightGBM/Qlib inference path.

Run this from a real script file instead of stdin so Qlib/joblib
multiprocessing can spawn safely on macOS.

Usage:
    python scripts/smoke_lgb_predict.py
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "storage" / "lgb_model.pkl"
try:
    from config.settings import LGB_MIN_PREDICTIONS, LGB_PREDICTION_CACHE_PATH
except Exception:
    LGB_MIN_PREDICTIONS = 100
    LGB_PREDICTION_CACHE_PATH = PROJECT_ROOT / "data" / "storage" / "lgb_latest_predictions.json"


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _override_qlib_provider(qlib_dir: Path | None) -> None:
    if qlib_dir is None:
        return

    provider_uri = str(qlib_dir.resolve())
    os.environ["QLIB_PROVIDER_URI"] = provider_uri

    try:
        import config.settings as settings

        settings.QLIB_PROVIDER_URI = provider_uri
    except Exception:
        pass

    short_term = sys.modules.get("models.short_term")
    if short_term is not None:
        short_term.QLIB_PROVIDER_URI = provider_uri


def run_smoke(
    model_path: Path,
    min_predictions: int,
    qlib_dir: Path | None = None,
    output: Path | None = LGB_PREDICTION_CACHE_PATH,
) -> dict:
    _override_qlib_provider(qlib_dir)

    from models.short_term import ShortTermModel
    from models.lgb_cache import write_prediction_cache

    if not model_path.exists():
        return {
            "ok": False,
            "error": f"model file not found: {model_path}",
            "model_path": str(model_path),
        }

    model = ShortTermModel.load_from_pickle(str(model_path))
    preds = model.predict_batch()
    finite_items = {
        code: score
        for code, score in preds.items()
        if isinstance(score, (int, float)) and math.isfinite(float(score))
    }
    nan_count = len(preds) - len(finite_items)

    top = sorted(finite_items.items(), key=lambda item: item[1], reverse=True)[:5]
    bottom = sorted(finite_items.items(), key=lambda item: item[1])[:5]

    ok = len(finite_items) >= min_predictions
    result = {
        "ok": ok,
        "model_path": str(model_path),
        "qlib_dir": str(qlib_dir.resolve()) if qlib_dir else None,
        "prediction_count": len(preds),
        "finite_prediction_count": len(finite_items),
        "nan_prediction_count": nan_count,
        "min_predictions": min_predictions,
        "top": top,
        "bottom": bottom,
    }
    if not ok:
        result["error"] = (
            f"finite predictions {len(finite_items)} < required {min_predictions}"
        )
    elif output is not None:
        cache_payload = write_prediction_cache(
            finite_items,
            output,
            latest_date=getattr(model, "latest_prediction_date", ""),
            model_path=str(model_path),
            qlib_dir=str(qlib_dir.resolve()) if qlib_dir else None,
            min_predictions=min_predictions,
        )
        result["cache_path"] = str(output)
        result["cache_latest_date"] = cache_payload.get("latest_date", "")
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--min-predictions", type=int, default=LGB_MIN_PREDICTIONS)
    parser.add_argument("--qlib-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=LGB_PREDICTION_CACHE_PATH,
        help="Write a validated LGB prediction cache for scheduler/RL fallback",
    )
    parser.add_argument("--no-cache", action="store_true", help="Do not write prediction cache")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    from scheduler.data_health import write_health, HealthStatus

    args = parse_args(argv)
    try:
        output = None if args.no_cache else args.output
        result = run_smoke(args.model_path, args.min_predictions, args.qlib_dir, output)
    except Exception as exc:
        result = {
            "ok": False,
            "model_path": str(args.model_path),
            "error": f"{type(exc).__name__}: {exc}",
        }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result["ok"]:
        logger.info(
            "LGB smoke OK: %s finite predictions (%s total)",
            result["finite_prediction_count"],
            result["prediction_count"],
        )
        logger.info("Top examples: %s", result["top"])
        if result.get("cache_path"):
            logger.info("Prediction cache written: %s", result["cache_path"])
    else:
        logger.error("LGB smoke failed: %s", result.get("error", "unknown error"))

    # Write health status
    if result["ok"]:
        write_health("lgb_smoke_predict", HealthStatus(
            success=True,
            n_items=result.get("finite_prediction_count", 0),
            latest_date=result.get("cache_latest_date", ""),
            network_profile="domestic",
        ))
    else:
        write_health("lgb_smoke_predict", HealthStatus(
            success=False,
            error_type="PredictionFailure",
            error_message=str(result.get("error", ""))[:200],
        ))

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
