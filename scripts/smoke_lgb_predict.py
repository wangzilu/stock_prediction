"""Smoke-test the production LightGBM/Qlib inference path.

Run this from a real script file instead of stdin so Qlib/joblib
multiprocessing can spawn safely on macOS.

Usage:
    python scripts/smoke_lgb_predict.py
"""
from __future__ import annotations

import os
# Must precede joblib/qlib import — see main.py header for the
# 2026-06-08 morning-hang root cause writeup.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# cx round 23 E.P2 #4: do NOT hardcode the legacy ``lgb_model.pkl`` here.
# The legacy symlink existed for back-compat but the active profile may
# point at ``lgb_model_xgb_209.pkl`` / ``lgb_model_xgb_242.pkl`` directly.
# Resolve to the active profile's binary via production_model_filename so
# the smoke check exercises the same artifact the cron does.
try:
    from config.production_features import production_model_filename
    DEFAULT_MODEL_PATH = (
        PROJECT_ROOT / "data" / "storage" / production_model_filename()
    )
except Exception:
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
    model_path: Path | None,
    min_predictions: int,
    qlib_dir: Path | None = None,
    output: Path | None = LGB_PREDICTION_CACHE_PATH,
    inference_end_date: str | None = None,
) -> dict:
    _override_qlib_provider(qlib_dir)

    from models.short_term import ShortTermModel
    from models.lgb_cache import write_prediction_cache

    # cx round 23 E.P2 #4: when --model-path is not explicitly passed
    # (model_path is None), delegate resolution to
    # ShortTermModel.load_from_pickle(None) — which loads the active
    # profile's binary. Only when the caller hands a literal path do we
    # exist-check it and pass it through.
    if model_path is not None:
        if not model_path.exists():
            return {
                "ok": False,
                "error": f"model file not found: {model_path}",
                "model_path": str(model_path),
            }
        model = ShortTermModel.load_from_pickle(
            str(model_path), inference_end_date=inference_end_date
        )
    else:
        # Resolve effective path for logging/cache metadata; mirror the
        # active-profile resolution that load_from_pickle(None) performs.
        from config.production_features import production_model_filename
        from config.settings import DATA_DIR as _DATA_DIR
        preferred = Path(_DATA_DIR) / production_model_filename()
        effective_path = preferred if preferred.exists() else Path(_DATA_DIR) / "lgb_model.pkl"
        model = ShortTermModel.load_from_pickle(
            None, inference_end_date=inference_end_date
        )
        model_path = effective_path
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
    # cx round 23 E.P2 #4: default is None so run_smoke delegates to
    # ShortTermModel.load_from_pickle(None) → active-profile binary.
    # DEFAULT_MODEL_PATH is shown only in --help epilog as a hint.
    parser.add_argument(
        "--model-path", type=Path, default=None,
        help=(
            f"Override the model binary. When omitted, use the active "
            f"profile's binary (currently {DEFAULT_MODEL_PATH.name})."
        ),
    )
    parser.add_argument("--min-predictions", type=int, default=LGB_MIN_PREDICTIONS)
    parser.add_argument("--qlib-dir", type=Path, default=None)
    parser.add_argument(
        "--date",
        default=None,
        help=(
            "Business date for upstream freshness checks and health output "
            "(YYYY-MM-DD). Defaults to today."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LGB_PREDICTION_CACHE_PATH,
        help="Write a validated LGB prediction cache for scheduler/RL fallback",
    )
    parser.add_argument("--no-cache", action="store_true", help="Do not write prediction cache")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--skip-gate", action="store_true",
        help=(
            "Bypass the upstream-source freshness gate. ONLY for manual "
            "rescue when the cron chain has been stuck and you accept "
            "that the resulting prediction cache is based on stale "
            "upstream signals. The output will still record the actual "
            "latest_date from the feature cache so consumers know how "
            "fresh the prediction window is. Use case: 2026-06-14 cron "
            "chain blocked 5 trading days, manually run smoke on the "
            "06-09 feature cache so users stop getting the same picks."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    from scheduler.data_health import write_health, HealthStatus, check_training_gate

    args = parse_args(argv)

    # --- Freshness gate: check upstream data before predicting ---
    gate_result = check_training_gate(args.date)
    gate = gate_result["gate"]

    if gate == "fail":
        if args.skip_gate:
            logger.warning(
                "Training gate FAIL but --skip-gate set: proceeding with "
                "stale upstream signals. Reason: %s",
                gate_result.get("reason", "unknown"),
            )
        else:
            logger.error(
                "Training gate FAILED — skipping prediction. Reason: %s",
                gate_result.get("reason", "unknown"),
            )
            write_health(
                "lgb_after_close_smoke",
                HealthStatus(
                    success=False,
                    error_type="GateFail",
                    error_message=gate_result.get("reason", "upstream data stale")[:200],
                ),
                date=args.date,
            )
            return 1

    if gate == "degrade":
        logger.warning(
            "Training gate DEGRADED — proceeding with degraded overlays: %s",
            gate_result.get("degraded_overlays", []),
        )

    try:
        output = None if args.no_cache else args.output
        result = run_smoke(
            args.model_path,
            args.min_predictions,
            args.qlib_dir,
            output,
            inference_end_date=args.date,
        )
    except Exception as exc:
        result = {
            "ok": False,
            "model_path": str(args.model_path) if args.model_path is not None
                          else f"<active-profile:{DEFAULT_MODEL_PATH.name}>",
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
        write_health(
            "lgb_after_close_smoke",
            HealthStatus(
                success=True,
                n_items=result.get("finite_prediction_count", 0),
                latest_date=result.get("cache_latest_date", ""),
                network_profile="domestic",
            ),
            date=args.date,
        )
    else:
        write_health(
            "lgb_after_close_smoke",
            HealthStatus(
                success=False,
                error_type="PredictionFailure",
                error_message=str(result.get("error", ""))[:200],
            ),
            date=args.date,
        )

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
