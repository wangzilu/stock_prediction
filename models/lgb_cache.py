"""Shared helpers for production LightGBM prediction cache."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Mapping

from config.settings import (
    LGB_CACHE_MAX_AGE_DAYS,
    LGB_MIN_PREDICTIONS,
    LGB_PREDICTION_CACHE_PATH,
)


def finite_prediction_map(predictions: Mapping[str, float]) -> dict[str, float]:
    """Normalize prediction mapping and keep only finite scores."""
    finite: dict[str, float] = {}
    for code, score in predictions.items():
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            finite[str(code).upper()] = value
    return finite


def write_prediction_cache(
    predictions: Mapping[str, float],
    path: Path = LGB_PREDICTION_CACHE_PATH,
    *,
    latest_date: str | None = None,
    model_path: str | None = None,
    qlib_dir: str | None = None,
    min_predictions: int = LGB_MIN_PREDICTIONS,
    source: str = "lgb_smoke",
) -> dict:
    """Write an atomic JSON cache for scheduler/RL fallback usage."""
    finite = finite_prediction_map(predictions)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "latest_date": latest_date or "",
        "model_path": model_path or "",
        "qlib_dir": qlib_dir or "",
        "source": source,
        "prediction_count": len(predictions),
        "finite_prediction_count": len(finite),
        "min_predictions": min_predictions,
        "predictions": finite,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    return payload


def _date_age_days(value: str) -> int | None:
    if not value:
        return None
    try:
        date_value = datetime.fromisoformat(value[:10]).date()
    except ValueError:
        return None
    return (datetime.now().date() - date_value).days


def load_prediction_cache(
    path: Path = LGB_PREDICTION_CACHE_PATH,
    *,
    min_predictions: int = LGB_MIN_PREDICTIONS,
    max_age_days: int = LGB_CACHE_MAX_AGE_DAYS,
) -> tuple[dict[str, float], dict]:
    """Load and validate a cached LGB prediction map."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LGB prediction cache not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = payload.get("predictions", {})
    if not isinstance(predictions, dict):
        raise RuntimeError("LGB prediction cache has invalid predictions payload")

    finite = finite_prediction_map(predictions)
    if len(finite) < min_predictions:
        raise RuntimeError(
            f"cached finite LGB predictions {len(finite)} < required {min_predictions}"
        )

    latest_date = str(payload.get("latest_date", ""))
    age_days = _date_age_days(latest_date)
    if age_days is None:
        # Fail-closed: an unparseable / empty latest_date used to be accepted
        # silently (research-time leniency). For live recommendation + paper
        # OMS that meant cache of unknown age could drive trading decisions
        # whenever live inference failed. Reject explicitly.
        raise RuntimeError(
            f"LGB prediction cache latest_date={latest_date!r} unparseable — "
            f"refusing to use a prediction file with no provenance date"
        )
    if age_days > max_age_days:
        raise RuntimeError(
            f"LGB prediction cache latest_date={latest_date} is {age_days} days old "
            f"(max={max_age_days})"
        )

    return finite, payload
