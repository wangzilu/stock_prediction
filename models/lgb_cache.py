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
    stale_count: int = 0,
    allow_red: bool = False,
) -> dict:
    """Write an atomic JSON cache for scheduler/RL fallback usage.

    2026-06-04 cx round 3 P0-2: distribution health is now validated
    HERE, at the single write-path bottleneck. Pre-fix the gate lived
    only in ``scheduler.jobs.DailyPipeline``, so the 18:35
    ``lgb_after_close_smoke`` cron could land an all-negative cache
    that downstream paper / shadow / ensemble consumers then read as
    truth. Sinking the gate into the writer means SMOKE, SCHEDULER,
    or any future producer cannot poison the cache without an
    explicit ``allow_red=True`` opt-out.

    Args:
        allow_red: only used by tests / debug. When True, RED writes
            are allowed (the JSON records ``distribution_status="RED"``
            so even loaders that ignore the exception can see it).
    """
    from models.prediction_health import (
        classify_status,
        PredictionDistributionRed,
    )

    finite = finite_prediction_map(predictions)
    status, dist_stats = classify_status(finite, stale_count=stale_count)
    if status == "RED" and not allow_red:
        raise PredictionDistributionRed(
            f"refusing to write LGB cache from source={source!r}: "
            f"distribution_status=RED (reason: {dist_stats.get('reason')}). "
            f"This is the 2026-06-03 22:00 incident signature; "
            f"investigate upstream feature contract / inject path."
        )

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "latest_date": latest_date or "",
        "model_path": model_path or "",
        "qlib_dir": qlib_dir or "",
        "source": source,
        "prediction_count": len(predictions),
        "finite_prediction_count": len(finite),
        "min_predictions": min_predictions,
        "distribution_status": status,
        "distribution_stats": dist_stats,
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
    allow_red: bool = False,
) -> tuple[dict[str, float], dict]:
    """Load and validate a cached LGB prediction map.

    2026-06-04 cx round 3 P0-3: the loader now ALSO checks distribution
    health, so callers cannot silently consume a RED cache even if a
    writer somehow produced one (older cache from before the writer
    gate, manual debug, etc.). ``paper/oms.py`` and shadow scripts
    that previously read the raw JSON must move to this function so
    they get freshness + distribution gating for free.

    Args:
        allow_red: tests / debug only. When True the caller is
            explicitly opting in to consume a RED cache (you will
            usually NOT want this for live trading paths).
    """
    from models.prediction_health import (
        classify_status,
        PredictionDistributionRed,
    )

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

    # 2026-06-04 cx round 9 P1-3: tighten by also requiring
    # latest_date to be at least the most-recent CN trading date.
    # The 7-calendar-day default is too permissive for daily-frequency
    # A-share recommendations — it effectively let last week's
    # predictions feed today's signal. Trading-date check refuses
    # any cache that pre-dates the last trading session.
    try:
        from scheduler.data_health import _expected_latest_trading_date
        expected = _expected_latest_trading_date()
        if latest_date < expected:
            raise RuntimeError(
                f"LGB prediction cache latest_date={latest_date} is older "
                f"than last trading date {expected} (CN trading-day gate)"
            )
    except RuntimeError:
        raise
    except Exception:
        # Fallback to calendar-day check (still applied above) if the
        # trading-day helper isn't available for any reason.
        pass

    # Distribution health gate. Prefer the writer-recorded status
    # (faster, no recomputation), fall back to classifying live for
    # caches written before the writer gate landed.
    recorded_status = str(payload.get("distribution_status", "")).upper()
    if recorded_status in ("GREEN", "YELLOW", "RED"):
        status = recorded_status
    else:
        status, _ = classify_status(finite)
    if status == "RED" and not allow_red:
        raise PredictionDistributionRed(
            f"refusing to consume LGB cache at {path}: "
            f"distribution_status=RED. Source={payload.get('source', '?')!r}, "
            f"latest_date={latest_date!r}. The cache itself is the 6-3 "
            f"incident signature; do not trade on it."
        )

    return finite, payload
