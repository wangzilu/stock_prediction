"""Shared health classifier for LGB short-term prediction maps.

Why this module exists (2026-06-04, cx round 3 P0-2):
    Pre-fix, the RED / YELLOW / GREEN logic lived ONLY inside
    ``scheduler.jobs.DailyPipeline._write_lgb_distribution_health``.
    That meant any other entry point that wrote the production cache
    bypassed the gate. The 18:35 ``lgb_after_close_smoke`` cron is
    exactly such an entry point — it calls
    ``models.lgb_cache.write_prediction_cache`` directly, no
    distribution check. An all-negative day produced by smoke would
    therefore land in ``data/storage/lgb_latest_predictions.json``
    and downstream paper / shadow / ensemble consumers (cx P0-3 +
    P1-8) would happily read it.

    Pulling the classifier here means EVERY writer and EVERY reader
    of the cache shares one definition of "RED", and the helper is
    cheap enough to invoke on every cache touch.

Status rules (unchanged from jobs.py):
    - RED   : empty, all-zero, OR no positive/no negative (the
              158→242 default-leaf signature from the 6-3 22:00
              incident).
    - YELLOW: positive_ratio < 10% (heavily skewed) OR
              stale_prediction_count > 10% of total.
    - GREEN : positive_ratio >= 10% AND fresh.
"""
from __future__ import annotations

from typing import Mapping


class PredictionDistributionRed(RuntimeError):
    """Raised when a prediction map is classified RED. Callers MUST
    NOT silently swallow this exception — that defeats the gate. The
    valid responses are: refuse to publish (writer), refuse to consume
    (reader), or surface an alert to the human."""


def classify_status(
    preds: Mapping[str, float],
    *,
    stale_count: int = 0,
    yellow_pos_ratio_threshold: float = 0.10,
    yellow_stale_ratio_threshold: float = 0.10,
) -> tuple[str, dict]:
    """Classify a prediction map and return (status, stats).

    ``stats`` is a dict with the canonical fields the scheduler's
    health JSON has historically used. Callers that only need the
    status can ignore it.
    """
    if not preds:
        return "RED", {
            "n_predictions": 0,
            "n_positive": 0,
            "n_negative": 0,
            "n_zero": 0,
            "positive_ratio": 0.0,
            "stale_prediction_count": int(stale_count),
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "reason": "empty prediction map",
        }

    values = []
    for v in preds.values():
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if x == x:  # NaN filter
            values.append(x)

    n = len(values)
    if n == 0:
        return "RED", {
            "n_predictions": 0,
            "n_positive": 0,
            "n_negative": 0,
            "n_zero": 0,
            "positive_ratio": 0.0,
            "stale_prediction_count": int(stale_count),
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "reason": "all values non-finite",
        }

    n_pos = sum(1 for v in values if v > 0)
    n_neg = sum(1 for v in values if v < 0)
    n_zero = sum(1 for v in values if v == 0)
    values_sorted = sorted(values)
    med = values_sorted[n // 2]
    mean = sum(values) / n
    pos_ratio = n_pos / n
    stale_ratio = stale_count / n if n else 0.0

    stats = {
        "n_predictions": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_zero": n_zero,
        "positive_ratio": round(pos_ratio, 4),
        "stale_prediction_count": int(stale_count),
        "mean": round(mean, 6),
        "median": round(med, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }

    if n_pos == 0 or n_neg == 0:
        stats["reason"] = (
            "all-positive or all-negative (default-leaf signature)"
        )
        return "RED", stats
    if pos_ratio < yellow_pos_ratio_threshold:
        stats["reason"] = (
            f"positive_ratio {pos_ratio:.2%} < "
            f"{yellow_pos_ratio_threshold:.0%}"
        )
        return "YELLOW", stats
    if stale_ratio > yellow_stale_ratio_threshold:
        stats["reason"] = (
            f"stale_ratio {stale_ratio:.2%} > "
            f"{yellow_stale_ratio_threshold:.0%}"
        )
        return "YELLOW", stats
    stats["reason"] = "balanced + fresh"
    return "GREEN", stats
