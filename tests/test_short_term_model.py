import numpy as np
import pandas as pd
import pytest


pytest.importorskip("qlib")

from models.short_term import ShortTermModel
from scripts.train_lgb import _prediction_health


def test_latest_finite_predictions_use_latest_score_per_instrument():
    dates = pd.to_datetime(["2026-05-06", "2026-05-07"])
    index = pd.MultiIndex.from_tuples(
        [
            (dates[0], "sh600519"),
            (dates[1], "sh600519"),
            (dates[0], "sz300750"),
            (dates[1], "sz300750"),
        ],
        names=["datetime", "instrument"],
    )
    predictions = pd.DataFrame(
        {"score": [0.03, np.nan, -0.01, 0.05]},
        index=index,
    )

    latest = ShortTermModel._latest_finite_predictions(predictions)

    assert latest.attrs["latest_date"] == "2026-05-07"
    assert latest.attrs["stale_prediction_count"] == 1
    assert latest.loc["SH600519", "score"] == 0.03
    assert latest.loc["SZ300750", "score"] == 0.05


def test_train_health_matches_latest_score_per_instrument_policy():
    dates = pd.to_datetime(["2026-05-06", "2026-05-07"])
    index = pd.MultiIndex.from_tuples(
        [
            (dates[0], "sh600519"),
            (dates[1], "sh600519"),
            (dates[0], "sz300750"),
            (dates[1], "sz300750"),
        ],
        names=["datetime", "instrument"],
    )
    predictions = pd.DataFrame(
        {"score": [0.03, np.nan, -0.01, 0.05]},
        index=index,
    )

    stats = _prediction_health(predictions, min_predictions=2)

    assert stats["latest_finite_prediction_count"] == 2
    assert stats["stale_prediction_count"] == 1
