from datetime import datetime, timedelta

import pytest

from models.lgb_cache import load_prediction_cache, write_prediction_cache


def test_lgb_prediction_cache_round_trip(tmp_path):
    cache_path = tmp_path / "lgb_cache.json"

    write_prediction_cache(
        {"SH600519": 0.1, "BAD": float("nan")},
        cache_path,
        latest_date=datetime.now().strftime("%Y-%m-%d"),
        min_predictions=1,
    )
    preds, payload = load_prediction_cache(cache_path, min_predictions=1)

    assert preds == {"SH600519": 0.1}
    assert payload["finite_prediction_count"] == 1


def test_lgb_prediction_cache_rejects_stale_latest_date(tmp_path):
    cache_path = tmp_path / "lgb_cache.json"
    old_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    write_prediction_cache(
        {"SH600519": 0.1},
        cache_path,
        latest_date=old_date,
        min_predictions=1,
    )

    with pytest.raises(RuntimeError, match="days old"):
        load_prediction_cache(cache_path, min_predictions=1, max_age_days=2)
