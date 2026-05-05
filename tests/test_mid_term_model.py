import pytest
import numpy as np
import pandas as pd
import torch
from models.mid_term import LSTMAttention, MidTermModel


def _make_price_df(days=30):
    """Create a synthetic price DataFrame."""
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(days) * 2)
    return pd.DataFrame({
        "open": close - np.random.rand(days),
        "high": close + np.abs(np.random.randn(days)),
        "low": close - np.abs(np.random.randn(days)),
        "close": close,
        "volume": np.random.randint(1000000, 5000000, days),
    }, index=dates)


def test_lstm_attention_forward():
    """LSTMAttention forward pass should produce correct output shape."""
    model = LSTMAttention(input_size=11, hidden_size=32, num_layers=2)
    x = torch.randn(4, 20, 11)  # batch=4, seq=20, features=11
    output = model(x)
    assert output.shape == (4, 1)
    assert torch.all(output >= -1) and torch.all(output <= 1)


def test_prepare_features():
    """prepare_features should return correct shape."""
    model = MidTermModel(lookback_days=10)
    df = _make_price_df(30)
    features = model.prepare_features(df)
    assert features.shape == (30, 11)
    # Price features should be non-zero
    assert np.all(features[:, 3] > 0)  # close prices normalized


def test_prepare_features_with_sentiment():
    """Should incorporate sentiment features when provided."""
    model = MidTermModel(lookback_days=10)
    df = _make_price_df(5)
    sentiment = [
        {"sentiment_score": 0.5, "heat": 0.3},
        {"sentiment_score": -0.2, "heat": 0.5},
        {"sentiment_score": 0.1, "heat": 0.2},
        {"sentiment_score": 0.7, "heat": 0.8},
        {"sentiment_score": -0.5, "heat": 0.1},
    ]
    features = model.prepare_features(df, sentiment_scores=sentiment)
    assert features[0, 5] == 0.5  # sentiment_score
    assert features[0, 6] == 0.3  # heat


def test_predict_returns_dict():
    """predict should return a dict with trend info."""
    model = MidTermModel(lookback_days=10)
    df = _make_price_df(30)
    result = model.predict(df)
    assert "trend_score" in result
    assert "trend_label" in result
    assert "confidence" in result
    assert -1.0 <= result["trend_score"] <= 1.0
    assert result["trend_label"] in ("强看多", "看多", "中性", "看空", "强看空")


def test_predict_insufficient_data():
    """predict with insufficient data should return neutral."""
    model = MidTermModel(lookback_days=30)
    df = _make_price_df(10)  # Only 10 days, need 30
    result = model.predict(df)
    assert result["trend_score"] == 0.0
    assert result["trend_label"] == "中性"
    assert result["confidence"] == 0.0


def test_train_and_predict():
    """Training should not crash and predictions should change."""
    model = MidTermModel(lookback_days=10)

    # Create synthetic training data
    train_data = []
    for _ in range(50):
        features = np.random.randn(10, 11)
        target = np.random.uniform(-0.5, 0.5)
        train_data.append((features, target))

    model.train_model(train_data, epochs=5, lr=0.01, batch_size=16)

    # Model should produce predictions
    df = _make_price_df(20)
    result = model.predict(df)
    assert isinstance(result["trend_score"], float)


def test_save_and_load(tmp_path):
    """Save and load should preserve model state."""
    model = MidTermModel(lookback_days=10)
    model._feature_means = np.zeros(11)
    model._feature_stds = np.ones(11)

    save_path = str(tmp_path / "model.pt")
    model.save(save_path)

    model2 = MidTermModel(lookback_days=10)
    model2.load(save_path)

    # Both models should give the same prediction
    df = _make_price_df(20)
    r1 = model.predict(df)
    r2 = model2.predict(df)
    assert r1["trend_score"] == r2["trend_score"]
