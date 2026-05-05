import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LSTMAttention(nn.Module):
    """LSTM with attention mechanism for mid-term stock trend prediction.

    Input features: price OHLCV (5) + sentiment (2) + geopolitical (4) = 11
    Output: trend direction and strength [-1, 1]
    """

    def __init__(self, input_size=11, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, input_size)

        Returns:
            Tensor of shape (batch, 1) with values in [-1, 1]
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden_size)

        return self.fc(context)


class MidTermModel:
    """Mid-term stock trend prediction model (1-4 weeks).

    Uses LSTM+Attention with price, sentiment, and geopolitical features.
    """

    FEATURE_NAMES = [
        "open", "high", "low", "close", "volume",  # Price features
        "sentiment_score", "sentiment_heat",  # Sentiment features
        "geo_risk_index", "china_us_temperature",  # Geopolitical features
        "policy_signal", "safe_haven_signal",
    ]

    def __init__(self, lookback_days=20, model_path=None):
        """
        Args:
            lookback_days: Number of historical days to use as input sequence
            model_path: Path to saved model weights (optional)
        """
        self.lookback_days = lookback_days
        self.model = LSTMAttention(input_size=len(self.FEATURE_NAMES))
        self.model_path = model_path

        # Feature normalization stats (updated during training)
        self._feature_means = None
        self._feature_stds = None

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def prepare_features(
        self,
        price_df: pd.DataFrame,
        sentiment_scores: list = None,
        geo_factors: list = None,
    ) -> np.ndarray:
        """Prepare feature matrix from raw data.

        Args:
            price_df: DataFrame with OHLCV columns, indexed by date
            sentiment_scores: List of dicts with sentiment_score and heat per day.
                             Length should match price_df. If None, fills with zeros.
            geo_factors: List of dicts with geo factor scores per day.
                        Length should match price_df. If None, fills with zeros.

        Returns:
            numpy array of shape (num_days, num_features)
        """
        n = len(price_df)

        # Price features (normalize by first close price for scale invariance)
        base_price = price_df["close"].iloc[0]
        features = np.zeros((n, len(self.FEATURE_NAMES)))
        features[:, 0] = price_df["open"].values / base_price
        features[:, 1] = price_df["high"].values / base_price
        features[:, 2] = price_df["low"].values / base_price
        features[:, 3] = price_df["close"].values / base_price
        features[:, 4] = np.log1p(price_df["volume"].values) / 20.0  # Log-normalize volume

        # Sentiment features
        if sentiment_scores and len(sentiment_scores) == n:
            for i, s in enumerate(sentiment_scores):
                features[i, 5] = s.get("sentiment_score", 0)
                features[i, 6] = s.get("heat", 0)

        # Geopolitical features
        if geo_factors and len(geo_factors) == n:
            for i, g in enumerate(geo_factors):
                features[i, 7] = g.get("geo_risk_index", 0)
                features[i, 8] = g.get("china_us_temperature", 0)
                features[i, 9] = g.get("policy_signal", 0)
                features[i, 10] = g.get("safe_haven_signal", 0)

        return features

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Z-score normalize features."""
        if self._feature_means is None:
            return features
        return (features - self._feature_means) / (self._feature_stds + 1e-8)

    def predict(
        self,
        price_df: pd.DataFrame,
        sentiment_scores: list = None,
        geo_factors: list = None,
    ) -> dict:
        """Predict mid-term trend for a stock.

        Args:
            price_df: At least lookback_days of OHLCV data
            sentiment_scores: Daily sentiment scores (optional)
            geo_factors: Daily geopolitical factors (optional)

        Returns:
            Dict with:
                - trend_score: float in [-1, 1] (negative=bearish, positive=bullish)
                - trend_label: str ("强看多", "看多", "中性", "看空", "强看空")
                - confidence: float in [0, 1]
        """
        self.model.eval()

        features = self.prepare_features(price_df, sentiment_scores, geo_factors)

        if len(features) < self.lookback_days:
            logger.warning(
                f"Not enough data: {len(features)} days, need {self.lookback_days}"
            )
            return {"trend_score": 0.0, "trend_label": "中性", "confidence": 0.0}

        # Take the last lookback_days
        seq = features[-self.lookback_days:]
        if self._feature_means is not None:
            seq = self._normalize(seq)

        # Convert to tensor
        x = torch.FloatTensor(seq).unsqueeze(0)  # (1, lookback_days, features)

        with torch.no_grad():
            score = self.model(x).item()

        # Map score to label
        abs_score = abs(score)
        if abs_score > 0.6:
            label = "强看多" if score > 0 else "强看空"
        elif abs_score > 0.2:
            label = "看多" if score > 0 else "看空"
        else:
            label = "中性"

        return {
            "trend_score": round(score, 4),
            "trend_label": label,
            "confidence": round(abs_score, 4),
        }

    def train_model(
        self,
        train_data: list,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 32,
    ):
        """Train the model on historical data.

        Args:
            train_data: List of (features_array, target_return) tuples.
                       features_array shape: (lookback_days, num_features)
                       target_return: float, the actual forward return
            epochs: Training epochs
            lr: Learning rate
            batch_size: Batch size
        """
        if not train_data:
            logger.warning("No training data provided")
            return

        # Prepare data
        X = np.array([d[0] for d in train_data])
        y = np.array([d[1] for d in train_data])

        # Compute normalization stats
        self._feature_means = X.reshape(-1, X.shape[-1]).mean(axis=0)
        self._feature_stds = X.reshape(-1, X.shape[-1]).std(axis=0)

        # Normalize
        X = (X - self._feature_means) / (self._feature_stds + 1e-8)

        # Clip targets to [-1, 1]
        y = np.clip(y, -1.0, 1.0)

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        self.model.eval()

    def save(self, path: str):
        """Save model weights and normalization stats."""
        save_dict = {
            "model_state": self.model.state_dict(),
            "feature_means": self._feature_means,
            "feature_stds": self._feature_stds,
        }
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model weights and normalization stats."""
        save_dict = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(save_dict["model_state"])
        self._feature_means = save_dict.get("feature_means")
        self._feature_stds = save_dict.get("feature_stds")
        self.model.eval()
        logger.info(f"Model loaded from {path}")
