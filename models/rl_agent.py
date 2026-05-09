"""Transformer + SAC reinforcement learning agent for per-stock timing.

Uses tianshou 2.0 with gymnasium interface.
Action space: 0=hold, 1=buy, 2=sell
State: Alpha158 features + qlib_score + sentiment + market regime + position
Reward: return - λ * max(drawdown, 0) - transaction costs
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import logging

logger = logging.getLogger(__name__)

# ── Gymnasium Environment ──────────────────────────────────────────────

class StockTradingEnv(gym.Env):
    """Single-stock trading environment.

    Each episode is one stock's full history. The agent sees
    features and decides buy/hold/sell each day.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        qlib_scores: np.ndarray | None = None,
        sentiment_scores: np.ndarray | None = None,
        sentiment_heat: np.ndarray | None = None,
        market_regime: np.ndarray | None = None,
        window: int = 20,
        drawdown_penalty: float = 2.0,
        transaction_cost: float = 0.001,
    ):
        """
        Args:
            features: shape (T, F) — Alpha158 features per day
            prices: shape (T,) — close prices per day
            qlib_scores: shape (T,) — current short-term model score per day
            sentiment_scores: shape (T,) — sentiment score [-1, 1]
            sentiment_heat: shape (T,) — sentiment intensity [0, 1]
            market_regime: shape (T,) — broad market/risk state proxy
            window: lookback window start offset
            drawdown_penalty: λ for drawdown penalty in reward
            transaction_cost: one-way trading cost deducted on buy/sell
        """
        super().__init__()
        self.features = np.nan_to_num(
            np.asarray(features, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.prices = np.nan_to_num(
            np.asarray(prices, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.qlib_scores = self._align_optional(qlib_scores, default=0.0)
        self.sentiment_scores = self._align_optional(sentiment_scores, default=0.0)
        self.sentiment_heat = self._align_optional(sentiment_heat, default=0.0)
        self.market_regime = self._align_optional(market_regime, default=0.0)
        self.window = window
        self.lam = drawdown_penalty
        self.transaction_cost = transaction_cost

        n_features = features.shape[1]
        # state = alpha158 + qlib + sentiment_score + sentiment_heat
        #       + market_regime + position + unrealized_return
        self.state_dim = n_features + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # hold=0, buy=1, sell=2

        self._step = 0
        self._position = 0  # 0=empty, 1=holding
        self._entry_price = 0.0
        self._peak_price = 0.0

    def _align_optional(self, values, default: float) -> np.ndarray:
        if values is None:
            return np.full(len(self.prices), default, dtype=np.float32)
        arr = np.nan_to_num(
            np.asarray(values, dtype=np.float32),
            nan=default,
            posinf=default,
            neginf=default,
        )
        if len(arr) < len(self.prices):
            padded = np.full(len(self.prices), default, dtype=np.float32)
            padded[:len(arr)] = arr
            return padded
        return arr[:len(self.prices)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step = self.window
        self._position = 0
        self._entry_price = 0.0
        self._peak_price = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        feat = self.features[self._step]
        price = self.prices[self._step]
        unrealized = 0.0
        if self._position == 1 and self._entry_price > 0:
            unrealized = (price - self._entry_price) / (self._entry_price + 1e-8)
        obs = np.append(
            feat,
            [
                self.qlib_scores[self._step],
                self.sentiment_scores[self._step],
                self.sentiment_heat[self._step],
                self.market_regime[self._step],
                self._position,
                unrealized,
            ],
        ).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def step(self, action):
        price = self.prices[self._step]
        prev_price = self.prices[self._step - 1]
        reward = 0.0

        if action == 1 and self._position == 0:  # buy
            self._position = 1
            self._entry_price = price
            self._peak_price = price
            reward -= self.transaction_cost
        elif action == 2 and self._position == 1:  # sell
            ret = (price - self._entry_price) / (self._entry_price + 1e-8)
            drawdown = (self._peak_price - price) / (self._peak_price + 1e-8)
            reward = ret - self.lam * max(drawdown, 0) - self.transaction_cost
            self._position = 0
            self._entry_price = 0.0
            self._peak_price = 0.0
        elif action == 1 and self._position == 1:
            reward -= self.transaction_cost * 0.25
        elif action == 2 and self._position == 0:
            reward -= self.transaction_cost * 0.25
        elif self._position == 1:  # holding
            daily_ret = (price - prev_price) / (prev_price + 1e-8)
            reward = daily_ret
            self._peak_price = max(self._peak_price, price)

        self._step += 1
        terminated = self._step >= len(self.prices) - 1
        truncated = False

        # Force sell at episode end
        if terminated and self._position == 1:
            ret = (self.prices[self._step] - self._entry_price) / (self._entry_price + 1e-8)
            drawdown = (self._peak_price - self.prices[self._step]) / (self._peak_price + 1e-8)
            reward += ret - self.lam * max(drawdown, 0)
            self._position = 0

        obs = self._get_obs()
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0))
        reward = float(np.clip(reward, -1.0, 1.0))
        return obs, reward, terminated, truncated, {}


# ── Transformer Network ───────────────────────────────────────────────

class TransformerActor(nn.Module):
    """Transformer encoder → discrete action logits for SAC."""

    def __init__(self, state_dim: int, action_dim: int = 3, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, obs, state=None, info=None):
        """Forward pass. obs: (batch, state_dim)"""
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        x = self.input_proj(obs).unsqueeze(1)  # (batch, 1, d_model)
        x = self.transformer(x)
        x = x.squeeze(1)  # (batch, d_model)
        logits = self.head(x)
        return logits, state


class TransformerCritic(nn.Module):
    """Transformer encoder → Q-value for SAC."""

    def __init__(self, state_dim: int, action_dim: int = 3, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, obs, state=None, info=None):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        x = self.input_proj(obs).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.head(x)


# ── Model Wrapper ─────────────────────────────────────────────────────

class RLAgent:
    """Wrapper for the trained RL agent for inference."""

    def __init__(self, model_path: str = None):
        self.state_dim = 160  # will be set on load
        self.actor = None
        if model_path:
            self.load(model_path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.state_dim = checkpoint["state_dim"]
        actor_kwargs = checkpoint.get("actor_kwargs", {})
        action_dim = int(checkpoint.get("action_dim", 3))
        self.actor = TransformerActor(
            state_dim=self.state_dim,
            action_dim=action_dim,
            **actor_kwargs,
        )
        self.actor.load_state_dict(checkpoint["actor_state"])
        self.actor.eval()
        logger.info(f"RL agent loaded from {path}")

    def predict(self, state: np.ndarray) -> dict:
        """Predict action for a single state.

        Returns:
            Dict with action (0=hold, 1=buy, 2=sell),
            action_label, and confidence.
        """
        if self.actor is None:
            return {"action": 0, "action_label": "hold", "confidence": 0.0}

        with torch.no_grad():
            obs = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, _ = self.actor(obs)
            probs = torch.softmax(logits, dim=-1).squeeze()

        action = int(probs.argmax())
        labels = {0: "hold", 1: "buy", 2: "sell"}
        return {
            "action": action,
            "action_label": labels[action],
            "confidence": float(probs[action]),
        }
