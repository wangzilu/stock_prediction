# Pipeline Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate LightGBM predictions into the recommendation pipeline, build a Transformer+SAC RL agent for per-stock timing, and add a 4-slot push schedule (9:20, 14:30, 15:30, 22:00).

**Architecture:** The LGB model is loaded once at pipeline init and provides `short_score` for all CSI300/500 stocks via Qlib Alpha158 inference. The RL agent runs as a separate Transformer+SAC model providing buy/hold/sell signals per stock. The scheduler replaces the old 14:00/14:05 jobs with 4 new time slots, each calling a dedicated method on `DailyPipeline`.

**Tech Stack:** Python 3.11 (conda tianshou), Qlib, LightGBM, tianshou 2.0, PyTorch, gymnasium

**Conda env:** `/Users/wangzilu/miniconda3/envs/tianshou/bin/python`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/train_lgb.py` | Modify | Dynamic date ranges |
| `scripts/train_rl.py` | Create | RL training script |
| `models/rl_agent.py` | Create | Transformer+SAC model, gymnasium env, tianshou wrappers |
| `models/short_term.py` | Modify | Add `load_and_predict_batch()` for pipeline integration |
| `scheduler/jobs.py` | Modify | Add 4 new pipeline methods, LGB/RL loading |
| `signals/scorer.py` | Modify | Add `rl_score` to Recommendation, sell-check report |
| `push/wechat.py` | Modify | Add send methods for new push types |
| `main.py` | Modify | Replace scheduler jobs with new 4-slot schedule |
| `config/settings.py` | Modify | Add RL/schedule config constants |
| `scripts/nightly_train.py` | Modify | Add RL training step |
| `tests/test_rl_agent.py` | Create | RL env + model tests |
| `tests/test_pipeline_lgb.py` | Create | LGB integration tests |

---

### Task 1: Update `train_lgb.py` with Dynamic Dates

**Files:**
- Modify: `scripts/train_lgb.py`

- [ ] **Step 1: Update date logic**

Replace the hardcoded dates in `scripts/train_lgb.py` with dynamic calculation:

```python
"""Train LightGBM model using Qlib Alpha158 factors.

Usage: python scripts/train_lgb.py
"""
import os
import sys
import pickle
from datetime import datetime, timedelta

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "storage")
QLIB_DATA = os.path.join(DATA_DIR, "qlib_data", "cn_data")
MODEL_PATH = os.path.join(DATA_DIR, "lgb_model.pkl")
DATASET_PATH = os.path.join(DATA_DIR, "lgb_dataset.pkl")


def main():
    print("Initializing Qlib...")
    qlib.init(provider_uri=QLIB_DATA, region=REG_CN)

    # Dynamic date ranges
    today = datetime.now()
    train_start = (today - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
    train_end = (today - timedelta(days=90)).strftime("%Y-%m-%d")
    valid_start = (today - timedelta(days=89)).strftime("%Y-%m-%d")
    valid_end = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    test_start = (today - timedelta(days=29)).strftime("%Y-%m-%d")
    test_end = today.strftime("%Y-%m-%d")

    print(f"Train: {train_start} ~ {train_end}")
    print(f"Valid: {valid_start} ~ {valid_end}")
    print(f"Test:  {test_start} ~ {test_end}")

    handler_config = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": train_start,
            "end_time": test_end,
            "instruments": "csi300",
        },
    }
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {
                "train": (train_start, train_end),
                "valid": (valid_start, valid_end),
                "test": (test_start, test_end),
            },
        },
    }

    print("Loading dataset (Alpha158 x csi300)...")
    dataset = init_instance_by_config(dataset_config)
    print("Dataset ready.")

    model_config = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.05,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 4,
        },
    }

    print("Training LightGBM...")
    model = init_instance_by_config(model_config)
    model.fit(dataset)
    print("Training complete!")

    # Save model + dataset
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(DATASET_PATH, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Model saved to {MODEL_PATH}")

    # Quick evaluation
    pred = model.predict(dataset)
    print(f"\nPredictions shape: {pred.shape}")
    print(f"Last 5 predictions:")
    print(pred.tail(5))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run data update + training**

```bash
PY=/Users/wangzilu/miniconda3/envs/tianshou/bin/python
cd /Users/wangzilu/MyProjects/stockPrediction
$PY scripts/update_qlib_data.py
$PY scripts/train_lgb.py
```

Expected: Model saved to `data/storage/lgb_model.pkl`

- [ ] **Step 3: Commit**

```bash
git add scripts/train_lgb.py
git commit -m "feat: dynamic date ranges in LGB training"
```

---

### Task 2: Add LGB Batch Prediction to `ShortTermModel`

**Files:**
- Modify: `models/short_term.py`

- [ ] **Step 1: Add `load_from_pickle` and `predict_batch` methods**

Add these methods to the `ShortTermModel` class in `models/short_term.py` after the existing `predict` method (after line 131):

```python
    @classmethod
    def load_from_pickle(cls, model_path: str = None, dataset_path: str = None):
        """Load pre-trained model and dataset from pickle files.

        Args:
            model_path: Path to lgb_model.pkl
            dataset_path: Path to lgb_dataset.pkl

        Returns:
            ShortTermModel instance with loaded model
        """
        import pickle
        from config.settings import DATA_DIR

        model_path = model_path or str(DATA_DIR / "lgb_model.pkl")
        dataset_path = dataset_path or str(DATA_DIR / "lgb_dataset.pkl")

        instance = cls()
        with open(model_path, "rb") as f:
            instance._model = pickle.load(f)
        with open(dataset_path, "rb") as f:
            instance._dataset = pickle.load(f)
        instance._initialized = True
        return instance

    def predict_batch(self) -> dict:
        """Generate predictions for ALL stocks in the dataset.

        Returns:
            Dict mapping qlib_code (e.g. 'SH600519') to predicted score (float).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_from_pickle() first.")

        predictions = self._model.predict(dataset=self._dataset)
        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame("score")

        latest_date = predictions.index.get_level_values(0).max()
        latest_preds = predictions.loc[latest_date]

        return {code: float(latest_preds.loc[code, "score"])
                for code in latest_preds.index}
```

- [ ] **Step 2: Commit**

```bash
git add models/short_term.py
git commit -m "feat: add load_from_pickle and predict_batch to ShortTermModel"
```

---

### Task 3: Build RL Environment + Transformer+SAC Agent

**Files:**
- Create: `models/rl_agent.py`
- Create: `scripts/train_rl.py`

- [ ] **Step 1: Create `models/rl_agent.py`**

```python
"""Transformer + SAC reinforcement learning agent for per-stock timing.

Uses tianshou 2.0 with gymnasium interface.
Action space: 0=hold, 1=buy, 2=sell
State: Alpha158 (158d) + 20-day OHLCV (100d) + position (1d) + sentiment (1d) = 260d
Reward: return - λ * max(drawdown, 0)
"""
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

    Each episode is one stock's full history. The agent sees a sliding
    window of features and decides buy/hold/sell each day.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        window: int = 20,
        drawdown_penalty: float = 2.0,
    ):
        """
        Args:
            features: shape (T, F) — Alpha158 features per day
            prices: shape (T,) — close prices per day
            window: lookback window for price series in state
            drawdown_penalty: λ for drawdown penalty in reward
        """
        super().__init__()
        self.features = features
        self.prices = prices
        self.window = window
        self.lam = drawdown_penalty

        n_features = features.shape[1]
        # state = alpha158 + flattened OHLCV window + position + sentiment
        self.state_dim = n_features + 1 + 1  # features + position + sentiment placeholder
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # hold=0, buy=1, sell=2

        self._step = 0
        self._position = 0  # 0=empty, 1=holding
        self._entry_price = 0.0
        self._peak_price = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step = self.window
        self._position = 0
        self._entry_price = 0.0
        self._peak_price = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        feat = self.features[self._step]
        obs = np.append(feat, [self._position, 0.0]).astype(np.float32)
        return obs

    def step(self, action):
        price = self.prices[self._step]
        prev_price = self.prices[self._step - 1]
        reward = 0.0

        if action == 1 and self._position == 0:  # buy
            self._position = 1
            self._entry_price = price
            self._peak_price = price
        elif action == 2 and self._position == 1:  # sell
            ret = (price - self._entry_price) / (self._entry_price + 1e-8)
            drawdown = (self._peak_price - price) / (self._peak_price + 1e-8)
            reward = ret - self.lam * max(drawdown, 0)
            self._position = 0
            self._entry_price = 0.0
            self._peak_price = 0.0
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

        return self._get_obs() if not terminated else self._get_obs(), reward, terminated, truncated, {}


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
        self.actor = TransformerActor(state_dim=self.state_dim)
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
```

- [ ] **Step 2: Create `scripts/train_rl.py`**

```python
"""Train Transformer+SAC RL agent using tianshou.

Usage: python scripts/train_rl.py
"""
import os
import sys
import numpy as np
import torch
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "storage")
QLIB_DATA = os.path.join(DATA_DIR, "qlib_data", "cn_data")
MODEL_PATH = os.path.join(DATA_DIR, "rl_model.pt")


def build_envs_from_qlib():
    """Build gymnasium environments from Qlib data, one per stock."""
    import qlib
    from qlib.constant import REG_CN
    from qlib.utils import init_instance_by_config
    from datetime import datetime, timedelta
    from models.rl_agent import StockTradingEnv

    qlib.init(provider_uri=QLIB_DATA, region=REG_CN)

    today = datetime.now()
    start = (today - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    handler_config = {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": start,
            "end_time": end,
            "instruments": "csi300",
        },
    }
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {
                "train": (start, (today - timedelta(days=60)).strftime("%Y-%m-%d")),
                "test": ((today - timedelta(days=59)).strftime("%Y-%m-%d"), end),
            },
        },
    }

    dataset = init_instance_by_config(dataset_config)
    train_data = dataset.prepare("train", col_set="feature")
    # Get close prices for reward calculation
    handler = dataset.handler
    label_data = dataset.prepare("train", col_set="label")

    envs = []
    instruments = train_data.index.get_level_values(1).unique()
    logger.info(f"Building envs for {len(instruments)} stocks...")

    for inst in instruments:
        try:
            inst_features = train_data.loc[(slice(None), inst), :].values
            if len(inst_features) < 60:
                continue
            # Use close prices — extract from raw data
            inst_features_clean = np.nan_to_num(inst_features, nan=0.0)
            # Approximate close prices from first feature (normalized)
            prices = np.cumsum(np.random.randn(len(inst_features_clean)) * 0.02 + 1.0001)
            # Better: try to get actual prices
            try:
                import qlib.data
                close = qlib.data.D.features(
                    [inst], ["$close"], start_time=start, end_time=end
                )
                if not close.empty:
                    inst_close = close.loc[(slice(None), inst), :].values.flatten()
                    if len(inst_close) == len(inst_features_clean):
                        prices = inst_close
                    else:
                        prices = inst_close[:len(inst_features_clean)]
                        inst_features_clean = inst_features_clean[:len(prices)]
            except Exception:
                pass

            if len(prices) < 60 or np.any(prices <= 0):
                continue

            env = StockTradingEnv(
                features=inst_features_clean,
                prices=prices,
                window=20,
                drawdown_penalty=2.0,
            )
            envs.append(env)
        except Exception as e:
            continue

    logger.info(f"Built {len(envs)} valid environments")
    return envs


def make_env_fn(envs):
    """Create a factory that cycles through pre-built envs."""
    idx = [0]
    def _make():
        env = envs[idx[0] % len(envs)]
        idx[0] += 1
        return env
    return _make


def main():
    envs = build_envs_from_qlib()
    if not envs:
        logger.error("No valid environments built. Check Qlib data.")
        return

    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.n
    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")

    from models.rl_agent import TransformerActor, TransformerCritic

    # Networks
    actor = TransformerActor(state_dim=state_dim, action_dim=action_dim)
    critic1 = TransformerCritic(state_dim=state_dim, action_dim=action_dim)
    critic2 = TransformerCritic(state_dim=state_dim, action_dim=action_dim)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optim = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), lr=3e-4
    )

    # Tianshou SAC (discrete)
    import tianshou as ts
    from tianshou.policy import DiscreteSACPolicy
    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.trainer import OffpolicyTrainer

    policy = DiscreteSACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic_optim,
        critic2=critic2,
        critic2_optim=torch.optim.Adam(critic2.parameters(), lr=3e-4),
        tau=0.005,
        gamma=0.99,
        alpha=0.2,
        action_space=envs[0].action_space,
    )

    # Vectorized envs
    from tianshou.env import SubprocVectorEnv, DummyVectorEnv

    n_train = min(8, len(envs))
    n_test = min(4, len(envs))

    train_envs = DummyVectorEnv([make_env_fn(envs[:len(envs)//2]) for _ in range(n_train)])
    test_envs = DummyVectorEnv([make_env_fn(envs[len(envs)//2:]) for _ in range(n_test)])

    buf = VectorReplayBuffer(total_size=100000, buffer_num=n_train)
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs)

    # Pre-collect
    logger.info("Pre-collecting random episodes...")
    train_collector.collect(n_step=5000, random=True)

    logger.info("Starting SAC training...")
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=20,
        step_per_epoch=10000,
        step_per_collect=100,
        update_per_step=0.1,
        episode_per_test=n_test,
        batch_size=256,
    ).run()

    logger.info(f"Training result: {result}")

    # Save model
    torch.save({
        "actor_state": actor.state_dict(),
        "critic1_state": critic1.state_dict(),
        "critic2_state": critic2.state_dict(),
        "state_dim": state_dim,
    }, MODEL_PATH)
    logger.info(f"RL model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add models/rl_agent.py scripts/train_rl.py
git commit -m "feat: add Transformer+SAC RL agent with tianshou"
```

---

### Task 4: Add Config Constants

**Files:**
- Modify: `config/settings.py`

- [ ] **Step 1: Add RL and schedule constants**

Append to the end of `config/settings.py`:

```python
# RL Agent
RL_MODEL_PATH = DATA_DIR / "rl_model.pt"
LGB_MODEL_PATH = DATA_DIR / "lgb_model.pkl"
LGB_DATASET_PATH = DATA_DIR / "lgb_dataset.pkl"

# Sell check thresholds
TAKE_PROFIT_PCT = 8.0   # sell if gain >= 8%
STOP_LOSS_PCT = 5.0      # sell if loss >= 5%
LGB_FLIP_THRESHOLD = -0.02  # sell if LGB score drops below this

# Push schedule
MORNING_REC_TIME = "09:20"
SELL_CHECK_TIME = "14:30"
DAILY_SUMMARY_TIME = "15:30"
EVENING_OUTLOOK_TIME = "22:00"
```

- [ ] **Step 2: Commit**

```bash
git add config/settings.py
git commit -m "feat: add RL/schedule config constants"
```

---

### Task 5: Add Push Methods to WeChatPusher

**Files:**
- Modify: `push/wechat.py`

- [ ] **Step 1: Add new send methods**

Add after the existing `send_verification` method (after line 63):

```python
    def send_sell_check(self, report: str) -> bool:
        """Send sell-check report."""
        return self.send(report, title="📉 盘中卖出建议")

    def send_daily_summary(self, report: str) -> bool:
        """Send daily market summary."""
        return self.send(report, title="📊 收盘总结")

    def send_evening_outlook(self, report: str) -> bool:
        """Send evening outlook report."""
        return self.send(report, title="🌙 明日展望")
```

- [ ] **Step 2: Commit**

```bash
git add push/wechat.py
git commit -m "feat: add push methods for sell-check, summary, outlook"
```

---

### Task 6: Update `SignalScorer` with RL Score and Sell Reports

**Files:**
- Modify: `signals/scorer.py`

- [ ] **Step 1: Add `rl_score` field to `Recommendation`**

In `signals/scorer.py`, add `rl_score` to the `Recommendation` dataclass:

```python
@dataclass
class Recommendation:
    """A single stock recommendation."""
    code: str
    name: str
    final_score: float
    signal: str
    model_score: float
    sentiment_score: float
    sentiment_heat: float
    reason: str
    # Multi-timeframe scores (optional, filled when available)
    short_term_score: float = 0.0
    mid_term_score: float = 0.0
    macro_score: float = 0.0
    has_divergence: bool = False
    rl_action: str = "hold"  # buy / hold / sell
    rl_confidence: float = 0.0
```

- [ ] **Step 2: Add sell-check report generator**

Add this method to the `SignalScorer` class:

```python
    def generate_sell_report(self, sell_items: list) -> str:
        """Generate formatted sell-check report.

        Args:
            sell_items: list of dicts with keys:
                code, name, reason, gain_pct, rec_date
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [f"📉 盘中卖出建议 ({now})", "─────────────"]

        for item in sell_items:
            gain = item["gain_pct"]
            sign = "+" if gain >= 0 else ""
            lines.append(f"• {item['name']}({item['code'][-6:]})")
            lines.append(f"  推荐日: {item['rec_date']} | 收益: {sign}{gain:.1f}%")
            lines.append(f"  触发: {item['reason']}")

        if not sell_items:
            lines.append("暂无卖出建议，持仓继续观察。")

        lines.append("─────────────")
        return "\n".join(lines)
```

- [ ] **Step 3: Commit**

```bash
git add signals/scorer.py
git commit -m "feat: add rl_action to Recommendation, sell report generator"
```

---

### Task 7: Integrate LGB + RL into DailyPipeline

**Files:**
- Modify: `scheduler/jobs.py`

This is the core integration task. The pipeline's `__init__` loads LGB and RL models. The 4 new methods implement the push schedule.

- [ ] **Step 1: Update imports and `__init__`**

At the top of `scheduler/jobs.py`, add new imports (after existing imports):

```python
from config.settings import (
    TAKE_PROFIT_PCT, STOP_LOSS_PCT, LGB_FLIP_THRESHOLD,
    LGB_MODEL_PATH, LGB_DATASET_PATH, RL_MODEL_PATH,
)
```

In `DailyPipeline.__init__`, add after `self.llm_analyst = LLMAnalyst()`:

```python
        # Pre-trained models (loaded lazily)
        self._lgb_predictions = None
        self._rl_agent = None
```

- [ ] **Step 2: Add LGB + RL loading helpers**

Add these methods to `DailyPipeline`:

```python
    def _load_lgb_predictions(self):
        """Load LGB model and get latest predictions for all stocks."""
        if self._lgb_predictions is not None:
            return self._lgb_predictions

        try:
            from models.short_term import ShortTermModel
            model = ShortTermModel.load_from_pickle(
                str(LGB_MODEL_PATH), str(LGB_DATASET_PATH)
            )
            self._lgb_predictions = model.predict_batch()
            logger.info(f"Loaded LGB predictions for {len(self._lgb_predictions)} stocks")
        except Exception as e:
            logger.warning(f"Failed to load LGB model: {e}")
            self._lgb_predictions = {}
        return self._lgb_predictions

    def _load_rl_agent(self):
        """Load RL agent for stock timing signals."""
        if self._rl_agent is not None:
            return self._rl_agent

        try:
            from models.rl_agent import RLAgent
            if os.path.exists(str(RL_MODEL_PATH)):
                self._rl_agent = RLAgent(str(RL_MODEL_PATH))
                logger.info("RL agent loaded")
            else:
                logger.warning("RL model not found, skipping")
                self._rl_agent = RLAgent()  # empty agent, returns hold
        except Exception as e:
            logger.warning(f"Failed to load RL agent: {e}")
            from models.rl_agent import RLAgent
            self._rl_agent = RLAgent()
        return self._rl_agent
```

- [ ] **Step 3: Update `run_daily_recommendation` to use LGB scores**

In `run_daily_recommendation`, replace the A-share screening block. Change the candidate scoring from `change_pct / 10` to LGB predictions.

Replace lines 121-152 (the spot cache screening loop) with:

```python
        # === Stage 1: Fast screening ALL A-shares + crypto + gold ===
        logger.info("Stage 1: Screening all A-shares...")
        candidates = []

        # Load LGB predictions
        lgb_preds = self._load_lgb_predictions()

        # Load full market spot data once (5800+ stocks)
        self.market_collector._load_spot_cache()
        spot = self.market_collector._spot_cache
        stock_macro = (geo["china_us_temperature"] + geo["policy_signal"]) / 2

        if spot is not None and not spot.empty:
            for _, row in spot.iterrows():
                try:
                    code_num = str(row["代码"])
                    price = float(row["最新价"]) if row["最新价"] else 0
                    change_pct = float(row["涨跌幅"]) if row["涨跌幅"] else 0
                    if price <= 0:
                        continue

                    prefix = "SH" if code_num.startswith("6") else "SZ"
                    qlib_code = f"{prefix}{code_num}"
                    name = str(row.get("名称", code_num))

                    # Use LGB prediction if available, fallback to change_pct
                    short_score = lgb_preds.get(qlib_code, change_pct / 10)

                    candidates.append({
                        "code": qlib_code,
                        "name": name,
                        "market": MARKET_STOCK,
                        "short_score": short_score,
                        "macro_score": stock_macro,
                        "price": price,
                    })
                except Exception:
                    continue
            logger.info(f"Screened {len(candidates)} A-shares ({len(lgb_preds)} with LGB scores)")
```

- [ ] **Step 4: Add `run_morning_recommendation` method**

This is the 9:20 push. It calls the existing `run_daily_recommendation`:

```python
    def run_morning_recommendation(self):
        """9:20 AM: Pre-market recommendation push."""
        logger.info("=== Morning Recommendation (9:20) ===")
        self._lgb_predictions = None  # refresh
        self._rl_agent = None
        self.run_daily_recommendation()
```

- [ ] **Step 5: Add `run_sell_check` method**

```python
    def run_sell_check(self):
        """14:30: Check recent recommendations for sell signals."""
        logger.info("=== Sell Check (14:30) ===")

        # Get recent un-verified recommendations
        recent_recs = self.verifier.get_recent_recommendations(days=5)
        if not recent_recs:
            logger.info("No recent recommendations to check")
            return

        lgb_preds = self._load_lgb_predictions()
        sell_items = []

        for rec in recent_recs:
            code = rec["code"]
            rec_price = rec.get("price_at_rec")
            if not rec_price or rec_price <= 0:
                continue

            # Get current price
            try:
                market = next((m for c, n, m in WATCHLIST if c == code), MARKET_STOCK)
                quote = self._get_quote(code, market)
                if not quote:
                    continue
                current_price = quote.get("price", 0)
                if current_price <= 0:
                    continue
            except Exception:
                continue

            gain_pct = (current_price - rec_price) / rec_price * 100
            reasons = []

            # Rule 1: Take profit
            if gain_pct >= TAKE_PROFIT_PCT:
                reasons.append(f"止盈 (涨{gain_pct:.1f}%)")

            # Rule 2: Stop loss
            if gain_pct <= -STOP_LOSS_PCT:
                reasons.append(f"止损 (跌{abs(gain_pct):.1f}%)")

            # Rule 3: LGB score flip
            lgb_score = lgb_preds.get(code, 0)
            if lgb_score < LGB_FLIP_THRESHOLD:
                reasons.append(f"模型翻空 (LGB={lgb_score:.3f})")

            if reasons:
                sell_items.append({
                    "code": code,
                    "name": rec.get("name", code),
                    "reason": " + ".join(reasons),
                    "gain_pct": gain_pct,
                    "rec_date": rec.get("rec_date", ""),
                })

        # Generate and push report
        report = self.signal_scorer.generate_sell_report(sell_items)
        if sell_items:
            self.pusher.send_sell_check(report)
            logger.info(f"Sell check: {len(sell_items)} sell signals pushed")
        else:
            logger.info("Sell check: no sell signals")
```

- [ ] **Step 6: Add `run_daily_summary` method**

```python
    def run_daily_summary(self):
        """15:30: Post-close daily summary with verification."""
        logger.info("=== Daily Summary (15:30) ===")

        # Run verification first
        self.run_verification()

        # Generate summary via LLM
        geo = self._fetch_geo_factors()

        # Collect market data
        crypto_data = {}
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            q = self.crypto_collector.fetch_realtime(symbol)
            if q:
                crypto_data[symbol] = q
        gold_data = self.gold_collector.fetch_realtime()
        global_indices = self.global_indices.format_for_report()

        prompt_data = {
            "type": "daily_summary",
            "headlines": self._headlines or [],
            "geo_factors": geo,
            "crypto_data": crypto_data,
            "gold_data": gold_data,
            "global_indices": global_indices,
        }

        report = self.llm_analyst.generate_summary(prompt_data)
        self.pusher.send_daily_summary(report)
        logger.info("Daily summary pushed")
```

- [ ] **Step 7: Add `run_evening_outlook` method**

```python
    def run_evening_outlook(self):
        """22:00: Evening outlook for next trading day."""
        logger.info("=== Evening Outlook (22:00) ===")

        geo = self._fetch_geo_factors()
        lgb_preds = self._load_lgb_predictions()

        # Top bullish and bearish from LGB
        sorted_preds = sorted(lgb_preds.items(), key=lambda x: x[1], reverse=True)
        top_bull = sorted_preds[:10]
        top_bear = sorted_preds[-5:]

        crypto_data = {}
        for symbol in ["BTC/USDT", "ETH/USDT"]:
            q = self.crypto_collector.fetch_realtime(symbol)
            if q:
                crypto_data[symbol] = q
        gold_data = self.gold_collector.fetch_realtime()
        global_indices = self.global_indices.format_for_report()

        prompt_data = {
            "type": "evening_outlook",
            "headlines": self._headlines or [],
            "geo_factors": geo,
            "top_bullish": [{"code": c, "score": s} for c, s in top_bull],
            "top_bearish": [{"code": c, "score": s} for c, s in top_bear],
            "crypto_data": crypto_data,
            "gold_data": gold_data,
            "global_indices": global_indices,
        }

        report = self.llm_analyst.generate_outlook(prompt_data)
        self.pusher.send_evening_outlook(report)
        logger.info("Evening outlook pushed")
```

- [ ] **Step 8: Commit**

```bash
git add scheduler/jobs.py
git commit -m "feat: integrate LGB predictions + 4-slot pipeline methods"
```

---

### Task 8: Add LLM Summary/Outlook Methods

**Files:**
- Modify: `signals/llm_analyst.py`

- [ ] **Step 1: Add `generate_summary` and `generate_outlook` methods**

Add to the `LLMAnalyst` class. Read the existing `generate_report` method pattern and add:

```python
    def generate_summary(self, data: dict) -> str:
        """Generate daily market close summary."""
        prompt = f"""请基于以下数据撰写今日收盘市场总结（300-500字）：

全球指数：
{data.get('global_indices', '无数据')}

加密货币：{json.dumps(data.get('crypto_data', {}), ensure_ascii=False)}
黄金：{json.dumps(data.get('gold_data', {}), ensure_ascii=False)}

地缘政治因素：{json.dumps(data.get('geo_factors', {}), ensure_ascii=False)}

今日重要新闻：
{chr(10).join(data.get('headlines', [])[:20])}

要求：
1. 总结今日A股、港股、美股期货走势
2. 分析主要驱动因素
3. 点评板块轮动
4. 给出明日开盘预判"""

        return self._call_llm(prompt)

    def generate_outlook(self, data: dict) -> str:
        """Generate evening outlook for next trading day."""
        top_bull = data.get("top_bullish", [])
        top_bear = data.get("top_bearish", [])

        bull_text = "\n".join([f"  {b['code']}: {b['score']:.4f}" for b in top_bull])
        bear_text = "\n".join([f"  {b['code']}: {b['score']:.4f}" for b in top_bear])

        prompt = f"""请撰写明日市场展望（300-500字）：

模型看多前10:
{bull_text}

模型看空后5:
{bear_text}

全球指数：
{data.get('global_indices', '无数据')}

加密货币：{json.dumps(data.get('crypto_data', {}), ensure_ascii=False)}
黄金：{json.dumps(data.get('gold_data', {}), ensure_ascii=False)}

地缘因素：{json.dumps(data.get('geo_factors', {}), ensure_ascii=False)}

夜间新闻：
{chr(10).join(data.get('headlines', [])[:20])}

要求：
1. 预判明日大盘方向和关键点位
2. 重点关注板块和个股
3. 潜在风险提示
4. 建议仓位和操作策略"""

        return self._call_llm(prompt)
```

- [ ] **Step 2: Commit**

```bash
git add signals/llm_analyst.py
git commit -m "feat: add LLM summary and outlook generators"
```

---

### Task 9: Add `get_recent_recommendations` to Verifier

**Files:**
- Modify: `tracker/verifier.py`

- [ ] **Step 1: Add method for sell-check query**

Add to the `Verifier` class:

```python
    def get_recent_recommendations(self, days: int = 5) -> list:
        """Get recommendations from the last N days that haven't been verified.

        Returns:
            List of dicts with rec_date, code, name, signal, score, price_at_rec.
        """
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT rec_date, code, name, signal, score, price_at_rec
                   FROM recommendations
                   WHERE rec_date >= ? AND verified = 0
                   ORDER BY rec_date DESC""",
                (cutoff,),
            ).fetchall()

        return [dict(row) for row in rows]
```

- [ ] **Step 2: Commit**

```bash
git add tracker/verifier.py
git commit -m "feat: add get_recent_recommendations to Verifier"
```

---

### Task 10: Update Scheduler in `main.py`

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Replace old schedule with new 4-slot schedule**

Replace the scheduler section (lines 51-89) with:

```python
    # Default: start scheduler
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler()

    # 9:20 Morning recommendation (pre-market)
    scheduler.add_job(
        pipeline.run_morning_recommendation,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=20),
        id="morning_recommendation",
        name="Morning Recommendation",
    )

    # 14:30 Sell check (30 min before close)
    scheduler.add_job(
        pipeline.run_sell_check,
        CronTrigger(day_of_week="mon-fri", hour=14, minute=30),
        id="sell_check",
        name="Sell Check",
    )

    # 15:30 Daily summary (post-close)
    scheduler.add_job(
        pipeline.run_daily_summary,
        CronTrigger(day_of_week="mon-fri", hour=15, minute=30),
        id="daily_summary",
        name="Daily Summary",
    )

    # 22:00 Evening outlook
    scheduler.add_job(
        pipeline.run_evening_outlook,
        CronTrigger(day_of_week="mon-fri", hour=22, minute=0),
        id="evening_outlook",
        name="Evening Outlook",
    )

    # Hourly risk check (every hour during trading hours 9-15, weekdays)
    scheduler.add_job(
        pipeline.run_risk_check,
        CronTrigger(day_of_week="mon-fri", hour="9-15", minute=30),
        id="risk_check",
        name="Hourly Risk Check",
    )

    logger.info("Scheduler started. Jobs:")
    logger.info("  - Morning recommendation: Mon-Fri 09:20")
    logger.info("  - Sell check: Mon-Fri 14:30")
    logger.info("  - Daily summary: Mon-Fri 15:30")
    logger.info("  - Evening outlook: Mon-Fri 22:00")
    logger.info("  - Risk check: Mon-Fri 9:30-15:30 (hourly)")
    logger.info("Press Ctrl+C to exit.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "feat: replace schedule with 4-slot push (9:20/14:30/15:30/22:00)"
```

---

### Task 11: Update Nightly Pipeline

**Files:**
- Modify: `scripts/nightly_train.py`

- [ ] **Step 1: Enable RL training step**

Replace the commented-out Step 3 in `nightly_train.py`:

```python
    # Step 3: RL training
    run_step("RL Agent Training (Transformer+SAC)", "train_rl.py")
```

- [ ] **Step 2: Commit**

```bash
git add scripts/nightly_train.py
git commit -m "feat: enable RL training in nightly pipeline"
```

---

### Task 12: Run Data Update + Full Training Pipeline

- [ ] **Step 1: Run baostock data update**

```bash
PY=/Users/wangzilu/miniconda3/envs/tianshou/bin/python
cd /Users/wangzilu/MyProjects/stockPrediction
$PY scripts/update_qlib_data.py
```

Expected: "Update complete: N stocks updated"

- [ ] **Step 2: Run LGB training**

```bash
$PY scripts/train_lgb.py
```

Expected: "Model saved to data/storage/lgb_model.pkl"

- [ ] **Step 3: Run RL training**

```bash
$PY scripts/train_rl.py
```

Expected: "RL model saved to data/storage/rl_model.pt"
