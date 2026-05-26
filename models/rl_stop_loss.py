"""Phase 5C: DQN-based adaptive stop-loss agent.

Learns when to exit positions based on market conditions,
replacing fixed ATR stop-loss rules in RiskGuard.

Actions: 0=hold, 1=exit
State (8 features): unrealized_pnl_pct, holding_days, volatility_20d,
    momentum_5d, volume_ratio, regime_risk, max_profit_pct, drawdown_from_peak

Usage:
    from models.rl_stop_loss import StopLossEnv, train_stop_loss_agent, should_exit
    agent = train_stop_loss_agent(episodes=500)
    exit_now = should_exit(agent, state_dict)
"""
from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "data" / "models"

# ── Constants ────────────────────────────────────────────────────────────

STATE_DIM = 8
N_ACTIONS = 2  # 0=hold, 1=exit
HARD_STOP_PCT = -0.20
TRANSACTION_COST = 0.0015  # one-way cost (stamp + commission)


# ── Gymnasium-style Environment ──────────────────────────────────────────

class StopLossEnv:
    """Stop-loss decision environment for a single position.

    The agent observes 8 market/position features each step and decides
    whether to hold (0) or exit (1). An episode represents one position
    lifecycle from entry to exit.

    Can run in two modes:
      1. Synthetic mode (default): generates random trajectories for training
      2. Replay mode: pass daily_returns array to replay a real position
    """

    def __init__(
        self,
        daily_returns: np.ndarray | None = None,
        max_holding_days: int = 60,
        hard_stop_pct: float = HARD_STOP_PCT,
        transaction_cost: float = TRANSACTION_COST,
    ):
        self.daily_returns = daily_returns
        self.max_holding_days = max_holding_days
        self.hard_stop_pct = hard_stop_pct
        self.transaction_cost = transaction_cost

        # State space info
        self.state_dim = STATE_DIM
        self.n_actions = N_ACTIONS

        # Episode state
        self._step = 0
        self._cum_return = 0.0
        self._max_profit = 0.0
        self._returns_buffer: list[float] = []
        self._volatility = 0.0
        self._momentum = 0.0
        self._volume_ratio = 1.0
        self._regime_risk = 0.0
        self._done = False

    def reset(self, daily_returns: np.ndarray | None = None) -> np.ndarray:
        """Reset environment for a new episode.

        Args:
            daily_returns: Optional array of daily returns for replay mode.
                If None and self.daily_returns is None, generates synthetic data.
        """
        if daily_returns is not None:
            self.daily_returns = daily_returns

        if self.daily_returns is None:
            self.daily_returns = self._generate_synthetic_returns()

        self._step = 0
        self._cum_return = 0.0
        self._max_profit = 0.0
        self._returns_buffer = []
        self._done = False

        # Randomize initial market context for training diversity
        self._volatility = abs(np.random.normal(0.02, 0.01))
        self._momentum = np.random.normal(0.0, 0.03)
        self._volume_ratio = max(0.3, np.random.lognormal(0.0, 0.3))
        self._regime_risk = np.random.beta(2, 5)  # skewed toward low risk

        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Take action and return (next_state, reward, done, info).

        Args:
            action: 0=hold, 1=exit

        Returns:
            Tuple of (state, reward, done, info)
        """
        if self._done:
            return self._get_state(), 0.0, True, {"reason": "already_done"}

        info: dict = {}
        reward = 0.0

        if action == 1:
            # Exit position
            reward = self._cum_return - self.transaction_cost
            self._done = True
            info["reason"] = "agent_exit"
            info["exit_return"] = self._cum_return
            return self._get_state(), reward, True, info

        # Action = 0: hold — advance one day
        if self._step >= len(self.daily_returns):
            # Forced exit at end of data
            reward = self._cum_return - self.transaction_cost
            self._done = True
            info["reason"] = "max_days"
            info["exit_return"] = self._cum_return
            return self._get_state(), reward, True, info

        daily_ret = float(self.daily_returns[self._step])
        self._returns_buffer.append(daily_ret)
        self._cum_return = (1 + self._cum_return) * (1 + daily_ret) - 1
        self._max_profit = max(self._max_profit, self._cum_return)
        self._step += 1

        # Update rolling features
        if len(self._returns_buffer) >= 5:
            self._momentum = sum(self._returns_buffer[-5:])
        if len(self._returns_buffer) >= 10:
            self._volatility = float(np.std(self._returns_buffer[-20:]))

        # Slowly drift volume ratio and regime
        self._volume_ratio *= np.random.lognormal(0.0, 0.05)
        self._volume_ratio = np.clip(self._volume_ratio, 0.2, 5.0)
        self._regime_risk = np.clip(
            self._regime_risk + np.random.normal(0.0, 0.02), 0.0, 1.0
        )

        # Check hard stop
        if self._cum_return <= self.hard_stop_pct:
            penalty = -0.10  # extra penalty for hitting hard stop
            reward = self._cum_return + penalty - self.transaction_cost
            self._done = True
            info["reason"] = "hard_stop"
            info["exit_return"] = self._cum_return
            return self._get_state(), reward, True, info

        # Check max holding days
        if self._step >= self.max_holding_days:
            reward = self._cum_return - self.transaction_cost
            self._done = True
            info["reason"] = "max_holding"
            info["exit_return"] = self._cum_return
            return self._get_state(), reward, True, info

        # Hold reward: small step penalty to encourage decisive action
        reward = 0.0

        return self._get_state(), reward, False, info

    def _get_state(self) -> np.ndarray:
        """Build 8-dimensional state vector."""
        drawdown_from_peak = (
            self._cum_return - self._max_profit
            if self._max_profit > 0
            else min(0.0, self._cum_return)
        )

        state = np.array([
            self._cum_return,                                   # unrealized_pnl_pct
            self._step / self.max_holding_days,                 # holding_days (normalized)
            self._volatility,                                   # volatility_20d
            self._momentum,                                     # momentum_5d
            self._volume_ratio - 1.0,                           # volume_ratio (centered)
            self._regime_risk,                                  # regime_risk
            self._max_profit,                                   # max_profit_pct
            drawdown_from_peak,                                 # drawdown_from_peak
        ], dtype=np.float32)

        return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

    @staticmethod
    def _generate_synthetic_returns(n_days: int = 60) -> np.ndarray:
        """Generate realistic synthetic daily returns for training.

        Mixes several market regimes: trending up, trending down,
        mean-reverting, and high-volatility.
        """
        regime = random.choice(["bull", "bear", "choppy", "crash"])

        if regime == "bull":
            mu, sigma = 0.002, 0.015
        elif regime == "bear":
            mu, sigma = -0.003, 0.02
        elif regime == "choppy":
            mu, sigma = 0.0, 0.025
        else:  # crash
            mu, sigma = -0.008, 0.035

        returns = np.random.normal(mu, sigma, n_days).astype(np.float32)

        # Add occasional jumps
        n_jumps = np.random.poisson(2)
        for _ in range(n_jumps):
            idx = np.random.randint(0, n_days)
            returns[idx] += np.random.normal(0, 0.05)

        return returns


# ── DQN Network ──────────────────────────────────────────────────────────

class DQN(nn.Module):
    """Simple 2-layer MLP for Q-value estimation."""

    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS,
                 hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay Buffer ────────────────────────────────────────────────────────

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Simple circular replay buffer."""

    def __init__(self, capacity: int = 50_000):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ────────────────────────────────────────────────────────────

class StopLossAgent:
    """DQN agent for stop-loss decisions."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        n_actions: int = N_ACTIONS,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 3000,
        target_update: int = 100,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.policy_net = DQN(state_dim, n_actions).to(self.device)
        self.target_net = DQN(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.steps_done = 0

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action selection."""
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay,
        )

        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def train_step(self) -> float:
        """One gradient step on a batch from replay buffer.

        Returns:
            Loss value, or 0.0 if buffer too small.
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.sample(self.batch_size)

        states = torch.as_tensor(
            np.array([t.state for t in batch]), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            [t.action for t in batch], dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            np.array([t.next_state for t in batch]), dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            [t.done for t in batch], dtype=torch.float32, device=self.device
        )

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            path = str(MODEL_DIR / "rl_stop_loss_dqn.pt")
        torch.save({
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "steps_done": self.steps_done,
            "epsilon": self.epsilon,
        }, path)
        logger.info(f"StopLossAgent saved to {path}")

    def load(self, path: Optional[str] = None):
        """Load model checkpoint."""
        if path is None:
            path = str(MODEL_DIR / "rl_stop_loss_dqn.pt")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_state"])
        self.target_net.load_state_dict(checkpoint["target_state"])
        self.steps_done = checkpoint.get("steps_done", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.policy_net.eval()
        logger.info(f"StopLossAgent loaded from {path}")


# ── Training ─────────────────────────────────────────────────────────────

def train_stop_loss_agent(
    episodes: int = 2000,
    max_steps_per_episode: int = 60,
    lr: float = 1e-3,
    gamma: float = 0.99,
    log_interval: int = 200,
    save_path: Optional[str] = None,
) -> StopLossAgent:
    """Train a DQN stop-loss agent on synthetic position trajectories.

    Args:
        episodes: Number of training episodes.
        max_steps_per_episode: Max days per episode.
        lr: Learning rate.
        gamma: Discount factor.
        log_interval: Print stats every N episodes.
        save_path: Where to save the trained model.

    Returns:
        Trained StopLossAgent.
    """
    agent = StopLossAgent(lr=lr, gamma=gamma)
    env = StopLossEnv(max_holding_days=max_steps_per_episode)

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    losses: list[float] = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.buffer.push(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss > 0:
                losses.append(loss)

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

        if (ep + 1) % log_interval == 0:
            avg_ret = np.mean(episode_returns[-log_interval:])
            avg_len = np.mean(episode_lengths[-log_interval:])
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            logger.info(
                f"Episode {ep+1}/{episodes} | "
                f"avg_return={avg_ret:.4f} | avg_steps={avg_len:.1f} | "
                f"loss={avg_loss:.4f} | epsilon={agent.epsilon:.3f}"
            )
            print(
                f"  [Episode {ep+1:>5}] return={avg_ret:+.4f}  "
                f"steps={avg_len:.1f}  loss={avg_loss:.4f}  "
                f"eps={agent.epsilon:.3f}"
            )

    agent.save(save_path)
    return agent


# ── Inference ────────────────────────────────────────────────────────────

def should_exit(agent: StopLossAgent, state: dict | np.ndarray) -> bool:
    """Convenience function: should we exit this position?

    Args:
        agent: Trained StopLossAgent.
        state: Either a dict with keys matching STATE features,
               or a raw numpy array of shape (8,).

    Returns:
        True if agent recommends exit.
    """
    if isinstance(state, dict):
        state_arr = np.array([
            state.get("unrealized_pnl_pct", 0.0),
            state.get("holding_days", 0.0),
            state.get("volatility_20d", 0.02),
            state.get("momentum_5d", 0.0),
            state.get("volume_ratio", 0.0),
            state.get("regime_risk", 0.0),
            state.get("max_profit_pct", 0.0),
            state.get("drawdown_from_peak", 0.0),
        ], dtype=np.float32)
    else:
        state_arr = np.asarray(state, dtype=np.float32)

    action = agent.select_action(state_arr, greedy=True)
    return action == 1


# ── Comparison with Fixed Rules ──────────────────────────────────────────

def compare_with_fixed_stop(
    agent: StopLossAgent,
    n_episodes: int = 500,
    fixed_stops: tuple[float, ...] = (-0.10, -0.15, -0.20),
) -> dict:
    """Compare RL agent vs fixed stop-loss thresholds.

    Runs the same set of synthetic trajectories through:
      1. RL agent (greedy)
      2. Fixed stop at each threshold level

    Args:
        agent: Trained StopLossAgent.
        n_episodes: Number of test trajectories.
        fixed_stops: Tuple of fixed stop-loss thresholds to compare.

    Returns:
        Dict with metrics for each strategy.
    """
    env = StopLossEnv()
    results: dict[str, list[dict]] = {f"fixed_{s:.0%}": [] for s in fixed_stops}
    results["rl_agent"] = []

    for _ in range(n_episodes):
        # Generate one trajectory
        returns = StopLossEnv._generate_synthetic_returns(60)

        # --- RL agent ---
        state = env.reset(daily_returns=returns.copy())
        while True:
            action = agent.select_action(state, greedy=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                results["rl_agent"].append({
                    "exit_return": info.get("exit_return", env._cum_return),
                    "reason": info.get("reason", "unknown"),
                    "holding_days": env._step,
                })
                break

        # --- Fixed stops ---
        for stop_pct in fixed_stops:
            cum_ret = 0.0
            exited = False
            for day_i, r in enumerate(returns):
                cum_ret = (1 + cum_ret) * (1 + r) - 1
                if cum_ret <= stop_pct:
                    results[f"fixed_{stop_pct:.0%}"].append({
                        "exit_return": cum_ret,
                        "reason": "stop_hit",
                        "holding_days": day_i + 1,
                    })
                    exited = True
                    break
            if not exited:
                # Held to end
                results[f"fixed_{stop_pct:.0%}"].append({
                    "exit_return": cum_ret,
                    "reason": "held_to_end",
                    "holding_days": len(returns),
                })

    # Compute summary metrics
    summary = {}
    for strategy, trades in results.items():
        if not trades:
            continue
        exit_rets = [t["exit_return"] for t in trades]
        summary[strategy] = {
            "avg_return": float(np.mean(exit_rets)),
            "median_return": float(np.median(exit_rets)),
            "win_rate": float(np.mean([1 if r > 0 else 0 for r in exit_rets])),
            "max_loss": float(np.min(exit_rets)),
            "avg_holding_days": float(np.mean([t["holding_days"] for t in trades])),
            "stop_hit_pct": float(
                np.mean([1 if t["reason"] in ("stop_hit", "hard_stop") else 0 for t in trades])
            ),
            "n_trades": len(trades),
        }

    return summary


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 60)
    print("Phase 5C: Training RL Stop-Loss Agent (DQN)")
    print("=" * 60)

    agent = train_stop_loss_agent(episodes=2000, log_interval=500)

    print("\n" + "=" * 60)
    print("Comparing RL agent vs fixed stop-loss rules")
    print("=" * 60)

    comparison = compare_with_fixed_stop(agent, n_episodes=1000)

    for strategy, metrics in sorted(comparison.items()):
        print(f"\n  {strategy}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k:>20s}: {v:+.4f}" if "return" in k or "loss" in k
                      else f"    {k:>20s}: {v:.4f}")
            else:
                print(f"    {k:>20s}: {v}")
