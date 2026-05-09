"""Train Transformer+SAC RL agent using tianshou.

The RL agent trains offline on historical data. It is evaluated daily
but not deployed to the recommendation pipeline until metrics mature.

Usage: python scripts/train_rl.py
"""
from __future__ import annotations

import os
import sys
import argparse
import json
import random
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import logging
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "storage")
QLIB_DATA = os.path.join(DATA_DIR, "qlib_data", "cn_data")
MODEL_PATH = os.path.join(DATA_DIR, "rl_model.pt")
METRICS_PATH = os.path.join(DATA_DIR, "rl_metrics.json")
LGB_MODEL_PATH = os.path.join(DATA_DIR, "lgb_model.pkl")
try:
    from config.settings import LGB_PREDICTION_CACHE_PATH
except Exception:
    LGB_PREDICTION_CACHE_PATH = os.path.join(DATA_DIR, "lgb_latest_predictions.json")


STATE_COMPONENTS = [
    "alpha158_features",
    "qlib_score",
    "sentiment_score",
    "sentiment_heat",
    "market_regime",
    "position",
    "unrealized_return",
]


def _prediction_score_series(predictions) -> pd.Series:
    if isinstance(predictions, pd.Series):
        series = predictions
    elif isinstance(predictions, pd.DataFrame):
        if "score" in predictions.columns:
            series = predictions["score"]
        elif len(predictions.columns) == 1:
            series = predictions.iloc[:, 0]
        else:
            numeric_cols = [
                col for col in predictions.columns
                if pd.api.types.is_numeric_dtype(predictions[col])
            ]
            if len(numeric_cols) != 1:
                raise RuntimeError("LGB prediction output does not contain one score column")
            series = predictions[numeric_cols[0]]
    else:
        raise RuntimeError(f"Unsupported prediction output: {type(predictions).__name__}")
    return pd.to_numeric(series, errors="coerce").astype("float32")


def _finite_number(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _contains_nonfinite(value) -> bool:
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, np.number)):
        return not _finite_number(value)
    return False


def _sanitize_json_value(value):
    if isinstance(value, dict):
        return {key: _sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.number)):
        return float(value) if _finite_number(value) else None
    return value


def _metrics_are_valid(metrics: dict) -> bool:
    required = ("mean_reward", "std_reward", "mean_length", "mean_loss")
    return all(_finite_number(metrics.get(key)) for key in required)


def _module_has_finite_params(module: torch.nn.Module) -> bool:
    return all(torch.isfinite(param).all().item() for param in module.parameters())


def append_metrics(metrics: dict) -> None:
    """Append metrics while keeping the JSON file standards-compliant."""
    history = []
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            history = json.load(f)
    for item in history:
        if isinstance(item, dict) and _contains_nonfinite(item):
            item.setdefault("valid", False)
            item.setdefault("invalid_reason", "non_finite_metric")
    history.append(metrics)
    with open(METRICS_PATH, "w") as f:
        json.dump(_sanitize_json_value(history), f, indent=2, allow_nan=False)


def load_lgb_prediction_series(dataset, enabled: bool = True) -> pd.Series | None:
    """Load the current LGB artifact and score the RL training dataset."""
    if not enabled:
        logger.warning("LGB score state disabled by CLI; RL qlib_score state will be neutral")
        return None
    logger.warning(
        "Historical LGB score state is disabled for tonight's RL run; "
        "use --lgb-score-mode latest to consume the validated cache"
    )
    return None


def load_latest_lgb_score_map(enabled: bool = True) -> dict[str, float]:
    """Load validated production LGB cache and broadcast by instrument in RL envs."""
    if not enabled:
        logger.warning("LGB score state disabled by CLI; RL qlib_score state will be neutral")
        return {}
    try:
        from models.lgb_cache import load_prediction_cache

        logger.info("Loading validated LGB score cache for RL state: %s", LGB_PREDICTION_CACHE_PATH)
        score_map, payload = load_prediction_cache(LGB_PREDICTION_CACHE_PATH)
        logger.info(
            "Loaded LGB score cache for RL: %s instruments, latest_date=%s",
            len(score_map),
            payload.get("latest_date", ""),
        )
        return score_map
    except Exception as exc:
        logger.warning("Failed to load LGB score cache; using neutral qlib_score: %s", exc)
        return {}


def _instrument_scores(scores: pd.Series | None, inst, length: int) -> np.ndarray:
    if scores is None:
        return np.zeros(length, dtype=np.float32)
    try:
        inst_scores = _instrument_slice(scores, inst)
        return np.nan_to_num(inst_scores.to_numpy(dtype=np.float32), nan=0.0)[:length]
    except Exception:
        return np.zeros(length, dtype=np.float32)


def _market_regime_scores(scores: pd.Series | None, dates, length: int) -> np.ndarray:
    if scores is None or not isinstance(scores.index, pd.MultiIndex):
        return np.zeros(length, dtype=np.float32)
    try:
        by_date = scores.groupby(level=0).mean()
        aligned = pd.Series(index=dates, dtype="float32")
        values = by_date.reindex(dates).fillna(0.0).to_numpy(dtype=np.float32)
        aligned.iloc[:len(values)] = values
        return np.nan_to_num(aligned.to_numpy(dtype=np.float32), nan=0.0)[:length]
    except Exception:
        return np.zeros(length, dtype=np.float32)


def _instrument_slice(frame: pd.DataFrame | pd.Series, inst):
    index = frame.index
    if isinstance(index, pd.MultiIndex) and "instrument" in index.names:
        return frame.xs(inst, level="instrument", drop_level=False)
    return frame.loc[(slice(None), inst)]


def _datetime_values(index) -> pd.Index:
    if isinstance(index, pd.MultiIndex) and "datetime" in index.names:
        return index.get_level_values("datetime")
    if isinstance(index, pd.MultiIndex):
        return index.get_level_values(0)
    return index


def _qlib_code_from_instrument(inst: str) -> str:
    text = str(inst).upper()
    if text.startswith("SH") or text.startswith("SZ"):
        return text
    if text.startswith("SH") or text.startswith("SZ"):
        return text
    return text


def build_envs_from_qlib(
    max_envs: int = 0,
    lgb_score_mode: str = "latest",
):
    """Build gymnasium environments from Qlib data, one per stock."""
    import qlib
    from qlib.constant import REG_CN
    from qlib.utils import init_instance_by_config
    from qlib.data import D
    from datetime import datetime, timedelta
    from models.rl_agent import StockTradingEnv

    qlib.init(provider_uri=QLIB_DATA, region=REG_CN)

    today = datetime.now()
    start = (today - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    end = (today - timedelta(days=60)).strftime("%Y-%m-%d")

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
                "train": (start, end),
            },
        },
    }

    dataset = init_instance_by_config(dataset_config)
    train_data = dataset.prepare("train", col_set="feature")
    lgb_scores = None
    lgb_score_map: dict[str, float] = {}
    if lgb_score_mode == "historical":
        lgb_scores = load_lgb_prediction_series(dataset, enabled=True)
    elif lgb_score_mode == "latest":
        lgb_score_map = load_latest_lgb_score_map(enabled=True)
    else:
        logger.warning("LGB score state disabled by CLI; RL qlib_score state will be neutral")

    envs = []
    instruments = train_data.index.get_level_values(1).unique()
    logger.info(f"Building envs for {len(instruments)} stocks...")
    logger.info("Historical sentiment store not found; using neutral sentiment states for tonight's RL training")

    for inst in instruments:
        try:
            inst_frame = _instrument_slice(train_data, inst)
            inst_features = inst_frame.values
            if len(inst_features) < 60:
                continue
            inst_features_clean = np.nan_to_num(
                inst_features,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).astype(np.float32)
            inst_features_clean = np.clip(inst_features_clean, -20.0, 20.0)
            dates = _datetime_values(inst_frame.index)

            # Get actual close prices
            try:
                close = D.features([inst], ["$close"], start_time=start, end_time=end)
                if close.empty:
                    continue
                inst_close = _instrument_slice(close, inst).values.flatten()
                # Align lengths
                min_len = min(len(inst_close), len(inst_features_clean))
                prices = np.nan_to_num(
                    inst_close[:min_len],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).astype(np.float32)
                inst_features_clean = inst_features_clean[:min_len]
                dates = dates[:min_len]
            except Exception:
                continue

            if len(prices) < 60 or np.any(prices <= 0) or not np.all(np.isfinite(prices)):
                continue

            if lgb_score_map:
                qlib_state = np.full(
                    len(prices),
                    lgb_score_map.get(_qlib_code_from_instrument(inst), 0.0),
                    dtype=np.float32,
                )
            else:
                qlib_state = _instrument_scores(lgb_scores, inst, len(prices))
            market_state = _market_regime_scores(lgb_scores, dates, len(prices))
            sentiment_state = np.zeros(len(prices), dtype=np.float32)
            heat_state = np.zeros(len(prices), dtype=np.float32)

            env = StockTradingEnv(
                features=inst_features_clean,
                prices=prices,
                qlib_scores=qlib_state,
                sentiment_scores=sentiment_state,
                sentiment_heat=heat_state,
                market_regime=market_state,
                window=20,
                drawdown_penalty=2.0,
            )
            envs.append(env)
            if max_envs and len(envs) >= max_envs:
                break
        except Exception:
            continue

    logger.info(f"Built {len(envs)} valid environments")
    return envs


def make_env_fn(envs, start_index: int = 0):
    """Create a factory that returns deep copies of pre-built envs."""
    import copy
    idx = [start_index]
    def _make():
        env = copy.deepcopy(envs[idx[0] % len(envs)])
        idx[0] += 1
        return env
    return _make


def evaluate_model(policy, test_envs, n_episodes=20):
    """Evaluate the RL model and return metrics."""
    from tianshou.data import Collector
    collector = Collector(policy, test_envs)
    result = collector.collect(n_episode=n_episodes)
    return {
        "mean_reward": float(np.mean(result["rews"])),
        "std_reward": float(np.std(result["rews"])),
        "mean_length": float(np.mean(result["lens"])),
    }


def evaluate_dqn(q_net, envs, n_episodes: int) -> dict:
    """Evaluate a greedy DQN-style policy on plain gym envs."""
    rewards = []
    lengths = []
    q_net.eval()
    for env in envs[:n_episodes]:
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        length = 0
        while not done:
            with torch.no_grad():
                obs_arr = np.nan_to_num(
                    np.asarray(obs, dtype=np.float32),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                logits, _ = q_net(torch.as_tensor(obs_arr, dtype=torch.float32).unsqueeze(0))
                action = 0 if not torch.isfinite(logits).all() else int(torch.argmax(logits, dim=-1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            reward = float(np.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0))
            total_reward += float(np.clip(reward, -1.0, 1.0))
            length += 1
            done = terminated or truncated
        rewards.append(total_reward)
        lengths.append(length)
    if not rewards:
        return {"mean_reward": 0.0, "std_reward": 0.0, "mean_length": 0.0}
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
    }


def train_dqn(envs, state_dim: int, action_dim: int, args):
    """Train a lightweight offline/online DQN execution layer."""
    from models.rl_agent import TransformerActor

    split = max(1, len(envs) * 3 // 4)
    train_envs = envs[:split]
    test_envs = envs[split:] if split < len(envs) else envs[: min(4, len(envs))]

    q_net = TransformerActor(
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=0.1,
    )
    target_net = TransformerActor(
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=0.1,
    )
    target_net.load_state_dict(q_net.state_dict())
    optim = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    replay = deque(maxlen=args.replay_size)

    live_envs = []
    for env in train_envs:
        obs, _ = env.reset()
        live_envs.append([env, obs])

    def choose_action(obs, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(action_dim)
        obs = np.nan_to_num(np.asarray(obs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            logits, _ = q_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            if not torch.isfinite(logits).all():
                return random.randrange(action_dim)
            return int(torch.argmax(logits, dim=-1).item())

    def step_env(index: int, epsilon: float) -> None:
        env, obs = live_envs[index]
        obs = np.nan_to_num(np.asarray(obs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        action = choose_action(obs, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_obs = np.nan_to_num(
            np.asarray(next_obs, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0))
        reward = float(np.clip(reward, -1.0, 1.0))
        replay.append((obs, action, reward, next_obs, done))
        if done:
            next_obs, _ = env.reset()
            next_obs = np.nan_to_num(
                np.asarray(next_obs, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        live_envs[index][1] = next_obs

    logger.info("DQN pre-collecting transitions...")
    for i in range(args.precollect_steps):
        step_env(i % len(live_envs), epsilon=1.0)

    losses = []
    skipped_updates = 0
    logger.info("Starting DQN training...")
    for epoch in range(1, args.max_epoch + 1):
        epsilon = max(args.epsilon_final, args.epsilon_start * (args.epsilon_decay ** (epoch - 1)))
        epoch_losses = []
        for step in range(args.step_per_epoch):
            step_env(step % len(live_envs), epsilon=epsilon)
            if len(replay) < args.batch_size:
                continue
            batch = random.sample(replay, args.batch_size)
            obs, actions, rewards, next_obs, dones = zip(*batch)
            obs_arr = np.nan_to_num(np.asarray(obs), nan=0.0, posinf=0.0, neginf=0.0)
            next_obs_arr = np.nan_to_num(np.asarray(next_obs), nan=0.0, posinf=0.0, neginf=0.0)
            rewards_arr = np.clip(
                np.nan_to_num(np.asarray(rewards), nan=0.0, posinf=1.0, neginf=-1.0),
                -1.0,
                1.0,
            )
            obs_t = torch.as_tensor(obs_arr, dtype=torch.float32)
            actions_t = torch.as_tensor(actions, dtype=torch.long).unsqueeze(1)
            rewards_t = torch.as_tensor(rewards_arr, dtype=torch.float32)
            next_obs_t = torch.as_tensor(next_obs_arr, dtype=torch.float32)
            dones_t = torch.as_tensor(dones, dtype=torch.float32)

            q_values, _ = q_net(obs_t)
            if not torch.isfinite(q_values).all():
                skipped_updates += 1
                continue
            current_q = q_values.gather(1, actions_t).squeeze(1)
            with torch.no_grad():
                next_q_values, _ = target_net(next_obs_t)
                if not torch.isfinite(next_q_values).all():
                    skipped_updates += 1
                    continue
                target_q = rewards_t + args.gamma * (1.0 - dones_t) * next_q_values.max(dim=1).values
                target_q = torch.clamp(target_q, -2.0, 2.0)

            if not torch.isfinite(current_q).all() or not torch.isfinite(target_q).all():
                skipped_updates += 1
                continue

            loss = F.smooth_l1_loss(current_q, target_q)
            if not torch.isfinite(loss):
                skipped_updates += 1
                continue
            optim.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                optim.zero_grad()
                skipped_updates += 1
                continue
            optim.step()
            epoch_losses.append(float(loss.item()))

            if step % args.target_update_interval == 0:
                target_net.load_state_dict(q_net.state_dict())

        losses.extend(epoch_losses)
        logger.info(
            "DQN epoch %s/%s epsilon=%.3f loss=%.6f replay=%s skipped_nonfinite=%s",
            epoch,
            args.max_epoch,
            epsilon,
            float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            len(replay),
            skipped_updates,
        )

    target_net.load_state_dict(q_net.state_dict())
    metrics = evaluate_dqn(q_net, test_envs, n_episodes=min(args.eval_episodes, len(test_envs)))
    metrics["mean_loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["skipped_nonfinite_updates"] = skipped_updates
    metrics["train_env_count"] = len(train_envs)
    metrics["test_env_count"] = len(test_envs)
    return q_net, metrics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-envs", type=int, default=0, help="Limit stock environments; 0 means all")
    parser.add_argument("--max-epoch", type=int, default=20)
    parser.add_argument("--step-per-epoch", type=int, default=10000)
    parser.add_argument("--step-per-collect", type=int, default=100)
    parser.add_argument("--precollect-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument(
        "--lgb-score-mode",
        choices=["latest", "historical", "none"],
        default="latest",
        help="latest broadcasts current production LGB score per stock; historical is experimental",
    )
    parser.add_argument("--no-lgb-score-state", action="store_true")
    parser.add_argument("--trainer", choices=["dqn", "tianshou_sac"], default="dqn")
    parser.add_argument("--replay-size", type=int, default=100000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epsilon-start", type=float, default=0.35)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.90)
    parser.add_argument("--target-update-interval", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None):
    args = parse_args(argv)
    lgb_score_mode = "none" if args.no_lgb_score_state else args.lgb_score_mode
    envs = build_envs_from_qlib(max_envs=args.max_envs, lgb_score_mode=lgb_score_mode)
    if not envs:
        logger.error("No valid environments built. Check Qlib data.")
        return 1

    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.n
    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")

    from models.rl_agent import TransformerActor, TransformerCritic

    if args.trainer == "dqn":
        actor, metrics = train_dqn(envs, state_dim, action_dim, args)

        from datetime import datetime
        metrics["date"] = datetime.now().strftime("%Y-%m-%d")
        metrics["deployed"] = False
        metrics["valid"] = _metrics_are_valid(metrics) and _module_has_finite_params(actor)
        metrics["state_components"] = STATE_COMPONENTS
        metrics["uses_qlib_score"] = lgb_score_mode != "none"
        metrics["lgb_score_mode"] = lgb_score_mode
        metrics["sentiment_mode"] = "neutral_placeholder_until_historical_store_exists"
        metrics["trainer"] = "dqn"
        metrics["max_epoch"] = args.max_epoch
        metrics["step_per_epoch"] = args.step_per_epoch

        if not metrics["valid"]:
            metrics["invalid_reason"] = "non_finite_metric"
            append_metrics(metrics)
            logger.error("RL training produced non-finite metrics; model artifact was not updated")
            return 1

        torch.save({
            "actor_state": actor.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "state_components": STATE_COMPONENTS,
            "trainer": "dqn",
            "actor_kwargs": {
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dropout": 0.1,
            },
        }, MODEL_PATH)
        logger.info(f"RL model saved to {MODEL_PATH}")

        append_metrics(metrics)
        logger.info(f"Metrics saved to {METRICS_PATH}")
        return 0

    # Networks
    actor = TransformerActor(state_dim=state_dim, action_dim=action_dim)
    critic1 = TransformerCritic(state_dim=state_dim, action_dim=action_dim)
    critic2 = TransformerCritic(state_dim=state_dim, action_dim=action_dim)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=3e-4)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=3e-4)

    # Tianshou SAC (discrete)
    from tianshou.policy import DiscreteSACPolicy
    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.trainer import OffpolicyTrainer
    from tianshou.env import DummyVectorEnv

    policy = DiscreteSACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=0.005,
        gamma=0.99,
        alpha=0.2,
        action_space=envs[0].action_space,
    )

    # Split envs for train/test
    split = max(1, len(envs) * 3 // 4)
    train_envs_list = envs[:split]
    test_envs_list = envs[split:] if split < len(envs) else envs[:4]

    n_train = min(8, len(train_envs_list))
    n_test = min(4, len(test_envs_list))

    train_vec = DummyVectorEnv([make_env_fn(train_envs_list, i) for i in range(n_train)])
    test_vec = DummyVectorEnv([make_env_fn(test_envs_list, i) for i in range(n_test)])

    buf = VectorReplayBuffer(total_size=100000, buffer_num=n_train)
    train_collector = Collector(policy, train_vec, buf, exploration_noise=True)
    test_collector = Collector(policy, test_vec)

    # Pre-collect
    logger.info("Pre-collecting random episodes...")
    train_collector.collect(n_step=args.precollect_steps, random=True)

    logger.info("Starting SAC training...")
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.max_epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        update_per_step=args.update_per_step,
        episode_per_test=n_test,
        batch_size=args.batch_size,
    ).run()

    logger.info(f"Training result: {result}")

    # Evaluate
    metrics = evaluate_model(policy, test_vec, n_episodes=n_test * 2)
    logger.info(f"Evaluation metrics: {metrics}")

    # Save model
    torch.save({
        "actor_state": actor.state_dict(),
        "critic1_state": critic1.state_dict(),
        "critic2_state": critic2.state_dict(),
        "state_dim": state_dim,
        "state_components": STATE_COMPONENTS,
    }, MODEL_PATH)
    logger.info(f"RL model saved to {MODEL_PATH}")

    # Save metrics for daily evaluation tracking
    from datetime import datetime
    metrics["date"] = datetime.now().strftime("%Y-%m-%d")
    metrics["deployed"] = False  # Not deployed until manually approved
    metrics["state_components"] = STATE_COMPONENTS
    metrics["uses_qlib_score"] = lgb_score_mode != "none"
    metrics["lgb_score_mode"] = lgb_score_mode
    metrics["sentiment_mode"] = "neutral_placeholder_until_historical_store_exists"
    metrics["train_env_count"] = len(train_envs_list)
    metrics["test_env_count"] = len(test_envs_list)
    metrics["max_epoch"] = args.max_epoch
    metrics["step_per_epoch"] = args.step_per_epoch

    metrics["valid"] = _metrics_are_valid(metrics)
    append_metrics(metrics)
    logger.info(f"Metrics saved to {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
