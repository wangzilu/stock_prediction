import numpy as np

from models.rl_agent import StockTradingEnv


def test_stock_trading_env_includes_qlib_sentiment_and_market_state():
    features = np.ones((80, 4), dtype=np.float32)
    prices = np.linspace(10, 12, 80, dtype=np.float32)
    qlib_scores = np.linspace(-0.1, 0.1, 80, dtype=np.float32)
    sentiment_scores = np.linspace(-0.5, 0.5, 80, dtype=np.float32)
    sentiment_heat = np.linspace(0.0, 1.0, 80, dtype=np.float32)
    market_regime = np.linspace(-0.2, 0.2, 80, dtype=np.float32)

    env = StockTradingEnv(
        features=features,
        prices=prices,
        qlib_scores=qlib_scores,
        sentiment_scores=sentiment_scores,
        sentiment_heat=sentiment_heat,
        market_regime=market_regime,
        window=20,
    )
    obs, _ = env.reset()

    assert env.observation_space.shape == (10,)
    assert obs.shape == (10,)
    assert obs[-6] == qlib_scores[20]
    assert obs[-5] == sentiment_scores[20]
    assert obs[-4] == sentiment_heat[20]
    assert obs[-3] == market_regime[20]
    assert obs[-2] == 0.0  # position


def test_stock_trading_env_applies_transaction_cost_on_buy():
    features = np.ones((80, 4), dtype=np.float32)
    prices = np.linspace(10, 12, 80, dtype=np.float32)
    env = StockTradingEnv(
        features=features,
        prices=prices,
        window=20,
        transaction_cost=0.01,
    )
    env.reset()

    _, reward, _, _, _ = env.step(1)

    assert reward == -0.01
