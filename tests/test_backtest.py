import pytest
import numpy as np
import pandas as pd
from backtest.engine import BacktestEngine, BacktestResult, TradeRecord


def _make_price_data(n_stocks=3, n_days=100):
    """Create synthetic price data for multiple stocks."""
    data = {}
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")  # Business days

    np.random.seed(42)
    for i in range(n_stocks):
        code = f"SH60051{i}"
        base_price = 100 + i * 50
        returns = np.random.randn(n_days) * 0.02  # 2% daily vol
        prices = base_price * np.cumprod(1 + returns)

        data[code] = pd.DataFrame({
            "open": prices * (1 + np.random.randn(n_days) * 0.005),
            "high": prices * (1 + abs(np.random.randn(n_days) * 0.01)),
            "low": prices * (1 - abs(np.random.randn(n_days) * 0.01)),
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, n_days),
        }, index=dates)

    return data


def test_backtest_runs_without_error():
    """Backtest should run on synthetic data and return results."""
    engine = BacktestEngine(holding_days=5, top_k=2)
    data = _make_price_data()
    result = engine.run(data)

    assert isinstance(result, BacktestResult)
    assert result.total_signals > 0
    assert 0 <= result.win_rate <= 100


def test_backtest_empty_data():
    """Empty data should return empty result."""
    engine = BacktestEngine()
    result = engine.run({})
    assert result.total_signals == 0


def test_backtest_with_custom_signal():
    """Custom signal function should be used."""
    engine = BacktestEngine(holding_days=5, top_k=1)
    data = _make_price_data(n_stocks=1, n_days=60)

    # Always bullish signal
    def always_bullish(code, history):
        return 0.8

    result = engine.run(data, signal_fn=always_bullish)
    assert result.total_signals > 0
    # All signals should be bullish
    assert all(t.signal == "看多" for t in result.trades)


def test_backtest_metrics_reasonable():
    """Metrics should be within reasonable ranges."""
    engine = BacktestEngine(holding_days=5, top_k=2)
    data = _make_price_data(n_stocks=5, n_days=200)
    result = engine.run(data)

    # Win rate should be somewhere between 30-70% for random data
    assert 20 <= result.win_rate <= 80
    # Max drawdown should be positive
    assert result.max_drawdown >= 0
    # Sharpe ratio for random signals should be within a reasonable range
    assert -5 < result.sharpe_ratio < 5


def test_backtest_date_range():
    """Should respect start_date and end_date."""
    engine = BacktestEngine(holding_days=5, top_k=2)
    data = _make_price_data(n_stocks=2, n_days=200)

    result_full = engine.run(data)
    result_half = engine.run(data, start_date="2025-04-01")

    assert result_half.total_signals < result_full.total_signals


def test_backtest_summary_string():
    """Summary should be a formatted string."""
    result = BacktestResult(
        total_signals=100,
        correct_signals=62,
        win_rate=62.0,
        avg_return=0.35,
        total_return=35.0,
        max_drawdown=8.5,
        sharpe_ratio=1.2,
    )
    summary = result.summary()
    assert "62.0%" in summary
    assert "35.00%" in summary
    assert "1.20" in summary


def test_trade_record_fields():
    """TradeRecord should have all required fields."""
    trade = TradeRecord(
        date="2025-05-01",
        code="SH600519",
        name="贵州茅台",
        signal="看多",
        score=0.75,
        entry_price=1800.0,
        exit_price=1860.0,
        return_pct=3.33,
        is_correct=True,
    )
    assert trade.code == "SH600519"
    assert trade.return_pct == 3.33
