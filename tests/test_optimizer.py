import pytest
import numpy as np
import pandas as pd
from backtest.optimizer import WeightOptimizer, OptimizationResult, generate_performance_report
from backtest.engine import BacktestResult, TradeRecord


def _make_price_data(n_stocks=3, n_days=100):
    """Create synthetic price data."""
    data = {}
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")
    np.random.seed(42)
    for i in range(n_stocks):
        code = f"SH60051{i}"
        base_price = 100 + i * 50
        returns = np.random.randn(n_days) * 0.02
        prices = base_price * np.cumprod(1 + returns)
        data[code] = pd.DataFrame({
            "open": prices * (1 + np.random.randn(n_days) * 0.005),
            "high": prices * (1 + abs(np.random.randn(n_days) * 0.01)),
            "low": prices * (1 - abs(np.random.randn(n_days) * 0.01)),
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, n_days),
        }, index=dates)
    return data


def test_optimizer_runs():
    """Optimizer should complete and return best params."""
    optimizer = WeightOptimizer(holding_days=5, top_k=2)
    data = _make_price_data()

    # Use small grid for speed
    result = optimizer.optimize(data, weight_grid=[0.2, 0.3, 0.5])

    assert isinstance(result, OptimizationResult)
    assert result.best_params != {}
    assert "weight_short" in result.best_params
    assert len(result.all_results) > 0


def test_optimizer_empty_data():
    """Empty data should not crash."""
    optimizer = WeightOptimizer()
    result = optimizer.optimize({}, weight_grid=[0.3, 0.7])
    assert result.best_sharpe <= 0 or result.best_result.total_signals == 0


def test_optimizer_summary():
    """Summary should be readable string."""
    optimizer = WeightOptimizer(holding_days=5, top_k=1)
    data = _make_price_data(n_stocks=2, n_days=80)
    result = optimizer.optimize(data, weight_grid=[0.2, 0.3, 0.5])

    summary = result.summary()
    assert "最优参数" in summary
    assert "夏普" in summary


def test_performance_report():
    """Performance report should format correctly."""
    trades = [
        TradeRecord("2025-01-10", "SH600510", "茅台", "看多", 0.8, 100, 105, 5.0, 5, True),
        TradeRecord("2025-01-15", "SH600511", "平安", "看多", 0.6, 50, 48, -4.0, 5, False),
        TradeRecord("2025-02-01", "SH600510", "茅台", "看空", 0.7, 110, 105, 4.5, 5, True),
    ]
    result = BacktestResult(
        total_signals=3,
        correct_signals=2,
        win_rate=66.7,
        avg_return=1.83,
        total_return=5.5,
        max_drawdown=4.0,
        sharpe_ratio=0.95,
        trades=trades,
        daily_returns=[1.0, -2.0, 3.0],
    )
    report = generate_performance_report(result)
    assert "策略表现报告" in report
    assert "66.7%" in report
    assert "月度表现" in report
    assert "最佳交易" in report


def test_performance_report_empty():
    """Empty result should still produce a report."""
    result = BacktestResult()
    report = generate_performance_report(result)
    assert "策略表现报告" in report
