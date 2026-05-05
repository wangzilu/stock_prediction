import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """A single trade/signal record in backtest."""
    date: str
    code: str
    name: str
    signal: str
    score: float
    entry_price: float
    exit_price: float = 0.0
    return_pct: float = 0.0
    holding_days: int = 5
    is_correct: bool = False


@dataclass
class BacktestResult:
    """Overall backtest performance metrics."""
    total_signals: int = 0
    correct_signals: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    trades: list = field(default_factory=list)
    daily_returns: list = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "═══════ 回测结果 ═══════",
            f"信号总数：{self.total_signals}",
            f"正确信号：{self.correct_signals}",
            f"胜率：{self.win_rate:.1f}%",
            f"平均收益：{self.avg_return:+.2f}%",
            f"累计收益：{self.total_return:+.2f}%",
            f"最大回撤：{self.max_drawdown:.2f}%",
            f"夏普比率：{self.sharpe_ratio:.2f}",
            "═══════════════════════",
        ]
        return "\n".join(lines)


class BacktestEngine:
    """Backtesting engine for signal strategy evaluation.

    Replays historical data day-by-day, generates signals using the same
    scoring logic as live trading, and evaluates performance.
    """

    def __init__(self, holding_days: int = 5, top_k: int = 3):
        """
        Args:
            holding_days: How many days to hold after a signal
            top_k: Number of top signals to act on each day
        """
        self.holding_days = holding_days
        self.top_k = top_k

    def run(
        self,
        price_data: dict,
        signal_fn=None,
        start_date: str = None,
        end_date: str = None,
    ) -> BacktestResult:
        """Run backtest on historical data.

        Args:
            price_data: Dict of {code: DataFrame} with OHLCV data indexed by date.
            signal_fn: Function(code, price_series_up_to_date) -> float score.
                      If None, uses simple momentum (5-day return).
            start_date: Backtest start date (YYYY-MM-DD). Defaults to 60 days ago.
            end_date: Backtest end date (YYYY-MM-DD). Defaults to today.

        Returns:
            BacktestResult with performance metrics.
        """
        if not price_data:
            return BacktestResult()

        # Determine date range
        all_dates = set()
        for df in price_data.values():
            all_dates.update(df.index.strftime("%Y-%m-%d"))
        all_dates = sorted(all_dates)

        if not all_dates:
            return BacktestResult()

        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]

        # Need at least holding_days + lookback
        lookback = 20
        if len(all_dates) < lookback + self.holding_days:
            logger.warning("Not enough data for backtest")
            return BacktestResult()

        if signal_fn is None:
            signal_fn = self._default_signal

        trades = []
        daily_returns = []

        # Pre-compute date string sets per stock for O(1) lookup
        stock_date_sets = {}
        for code, df in price_data.items():
            stock_date_sets[code] = set(df.index.strftime("%Y-%m-%d"))

        # Iterate through tradeable dates
        for i in range(lookback, len(all_dates) - self.holding_days):
            current_date = all_dates[i]
            exit_date = all_dates[min(i + self.holding_days, len(all_dates) - 1)]

            # Generate signals for all stocks on this date
            signals = []
            for code, df in price_data.items():
                if current_date not in stock_date_sets[code]:
                    continue

                # Get data up to current date
                history = df[df.index <= pd.Timestamp(current_date)]

                if len(history) < lookback:
                    continue

                score = signal_fn(code, history)
                if score != 0:
                    signals.append((code, score, history.iloc[-1]["close"]))

            # Select top-k signals
            signals.sort(key=lambda x: abs(x[1]), reverse=True)
            top_signals = signals[:self.top_k]

            # Evaluate each signal
            day_return = 0.0
            for code, score, entry_price in top_signals:
                df = price_data[code]
                if exit_date not in stock_date_sets[code]:
                    continue

                exit_row = df[df.index == pd.Timestamp(exit_date)]
                if exit_row.empty:
                    continue

                exit_price = exit_row.iloc[0]["close"]

                # Long if score > 0, short if score < 0
                if score > 0:
                    ret = (exit_price - entry_price) / entry_price * 100
                else:
                    ret = (entry_price - exit_price) / entry_price * 100

                is_correct = ret > 0
                signal_text = "看多" if score > 0 else "看空"

                trade = TradeRecord(
                    date=current_date,
                    code=code,
                    name=code,
                    signal=signal_text,
                    score=score,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    return_pct=round(ret, 2),
                    holding_days=self.holding_days,
                    is_correct=is_correct,
                )
                trades.append(trade)
                day_return += ret / len(top_signals) if top_signals else 0

            daily_returns.append(day_return)

        # Compute metrics
        result = self._compute_metrics(trades, daily_returns)
        return result

    def _default_signal(self, code: str, history: pd.DataFrame) -> float:
        """Default signal: 5-day momentum normalized to [-1, 1]."""
        if len(history) < 5:
            return 0.0
        ret_5d = (history.iloc[-1]["close"] - history.iloc[-5]["close"]) / history.iloc[-5]["close"]
        return float(np.clip(ret_5d * 10, -1.0, 1.0))

    def _compute_metrics(self, trades: list, daily_returns: list) -> BacktestResult:
        """Compute backtest performance metrics."""
        if not trades:
            return BacktestResult()

        returns = [t.return_pct for t in trades]
        correct = [t for t in trades if t.is_correct]

        total_signals = len(trades)
        correct_signals = len(correct)
        win_rate = correct_signals / total_signals * 100 if total_signals > 0 else 0
        avg_return = np.mean(returns) if returns else 0
        total_return = np.sum(returns) if returns else 0

        # Max drawdown from cumulative returns
        cum_returns = np.cumsum(daily_returns) if daily_returns else [0]
        peak = np.maximum.accumulate(cum_returns)
        drawdown = cum_returns - peak
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0

        # Sharpe ratio (annualized, assuming 252 trading days)
        if daily_returns and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        return BacktestResult(
            total_signals=total_signals,
            correct_signals=correct_signals,
            win_rate=round(win_rate, 1),
            avg_return=round(float(avg_return), 2),
            total_return=round(float(total_return), 2),
            max_drawdown=round(float(max_drawdown), 2),
            sharpe_ratio=round(float(sharpe), 2),
            trades=trades,
            daily_returns=daily_returns,
        )
