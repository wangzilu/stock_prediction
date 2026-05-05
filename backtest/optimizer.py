import numpy as np
import pandas as pd
import logging
from itertools import product
from dataclasses import dataclass

from backtest.engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    best_params: dict
    best_sharpe: float
    best_win_rate: float
    best_result: BacktestResult
    all_results: list  # List of (params, BacktestResult)

    def summary(self) -> str:
        lines = [
            "═══════ 优化结果 ═══════",
            f"最优参数：{self.best_params}",
            f"最优夏普：{self.best_sharpe:.2f}",
            f"最优胜率：{self.best_win_rate:.1f}%",
            f"测试组合数：{len(self.all_results)}",
            "───────────────────────",
            self.best_result.summary(),
        ]
        return "\n".join(lines)


class WeightOptimizer:
    """Grid search optimizer for signal weights.

    Optimizes the weights for multi-timeframe signal fusion:
    - weight_short: Short-term model weight
    - weight_mid: Mid-term model weight
    - weight_sentiment: Sentiment weight
    - weight_macro: Macro/geo weight
    """

    def __init__(self, holding_days: int = 5, top_k: int = 3):
        self.holding_days = holding_days
        self.top_k = top_k

    def optimize(
        self,
        price_data: dict,
        short_scores: dict = None,
        mid_scores: dict = None,
        sentiment_scores: dict = None,
        macro_score: float = 0.0,
        weight_grid: list = None,
    ) -> OptimizationResult:
        """Run grid search over weight combinations.

        Args:
            price_data: Dict of {code: DataFrame} with OHLCV
            short_scores: Dict of {code: {date: score}} for short-term
            mid_scores: Dict of {code: {date: score}} for mid-term
            sentiment_scores: Dict of {code: {date: score}} for sentiment
            macro_score: Single macro score (constant for simplicity)
            weight_grid: List of weight values to try (default [0.1, 0.2, ..., 0.6])

        Returns:
            OptimizationResult with best parameters
        """
        if weight_grid is None:
            weight_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        # Generate valid weight combinations (must sum to ~1.0)
        valid_combos = []
        for ws, wm, wsen, wmac in product(weight_grid, repeat=4):
            total = ws + wm + wsen + wmac
            if 0.9 <= total <= 1.1:  # Allow small tolerance
                valid_combos.append({
                    "weight_short": ws,
                    "weight_mid": wm,
                    "weight_sentiment": wsen,
                    "weight_macro": wmac,
                })

        if not valid_combos:
            logger.warning("No valid weight combinations found")
            return OptimizationResult(
                best_params={}, best_sharpe=0, best_win_rate=0,
                best_result=BacktestResult(), all_results=[],
            )

        logger.info(f"Testing {len(valid_combos)} weight combinations...")

        all_results = []
        best_sharpe = -float("inf")
        best_idx = 0

        for i, params in enumerate(valid_combos):
            # Create signal function with these weights
            def make_signal_fn(w):
                def signal_fn(code, history):
                    # Simple proxy signals from price data
                    if len(history) < 20:
                        return 0.0

                    # Short-term: 5-day momentum
                    ret_5d = (history.iloc[-1]["close"] - history.iloc[-5]["close"]) / history.iloc[-5]["close"]
                    short = float(np.clip(ret_5d * 10, -1, 1))

                    # Mid-term: 20-day momentum
                    ret_20d = (history.iloc[-1]["close"] - history.iloc[-20]["close"]) / history.iloc[-20]["close"]
                    mid = float(np.clip(ret_20d * 5, -1, 1))

                    # Sentiment proxy: volume change
                    vol_ratio = history.iloc[-1]["volume"] / history["volume"].mean()
                    sent = float(np.clip((vol_ratio - 1) * 2, -1, 1))

                    # Macro (constant)
                    mac = macro_score

                    score = (
                        short * w["weight_short"]
                        + mid * w["weight_mid"]
                        + sent * w["weight_sentiment"]
                        + mac * w["weight_macro"]
                    )
                    return float(np.clip(score, -1, 1))
                return signal_fn

            engine = BacktestEngine(holding_days=self.holding_days, top_k=self.top_k)
            result = engine.run(price_data, signal_fn=make_signal_fn(params))
            all_results.append((params, result))

            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_idx = i

        best_params, best_result = all_results[best_idx]

        return OptimizationResult(
            best_params=best_params,
            best_sharpe=best_result.sharpe_ratio,
            best_win_rate=best_result.win_rate,
            best_result=best_result,
            all_results=all_results,
        )


def generate_performance_report(result: BacktestResult, title: str = "策略表现报告") -> str:
    """Generate a detailed performance report.

    Args:
        result: BacktestResult from backtesting
        title: Report title

    Returns:
        Formatted report string
    """
    lines = [
        f"{'═' * 30}",
        f"  {title}",
        f"{'═' * 30}",
        "",
        "【总体表现】",
        f"  信号总数：{result.total_signals}",
        f"  正确信号：{result.correct_signals}",
        f"  胜率：{result.win_rate:.1f}%",
        f"  平均每笔收益：{result.avg_return:+.2f}%",
        f"  累计收益：{result.total_return:+.2f}%",
        f"  最大回撤：{result.max_drawdown:.2f}%",
        f"  夏普比率：{result.sharpe_ratio:.2f}",
        "",
    ]

    # Monthly breakdown if enough data
    if result.trades:
        lines.append("【月度表现】")
        monthly = {}
        for trade in result.trades:
            month = trade.date[:7]  # YYYY-MM
            if month not in monthly:
                monthly[month] = {"trades": 0, "correct": 0, "returns": []}
            monthly[month]["trades"] += 1
            if trade.is_correct:
                monthly[month]["correct"] += 1
            monthly[month]["returns"].append(trade.return_pct)

        for month in sorted(monthly.keys()):
            m = monthly[month]
            wr = m["correct"] / m["trades"] * 100 if m["trades"] > 0 else 0
            avg_ret = np.mean(m["returns"]) if m["returns"] else 0
            lines.append(
                f"  {month}: {m['trades']}笔 | 胜率{wr:.0f}% | 平均{avg_ret:+.1f}%"
            )

        lines.append("")

    # Top/worst trades
    if result.trades:
        sorted_trades = sorted(result.trades, key=lambda t: t.return_pct, reverse=True)
        lines.append("【最佳交易 Top 3】")
        for t in sorted_trades[:3]:
            lines.append(f"  {t.date} {t.code} {t.signal} → {t.return_pct:+.2f}%")

        lines.append("【最差交易 Top 3】")
        for t in sorted_trades[-3:]:
            lines.append(f"  {t.date} {t.code} {t.signal} → {t.return_pct:+.2f}%")

    lines.append(f"\n{'═' * 30}")
    return "\n".join(lines)
