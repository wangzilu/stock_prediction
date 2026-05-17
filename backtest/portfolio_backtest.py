"""Portfolio backtest engine for Qlib model predictions.

Takes daily model predictions → forms TopK portfolio → tracks PnL with costs.

Execution assumption: T日收盘后出信号, T+1 close-to-close 记账.
T+1 open/VWAP 成交模拟可作为后续扩展，但不能和训练 label 混用。

Constraints:
- T+1 (no same-day sell)
- Limit-up: cannot buy (涨停不可买)
- Limit-down: cannot sell (跌停不可卖)
- Suspended: cannot trade (停牌不可交易)
- ST: excluded from universe
- Min ADV filter (minimum daily turnover)

Usage:
    from backtest.portfolio_backtest import PortfolioBacktest
    from backtest.cost_model import CostModel

    bt = PortfolioBacktest(top_k=20, cost_model=CostModel())
    result = bt.run(predictions, price_data)
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .cost_model import CostModel

logger = logging.getLogger(__name__)


@dataclass
class PortfolioResult:
    """Complete backtest result with cost breakdown."""
    # Performance
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0

    # Cost
    total_cost: float = 0.0
    cost_to_return_ratio: float = 0.0
    avg_turnover: float = 0.0

    # Raw (before cost)
    raw_total_return: float = 0.0
    raw_annual_return: float = 0.0
    raw_sharpe: float = 0.0

    # Metadata
    n_days: int = 0
    avg_holdings: float = 0.0

    # Time series
    daily_pnl: pd.Series = field(default_factory=pd.Series)
    daily_turnover: pd.Series = field(default_factory=pd.Series)
    daily_cost: pd.Series = field(default_factory=pd.Series)

    def summary(self) -> str:
        lines = [
            "═" * 50,
            "PORTFOLIO BACKTEST RESULT",
            "═" * 50,
            f"Period:          {self.n_days} trading days",
            f"Avg holdings:    {self.avg_holdings:.1f} stocks",
            "",
            "--- Raw (before cost) ---",
            f"Total return:    {self.raw_total_return*100:+.2f}%",
            f"Annual return:   {self.raw_annual_return*100:+.2f}%",
            f"Sharpe:          {self.raw_sharpe:.3f}",
            "",
            "--- After cost ---",
            f"Total return:    {self.total_return*100:+.2f}%",
            f"Annual return:   {self.annual_return*100:+.2f}%",
            f"Annual vol:      {self.annual_volatility*100:.2f}%",
            f"Sharpe:          {self.sharpe_ratio:.3f}",
            f"Calmar:          {self.calmar_ratio:.3f}",
            f"Max drawdown:    {self.max_drawdown*100:.2f}%",
            f"Win rate (day):  {self.win_rate*100:.1f}%",
            "",
            "--- Cost ---",
            f"Total cost:      {self.total_cost*100:.3f}%",
            f"Cost/Return:     {self.cost_to_return_ratio*100:.1f}%",
            f"Avg turnover:    {self.avg_turnover*100:.1f}%",
            "═" * 50,
        ]
        return "\n".join(lines)


class PortfolioBacktest:
    """TopK equal-weight portfolio backtest with T+1 and cost model.

    Supports multiple execution modes:
    - "fixed": fixed-frequency rebalance with optional dropout/bonus
    - "buffered_partial": Garleanu-Pedersen style partial trading + buffer zone + vol throttle
    """

    def __init__(
        self,
        top_k: int = 20,
        cost_model: Optional[CostModel] = None,
        min_adv: float = 5e6,       # 最小日成交额 500 万
        max_weight: float = 0.08,   # 单票最大权重 8%
        rebalance_freq: int = 5,    # 每 N 天换仓 (default: weekly)
        dropout_k: int = 0,         # TopK dropout: only sell if falls below top_k + dropout_k
        hold_bonus: float = 0.0,    # Score bonus for currently held stocks (reduces turnover)
        # --- Buffered Partial Rebalance params ---
        mode: str = "fixed",        # "fixed" or "buffered_partial"
        buffer: int = 5,            # no-trade zone: stocks ranked top_k+1 ~ top_k+buffer are safe
        trade_rate: float = 0.35,   # fraction to trade toward target per day (Garleanu-Pedersen)
        min_hold_days: int = 2,     # minimum holding days (T+1 + 1 extra)
        max_daily_turnover: float = 0.15,  # cap daily turnover
        vol_window: int = 20,       # lookback for volatility estimation
        vol_threshold: float = 1.5, # if current vol > threshold * median, reduce trading
        # --- Drawdown stop-loss ---
        drawdown_stop: float = 0.0, # if > 0, force sell all when drawdown exceeds this (e.g. 0.08 = 8%)
    ):
        self.top_k = top_k
        self.cost = cost_model or CostModel()
        self.min_adv = min_adv
        self.max_weight = max_weight
        self.rebalance_freq = rebalance_freq
        self.dropout_k = dropout_k
        self.hold_bonus = hold_bonus
        self.mode = mode
        self.buffer = buffer
        self.trade_rate = trade_rate
        self.min_hold_days = min_hold_days
        self.max_daily_turnover = max_daily_turnover
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.drawdown_stop = drawdown_stop

    def run(
        self,
        predictions: pd.DataFrame,
        returns: pd.DataFrame,
        limit_up: Optional[pd.DataFrame] = None,
        limit_down: Optional[pd.DataFrame] = None,
        suspended: Optional[pd.DataFrame] = None,
        adv: Optional[pd.DataFrame] = None,
        return_horizon_days: int = 1,
    ) -> PortfolioResult:
        """Run backtest.

        Args:
            predictions: DataFrame indexed by (datetime, instrument) with column 'score'
            returns: DataFrame indexed by (datetime, instrument) with 1-day realized return
                     (close-to-close daily return, NOT model training label)
            return_horizon_days: Must be 1. Safety check to prevent passing multi-day labels.
            limit_up: Boolean DataFrame - True if stock hit limit up (cannot buy)
            limit_down: Boolean DataFrame - True if stock hit limit down (cannot sell)
            suspended: Boolean DataFrame - True if stock is suspended
            adv: Average daily volume (for liquidity filter)

        Returns:
            PortfolioResult
        """
        # Safety: prevent passing multi-day model labels as daily PnL
        if return_horizon_days != 1:
            raise ValueError(
                f"PortfolioBacktest requires daily realized returns (horizon=1), "
                f"got horizon={return_horizon_days}. "
                f"Do NOT pass model training labels as PnL returns."
            )

        # Align predictions and returns
        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame("score")
        if isinstance(returns, pd.Series):
            returns = returns.to_frame("return")

        dates = sorted(predictions.index.get_level_values(0).unique())
        logger.info(f"Backtest: {len(dates)} dates, top_k={self.top_k}, "
                    f"rebal_freq={self.rebalance_freq}, dropout_k={self.dropout_k}, "
                    f"hold_bonus={self.hold_bonus}")

        daily_pnl_raw = []
        daily_pnl_net = []
        daily_costs = []
        daily_turnovers = []
        daily_holdings_count = []
        prev_portfolio = set()
        days_since_rebal = 0
        holding_days = {}  # {stock: days_held}
        recent_pnl = []    # last N daily returns for vol estimation

        for i, date in enumerate(dates):
            # Get predictions for this date
            if date not in predictions.index.get_level_values(0):
                continue

            day_pred = predictions.loc[date]
            if isinstance(day_pred, pd.DataFrame):
                scores = day_pred["score"] if "score" in day_pred.columns else day_pred.iloc[:, 0]
            else:
                scores = day_pred

            # Filter: remove NaN, apply liquidity filter
            scores = scores.dropna()
            if adv is not None and date in adv.index.get_level_values(0):
                day_adv = adv.loc[date]
                liquid = day_adv[day_adv > self.min_adv].index
                scores = scores[scores.index.isin(liquid)]

            # Filter: suspended stocks cannot be traded
            if suspended is not None and date in suspended.index.get_level_values(0):
                day_susp = suspended.loc[date]
                if isinstance(day_susp, pd.Series):
                    susp_set = set(day_susp[day_susp == True].index)
                    scores = scores[~scores.index.isin(susp_set)]
                elif isinstance(day_susp, pd.DataFrame):
                    susp_set = set(day_susp[day_susp.iloc[:, 0] == True].index)
                    scores = scores[~scores.index.isin(susp_set)]

            # Filter: cannot buy limit-up stocks (only affects new buys)
            if limit_up is not None and date in limit_up.index.get_level_values(0):
                day_lu = limit_up.loc[date]
                if isinstance(day_lu, pd.Series):
                    blocked = set(day_lu[day_lu == True].index)
                elif isinstance(day_lu, pd.DataFrame):
                    blocked = set(day_lu[day_lu.iloc[:, 0] == True].index)
                else:
                    blocked = set()
                new_candidates = scores.index.difference(prev_portfolio)
                new_blocked = blocked & set(new_candidates)
                if new_blocked:
                    scores = scores[~scores.index.isin(new_blocked)]

            # Filter: cannot sell limit-down stocks (force keep in portfolio)
            cannot_sell = set()
            if limit_down is not None and date in limit_down.index.get_level_values(0):
                day_ld = limit_down.loc[date]
                if isinstance(day_ld, pd.Series):
                    cannot_sell = set(day_ld[day_ld == True].index)
                elif isinstance(day_ld, pd.DataFrame):
                    cannot_sell = set(day_ld[day_ld.iloc[:, 0] == True].index)

            # === Drawdown stop-loss check ===
            in_drawdown_stop = False
            if self.drawdown_stop > 0 and len(daily_pnl_net) >= 5:
                # Compute recent peak and current drawdown
                cum = np.cumprod([1 + r for r in daily_pnl_net])
                peak = np.max(cum)
                current_dd = (cum[-1] - peak) / peak
                if current_dd < -self.drawdown_stop:
                    in_drawdown_stop = True

            # === Portfolio construction ===
            if in_drawdown_stop:
                # Force to cash: sell everything
                target_portfolio = set()
                turnover_override = 1.0 if prev_portfolio else 0.0
            elif self.mode == "buffered_partial":
                target_portfolio, turnover_override = self._buffered_partial_step(
                    scores, prev_portfolio, holding_days, recent_pnl, cannot_sell)
            else:
                # Fixed-frequency mode (original logic)
                days_since_rebal += 1
                is_rebal_day = (days_since_rebal >= self.rebalance_freq) or (not prev_portfolio)
                turnover_override = None

                if is_rebal_day:
                    days_since_rebal = 0

                    if self.hold_bonus > 0 and prev_portfolio:
                        scores = scores.copy()
                        held_mask = scores.index.isin(prev_portfolio)
                        scores[held_mask] += self.hold_bonus

                    if self.dropout_k > 0 and prev_portfolio:
                        sell_threshold = self.top_k + self.dropout_k
                        if len(scores) >= sell_threshold:
                            safe_zone = set(scores.nlargest(sell_threshold).index)
                        else:
                            safe_zone = set(scores.index)
                        forced_keep = prev_portfolio & safe_zone
                        remaining_slots = self.top_k - len(forced_keep)
                        if remaining_slots > 0:
                            available = scores[~scores.index.isin(forced_keep)]
                            new_picks = set(available.nlargest(remaining_slots).index)
                            target_portfolio = forced_keep | new_picks
                        else:
                            target_portfolio = set(list(forced_keep)[:self.top_k])
                    else:
                        if len(scores) < self.top_k:
                            target_portfolio = set(scores.index)
                        else:
                            target_portfolio = set(scores.nlargest(self.top_k).index)
                else:
                    target_portfolio = prev_portfolio

            # Force keep cannot_sell stocks from prev portfolio
            forced_keep_final = prev_portfolio & cannot_sell
            target_portfolio = target_portfolio | forced_keep_final

            # Compute turnover
            if turnover_override is not None:
                turnover = turnover_override
            elif prev_portfolio:
                sells = prev_portfolio - target_portfolio
                buys = target_portfolio - prev_portfolio
                turnover = (len(sells) + len(buys)) / (2 * max(len(prev_portfolio), 1))
            else:
                buys = target_portfolio
                sells = set()
                turnover = 1.0  # first day: full buy

            # Update holding days
            new_holding_days = {}
            for s in target_portfolio:
                new_holding_days[s] = holding_days.get(s, 0) + 1
            holding_days = new_holding_days

            # Compute returns for target portfolio (T+1 assumption)
            # The return is realized on the NEXT trading day
            if i + 1 < len(dates):
                next_date = dates[i + 1]
                if next_date in returns.index.get_level_values(0):
                    day_returns = returns.loc[next_date]
                    if isinstance(day_returns, pd.DataFrame):
                        day_returns = day_returns.iloc[:, 0]

                    # Equal weight portfolio return
                    port_stocks = list(target_portfolio)
                    port_rets = day_returns.reindex(port_stocks).dropna()

                    if len(port_rets) > 0:
                        raw_ret = port_rets.mean()
                    else:
                        raw_ret = 0.0
                else:
                    raw_ret = 0.0
            else:
                raw_ret = 0.0

            # Cost: proportional to turnover
            cost_rate = self.cost.round_trip_rate() * turnover
            net_ret = raw_ret - cost_rate

            daily_pnl_raw.append(raw_ret)
            daily_pnl_net.append(net_ret)
            daily_costs.append(cost_rate)
            daily_turnovers.append(turnover)
            daily_holdings_count.append(len(target_portfolio))

            # Track recent PnL for vol estimation (buffered_partial mode)
            recent_pnl.append(raw_ret)
            if len(recent_pnl) > self.vol_window:
                recent_pnl = recent_pnl[-self.vol_window:]

            prev_portfolio = target_portfolio

        # Compute result metrics
        if not daily_pnl_net:
            return PortfolioResult()

        pnl_raw = np.array(daily_pnl_raw)
        pnl_net = np.array(daily_pnl_net)
        costs = np.array(daily_costs)
        turnovers = np.array(daily_turnovers)

        n_days = len(pnl_net)
        annual_factor = 250 / n_days if n_days > 0 else 1

        # Raw metrics
        raw_total = float(np.prod(1 + pnl_raw) - 1)
        raw_annual = float((1 + raw_total) ** annual_factor - 1)
        raw_vol = float(np.std(pnl_raw) * np.sqrt(250))
        raw_sharpe = raw_annual / (raw_vol + 1e-8)

        # Net metrics
        total_ret = float(np.prod(1 + pnl_net) - 1)
        annual_ret = float((1 + total_ret) ** annual_factor - 1)
        annual_vol = float(np.std(pnl_net) * np.sqrt(250))
        sharpe = annual_ret / (annual_vol + 1e-8)

        # Max drawdown
        cum = np.cumprod(1 + pnl_net)
        running_max = np.maximum.accumulate(cum)
        drawdowns = (cum - running_max) / running_max
        max_dd = float(np.min(drawdowns))

        calmar = annual_ret / (abs(max_dd) + 1e-8)
        win_rate = float(np.mean(pnl_net > 0))
        total_cost = float(np.sum(costs))
        cost_ratio = total_cost / (abs(raw_total) + 1e-8) if raw_total != 0 else 0

        result = PortfolioResult(
            total_return=total_ret,
            annual_return=annual_ret,
            annual_volatility=annual_vol,
            sharpe_ratio=sharpe,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_cost=total_cost,
            cost_to_return_ratio=cost_ratio,
            avg_turnover=float(np.mean(turnovers)),
            raw_total_return=raw_total,
            raw_annual_return=raw_annual,
            raw_sharpe=raw_sharpe,
            n_days=n_days,
            avg_holdings=float(np.mean(daily_holdings_count)),
            daily_pnl=pd.Series(pnl_net, index=dates[:n_days]),
            daily_turnover=pd.Series(turnovers, index=dates[:n_days]),
            daily_cost=pd.Series(costs, index=dates[:n_days]),
        )

        return result

    def _buffered_partial_step(
        self,
        scores: pd.Series,
        prev_portfolio: set,
        holding_days: dict,
        recent_pnl: list,
        cannot_sell: set,
    ) -> tuple[set, float]:
        """Buffered Partial Rebalance: Garleanu-Pedersen + Smart Rebalancing + Vol throttle.

        Returns (target_portfolio, turnover).
        """
        if not prev_portfolio:
            # First day: buy top_k
            if len(scores) < self.top_k:
                return set(scores.index), 1.0
            return set(scores.nlargest(self.top_k).index), 1.0

        # Step 1: Determine sell candidates (only sell if dropped BELOW buffer zone)
        ranked = scores.sort_values(ascending=False)
        top_candidates = set(ranked.index[:self.top_k])
        buffer_zone = set(ranked.index[self.top_k:self.top_k + self.buffer])
        safe_zone = top_candidates | buffer_zone  # stocks ranked 1 ~ top_k+buffer

        # Identify stocks to sell: held but dropped out of safe zone AND held long enough
        sell_candidates = []
        for stock in prev_portfolio:
            if stock not in safe_zone and stock not in cannot_sell:
                if holding_days.get(stock, 999) >= self.min_hold_days:
                    sell_candidates.append(stock)

        # Step 2: Determine buy candidates (must be in top_k and not already held)
        buy_candidates = [s for s in ranked.index[:self.top_k] if s not in prev_portfolio]

        # Step 3: Adaptive trade rate based on recent volatility
        effective_rate = self.trade_rate
        if len(recent_pnl) >= self.vol_window:
            current_vol = np.std(recent_pnl[-self.vol_window:])
            # Compare to full history median (approximate with half the window)
            if len(recent_pnl) >= self.vol_window:
                median_vol = np.median([
                    np.std(recent_pnl[max(0, j-self.vol_window):j])
                    for j in range(self.vol_window, len(recent_pnl), 5)
                ]) if len(recent_pnl) > self.vol_window * 2 else current_vol
                if median_vol > 0 and current_vol / median_vol > self.vol_threshold:
                    effective_rate *= 0.5  # halve trade speed in high-vol

        # Step 4: Partial trading — only trade a fraction of desired changes
        n_sells = int(len(sell_candidates) * effective_rate + 0.5)
        n_sells = min(n_sells, int(self.max_daily_turnover * len(prev_portfolio)))

        # Priority: sell weakest first (lowest score among sell candidates)
        if n_sells > 0 and sell_candidates:
            sell_scores = scores.reindex(sell_candidates).dropna().sort_values()
            actual_sells = set(sell_scores.index[:n_sells])
        else:
            actual_sells = set()

        # Buy to fill vacated slots (priority: highest score among buy candidates)
        n_buys = min(len(actual_sells), len(buy_candidates))
        n_buys = min(n_buys, int(self.max_daily_turnover * len(prev_portfolio)) - len(actual_sells))
        n_buys = max(0, n_buys)
        actual_buys = set(buy_candidates[:n_buys])

        # Build target portfolio
        target_portfolio = (prev_portfolio - actual_sells) | actual_buys

        # Ensure we don't exceed top_k too much (can happen if few sells)
        if len(target_portfolio) > self.top_k + self.buffer:
            # Trim by removing lowest-scored excess stocks
            excess = len(target_portfolio) - self.top_k
            port_scores = scores.reindex(list(target_portfolio)).dropna().sort_values()
            to_remove = set(port_scores.index[:excess]) - cannot_sell
            # Only remove if held long enough
            to_remove = {s for s in to_remove if holding_days.get(s, 999) >= self.min_hold_days}
            target_portfolio -= to_remove

        # Compute actual turnover
        turnover = (len(actual_sells) + len(actual_buys)) / (2 * max(len(prev_portfolio), 1))

        return target_portfolio, turnover
