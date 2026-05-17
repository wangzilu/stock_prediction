"""Transaction cost model for A-share backtest.

Covers:
- Commission (买卖佣金)
- Stamp tax (印花税, sell-only)
- Slippage (滑点)
- Impact cost placeholder

A-share assumptions:
- T+1: cannot sell on buy day
- Limit-up: cannot buy
- Limit-down: cannot sell
- Suspended: cannot trade
- ST: filtered by default
- Minimum turnover filter

Usage:
    cost = CostModel()
    buy_cost = cost.buy_cost(price=10.0, shares=1000)
    sell_cost = cost.sell_cost(price=11.0, shares=1000)
"""
from dataclasses import dataclass


@dataclass
class CostModel:
    """A-share transaction cost model."""

    # Commission: both sides, min 5 yuan per trade
    commission_rate: float = 0.0003  # 万三 (most retail brokers)
    min_commission: float = 5.0  # 最低佣金 5 元

    # Stamp tax: sell side only
    stamp_tax_rate: float = 0.0005  # 万五 (since 2023-08-28 reduced from 千一)

    # Slippage: simulates market impact of crossing spread
    slippage_rate: float = 0.001  # 单边 0.1% (conservative for small cap)

    # Impact cost placeholder (for capacity estimation)
    impact_rate: float = 0.0  # disabled by default

    def buy_cost(self, price: float, shares: int) -> float:
        """Total cost of buying (commission + slippage)."""
        amount = price * shares
        commission = max(amount * self.commission_rate, self.min_commission)
        slippage = amount * self.slippage_rate
        impact = amount * self.impact_rate
        return commission + slippage + impact

    def sell_cost(self, price: float, shares: int) -> float:
        """Total cost of selling (commission + stamp tax + slippage)."""
        amount = price * shares
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_tax = amount * self.stamp_tax_rate
        slippage = amount * self.slippage_rate
        impact = amount * self.impact_rate
        return commission + stamp_tax + slippage + impact

    def round_trip_rate(self) -> float:
        """Total cost rate for a complete buy+sell round trip."""
        return (self.commission_rate * 2  # buy + sell commission
                + self.stamp_tax_rate     # sell stamp tax
                + self.slippage_rate * 2  # buy + sell slippage
                + self.impact_rate * 2)   # buy + sell impact

    def summary(self) -> str:
        rt = self.round_trip_rate()
        return (
            f"CostModel: commission={self.commission_rate*10000:.1f}‱×2, "
            f"stamp={self.stamp_tax_rate*10000:.1f}‱, "
            f"slippage={self.slippage_rate*10000:.1f}‱×2, "
            f"round_trip={rt*100:.3f}%"
        )
