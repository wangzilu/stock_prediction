"""Transaction cost model for A-share backtest.

Covers:
- Commission (买卖佣金)
- Stamp tax (印花税, sell-only)
- Slippage (滑点) — fixed or impact-based (sqrt-ADV model)
- Impact cost placeholder

A-share assumptions:
- T+1: cannot sell on buy day
- Limit-up: cannot buy
- Limit-down: cannot sell
- Suspended: cannot trade
- ST: filtered by default
- Minimum turnover filter

Impact model (sqrt_adv):
    slippage = volatility * sqrt(trade_value / ADV) * impact_coefficient
    Per Almgren-Chriss / CX plan: Impact = sigma * sqrt(Q/ADV) * sign(side)

Usage:
    # Fixed slippage (default, backward compatible)
    cost = CostModel()

    # Impact-based slippage
    cost = CostModel(impact_model="sqrt_adv", impact_coefficient=0.1)
    buy_cost = cost.buy_cost(price=10.0, shares=1000,
                             daily_volatility=0.02, adv=50_000_000)
"""
import math
from dataclasses import dataclass
from typing import Optional


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

    # --- Impact model parameters ---
    impact_model: str = "fixed"  # "fixed" or "sqrt_adv"
    impact_coefficient: float = 0.1  # scaling factor for sqrt_adv model

    def _slippage(
        self,
        amount: float,
        daily_volatility: Optional[float] = None,
        adv: Optional[float] = None,
    ) -> float:
        """Compute slippage cost in yuan.

        For "fixed" model: amount * slippage_rate (backward compatible).
        For "sqrt_adv" model: volatility * sqrt(trade_value / ADV) * impact_coefficient * amount.
            Falls back to fixed if daily_volatility or adv is missing/invalid.
        """
        if self.impact_model == "sqrt_adv":
            if (
                daily_volatility is not None
                and adv is not None
                and adv > 0
                and daily_volatility > 0
            ):
                # slippage rate = sigma * sqrt(Q / ADV) * coefficient
                ratio = amount / adv
                # Clamp ratio to avoid unrealistic values
                ratio = min(ratio, 1.0)
                slip_rate = daily_volatility * math.sqrt(ratio) * self.impact_coefficient
                return amount * slip_rate
            # Fallback to fixed when data unavailable
        return amount * self.slippage_rate

    def buy_cost(
        self,
        price: float,
        shares: int,
        daily_volatility: Optional[float] = None,
        adv: Optional[float] = None,
    ) -> float:
        """Total cost of buying (commission + slippage)."""
        amount = price * shares
        commission = max(amount * self.commission_rate, self.min_commission)
        slippage = self._slippage(amount, daily_volatility, adv)
        impact = amount * self.impact_rate
        return commission + slippage + impact

    def sell_cost(
        self,
        price: float,
        shares: int,
        daily_volatility: Optional[float] = None,
        adv: Optional[float] = None,
    ) -> float:
        """Total cost of selling (commission + stamp tax + slippage)."""
        amount = price * shares
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_tax = amount * self.stamp_tax_rate
        slippage = self._slippage(amount, daily_volatility, adv)
        impact = amount * self.impact_rate
        return commission + stamp_tax + slippage + impact

    def round_trip_rate(
        self,
        daily_volatility: Optional[float] = None,
        adv: Optional[float] = None,
        trade_value: Optional[float] = None,
    ) -> float:
        """Total cost rate for a complete buy+sell round trip.

        For fixed model (or when vol/adv not provided): returns static rate.
        For sqrt_adv model with vol/adv: returns estimated rate for given trade size.
        """
        if (
            self.impact_model == "sqrt_adv"
            and daily_volatility is not None
            and adv is not None
            and adv > 0
            and daily_volatility > 0
        ):
            tv = trade_value if trade_value is not None else adv * 0.01  # assume 1% ADV
            ratio = min(tv / adv, 1.0)
            slip_rate = daily_volatility * math.sqrt(ratio) * self.impact_coefficient
            return (
                self.commission_rate * 2
                + self.stamp_tax_rate
                + slip_rate * 2
                + self.impact_rate * 2
            )
        return (
            self.commission_rate * 2  # buy + sell commission
            + self.stamp_tax_rate  # sell stamp tax
            + self.slippage_rate * 2  # buy + sell slippage
            + self.impact_rate * 2  # buy + sell impact
        )

    def summary(self) -> str:
        rt = self.round_trip_rate()
        model_str = f"model={self.impact_model}"
        if self.impact_model == "sqrt_adv":
            model_str += f"(coeff={self.impact_coefficient})"
        return (
            f"CostModel: commission={self.commission_rate*10000:.1f}\u2031\u00d72, "
            f"stamp={self.stamp_tax_rate*10000:.1f}\u2031, "
            f"slippage={self.slippage_rate*10000:.1f}\u2031\u00d72, "
            f"{model_str}, "
            f"round_trip={rt*100:.3f}%"
        )
