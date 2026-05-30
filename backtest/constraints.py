"""Portfolio constraint definitions for turnover-constrained optimization."""
from dataclasses import dataclass, field


@dataclass
class PortfolioConstraints:
    """Constraints for portfolio optimization."""
    max_turnover: float = 0.20          # max sum(|w_new - w_old|) per rebalance
    max_single_weight: float = 0.05     # per-stock weight cap
    max_industry_deviation: float = 0.10  # active industry weight vs benchmark
    max_adv_participation: float = 0.03  # per-stock trade size / ADV
    min_hold_days: int = 2              # min days before selling
    cannot_sell: set = field(default_factory=set)
    cannot_buy: set = field(default_factory=set)
    industry_map: dict = field(default_factory=dict)   # stock -> industry
    adv: dict = field(default_factory=dict)             # stock -> ADV value
    volatility: dict = field(default_factory=dict)      # stock -> daily vol
    # Soft position-sizing multipliers from RiskGuard (e.g. crash_prob 0.50
    # → 0.5x, 0.70 → 0.25x). Applied to raw alpha-proportional weights
    # BEFORE the per-stock cap + turnover constraint so the cap can still
    # enforce diversification on the down-weighted name.
    reduce_weight: dict = field(default_factory=dict)   # stock -> multiplier in (0, 1]
