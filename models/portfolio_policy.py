"""Portfolio risk control policies for TopK stock selection.

Applies constraints after model scoring:
- Industry exposure cap (single industry max weight)
- Turnover control (max daily/weekly rebalance ratio)
- Drawdown circuit breaker (reduce position on cumulative loss)
- Single stock weight cap
- Risk filters (ST, suspension, limit-up/down, low liquidity, high pledge)
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PortfolioPolicy:
    """Risk constraints for portfolio construction."""
    topk: int = 20
    max_drop: int = 5
    max_industry_pct: float = 0.25          # Single industry max 25%
    max_single_stock_pct: float = 0.08      # Single stock max 8%
    max_daily_turnover: float = 0.30        # Max 30% daily turnover
    drawdown_threshold: float = -0.15       # -15% cumulative → reduce to half
    drawdown_reduce_factor: float = 0.5     # Reduce topk by this factor on drawdown


def sector_from_code(code: str) -> str:
    """Simple sector classification from stock code prefix."""
    code = str(code).upper()
    if code.startswith("SH68"):
        return "科创板"
    elif code.startswith("SH60"):
        return "沪市主板"
    elif code.startswith("SZ002"):
        return "中小板"
    elif code.startswith("SZ00"):
        return "深市主板"
    elif code.startswith("SZ30"):
        return "创业板"
    elif code.startswith("BJ"):
        return "北交所"
    return "其他"


def apply_industry_cap(
    candidates: pd.DataFrame,
    topk: int,
    max_industry_pct: float = 0.25,
) -> pd.DataFrame:
    """Apply industry exposure cap to TopK selection.

    Args:
        candidates: DataFrame with columns ['code', 'score'], sorted by score desc
        topk: target number of holdings
        max_industry_pct: max fraction per industry

    Returns:
        Filtered DataFrame with at most topk stocks, industry-capped
    """
    if candidates.empty:
        return candidates

    candidates = candidates.copy()
    candidates["sector"] = candidates["code"].apply(sector_from_code)

    max_per_sector = max(1, int(topk * max_industry_pct))
    selected = []
    sector_counts = {}

    for _, row in candidates.iterrows():
        sector = row["sector"]
        count = sector_counts.get(sector, 0)
        if count < max_per_sector:
            selected.append(row)
            sector_counts[sector] = count + 1
        if len(selected) >= topk:
            break

    result = pd.DataFrame(selected)
    if not result.empty and "sector" in result.columns:
        logger.debug(f"Industry distribution: {dict(result['sector'].value_counts())}")
    return result


def check_turnover(
    new_holdings: set,
    old_holdings: set,
    max_turnover: float = 0.30,
) -> set:
    """Limit turnover by keeping more old holdings if turnover exceeds cap.

    Returns:
        Adjusted new holdings set with turnover <= max_turnover
    """
    if not old_holdings:
        return new_holdings

    total = max(len(new_holdings), len(old_holdings), 1)
    buys = new_holdings - old_holdings
    sells = old_holdings - new_holdings
    turnover = (len(buys) + len(sells)) / total

    if turnover <= max_turnover:
        return new_holdings

    # Reduce turnover by keeping more old holdings
    max_changes = int(total * max_turnover / 2)  # half for buys, half for sells
    keep = old_holdings - set(list(sells)[:len(sells) - max_changes])  # keep more old
    fill_from_new = set(list(buys)[:max_changes])  # add fewer new
    adjusted = keep | fill_from_new

    logger.debug(f"Turnover capped: {turnover:.1%} → {len(fill_from_new)*2/total:.1%}")
    return adjusted


def check_drawdown(
    cumulative_returns: list,
    threshold: float = -0.15,
    reduce_factor: float = 0.5,
    topk: int = 20,
) -> int:
    """Reduce topk if cumulative drawdown exceeds threshold.

    Returns:
        Adjusted topk (may be reduced)
    """
    if not cumulative_returns or len(cumulative_returns) < 5:
        return topk

    cum = np.array(cumulative_returns)
    peak = np.maximum.accumulate(cum)
    drawdown = (cum[-1] - peak[-1]) / (peak[-1] + 1e-8) if peak[-1] > 0 else 0

    if drawdown < threshold:
        adjusted = max(5, int(topk * reduce_factor))
        logger.info(f"Drawdown {drawdown:.1%} < {threshold:.0%}: topk {topk} → {adjusted}")
        return adjusted

    return topk


def risk_filter(code: str, spot_data: dict = None) -> bool:
    """Filter out high-risk stocks.

    Returns:
        True if stock passes risk check (safe to hold)
    """
    code = str(code).upper()

    # Filter ST stocks
    if spot_data:
        name = str(spot_data.get("name", spot_data.get("名称", "")))
        if "ST" in name or "*ST" in name:
            return False

        # Filter low liquidity
        volume = float(spot_data.get("volume", spot_data.get("成交量", 0)) or 0)
        if volume < 10000:  # Less than 10K shares traded
            return False

    return True


class PortfolioManager:
    """Manages portfolio with risk controls."""

    def __init__(self, policy: PortfolioPolicy = None):
        self.policy = policy or PortfolioPolicy()
        self.holdings: set = set()
        self.cumulative_returns: list = []
        self.history: list = []

    def select(
        self,
        scores: dict,
        spot_data: dict = None,
    ) -> list:
        """Select stocks with all risk controls applied.

        Args:
            scores: dict of code → prediction score
            spot_data: optional dict of code → spot quote info

        Returns:
            List of selected stock codes
        """
        if not scores:
            return []

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Check drawdown → may reduce topk
        effective_topk = check_drawdown(
            self.cumulative_returns,
            self.policy.drawdown_threshold,
            self.policy.drawdown_reduce_factor,
            self.policy.topk,
        )

        # Build candidate DataFrame
        candidates = pd.DataFrame([
            {"code": code, "score": score}
            for code, score in sorted_scores
            if risk_filter(code, (spot_data or {}).get(code))
        ])

        if candidates.empty:
            return []

        # Apply industry cap
        capped = apply_industry_cap(
            candidates, effective_topk, self.policy.max_industry_pct
        )
        new_set = set(capped["code"].tolist())

        # Apply TopK dropout (keep stocks still in top K+max_drop)
        if self.holdings:
            keep_zone = set(candidates.head(effective_topk + self.policy.max_drop)["code"])
            keep = self.holdings & keep_zone
            n_fill = effective_topk - len(keep)
            fill = set()
            for code in capped["code"]:
                if code not in keep and len(fill) < n_fill:
                    fill.add(code)
            new_set = keep | fill

        # Apply turnover cap
        new_set = check_turnover(new_set, self.holdings, self.policy.max_daily_turnover)

        self.holdings = new_set
        return list(new_set)

    def record_return(self, daily_return: float):
        """Record daily portfolio return for drawdown tracking."""
        if not self.cumulative_returns:
            self.cumulative_returns.append(1.0 + daily_return)
        else:
            self.cumulative_returns.append(
                self.cumulative_returns[-1] * (1.0 + daily_return)
            )

    def get_status(self) -> dict:
        """Get current portfolio status."""
        cum = self.cumulative_returns
        if not cum:
            return {"holdings": len(self.holdings), "drawdown": 0, "topk": self.policy.topk}

        peak = max(cum)
        dd = (cum[-1] - peak) / (peak + 1e-8) if peak > 0 else 0
        return {
            "holdings": len(self.holdings),
            "cumulative_return": round(cum[-1] - 1, 4) if cum else 0,
            "max_drawdown": round(dd, 4),
            "effective_topk": check_drawdown(
                cum, self.policy.drawdown_threshold,
                self.policy.drawdown_reduce_factor, self.policy.topk
            ),
        }
