"""Turnover-constrained portfolio optimizer (v2).

Heuristic "trade-toward-target" approach:
1. Compute unconstrained alpha-proportional target weights among top-N candidates
2. Compute desired trades (delta = target - current)
3. If total turnover exceeds max, scale all trades proportionally
4. Apply per-stock weight bounds and re-normalize

This is fast, deterministic, and avoids scipy convergence issues.
Can be upgraded to full SLSQP later.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from backtest.constraints import PortfolioConstraints
from backtest.cost_model import CostModel

logger = logging.getLogger(__name__)


class TurnoverConstrainedOptimizer:
    """Alpha-proportional portfolio with turnover constraint."""

    def __init__(
        self,
        top_k: int = 100,
        max_turnover: float = 0.20,
        max_single_weight: float = 0.05,
        min_weight: float = 0.002,
        lambda_cost: float = 0.0,
        cost_model: Optional[CostModel] = None,
        weight_method: str = "alpha_proportional",
    ):
        """
        Args:
            top_k: number of top candidates to consider
            max_turnover: max sum(|w_new - w_old|) per rebalance (one-way)
            max_single_weight: per-stock weight cap
            min_weight: minimum per-stock weight (positions below this are dropped)
            lambda_cost: cost penalty multiplier (0 = no cost awareness in target)
            cost_model: for cost-aware target adjustment
            weight_method: "equal" or "alpha_proportional"
        """
        self.top_k = top_k
        self.max_turnover = max_turnover
        self.max_single_weight = max_single_weight
        self.min_weight = min_weight
        self.lambda_cost = lambda_cost
        self.cost_model = cost_model
        self.weight_method = weight_method

    def optimize(
        self,
        alpha_scores: pd.Series,
        prev_weights: dict[str, float],
        constraints: Optional[PortfolioConstraints] = None,
        holding_days: Optional[dict[str, int]] = None,
    ) -> dict[str, float]:
        """Compute target weights with turnover constraint.

        Args:
            alpha_scores: Series indexed by instrument, model prediction scores
            prev_weights: current portfolio weights {instrument: weight}
            constraints: optional additional constraints
            holding_days: {instrument: days_held} for min hold enforcement

        Returns:
            dict {instrument: target_weight}, summing to ~1.0
        """
        if alpha_scores.empty:
            return dict(prev_weights)

        # Clean scores
        scores = alpha_scores.dropna().sort_values(ascending=False)

        # Step 1: Select candidate universe (top-K by score)
        candidates = scores.head(self.top_k)

        # Also keep current holdings that are in safe zone (top_k + buffer)
        buffer_k = int(self.top_k * 1.2)
        safe_zone = set(scores.head(buffer_k).index)

        # Force keep: stocks held < min_hold_days
        min_hold = 2
        if constraints:
            min_hold = constraints.min_hold_days
        force_keep = set()
        if holding_days:
            for stock, days in holding_days.items():
                if days < min_hold and stock in prev_weights:
                    force_keep.add(stock)

        # Cannot sell
        if constraints and constraints.cannot_sell:
            force_keep |= (constraints.cannot_sell & set(prev_weights.keys()))

        # Expand candidates to include force-keep stocks
        all_stocks = set(candidates.index) | force_keep
        # Also keep current holdings in safe zone
        for stock in prev_weights:
            if stock in safe_zone:
                all_stocks.add(stock)

        # Filter to stocks with valid scores
        all_stocks = [s for s in all_stocks if s in scores.index]
        if not all_stocks:
            return dict(prev_weights)

        stock_scores = scores.reindex(all_stocks).dropna()
        all_stocks = list(stock_scores.index)

        # Step 2: Compute unconstrained target weights
        if self.weight_method == "equal":
            raw_weights = {s: 1.0 / len(all_stocks) for s in all_stocks}
        else:
            # Alpha-proportional: shift scores to positive, then normalize
            shifted = stock_scores - stock_scores.min() + 1e-8
            total = shifted.sum()
            raw_weights = {s: float(shifted[s] / total) for s in all_stocks}

        # Step 2.5: RiskGuard soft penalty (crash_prob 0.5/0.7 tiers).
        # Multiply target weights by reduce_weight[code] (typically 0.25 or 0.5)
        # before bounds + turnover so a flagged stock can still appear in the
        # top-K universe but at a clamped size. Then renormalize so total
        # remains 1.0 (the freed weight gets redistributed to non-penalized
        # stocks proportionally via the renorm).
        if constraints and getattr(constraints, "reduce_weight", None):
            n_applied = 0
            for code, mult in constraints.reduce_weight.items():
                if code in raw_weights and 0.0 < float(mult) < 1.0:
                    raw_weights[code] *= float(mult)
                    n_applied += 1
            if n_applied:
                total = sum(raw_weights.values())
                if total > 0:
                    raw_weights = {s: w / total for s, w in raw_weights.items()}
                logger.info("optimizer_v2: applied reduce_weight to %d stocks", n_applied)

        # Apply per-stock weight cap
        raw_weights = self._apply_weight_bounds(raw_weights)

        # Step 3: Compute trades and apply turnover constraint
        target_weights = self._apply_turnover_constraint(
            raw_weights, prev_weights, force_keep
        )

        # Step 4: Drop tiny positions and re-normalize
        target_weights = {s: w for s, w in target_weights.items() if w >= self.min_weight}
        total_w = sum(target_weights.values())
        if total_w > 0:
            target_weights = {s: w / total_w for s, w in target_weights.items()}

        return target_weights

    def _apply_weight_bounds(self, weights: dict[str, float]) -> dict[str, float]:
        """Cap individual weights at ``max_single_weight`` using a
        proper capped-simplex allocation.

        2026-06-04 cx round 5 P1-1: pre-fix this did
            ``capped = {s: min(w, cap) for s, w in weights.items()}``
        then normalized ALL stocks back to sum=1. When only K of N
        stocks needed capping, the renormalize re-amplified the
        un-capped weights past the cap — e.g. 10 stocks each capped
        at 5% summed to 50%, after renormalize each became 10%, cap
        defeated.
        The correct treatment: fix the capped stocks at ``cap`` and
        re-distribute the residual weight proportionally to the
        uncapped stocks. If that re-distribution pushes any
        previously-uncapped stock above ``cap``, iterate: cap it,
        re-distribute again. The loop converges in at most N
        iterations because the capped set only grows.
        """
        cap = float(self.max_single_weight)
        total_target = sum(weights.values()) or 1.0  # preserve gross
        if cap <= 0 or cap >= 1.0:
            # cap disabled or pathological — fall back to normalization
            return {s: float(w) / total_target for s, w in weights.items()}

        capped_set: set[str] = set()
        result = {s: float(w) for s, w in weights.items()}

        for _ in range(len(result) + 1):  # safe upper bound
            # Stocks currently above cap (excluding already-capped set)
            over = {s for s in result if s not in capped_set and result[s] > cap}
            if not over:
                break
            # Cap them, then redistribute the released weight to the
            # remaining uncapped stocks proportionally.
            released = sum(result[s] - cap for s in over)
            capped_set |= over
            for s in over:
                result[s] = cap
            uncapped = [s for s in result if s not in capped_set]
            if not uncapped or released <= 0:
                break
            uncapped_sum = sum(result[s] for s in uncapped)
            if uncapped_sum <= 0:
                # Nothing to scale — distribute equally
                each = released / len(uncapped)
                for s in uncapped:
                    result[s] += each
            else:
                # Scale proportionally
                scale = (uncapped_sum + released) / uncapped_sum
                for s in uncapped:
                    result[s] *= scale

        # Final normalization to preserve the original gross. If the
        # capped portion alone exceeds total_target (e.g. cap=0.05 +
        # 25 stocks = 125%), shrink uniformly.
        gross = sum(result.values())
        if gross > 0:
            result = {s: w * total_target / gross for s, w in result.items()}
        # Re-clip after final scale — gross-conservation may push a
        # capped stock fractionally above cap when total_target < gross.
        result = {s: min(w, cap) for s, w in result.items()}
        return result

    def _apply_turnover_constraint(
        self,
        target: dict[str, float],
        prev: dict[str, float],
        force_keep: set,
    ) -> dict[str, float]:
        """Scale trades proportionally if total turnover exceeds max.

        Turnover = sum(|w_new_i - w_old_i|) / 2  (one-way)
        """
        # All stocks in either target or prev
        all_stocks = set(target.keys()) | set(prev.keys())

        # Compute desired deltas
        deltas = {}
        for s in all_stocks:
            w_new = target.get(s, 0.0)
            w_old = prev.get(s, 0.0)
            deltas[s] = w_new - w_old

        # Force keep: don't sell these stocks
        for s in force_keep:
            if s in deltas and deltas[s] < 0:
                deltas[s] = 0.0

        # Total turnover (one-way = half of two-way)
        total_turnover = sum(abs(d) for d in deltas.values()) / 2.0

        if total_turnover <= self.max_turnover:
            # Within budget — apply full trade
            result = {}
            for s in all_stocks:
                w = prev.get(s, 0.0) + deltas[s]
                if w > 0:
                    result[s] = w
            return result

        # Exceeds budget — scale trades proportionally
        scale = self.max_turnover / (total_turnover + 1e-10)
        result = {}
        for s in all_stocks:
            w = prev.get(s, 0.0) + deltas[s] * scale
            if w > 0:
                result[s] = w

        # Re-normalize
        total = sum(result.values())
        if total > 0:
            result = {s: w / total for s, w in result.items()}

        return result
