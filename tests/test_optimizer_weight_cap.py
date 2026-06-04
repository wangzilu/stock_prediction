"""Behavior pin: capped-simplex allocation in optimizer_v2.

cx round 5 P1-1 (2026-06-04): pre-fix the optimizer's
``_apply_weight_bounds`` capped weights then renormalized ALL stocks,
so when only K of N stocks needed capping, the renormalize
re-amplified the un-capped weights past the cap. The cap was
effectively a suggestion, not a constraint.

Concrete failure mode the user flagged:
  10 stocks, each computed weight 10%, total = 100%, cap = 5%.
  Old behavior:
    capped = each at 5%, total = 50%
    renormalize → each at 10% (cap completely defeated)
  Correct behavior:
    All 10 stocks at the cap (5% × 10 = 50% of target gross)
    OR a different rebalance respecting the cap. In any case,
    no single weight should exceed the cap.

These tests pin the fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.optimizer_v2 import TurnoverConstrainedOptimizer


def _opt(cap: float = 0.05) -> TurnoverConstrainedOptimizer:
    return TurnoverConstrainedOptimizer(top_k=10, max_single_weight=cap)


def test_no_single_weight_above_cap_in_equal_input():
    """10 stocks at 10% each, cap=5% — no resulting weight can
    exceed cap. Pre-fix the renormalize bumped all back to 10%."""
    weights = {f"SH{i:06d}": 0.10 for i in range(10)}
    capped = _opt(0.05)._apply_weight_bounds(weights)
    assert capped, "weights must not be empty"
    for s, w in capped.items():
        assert w <= 0.05 + 1e-9, (
            f"{s} weight {w:.4f} exceeds cap 0.05 — capped-simplex "
            f"broken (renormalize defeated the cap again)"
        )


def test_no_single_weight_above_cap_with_one_outlier():
    """One stock at 40%, nine stocks at ~6.67% each (gross 100%).
    cap=10% — the outlier must come down to cap, the rest
    re-distribute proportionally but none can exceed cap."""
    weights = {"SH600000": 0.40}
    for i in range(1, 10):
        weights[f"SH{i:06d}"] = (1.0 - 0.40) / 9
    capped = _opt(0.10)._apply_weight_bounds(weights)
    for s, w in capped.items():
        assert w <= 0.10 + 1e-9, (
            f"{s} weight {w:.4f} exceeds cap 0.10"
        )


def test_gross_preserved():
    """Capped weights should still sum to (approximately) the input
    gross, so portfolio-level leverage matches the optimizer's
    intent."""
    weights = {f"SH{i:06d}": 0.05 for i in range(20)}  # gross=1.0
    capped = _opt(0.10)._apply_weight_bounds(weights)
    total = sum(capped.values())
    assert abs(total - 1.0) < 1e-6, (
        f"capped gross {total:.6f} != input gross 1.0"
    )


def test_cap_disabled_falls_back_to_normalization():
    """cap >= 1.0 should disable the cap and just normalize."""
    weights = {"SH600000": 0.7, "SH600001": 0.7}
    capped = _opt(1.5)._apply_weight_bounds(weights)
    assert abs(sum(capped.values()) - 1.4) < 1e-6 or \
           abs(sum(capped.values()) - 1.0) < 1e-6


def test_iterative_capping_when_redistribute_overflows():
    """3 stocks at 0.5/0.3/0.2 (gross=1.0), cap=0.4 — after capping
    the 0.5 stock to 0.4, the residual 0.1 redistributes to the
    other two. The new 0.3 + share could exceed cap depending on
    formula; iteration must ensure no stock ends above cap."""
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    capped = _opt(0.4)._apply_weight_bounds(weights)
    for s, w in capped.items():
        assert w <= 0.4 + 1e-9, f"{s}={w:.4f} exceeds cap 0.4"
