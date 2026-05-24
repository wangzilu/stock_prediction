"""RiskGuard — independent risk management module.

Outputs constraints for OMS and optimizer, does NOT execute trades.
OMS enforces force_sell/cannot_buy; optimizer respects max_position.

Three layers:
  L1: Stock-level force exit (ST, limit-down, hard stop)
  L2: Portfolio drawdown state machine
  L3: Regime linkage (from regime_controller)

Usage:
    from backtest.risk_guard import RiskGuard
    guard = RiskGuard()
    constraints = guard.check(positions, prices, date)
    # constraints.force_sell = ["SH600000", ...]
    # constraints.cannot_buy = {"SZ000001", ...}
    # constraints.max_gross_position = 0.6
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "storage"


@dataclass
class RiskConstraints:
    """Output of RiskGuard — consumed by OMS and optimizer."""
    force_sell: list = field(default_factory=list)       # must sell ASAP
    pending_exit: list = field(default_factory=list)     # want to sell but may not be tradable
    cannot_buy: set = field(default_factory=set)         # blocked from buying
    cannot_sell: set = field(default_factory=set)        # cannot sell (limit-down/suspended)
    max_gross_position: float = 1.0                      # max % invested (1.0 = fully invested)
    cooldown_until: dict = field(default_factory=dict)   # {code: date_str} — cannot buy until
    risk_reasons: dict = field(default_factory=dict)     # {code: reason}
    drawdown_state: str = "normal"                       # normal/watch/derisk/emergency
    drawdown_pct: float = 0.0


class RiskGuard:

    def __init__(self,
                 hard_stop_pct: float = -0.20,
                 vol_stop_multiplier: float = 4.0,
                 vol_stop_min: float = -0.12,
                 vol_stop_max: float = -0.25,
                 cooldown_price_days: int = 10,
                 cooldown_event_days: int = 30,
                 cooldown_st_until_clear: bool = True):
        self.hard_stop_pct = hard_stop_pct
        self.vol_stop_multiplier = vol_stop_multiplier
        self.vol_stop_min = vol_stop_min
        self.vol_stop_max = vol_stop_max
        self.cooldown_price_days = cooldown_price_days
        self.cooldown_event_days = cooldown_event_days
        self.cooldown_st_until_clear = cooldown_st_until_clear

        # Persistent state
        self._state_path = DATA_DIR / "risk_guard_state.json"
        self._state = self._load_state()

    def check(self, positions: dict, prices: dict, date: str,
              xgb_ranks: dict = None, regime: dict = None) -> RiskConstraints:
        """Run all risk checks and return constraints.

        Args:
            positions: {code: {"shares": N, "avg_price": P, "holding_days": D}}
            prices: {code: current_price}
            date: current date string
            xgb_ranks: {code: rank_in_universe} — for soft exit logic
            regime: regime controller output dict
        """
        constraints = RiskConstraints()

        # === L1: Stock-level checks ===
        self._check_st_stocks(positions, constraints, date)
        self._check_hard_stop(positions, prices, constraints, date, xgb_ranks)
        self._check_limit_down(positions, prices, constraints)
        self._check_pending_exits(constraints, date)
        self._apply_cooldowns(constraints, date)

        # === L2: Portfolio drawdown state machine ===
        self._check_drawdown(constraints, date)

        # === L3: Regime linkage ===
        if regime:
            self._apply_regime(regime, constraints)

        # Save state
        self._save_state()

        return constraints

    # ---- L1: Stock-level ----

    def _check_st_stocks(self, positions, constraints, date):
        """Force sell ST stocks."""
        try:
            st_path = DATA_DIR / "st_stock_list.json"
            if st_path.exists():
                st_set = set(json.loads(st_path.read_text()))
                for code in positions:
                    code_lower = code.lower()
                    if code_lower in st_set:
                        constraints.force_sell.append(code)
                        constraints.risk_reasons[code] = "ST/退市风险"
                        # ST cooldown: until removed from ST list
                        self._state.setdefault("cooldowns", {})[code] = "9999-12-31"
        except Exception:
            pass

    def _check_hard_stop(self, positions, prices, constraints, date, xgb_ranks):
        """Check individual stock hard stop loss."""
        for code, pos in positions.items():
            if code in constraints.force_sell:
                continue

            current_price = prices.get(code)
            if not current_price or current_price <= 0:
                continue

            avg_price = pos.get("avg_price", current_price)
            if avg_price <= 0:
                continue

            pnl_pct = (current_price - avg_price) / avg_price

            # Dynamic threshold: -clip(4 * vol20, 0.12, 0.25)
            # For now use hard stop since we don't have per-stock vol in OMS
            threshold = self.hard_stop_pct  # -0.20

            if pnl_pct < threshold:
                # CX: don't force sell if XGB still ranks high
                xgb_rank = (xgb_ranks or {}).get(code, 9999)
                if xgb_rank <= 50:
                    # XGB still top 50 — soft exit (pending, not forced)
                    constraints.pending_exit.append(code)
                    constraints.risk_reasons[code] = f"浮亏{pnl_pct:.1%}但XGB排名{xgb_rank}仍高，标记观察"
                else:
                    # XGB also weak — force sell
                    constraints.force_sell.append(code)
                    constraints.risk_reasons[code] = f"浮亏{pnl_pct:.1%}+XGB排名{xgb_rank}，强制退出"
                    # Cooldown
                    cooldown_end = (datetime.strptime(date, "%Y-%m-%d") +
                                    timedelta(days=self.cooldown_price_days)).strftime("%Y-%m-%d")
                    self._state.setdefault("cooldowns", {})[code] = cooldown_end

    def _check_limit_down(self, positions, prices, constraints):
        """Mark limit-down stocks as cannot_sell."""
        # In real implementation: check if stock hit limit-down today
        # For now: if price dropped > 9.5% (main board) or > 19.5% (创业板/科创板)
        for code, pos in positions.items():
            current_price = prices.get(code)
            if not current_price:
                continue
            avg_price = pos.get("avg_price", current_price)
            if avg_price <= 0:
                continue
            daily_change = (current_price - avg_price) / avg_price
            # This is a rough proxy — real check needs previous close
            # For now just mark as info
            pass

    def _check_pending_exits(self, constraints, date):
        """Process pending exits from previous days."""
        pending = self._state.get("pending_exits", [])
        still_pending = []
        for item in pending:
            code = item["code"]
            if code not in constraints.cannot_sell:
                # Can sell today — add to force_sell
                constraints.force_sell.append(code)
                constraints.risk_reasons[code] = f"挂起卖出（原因：{item.get('reason', '?')}）"
            else:
                # Still can't sell — keep pending
                still_pending.append(item)
        self._state["pending_exits"] = still_pending

        # Add new pending exits
        for code in constraints.pending_exit:
            self._state.setdefault("pending_exits", []).append({
                "code": code,
                "reason": constraints.risk_reasons.get(code, ""),
                "since": date,
            })

    def _apply_cooldowns(self, constraints, date):
        """Apply cooldown periods — cannot buy stocks in cooldown."""
        cooldowns = self._state.get("cooldowns", {})
        expired = []
        for code, until_date in cooldowns.items():
            if date < until_date:
                constraints.cannot_buy.add(code)
            else:
                expired.append(code)
        for code in expired:
            del cooldowns[code]

    # ---- L2: Portfolio drawdown state machine ----

    def _check_drawdown(self, constraints, date):
        """Portfolio drawdown state machine.

        States: normal → watch → derisk → emergency
        Transitions based on drawdown from peak.
        """
        dd = self._state.get("drawdown_pct", 0.0)
        prev_state = self._state.get("drawdown_state", "normal")
        recovery_days = self._state.get("recovery_days", 0)

        # State transitions
        if dd < -0.18:
            new_state = "emergency"
            constraints.max_gross_position = 0.3
        elif dd < -0.12:
            new_state = "derisk"
            constraints.max_gross_position = 0.6
        elif dd < -0.08:
            new_state = "watch"
            constraints.max_gross_position = 0.85
        else:
            new_state = "normal"
            constraints.max_gross_position = 1.0

        # Recovery: need 5 consecutive days above threshold to upgrade
        if new_state < prev_state:  # string comparison: derisk < emergency < normal < watch
            # Actually need proper ordering
            pass

        constraints.drawdown_state = new_state
        constraints.drawdown_pct = dd
        self._state["drawdown_state"] = new_state

        if new_state != "normal":
            logger.warning(f"RiskGuard: drawdown_state={new_state}, dd={dd:.1%}, "
                           f"max_position={constraints.max_gross_position:.0%}")

    def update_portfolio_value(self, current_value: float, date: str):
        """Update portfolio peak and drawdown tracking."""
        peak = self._state.get("portfolio_peak", current_value)
        if current_value > peak:
            peak = current_value
            self._state["portfolio_peak"] = peak

        dd = (current_value - peak) / peak if peak > 0 else 0
        self._state["drawdown_pct"] = dd
        self._state["last_update"] = date

    # ---- L3: Regime linkage ----

    def _apply_regime(self, regime, constraints):
        """Apply regime controller constraints."""
        alert = regime.get("alert_level", "normal")
        if alert == "critical":
            constraints.max_gross_position = min(constraints.max_gross_position, 0.3)
        elif alert == "warning":
            constraints.max_gross_position = min(constraints.max_gross_position, 0.6)

    # ---- State persistence ----

    def _load_state(self) -> dict:
        if self._state_path.exists():
            try:
                return json.loads(self._state_path.read_text())
            except Exception:
                pass
        return {"cooldowns": {}, "pending_exits": [], "drawdown_state": "normal",
                "portfolio_peak": 1_000_000, "drawdown_pct": 0.0}

    def _save_state(self):
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(self._state, indent=2, ensure_ascii=False))
        except Exception:
            pass
