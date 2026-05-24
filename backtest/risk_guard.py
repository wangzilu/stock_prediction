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
import pandas as pd

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
                 # CX: cooldown by trigger reason
                 cooldown_price_days: int = 10,          # 普通价格止损
                 cooldown_event_days: int = 30,           # 连续跌停/监管/负面事件
                 cooldown_st_until_clear: bool = True,    # ST: 直到摘帽
                 # CX: drawdown recovery needs 5 consecutive days
                 drawdown_recovery_days: int = 5,
                 # State isolation: champion/shadow/backtest use separate dirs
                 state_dir: str = None):
        self.hard_stop_pct = hard_stop_pct
        self.vol_stop_multiplier = vol_stop_multiplier
        self.vol_stop_min = vol_stop_min
        self.vol_stop_max = vol_stop_max
        self.cooldown_price_days = cooldown_price_days
        self.cooldown_event_days = cooldown_event_days
        self.cooldown_st_until_clear = cooldown_st_until_clear
        self.drawdown_recovery_days = drawdown_recovery_days

        # Persistent state — isolated per state_dir to prevent
        # champion/shadow/backtest from polluting each other
        if state_dir:
            state_base = DATA_DIR / "paper" / state_dir
        else:
            state_base = DATA_DIR
        state_base.mkdir(parents=True, exist_ok=True)
        self._state_path = state_base / "risk_guard_state.json"
        self._state = self._load_state()

    def check(self, positions: dict, prices: dict, date: str,
              xgb_ranks: dict = None, regime: dict = None,
              events: dict = None,
              prev_closes: dict = None,
              crash_probs: dict = None) -> RiskConstraints:
        """Run all risk checks and return constraints.

        Args:
            positions: {code: {"shares": N, "avg_price": P, "holding_days": D}}
            prices: {code: current_price}
            date: current date string
            xgb_ranks: {code: rank_in_universe} — for soft exit logic
            regime: regime controller output dict
            events: {code: impact} — LLM event alphas for today
            prev_closes: {code: prev_close_price} — for limit-down check
            crash_probs: {code: crash_prob_5d} — crash model output (optional)
        """
        constraints = RiskConstraints()

        # === L1: Stock-level checks ===
        self._check_st_stocks(positions, constraints, date)
        self._check_hard_stop(positions, prices, constraints, date, xgb_ranks)
        # CX: trailing stop / profit giveback 暂缓，第一版不启用
        # self._check_profit_giveback(positions, prices, constraints, date, xgb_ranks, events)
        self._check_limit_down(positions, prices, constraints, prev_closes)
        self._check_pending_exits(constraints, date)
        self._apply_cooldowns(constraints, date)

        # === L1.5: Crash model checks ===
        if crash_probs is not None:
            self._check_crash_risk(positions, crash_probs, date, constraints)

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

    def _get_dynamic_stop(self, code: str) -> float:
        """CX: ATR/vol dynamic threshold with clip(0.12, 0.25).

        Uses vol20 from feature cache if available, else hard_stop_pct.
        """
        try:
            cache = pd.read_parquet(
                DATA_DIR / "feature_cache_174_holder_regime_ma.parquet",
                columns=["STD20"],
            )
            dates = cache.index.get_level_values(0)
            latest = dates.max()
            code_lower = code.lower()
            if (latest, code_lower) in cache.index:
                vol20 = float(cache.loc[(latest, code_lower), "STD20"])
                if np.isfinite(vol20) and vol20 > 0:
                    # CX formula: -clip(4 * vol20, 0.12, 0.25)
                    return -max(0.12, min(0.25, self.vol_stop_multiplier * vol20))
        except Exception:
            pass
        return self.hard_stop_pct  # fallback: -0.20

    def _check_hard_stop(self, positions, prices, constraints, date, xgb_ranks):
        """Check individual stock hard stop loss with dynamic threshold."""
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
            threshold = self._get_dynamic_stop(code)

            if pnl_pct < threshold:
                xgb_rank = (xgb_ranks or {}).get(code, 9999)
                if xgb_rank <= 50:
                    # CX: XGB still top 50 → soft exit only
                    constraints.pending_exit.append(code)
                    constraints.risk_reasons[code] = (
                        f"浮亏{pnl_pct:.1%}(阈值{threshold:.1%})但XGB排名{xgb_rank}仍高，标记观察"
                    )
                elif xgb_rank <= 200:
                    # CX: soft_exit = 浮亏大 + XGB跌出Top200 + 无正面事件
                    constraints.pending_exit.append(code)
                    constraints.risk_reasons[code] = (
                        f"浮亏{pnl_pct:.1%}+XGB排名{xgb_rank}，软退出"
                    )
                else:
                    # XGB very weak → force sell
                    constraints.force_sell.append(code)
                    constraints.risk_reasons[code] = (
                        f"浮亏{pnl_pct:.1%}+XGB排名{xgb_rank}，强制退出"
                    )
                    # CX: cooldown by reason — price stop = 10 days
                    cooldown_end = (datetime.strptime(date, "%Y-%m-%d") +
                                    timedelta(days=self.cooldown_price_days)).strftime("%Y-%m-%d")
                    self._state.setdefault("cooldowns", {})[code] = cooldown_end

    def _check_profit_giveback(self, positions, prices, constraints, date,
                               xgb_ranks, events):
        """Soft trailing stop: profit giveback + weak signal → reduce weight.

        Triggers when ALL three conditions met:
          1. Current price dropped > 15% from holding-period high
          2. XGB rank dropped below Top100
          3. No positive LLM event

        Action: not force sell, but mark for weight reduction (half weight).
        If continues next day → force sell.
        """
        for code, pos in positions.items():
            if code in constraints.force_sell:
                continue

            current_price = prices.get(code)
            if not current_price or current_price <= 0:
                continue

            avg_price = pos.get("avg_price", current_price)
            if avg_price <= 0:
                continue

            # Track holding-period high
            peak_key = f"peak_{code}"
            peak = self._state.get(peak_key, avg_price)
            if current_price > peak:
                peak = current_price
                self._state[peak_key] = peak

            # Condition 1: drawdown from peak > 15%
            dd_from_peak = (current_price - peak) / peak
            if dd_from_peak >= -0.15:
                continue  # not enough drawdown

            # Only trigger if was profitable (peak > avg_price * 1.05)
            if peak <= avg_price * 1.05:
                continue  # was never significantly profitable

            # Condition 2: XGB rank weak
            xgb_rank = (xgb_ranks or {}).get(code, 9999)
            if xgb_rank <= 100:
                continue  # XGB still likes it

            # Condition 3: no positive event
            event_impact = (events or {}).get(code, 0)
            if event_impact > 0:
                continue  # positive event, don't trigger

            # All three conditions met → soft exit
            profit_pct = (current_price - avg_price) / avg_price
            constraints.pending_exit.append(code)
            constraints.risk_reasons[code] = (
                f"利润回吐: 高点{peak:.2f}→现价{current_price:.2f} "
                f"(回撤{dd_from_peak:.1%}), XGB排名{xgb_rank}, "
                f"仍盈利{profit_pct:.1%}"
            )

            # Check if this is second consecutive day of giveback trigger
            prev_giveback = self._state.get(f"giveback_{code}")
            if prev_giveback and prev_giveback == self._state.get("last_update"):
                # Second day → force sell
                constraints.force_sell.append(code)
                constraints.risk_reasons[code] += " → 连续触发，强制卖出"
                # Remove from pending
                constraints.pending_exit = [c for c in constraints.pending_exit if c != code]
                # Cooldown
                cooldown_end = (datetime.strptime(date, "%Y-%m-%d") +
                                timedelta(days=self.cooldown_price_days)).strftime("%Y-%m-%d")
                self._state.setdefault("cooldowns", {})[code] = cooldown_end
            else:
                self._state[f"giveback_{code}"] = date

    @staticmethod
    def _limit_pct_for_code(code: str) -> float:
        """Return the limit-down percentage for a given stock code.

        A-share rules:
          - 创业板 (30xxxx): 20%
          - 科创板 (688xxx): 20%
          - ST stocks (name-based, but code heuristic via caller): 5%
          - Normal main board: 10%
        """
        # Normalise: accept sh600000 / SH600000 / 600000.SH etc.
        c = code.lower().replace(".", "")
        # Extract the 6-digit numeric portion
        digits = ""
        for ch in c:
            if ch.isdigit():
                digits += ch
        if len(digits) < 6:
            return 0.10  # fallback
        d6 = digits[:6]
        if d6.startswith("30"):
            return 0.20  # 创业板
        if d6.startswith("688"):
            return 0.20  # 科创板
        return 0.10      # 主板 default

    def _check_limit_down(self, positions, prices, constraints,
                          prev_closes: dict = None):
        """Mark limit-down stocks as cannot_sell.

        A stock is at limit-down when:
            current_price <= prev_close * (1 - limit_pct) + tolerance

        Args:
            positions: {code: pos_dict}
            prices: {code: current_price}
            constraints: RiskConstraints to mutate
            prev_closes: {code: previous_close_price}
                         If not supplied, fall back to pos["prev_close"]
                         or pos["avg_price"] as rough proxy.
        """
        TOLERANCE = 0.001  # 0.1% tolerance for floating-point / tick rounding

        # Also detect ST via the st_stock_list
        st_set: set = set()
        try:
            st_path = DATA_DIR / "st_stock_list.json"
            if st_path.exists():
                st_set = set(json.loads(st_path.read_text()))
        except Exception:
            pass

        for code, pos in positions.items():
            current_price = prices.get(code)
            if not current_price or current_price <= 0:
                continue

            # Determine previous close
            prev_close = None
            if prev_closes:
                prev_close = prev_closes.get(code)
            if prev_close is None:
                prev_close = pos.get("prev_close")
            if prev_close is None:
                # Last-resort fallback: use avg_price (rough proxy)
                prev_close = pos.get("avg_price")
            if not prev_close or prev_close <= 0:
                continue

            # Determine limit percentage
            code_lower = code.lower()
            if code_lower in st_set:
                limit_pct = 0.05  # ST stocks: 5%
            else:
                limit_pct = self._limit_pct_for_code(code)

            limit_down_price = prev_close * (1.0 - limit_pct)

            if current_price <= limit_down_price * (1.0 + TOLERANCE):
                constraints.cannot_sell.add(code)
                constraints.risk_reasons.setdefault(
                    code,
                    f"跌停(现价{current_price:.2f}≤跌停价{limit_down_price:.2f}, "
                    f"限制{limit_pct:.0%})"
                )

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

    # ---- L1.5: Crash model ----

    def check_crash_risk(self, positions: dict, crash_probs: dict,
                         date: str) -> list:
        """Standalone crash risk check — returns list of flagged stocks.

        Args:
            positions: {code: pos_dict} — currently held positions
            crash_probs: {code: crash_prob_5d} — crash model output
            date: current date string

        Returns:
            List of dicts with flagged stock info:
              [{"code": ..., "crash_prob": ..., "action": "block"/"exit",
                "reason": ...}, ...]
        """
        flagged = []
        for code, prob in crash_probs.items():
            if not np.isfinite(prob):
                continue
            if prob > 0.80 and code in positions:
                flagged.append({
                    "code": code,
                    "crash_prob": prob,
                    "action": "exit",
                    "reason": f"崩盘概率{prob:.1%}>80%，建议退出",
                })
            elif prob > 0.65:
                flagged.append({
                    "code": code,
                    "crash_prob": prob,
                    "action": "block",
                    "reason": f"崩盘概率{prob:.1%}>65%，禁止买入",
                })
        return flagged

    def _check_crash_risk(self, positions: dict, crash_probs: dict,
                          date: str, constraints: RiskConstraints):
        """Wire crash model into RiskConstraints (called from check()).

        - crash_prob > 0.65 → cannot_buy
        - crash_prob > 0.80 and held → pending_exit
        """
        for code, prob in crash_probs.items():
            if not np.isfinite(prob):
                continue
            if prob > 0.65:
                constraints.cannot_buy.add(code)
                if prob > 0.80 and code in positions:
                    constraints.pending_exit.append(code)
                    constraints.risk_reasons[code] = (
                        f"崩盘概率{prob:.1%}>80%，标记退出"
                    )
                else:
                    constraints.risk_reasons.setdefault(
                        code,
                        f"崩盘概率{prob:.1%}>65%，禁止买入"
                    )

    # ---- L2: Portfolio drawdown state machine ----

    def _check_drawdown(self, constraints, date):
        """Portfolio drawdown state machine with recovery conditions.

        States: normal → watch → derisk → emergency
        Downgrade: immediate on threshold breach
        Upgrade: requires N consecutive days above threshold (CX: 5 days)
        """
        dd = self._state.get("drawdown_pct", 0.0)
        prev_state = self._state.get("drawdown_state", "normal")
        recovery_count = self._state.get("recovery_count", 0)

        STATE_ORDER = {"normal": 0, "watch": 1, "derisk": 2, "emergency": 3}
        STATE_CONFIG = {
            "emergency": {"threshold": -0.18, "max_pos": 0.3},
            "derisk":    {"threshold": -0.12, "max_pos": 0.6},
            "watch":     {"threshold": -0.08, "max_pos": 0.85},
            "normal":    {"threshold": 0.0,   "max_pos": 1.0},
        }

        # Determine raw state from current drawdown
        if dd < -0.18:
            raw_state = "emergency"
        elif dd < -0.12:
            raw_state = "derisk"
        elif dd < -0.08:
            raw_state = "watch"
        else:
            raw_state = "normal"

        # Downgrade: immediate
        if STATE_ORDER.get(raw_state, 0) > STATE_ORDER.get(prev_state, 0):
            new_state = raw_state
            recovery_count = 0
        # Upgrade: requires consecutive recovery days
        elif STATE_ORDER.get(raw_state, 0) < STATE_ORDER.get(prev_state, 0):
            recovery_count += 1
            if recovery_count >= self.drawdown_recovery_days:
                # CX: upgrade one level at a time, not jump to normal
                state_list = ["emergency", "derisk", "watch", "normal"]
                prev_idx = state_list.index(prev_state) if prev_state in state_list else 0
                new_state = state_list[min(prev_idx + 1, 3)]
                recovery_count = 0
            else:
                new_state = prev_state  # stay in current state until recovery confirmed
        else:
            new_state = prev_state
            recovery_count = 0

        constraints.max_gross_position = STATE_CONFIG[new_state]["max_pos"]
        constraints.drawdown_state = new_state
        constraints.drawdown_pct = dd

        self._state["drawdown_state"] = new_state
        self._state["recovery_count"] = recovery_count

        if new_state != "normal":
            logger.warning(
                f"RiskGuard: state={new_state} (was {prev_state}), dd={dd:.1%}, "
                f"max_pos={constraints.max_gross_position:.0%}, recovery={recovery_count}/{self.drawdown_recovery_days}"
            )

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
