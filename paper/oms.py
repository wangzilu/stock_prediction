"""Paper Trading OMS (Order Management System).

Simulates: signal → target portfolio → orders → fills → positions → PnL.

Supports two modes:
- "legacy": original single-step flow (generate + fill in one call)
- "pending": two-step pending-order model
    Step 1 (T close): generate_orders() — creates pending orders based on signals
    Step 2 (T+1 open): reconcile() — fills orders at actual T+1 open prices

Daily flow (legacy):
1. Load model predictions (from lgb_latest_predictions.json)
2. Generate target portfolio (buffered_partial logic)
3. Compare with current positions → generate orders
4. Simulate fills (T+1, limit-up/down/suspended check)
5. Update positions and PnL
6. Write daily ledger

Usage:
    from paper.oms import PaperOMS
    oms = PaperOMS()
    oms.run_daily()  # legacy mode (default)

    # Or use pending order model:
    oms = PaperOMS(mode="pending")
    oms.generate_orders("2026-05-24")   # after T close
    oms.reconcile("2026-05-24")         # after T+1 open
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Qlib init deferred to method call to avoid import at module load

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "storage"
PAPER_DIR = DATA_DIR / "paper"


class Position:
    """Single stock position."""
    def __init__(self, code: str, shares: int, avg_price: float, entry_date: str):
        self.code = code
        self.shares = shares
        self.avg_price = avg_price
        self.entry_date = entry_date
        self.holding_days = 0
        self.unrealized_pnl = 0.0

    def to_dict(self):
        return {
            "code": self.code,
            "shares": self.shares,
            "avg_price": round(self.avg_price, 4),
            "entry_date": self.entry_date,
            "holding_days": self.holding_days,
            "unrealized_pnl": round(self.unrealized_pnl, 4),
        }


class PaperOMS:
    """Unified paper trading OMS — supports both buffered_partial and optimizer_v2.

    Args:
        execution_mode: "buffered_partial" or "optimizer_v2"
        state_dir: directory for state/trades files (allows champion/shadow separation)
    """

    def __init__(self,
                 initial_capital: float = 1_000_000,
                 top_k: int = 20,
                 buffer: int = 5,
                 trade_rate: float = 0.35,
                 min_hold_days: int = 2,
                 max_daily_turnover: float = 0.15,
                 commission_rate: float = 0.0003,
                 stamp_tax_rate: float = 0.0005,
                 slippage_rate: float = 0.001,
                 execution_mode: str = "buffered_partial",
                 max_turnover: float = 0.10,
                 max_single_weight: float = 0.05,
                 weight_method: str = "alpha_proportional",
                 state_dir: str = None,
                 mode: str = "legacy"):

        self.initial_capital = initial_capital
        self.top_k = top_k
        self.buffer = buffer
        self.trade_rate = trade_rate
        self.min_hold_days = min_hold_days
        self.max_daily_turnover = max_daily_turnover
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage_rate = slippage_rate
        self.execution_mode = execution_mode
        self.max_turnover = max_turnover
        self.max_single_weight = max_single_weight
        self.weight_method = weight_method
        self.mode = mode  # "legacy" or "pending"

        # State directory (separate for champion vs shadow)
        if state_dir:
            self._state_dir = Path(state_dir)
        else:
            self._state_dir = PAPER_DIR
        self._state_dir.mkdir(parents=True, exist_ok=True)

        PAPER_DIR.mkdir(parents=True, exist_ok=True)

        # Load state
        self.state = self._load_state()

    def _state_path(self):
        return self._state_dir / "oms_state.json"

    def _load_state(self) -> dict:
        path = self._state_path()
        if path.exists():
            return json.loads(path.read_text())
        return {
            "cash": self.initial_capital,
            "positions": {},
            "total_value": self.initial_capital,
            "trade_count": 0,
            "start_date": datetime.now().strftime("%Y-%m-%d"),
            "last_update": None,
            "daily_pnl_history": [],
        }

    def _save_state(self):
        self.state["last_update"] = datetime.now().isoformat(timespec="seconds")
        path = self._state_path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False))
        os.replace(tmp, path)

    def load_predictions(self) -> dict:
        """Load latest model predictions."""
        cache_path = DATA_DIR / "lgb_latest_predictions.json"
        if not cache_path.exists():
            logger.warning("No prediction cache found")
            return {}
        payload = json.loads(cache_path.read_text())
        return payload.get("predictions", {})

    def get_current_positions(self) -> dict:
        """Get current positions as {code: Position}."""
        positions = {}
        for code, info in self.state.get("positions", {}).items():
            pos = Position(code, info["shares"], info["avg_price"], info["entry_date"])
            pos.holding_days = info.get("holding_days", 0)
            positions[code] = pos
        return positions

    def generate_target(self, predictions: dict) -> list:
        """Generate target portfolio.

        For buffered_partial: returns (target_list, sells, buys)
        For optimizer_v2: returns (target_list, sells, buys) derived from weight changes
        """
        if not predictions:
            return list(self.state.get("positions", {}).keys()), [], []

        if self.execution_mode == "optimizer_v2":
            return self._generate_target_optimizer(predictions)

        return self._generate_target_buffered(predictions)

    def _generate_target_optimizer(self, predictions: dict):
        """Generate target via optimizer_v2 (alpha-proportional + turnover constraint)."""
        import pandas as pd
        from backtest.optimizer_v2 import TurnoverConstrainedOptimizer
        from backtest.constraints import PortfolioConstraints

        scores = pd.Series(predictions).sort_values(ascending=False)
        optimizer = TurnoverConstrainedOptimizer(
            top_k=self.top_k, max_turnover=self.max_turnover,
            max_single_weight=self.max_single_weight,
            weight_method=self.weight_method,
        )
        prev_weights = self.state.get("prev_weights", {})
        holding_days = {code: pos.get("holding_days", 0)
                        for code, pos in self.state.get("positions", {}).items()}
        constraints = PortfolioConstraints(min_hold_days=self.min_hold_days)

        target_weights = optimizer.optimize(
            alpha_scores=scores, prev_weights=prev_weights,
            constraints=constraints, holding_days=holding_days,
        )

        # Store target weights — in pending mode these are "intended" weights,
        # only committed to prev_weights after successful reconciliation.
        # In legacy mode, write immediately (backward compatible).
        if self.mode == "legacy":
            self.state["prev_weights"] = target_weights
        else:
            # Pending mode: store as pending, not yet committed
            self.state["pending_target_weights"] = target_weights

        # Derive sells/buys from weight changes
        current = set(self.state.get("positions", {}).keys())
        target = set(target_weights.keys())
        sells = list(current - target)
        buys = list(target - current)

        return list(target), sells, buys

    def _generate_target_buffered(self, predictions: dict):
        """Original buffered_partial logic."""

        # Rank all stocks
        sorted_preds = sorted(predictions.items(), key=lambda x: -x[1])
        top_candidates = set(k for k, _ in sorted_preds[:self.top_k])
        buffer_zone = set(k for k, _ in sorted_preds[self.top_k:self.top_k + self.buffer])
        safe_zone = top_candidates | buffer_zone

        current_positions = set(self.state.get("positions", {}).keys())

        # Sell: only if dropped below safe zone AND held long enough
        sells = []
        for code in current_positions:
            if code not in safe_zone:
                hold_days = self.state["positions"][code].get("holding_days", 0)
                if hold_days >= self.min_hold_days:
                    sells.append(code)

        # Partial trading: only sell a fraction
        n_sells = int(len(sells) * self.trade_rate + 0.5)
        n_sells = min(n_sells, int(self.max_daily_turnover * max(len(current_positions), 1)))

        # Priority: sell weakest first
        if n_sells > 0 and sells:
            sell_scores = [(code, predictions.get(code, -999)) for code in sells]
            sell_scores.sort(key=lambda x: x[1])
            actual_sells = set(code for code, _ in sell_scores[:n_sells])
        else:
            actual_sells = set()

        # Buy: fill vacated slots from top candidates
        remaining = current_positions - actual_sells
        open_slots = self.top_k - len(remaining)
        if current_positions:
            # Existing portfolio: only buy to replace sold stocks (partial rebalance)
            n_buys = min(len(actual_sells), open_slots)
        else:
            # Empty portfolio (first day or after reset): buy up to top_k
            n_buys = open_slots
        buy_candidates = [k for k, _ in sorted_preds[:self.top_k] if k not in remaining]
        actual_buys = set(buy_candidates[:max(0, n_buys)])

        target = list(remaining | actual_buys)
        return target, list(actual_sells), list(actual_buys)

    def _load_real_prices(self, date: str, extra_codes: list = None,
                          use_next_open: bool = True) -> dict:
        """Load execution prices from Qlib.

        Two modes depending on when this runs:
        - Historical backtest replay: T+1 open is known → use it for realistic fill.
        - Live/paper (same-day): T+1 open is NOT yet known → use T close as
          estimate. Orders are "pending next open" semantically.

        The caller (RiskGuard, PnL) must understand that live prices are
        estimates until next-day reconciliation.

        Args:
            date: signal date (today)
            extra_codes: additional stock codes to load
            use_next_open: if True, try next day's open first (only works
                for historical dates where T+1 data exists in Qlib)
        """
        prices = {}
        try:
            from qlib.data import D
            import pandas as pd
            try:
                D.calendar(freq="day", start_time="2020-01-01", end_time="2020-01-02")
            except Exception:
                from config.qlib_runtime import init_qlib
                init_qlib(str(DATA_DIR / "qlib_data" / "cn_data"))

            insts = list(self.state.get("positions", {}).keys())
            if extra_codes:
                insts = list(set(insts + extra_codes))
            if not insts:
                return prices

            qlib_insts = [c.lower() for c in insts]

            if use_next_open:
                # Try T+1 open: only valid for historical replay where
                # next trading day's data already exists in Qlib.
                # For live/paper on today's date, this will return NaN/empty
                # and we fall through to T close below.
                df = D.features(qlib_insts, ["Ref($open, -1)"],
                               start_time=date, end_time=date)
                if df is not None and not df.empty:
                    for idx, row in df.iterrows():
                        inst = str(idx[0]).upper()
                        price = float(row.iloc[0])
                        if np.isfinite(price) and price > 0:
                            prices[inst] = price
                    if prices:
                        logger.info(f"  Using T+1 open prices ({len(prices)} stocks)")
                        self._price_type = "next_open"
                        return prices

            # Fallback: today's close (used for live/paper when T+1 not yet available)
            # NOTE: This is an ESTIMATE. Real execution happens at next day's open.
            # RiskGuard decisions based on this price are preliminary.
            df = D.features(qlib_insts, ["$close"],
                           start_time=date, end_time=date)
            if df is not None and not df.empty:
                for idx, row in df.iterrows():
                    inst = str(idx[0]).upper()
                    price = float(row.iloc[0])
                    if np.isfinite(price) and price > 0:
                        prices[inst] = price
                if prices:
                    logger.info(f"  Using T close prices ({len(prices)} stocks, T+1 open unavailable)")
                    self._price_type = "close_estimate"

        except Exception as e:
            logger.warning(f"Failed to load prices: {e}")
            self._price_type = "unavailable"
        return prices

    def execute_orders(self, sells: list, buys: list, date: str):
        """Simulate order execution with costs.

        Execution price logic:
        - If T+1 open price available (next trading day open): use it (realistic)
        - Else fall back to today's close (optimistic but usable for live trading)
        - Apply slippage on top of execution price
        """
        cash = self.state["cash"]
        positions = self.state["positions"]
        trades = []

        # Load real prices for all relevant stocks (positions + buy candidates)
        all_codes = list(set(sells + buys + list(positions.keys())))
        prices = self._load_real_prices(date, extra_codes=all_codes)

        # Execute sells
        for code in sells:
            if code not in positions:
                continue
            pos = positions[code]
            sell_price = prices.get(code, pos["avg_price"])  # real price or fallback to avg
            amount = pos["shares"] * sell_price
            commission = max(amount * self.commission_rate, 5.0)
            stamp_tax = amount * self.stamp_tax_rate
            slippage = amount * self.slippage_rate
            net = amount - commission - stamp_tax - slippage

            cash += net
            del positions[code]
            trades.append({
                "date": date, "code": code, "side": "sell",
                "shares": pos["shares"], "price": round(sell_price, 4),
                "cost": round(commission + stamp_tax + slippage, 2),
                "net": round(net, 2),
            })
            self.state["trade_count"] += 1

        # Execute buys — use target weights if optimizer mode, else equal weight
        if buys:
            target_weights = self.state.get("prev_weights", {})
            total_portfolio_value = cash
            for code, pos in positions.items():
                total_portfolio_value += pos["shares"] * prices.get(code, pos["avg_price"])

            for code in buys:
                # Determine target value: use optimizer weight or equal weight
                if self.execution_mode == "optimizer_v2" and code in target_weights:
                    per_stock_value = total_portfolio_value * target_weights[code]
                else:
                    n_target = len(positions) + len(buys)
                    per_stock_value = cash * 0.95 / max(n_target, 1)

                if per_stock_value < 1000:
                    continue
                buy_price = prices.get(code, 0)
                if buy_price <= 0:
                    logger.warning(f"  No price for {code}, skip buy")
                    continue
                shares = int(per_stock_value / buy_price / 100) * 100  # round to 100 shares
                if shares <= 0:
                    continue

                amount = shares * buy_price
                commission = max(amount * self.commission_rate, 5.0)
                slippage = amount * self.slippage_rate
                total_cost = amount + commission + slippage

                if total_cost > cash:
                    continue

                cash -= total_cost
                positions[code] = {
                    "shares": shares,
                    "avg_price": buy_price,
                    "entry_date": date,
                    "holding_days": 0,
                }
                trades.append({
                    "date": date, "code": code, "side": "buy",
                    "shares": shares, "price": buy_price,
                    "cost": round(commission + slippage, 2),
                    "net": round(-total_cost, 2),
                })
                self.state["trade_count"] += 1

        self.state["cash"] = round(cash, 2)
        return trades

    def update_holding_days(self):
        """Increment holding days for all positions."""
        for code in self.state["positions"]:
            self.state["positions"][code]["holding_days"] = \
                self.state["positions"][code].get("holding_days", 0) + 1

    def compute_daily_pnl(self, date: str) -> dict:
        """Compute daily PnL using real prices where available."""
        # Dedup guard: skip if already recorded this date
        history = self.state.get("daily_pnl_history", [])
        if history and history[-1].get("date") == date:
            logger.warning(f"  Date {date} already recorded, skipping duplicate")
            return history[-1]

        positions = self.state["positions"]
        n_positions = len(positions)

        # Use real prices for valuation
        prices = self._load_real_prices(date)
        total_value = self.state["cash"]
        for code, p in positions.items():
            price = prices.get(code, p["avg_price"])
            total_value += p["shares"] * price

        pnl_record = {
            "date": date,
            "cash": round(self.state["cash"], 2),
            "n_positions": n_positions,
            "total_value": round(total_value, 2),
            "trade_count": self.state["trade_count"],
        }

        if self.state["daily_pnl_history"]:
            prev = self.state["daily_pnl_history"][-1]
            daily_return = (total_value - prev["total_value"]) / prev["total_value"]
            pnl_record["daily_return"] = round(daily_return, 6)
        else:
            pnl_record["daily_return"] = 0.0

        self.state["daily_pnl_history"].append(pnl_record)
        self.state["total_value"] = round(total_value, 2)

        return pnl_record

    # ------------------------------------------------------------------
    # Pending-order model: two-step flow
    # ------------------------------------------------------------------

    def _pending_orders_path(self, date: str) -> Path:
        return self._state_dir / f"pending_orders_{date}.json"

    def _filled_orders_path(self, date: str) -> Path:
        return self._state_dir / f"filled_orders_{date}.json"

    def _load_and_filter_predictions(self, date: str):
        """Load predictions, apply universe filter, return dict or None."""
        predictions = self.load_predictions()
        if predictions:
            try:
                from models.universe_filter import UniverseFilter
                uf = UniverseFilter()
                n_before = len(predictions)
                predictions = uf.filter_predictions(predictions)
                n_after = len(predictions)
                if n_before != n_after:
                    logger.info(f"  Universe filter: {n_before} -> {n_after} predictions")
            except Exception as e:
                logger.warning(f"  Universe filter failed (continuing unfiltered): {e}")
        return predictions or None

    def _apply_risk_guard(self, predictions: dict, date: str):
        """Run RiskGuard checks.  Returns (filtered_predictions, extra_sells, risk_info)."""
        extra_sells = []
        risk = None
        risk_info = {"force_sell": [], "alert_level": "normal"}
        try:
            from backtest.risk_guard import RiskGuard
            # Pass state_dir to RiskGuard for champion/shadow isolation
            _sd = str(self._state_dir.name) if self._state_dir != PAPER_DIR else None
            guard = RiskGuard(state_dir=_sd)
            prices = self._load_real_prices(
                date, extra_codes=list(self.state.get("positions", {}).keys()))
            risk = guard.check(
                positions=self.state.get("positions", {}),
                prices=prices,
                date=date,
            )
            guard.update_portfolio_value(self.state.get("total_value", 1e6), date)

            if risk.force_sell:
                logger.warning(f"  RiskGuard force_sell: {risk.force_sell}")
                for code in risk.force_sell:
                    reason = risk.risk_reasons.get(code, "")
                    logger.warning(f"    {code}: {reason}")
                risk_info["force_sell"] = list(risk.force_sell)
                risk_info["alert_level"] = "force_sell"

            if risk.cannot_buy:
                predictions = {k: v for k, v in predictions.items()
                               if k not in risk.cannot_buy and k.lower() not in risk.cannot_buy}

            extra_sells = [code for code in risk.force_sell
                           if code in self.state.get("positions", {})]
        except Exception as e:
            logger.warning(f"  RiskGuard failed (continuing without): {e}")

        return predictions, extra_sells, risk_info

    def generate_orders(self, date: str = None) -> dict:
        """Step 1 of pending-order model.

        Generate orders on day T using T close prices for preliminary valuation.
        Orders are saved to pending_orders_{date}.json but positions/PnL are NOT
        modified.

        Returns:
            Order summary dict (also written to disk).
        """
        date = date or datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Paper OMS generate_orders: {date}")

        predictions = self._load_and_filter_predictions(date)
        if not predictions:
            logger.warning("  No predictions — no orders generated")
            return {"signal_date": date, "status": "no_predictions", "orders": []}

        predictions, extra_sells, risk_info = self._apply_risk_guard(predictions, date)

        # Generate target
        target, sells, buys = self.generate_target(predictions)

        if extra_sells:
            sells = list(set(sells + extra_sells))
            target = [t for t in target if t not in set(extra_sells)]

        logger.info(f"  Target: {len(target)} stocks, sell={len(sells)}, buy={len(buys)}")

        # Load T close prices for preliminary valuation
        all_codes = list(set(sells + buys + list(self.state.get("positions", {}).keys())))
        prices = self._load_real_prices(date, extra_codes=all_codes, use_next_open=False)
        price_type = getattr(self, "_price_type", "close_estimate")

        # Compute target weights for buy sizing
        target_weights = {}
        if self.execution_mode == "optimizer_v2":
            target_weights = dict(self.state.get("prev_weights", {}))
        else:
            # Equal weight for buffered_partial
            if target:
                w = round(1.0 / len(target), 6)
                for code in target:
                    target_weights[code] = w

        # Build order list
        orders = []
        for code in sells:
            orders.append({
                "code": code,
                "action": "sell",
                "target_weight": 0.0,
                "preliminary_price": round(prices.get(code, 0.0), 4),
                "reason": "force_sell" if code in risk_info.get("force_sell", []) else "dropped_from_target",
            })
        for code in buys:
            orders.append({
                "code": code,
                "action": "buy",
                "target_weight": round(target_weights.get(code, 0.0), 6),
                "preliminary_price": round(prices.get(code, 0.0), 4),
            })

        pending = {
            "signal_date": date,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "target_portfolio": target,
            "target_weights": {k: round(v, 6) for k, v in target_weights.items()},
            "orders": orders,
            "risk_guard": risk_info,
            "price_type": price_type,
            "status": "pending",
        }

        # Save to disk
        path = self._pending_orders_path(date)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(pending, indent=2, ensure_ascii=False))
        os.replace(tmp, path)
        logger.info(f"  Pending orders saved: {path} ({len(orders)} orders)")

        return pending

    def reconcile(self, date: str, fill_date: str = None) -> dict:
        """Step 2 of pending-order model.

        Load pending orders from *date*, fill them at T+1 (fill_date) open
        prices from Qlib.  Updates positions, PnL, and state.

        Args:
            date: signal date (when orders were generated).
            fill_date: execution date (T+1).  If None, determined automatically.

        Returns:
            Fill summary dict.  status="pending" if T+1 prices not yet available.
        """
        logger.info(f"Paper OMS reconcile: signal_date={date}, fill_date={fill_date}")

        # Load pending orders
        pending_path = self._pending_orders_path(date)
        if not pending_path.exists():
            logger.warning(f"  No pending orders for {date}")
            return {"signal_date": date, "status": "no_pending_orders"}

        pending = json.loads(pending_path.read_text())
        if pending.get("status") == "filled":
            logger.info(f"  Orders for {date} already filled")
            return pending

        orders = pending.get("orders", [])
        if not orders:
            # No orders to fill — just update holding days / PnL
            logger.info("  No orders to fill, updating holding days & PnL")
            self.update_holding_days()
            pnl = self.compute_daily_pnl(fill_date or date)
            self._save_state()
            return {"signal_date": date, "status": "filled", "fills": [],
                    "daily_pnl": pnl.get("daily_return", 0.0)}

        sells = [o["code"] for o in orders if o["action"] == "sell"]
        buys = [o["code"] for o in orders if o["action"] == "buy"]
        all_codes = list(set(sells + buys + list(self.state.get("positions", {}).keys())))

        # Try to load T+1 open prices
        # We query using the signal date with use_next_open=True — Qlib's
        # Ref($open, -1) on signal_date gives next-day open.
        prices = self._load_real_prices(date, extra_codes=all_codes, use_next_open=True)
        fill_price_type = getattr(self, "_price_type", "unavailable")

        if fill_price_type != "next_open" or not prices:
            logger.info("  T+1 open prices not available yet — reconciliation deferred")
            return {"signal_date": date, "status": "pending",
                    "reason": "next_open_not_available"}

        # --- Execute fills ---
        cash = self.state["cash"]
        positions = self.state["positions"]
        fills = []

        # Sells
        for code in sells:
            if code not in positions:
                continue
            pos = positions[code]
            sell_price = prices.get(code, pos["avg_price"])
            amount = pos["shares"] * sell_price
            commission = max(amount * self.commission_rate, 5.0)
            stamp_tax = amount * self.stamp_tax_rate
            slippage = amount * self.slippage_rate
            net = amount - commission - stamp_tax - slippage

            cash += net
            shares_sold = pos["shares"]
            del positions[code]
            fills.append({
                "code": code, "action": "sell",
                "fill_price": round(sell_price, 4),
                "shares": shares_sold,
                "cost": round(commission + stamp_tax + slippage, 2),
                "net": round(net, 2),
            })
            self.state["trade_count"] += 1

        # Buys
        if buys:
            target_weights = pending.get("target_weights", {})
            total_portfolio_value = cash
            for code, pos in positions.items():
                total_portfolio_value += pos["shares"] * prices.get(code, pos["avg_price"])

            for code in buys:
                if self.execution_mode == "optimizer_v2" and code in target_weights:
                    per_stock_value = total_portfolio_value * target_weights[code]
                else:
                    n_target = len(positions) + len(buys)
                    per_stock_value = cash * 0.95 / max(n_target, 1)

                if per_stock_value < 1000:
                    continue
                buy_price = prices.get(code, 0)
                if buy_price <= 0:
                    logger.warning(f"  No fill price for {code}, skip buy")
                    continue
                shares = int(per_stock_value / buy_price / 100) * 100
                if shares <= 0:
                    continue

                amount = shares * buy_price
                commission = max(amount * self.commission_rate, 5.0)
                slippage_cost = amount * self.slippage_rate
                total_cost = amount + commission + slippage_cost

                if total_cost > cash:
                    continue

                cash -= total_cost
                positions[code] = {
                    "shares": shares,
                    "avg_price": buy_price,
                    "entry_date": fill_date or date,
                    "holding_days": 0,
                }
                fills.append({
                    "code": code, "action": "buy",
                    "fill_price": round(buy_price, 4),
                    "shares": shares,
                    "cost": round(commission + slippage_cost, 2),
                    "net": round(-total_cost, 2),
                })
                self.state["trade_count"] += 1

        self.state["cash"] = round(cash, 2)

        # Commit pending target weights as actual prev_weights after successful fills
        if "pending_target_weights" in self.state:
            self.state["prev_weights"] = self.state.pop("pending_target_weights")

        # Update holding days & PnL
        self.update_holding_days()
        use_date = fill_date or date
        pnl = self.compute_daily_pnl(use_date)
        logger.info(f"  Reconciled: {len(fills)} fills, PnL={pnl['daily_return']:+.4f}")

        # Write trade log
        if fills:
            trades_path = self._state_dir / "trades.jsonl"
            with open(str(trades_path), "a") as f:
                for fill in fills:
                    record = dict(fill, date=use_date, side=fill["action"],
                                  price=fill["fill_price"])
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._save_state()

        # Build filled order file
        filled = {
            "signal_date": date,
            "fill_date": use_date,
            "reconciled_at": datetime.now().isoformat(timespec="seconds"),
            "fills": fills,
            "price_type": "next_open",
            "daily_pnl": pnl.get("daily_return", 0.0),
            "status": "filled",
        }

        # Archive filled orders
        filled_path = self._filled_orders_path(use_date)
        tmp = filled_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(filled, indent=2, ensure_ascii=False))
        os.replace(tmp, filled_path)

        # Mark pending as filled
        pending["status"] = "filled"
        pending_path.write_text(json.dumps(pending, indent=2, ensure_ascii=False))

        return filled

    # ------------------------------------------------------------------
    # Legacy single-step entry point (backward compatible)
    # ------------------------------------------------------------------

    def run_daily(self, date: str = None):
        """Run one day of paper trading.

        In "legacy" mode (default): runs the original single-step flow where
        orders are generated and filled in the same call.

        In "pending" mode: calls generate_orders() then reconcile() in
        sequence.  reconcile() may return status="pending" if T+1 prices
        are not yet available.
        """
        date = date or datetime.now().strftime("%Y-%m-%d")

        if self.mode == "pending":
            gen = self.generate_orders(date)
            if gen.get("status") in ("no_predictions",):
                # Still update holding days and PnL
                self.update_holding_days()
                pnl = self.compute_daily_pnl(date)
                self._save_state()
                return pnl
            rec = self.reconcile(date)
            if rec.get("status") == "pending":
                logger.info("  Reconciliation deferred (T+1 data not available)")
                return rec
            # Return PnL-style dict for backward compat
            pnl_history = self.state.get("daily_pnl_history", [])
            return pnl_history[-1] if pnl_history else rec

        # --- Legacy mode (original behaviour, unchanged) ---
        logger.info(f"Paper OMS: running for {date}")

        predictions = self._load_and_filter_predictions(date)
        if not predictions:
            logger.warning("  No predictions, holding current positions")
            self.update_holding_days()
            pnl = self.compute_daily_pnl(date)
            self._save_state()
            return pnl

        predictions, extra_sells, _risk_info = self._apply_risk_guard(predictions, date)

        # Generate target
        target, sells, buys = self.generate_target(predictions)

        # Add risk-forced sells
        if extra_sells:
            sells = list(set(sells + extra_sells))
            # Remove force-sold stocks from target
            target = [t for t in target if t not in set(extra_sells)]

        logger.info(f"  Target: {len(target)} stocks, sell={len(sells)}, buy={len(buys)}")

        # Execute
        trades = self.execute_orders(sells, buys, date)
        logger.info(f"  Trades: {len(trades)}")

        # Update
        self.update_holding_days()
        pnl = self.compute_daily_pnl(date)
        logger.info(f"  PnL: value={pnl['total_value']}, return={pnl['daily_return']:+.4f}")

        # Write trade log
        if trades:
            trades_path = self._state_dir / "trades.jsonl"
            with open(str(trades_path), "a") as f:
                for t in trades:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")

        self._save_state()
        return pnl

    def status(self) -> dict:
        """Get current paper trading status."""
        n_days = len(self.state["daily_pnl_history"])
        total_return = (self.state["total_value"] / self.initial_capital - 1) if n_days > 0 else 0

        return {
            "start_date": self.state["start_date"],
            "n_days": n_days,
            "cash": self.state["cash"],
            "n_positions": len(self.state["positions"]),
            "total_value": self.state["total_value"],
            "total_return": round(total_return, 4),
            "trade_count": self.state["trade_count"],
            "positions": list(self.state["positions"].keys())[:10],
        }
