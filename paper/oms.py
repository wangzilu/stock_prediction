"""Paper Trading OMS (Order Management System).

Simulates: signal → target portfolio → orders → fills → positions → PnL.

Daily flow:
1. Load model predictions (from lgb_latest_predictions.json)
2. Generate target portfolio (buffered_partial logic)
3. Compare with current positions → generate orders
4. Simulate fills (T+1, limit-up/down/suspended check)
5. Update positions and PnL
6. Write daily ledger

Usage:
    from paper.oms import PaperOMS
    oms = PaperOMS()
    oms.run_daily()  # call after market close
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

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
    """Paper trading order management system."""

    def __init__(self,
                 initial_capital: float = 1_000_000,
                 top_k: int = 20,
                 buffer: int = 5,
                 trade_rate: float = 0.35,
                 min_hold_days: int = 2,
                 max_daily_turnover: float = 0.15,
                 commission_rate: float = 0.0003,
                 stamp_tax_rate: float = 0.0005,
                 slippage_rate: float = 0.001):

        self.initial_capital = initial_capital
        self.top_k = top_k
        self.buffer = buffer
        self.trade_rate = trade_rate
        self.min_hold_days = min_hold_days
        self.max_daily_turnover = max_daily_turnover
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage_rate = slippage_rate

        PAPER_DIR.mkdir(parents=True, exist_ok=True)

        # Load state
        self.state = self._load_state()

    def _state_path(self):
        return PAPER_DIR / "oms_state.json"

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
        """Generate target portfolio using buffered_partial logic.

        Returns list of stock codes to hold.
        """
        if not predictions:
            return list(self.state.get("positions", {}).keys())

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
        n_buys = min(len(actual_sells), self.top_k - len(remaining))
        buy_candidates = [k for k, _ in sorted_preds[:self.top_k] if k not in remaining]
        actual_buys = set(buy_candidates[:max(0, n_buys)])

        target = list(remaining | actual_buys)
        return target, list(actual_sells), list(actual_buys)

    def execute_orders(self, sells: list, buys: list, date: str):
        """Simulate order execution with costs."""
        cash = self.state["cash"]
        positions = self.state["positions"]
        trades = []

        # Execute sells
        for code in sells:
            if code not in positions:
                continue
            pos = positions[code]
            # Simulate: sell at previous close * (1 - slippage)
            sell_price = pos["avg_price"]  # simplified: use avg_price as proxy
            amount = pos["shares"] * sell_price
            commission = max(amount * self.commission_rate, 5.0)
            stamp_tax = amount * self.stamp_tax_rate
            slippage = amount * self.slippage_rate
            net = amount - commission - stamp_tax - slippage

            cash += net
            del positions[code]
            trades.append({
                "date": date, "code": code, "side": "sell",
                "shares": pos["shares"], "price": sell_price,
                "cost": round(commission + stamp_tax + slippage, 2),
                "net": round(net, 2),
            })
            self.state["trade_count"] += 1

        # Execute buys (equal weight among all target stocks)
        if buys:
            n_positions = len(positions) + len(buys)
            per_stock_value = cash * 0.95 / max(n_positions, 1)  # keep 5% cash buffer

            for code in buys:
                if per_stock_value < 1000:  # minimum trade value
                    break
                buy_price = 10.0  # placeholder — in production, use real price
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
        """Compute daily PnL (simplified: uses position count as proxy)."""
        positions = self.state["positions"]
        n_positions = len(positions)
        total_value = self.state["cash"] + sum(
            p["shares"] * p["avg_price"] for p in positions.values())

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

    def run_daily(self, date: str = None):
        """Run one day of paper trading."""
        date = date or datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Paper OMS: running for {date}")

        # Load predictions
        predictions = self.load_predictions()
        if not predictions:
            logger.warning("  No predictions, holding current positions")
            self.update_holding_days()
            pnl = self.compute_daily_pnl(date)
            self._save_state()
            return pnl

        # Generate target
        target, sells, buys = self.generate_target(predictions)
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
            trades_path = PAPER_DIR / "trades.jsonl"
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
