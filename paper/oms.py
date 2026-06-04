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
                 cost_model=None,
                 vol_adv_snapshot=None,
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
        # cx code review round 3 P2: sqrt_adv exists in backtest/cost_model.py
        # but paper OMS computed slippage as `amount * slippage_rate` inline.
        # Accept an optional CostModel; when supplied with impact_model
        # "sqrt_adv" plus per-fill (daily_volatility, adv), slippage scales
        # with sqrt(trade_value / ADV). Default behaviour (cost_model=None)
        # preserves the bare-rate path.
        self.cost_model = cost_model
        # cx round-3 P2 #84 follow-up: per-stock vol + ADV snapshot fed
        # into fill sites. Without this dict, _compute_slippage(amount)
        # at the fill sites passes vol=None,adv=None → CostModel._slippage
        # falls back to bare slippage_rate → sqrt_adv is dead code in
        # production paper even with cost_model=CostModel(impact_model=
        # "sqrt_adv"). The snapshot is a dict {code: {"vol": float,
        # "adv": float}} typically built by paper.cost_inputs at the
        # start of a daily run from qlib historical data. None preserves
        # pre-fix behaviour.
        self.vol_adv_snapshot = vol_adv_snapshot
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

    def _lookup_vol_adv(self, code) -> tuple:
        """Look up (daily_volatility, adv) for `code` in the snapshot.

        Returns (None, None) if the snapshot is absent or the code is
        not in it. The fill sites pass these into _compute_slippage,
        which itself falls back to bare slippage_rate when either is
        None — so a missing snapshot or a code that wasn't computed
        today both gracefully degrade to pre-fix behaviour rather
        than raising.

        Accepts the snapshot in either of two shapes:
          {code: {"vol": ..., "adv": ...}}
          {code: (vol, adv)}
        """
        if not self.vol_adv_snapshot:
            return None, None
        entry = self.vol_adv_snapshot.get(code)
        if entry is None:
            return None, None
        if isinstance(entry, dict):
            return entry.get("vol"), entry.get("adv")
        if isinstance(entry, (tuple, list)) and len(entry) >= 2:
            return entry[0], entry[1]
        return None, None

    def _compute_slippage(self, amount, daily_volatility=None, adv=None):
        """Compute slippage cost for a fill.

        When self.cost_model is provided (with impact_model="sqrt_adv")
        AND we have daily_volatility + ADV for the stock, delegate to
        CostModel._slippage which uses
            slip_rate = sigma * sqrt(trade_value / ADV) * coefficient

        Otherwise fall back to the bare-rate path
        (amount * self.slippage_rate) — identical to pre-fix behaviour.

        Argument convention: daily_volatility is the stock's daily return
        std (e.g. 0.02 for a 2% sigma); adv is the stock's average daily
        traded value (yuan). Both default to None so existing callers
        that don't yet pipe vol/ADV continue to work.
        """
        if self.cost_model is not None:
            try:
                return self.cost_model._slippage(
                    amount, daily_volatility=daily_volatility, adv=adv,
                )
            except Exception:
                # Defensive: any cost-model failure falls back to bare rate
                # so a misconfigured CostModel can never break paper fills.
                pass
        return amount * self.slippage_rate

    def load_predictions(self) -> dict:
        """Load latest model predictions through the validated loader.

        2026-06-04 cx round 3 P0-3: previously this method called
        ``json.loads(cache_path.read_text())`` directly, bypassing
        the freshness + distribution gates in
        ``models.lgb_cache.load_prediction_cache``. That meant a
        polluted cache (smoke RED, manual debug, anything) would
        drive paper trades. The validated loader raises on:
            - missing file
            - undated / stale latest_date
            - RED distribution (all-negative / all-zero / etc.)
        and paper's loop logs the failure and skips this cycle
        rather than trading on garbage."""
        from models.lgb_cache import load_prediction_cache
        from models.prediction_health import PredictionDistributionRed
        try:
            finite, _payload = load_prediction_cache()
        except FileNotFoundError:
            logger.warning("No prediction cache found")
            return {}
        except PredictionDistributionRed as exc:
            logger.error(
                "Paper OMS refusing to trade on RED-distribution "
                "prediction cache: %s", exc,
            )
            return {}
        except RuntimeError as exc:
            logger.error(
                "Paper OMS refusing to trade on invalid prediction "
                "cache: %s", exc,
            )
            return {}
        return finite

    def get_current_positions(self) -> dict:
        """Get current positions as {code: Position}."""
        positions = {}
        for code, info in self.state.get("positions", {}).items():
            pos = Position(code, info["shares"], info["avg_price"], info["entry_date"])
            pos.holding_days = info.get("holding_days", 0)
            positions[code] = pos
        return positions

    def generate_target(self, predictions: dict, risk_info: dict | None = None) -> list:
        """Generate target portfolio.

        For buffered_partial: returns (target_list, sells, buys)
        For optimizer_v2: returns (target_list, sells, buys) derived from weight changes

        risk_info (optional) carries RiskGuard outputs that influence sizing
        beyond simple include/exclude — currently `reduce_weight` (soft crash
        tier multipliers). Threaded through to the optimizer.
        """
        if not predictions:
            return list(self.state.get("positions", {}).keys()), [], []

        if self.execution_mode == "optimizer_v2":
            return self._generate_target_optimizer(predictions, risk_info=risk_info)

        return self._generate_target_buffered(predictions)

    def _generate_target_optimizer(self, predictions: dict, risk_info: dict | None = None):
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
        reduce_weight = dict((risk_info or {}).get("reduce_weight", {}))
        constraints = PortfolioConstraints(
            min_hold_days=self.min_hold_days,
            reduce_weight=reduce_weight,
        )

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
            _vol, _adv = self._lookup_vol_adv(code)
            slippage = self._compute_slippage(amount, daily_volatility=_vol, adv=_adv)
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
                _vol, _adv = self._lookup_vol_adv(code)
                slippage = self._compute_slippage(amount, daily_volatility=_vol, adv=_adv)
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
        """Load predictions, apply unified candidate sanitizer, return dict or None.

        Previously called models.universe_filter.UniverseFilter which only
        excluded ST + BJ — narrower than the training-time tradable_mask
        (which also handles IPO<60d, 一字板, suspended, low liquidity). The
        unified CandidateSanitizer brings inference-side filtering in line.
        Quote data isn't reliably available at OMS load time (before market
        open in some paths), so require_quote=False.
        """
        predictions = self.load_predictions()
        if not predictions:
            return None
        # 2026-06-04 cx round 7 P1-6: previously this caught any
        # sanitizer error and returned the UNFILTERED predictions —
        # paper would then trade ST / new / suspended / 一字板 /
        # high-crash-risk stocks that the production recommendation
        # path explicitly excludes. The opposite of what "safety
        # filter" means. Fail-closed instead: sanitizer error → no
        # trade this cycle.
        try:
            from factors.candidate_sanitizer import CandidateSanitizer
            sanitizer = CandidateSanitizer(today=date, require_quote=False)
        except Exception as e:
            logger.error(
                "Paper OMS CandidateSanitizer import/init failed: %s — "
                "refusing to trade unfiltered.", e,
            )
            return None
        filtered = {}
        try:
            for code, score in predictions.items():
                ok, _reason = sanitizer.check(code, None)
                if ok:
                    filtered[code] = score
            sanitizer.log_summary(label=f"paper_oms[{date}]")
        except Exception as e:
            logger.error(
                "Paper OMS sanitizer.check raised: %s — refusing to "
                "trade unfiltered predictions.", e,
            )
            return None
        return filtered or None

    def _load_crash_predictions(self, date: str) -> dict | None:
        """Try to load crash predictions from crash_predictions_latest.json.

        Returns:
            Dict {code: crash_prob} if available and date matches, else None.
        """
        crash_path = DATA_DIR / "crash_predictions_latest.json"
        if not crash_path.exists():
            return None
        try:
            payload = json.loads(crash_path.read_text())
            pred_date = payload.get("date", "")
            # Accept if prediction date is today or yesterday (stale by 1 day is OK)
            if pred_date and pred_date < date[:10]:
                # More than a day stale — check how old
                from datetime import datetime as _dt
                age = (_dt.strptime(date[:10], "%Y-%m-%d") -
                       _dt.strptime(pred_date, "%Y-%m-%d")).days
                if age > 2:
                    logger.info(f"  Crash predictions stale ({pred_date}, {age}d old), skipping")
                    return None
            crash_probs = payload.get("predictions", {})
            if crash_probs:
                logger.info(f"  Loaded crash predictions: {len(crash_probs)} stocks (date={pred_date})")
            return crash_probs if crash_probs else None
        except Exception as e:
            logger.warning(f"  Failed to load crash predictions: {e}")
            return None

    def _load_chain_factors(self, date: str) -> dict | None:
        """Try to load global supply chain factors from parquet.

        Returns:
            Dict {code: global_chain_alpha} if available, else None.
        """
        chain_path = DATA_DIR / "global_chain_factors.parquet"
        if not chain_path.exists():
            return None
        try:
            df = pd.read_parquet(chain_path)
            if df.empty:
                return None

            dt = pd.Timestamp(date)
            dates = df.index.get_level_values("datetime")

            if dt in dates:
                chain_today = df.xs(dt, level="datetime")
            else:
                # Fall back to the latest available date (max 2 days stale)
                latest = dates.max()
                age = (dt - latest).days
                if age > 2:
                    logger.info(f"  Chain factors stale ({latest.date()}, {age}d old), skipping")
                    return None
                chain_today = df.xs(latest, level="datetime")

            if "global_chain_alpha" not in chain_today.columns:
                return None

            alpha = chain_today["global_chain_alpha"]
            alpha.index = alpha.index.str.upper()
            result = {code: float(val) for code, val in alpha.items()
                      if np.isfinite(val)}
            if result:
                logger.info(f"  Loaded chain factors: {len(result)} stocks")
            return result if result else None
        except Exception as e:
            logger.warning(f"  Failed to load chain factors: {e}")
            return None

    def _apply_risk_guard(self, predictions: dict, date: str):
        """Run RiskGuard checks.  Returns (filtered_predictions, extra_sells, risk_info).

        risk_info now also propagates reduce_weight (the soft crash-tier
        penalty multipliers, e.g. crash_prob 0.50 → 0.5x, 0.70 → 0.25x). The
        optimizer reads constraints.reduce_weight to scale target weights
        before the per-stock cap, so this layer is no longer dead code.
        """
        extra_sells = []
        risk = None
        risk_info = {"force_sell": [], "alert_level": "normal", "reduce_weight": {}}
        try:
            from backtest.risk_guard import RiskGuard
            # Pass state_dir to RiskGuard for champion/shadow isolation
            _sd = str(self._state_dir.name) if self._state_dir != PAPER_DIR else None
            guard = RiskGuard(state_dir=_sd)
            prices = self._load_real_prices(
                date, extra_codes=list(self.state.get("positions", {}).keys()))

            # Try to load crash predictions (Phase 4O)
            crash_probs = self._load_crash_predictions(date)

            # Try to load global supply chain factors
            chain_factors = self._load_chain_factors(date)

            risk = guard.check(
                positions=self.state.get("positions", {}),
                prices=prices,
                date=date,
                crash_probs=crash_probs,
                chain_factors=chain_factors,
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

            # Surface reduce_weight for the optimizer downstream
            if risk.reduce_weight:
                risk_info["reduce_weight"] = dict(risk.reduce_weight)
                logger.info(
                    "  RiskGuard reduce_weight: %d stocks (soft crash tier)",
                    len(risk.reduce_weight),
                )

            extra_sells = [code for code in risk.force_sell
                           if code in self.state.get("positions", {})]
        except Exception as e:
            logger.warning(f"  RiskGuard FAILED: {e}")
            # Fail-closed: if RiskGuard crashes, block all new buys.
            # Existing positions can still be reconciled/sold.
            risk_info["alert_level"] = "riskguard_error"
            risk_info["error"] = str(e)[:200]
            # Clear all predictions → no new buys
            predictions = {}
            logger.warning("  Fail-closed: blocked all new buys due to RiskGuard error")

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

        # Generate target — pass risk_info so optimizer applies reduce_weight
        target, sells, buys = self.generate_target(predictions, risk_info=risk_info)

        if extra_sells:
            sells = list(set(sells + extra_sells))
            target = [t for t in target if t not in set(extra_sells)]

        logger.info(f"  Target: {len(target)} stocks, sell={len(sells)}, buy={len(buys)}")

        # Load T close prices for preliminary valuation
        all_codes = list(set(sells + buys + list(self.state.get("positions", {}).keys())))
        prices = self._load_real_prices(date, extra_codes=all_codes, use_next_open=False)
        price_type = getattr(self, "_price_type", "close_estimate")

        # Compute target weights for buy sizing.
        # CRITICAL: in optimizer_v2 + pending mode, _generate_target_optimizer
        # stores the just-computed (RiskGuard + reduce_weight aware) target
        # into state["pending_target_weights"]; prev_weights still holds the
        # PREVIOUS reconcile's weights. Reading prev_weights here meant new
        # buys had no weight at all (they weren't in last cycle's portfolio),
        # falling back to cash * 0.95 / N and silently bypassing
        # reduce_weight. Read pending_target_weights first; only legacy mode
        # commits straight to prev_weights so it's still correct there.
        target_weights = {}
        if self.execution_mode == "optimizer_v2":
            target_weights = dict(self.state.get("pending_target_weights")
                                  or self.state.get("prev_weights", {}))
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
            _vol, _adv = self._lookup_vol_adv(code)
            slippage = self._compute_slippage(amount, daily_volatility=_vol, adv=_adv)
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
        unfilled_buys: list[dict] = []
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
                    unfilled_buys.append({"code": code, "reason": "size_below_min"})
                    continue
                buy_price = prices.get(code, 0)
                if buy_price <= 0:
                    logger.warning(f"  No fill price for {code}, skip buy")
                    unfilled_buys.append({"code": code, "reason": "no_price_likely_halted"})
                    continue
                shares = int(per_stock_value / buy_price / 100) * 100
                if shares <= 0:
                    unfilled_buys.append({"code": code, "reason": "zero_shares_after_rounding"})
                    continue

                amount = shares * buy_price
                commission = max(amount * self.commission_rate, 5.0)
                _vol, _adv = self._lookup_vol_adv(code)
                slippage_cost = self._compute_slippage(amount, daily_volatility=_vol, adv=_adv)
                total_cost = amount + commission + slippage_cost

                if total_cost > cash:
                    unfilled_buys.append({"code": code, "reason": "insufficient_cash"})
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

        if unfilled_buys:
            logger.warning(
                "  %d intended buy(s) could not fill: %s",
                len(unfilled_buys),
                {u["code"]: u["reason"] for u in unfilled_buys},
            )

        # Build filled order file. status="filled" historically applied even
        # when many intended buys couldn't transact (no price = likely halted,
        # cash exhausted, etc.). Now we surface unfilled_buys so consumers can
        # tell a partial reconcile from a clean one. status="filled" still
        # means "no more work to do for this signal_date", not "all buys went
        # through".
        filled = {
            "signal_date": date,
            "fill_date": use_date,
            "reconciled_at": datetime.now().isoformat(timespec="seconds"),
            "fills": fills,
            "unfilled_buys": unfilled_buys,
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

    def reconcile_pending_history(self, today: str = None) -> dict:
        """Reconcile any prior pending orders whose T+1 data is now available.

        Walks pending_orders_*.json in chronological order. For each with
        status != "filled" and signal_date < today, calls reconcile(). Stops
        at the first deferral — by construction, if signal_date N defers
        because T+1=N+1 has no Qlib data, no later signal_date will either.

        Without this, run_daily(today) only reconciles today's orders
        (which always defer for live runs since T+1 is the future), so the
        OMS state never advances past the last manual reconcile.
        """
        today = today or datetime.now().strftime("%Y-%m-%d")
        pending_files = sorted(self._state_dir.glob("pending_orders_*.json"))

        replayed, deferred, skipped = 0, 0, 0
        for path in pending_files:
            signal_date = path.stem.replace("pending_orders_", "")
            if signal_date >= today:
                continue
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            if data.get("status") == "filled":
                continue
            result = self.reconcile(signal_date)
            status = result.get("status")
            if status == "pending":
                deferred += 1
                break
            elif status == "filled":
                replayed += 1
            else:
                skipped += 1

        if replayed or deferred:
            logger.info(
                "  Reconcile history: %d replayed, %d deferred, %d skipped",
                replayed, deferred, skipped,
            )
        return {"replayed": replayed, "deferred": deferred, "skipped": skipped}

    # ------------------------------------------------------------------
    # Legacy single-step entry point (backward compatible)
    # ------------------------------------------------------------------

    def run_daily(self, date: str = None):
        """Run one day of paper trading.

        In "legacy" mode (default): runs the original single-step flow where
        orders are generated and filled in the same call.

        In "pending" mode: replays any historical pending orders whose T+1
        data is now available, then calls generate_orders() and reconcile()
        for *date*. reconcile(date) typically defers in live runs since T+1
        prices are not yet known.
        """
        date = date or datetime.now().strftime("%Y-%m-%d")

        if self.mode == "pending":
            # Walk forward through any unfilled history first
            self.reconcile_pending_history(today=date)

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

        # Generate target — pass risk_info so optimizer applies reduce_weight
        target, sells, buys = self.generate_target(predictions, risk_info=_risk_info)

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
