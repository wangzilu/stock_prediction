"""Run daily paper trading — call after market close.

Loads latest predictions, generates orders, simulates fills, updates PnL.

Usage:
    python scripts/run_paper_trading.py                # run for today
    python scripts/run_paper_trading.py --status       # show current status
    python scripts/run_paper_trading.py --report       # show PnL history
    python scripts/run_paper_trading.py --reset        # reset to initial state
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from paper.oms import PaperOMS

DATA_DIR = PROJECT_ROOT / "data" / "storage"
PAPER_DIR = DATA_DIR / "paper"


def _build_cost_inputs(date: str | None,
                        impact_model: str,
                        impact_coefficient: float,
                        lookback_days: int):
    """Return (cost_model, vol_adv_snapshot) for PaperOMS.

    Per cx round 3 P2 #84 follow-up (Task #87): production paper was
    using bare slippage_rate even after the sqrt_adv plumbing landed,
    because run_paper_trading.py never instantiated a CostModel with
    impact_model="sqrt_adv" and never loaded the per-stock vol/ADV
    snapshot. This helper does both.

    When impact_model is "fixed" (default), returns (None, None) so
    PaperOMS sees no cost_model and no snapshot — behaviour identical
    to pre-fix. When "sqrt_adv", builds + caches today's snapshot
    from qlib historical data and returns the matching CostModel.

    Any failure in snapshot construction degrades gracefully to
    (cost_model, None) so a missing qlib path can never break paper
    trading — the sqrt_adv path then falls back per-fill to bare rate.
    """
    if impact_model == "fixed":
        return None, None

    from backtest.cost_model import CostModel
    cm = CostModel(impact_model="sqrt_adv",
                    impact_coefficient=impact_coefficient)
    try:
        from paper.cost_inputs import load_or_build_snapshot
        # init qlib lazily (only when sqrt_adv is requested — keeps
        # cron startup cheap when running in fixed mode)
        from config.qlib_runtime import init_qlib
        from config.settings import QLIB_PROVIDER_URI
        init_qlib(QLIB_PROVIDER_URI)
        asof = date or datetime.now().strftime("%Y-%m-%d")
        snapshot = load_or_build_snapshot(asof, lookback_days=lookback_days)
        logger.info(
            "Cost inputs: sqrt_adv (coeff=%.3f), snapshot codes=%d",
            impact_coefficient, len(snapshot),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Cost-input snapshot build failed (%s) — sqrt_adv path will "
            "fall back to bare rate per-fill", e,
        )
        snapshot = None
    return cm, snapshot


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run", action="store_true", default=True, help="Run daily paper trading")
    group.add_argument("--status", action="store_true", help="Show current status")
    group.add_argument("--report", action="store_true", help="Show PnL report")
    group.add_argument("--reset", action="store_true", help="Reset paper trading")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--force-stale", action="store_true",
                        help="Run even if prediction freshness check fails (default: abort)")
    # cx round 3 P2 #84 follow-up: sqrt_adv activation knobs
    parser.add_argument(
        "--impact-model", choices=["fixed", "sqrt_adv"], default="fixed",
        help="Cost-model impact branch. 'fixed' (default) preserves pre-fix "
             "bare slippage_rate behaviour. 'sqrt_adv' activates the "
             "Almgren-Chriss path; requires qlib historical data for the "
             "per-stock vol/ADV snapshot."
    )
    parser.add_argument(
        "--impact-coefficient", type=float, default=0.1,
        help="Coefficient on the sqrt_adv slippage term. Ignored when "
             "--impact-model=fixed.",
    )
    parser.add_argument(
        "--cost-lookback-days", type=int, default=20,
        help="Rolling window for vol/ADV snapshot. Default 20 matches "
             "PortfolioBacktest.cost_vol_window default.",
    )
    args = parser.parse_args()

    cost_model, vol_adv_snapshot = _build_cost_inputs(
        date=args.date,
        impact_model=args.impact_model,
        impact_coefficient=args.impact_coefficient,
        lookback_days=args.cost_lookback_days,
    )
    oms = PaperOMS(
        initial_capital=args.capital,
        mode="pending",
        cost_model=cost_model,
        vol_adv_snapshot=vol_adv_snapshot,
    )

    if args.status:
        s = oms.status()
        logger.info("=== Paper Trading Status ===")
        for k, v in s.items():
            logger.info(f"  {k}: {v}")
        return

    if args.report:
        history = oms.state.get("daily_pnl_history", [])
        if not history:
            logger.info("No trading history yet")
            return

        logger.info(f"=== Paper Trading Report ({len(history)} days) ===")
        logger.info(f"{'Date':<12} {'Value':>12} {'Return':>10} {'Positions':>10}")
        logger.info("-" * 50)
        for h in history[-20:]:  # last 20 days
            logger.info(f"{h['date']:<12} {h['total_value']:>12,.2f} "
                        f"{h['daily_return']:>+10.4f} {h['n_positions']:>10}")

        import numpy as np
        returns = [h["daily_return"] for h in history]
        total_ret = oms.state["total_value"] / oms.initial_capital - 1
        logger.info(f"\n  Total return: {total_ret*100:+.2f}%")
        logger.info(f"  Win rate: {np.mean([r > 0 for r in returns])*100:.0f}%")
        logger.info(f"  Avg daily return: {np.mean(returns)*100:+.3f}%")
        logger.info(f"  Total trades: {oms.state['trade_count']}")
        return

    if args.reset:
        state_path = PAPER_DIR / "oms_state.json"
        trades_path = PAPER_DIR / "trades.jsonl"
        if state_path.exists():
            os.remove(str(state_path))
        if trades_path.exists():
            os.remove(str(trades_path))
        logger.info("Paper trading reset")
        return

    # Run daily
    date = args.date or datetime.now().strftime("%Y-%m-%d")
    logger.info(f"=== Paper Trading: {date} ===")

    # --- Freshness check: abort if predictions are stale ---
    # Previously this only logged a warning and continued, generating fresh
    # orders against an old signal. For live paper trading, that's the same
    # silent-degradation pattern as the ST-leak / frozen-state bugs — looks
    # successful while quietly bad. Hard-abort unless the user passes --force-stale.
    # cx round 9 P0-2: require the prediction's recorded latest_date
    # to be at least the expected most-recent trading date, not just
    # that the smoke job exited green. Pre-fix, a same-day re-run of
    # smoke against yesterday's qlib data would write success=True
    # latest_date=yesterday and is_fresh() would say yes — paper would
    # then trade on yesterday's signals against today's market.
    from scheduler.data_health import is_fresh
    pred_fresh = is_fresh("lgb_after_close_smoke", require_latest_date=True)
    if not pred_fresh:
        if not getattr(args, "force_stale", False):
            logger.error(
                "Refusing to run paper trading: no fresh prediction health for today "
                "(lgb_after_close_smoke). Pass --force-stale to override."
            )
            sys.exit(2)
        logger.warning("Continuing on stale predictions — --force-stale set")

    pnl = oms.run_daily(date)

    # Handle pending mode: run_daily may return {"status": "pending"} when
    # T+1 open prices are not yet available (e.g., running same-day after close)
    if isinstance(pnl, dict) and pnl.get("status") == "pending":
        logger.info(f"\n  Orders generated, awaiting T+1 open for reconciliation.")
        logger.info(f"  Run again tomorrow after 10:00 to reconcile.")
        return

    logger.info(f"\nDaily summary:")
    logger.info(f"  Value: {pnl.get('total_value', 0):,.2f}")
    logger.info(f"  Return: {pnl.get('daily_return', 0):+.4f}")
    logger.info(f"  Positions: {pnl.get('n_positions', 0)}")

    # Push notification
    try:
        from push.wechat import WeChatPusher
        s = oms.status()
        stale_tag = " [STALE PREDICTIONS]" if not pred_fresh else ""
        msg = (f"📋 Paper Trading {date}{stale_tag}\n"
               f"Value: {s['total_value']:,.0f}\n"
               f"Return: {s['total_return']*100:+.2f}%\n"
               f"Positions: {s['n_positions']}\n"
               f"Day {s['n_days']}/20")
        WeChatPusher().send(msg, title="Paper Trading")
    except Exception as e:
        logger.warning(f"Push failed: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
