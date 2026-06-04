"""Run the Phase Crypto-C funding-arbitrage backtest on real or
synthetic data.

Usage:
    # Real data (requires Phase Crypto-A step 5 has populated funding parquet)
    python scripts/crypto_backtest_funding_arb.py \\
        --symbol BTC/USDT:USDT --venue binance \\
        --start 2024-01-01 --end 2026-06-01

    # Synthetic stress test (for sanity / contract demos)
    python scripts/crypto_backtest_funding_arb.py --synthetic

Output: a JSON report under `crypto_root/reports/backtest/funding_arb/
{venue}/{symbol_canonical}_{startdate}_{enddate}.json` (when crypto_root
is mounted) AND a console summary including the acceptance verdict.

Exit code: 0 on backtest success (verdict regardless), 1 if data
loading failed, 2 if storage is unreachable.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.crypto.funding_arb import (
    FundingArbBacktestConfig, backtest_funding_arb,
)

logger = logging.getLogger(__name__)


def _load_funding_parquet(symbol: str, venue: str) -> Optional[pd.DataFrame]:
    """Load all month partitions for (symbol, venue) under crypto_root.
    Returns None when storage is unreachable."""
    try:
        from config.crypto_storage import crypto_root
        from data.collectors.crypto_derivatives import to_canonical_perp_symbol
    except Exception as e:  # noqa: BLE001
        logger.warning("config load failed: %s", e)
        return None

    try:
        root = crypto_root()
    except Exception as e:  # noqa: BLE001
        logger.error("crypto_root unreachable: %s", e)
        return None

    sym_canon = to_canonical_perp_symbol(symbol, venue)
    base = root / "raw" / "funding" / venue.lower() / sym_canon
    if not base.exists():
        logger.warning("no funding partitions under %s", base)
        return None
    parquet_files = sorted(base.rglob("*.parquet"))
    if not parquet_files:
        logger.warning("funding directory empty: %s", base)
        return None
    frames = [pd.read_parquet(p) for p in parquet_files]
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp_utc")
    logger.info(
        "loaded %d funding events from %d files for %s",
        len(df), len(parquet_files), sym_canon,
    )
    return df


def _build_synthetic_funding(n_events: int = 1095,
                              seed: int = 0,
                              cadence_sec: int = 8 * 3600) -> pd.DataFrame:
    """Synthetic funding-rate series for sanity tests / demos.

    Designed so the strategy actually opens, closes and flips at
    realistic rates. Funding rates drawn from N(2 bps, 5 bps) with a
    long-tail mix (10% chance of |rate| in 10-30 bps).
    """
    rng = np.random.default_rng(seed)
    base = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    rows = []
    rates = rng.normal(loc=0.0002, scale=0.0005, size=n_events)
    big_mask = rng.random(n_events) < 0.1
    rates[big_mask] = rng.uniform(0.001, 0.003, size=big_mask.sum()) * rng.choice([1, -1], size=big_mask.sum())
    for i, r in enumerate(rates):
        rows.append({
            "timestamp_utc": base + i * cadence_sec * 1000,
            "exchange": "synthetic",
            "symbol": "BTC/USDT:USDT",
            "funding_rate": float(r),
            "next_funding_ts": None,
            "mark_price": None,
            "index_price": None,
            "ingested_at": base + i * cadence_sec * 1000 + 5000,
        })
    return pd.DataFrame(rows)


def _write_report(report: dict, *, symbol: str, venue: str,
                   start: str, end: str) -> Optional[Path]:
    try:
        from config.crypto_storage import crypto_root
        from data.collectors.crypto_derivatives import to_canonical_perp_symbol
    except Exception:
        return None
    try:
        root = crypto_root()
    except Exception:
        return None
    out_dir = root / "reports" / "backtest" / "funding_arb" / venue.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    sym_canon = to_canonical_perp_symbol(symbol, venue)
    out = out_dir / f"{sym_canon}_{start}_{end}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return out


def run(symbol: str, venue: str, *,
         start: Optional[str], end: Optional[str],
         synthetic: bool, config: FundingArbBacktestConfig) -> dict:
    if synthetic:
        df = _build_synthetic_funding()
    else:
        df = _load_funding_parquet(symbol, venue)

    if df is None or df.empty:
        return {"status": "ERROR", "reason": "no_funding_data"}

    config.start_date_utc = start
    config.end_date_utc = end
    result = backtest_funding_arb(df, config)
    passes, note = result.passes_acceptance()
    report = {
        "symbol": symbol,
        "venue": venue,
        "start": start,
        "end": end,
        "config": config.__dict__,
        "metrics": {
            "n_events": result.n_events,
            "n_open_events": result.n_open_events,
            "n_close_events": result.n_close_events,
            "n_flip_events": result.n_flip_events,
            "gross_pnl_usd": result.gross_pnl_usd,
            "net_pnl_usd": result.net_pnl_usd,
            "fees_paid_usd": result.fees_paid_usd,
            "slippage_paid_usd": result.slippage_paid_usd,
            # cx round 29 P2: report BOTH denominators so the operator
            # never reads the rosier notional number by accident.
            "after_cost_apr": result.after_cost_apr,  # alias of on_capital
            "after_cost_apr_on_notional": result.after_cost_apr_on_notional,
            "after_cost_apr_on_capital": result.after_cost_apr_on_capital,
            "after_cost_sharpe": result.after_cost_sharpe,
            "max_drawdown": result.max_drawdown,  # alias of on_capital
            "max_drawdown_on_notional": result.max_drawdown_on_notional,
            "max_drawdown_on_capital": result.max_drawdown_on_capital,
            "capital_required_usd": result.capital_required_usd,
            "effective_notional_usd": result.effective_notional_usd,
        },
        "acceptance": {"pass": passes, "note": note},
        "summary": result.summary(),
    }
    return report


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--venue", default="binance")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--synthetic", action="store_true",
                    help="Run on synthetic funding series instead of parquet")
    p.add_argument("--notional", type=float, default=5_000.0,
                    help="USD notional per leg")
    p.add_argument("--write", action="store_true",
                    help="Write the JSON report to crypto_root/reports/...")
    args = p.parse_args()

    cfg = FundingArbBacktestConfig(notional_per_trade_usd=args.notional)
    report = run(
        symbol=args.symbol, venue=args.venue,
        start=args.start, end=args.end,
        synthetic=args.synthetic, config=cfg,
    )

    if report.get("status") == "ERROR":
        logger.error("Backtest aborted: %s", report["reason"])
        sys.exit(1)

    if args.write:
        out = _write_report(report, symbol=args.symbol, venue=args.venue,
                              start=args.start or "all", end=args.end or "now")
        if out is not None:
            logger.info("Report written: %s", out)

    # Console summary
    print(json.dumps({
        "summary": report["summary"],
        "acceptance": report["acceptance"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
