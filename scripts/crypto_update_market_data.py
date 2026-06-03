"""Crypto market-data backfill driver (Phase Crypto-A step 5).

Drives the REST collectors to keep OHLCV + funding + open interest
parquet stores under `/Volumes/DATA/crypto/raw/` up to date. Intended
to run under cron with `--network crypto`:

    python scripts/crypto_update_market_data.py
        [--backfill-days N]   # default 1; deeper backfill on demand
        [--ohlcv-only]
        [--derivatives-only]
        [--no-backfill]       # only fetch recent (current window)
        [--asof YYYY-MM-DD]

Per architecture pivot (`plans/crypto-daemon-architecture-2026-06-03.md`
§7), this script is the BACKFILL data path. It is NEVER the trading
source — the daemon owns live state via WebSocket. Daily reconciliation
between daemon's 1m bar and this script's REST output is what proves
the system is consistent.

A-share isolation: any failure here exits non-zero and the cron
wrapper marks the job failed. A-share crons run in their own
processes and are not affected.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Project root on path so direct invocation works in cron
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.crypto_network import assert_proxy_active
from config.crypto_storage import crypto_root
from config.crypto_universe import (
    PRIMARY_EXCHANGE,
    PHASE_A_TIMEFRAMES,
    spot_symbols_ccxt,
    perp_symbols_ccxt,
)

logger = logging.getLogger(__name__)


# Sleep between fetches to be polite to exchange rate limits even when
# CCXT's enableRateLimit handles the bulk.
_INTER_FETCH_SLEEP_SEC = 0.25


# -----------------------------------------------------------------------------
# OHLCV
# -----------------------------------------------------------------------------

def update_ohlcv(
    *,
    venue: str,
    symbols: list[str],
    timeframes: list[str],
    asof_utc: datetime,
    backfill_days: int,
    no_backfill: bool,
) -> dict:
    """Fetch + write OHLCV for the given (venue, symbol, timeframe) grid.

    Returns a per-(symbol, timeframe) summary dict that the health
    script can consume.
    """
    from data.collectors.crypto_market import (
        fetch_recent, fetch_historical, write_ohlcv_partitions,
    )

    summary: dict = {}
    end_ts_ms = int(asof_utc.timestamp() * 1000)
    start_ts_ms = end_ts_ms - backfill_days * 86_400_000

    for symbol in symbols:
        for tf in timeframes:
            key = f"{symbol}|{tf}"
            try:
                if no_backfill:
                    df = fetch_recent(symbol, tf, venue=venue, limit=200)
                else:
                    df = fetch_historical(
                        symbol, tf,
                        start_ts_ms=start_ts_ms, end_ts_ms=end_ts_ms,
                        venue=venue,
                    )
                if df.empty:
                    summary[key] = {"rows": 0, "written": 0}
                    logger.warning("  %s %s: no rows returned", symbol, tf)
                    continue

                # Drop unclosed bars before writing — backfill stores
                # only closed bars; live unclosed handling is the
                # daemon's job (per architecture doc §7).
                closed = df[df["is_closed_bar"]].copy()
                paths = write_ohlcv_partitions(closed, venue=venue)
                summary[key] = {
                    "rows": len(closed),
                    "written": len(paths),
                }
                logger.info(
                    "  %s %s: %d closed rows → %d partitions",
                    symbol, tf, len(closed), len(paths),
                )
            except Exception as e:  # noqa: BLE001
                summary[key] = {"error": str(e)}
                logger.error("  %s %s: fetch/write FAILED — %s",
                              symbol, tf, e)
            time.sleep(_INTER_FETCH_SLEEP_SEC)
    return summary


# -----------------------------------------------------------------------------
# Funding + OI
# -----------------------------------------------------------------------------

def update_derivatives(
    *,
    venue: str,
    perp_symbols: list[str],
    asof_utc: datetime,
    backfill_days: int,
    no_backfill: bool,
) -> dict:
    """Fetch + write funding history + OI snapshot for the perp list."""
    from data.collectors.crypto_derivatives import (
        fetch_funding_recent,
        fetch_funding_history,
        fetch_open_interest_recent,
        write_funding_partitions,
        write_oi_partitions,
    )

    summary: dict = {}
    end_ts_ms = int(asof_utc.timestamp() * 1000)
    start_ts_ms = end_ts_ms - backfill_days * 86_400_000

    for symbol in perp_symbols:
        # Funding history
        try:
            if no_backfill:
                fdf = fetch_funding_recent(symbol, venue=venue, limit=64)
            else:
                fdf = fetch_funding_history(
                    symbol, start_ts_ms, end_ts_ms, venue=venue,
                )
            written = write_funding_partitions(fdf, venue=venue)
            summary[f"funding|{symbol}"] = {
                "rows": len(fdf), "written": len(written),
            }
            logger.info(
                "  funding %s: %d rows → %d partitions",
                symbol, len(fdf), len(written),
            )
        except Exception as e:  # noqa: BLE001
            summary[f"funding|{symbol}"] = {"error": str(e)}
            logger.error("  funding %s FAILED — %s", symbol, e)
        time.sleep(_INTER_FETCH_SLEEP_SEC)

        # OI snapshot (single point; history needs the live path)
        try:
            oi_df = fetch_open_interest_recent(symbol, venue=venue)
            written = write_oi_partitions(oi_df, venue=venue)
            summary[f"oi|{symbol}"] = {
                "rows": len(oi_df), "written": len(written),
            }
            logger.info(
                "  oi %s: %d row → %d partitions",
                symbol, len(oi_df), len(written),
            )
        except Exception as e:  # noqa: BLE001
            summary[f"oi|{symbol}"] = {"error": str(e)}
            logger.error("  oi %s FAILED — %s", symbol, e)
        time.sleep(_INTER_FETCH_SLEEP_SEC)
    return summary


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def run_update(
    *,
    venue: str = PRIMARY_EXCHANGE,
    backfill_days: int = 1,
    asof_utc: Optional[datetime] = None,
    ohlcv_only: bool = False,
    derivatives_only: bool = False,
    no_backfill: bool = False,
) -> dict:
    """Orchestrate one update pass. Returns the merged summary."""
    assert_proxy_active()  # fail fast if wrapper didn't set sentinels
    root = crypto_root()  # fail fast if storage volume not mounted
    logger.info("Crypto update_market_data: venue=%s root=%s backfill=%dd",
                 venue, root, backfill_days)

    if asof_utc is None:
        asof_utc = datetime.now(timezone.utc)

    summary: dict = {"asof_utc": asof_utc.isoformat(), "venue": venue}

    if not derivatives_only:
        logger.info("== OHLCV pass ==")
        summary["ohlcv"] = update_ohlcv(
            venue=venue,
            symbols=spot_symbols_ccxt(),
            timeframes=list(PHASE_A_TIMEFRAMES),
            asof_utc=asof_utc,
            backfill_days=backfill_days,
            no_backfill=no_backfill,
        )

    if not ohlcv_only:
        logger.info("== Derivatives pass ==")
        summary["derivatives"] = update_derivatives(
            venue=venue,
            perp_symbols=perp_symbols_ccxt(),
            asof_utc=asof_utc,
            backfill_days=backfill_days,
            no_backfill=no_backfill,
        )

    return summary


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--venue", default=PRIMARY_EXCHANGE)
    p.add_argument("--backfill-days", type=int, default=1)
    p.add_argument("--asof", type=str, default=None,
                    help="UTC date YYYY-MM-DD; default = now()")
    p.add_argument("--ohlcv-only", action="store_true")
    p.add_argument("--derivatives-only", action="store_true")
    p.add_argument("--no-backfill", action="store_true",
                    help="Only fetch recent (limit 200); skip historical pagination")
    args = p.parse_args()

    asof_utc = None
    if args.asof:
        asof_utc = datetime.fromisoformat(args.asof).replace(tzinfo=timezone.utc)

    summary = run_update(
        venue=args.venue,
        backfill_days=args.backfill_days,
        asof_utc=asof_utc,
        ohlcv_only=args.ohlcv_only,
        derivatives_only=args.derivatives_only,
        no_backfill=args.no_backfill,
    )

    # Exit non-zero if any task errored — cron wrapper marks job failed
    n_errors = sum(
        1 for section in ("ohlcv", "derivatives")
        for v in summary.get(section, {}).values()
        if isinstance(v, dict) and "error" in v
    )
    if n_errors:
        logger.error("Update pass had %d errors", n_errors)
        sys.exit(1)
    logger.info("Update pass clean")


if __name__ == "__main__":
    main()
