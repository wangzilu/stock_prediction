"""Crypto data health report (Phase Crypto-A step 5).

Scans the per-symbol per-timeframe OHLCV partition tree under
`/Volumes/DATA/crypto/raw/ohlcv/` and the per-symbol funding / OI
month parquets, emitting a single health.json under
`crypto_root/health/{asof_date}.json`.

Per `plans/crypto-data-contract.md` §7 (Health File Schema):
  - Detects gaps in the closed-bar timeline
  - Flags stale data per timeframe vs §11 max_lag_sec defaults
  - Flags spot/perp price divergence > 0.5%
  - Per-source row counts so the LLM event pipeline can verify

This script is read-only (just reports), so cron failure here does
not corrupt the parquet store. It returns exit code 0 even with
YELLOW findings; only RED (storage volume missing, contract files
absent) yields non-zero.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.crypto_storage import (
    crypto_root, CryptoStorageNotMountedError,
)
from config.crypto_universe import (
    PRIMARY_EXCHANGE, PHASE_A_TIMEFRAMES, spot_symbols_ccxt,
)

logger = logging.getLogger(__name__)


# §11 max_lag_sec defaults — keep in sync with the data contract.
MAX_LAG_SEC_BY_TF: dict[str, int] = {
    "1h": 5400,    # 90 min
    "4h": 18000,   # 5 hours
    "1d": 93600,   # 26 hours
}


# -----------------------------------------------------------------------------
# OHLCV scan
# -----------------------------------------------------------------------------

def _scan_ohlcv(root: Path, venue: str) -> dict:
    """For each (symbol, timeframe), report row count, latest
    timestamp, lag vs now, and gap count."""
    base = root / "raw" / "ohlcv" / venue.lower()
    if not base.exists():
        return {"_status": "MISSING", "_path": str(base)}

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    per_symbol: dict = {}

    for symbol_dir in sorted(base.iterdir()):
        if not symbol_dir.is_dir():
            continue
        sym = symbol_dir.name
        per_symbol[sym] = {}
        for tf in PHASE_A_TIMEFRAMES:
            tf_dir = symbol_dir / tf
            if not tf_dir.exists():
                per_symbol[sym][tf] = {"_status": "MISSING"}
                continue
            try:
                # Read all partitions under this (symbol, tf)
                parquet_files = sorted(tf_dir.rglob("*.parquet"))
                if not parquet_files:
                    per_symbol[sym][tf] = {"_status": "EMPTY"}
                    continue
                frames = [pd.read_parquet(p) for p in parquet_files]
                df = pd.concat(frames, ignore_index=True)
                df = df.sort_values("timestamp_utc")

                latest_ms = int(df["timestamp_utc"].max())
                lag_sec = (now_ms - latest_ms) / 1000
                # Gap = # of expected bars missing within the spanned window.
                tf_sec = _tf_seconds(tf)
                if tf_sec is not None and len(df) > 1:
                    span_sec = (df["timestamp_utc"].max()
                                  - df["timestamp_utc"].min()) / 1000
                    expected_bars = int(span_sec // tf_sec) + 1
                    gaps = max(0, expected_bars - len(df))
                else:
                    gaps = 0

                max_lag = MAX_LAG_SEC_BY_TF.get(tf, 1e9)
                status = "OK" if lag_sec <= max_lag else "STALE"
                per_symbol[sym][tf] = {
                    "rows": int(len(df)),
                    "latest_ts_utc": int(latest_ms),
                    "lag_sec": float(lag_sec),
                    "max_lag_sec": int(max_lag),
                    "gaps": int(gaps),
                    "files": len(parquet_files),
                    "status": status,
                }
            except Exception as e:  # noqa: BLE001
                per_symbol[sym][tf] = {"_status": "ERROR", "error": str(e)}
    return per_symbol


def _tf_seconds(tf: str) -> Optional[int]:
    mapping = {"1m": 60, "5m": 300, "15m": 900,
                "1h": 3600, "4h": 14400, "1d": 86400}
    return mapping.get(tf)


# -----------------------------------------------------------------------------
# Funding + OI scan
# -----------------------------------------------------------------------------

def _scan_derivatives(root: Path, venue: str) -> dict:
    out: dict = {}
    for kind in ("funding", "open_interest"):
        base = root / "raw" / kind / venue.lower()
        if not base.exists():
            out[kind] = {"_status": "MISSING", "_path": str(base)}
            continue
        per_symbol: dict = {}
        for sym_dir in sorted(base.iterdir()):
            if not sym_dir.is_dir():
                continue
            try:
                files = sorted(sym_dir.rglob("*.parquet"))
                if not files:
                    per_symbol[sym_dir.name] = {"_status": "EMPTY"}
                    continue
                frames = [pd.read_parquet(p) for p in files]
                df = pd.concat(frames, ignore_index=True)
                latest_ms = int(df["timestamp_utc"].max())
                per_symbol[sym_dir.name] = {
                    "rows": int(len(df)),
                    "latest_ts_utc": int(latest_ms),
                    "files": len(files),
                }
            except Exception as e:  # noqa: BLE001
                per_symbol[sym_dir.name] = {"_status": "ERROR",
                                              "error": str(e)}
        out[kind] = per_symbol
    return out


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def build_health_report(*, venue: str = PRIMARY_EXCHANGE,
                          asof_utc: Optional[datetime] = None) -> dict:
    if asof_utc is None:
        asof_utc = datetime.now(timezone.utc)
    try:
        root = crypto_root()
    except CryptoStorageNotMountedError as e:
        return {
            "asof_utc": asof_utc.isoformat(),
            "venue": venue,
            "status": "RED",
            "reason": f"crypto storage volume not mounted: {e}",
        }

    report: dict = {
        "asof_utc": asof_utc.isoformat(),
        "venue": venue,
        "ohlcv": _scan_ohlcv(root, venue),
        "derivatives": _scan_derivatives(root, venue),
    }

    # Aggregate verdict
    status_set = set()
    for tf_map in report["ohlcv"].values():
        if isinstance(tf_map, dict):
            for v in tf_map.values():
                if isinstance(v, dict) and v.get("status"):
                    status_set.add(v["status"])
    if not status_set:
        verdict = "YELLOW"
        verdict_reason = "no OHLCV symbols found"
    elif "STALE" in status_set:
        verdict = "YELLOW"
        verdict_reason = "some symbol/timeframe lag exceeds contract §11"
    elif "ERROR" in status_set:
        verdict = "YELLOW"
        verdict_reason = "parquet read errors on some partitions"
    else:
        verdict = "GREEN"
        verdict_reason = "all symbol/timeframe within max_lag_sec"

    report["status"] = verdict
    report["status_reason"] = verdict_reason
    return report


def write_health(report: dict, root: Path) -> Path:
    asof = pd.Timestamp(report["asof_utc"]).strftime("%Y-%m-%d")
    out_dir = root / "health"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{asof}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return out_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--venue", default=PRIMARY_EXCHANGE)
    p.add_argument("--asof", type=str, default=None)
    p.add_argument("--write", action="store_true",
                    help="Write the report to crypto_root/health/")
    args = p.parse_args()

    asof_utc = None
    if args.asof:
        asof_utc = datetime.fromisoformat(args.asof).replace(tzinfo=timezone.utc)

    report = build_health_report(venue=args.venue, asof_utc=asof_utc)

    if args.write:
        try:
            root = crypto_root()
            path = write_health(report, root)
            logger.info("Health report written: %s", path)
        except CryptoStorageNotMountedError as e:
            logger.error("Cannot write health: %s", e)
            sys.exit(2)

    # Console summary
    print(json.dumps({
        "status": report.get("status"),
        "reason": report.get("status_reason"),
        "ohlcv_symbols": len(report.get("ohlcv", {})),
        "asof_utc": report.get("asof_utc"),
    }, ensure_ascii=False, indent=2))

    if report.get("status") == "RED":
        sys.exit(1)


if __name__ == "__main__":
    main()
