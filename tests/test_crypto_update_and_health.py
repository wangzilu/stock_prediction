"""Tests for Crypto-A step 5 — driver scripts.

scripts/crypto_update_market_data.py and scripts/crypto_data_health.py
are driver scripts on top of the already-tested REST collectors.
These tests are offline:
  - REST collectors are monkeypatched with stub functions returning
    fixture frames
  - crypto_root via CRYPTO_STORAGE_ROOT env to tmp_path
  - assert_proxy_active sentinels set/unset per test

Coverage:
  - run_update fails fast when ssproxy sentinels missing
  - run_update fails fast when storage volume not mounted
  - run_update with --ohlcv-only does NOT call derivatives collectors
  - run_update with --derivatives-only does NOT call OHLCV collectors
  - main_pipeline writes partitions via the existing write_ohlcv_*
    helpers (smoke test: result frames non-empty)
  - build_health_report categorises status correctly:
      MISSING storage volume → RED
      missing symbol directories → YELLOW
      stale data per §11 → YELLOW
      fresh data → GREEN
  - build_health_report writes health.json under
    crypto_root/health/{date}.json when --write
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest


def _activate_proxy_env(monkeypatch):
    from config import crypto_network as cn
    monkeypatch.setenv(cn.CRYPTO_NETWORK_ENV, "crypto")
    monkeypatch.setenv(cn.CRYPTO_SSPROXY_ENV, "verified")


# -----------------------------------------------------------------------------
# update_market_data
# -----------------------------------------------------------------------------

def test_run_update_refuses_without_proxy_sentinels(monkeypatch, tmp_path):
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    from config import crypto_network as cn
    monkeypatch.delenv(cn.CRYPTO_NETWORK_ENV, raising=False)
    monkeypatch.delenv(cn.CRYPTO_SSPROXY_ENV, raising=False)
    from scripts.crypto_update_market_data import run_update
    with pytest.raises(cn.CryptoProxyNotActiveError):
        run_update()


def test_run_update_refuses_without_storage(monkeypatch):
    _activate_proxy_env(monkeypatch)
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", "/no/such/path")
    from config.crypto_storage import CryptoStorageNotMountedError
    from scripts.crypto_update_market_data import run_update
    with pytest.raises(CryptoStorageNotMountedError):
        run_update()


def _stub_market_collectors(monkeypatch):
    """Patch fetch_recent / fetch_historical / write_ohlcv_partitions."""
    from data.collectors import crypto_market
    calls = {"fetch_recent": 0, "fetch_historical": 0, "write": 0}

    def _fake_recent(symbol, tf, *, venue="binance", limit=200, now_ms=None):
        calls["fetch_recent"] += 1
        return _ohlcv_fixture_df(symbol, tf, venue, n=5)

    def _fake_historical(symbol, tf, start_ts_ms, end_ts_ms,
                         *, venue="binance", page_size=1000, now_ms=None):
        calls["fetch_historical"] += 1
        return _ohlcv_fixture_df(symbol, tf, venue, n=10)

    def _fake_write(df, *, venue, instrument_class="spot", root=None):
        calls["write"] += 1
        return [Path("/tmp/fake.parquet")]

    monkeypatch.setattr(crypto_market, "fetch_recent", _fake_recent)
    monkeypatch.setattr(crypto_market, "fetch_historical", _fake_historical)
    monkeypatch.setattr(crypto_market, "write_ohlcv_partitions", _fake_write)
    return calls


def _stub_derivative_collectors(monkeypatch):
    from data.collectors import crypto_derivatives
    calls = {"funding_recent": 0, "funding_history": 0,
             "oi_recent": 0, "write_funding": 0, "write_oi": 0}

    def _fake_funding_recent(symbol, *, venue="binance", limit=64, now_ms=None):
        calls["funding_recent"] += 1
        return _funding_fixture_df(symbol, venue, n=3)

    def _fake_funding_history(symbol, start, end, *, venue="binance",
                               page_size=1000, now_ms=None):
        calls["funding_history"] += 1
        return _funding_fixture_df(symbol, venue, n=6)

    def _fake_oi_recent(symbol, *, venue="binance", now_ms=None):
        calls["oi_recent"] += 1
        return _oi_fixture_df(symbol, venue, n=1)

    def _fake_write_funding(df, *, venue, root=None):
        calls["write_funding"] += 1
        return [Path("/tmp/funding.parquet")]

    def _fake_write_oi(df, *, venue, root=None):
        calls["write_oi"] += 1
        return [Path("/tmp/oi.parquet")]

    monkeypatch.setattr(crypto_derivatives, "fetch_funding_recent", _fake_funding_recent)
    monkeypatch.setattr(crypto_derivatives, "fetch_funding_history", _fake_funding_history)
    monkeypatch.setattr(crypto_derivatives, "fetch_open_interest_recent", _fake_oi_recent)
    monkeypatch.setattr(crypto_derivatives, "write_funding_partitions", _fake_write_funding)
    monkeypatch.setattr(crypto_derivatives, "write_oi_partitions", _fake_write_oi)
    return calls


def _ohlcv_fixture_df(symbol, tf, venue, n=5):
    """Build a fake OHLCV frame matching contract §3 schema."""
    base_ts = 1_700_000_000_000
    tf_ms = {"1m": 60_000, "5m": 300_000, "15m": 900_000,
             "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}[tf]
    return pd.DataFrame([{
        "timestamp_utc": base_ts + i * tf_ms,
        "exchange": venue,
        "symbol": symbol,
        "timeframe": tf,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume_base": 10.0, "volume_quote": 1000.0,
        "quote_volume_estimated": True,
        "trades": -1,
        "is_closed_bar": True,
        "ingested_at": base_ts + i * tf_ms + 5000,
    } for i in range(n)])


def _funding_fixture_df(symbol, venue, n=3):
    base_ts = 1_700_000_000_000
    return pd.DataFrame([{
        "timestamp_utc": base_ts + i * 28_800_000,
        "exchange": venue, "symbol": symbol,
        "funding_rate": 0.0001 * (i + 1),
        "next_funding_ts": None, "mark_price": 50000.0,
        "index_price": 50001.0,
        "ingested_at": base_ts + i * 28_800_000 + 5000,
    } for i in range(n)])


def _oi_fixture_df(symbol, venue, n=1):
    base_ts = 1_700_000_000_000
    return pd.DataFrame([{
        "timestamp_utc": base_ts + i * 900_000,
        "exchange": venue, "symbol": symbol,
        "open_interest": 100.0 + i, "oi_quote": 5_000_000.0,
        "long_short_ratio": 1.5,
        "ingested_at": base_ts + i * 900_000 + 1000,
    } for i in range(n)])


def test_run_update_calls_both_ohlcv_and_derivatives_by_default(
    monkeypatch, tmp_path,
):
    _activate_proxy_env(monkeypatch)
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    market_calls = _stub_market_collectors(monkeypatch)
    deriv_calls = _stub_derivative_collectors(monkeypatch)

    from scripts.crypto_update_market_data import run_update
    summary = run_update(backfill_days=1)
    assert market_calls["fetch_historical"] > 0
    assert deriv_calls["funding_history"] > 0
    assert deriv_calls["oi_recent"] > 0
    assert "ohlcv" in summary
    assert "derivatives" in summary


def test_run_update_ohlcv_only_skips_derivatives(monkeypatch, tmp_path):
    _activate_proxy_env(monkeypatch)
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    market_calls = _stub_market_collectors(monkeypatch)
    deriv_calls = _stub_derivative_collectors(monkeypatch)

    from scripts.crypto_update_market_data import run_update
    summary = run_update(ohlcv_only=True)
    assert market_calls["fetch_historical"] > 0
    assert deriv_calls["funding_history"] == 0
    assert deriv_calls["oi_recent"] == 0
    assert "derivatives" not in summary


def test_run_update_derivatives_only_skips_ohlcv(monkeypatch, tmp_path):
    _activate_proxy_env(monkeypatch)
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    market_calls = _stub_market_collectors(monkeypatch)
    deriv_calls = _stub_derivative_collectors(monkeypatch)

    from scripts.crypto_update_market_data import run_update
    summary = run_update(derivatives_only=True)
    assert market_calls["fetch_historical"] == 0
    assert deriv_calls["funding_history"] > 0
    assert "ohlcv" not in summary


def test_run_update_no_backfill_uses_recent_endpoint(monkeypatch, tmp_path):
    _activate_proxy_env(monkeypatch)
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    market_calls = _stub_market_collectors(monkeypatch)
    deriv_calls = _stub_derivative_collectors(monkeypatch)

    from scripts.crypto_update_market_data import run_update
    run_update(no_backfill=True)
    assert market_calls["fetch_recent"] > 0
    assert market_calls["fetch_historical"] == 0
    assert deriv_calls["funding_recent"] > 0
    assert deriv_calls["funding_history"] == 0


# -----------------------------------------------------------------------------
# data_health
# -----------------------------------------------------------------------------

def test_health_red_when_storage_missing(monkeypatch):
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", "/no/such/volume")
    from scripts.crypto_data_health import build_health_report
    report = build_health_report()
    assert report["status"] == "RED"
    assert "not mounted" in report["reason"].lower()


def test_health_yellow_when_no_symbols(monkeypatch, tmp_path):
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    from scripts.crypto_data_health import build_health_report
    report = build_health_report()
    assert report["status"] == "YELLOW"


def test_health_green_when_data_is_fresh(monkeypatch, tmp_path):
    """Write a fresh 1h OHLCV partition and verify status flips to
    GREEN."""
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))

    # Build a fixture parquet at the §3 layout
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    df = pd.DataFrame([{
        "timestamp_utc": now_ms - 60_000,   # 1 min ago — well within 1h lag
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
        "volume_base": 1.0, "volume_quote": 1.0,
        "quote_volume_estimated": True,
        "trades": -1,
        "is_closed_bar": True,
        "ingested_at": now_ms,
    }])
    part_dir = (
        tmp_path / "raw" / "ohlcv" / "binance" / "binance__btc_usdt__spot"
        / "1h" / "year=2025" / "month=11"
    )
    part_dir.mkdir(parents=True)
    df.to_parquet(part_dir / "day=15.parquet")

    from scripts.crypto_data_health import build_health_report
    report = build_health_report()
    assert report["status"] == "GREEN"


def test_health_yellow_when_stale(monkeypatch, tmp_path):
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    # 1h timeframe max lag is 5400s; write a row 10 hours old
    old_ms = int(datetime.now(timezone.utc).timestamp() * 1000) - 10 * 3600_000
    df = pd.DataFrame([{
        "timestamp_utc": old_ms,
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
        "volume_base": 1.0, "volume_quote": 1.0,
        "quote_volume_estimated": True,
        "trades": -1,
        "is_closed_bar": True,
        "ingested_at": old_ms,
    }])
    part_dir = (
        tmp_path / "raw" / "ohlcv" / "binance" / "binance__btc_usdt__spot"
        / "1h" / "year=2025" / "month=11"
    )
    part_dir.mkdir(parents=True)
    df.to_parquet(part_dir / "day=15.parquet")

    from scripts.crypto_data_health import build_health_report
    report = build_health_report()
    assert report["status"] == "YELLOW"
    assert "max_lag" in report["status_reason"].lower() or "stale" in report["status_reason"].lower() or "exceeds" in report["status_reason"].lower()


def test_health_writes_json_to_health_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    from scripts.crypto_data_health import build_health_report, write_health
    from config.crypto_storage import crypto_root

    report = build_health_report(asof_utc=datetime(2026, 6, 3, tzinfo=timezone.utc))
    out = write_health(report, crypto_root())
    assert out.exists()
    assert out.name == "2026-06-03.json"
    parsed = json.loads(out.read_text())
    assert parsed["status"] == report["status"]
