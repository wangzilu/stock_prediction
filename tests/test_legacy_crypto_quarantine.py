"""Tests for Legacy Crypto Quarantine.

Per `plans/cc-crypto-implementation-spec-2026-05-30.md` §6.5 + cx
system-design-review punch list #2.

With LEGACY_MARKET_CONTEXT_ENABLED=false (default):
1. Importing scheduler.jobs must not load data.collectors.crypto.
2. DailyPipeline must not instantiate any CryptoCollector at init.
3. A-share daily run must not make any network call to crypto exchanges.
"""

from __future__ import annotations

import importlib
import sys

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _force_default_flag(monkeypatch):
    """Ensure LEGACY_MARKET_CONTEXT_ENABLED is observed as false from env
    and reflected in config.feature_flags after a reload."""
    monkeypatch.delenv("LEGACY_MARKET_CONTEXT_ENABLED", raising=False)
    monkeypatch.setenv("LEGACY_MARKET_CONTEXT_ENABLED", "false")
    if "config.feature_flags" in sys.modules:
        importlib.reload(sys.modules["config.feature_flags"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_init_does_not_import_legacy_crypto(monkeypatch):
    """With flag default-false, importing + initializing DailyPipeline
    must not load `data.collectors.crypto`. Module-level import was
    removed in the quarantine PR — lazy access is gated by the flag.
    """
    _force_default_flag(monkeypatch)
    # Wipe any prior load of the legacy module so we can assert
    # cleanly that the pipeline import does not bring it back.
    sys.modules.pop("data.collectors.crypto", None)

    # Reload scheduler.jobs to pick up any prior state changes.
    if "scheduler.jobs" in sys.modules:
        importlib.reload(sys.modules["scheduler.jobs"])
    else:
        importlib.import_module("scheduler.jobs")

    assert "data.collectors.crypto" not in sys.modules, (
        "scheduler.jobs import triggered legacy crypto module load. "
        "Quarantine requires the module-level import to remain removed."
    )

    from scheduler.jobs import DailyPipeline

    pipeline = DailyPipeline()

    # After init, legacy crypto module must still be unloaded.
    assert "data.collectors.crypto" not in sys.modules
    # And no CryptoCollector instance has been built.
    assert pipeline._crypto_collector is None
    assert pipeline._get_crypto_collector() is None


def test_get_crypto_collector_returns_none_when_flag_off(monkeypatch):
    """The lazy accessor returns None when the flag is off."""
    _force_default_flag(monkeypatch)
    sys.modules.pop("data.collectors.crypto", None)

    from scheduler.jobs import DailyPipeline

    pipeline = DailyPipeline()
    assert pipeline._get_crypto_collector() is None
    # _fetch_crypto_market_data returns an empty dict, not None, so
    # downstream consumers (LLM analyst, report builders) handle it
    # the same as "no quote data available".
    assert pipeline._fetch_crypto_market_data() == {}


def test_no_crypto_network_call_during_ashare_imports(monkeypatch):
    """Per cx system-design-review punch list #2: with flag false, just
    importing and instantiating the A-share daily pipeline must not
    cause any outbound HTTP request to known crypto exchange hosts.

    This is the weakest possible version of the contract — it does not
    run the full pipeline, only proves the import + init path is clean.
    A stronger version that runs morning_recommendation needs more
    fixtures and is owned by the full quarantine PR.
    """
    _force_default_flag(monkeypatch)

    outbound_urls: list[str] = []

    # Best-effort instrumentation: patch requests.Session.send. CCXT
    # builds HTTP clients through requests under the hood.
    try:
        import requests.sessions
    except ImportError:
        pytest.skip("requests not installed; cannot instrument network")
        return

    orig_send = requests.sessions.Session.send

    def _trace_send(self, request, **kwargs):  # noqa: ANN001
        outbound_urls.append(request.url)
        return orig_send(self, request, **kwargs)

    monkeypatch.setattr(requests.sessions.Session, "send", _trace_send)

    sys.modules.pop("data.collectors.crypto", None)

    # Reload the relevant modules in a clean state.
    if "scheduler.jobs" in sys.modules:
        importlib.reload(sys.modules["scheduler.jobs"])

    from scheduler.jobs import DailyPipeline

    DailyPipeline()  # init only — no methods called

    crypto_host_patterns = (
        "binance.com",
        "okx.com",
        "bybit.com",
        "kraken.com",
        "coinbase.com",
    )
    offending = [
        url for url in outbound_urls
        if any(p in url for p in crypto_host_patterns)
    ]
    assert not offending, (
        f"DailyPipeline init triggered crypto network calls "
        f"despite LEGACY_MARKET_CONTEXT_ENABLED=false: {offending}"
    )
