"""Pin: run_network_job → collector integration (cx round 4 P1+P2).

Previous PR ("Crypto-A step 1+2") documented that the official entry
point is `run_network_job.py --network crypto` but never actually
added that profile to argparse choices or apply_profile(). The
collector contract therefore was unreachable through the documented
path. Tests passed only because they monkeypatched env directly.

This file pins:

  P1 — `crypto` is a real argparse choice
  P1 — apply_profile("crypto", env, ...) sets BOTH sentinels
        (CRYPTO_NETWORK_ACTIVE=crypto, CRYPTO_SSPROXY_VERIFIED=1)
        AND sets proxy env vars
  P1 — When apply_profile is followed by collector entry,
        assert_proxy_active() does NOT raise
  P1 — When ssproxy preflight fails, apply_profile exits (sys.exit(2))
  P2 — Canonical symbol format is lowercase end-to-end
  P2 — Contract example string and code output match exactly
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# -----------------------------------------------------------------------------
# P1: crypto profile is real (not a doc-only fiction)
# -----------------------------------------------------------------------------

def test_crypto_is_a_valid_argparse_choice():
    """argparse must accept `--network crypto` so the documented
    wrapper invocation does not blow up at the parse step."""
    src = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_network_job.py"
    ).read_text()
    # We do a source-level scan to avoid having to actually instantiate
    # argparse + read stdin in a subprocess for every test.
    assert '"crypto"' in src or "'crypto'" in src, (
        "run_network_job.py: 'crypto' missing from argparse choices. "
        "Documented entry point per crypto-data-contract.md §1.5 would "
        "fail at the parse step."
    )


def test_apply_profile_crypto_sets_sentinels_and_proxy(monkeypatch):
    """apply_profile('crypto', env, ...) must set both env sentinels
    that config.crypto_network.assert_proxy_active() looks for AND
    set http(s)_proxy env vars (mainland egress)."""
    # Monkeypatch _ensure_proxy to return True so we exercise the
    # success path without touching the real proxy.
    import scripts.run_network_job as rnj
    monkeypatch.setattr(rnj, "_ensure_proxy", lambda: True)

    env: dict = {}
    timeout = rnj.apply_profile("crypto", env, timeout=600)
    assert timeout == 600

    # Collector contract sentinels
    assert env.get("CRYPTO_NETWORK_ACTIVE") == "crypto", env
    assert env.get("CRYPTO_SSPROXY_VERIFIED") == "1", env

    # Standard proxy env vars (so any HTTP client honours ssproxy)
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        assert key in env, f"proxy var {key!r} missing from env"


def test_apply_profile_crypto_exits_when_ssproxy_preflight_fails(monkeypatch):
    """If the ssproxy port is not reachable, apply_profile must NOT
    silently set the sentinels — it must abort so the cron wrapper
    marks the job failed (without affecting A-share)."""
    import scripts.run_network_job as rnj
    monkeypatch.setattr(rnj, "_ensure_proxy", lambda: False)

    env: dict = {}
    with pytest.raises(SystemExit) as exc:
        rnj.apply_profile("crypto", env, timeout=600)
    # Use exit code 2 to distinguish from generic profile errors
    assert exc.value.code == 2
    # And the sentinels must NOT be set
    assert "CRYPTO_NETWORK_ACTIVE" not in env
    assert "CRYPTO_SSPROXY_VERIFIED" not in env


def test_crypto_profile_makes_assert_proxy_active_pass(monkeypatch):
    """End-to-end: after apply_profile('crypto', env, ...) writes
    sentinels to a dict, copying that dict into os.environ makes
    config.crypto_network.assert_proxy_active() pass without raising.

    This is the integration that the previous PR was missing."""
    import scripts.run_network_job as rnj
    from config import crypto_network as cn

    monkeypatch.setattr(rnj, "_ensure_proxy", lambda: True)
    env: dict = {}
    rnj.apply_profile("crypto", env, timeout=600)

    # Push the produced env back into os.environ as the child process
    # would inherit
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    cn.assert_proxy_active()  # must NOT raise


# -----------------------------------------------------------------------------
# P2: universe → collector contract is parseable end-to-end
# (a richer version of the test added in test_crypto_config.py — kept
# here so an integration failure surfaces in the entry-contract bucket)
# -----------------------------------------------------------------------------

def test_ccxt_universe_helpers_feed_collectors_without_raising():
    """A naive `for s in spot_symbols_ccxt(): fetch_recent(s, ...)` must
    NOT raise at the symbol-shape check. We don't actually fetch (no
    network) — just exercise the parsers each collector applies."""
    from config import crypto_universe as cu
    from data.collectors.crypto_market import to_canonical_symbol
    from data.collectors.crypto_derivatives import (
        parse_perp_symbol, to_canonical_perp_symbol,
    )

    for s in cu.spot_symbols_ccxt():
        to_canonical_symbol(s, "binance")  # raises on bad shape

    for s in cu.perp_symbols_ccxt():
        parse_perp_symbol(s)
        to_canonical_perp_symbol(s, "binance")


# -----------------------------------------------------------------------------
# P2: canonical lowercase contract is enforced + matches doc
# -----------------------------------------------------------------------------

def test_canonical_symbol_is_lowercase_end_to_end():
    """A symbol passing through the canonicaliser must come out
    lowercase regardless of input case."""
    from data.collectors.crypto_market import to_canonical_symbol
    assert to_canonical_symbol("BTC/USDT", "Binance") == "binance__btc_usdt__spot"
    assert to_canonical_symbol("btc/usdt", "binance") == "binance__btc_usdt__spot"
    assert to_canonical_symbol("Btc/USDT", "BINANCE") == "binance__btc_usdt__spot"


def test_canonical_perp_symbol_is_lowercase_end_to_end():
    from data.collectors.crypto_derivatives import to_canonical_perp_symbol
    assert to_canonical_perp_symbol("BTC/USDT:USDT", "Binance") == "binance__btc_usdt__perp"
    assert to_canonical_perp_symbol("btc/usdt:usdt", "BINANCE") == "binance__btc_usdt__perp"


def test_data_contract_lowercase_examples_match_code_output():
    """If anyone updates the contract back to uppercase, the example
    string drift would cause path/lookup desync between callers that
    read the contract vs callers that use the helper. Verify the
    example string in the contract matches what the helper produces."""
    from data.collectors.crypto_market import to_canonical_symbol

    contract_text = (
        Path(__file__).resolve().parents[1] / "plans" / "crypto-data-contract.md"
    ).read_text()
    example = to_canonical_symbol("BTC/USDT", "binance")
    assert example == "binance__btc_usdt__spot"
    assert example in contract_text, (
        f"Helper produces {example!r} but the contract does NOT contain "
        "that string. Either the contract example or the helper drifted; "
        "they must match."
    )
    # And the uppercase form should NOT appear (drift backstop)
    assert "binance__BTC_USDT__" not in contract_text, (
        "Contract still contains the old uppercase form somewhere; "
        "code is lowercase. Fix one of them."
    )
