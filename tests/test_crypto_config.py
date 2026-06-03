"""Phase Crypto-A step 1 — tests for crypto config modules.

Pin three contracts:

  1. config.crypto_storage refuses the in-repo sentinel path AND
     raises CryptoStorageNotMountedError when the real volume is
     unavailable (or when CRYPTO_STORAGE_ROOT points somewhere that
     doesn't exist).

  2. config.crypto_network.assert_proxy_active() raises
     CryptoProxyNotActiveError when the wrapper env vars are missing
     or wrong, and returns cleanly when both are set correctly.

  3. config.crypto_universe constants match the data-contract §10
     numbers and the helper functions produce the venue-native
     ticker strings expected by the collector layer.

These tests perform NO network I/O and NO mounting. They monkeypatch
env vars and tmp_path filesystem to exercise both happy and refusal
paths.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# -----------------------------------------------------------------------------
# crypto_storage
# -----------------------------------------------------------------------------

def test_crypto_storage_raises_when_root_missing(monkeypatch, tmp_path):
    """Pointing CRYPTO_STORAGE_ROOT at a non-existent path must raise
    CryptoStorageNotMountedError immediately. No silent fallback."""
    from config import crypto_storage as cs

    monkeypatch.setenv("CRYPTO_STORAGE_ROOT",
                        str(tmp_path / "no_such_volume"))

    with pytest.raises(cs.CryptoStorageNotMountedError) as exc_info:
        cs.crypto_root()
    assert "not mounted" in str(exc_info.value).lower(), (
        "error message should explain the volume is not mounted"
    )


def test_crypto_storage_returns_path_when_present(monkeypatch, tmp_path):
    """When the override path exists and is writable, crypto_root
    returns it. Tests use tmp_path as the stand-in for
    /Volumes/DATA/crypto."""
    from config import crypto_storage as cs

    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    root = cs.crypto_root()
    assert root == tmp_path


def test_crypto_storage_refuses_in_repo_sentinel(monkeypatch):
    """Even if CRYPTO_STORAGE_ROOT points at the in-repo sentinel, the
    resolver refuses. This is the policy-vs-mistake guard: silent
    repo-tree writes are exactly what the contract forbids."""
    from config import crypto_storage as cs

    repo_sentinel = Path(__file__).resolve().parents[1] / "data" / "storage" / "crypto"
    # Sentinel directory exists (we created the README), so the failure
    # path here is the "refuses repo-tree" branch, not the "doesn't
    # exist" branch.
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(repo_sentinel))
    with pytest.raises(cs.CryptoStorageNotMountedError) as exc:
        cs.crypto_root()
    assert "in-repo sentinel" in str(exc.value).lower()


def test_crypto_storage_raises_when_root_is_a_file(monkeypatch, tmp_path):
    """If the override resolves to a file (rare but possible), raise
    clearly — a file is not a usable storage root."""
    from config import crypto_storage as cs

    bad = tmp_path / "this_is_a_file"
    bad.write_text("x")
    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(bad))
    with pytest.raises(cs.CryptoStorageNotMountedError) as exc:
        cs.crypto_root()
    assert "not a directory" in str(exc.value).lower()


def test_required_subdirs_lists_contract_paths(monkeypatch, tmp_path):
    """The required-subdir list matches the data-contract §1 listing."""
    from config import crypto_storage as cs

    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    subs = cs.required_subdirs()
    names = {p.relative_to(tmp_path).as_posix() for p in subs}
    expected = {
        "raw/ohlcv", "raw/funding", "raw/open_interest",
        "features", "predictions", "health", "audit",
        "paper", "reports/daily",
    }
    assert names == expected, (
        f"required_subdirs drift from data-contract §1: got {names}, "
        f"expected {expected}"
    )


def test_is_inside_crypto_root_only_for_root_paths(monkeypatch, tmp_path):
    from config import crypto_storage as cs

    monkeypatch.setenv("CRYPTO_STORAGE_ROOT", str(tmp_path))
    inside = tmp_path / "raw" / "ohlcv" / "BTC.parquet"
    inside.parent.mkdir(parents=True)
    inside.touch()
    assert cs.is_inside_crypto_root(inside) is True
    assert cs.is_inside_crypto_root(Path("/tmp/elsewhere")) is False


# -----------------------------------------------------------------------------
# crypto_network
# -----------------------------------------------------------------------------

def test_assert_proxy_active_raises_when_both_envs_missing(monkeypatch):
    from config import crypto_network as cn

    monkeypatch.delenv(cn.CRYPTO_NETWORK_ENV, raising=False)
    monkeypatch.delenv(cn.CRYPTO_SSPROXY_ENV, raising=False)
    with pytest.raises(cn.CryptoProxyNotActiveError) as exc:
        cn.assert_proxy_active()
    assert cn.CRYPTO_NETWORK_ENV in str(exc.value), (
        "error must name the missing env var so debugging is fast"
    )


def test_assert_proxy_active_raises_when_network_is_wrong_profile(monkeypatch):
    """Wrapper invoked with `--network domestic` (A-share) but caller
    is a crypto collector — must refuse."""
    from config import crypto_network as cn

    monkeypatch.setenv(cn.CRYPTO_NETWORK_ENV, "domestic")
    monkeypatch.setenv(cn.CRYPTO_SSPROXY_ENV, "yes")
    with pytest.raises(cn.CryptoProxyNotActiveError) as exc:
        cn.assert_proxy_active()
    assert "crypto" in str(exc.value).lower()


def test_assert_proxy_active_raises_when_ssproxy_marker_missing(monkeypatch):
    """Wrapper started but did not confirm ssproxy pre-flight — refuse."""
    from config import crypto_network as cn

    monkeypatch.setenv(cn.CRYPTO_NETWORK_ENV, "crypto")
    monkeypatch.delenv(cn.CRYPTO_SSPROXY_ENV, raising=False)
    with pytest.raises(cn.CryptoProxyNotActiveError) as exc:
        cn.assert_proxy_active()
    assert cn.CRYPTO_SSPROXY_ENV in str(exc.value)


def test_assert_proxy_active_passes_when_both_envs_set(monkeypatch):
    """Happy path: both wrapper sentinels are present and correct."""
    from config import crypto_network as cn

    monkeypatch.setenv(cn.CRYPTO_NETWORK_ENV, "crypto")
    monkeypatch.setenv(cn.CRYPTO_SSPROXY_ENV, "verified")
    cn.assert_proxy_active()  # must not raise


def test_proxy_is_active_returns_bool_not_raise(monkeypatch):
    from config import crypto_network as cn

    monkeypatch.delenv(cn.CRYPTO_NETWORK_ENV, raising=False)
    monkeypatch.delenv(cn.CRYPTO_SSPROXY_ENV, raising=False)
    assert cn.proxy_is_active() is False

    monkeypatch.setenv(cn.CRYPTO_NETWORK_ENV, "crypto")
    monkeypatch.setenv(cn.CRYPTO_SSPROXY_ENV, "ok")
    assert cn.proxy_is_active() is True


# -----------------------------------------------------------------------------
# crypto_universe
# -----------------------------------------------------------------------------

def test_universe_matches_contract_section_10():
    """The constants must equal the contract numbers. If the data
    contract is updated, both should change together."""
    from config import crypto_universe as cu

    assert cu.PRIMARY_EXCHANGE == "binance"
    assert cu.FALLBACK_EXCHANGES == ("okx", "bybit")
    assert cu.PHASE_A_SPOT_BASES == ("BTC", "ETH", "SOL", "BNB", "XRP")
    assert cu.PHASE_A_PERP_BASES == ("BTC", "ETH", "SOL")
    assert cu.PHASE_A_TIMEFRAMES == ("1h", "4h", "1d")
    assert cu.QUOTE_CURRENCY == "USDT"


def test_spot_symbols_default_to_usdt():
    from config import crypto_universe as cu

    assert cu.spot_symbols() == ["BTCUSDT", "ETHUSDT", "SOLUSDT",
                                   "BNBUSDT", "XRPUSDT"]


def test_perp_symbols_default_to_usdt():
    from config import crypto_universe as cu

    assert cu.perp_symbols() == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_spot_symbols_with_custom_quote():
    from config import crypto_universe as cu

    assert cu.spot_symbols("USDC") == [
        "BTCUSDC", "ETHUSDC", "SOLUSDC", "BNBUSDC", "XRPUSDC",
    ]


def test_backfill_depth_includes_all_timeframes():
    from config import crypto_universe as cu

    for tf in cu.PHASE_A_TIMEFRAMES:
        assert tf in cu.BACKFILL_DEPTH_DAYS_BY_TF, (
            f"timeframe {tf} missing from BACKFILL_DEPTH_DAYS_BY_TF"
        )
        assert cu.BACKFILL_DEPTH_DAYS_BY_TF[tf] > 0


def test_expansion_prereqs_match_spec_count():
    """7 prerequisites per spec §4.2. If the spec is updated, both
    should change together."""
    from config import crypto_universe as cu

    assert len(cu.PHASE_B_EXPANSION_PREREQS) == 7


# -----------------------------------------------------------------------------
# Import hygiene
# -----------------------------------------------------------------------------

def test_config_modules_import_without_io():
    """Importing any of the 3 modules must NOT touch network or disk.
    Re-import on a fresh interpreter would be the cleanest probe, but
    we can at least check that the modules don't define a network
    call at module top-level by inspecting their source."""
    import config.crypto_storage as cs
    import config.crypto_network as cn
    import config.crypto_universe as cu

    for mod in (cs, cn, cu):
        src = Path(mod.__file__).read_text()
        # No top-level requests / urllib / ccxt / open() at module scope.
        for tok in ("requests.", "urllib.", "ccxt.", "socket."):
            # Allow appearance INSIDE a function body but not as a
            # top-level statement. Cheap heuristic: scan for the token
            # not preceded by 4 spaces / tab (i.e. not indented).
            for line in src.splitlines():
                if not line.lstrip().startswith(tok):
                    continue
                # Indented → fine
                if line.startswith((" ", "\t")):
                    continue
                pytest.fail(
                    f"{mod.__name__} appears to use {tok} at module "
                    f"top-level: {line!r}"
                )
