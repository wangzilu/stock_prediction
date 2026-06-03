"""Crypto network policy (ssproxy MANDATORY).

Per `plans/crypto-data-contract.md` §1.5: every crypto data fetch
must traverse `ssproxy`. The cron wrapper
(`scripts/run_network_job.py --network crypto`) sets a sentinel env
var; collector entrypoints call `assert_proxy_active()` to verify
the wrapper actually ran.

The module is import-cheap: no network probe, no I/O at import time.
Activation of the proxy itself is the wrapper's job — this module
only verifies the wrapper completed.

Forbidden patterns (enforced via lint in `scripts/check_namespace_isolation.py`):
- `requests.get(... exchange URL ...)` directly inside any crypto module
- `ccxt.binance().fetch_*` outside the wrapper's env-check
- Reading from a network library at import time (must be lazy)

A-share modules are NOT touched by this layer and continue using
`--network domestic`.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Env var the cron wrapper sets when --network crypto is active. The
# exact value is chosen by the wrapper; collectors only check for
# presence (truthy).
CRYPTO_NETWORK_ENV = "CRYPTO_NETWORK_ACTIVE"

# Secondary safety: an explicit "ssproxy active" marker the wrapper
# emits after verifying the SOCKS / HTTP proxy responded to a
# pre-flight probe.
CRYPTO_SSPROXY_ENV = "CRYPTO_SSPROXY_VERIFIED"


class CryptoProxyNotActiveError(RuntimeError):
    """Raised by collector entrypoints when the cron wrapper did not
    set the network sentinel — meaning the proxy is NOT confirmed
    active and the collector MUST refuse to proceed."""


def assert_proxy_active() -> None:
    """Verify the cron wrapper successfully activated ssproxy.

    Called as the FIRST action in every crypto collector entrypoint.
    Raises `CryptoProxyNotActiveError` if either:
      1. `CRYPTO_NETWORK_ENV` is unset / empty (the wrapper never ran)
      2. `CRYPTO_SSPROXY_ENV` is unset / empty (the wrapper ran but
         the ssproxy pre-flight probe failed)

    There is no silent fallback. A-share isolation requires that
    crypto collectors crash loud rather than fall back to direct
    egress (which mainland network would block or route through
    unexpected paths).
    """
    network = os.environ.get(CRYPTO_NETWORK_ENV, "").strip()
    if not network:
        raise CryptoProxyNotActiveError(
            f"{CRYPTO_NETWORK_ENV} is not set. Crypto collectors must "
            "be invoked via `run_network_job.py --network crypto` so "
            "the wrapper activates ssproxy first. Direct invocation "
            "from mainland network is forbidden — see "
            "plans/crypto-data-contract.md §1.5."
        )
    if network != "crypto":
        raise CryptoProxyNotActiveError(
            f"{CRYPTO_NETWORK_ENV}={network!r} is not 'crypto'. The "
            "wrapper appears to be running in a non-crypto profile; "
            "crypto collectors must NOT proceed in that case."
        )

    ssproxy = os.environ.get(CRYPTO_SSPROXY_ENV, "").strip()
    if not ssproxy:
        raise CryptoProxyNotActiveError(
            f"{CRYPTO_SSPROXY_ENV} is not set. The cron wrapper started "
            "with the crypto profile but did not confirm ssproxy is "
            "responding. This means a pre-flight probe failed and the "
            "collector MUST NOT attempt the fetch."
        )

    # Optional debug breadcrumb — useful when audit logs need to show
    # that the assertion was reached and passed.
    logger.debug(
        "crypto network policy: profile=%s ssproxy=verified", network,
    )


def proxy_is_active() -> bool:
    """Non-raising variant for places that want a bool. Returns True
    iff the wrapper sentinels are both present. Useful for tests and
    for module-level guards that prefer to skip rather than crash."""
    try:
        assert_proxy_active()
    except CryptoProxyNotActiveError:
        return False
    return True
