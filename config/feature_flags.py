"""Feature flags for runtime gating of legacy / experimental code paths.

Per `plans/cc-crypto-implementation-spec-2026-05-30.md` §6.5 (Legacy
Crypto Quarantine). User direction 2026-05-30: A-share daily pipeline
must not touch legacy crypto code paths by default.
"""

import os


def _flag_truthy(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() in ("true", "1", "yes")


# Legacy crypto context (BTC/ETH evening-report background, BTC/ETH
# candidates in morning recommendation, crypto market dispatcher).
#
# Default FALSE per user direction 2026-05-30. Goal: keep A-share runtime
# clean of legacy crypto. Set TRUE manually only if the old BTC/ETH
# evening-report background is needed for a specific run.
LEGACY_MARKET_CONTEXT_ENABLED = _flag_truthy(
    "LEGACY_MARKET_CONTEXT_ENABLED", "false"
)
