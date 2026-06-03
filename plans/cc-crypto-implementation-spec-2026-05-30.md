# CC Crypto Implementation Spec (Pre-Phase-0 Sign-off)

Date: 2026-05-30

Predecessor documents (all in `plans/`):

- `cc-crypto-quant-integration-plan-2026-05-30.md` (original CC plan)
- `crypto-quant-roadmap-2026-05-30.md` (CX roadmap, revised)
- `crypto-quant-literature-and-engineering-review-2026-05-30.md` (CX lit review, revised)
- `cx-review-cc-crypto-quant-integration-plan-2026-05-30.md` (CX review of CC + Addendum)
- `cc-review-cx-crypto-quant-docs-2026-05-30.md` (CC review of CX + convergence record)

This document is the implementation contract before Phase 0 starts.

## §−1. User Operational Constraint (HARDEST GATE)

User instruction, 2026-05-30, before Phase 0 sign-off:

> "先不实盘先不加杠杆，每天虚拟盘，把结果给我，我需要先仔细观察一段时间"

This constraint is **above** every phase acceptance gate, every CX
review, every CC review, and every numeric evidence threshold in this
spec. It is enforced as follows:

### Hard Rules

1. **No live exchange keys at any phase.** Phase D paper OMS, Phase E
   funding-arb paper, Phase G Nautilus testnet — all configured with
   `LIVE_TRADING=False` as a hard-coded default that cannot be
   overridden without user written sign-off.
2. **No leverage anywhere, including paper.** All paper OMS configs
   set `max_leverage=1.0`. Perp funding-arb simulation uses
   `margin=1.0` (cash-equivalent simulation). No margin call paths
   are exercised in paper.
3. **Daily paper trading report is the operational deliverable.**
   Every day produces a report file the user reads. Format specified
   in §11.
4. **Observation window is open-ended and user-decided.** No
   automatic promotion from paper to testnet to live based on
   Sharpe/IC/days-elapsed. User explicitly says when to advance.

### What Changes In This Spec

- All occurrences of "live capital" / "real money" / "promotion gate"
  / "30-day paper before promotion" are replaced or struck through.
- "Phase Crypto-D" reframes from "transitional to live" → "operational
  paper trading with daily reports, indefinite duration".
- "Phase Crypto-G" (Nautilus + physical migration) — the Nautilus
  live/testnet portion is gated to user written sign-off, separately
  from the namespace migration portion.
- "Phase Crypto-H" (RL) — paper-only evaluation; RL never controls
  real capital under this constraint.

### What This Does NOT Change

- The technical implementation of all phases stays the same: paper
  OMS still simulates fills, fees, slippage, funding cost. Backtest
  still validates IC/Sharpe. CryptoSanitizer + CryptoRiskGuard still
  enforce data hygiene. This constraint changes the *output channel*
  (paper report → user), not the *correctness machinery*.
- §14.1 audit and `core/` Protocol work still proceed. They are
  asset-class-neutral infrastructure unaffected by paper-only mode.

### Why This Constraint Is The Right Default

- Crypto failure modes (exchange downtime / stablecoin depeg /
  liquidation cascade / withdrawal halt / venue ban / API outage) are
  more numerous and faster than A-share failure modes. Observation
  period must be empirical, not date-based.
- A-share side of this project already validated the discipline:
  paper → user observation → live, with user as gating authority.
  Crypto inherits this with stricter enforcement because the failure
  surface is larger.
- Per CX Addendum Final Position: "Do not trade real leverage before
  local validation." This user constraint extends "before local
  validation" to "before user explicit sign-off" — strictly stronger
  than the CX rule.

### Enforcement Mechanism

- `paper/crypto_oms.py` line 1 has:
  ```python
  LIVE_TRADING_ALLOWED = False  # User constraint 2026-05-30. Do not
                                # change without explicit user sign-off.
  MAX_LEVERAGE = 1.0            # User constraint 2026-05-30.
  ```
- Any code path attempting `if LIVE_TRADING_ALLOWED:` is rejected at
  PR review.
- Daily report cron job (§11) is a Phase D acceptance hard
  requirement.

## §−0.4. Storage: All Crypto Data on /Volumes/DATA (FROZEN)

User instruction, 2026-05-30:

> "和加密货币相关的数据，都拉取在/Volume/DATA上，建一个文件夹"

Verified `/Volumes/DATA` (note plural per macOS convention) is mounted
as a 1.9 TB external volume, 1.6 TB free, already hosts `database/`
for other project data.

### Storage Root (FROZEN)

```
CRYPTO_STORAGE_ROOT = Path("/Volumes/DATA/crypto")
```

This replaces the earlier r1-r3 references to
`data/storage/crypto/**` (which would have written to the repo
working tree). All crypto parquet/jsonl/json under this root.

Layout (replaces §1.1 crypto storage layout):

```
/Volumes/DATA/crypto/
  raw/
    ohlcv/{exchange}/{symbol_canonical}/{timeframe}/year={Y}/month={M}/day={D}.parquet
    funding/{exchange}/{symbol_canonical}/year={Y}/month={M}.parquet
    open_interest/{exchange}/{symbol_canonical}/year={Y}/month={M}.parquet
  features/
    crypto_feature_cache.parquet
  predictions/
    crypto_predictions_latest.json
  health/
    crypto_data_health.json
    incidents.md
  audit/
    delisted_coins.parquet
  paper/
    crypto_oms_state.json
  reports/
    daily/{YYYY-MM-DD}_paper_report.md
  phase0_measurements.json
```

### Single Source of Truth

Add to `config/crypto_storage.py` (Phase 0a deliverable):

```python
"""Crypto storage root. Mounted external volume per user direction
2026-05-30. All crypto pipeline code MUST import CRYPTO_STORAGE_ROOT
from here, never hardcode the path."""
import os
from pathlib import Path

# Production default: /Volumes/DATA/crypto on the Mac Studio.
# Override via env for CI tests, alternate machines, or volume rename:
#     CRYPTO_STORAGE_ROOT=/tmp/crypto_test pytest ...
CRYPTO_STORAGE_ROOT = Path(
    os.environ.get("CRYPTO_STORAGE_ROOT", "/Volumes/DATA/crypto")
)

REQUIRED_SUBDIRS = [
    "raw/ohlcv", "raw/funding", "raw/open_interest",
    "features", "predictions", "health", "audit", "paper",
    "reports/daily",
]


def ensure_mounted_and_writable() -> None:
    """Hard-fail if /Volumes/DATA is not mounted or not writable.

    Crypto cron/scripts MUST call this before any write. Silent
    fallback to repo path is forbidden — if the external drive is
    not available, the job must fail, not write to the wrong place.
    """
    root = CRYPTO_STORAGE_ROOT
    if not root.parent.exists():
        raise RuntimeError(
            f"External volume {root.parent} not mounted. "
            f"Eject/re-mount /Volumes/DATA before running crypto jobs."
        )
    if not root.exists():
        # First-time setup: create the crypto/ subdir + REQUIRED_SUBDIRS.
        # Fail loudly if we can't (permission, FS error, etc.).
        try:
            root.mkdir(parents=True, exist_ok=True)
            for sub in REQUIRED_SUBDIRS:
                (root / sub).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(
                f"Cannot create {root} on external volume: {e}"
            ) from e
    # Quick writability probe
    probe = root / ".writable_probe"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        raise RuntimeError(
            f"{root} not writable: {e}. Check volume mount + permissions."
        ) from e
```

Every crypto entrypoint script (`crypto_update_market_data.py`,
`crypto_update_derivatives.py`, `crypto_data_health.py`,
`crypto_daily_report.py`, `crypto_phase0_spike.py`) calls
`ensure_mounted_and_writable()` as its first action after argparse.

### Cron Pre-flight (FROZEN)

`run_network_job.py` is extended OR a thin pre-flight wrapper added
so crypto jobs check mount BEFORE the proxy lifecycle:

```python
# scripts/run_crypto_job.py (NEW)
"""Pre-flight wrapper for crypto jobs.

Sequence: check /Volumes/DATA mount → start ssproxy →
exec wrapped command.

If mount fails, fail before bringing up ssproxy. If proxy fails,
fail before exec. Both failures land in run_with_status reporting
so health monitoring catches them.
"""
import sys
from config.crypto_storage import ensure_mounted_and_writable

def main():
    ensure_mounted_and_writable()
    # Delegate to run_network_job.py for proxy + env + exec
    from scripts.run_network_job import main as run_net
    sys.exit(run_net())
```

Cron entries (per §2.6) use this wrapper:

```python
CronJob(
    name="crypto_ohlcv_1h",
    schedule="5 * * * *",
    command=(
        "scripts/run_crypto_job.py --network crypto_global -- "
        "python scripts/crypto_update_market_data.py --timeframe 1h"
    ),
    timeout_sec=600,
)
```

### Safety Properties

- **Mount disappearance is loud, not silent**: if `/Volumes/DATA`
  unmounts mid-run (e.g. cable disconnect), the next `mkdir` /
  `write_parquet` raises `OSError`; the script does not fall back to
  the repo path. Cron job fails, monitoring catches it.
- **No code writes to repo `data/storage/crypto/`**: this path is
  forbidden. CI lint rule below enforces.
- **Backup question is deferred**: external drive backup strategy is
  user's responsibility for now; spec does not assume RAID/redundancy
  for `/Volumes/DATA`. Phase 0a `crypto_phase0_spike.py` records the
  drive's free space and SMART status snapshot so a baseline exists.

### CI Lint Addition

`scripts/check_namespace_isolation.py` adds:

```python
("crypto_market.py|crypto_derivatives.py|crypto_.*\\.py",
 "data/storage/crypto",
 "Crypto code must use config.crypto_storage.CRYPTO_STORAGE_ROOT "
 "(/Volumes/DATA/crypto), not the repo path data/storage/crypto"),
```

## §−0.3. Network: All Crypto Traffic Through ssproxy (FROZEN)

User instruction, 2026-05-30:

> "和加密货币相关业务数据拉取的都需要 ssproxy"

All crypto data fetches (OHLCV / funding / OI / event / on-chain) must
route through ssproxy. There is no exception. Direct connection to
exchange APIs is forbidden because (a) it may fail silently in this
network environment, (b) it would bypass the existing proxy
lifecycle / health discipline that A-share global jobs already use.

### Existing Infra (re-use, do not reinvent)

The project already has the right primitives:

- `config/network_profiles.py` (file path: `config/`, **not**
  `scheduler/`):
  ```python
  PROXY_URL = "http://127.0.0.1:10818"   # HTTP proxy
  PROXY_PORT = 10818                     # shadowsocks-http-auto.js
                                         # bridge to SOCKS5:10808
  PROXY_START_CMD = ["zsh", "-ic", "ssproxy"]
  ```
- `scripts/run_network_job.py --network <profile> -- <command>`:
  - For `--network global`: invokes `_ensure_proxy()` (launches
    ssproxy if needed, 10s health wait), sets `HTTP_PROXY` /
    `HTTPS_PROXY` env vars to `PROXY_URL`, then execs the wrapped
    command.
  - Failure to bring proxy up → exit 1 (fail-fast).

### Add `crypto_global` Profile (Phase 0a deliverable)

Add to `config/network_profiles.py`:

```python
# Crypto exchanges (Binance/OKX/Bybit/Kraken) require global network.
# Use the same ssproxy lifecycle as A-share global jobs.
CRYPTO_NETWORK = "global"  # alias — keeps semantics identical to global
```

Add to `scripts/run_network_job.py` `apply_profile()`:

```python
elif profile == "crypto_global":
    return apply_profile(CRYPTO_NETWORK, env, timeout)
```

This is one-line aliasing. `crypto_global` exists as a distinct name
so logs / health files / cron entries clearly mark which jobs are
crypto-network vs which are generic global; if we ever change crypto
routing (e.g. different exchange-specific proxy), only the alias
changes.

### Collector-side Contract (FROZEN)

`CryptoSpotCollector` and `CryptoDerivativesCollector` **never load
proxy config directly**. They assume the wrapper (`run_network_job.py`)
has already set env vars. CCXT respects `HTTP_PROXY` / `HTTPS_PROXY`
automatically.

```python
# data/collectors/crypto_market.py
class CryptoSpotCollector:
    def __init__(self, exchange: str):
        self.exchange = exchange.lower()
        self.client = self._make_client()

    def _make_client(self) -> ccxt.Exchange:
        # Per §−0.3: proxy is set by run_network_job.py via env vars
        # before this script runs. We assert env is configured so a
        # script run outside the wrapper fails loudly instead of
        # silently bypassing ssproxy.
        if not os.environ.get("HTTPS_PROXY"):
            raise RuntimeError(
                "HTTPS_PROXY env not set. Crypto collectors must be "
                "invoked via `run_network_job.py --network crypto_global`."
            )
        ex_cls = getattr(ccxt, self.exchange)
        return ex_cls({
            "enableRateLimit": True,
            "timeout": 30_000,
            # CCXT picks up HTTPS_PROXY automatically; explicit pass
            # below makes it visible in client config for debugging.
            "proxies": {
                "http": os.environ["HTTPS_PROXY"],
                "https": os.environ["HTTPS_PROXY"],
            },
        })
```

The over-engineered `_load_proxy(strict=True)` from r3 is **removed**.
It assumed a `Profile` object model that this project doesn't have
and duplicated work that `run_network_job.py` already does. The env-
assertion replaces it: simpler, fail-loud, aligned with A-share
network-job conventions.

### Cron Wrap (FROZEN — single entry point)

**Every crypto cron entry MUST go through `scripts/run_crypto_job.py`.**
Direct invocation of `run_network_job.py --network crypto_global` is
forbidden in CRYPTO_JOBS, because that path skips the mount check.

`run_crypto_job.py` is the only entrypoint and internally delegates:

```
run_crypto_job.py
  1. ensure_mounted_and_writable() — fail-fast if /Volumes/DATA missing
  2. delegate to run_network_job.py --network crypto_global -- <command>
     (which handles ssproxy lifecycle + env vars)
```

Cron entries:

```python
CronJob(
    name="crypto_ohlcv_1h",
    schedule="5 * * * *",
    command=(
        "scripts/run_crypto_job.py -- "
        "python scripts/crypto_update_market_data.py --timeframe 1h"
    ),
    timeout_sec=600,
)
```

(The `--network crypto_global` is baked into `run_crypto_job.py`, not
passed per-cron — every crypto job needs ssproxy by definition, so
the choice is not configurable per call site.)

### Phase 0a Audit Item

`scripts/crypto_phase0_spike.py` must:

1. Verify `config/network_profiles.py` has `CRYPTO_NETWORK` defined
2. Verify `run_network_job.py` recognizes `crypto_global` profile
3. Run a single `ccxt.binance().fetch_ohlcv('BTC/USDT', '1h', limit=1)`
   call wrapped in `run_network_job.py --network crypto_global --`
4. Record: ssproxy startup time, ccxt round-trip time, response size,
   any 429/timeout/proxy error

If any of these fail, Phase 0a does not proceed to schema work until
the network plumbing is fixed.

**Failure reporting (FROZEN)**: even on failure, the spike script
MUST write `/Volumes/DATA/crypto/phase0_measurements.json` (or, if
mount itself failed, write to repo `data/storage/crypto_spike_failure.json`
as last-resort). The file records `step` (which check failed),
`reason` (free-text), `error_class`, `traceback`, and what was
attempted. Reasons we distinguish:

- `mount_unavailable`: `/Volumes/DATA` not mounted
- `ssproxy_not_started`: PROXY_START_CMD failed
- `proxy_port_unreachable`: 10818 not listening
- `dns_failure`: cannot resolve exchange host
- `tls_handshake_failure`: cert / handshake issue through proxy
- `http_429`: exchange rate-limited
- `ccxt_symbol_invalid`: symbol naming wrong
- `ccxt_other_error`: catch-all with traceback
- `network_profile_missing`: `crypto_global` not registered

Without this, "spike failed" is opaque and the next debugging step is
guessing. The failure report turns the spike into a diagnostic.

## §−0.5. A-share Production Isolation Guarantee (HARDER GATE)

User instruction, 2026-05-30, after the §−1 paper-only constraint:

> "一定要确保解耦，开发过程中的 bug 不能影响 A 股每天的工作进度"

This constraint is enforced through 6 isolation layers. Any of them
failing is a P0 incident; collectively they guarantee A-share daily
cron / paper OMS / recommendations / reports continue uninterrupted
regardless of crypto code state.

### Layer 1: Import Isolation

- `core/` exports only. Forbidden: `from ashare...` or `from crypto...`
  inside any file under `core/`.
- `ashare/` forbidden: `from crypto...`, `from data.collectors.crypto`
  (legacy)
- `crypto/` (new namespace) forbidden: `from ashare...`,
  `from data.collectors.crypto` (legacy), `from scheduler.jobs`,
  `from config.watchlist`
- `scheduler/jobs.py` forbidden to have a module-level
  `from data.collectors.crypto import` — must be lazy per §6.5
- CI lint enforces these rules via `scripts/check_namespace_isolation.py`:

```python
# scripts/check_namespace_isolation.py
"""Fails CI if forbidden cross-namespace imports are introduced."""
RULES = [
    ("core/", "ashare\\.", "core may not depend on ashare"),
    ("core/", "crypto\\.", "core may not depend on crypto"),
    ("ashare/", "crypto\\.", "ashare may not depend on crypto"),
    ("ashare/", "data\\.collectors\\.crypto", "ashare may not import legacy crypto collector"),
    ("crypto/", "ashare\\.", "crypto may not depend on ashare"),
    ("crypto/", "data\\.collectors\\.crypto", "new crypto code may not import legacy crypto collector"),
    ("crypto/", "scheduler\\.jobs", "new crypto code may not import scheduler.jobs"),
    ("crypto/", "config\\.watchlist", "new crypto code may not import config.watchlist (legacy market enum)"),
    # AST-level rule: scheduler/jobs.py has no module-level legacy import
    AST_RULE("scheduler/jobs.py",
             forbidden_module_imports=["data.collectors.crypto"],
             reason="legacy crypto must be lazy-imported per §6.5"),
]
```

### Layer 2: Storage Isolation

- **Crypto storage root**: `/Volumes/DATA/crypto/**` (external volume,
  per §−0.4). Single source of truth is
  `config.crypto_storage.CRYPTO_STORAGE_ROOT`.
- **A-share storage paths**: in-repo `data/storage/factor_store/`,
  `data/storage/predictions/`, `data/storage/recommendations/`,
  `data/storage/llm_events_v2/`, etc.
- No shared parquet / jsonl / json file between the two
- Crypto NEVER reads from or writes to A-share paths; A-share NEVER
  reads from or writes to crypto paths
- **Repo path `data/storage/crypto/` is FORBIDDEN** — CI lint rejects
  any code referencing it (see §−0.4 lint rule)
- Crypto entrypoints must call
  `config.crypto_storage.ensure_mounted_and_writable()` before any
  write; silent fallback to alternative storage is forbidden

### Layer 3: Cron Isolation

- Crypto cron jobs in `install_crontab.py` placed in dedicated group
  `CRYPTO_JOBS = [...]`, separate from existing A-share groups
- Crypto `enforce_deps` never references A-share job names
- A-share `enforce_deps` never references crypto job names
- Crypto cron failure does NOT raise A-share `mark_blocked`
- A-share cron failure does NOT block crypto crons (independent failure
  domains)
- Single-command kill switch:

```bash
python scripts/disable_crypto_cron.py
# Removes ONLY crypto cron entries from crontab.
# Leaves A-share entries untouched.
# Idempotent.
```

### Layer 4: CI Gate

Every PR touching crypto code must pass:

- [ ] `pytest tests/test_ashare_unchanged_after_crypto_split.py` —
      §14.1 byte-identical regression tests (added incrementally per
      §1.3)
- [ ] `python scripts/check_namespace_isolation.py` — import lint
- [ ] `pytest tests/test_ashare_smoke.py` — A-share daily-cron
      simulation on a fixed fixture date, must produce byte-identical
      recommendations vs main branch
- [ ] PR description includes: "A-share regression check: [PASS / FAIL]"

Crypto PRs that fail any of the above are not mergeable. The CI gate is
mandatory, not advisory.

### Layer 5: Staged Deployment

§14.1 audit refactor lands in 3 staged PRs (per cx review):

- **PR1**: Create `core/` Protocols + A-share wrapper. **Zero behavior
  change**. A-share still uses the old code path; new wrapper is
  unused. Gates: 3 consecutive days of A-share cron GREEN after deploy.
- **PR2**: Wire `ashare/` to use `core/` Protocols via adapter.
  Behavior preserving — same recommendations output. Gates: byte-
  identical A-share recommendations vs PR1 baseline for 3 days.
- **PR3**: Add `crypto/` greenfield code using `core/` Protocols. No
  A-share path touched. Gates: A-share cron continues GREEN.

Each PR must wait the 3-day GREEN gate before the next is merged.
**No bundling of PR1+PR2 or PR2+PR3 is allowed.**

### Layer 6: Monitoring + Manual Rollback

First 14 days after each PR deploy:

- A-share health monitored every 2 hours (existing
  `scripts/check_health.py` extended)
- A-share **business output** compared byte-identical to pre-deploy
  baseline. Per cx review r3 #5: **never compare file mtime** —
  production files regenerate daily so mtime always differs and
  produces false positives.
- "Business output" means normalized hashes of:
  - `recommendations` list (sorted by code, fields normalized)
  - `paper_oms_state.json` positions + pending_target_weights
  - `daily_pnl.json` realized + unrealized PnL
  - cron `mark_complete` success/failure pattern
- Excluded from comparison: file mtime, log files, ingestion timestamps,
  intermediate parquet ingested_at columns
- Any business-output divergence OR cron failure raises a P0 alert

Rollback procedure if A-share degradation detected:

```bash
# Step 1: stop the bleeding
python scripts/disable_crypto_cron.py

# Step 2: feature-flag the refactor off
export ENABLE_CORE_REFACTOR=false
# (or edit config/feature_flags.py)

# Step 3: investigate logs, do not panic-revert
tail -f data/logs/ashare/*.log

# Step 4: if confirmed crypto-caused regression, git revert the
# offending commit (creates new commit, preserves history,
# does NOT clobber parallel session work)
git log --oneline -10
git revert <sha-of-bad-commit>
git push
```

**Explicit prohibition**: Auto-rollback via `git reset --hard` is
forbidden. It can destroy uncommitted user work and parallel session
state. All rollback paths use `git revert` (new commit, reversible)
or feature flags.

### Why These 6 Layers

- A-share has ~35 active cron jobs / 4 daily shadow overlays / paper
  OMS / user-visible recommendations
- Past audits (`project_audit_full_review_20260529`) showed even
  small refactors can introduce silent bugs (ST filter / OMS pending
  / news collection); crypto is a larger surface
- The user has explicitly stated A-share daily work must not be
  affected; this elevates "best-effort isolation" to "guaranteed
  isolation"
- Without these layers, the failure mode is: crypto bug → shared
  module → A-share recommendation breaks silently → user notices
  hours/days later → trust eroded

### Enforcement Mechanism

- Layers 1-2: CI lint (automated, runs on every commit)
- Layer 3: cron install script structural assertion
- Layer 4: pre-merge required checks
- Layer 5: per-PR deployment manual gate (3-day waiting period)
- Layer 6: extended monitoring window + manual rollback only

If any layer fails, the offending PR / deploy / commit is rolled back
before crypto work resumes. A-share takes absolute priority.

## §0. Honest Scope Statement

This spec commits to two levels of detail:

- **DETAIL-FROZEN**: Phase Crypto-0 and Phase Crypto-A. File paths,
  function signatures, schema columns, test names, cron entries.
  Every item below labeled FROZEN is a commitment.
- **DETAIL-STRUCTURAL**: Phase Crypto-B and Phase Crypto-C. Algorithm
  sketches, interfaces, acceptance criteria. Sign-off here is for
  shape, not for every line.
- **DETAIL-ACCEPTANCE-ONLY**: Phase Crypto-D through Crypto-H.
  Acceptance gates + architectural sketch. Each phase gets its own
  design doc before that phase starts.

**This document is for sign-off on Phase 0 + A. Phases B-H are sketched
to prove the eventual roadmap is coherent, not to lock implementation.**

What I do *not* yet know with certainty (DETAIL-DEFERRED, must be
measured during Phase 0 spike):

- Exact Binance/OKX/Bybit rate limits in the project's proxy config
- Fee tiers at 0-100k USDT / 100k-1M / 1M+ volume per exchange
- Actual historical depth available via CCXT fetch_ohlcv vs Tardis vs
  direct exchange archives (some exchanges throttle deep history)
- Initial 5-coin USDT dollar-volume thresholds for "liquid major"
  (need 30-day rolling read)
- Whether `data/collectors/crypto.py` AKShare fallback still works
  (last touched 2026-05-15)
- Whether `scheduler/network_profiles.py` has a `global` or
  `crypto_global` profile, or if one must be added

Items in this category are gated to a **Phase 0 measurement spike**
(see §1.7). They are not blockers for sign-off on this spec; they are
blockers for Phase 0 acceptance.

## §1. Phase Crypto-0: Data Contract + Architecture Interfaces

Per cx review of this spec (2026-05-30), Phase 0 is **split into
three sub-phases** to keep A-share production isolated:

- **Phase 0a**: documentation, schemas, measurement spike, numeric
  audit. **Zero code change to A-share or `core/`.** All read-only
  / document work.
- **Phase 0b**: build `core/` Protocol + minimal A-share wrapper.
  **A-share production code paths NOT modified.** New wrapper exists
  but is unused by ashare/. Greenfield only.
- **Phase 0c**: per-§14.1-row adapter wiring + regression tests. Each
  row gets a regression test FIRST, then an adapter in `ashare/` that
  delegates through `core/` Protocol. The original production code
  path (`paper/oms.py`, `scheduler/jobs.py`, `optimizer_v2.py`, etc.)
  is **not physically moved** — moves are deferred to Phase G.

Duration: **0a 3-5 days / 0b 3-5 days / 0c 1-2 weeks** (was: "1-2
weeks total" or cx-roadmap's "1-2 days").

Why this split: per cx review, batching Phase 0 risks turning crypto
startup into A-share refactor. By gating sub-phases and **forbidding
file moves until Phase G**, A-share production paths remain
operationally untouched throughout Phase 0.

### 1.1 Directory Layout (FROZEN, Phase 0b)

```
core/                                    # NEW
  __init__.py
  asset.py                               # AssetClass, InstrumentClass enums
  instrument.py                          # Symbol, Instrument dataclasses
  calendar.py                            # TradingCalendar Protocol + Always24x7Calendar
  settlement.py                          # SettlementModel Protocol + T1/Instant impls
  cost.py                                # CommissionModel/TaxModel/ImpactModel Protocols
  data_availability.py                   # MarketDataSource Protocol
  universe.py                            # UniverseFilter Protocol

data/collectors/
  crypto_market.py                       # NEW: CCXT spot OHLCV
  crypto_derivatives.py                  # NEW: funding/OI
  crypto.py                              # EXISTING: legacy single-symbol fetcher (kept for compat)

# Crypto data lives on external volume per §−0.4, NOT in repo.
# Path: /Volumes/DATA/crypto/  (see config/crypto_storage.py for SSOT)
#   raw/ohlcv/...
#   raw/funding/...
#   raw/open_interest/...
#   features/crypto_feature_cache.parquet
#   predictions/crypto_predictions_latest.json
#   health/crypto_data_health.json
#   audit/delisted_coins.parquet
#   paper/crypto_oms_state.json
#   reports/daily/{YYYY-MM-DD}_paper_report.md
#
# Repo tree does NOT contain data/storage/crypto/. CI lint forbids it.

scripts/
  crypto_phase0_spike.py                 # NEW: measurement spike (§1.7)
  crypto_update_market_data.py           # NEW: Phase A entrypoint
  crypto_update_derivatives.py           # NEW: Phase A
  crypto_data_health.py                  # NEW: Phase A
  crypto_build_features.py               # Phase B (sketch)
  crypto_train_model.py                  # Phase C (sketch)
  crypto_predict.py                      # Phase C (sketch)

config/
  crypto_universe.py                     # NEW: 5-coin Phase A universe
  crypto_exchanges.py                    # NEW: exchange config + fee tiers

paper/
  crypto_oms.py                          # Phase D (sketch)

risk/
  crypto_risk_guard.py                   # Phase E (sketch)

plans/
  crypto-data-contract.md                # NEW: Phase 0 deliverable, schema spec
```

### 1.2 Schema Doc Spec (FROZEN, Phase 0a)

File: `plans/crypto-data-contract.md`

Spot OHLCV bar schema:

```
Column          | Type      | Nullable | Description
----------------|-----------|----------|-------------------------------------------
timestamp_utc   | int64     | NO       | Bar OPEN timestamp, millisec since epoch UTC
exchange        | string    | NO       | Lowercase: "binance" / "okx" / "bybit"
symbol          | string    | NO       | CCXT canonical: "BTC/USDT" (no venue suffix here)
timeframe       | string    | NO       | "1h" | "4h" | "1d"
open            | float64   | NO       |
high            | float64   | NO       |
low             | float64   | NO       |
close           | float64   | NO       |
volume_base     | float64   | NO       | In base currency (BTC for BTC/USDT)
volume_quote    | float64   | NO       | In quote currency (USDT for BTC/USDT)
quote_volume_estimated | bool | NO     | True if volume_quote is volume_base * mid_price approximation; False if from exchange (per cx r3 #6)
trades          | int32     | YES      | -1 if exchange doesn't report
is_closed_bar   | bool      | NO       | True only if bar_close_ts + CLOSED_BUFFER_SEC ≤ ingestion_ts
ingested_at     | int64     | NO       | Ingestion timestamp UTC ms, for PIT replay
```

Partition: `(exchange, symbol_canonical, timeframe, date)` where
`symbol_canonical = symbol.replace("/", "_")` so file paths are
filesystem-safe.

Funding schema:

```
Column            | Type      | Nullable
------------------|-----------|----------
timestamp_utc     | int64     | NO   (funding event UTC ms)
exchange          | string    | NO
symbol            | string    | NO   (perp symbol, e.g. "BTC/USDT:USDT")
funding_rate      | float64   | NO   (signed, e.g. 0.0001 = 1bp/8h)
next_funding_ts   | int64     | YES
mark_price        | float64   | YES
index_price       | float64   | YES
ingested_at       | int64     | NO
```

Open interest schema (sampled, not event-driven):

```
Column            | Type      | Nullable
------------------|-----------|----------
timestamp_utc     | int64     | NO   (sample ts UTC ms, aligned to 15-min grid)
exchange          | string    | NO
symbol            | string    | NO
open_interest     | float64   | NO   (in base currency)
oi_quote          | float64   | YES  (in quote currency if reported)
long_short_ratio  | float64   | YES
ingested_at       | int64     | NO
```

Delisted-coin audit schema (for survivorship):

```
Column            | Type      | Nullable
------------------|-----------|----------
exchange          | string    | NO
symbol            | string    | NO
listed_at_utc     | int64     | YES
delisted_at_utc   | int64     | NO
last_close        | float64   | YES   (last available close before delisting)
reason            | string    | YES   ("exchange_removed" | "scam" | "unknown")
detected_at       | int64     | NO
```

Production rule: **the universe construction at any historical
timestamp T must read both `ohlcv/` and `delisted_coins.parquet` with
`detected_at <= T`**, so backtests do not survivorship-bias.

### 1.3 §14.1 Asset-implicit Audit Version-Locking (FROZEN, Phase 0a + 0c)

The CC plan §14.1 audit (20 file:line rows) gets a header at Phase 0
sign-off:

```
AUDIT-FROZEN-AT: <sha1 of cc-crypto-quant-integration-plan-2026-05-30.md at sign-off>
PHASE: Crypto-0
DATE: <sign-off date>
```

Rule: future edits create §14.2, §14.3 (new sections). Never in-place
rewrite §14.1.

Each of the 20 rows gets a regression smoke test added to
`tests/test_ashare_unchanged_after_crypto_split.py`:

```python
def test_st_filter_unchanged():
    # Snapshot A-share recommendation behavior on a fixed test date
    # before any core/ refactor. Must produce byte-identical output
    # after refactor.
    ...

def test_t1_settlement_unchanged():
    ...

def test_stamp_tax_unchanged():
    ...

# ... 17 more, one per §14.1 row
```

These 20 tests must pass before any Phase 0 work merges to master.

### 1.4 Core Protocols (FROZEN, Phase 0b)

```python
# core/asset.py
from enum import Enum

class AssetClass(str, Enum):
    ASHARE = "ashare"
    CRYPTO = "crypto"

class InstrumentClass(str, Enum):
    SPOT = "spot"
    PERPETUAL = "perp"
    FUTURE = "future"
    OPTION = "option"
```

```python
# core/instrument.py
from dataclasses import dataclass
from .asset import AssetClass, InstrumentClass

@dataclass(frozen=True, slots=True)
class Symbol:
    base: str               # "BTC"
    quote: str              # "USDT"
    venue: str              # "binance" lowercase
    asset_class: AssetClass
    instrument_class: InstrumentClass

    def canonical(self) -> str:
        """Fully-qualified identifier for dict keys / reports / cache keys.

        Per cx review: must include venue + instrument_class so that
        cross-venue collisions don't silently occur when Symbol is used
        as a dict/cache key downstream.

        Format: "{venue}__{base}_{quote}__{instrument_class}"
        Example: "binance__BTC_USDT__spot", "okx__BTC_USDT__perp"
        """
        return f"{self.venue}__{self.base}_{self.quote}__{self.instrument_class.value}"

    def filesystem_safe(self) -> str:
        """Same shape as canonical() — filesystem-safe by construction."""
        return self.canonical()

    def display(self) -> str:
        """Short human-readable form for reports."""
        return f"{self.base}/{self.quote}@{self.venue}"

@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: Symbol
    lot_size: float          # 100 A-share, 1e-5 BTC spot
    tick_size: float
    min_notional: float      # USDT minimum order size
    margin_supported: bool = False
    funding_interval_sec: int = 0   # 0 for spot, 28800 (8h) for perp
```

```python
# core/calendar.py
from datetime import datetime
from typing import Protocol

class TradingCalendar(Protocol):
    def is_open(self, ts: datetime) -> bool: ...
    def next_open(self, ts: datetime) -> datetime: ...
    def next_close(self, ts: datetime) -> datetime: ...

class Always24x7Calendar:
    """No closure ever. UTC-only by convention."""
    def is_open(self, ts: datetime) -> bool:
        return True
    def next_open(self, ts: datetime) -> datetime:
        return ts
    def next_close(self, ts: datetime) -> datetime:
        # Conventionally end-of-day UTC for daily reports.
        return ts.replace(hour=23, minute=59, second=59)

# AShareCalendar already exists conceptually in qlib; wrap it in a
# class with the Protocol shape but do NOT modify Qlib internals at
# Phase 0. Phase G+ can replace the Qlib wrapper with a clean impl.
```

```python
# core/settlement.py
from datetime import datetime
from typing import Protocol

class SettlementModel(Protocol):
    """Decides whether a position opened at opened_ts can be CLOSED now."""
    def can_close(self, opened_ts: datetime, now_ts: datetime) -> bool: ...
    def lock_pending(self, target_weights: dict) -> dict: ...

class T1SettlementModel:
    """A-share: today's buys lock until tomorrow's market open."""
    def can_close(self, opened_ts, now_ts) -> bool:
        return opened_ts.date() < now_ts.date()
    def lock_pending(self, target_weights):
        # Returns dict of {code: weight} that is held pending until T+1.
        # Caller uses this to populate paper/oms.py pending_target_weights.
        return dict(target_weights)

class InstantSettlementModel:
    """Crypto spot: positions closable immediately after fill."""
    def can_close(self, opened_ts, now_ts) -> bool:
        return True
    def lock_pending(self, target_weights):
        return {}  # nothing locked
```

```python
# core/cost.py
from typing import Protocol

class CommissionModel(Protocol):
    def commission(self, notional: float, side: str, taker: bool = True) -> float: ...

class TaxModel(Protocol):
    def tax(self, notional: float, side: str) -> float: ...

class ImpactModel(Protocol):
    def impact(self, notional: float, adv_quote: float) -> float: ...  # bps

class AShareCommission:
    def commission(self, notional, side, taker=True):
        return max(notional * 0.0003, 5.0)

class AShareStampTax:
    def tax(self, notional, side):
        return notional * 0.001 if side == "sell" else 0.0

class CryptoFlatCommission:
    """Configured per exchange + fee tier."""
    def __init__(self, taker_bps: float, maker_bps: float):
        self.taker_bps = taker_bps
        self.maker_bps = maker_bps
    def commission(self, notional, side, taker=True):
        bps = self.taker_bps if taker else self.maker_bps
        return notional * bps / 10000.0

class NoTax:
    def tax(self, notional, side): return 0.0

class LinearImpact:
    """Slippage = k * (notional / adv_quote) bps. Conservative default k=20."""
    def __init__(self, k_bps: float = 20.0):
        self.k_bps = k_bps
    def impact(self, notional, adv_quote):
        if adv_quote <= 0:
            return 50.0  # safety floor: 50bps if ADV unknown
        return self.k_bps * (notional / adv_quote)
```

```python
# core/data_availability.py
from datetime import datetime
from typing import Protocol
from .instrument import Symbol

class MarketDataSource(Protocol):
    def latest_bar_ts(self, symbol: Symbol, timeframe: str) -> datetime: ...
    def is_stale(self, symbol: Symbol, timeframe: str,
                 now_ts: datetime, max_lag_sec: int) -> bool: ...
```

```python
# core/universe.py
from typing import Protocol
from datetime import datetime

class UniverseFilter(Protocol):
    def eligible(self, asof_ts: datetime) -> list[str]: ...  # canonical symbols
```

Acceptance for §1.4: each Protocol has at least 2 implementations
(A-share + crypto where applicable) with byte-identical A-share
regression test pass.

### 1.5 Settlement-vs-Data-Latency Design (FROZEN, Phase 0a doc + 0b code)

CX roadmap §"Phase Crypto-0 Tasks" already specifies:

> - T+0 settlement is a Phase Crypto-0 OMS state-machine requirement.
> - WebSocket-class real-time data is deferred to Phase Crypto-G+.
> - Phase A-F use REST closed bars with stale/incomplete-bar guards.

This spec adds the implementation contract:

- `paper/crypto_oms.py` (Phase D) wires `InstantSettlementModel` from
  `core.settlement`. It does NOT carry over `pending_target_weights`
  from `paper/oms.py`. Test: round-trip BUY+SELL in same bar
  succeeds.
- `data/collectors/crypto_market.py` (Phase A) emits ONLY closed
  bars with a stabilization buffer: a bar is accepted only when
  `bar_close_ts + CLOSED_BUFFER_SEC <= ingested_at`
  (`CLOSED_BUFFER_SEC = 120` default, per cx review #4).
- Stale-bar guard in `MarketDataSource.is_stale` defaults
  `max_lag_sec`:
  - `1h` bars: 90 minutes (allows for exchange API lag + 1 retry +
    `CLOSED_BUFFER_SEC`)
  - `4h` bars: 5 hours
  - `1d` bars: 26 hours

These defaults are conservative; can be tightened after Phase A
measurement using Phase 0a spike data on actual exchange revision
windows.

### 1.6 Evidence Tagging (FROZEN, Phase 0a)

Tagging system (per CX Addendum):

- `[paper-reported]`
- `[exchange-dashboard]`
- `[open-source-backtest]`
- `[validated-on-local]`

Implementation:

- Add `numeric_claims_audit.md` to `plans/`. Every numeric claim used
  for sizing/risk in any plan doc is listed in this file with its
  tag and source.
- Phase 0 acceptance: every claim used in CC plan §3-7 and CX
  roadmap/lit-review is tagged. Self-retag of cx-review-of-cc body
  also lands here (Implementation Punch List item D).
- Phase D-H acceptance: any new claim used for sizing must be
  `[validated-on-local]` before merging.

### 1.7 Phase 0 Measurement Spike (FROZEN, Phase 0a)

File: `scripts/crypto_phase0_spike.py`

This script measures the DETAIL-DEFERRED items from §0:

```python
"""Phase 0 measurement spike. Runs against public mainnet REST APIs
only — read-only, no keys, no testnet (per cx review #7).
Output: /Volumes/DATA/crypto/phase0_measurements.json (per §−0.4)"""

def measure_rate_limits():
    """Test Binance/OKX/Bybit fetch_ohlcv throughput.
    Record: requests/min until 429, recovery time, weight-based limits."""

def measure_history_depth():
    """For each (exchange, symbol, timeframe), measure how far back
    fetch_ohlcv returns data. Record max retrievable history."""

def measure_funding_history():
    """fetch_funding_rate_history for BTC/ETH/SOL perp. Record depth."""

def measure_fee_tiers():
    """Read documented fee schedules from exchange docs/config.
    Record taker/maker bps at 0-100k / 100k-1M / 1M+ tiers."""

def measure_network_profile():
    """Check if scheduler/network_profiles.py has 'crypto_global' profile.
    Test proxy reachability if configured."""

def measure_existing_crypto_collector():
    """Check if data/collectors/crypto.py + AKShare fallback still works.
    Document its current state before refactor."""
```

Run order: **before** any Phase A implementation. Results go into
`crypto-data-contract.md` as the empirical foundation.

### 1.8 Phase 0 Acceptance Gates (FROZEN, split per cx review)

#### 1.8.a Phase 0a Acceptance (documentation + measurement)

Per cx review #7: Phase 0a allows **new read-only measurement/audit
scripts** (e.g. `crypto_phase0_spike.py`) — what is forbidden is
modification of A-share production code paths or `core/` Protocol
code. Adding a new isolated script under `scripts/` that touches no
existing files is consistent with "zero A-share/core production code
change".

Phase 0a is signed off when ALL of these pass (zero A-share/core
production code change; new isolated read-only scripts allowed):

- [ ] `plans/crypto-data-contract.md` exists with §1.2 schemas filled in
- [ ] §14.1 audit has `AUDIT-FROZEN-AT: <sha>` header committed
- [ ] `plans/numeric_claims_audit.md` covers all sizing/risk claims
- [ ] `scripts/crypto_phase0_spike.py` ran successfully (new isolated
      read-only script per cx #7), results in
      `/Volumes/DATA/crypto/phase0_measurements.json` (per §−0.4)
- [ ] CC plan self-corrections (§"CC Plan Self-corrections Required"
      from cc-review) merged into `cc-crypto-quant-integration-plan-...md`
- [ ] CX roadmap Implementation Punch List items A/B/C/D resolved
- [ ] A-share daily cron continues GREEN — production paths untouched
      so this is structurally guaranteed

#### 1.8.b Phase 0b Acceptance (core/ Protocol greenfield)

Phase 0b is signed off when ALL of these pass:

- [ ] `core/` directory created with §1.4 Protocols
- [ ] **2+ impls each**: A-share-flavored impl exists (e.g.
      `T1SettlementModel`, `AShareCommission`) and crypto-flavored
      impl exists (e.g. `InstantSettlementModel`, `CryptoFlatCommission`)
- [ ] **No `ashare/` file imports from `core/` yet** (zero behavior
      change to A-share production paths)
- [ ] Namespace isolation lint passes
      (`scripts/check_namespace_isolation.py`)
- [ ] A-share daily cron GREEN for 3 consecutive days after 0b PR merge

#### 1.8.c Phase 0c Acceptance (per-row regression tests + adapter wiring)

Phase 0c is signed off when ALL of these pass:

- [ ] 20 regression smoke tests in
      `tests/test_ashare_unchanged_after_crypto_split.py` exist and
      pass (one per §14.1 row)
- [ ] Per-row adapter: `ashare/<wrapper>.py` exists that calls
      `core/` Protocol with A-share impl; original production file
      (`paper/oms.py`, etc.) **delegates** to adapter, **does not get
      physically moved**
- [ ] A-share daily recommendations byte-identical to pre-0c baseline
      for 3 consecutive days
- [ ] A-share daily cron GREEN for 3 consecutive days after each 0c PR
      merge (per Layer 5 of §−0.5)

Physical file moves (`paper/oms.py` → `ashare/oms.py`, etc.) are
**deferred to Phase G**, not Phase 0c.

## §2. Phase Crypto-A: Spot OHLCV + Funding/OI Foundation

Per cx review r3 #2: Phase A splits into two parts so duration and
acceptance are consistent.

- **Phase A.impl** (3-5 days): build collectors, cron entries,
  health monitoring, parquet writes. End state: data flowing.
- **Phase A.soak** (7 consecutive days of GREEN): cron runs
  uninterrupted, gap rate < 0.5%, fallback exchanges reachable.

**Total Phase A duration: 7-10 days.**

Phase A.soak starts the day Phase A.impl ends. No code change permitted
during soak — any change resets the 7-day soak clock.

### 2.1 CCXT Spot Collector (FROZEN)

File: `data/collectors/crypto_market.py`

```python
"""Crypto spot OHLCV collector via CCXT.

Designed to be cron-driven, not actor-driven. Idempotent writes to
parquet partitioned by (exchange, symbol, timeframe, date).
"""
from __future__ import annotations
import ccxt
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime, timezone
from pathlib import Path
import logging
import time

from core.instrument import Symbol
from core.asset import AssetClass, InstrumentClass

logger = logging.getLogger(__name__)

from config.crypto_storage import CRYPTO_STORAGE_ROOT, ensure_mounted_and_writable
STORAGE_ROOT = CRYPTO_STORAGE_ROOT / "raw" / "ohlcv"
# Per §−0.4: never hardcode the path. Every entrypoint script must
# call ensure_mounted_and_writable() before any write.

TIMEFRAME_SEC = {
    "1m": 60, "5m": 300, "15m": 900, "1h": 3600,
    "4h": 14400, "1d": 86400,
}

# Per cx review #4: data may be revised within 30-90s after bar close
# (late trades, rollups). 120s buffer prevents ingesting still-stabilizing
# bars. This constant is the SINGLE SOURCE OF TRUTH for closed-bar logic
# — both fetch_recent and fetch_historical must call _is_closed_with_buffer.
CLOSED_BUFFER_SEC = 120


def _is_closed_with_buffer(bar_open_ms: int, tf_sec: int, now_ms: int) -> bool:
    """Returns True iff bar_open is far enough in the past that the
    bar has closed AND data has had CLOSED_BUFFER_SEC to stabilize.

    Per cx review #2 + #3: this is the ONLY closed-bar check. Direct
    `bar_open + tf_ms <= now_ms` is forbidden — it allows just-closed
    unstable bars through.
    """
    bar_close_ms = bar_open_ms + tf_sec * 1000
    return bar_close_ms + CLOSED_BUFFER_SEC * 1000 <= now_ms


class CryptoSpotCollector:
    def __init__(self, exchange: str, network_profile: str = "crypto_global"):
        self.exchange = exchange.lower()
        self.client = self._make_client(network_profile)

    def _make_client(self, profile: str) -> ccxt.Exchange:
        ex_cls = getattr(ccxt, self.exchange)
        config = {"enableRateLimit": True, "timeout": 30_000}
        # Proxy injection from scheduler/network_profiles.py:
        proxy = _load_proxy(profile)
        if proxy:
            config["proxies"] = {"http": proxy, "https": proxy}
        return ex_cls(config)

    def fetch_recent(
        self,
        symbol: Symbol,
        timeframe: str,
        n_bars: int = 1000,
    ) -> pd.DataFrame:
        """Fetch the most recent n_bars CLOSED-AND-STABILIZED bars. Both
        the in-progress bar and the most-recently-closed-but-unstable
        bar (within CLOSED_BUFFER_SEC) are filtered out."""
        ccxt_symbol = f"{symbol.base}/{symbol.quote}"
        raw = self.client.fetch_ohlcv(ccxt_symbol, timeframe, limit=n_bars)
        df = pd.DataFrame(
            raw, columns=["timestamp_utc", "open", "high", "low", "close", "volume_base"]
        )
        df["timestamp_utc"] = df["timestamp_utc"].astype("int64")
        # SINGLE closed-bar gate — uses buffer per cx review #2.
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        tf_sec = TIMEFRAME_SEC[timeframe]
        df["is_closed_bar"] = df["timestamp_utc"].apply(
            lambda t: _is_closed_with_buffer(int(t), tf_sec, now_ms)
        )
        df = df[df["is_closed_bar"]].copy()
        # Augment
        df["exchange"] = self.exchange
        df["symbol"] = ccxt_symbol
        df["timeframe"] = timeframe
        df["volume_quote"] = df["volume_base"] * (df["open"] + df["close"]) / 2.0
        df["quote_volume_estimated"] = True  # per cx r3 #6
        df["trades"] = -1
        df["ingested_at"] = now_ms
        return df

    def fetch_historical(
        self,
        symbol: Symbol,
        timeframe: str,
        start_ts_ms: int,
        end_ts_ms: int,
    ) -> pd.DataFrame:
        """Paginate fetch_ohlcv from start to end. Rate-limited by CCXT.

        Per cx review #3: even in historical mode, the last few bars
        near end_ts_ms may be unstable. Same _is_closed_with_buffer
        filter applies — NO unconditional is_closed_bar=True.
        """
        ccxt_symbol = f"{symbol.base}/{symbol.quote}"
        tf_ms = TIMEFRAME_SEC[timeframe] * 1000
        tf_sec = TIMEFRAME_SEC[timeframe]
        all_rows = []
        cursor = start_ts_ms
        while cursor < end_ts_ms:
            try:
                raw = self.client.fetch_ohlcv(
                    ccxt_symbol, timeframe, since=cursor, limit=1000
                )
            except ccxt.RateLimitExceeded:
                logger.warning("Rate limited, sleeping 30s")
                time.sleep(30)
                continue
            if not raw:
                break
            all_rows.extend(raw)
            cursor = raw[-1][0] + tf_ms
        df = pd.DataFrame(
            all_rows,
            columns=["timestamp_utc", "open", "high", "low", "close", "volume_base"],
        )
        df = df.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")
        # Per cx review r3 #3: CCXT fetch_ohlcv has `since=` but no
        # `until=`. The last page can include bars beyond end_ts_ms.
        # Drop them so PIT backfill doesn't ingest data outside the
        # requested window.
        df = df[(df["timestamp_utc"] >= start_ts_ms) & (df["timestamp_utc"] < end_ts_ms)].copy()
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        # Same closed-bar gate as fetch_recent — protects tail.
        df["is_closed_bar"] = df["timestamp_utc"].apply(
            lambda t: _is_closed_with_buffer(int(t), tf_sec, now_ms)
        )
        df = df[df["is_closed_bar"]].copy()
        df["exchange"] = self.exchange
        df["symbol"] = ccxt_symbol
        df["timeframe"] = timeframe
        # Per cx review r3 #6: quote_volume here is an estimate
        # (volume_base * mid_price). The schema flag records this so
        # downstream code can distinguish from exchange-reported
        # quote_volume if later sourced.
        df["volume_quote"] = df["volume_base"] * (df["open"] + df["close"]) / 2.0
        df["quote_volume_estimated"] = True
        df["trades"] = -1
        df["ingested_at"] = now_ms
        return df

    def write_parquet(self, df: pd.DataFrame) -> int:
        """Idempotent write partitioned by (exchange, symbol, timeframe, date).

        Uses upsert semantics on (timestamp_utc, exchange, symbol, timeframe):
        existing rows with same key are overwritten (handles late-arriving
        revisions from exchange data corrections).
        """
        if df.empty:
            return 0
        # Group by date for partitioning
        df["_date"] = pd.to_datetime(df["timestamp_utc"], unit="ms", utc=True).dt.date
        n_written = 0
        for (date, exch, sym, tf), group in df.groupby(["_date", "exchange", "symbol", "timeframe"]):
            sym_fs = sym.replace("/", "_")
            out_dir = STORAGE_ROOT / exch / sym_fs / tf / f"year={date.year}" / f"month={date.month:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"day={date.day:02d}.parquet"
            group = group.drop(columns=["_date"])
            if out_path.exists():
                existing = pd.read_parquet(out_path)
                combined = pd.concat([existing, group])
                combined = combined.drop_duplicates(
                    subset=["timestamp_utc", "exchange", "symbol", "timeframe"],
                    keep="last",  # last-write-wins for exchange corrections
                )
                combined = combined.sort_values("timestamp_utc")
                combined.to_parquet(out_path, index=False)
                n_written += len(combined) - len(existing)
            else:
                group.sort_values("timestamp_utc").to_parquet(out_path, index=False)
                n_written += len(group)
        return n_written


def _load_proxy(profile: str, strict: bool = True) -> str | None:
    """Read proxy URL from scheduler/network_profiles.py.

    Per cx review r3 #4: when `strict=True` (default for crypto
    collectors), missing profile or missing module raises with a clear
    error. Silently swallowing means a misconfigured network profile
    would cause crypto jobs to fall through to direct connection,
    which then might or might not work depending on machine — that's
    exactly the kind of silent drift we want to prevent.

    Set strict=False only for opt-in use cases where direct connection
    is an explicit fallback.
    """
    try:
        from scheduler.network_profiles import get_profile
    except ImportError as e:
        if strict:
            raise RuntimeError(
                f"Network profile '{profile}' requested but "
                f"scheduler.network_profiles is not importable: {e}. "
                f"Crypto collectors require strict network config."
            ) from e
        return None
    try:
        p = get_profile(profile)
    except (KeyError, AttributeError, ValueError) as e:
        if strict:
            raise RuntimeError(
                f"Network profile '{profile}' not defined in "
                f"scheduler.network_profiles. Add it before running crypto "
                f"collectors. Underlying error: {e}"
            ) from e
        return None
    if p is None:
        if strict:
            raise RuntimeError(
                f"Network profile '{profile}' resolved to None. "
                f"Configure it in scheduler.network_profiles."
            )
        return None
    return p.proxy_url
```

### 2.2 Universe Construction (FROZEN, Phase A initial)

File: `config/crypto_universe.py`

Per cx review #5: Phase A must validate OKX/Bybit fallback to avoid
sole dependency on Binance (China-network access risk). Primary
collection from Binance; OKX/Bybit health-checked + ready to swap.

```python
"""Phase A initial universe: 5 USDT spot majors, primary on Binance
with OKX/Bybit fallback validated.

Phase 0a measurement spike confirmed which exchanges are accessible
from the user's network. PRIMARY_EXCHANGE may switch based on spike
results.
"""
from core.instrument import Symbol
from core.asset import AssetClass, InstrumentClass

PRIMARY_EXCHANGE = "binance"     # Default; may be overridden by 0a spike
FALLBACK_EXCHANGES = ["okx", "bybit"]

def _make_spot(base: str, exchange: str) -> Symbol:
    return Symbol(base, "USDT", exchange, AssetClass.CRYPTO, InstrumentClass.SPOT)

def _make_perp(base: str, exchange: str) -> Symbol:
    return Symbol(base, "USDT", exchange, AssetClass.CRYPTO, InstrumentClass.PERPETUAL)

PHASE_A_SPOT_BASES = ["BTC", "ETH", "SOL", "BNB", "XRP"]
PHASE_A_PERP_BASES = ["BTC", "ETH", "SOL"]

# Primary universe: actively collected
PHASE_A_UNIVERSE = [_make_spot(b, PRIMARY_EXCHANGE) for b in PHASE_A_SPOT_BASES]
PHASE_A_PERP_UNIVERSE = [_make_perp(b, PRIMARY_EXCHANGE) for b in PHASE_A_PERP_BASES]

# Fallback universe: health-checked only (collector instantiated +
# fetch_ohlcv test) but data not stored. If primary unhealthy, daily
# report flags "fallback available: okx YES, bybit YES" and the
# operator can switch PRIMARY_EXCHANGE.
PHASE_A_FALLBACK_UNIVERSE = [
    _make_spot(b, ex)
    for ex in FALLBACK_EXCHANGES
    for b in PHASE_A_SPOT_BASES
]

PHASE_A_TIMEFRAMES = ["1h", "4h", "1d"]
```

Health check responsibility: §2.7 health file includes per-fallback
exchange reachability status. Switching PRIMARY_EXCHANGE is a config
change (single edit + cron restart), not a code change.

### 2.3 Gap Detection + Idempotency (FROZEN)

```python
# data/collectors/crypto_gap_detector.py
def detect_gaps(
    symbol: Symbol,
    timeframe: str,
    start_ts_ms: int,
    end_ts_ms: int,
) -> list[tuple[int, int]]:
    """Returns list of (gap_start_ms, gap_end_ms) windows missing from parquet.

    Algorithm:
    1. Read all parquet partitions for (symbol, timeframe) in [start, end]
    2. Build expected timestamp grid: range(start, end, tf_ms)
    3. Set-diff against actual timestamps
    4. Coalesce consecutive gaps into windows
    """
    ...
```

Used by `scripts/crypto_update_market_data.py` to backfill identified
gaps via `fetch_historical`.

### 2.4 Survivorship-aware History (FROZEN)

Two-list discipline:

- Current eligible universe: `config/crypto_universe.PHASE_A_UNIVERSE`
- Historical eligible (PIT) universe: derived at query time as

```python
def eligible_at(asof_ts_ms: int) -> list[Symbol]:
    """Returns symbols that were tradable AT asof_ts_ms.

    Includes:
    - currently listed symbols whose `listed_at <= asof_ts_ms`
    - symbols in delisted_coins.parquet where
      `listed_at <= asof_ts_ms < delisted_at`
    """
```

Phase A acceptance: backtest at any T must read both lists.

For Phase A's 5 majors, delisting hasn't happened, so the delisted
table is empty. But the code path must exist so Phase B universe
expansion (top-30) does not silently survivorship-bias.

### 2.5 Funding/OI Collector (FROZEN)

File: `data/collectors/crypto_derivatives.py`

Per cx review #6: funding 8h × 365 days ≈ 1095 records exceeds the
typical `limit=1000` single-call cap, and many exchanges only return
the most recent window per call. Pagination is mandatory.

```python
class CryptoDerivativesCollector:
    def __init__(self, exchange: str, network_profile: str = "crypto_global"):
        self.exchange = exchange.lower()
        self.client = _make_client(self.exchange, network_profile)

    def fetch_funding_history(
        self,
        symbol: Symbol,
        start_ts_ms: int,
        end_ts_ms: int,
    ) -> pd.DataFrame:
        """Returns funding events from start to end, paginated.

        Per cx review #6: paginate via cursor advance, do not assume
        single-call returns full window. 8h funding × 1y ≈ 1095 events.
        """
        ccxt_symbol = f"{symbol.base}/{symbol.quote}:{symbol.quote}"
        # Funding interval varies per exchange; Binance perp = 8h = 28800000 ms
        FUNDING_INTERVAL_MS = 8 * 3600 * 1000
        all_rows = []
        cursor = start_ts_ms
        while cursor < end_ts_ms:
            try:
                raw = self.client.fetch_funding_rate_history(
                    ccxt_symbol, since=cursor, limit=1000
                )
            except ccxt.RateLimitExceeded:
                logger.warning("Funding rate limited, sleeping 30s")
                time.sleep(30)
                continue
            if not raw:
                break
            all_rows.extend(raw)
            last_ts = raw[-1].get("timestamp") or raw[-1].get("fundingTimestamp")
            if last_ts is None or last_ts <= cursor:
                # Defensive: prevent infinite loop on malformed response
                break
            cursor = last_ts + FUNDING_INTERVAL_MS
        # Convert to schema in §1.2; dedup by (exchange, symbol, timestamp_utc)
        ...

    def fetch_open_interest_snapshot(
        self,
        symbol: Symbol,
    ) -> dict:
        """Single-point OI snapshot. Cron-scheduled every 15 minutes."""
        ccxt_symbol = f"{symbol.base}/{symbol.quote}:{symbol.quote}"
        return self.client.fetch_open_interest(ccxt_symbol)

    def fetch_open_interest_history(
        self,
        symbol: Symbol,
        start_ts_ms: int,
        end_ts_ms: int,
        period: str = "15m",
    ) -> pd.DataFrame:
        """OI historical backfill via fetch_open_interest_history.

        Per cx review #5: NOT used in Phase A (15min cron sampling is
        the Phase A source, accumulating over time). This method is
        provided for Phase B where deeper historical OI is needed.
        Many exchanges restrict OI history depth to 30 days — Phase B
        design will document the depth limit per exchange.
        """
        # Implementation deferred to Phase B; placeholder here for
        # interface stability.
        raise NotImplementedError("OI historical backfill is Phase B work.")
```

- **Funding**: event-driven (every 8h on Binance). Cron pulls
  historically from `start_ts_ms` (last successful fetch) to now.
- **OI**: sampled every 15 minutes via cron polling. Phase A
  accumulates samples forward — no historical backfill in A.

### 2.6 Cron Schedule (FROZEN)

Per cx review #4: 1h cron runs ONCE per hour (was `5,35` which
duplicates work on already-closed bar); all collectors honor a
`closed_buffer_sec` so just-closed exchange data has time to stabilize
before ingestion.

Per §−0.5 Layer 3: all crypto cron jobs in dedicated `CRYPTO_JOBS`
group, separate from A-share, no `enforce_deps` references across
groups.

To be added to `scripts/install_crontab.py`:

**Timezone rule (per cx review #1)**: macOS crontab does not honor
`CRON_TZ=UTC`. The machine timezone is `Asia/Shanghai` (UTC+8).
Therefore:

- **All `schedule` strings below are in `Asia/Shanghai` local time.**
- **All collector / health internal logic is UTC.**
- Each cron entry comment shows both the local schedule and the UTC
  event it tracks, to make alignment auditable.

Example mapping:
- Binance funding settles at `00:00 / 08:00 / 16:00 UTC`
- That is `08:00 / 16:00 / 00:00 CST` (UTC+8)
- We want to run 15 min after each settlement
- Local cron: `15 0,8,16 * * *` is WRONG (would fire 0/8/16 CST = 16/00/08 UTC)
- Local cron: `15 8,16,0 * * *` (still 15 8,16,0) is the CST equivalent of 00/08/16 UTC

```python
# scripts/install_crontab.py
# CRYPTO_JOBS — all schedules in Asia/Shanghai local time.
# Internal logic in collectors uses UTC.

CRYPTO_JOBS = [
    CronJob(
        name="crypto_ohlcv_1h",
        # Local 05min past every hour. Tracks UTC hour-close + buffer.
        # 1h bars close every UTC hour boundary; CST hour boundary is
        # the same minute mark, so this works identically in any TZ.
        schedule="5 * * * *",
        command="scripts/crypto_update_market_data.py --timeframe 1h --network crypto_global",
        timeout_sec=600,
        enforce_deps=False,
    ),
    CronJob(
        name="crypto_ohlcv_4h",
        # 4h bars close at UTC 00/04/08/12/16/20.
        # In CST that is 08/12/16/20/00/04. Fire 10min after each.
        schedule="10 0,4,8,12,16,20 * * *",
        command="scripts/crypto_update_market_data.py --timeframe 4h --network crypto_global",
        timeout_sec=900,
        enforce_deps=False,
    ),
    CronJob(
        name="crypto_ohlcv_1d",
        # 1d bars close at UTC 00:00 = CST 08:00. Fire 08:05 CST.
        schedule="5 8 * * *",
        command="scripts/crypto_update_market_data.py --timeframe 1d --network crypto_global",
        timeout_sec=1200,
        enforce_deps=False,
    ),
    CronJob(
        name="crypto_funding",
        # Binance funding settles UTC 00/08/16 = CST 08/16/00.
        # Fire 15min after each settlement.
        schedule="15 0,8,16 * * *",
        command="scripts/crypto_update_derivatives.py --kind funding --network crypto_global",
        timeout_sec=300,
    ),
    CronJob(
        name="crypto_oi",
        # OI sampling every 15min. TZ-independent.
        schedule="*/15 * * * *",
        command="scripts/crypto_update_derivatives.py --kind oi --network crypto_global",
        timeout_sec=300,
    ),
    CronJob(
        name="crypto_data_health",
        schedule="20 */4 * * *",
        command="scripts/crypto_data_health.py --network crypto_global",
        timeout_sec=300,
        enforce_deps=True,
        dep_wait_seconds=900,
    ),
    # ↑ enforce_deps within CRYPTO_JOBS group only; never refers to
    # A-share job names. Per §−0.5 Layer 3.
    # ↑ All commands include --network crypto_global per cx review #4
    # so health monitoring knows network profile.
]
```

**Boot-time assertion (FROZEN)**: `scripts/install_crontab.py` must
print and verify the timezone at install time:

```python
import time
print(
    f"Installing crontab.\n"
    f"  Machine TZ name: {time.tzname}\n"
    f"  UTC offset (sec): {time.timezone} ({-time.timezone//3600:+d}h)\n"
    f"  Local timezone abbreviations: {time.tzname[0]} / {time.tzname[1]}"
)
if time.timezone != -28800:  # CST = UTC+8 → offset -28800s
    raise SystemExit(
        f"Timezone mismatch. CRYPTO_JOBS schedules assume Asia/Shanghai "
        f"(UTC+8, offset -28800).\n"
        f"Detected: tzname={time.tzname}, offset={time.timezone}.\n"
        f"Either set TZ=Asia/Shanghai before running, or update "
        f"CRYPTO_JOBS schedules and their //UTC comments to match the "
        f"current local timezone."
    )
```

If the machine timezone changes (e.g. UTC server), all CRYPTO_JOBS
schedules must be retranslated. The assertion (per cx review r3 #8)
prints `time.tzname` so the operator can see exactly which TZ was
detected, not just the numeric offset.

**Retry policy**: if `crypto_ohlcv_1h` fails (e.g. transient API
error), retry is handled by `run_with_status.py` standard retry
mechanism (existing infra), not by scheduling a second cron at `:35`.
Spurious duplicate cron entries waste API quota and obscure failures.

**Closed-bar buffer**: the single source of truth is
`_is_closed_with_buffer()` in `data/collectors/crypto_market.py` (see
§2.1). Both `fetch_recent` and `fetch_historical` call it. Per cx
review #2 + #3, **direct `bar_open + tf_ms <= now_ms` comparisons or
unconditional `is_closed_bar=True` assignments are forbidden** — they
allow just-closed unstable bars through.

Default: `CLOSED_BUFFER_SEC = 120`. Tighter values can be calibrated
after Phase 0a measurement spike confirms per-exchange revision
windows.

Phase A acceptance: all 6 cron entries run for 3 consecutive days
without manual intervention.

### 2.7 Health File Format (FROZEN)

File: `/Volumes/DATA/crypto/health/crypto_data_health.json`
(via `CRYPTO_STORAGE_ROOT / "health" / "crypto_data_health.json"` —
see §−0.4)

Per cx review #9: JSON cannot contain `|` as a value alternation;
`overall_status` is a single string from the enum below.

```json
{
  "generated_at_utc": "2026-05-30T12:34:56Z",
  "network_profile": "crypto_global",
  "primary_exchange": "binance",
  "fallback_exchanges_reachable": {
    "okx": true,
    "bybit": true
  },
  "ohlcv": {
    "binance__BTC_USDT__spot/1h": {
      "latest_bar_ts": 1748600000000,
      "latest_bar_age_sec": 3245,
      "stale": false,
      "stale_threshold_sec": 5400,
      "gap_count_30d": 0,
      "gap_rate_30d": 0.0
    }
  },
  "funding": {
    "binance__BTC_USDT__perp": {
      "latest_funding_ts": 1748592000000,
      "latest_funding_age_sec": 12000,
      "stale": false
    }
  },
  "open_interest": {},
  "overall_status": "GREEN"
}
```

**`overall_status` enum** (one of):

- `"GREEN"`
- `"YELLOW"`
- `"RED"`

Decision rules:

- `RED` if ANY 1h bar > 90 min stale, OR any 4h bar > 5h stale, OR
  any 1d bar > 26h stale, OR primary exchange unreachable
- `YELLOW` if any gap rate > 1% in last 30 days, OR **any fallback
  exchange unreachable (degraded resilience but operational —
  primary still working, fallback path not available if needed)**.
  Per cx review r3 #7: fallback unreachable is YELLOW not RED;
  daily report must surface it so operator can investigate, but
  Phase A continues since primary is fine.
- `GREEN` otherwise

Phase A acceptance: 7 consecutive days `GREEN` before Phase B starts.

Symbol identifiers in this file use `Symbol.canonical()` format per
§1.4 (e.g. `binance__BTC_USDT__spot`) so cross-venue collisions
cannot occur.

#### Cross-source Sanity (FROZEN, per cx system design review §2)

Phase A is single-source for storage (Binance primary) but
**multi-source for sanity** — we sample OKX and Bybit on BTC/ETH/SOL
purely to compute health metrics, not to merge into training data.

Health file extended with `cross_source` section:

```json
"cross_source": {
  "binance__BTC_USDT__spot/1h": {
    "primary_close": 65432.10,
    "okx_close": 65430.50,
    "bybit_close": 65433.20,
    "max_spread_bps": 4.1,
    "binance_latency_ms": 420,
    "okx_latency_ms": 510,
    "bybit_latency_ms": 480
  },
  "binance__BTC_USDT__perp/funding": {
    "binance_rate": 0.00010,
    "okx_rate": 0.00012,
    "bybit_rate": 0.00009,
    "sign_agreement": true
  }
}
```

Decision rules added:

- `YELLOW` if any cross-source close spread > 25 bps OR funding sign
  disagrees across primary/fallback
- `RED` rules unchanged (primary stale / unreachable)

Cron entry: cross-source sanity runs piggybacked on
`crypto_data_health` job (every 4h, see §2.6) — it does NOT spin up
separate cron, only adds ~3 extra REST calls per health run.

**Storage**: cross-source spread/latency samples land in
`/Volumes/DATA/crypto/health/cross_source_samples.parquet`, kept for
30 days for trend visualization. No merge into `raw/ohlcv/`.

Phase A acceptance gate (§2.9.b) adds:
- [ ] Cross-source samples flowing for BTC/ETH/SOL during soak
- [ ] No primary `RED` events during the 7-day soak (cross-source by
      construction can only trigger YELLOW, not RED; this bullet
      avoids the impossible "no RED-by-cross-source" wording)
- [ ] Cross-source YELLOW events allowed but documented in
      `/Volumes/DATA/crypto/health/incidents.md` with cause; soak does
      not fail unless YELLOW is persistent (>50% of 7-day window)

### 2.8 Tests (FROZEN)

Per cx review #7: tests split into unit (fixture-driven) and
integration (public mainnet read-only). **Testnet is not used** —
testnet historical data quality may differ from mainnet and adds
operational complexity. **No API key configuration permitted in any
test** — keys are paid-account credentials and integration tests must
prove they work without them.

```python
# tests/test_crypto_collector_unit.py (NEW — fixture-driven, fast)

def test_closed_bar_filter_with_buffer():
    """A bar where bar_close + CLOSED_BUFFER_SEC > now is rejected."""
    # Pure unit, uses fake timestamps.

def test_gap_detection_from_fixture():
    """Given a parquet fixture with known gaps, detect_gaps returns them."""
    # Reads tests/fixtures/crypto_ohlcv_with_gaps.parquet

def test_universe_survivorship_path_exists():
    """eligible_at returns delisted+current union. Empty delisted table OK."""

def test_oi_15min_grid_alignment():
    """Given raw OI samples, write_parquet aligns to 15-min grid."""

def test_health_red_on_stale_bar():
    """is_stale=True correctly flips overall_status to RED."""

def test_symbol_canonical_includes_venue_and_instrument():
    """Per cx #6: cross-venue collisions don't happen via canonical()."""
    assert (
        Symbol("BTC", "USDT", "binance", AssetClass.CRYPTO, InstrumentClass.SPOT).canonical()
        != Symbol("BTC", "USDT", "okx", AssetClass.CRYPTO, InstrumentClass.SPOT).canonical()
    )


# tests/test_crypto_collector_integration.py (NEW — public mainnet REST, slow, opt-in)
#
# Marker: @pytest.mark.integration — excluded from default test run,
# included in nightly CI. No API keys. Read-only fetch_ohlcv only.

@pytest.mark.integration
def test_phase_a_collector_idempotent_mainnet():
    """Two consecutive fetch_recent calls against PUBLIC mainnet REST API
    produce same parquet rows after deduplication. No API key used.
    Throttled to respect documented rate limits."""

@pytest.mark.integration
def test_funding_8h_alignment_mainnet():
    """Funding history fetched from public REST aligns to exchange
    settlement schedule (00/08/16 UTC for Binance)."""

@pytest.mark.integration
def test_fallback_exchanges_reachable():
    """Per cx #5: OKX/Bybit reachable and return valid OHLCV format,
    even though we don't store their data in Phase A."""


# tests/test_ashare_unchanged_after_crypto_split.py (NEW, Phase 0c)
# 20 tests from §1.3 §14.1 audit — byte-identical regression on fixed
# A-share fixture date. Per §−0.5 Layer 4 CI gate.
```

CI policy:
- Unit tests run on every commit.
- Integration tests run nightly (mainnet rate-limited, slow).
- A-share regression tests (`test_ashare_unchanged_...`) run on
  every PR per §−0.5 Layer 4. Required to pass for merge.

No test ever configures an API key. If a key is needed for a feature,
the feature is gated to a manual operator workflow, not CI.

### 2.9 Phase A Acceptance Gate (FROZEN)

Per cx review r1 #5 + r3 #2: Phase A.impl is 3-5 days; Phase A.soak
is 7 days. OI accumulation in Phase A is forward-only (no historical
backfill until Phase B).

#### 2.9.a Phase A.impl Acceptance (end of build, 3-5 days)

Phase A.impl is signed off when ALL of these pass:

- [ ] All 6 cron entries in §2.6 deployed and running
- [ ] 1y of 1h bars for 5 symbols in parquet (historical backfill via
      `fetch_historical`)
- [ ] 1y of 4h bars for 5 symbols
- [ ] All available 1d bars for 5 symbols
- [ ] 1y of funding history for 3 perp symbols (paginated, per §2.5)
- [ ] OI samples for 3 perp symbols flowing at 15-min cadence
- [ ] All §2.8 tests pass in CI
- [ ] All §1.3 regression tests still pass (no A-share behavior change)
- [ ] OKX/Bybit fallback exchange reachability verified in health
      file (per §2.2)

#### 2.9.b Phase A.soak Acceptance (7 consecutive days post-impl)

After Phase A.impl signs off, the soak begins. During soak NO code
change is permitted (any change resets the soak clock to day 0).
Phase A.soak is signed off when ALL of these pass:

- [ ] 7 consecutive days of `GREEN` health status (per §2.7 enum)
- [ ] Gap rate < 0.5% across all (symbol, timeframe) combinations
      during the 7-day window
- [ ] **3+ consecutive days of OI samples** for 3 perp symbols (15-min
      cadence). 30-day OI accumulation moves to Phase B acceptance.
- [ ] No `RED` events during the 7-day window
- [ ] Any `YELLOW` events documented with cause + resolution in
      `/Volumes/DATA/crypto/health/incidents.md`
- [ ] A-share daily cron continues GREEN through the entire soak
      period (per §−0.5 Layer 6)

## §3. Phase Crypto-B: Feature Cache + Fee-aware Baseline (STRUCTURAL)

Duration: 5-7 days (matches cx-roadmap).

### 3.1 Features (STRUCTURAL)

File: `models/crypto_feature_pipeline.py`

Feature families:
- Returns: 1h/4h/12h/24h/3d/7d (computed from `close`)
- Momentum: ROC, distance to rolling high/low
- Reversal: last-bar return, 4h reversal
- Volatility: realized vol 12h/24h/7d, Parkinson H-L
- Volume: z-score, quote-volume rank
- Liquidity: dollar volume rank, Amihud illiquidity
- Cross-section vs BTC: relative strength, BTC beta
- Funding: current funding, rolling 8x funding mean, funding z-score
- OI: OI change 1h/4h/24h, OI/price divergence
- **Hard Gotcha enforcement**: IVOL computed but stored WITHOUT a
  sign assumption; sign is determined by local IC validation at
  Phase B acceptance, not copied from A-share

### 3.2 Rule-based Baseline Backtest (STRUCTURAL)

File: `scripts/crypto_backtest_baseline.py`

Per cx review #8: Phase B has **no supervised model** — that's Phase C.
Baseline strategies are simple rules that exist to validate the cost
model, backtest engine, and data pipeline. No "prediction" in Phase B.

Baseline rule menu (run each as a separate baseline):

1. **Equal-weight top-5**: hold the 5 Phase A coins at equal weight,
   4h rebalance. Establishes a no-skill benchmark.
2. **Momentum**: long-only top-quintile by realized `ret_24h`,
   4h rebalance. Pure backward-looking signal, no model.
3. **Reversal**: long-only bottom-quintile by realized `ret_4h`,
   4h rebalance.
4. **BTC-beta-neutral**: hold equal-weight Phase A coins minus a
   BTC short proportional to portfolio beta. Tests beta hedging
   plumbing.

Cost model: uses `CryptoFlatCommission(taker_bps=10, maker_bps=4)` +
`LinearImpact(k_bps=20)` + `NoTax()`. (Bps values from Phase 0a spike,
tagged `[exchange-dashboard]`.)

Why these and not "prediction": rule-based baselines test the
backtest engine independently of model risk. If equal-weight Sharpe
is wrong, the bug is in the engine. If momentum Sharpe is wrong, the
bug is in either engine or features. Only Phase C introduces a
trained model whose prediction joins the loop.

Acceptance: each baseline produces a report with RankIC (where
applicable), turnover, max drawdown, Sharpe (fee-adjusted). All
numbers tagged `[validated-on-local]`. The 4 baseline reports become
the reference against which Phase C model improvement is measured.

### 3.3 Phase B Acceptance Gate

- [ ] Feature cache parquet exists with all feature families
- [ ] All 4 rule-based baselines run on 1y data with reports
- [ ] Equal-weight baseline Sharpe matches a manual hand-calc (sanity
      check on backtest engine — not a profitability claim)
- [ ] IVOL sign explicitly validated; sign-flip check passes
- [ ] All features and rule baselines tagged `[validated-on-local]` in
      `numeric_claims_audit.md`
- [ ] No model trained yet (model is Phase C)

## §4. Phase Crypto-C: Supervised Model (STRUCTURAL)

Duration: 7-10 days.

### 4.1 Model (STRUCTURAL)

Files:
- `models/crypto_model.py`
- `scripts/crypto_train_model.py`
- `scripts/crypto_predict.py`

XGB + LGB ensemble. Walk-forward weekly retrain (per cx Hard Gotchas
"alpha decay 5-10x faster than A-share"). Rolling 60-day training
window (NOT A-share's 250d default).

Labels: `ret_24h_after_fee` and `ret_4h_after_fee` (two-head).

### 4.2 Phase C Acceptance Gate

Per cx review #10 + system-design review §3: with Phase A's N=5
universe, cross-sectional RankIC statistics are too noisy to support
hard thresholds. Phase C is therefore defined as a **shadow model**
— its job is to validate the training pipeline + walk-forward
semantics + retrain schedule, NOT to clear an IC bar.

Universe expansion to top-20 / top-30 is deferred to Phase E+ and
**gated by 7 hard pre-conditions** (per cx system design review §3 —
universe expansion is a meaningful capital/risk surface change, not
just a config edit):

1. Delisted-coin handling exists (`delisted_coins.parquet` write path
   tested with synthetic delisting event)
2. Stablecoin / depeg filter exists and passes a unit test on
   historical USDT/USDC depeg episodes
3. Scam / meme / wash-volume filter exists (rules from CC §11
   sanitizer)
4. Min dollar-volume floor enforced (default 50M USD/day rolling)
5. Min listed-days floor enforced (default 60 days)
6. Exchange coverage measured (≥2 exchanges list the symbol)
7. Survivorship test runs in CI (random sample of 5 delisted coins
   reappears in historical universe at `eligible_at(asof_ts)` queries)

Universe expansion PR also requires explicit user written sign-off
(per §−1 — non-trivial risk-surface change).

Acceptance:
- [ ] Rolling split walk-forward backtest runs end-to-end
- [ ] Weekly retrain cadence runs without error (per Hard Gotcha
      "alpha decay 5-10x faster")
- [ ] Model output appears in §11 daily report as a SHADOW column
      next to rule-based baselines (per §3.2)
- [ ] No single coin contributes > 50% of paper-shadow long-only PnL
      (sanity check, not skill check)
- [ ] Predictions saved to `crypto_predictions_latest.json` for
      downstream phases
- [ ] **No hard RankIC / ICIR threshold** — too noisy with N=5.
      Promotion to paper portfolio policy requires user sign-off
      based on accumulated observation, not metric threshold.

## §5. Phase Crypto-D-H: Acceptance Gates Only

Each phase below has acceptance gates frozen; full design is owed
when that phase is reached.

### 5.1 Phase Crypto-D: Paper OMS — Operational Mode (ACCEPTANCE-ONLY)

Per §−1, Phase D is the **operational mode**, not transitional.
Duration is open-ended, governed by user observation.

**Pre-Phase-D Spec Requirement (per cx system design review §4)**:
Before any Phase D code is written, CC writes a dedicated paper OMS
design doc at `plans/crypto-paper-oms-spec-YYYY-MM-DD.md`. cx's
proposed minimum state model (Account / Position / Order) and refusal
rules (data RED / stale quote / min_notional / lot precision / max
weight / depeg) are the floor — the dedicated spec will detail every
field, every refusal rule, every reconciliation invariant.

Reason: paper PnL accuracy is foundational. If account ledger
reconciliation is wrong, every downstream model evaluation
(including the supervised shadow model in §11 daily report) is fake.
This deserves its own design document, not a 1-paragraph summary
under Phase D.

Files (sketch):
- `paper/crypto_oms.py` — `LIVE_TRADING_ALLOWED = False`, `MAX_LEVERAGE = 1.0`
- `factors/crypto_sanitizer.py`
- `scripts/run_crypto_paper_trading.py`
- `scripts/crypto_daily_report.py` — see §11 for format

Acceptance:
- [ ] Paper orders generated every 4h
- [ ] `InstantSettlementModel` wired (NO `pending_target_weights`)
- [ ] 12-rule `CryptoSanitizer` from CC plan §11 applied
- [ ] Fee + slippage + funding cost accounted
- [ ] **Collateral ledger reconciliation prototype** (per Punch List C) — simulated against fake exchange statement, not real
- [ ] `LIVE_TRADING_ALLOWED = False` hard-coded; cannot toggle without user written sign-off
- [ ] `MAX_LEVERAGE = 1.0` hard-coded
- [ ] Daily report generated every day in §11 format
- [ ] User reads at least 1 week of reports before signing off Phase D
- [ ] **No "30-day to promotion" gate**: phase remains operational until
      user decides otherwise

### 5.2 Phase Crypto-E: Perp RiskGuard + Paper Funding-arb — Paper Only (ACCEPTANCE-ONLY)

Per cx system design review §5: RiskGuard gets 10 crypto-native
layers + Red/Yellow/Green action gating. RiskGuard's job is to
**reduce or block paper actions**, never to "improve alpha".

**10 RiskGuard Layers (FROZEN)**:

1. Stale primary exchange data
2. Fallback exchange unavailable
3. Cross-exchange close spread abnormal (> 25 bps per §2.7)
4. Funding-rate z-score extreme (|z| > 3 over 90-day rolling)
5. OI surge with price divergence (OI 24h change > 50% + price flat)
6. Market-wide liquidation cascade proxy (BTC perp 1h vol > 3σ + OI drop)
7. Stablecoin depeg basket risk (USDT/USDC > 50bps off peg)
8. Withdrawal halt / exchange incident flag (from event feed)
9. BTC dominance break (BTC.D 30d Z-score |> 2|)
10. High-volatility circuit breaker (BTC 1h realized vol > X%)

**Red / Yellow / Green Action Gating**:

```
RED:    no new paper orders, report-only mode
YELLOW: cap max weight, reduce turnover, refuse new perp /
        funding-arb paper exposure
GREEN:  normal paper policy
```

The status applies per-symbol and aggregate. Per-symbol RED means
that one symbol is frozen; aggregate RED means whole paper book
freezes.

**Minimum acceptance for trigger handling**:

- Every blocked action appears in §11 daily report (Sanitizer
  Rejections + RiskGuard Triggers sections)
- Every trigger row includes: timestamp / rule name / input values
  / action taken
- Repeated identical triggers within 1h are deduplicated (one row,
  count column) to avoid log spam

Per §−1, funding-arb in Phase E runs **paper only**, leverage = 1.0
cash-equivalent. No testnet, no margin, no liquidation simulator
exercising real risk.

Files:
- `risk/crypto_risk_guard.py`
- `models/crypto_derivative_features.py`

Acceptance:
- [ ] 8 RiskGuard rules from CC §11+cx convergence list
- [ ] **Negative-funding stress window backtest** (per Punch List C)
- [ ] **API outage simulation** (per Punch List C, simulated)
- [ ] **Withdrawal halt simulation** (per Punch List C, simulated)
- [ ] **Liquidation-buffer max-loss calculation** (per Punch List C,
      simulated against fake collateral ledger)
- [ ] Paper funding-arb strategy runs in shadow mode within Phase D
      daily report
- [ ] Funding-arb paper PnL appears in §11 report as a separate line
      so user can compare to spot-only PnL
- [ ] **`MAX_LEVERAGE = 1.0` enforced for funding-arb simulation too**:
      the strategy that historically wants 2-5x leverage runs at 1x
      under this constraint. User must explicitly sign off before
      any leverage > 1 is enabled, even in paper

### 5.3 Phase Crypto-F: Event/On-chain + Frontier Shadow (ACCEPTANCE-ONLY)

Files:
- `data/collectors/crypto_events.py`
- `factors/crypto_event_store.py` (or extend `EventStore` with asset_class)
- Kronos shadow predictor evaluation harness

Acceptance:
- [ ] LLM event extraction for ETF/listing/hack/depeg/unlock
- [ ] EventStore handles asset_class partitioning
- [ ] RD-Agent / Kronos / CryptoTrade / GraphSAGE evaluated in shadow
- [ ] No main-model inclusion until incremental IC > 0.02

### 5.4 Phase Crypto-G: Nautilus Prototype + Physical Migration (ACCEPTANCE-ONLY)

This phase splits into two sub-phases. Per cx review #8: **both
require explicit user written sign-off**, because both touch A-share
production structure.

**G.a: Physical Migration** (requires user written sign-off — moves
A-share production files into `ashare/` namespace, even though
behavior is preserved)

Acceptance:
- [ ] All §14.1 regression tests pass after migration
- [ ] One full A-share cron cycle with new namespace, byte-identical
      recommendation output vs pre-migration baseline
- [ ] User has reviewed the migration PR and signed off in writing
      (issue / commit message / chat — recorded in
      `plans/phase_g_signoff_log.md`)

**G.b: Nautilus Prototype** (paper backtest only)

Acceptance:
- [ ] Nautilus BTC/ETH spot backtest runs against historical parquet
- [ ] Internal vs Nautilus fills/costs comparison documented
- [ ] **Nautilus testnet/live connection NOT enabled** — gated to
      separate user written sign-off after observation

### 5.5 Phase Crypto-H: RL Allocation — Paper Only (ACCEPTANCE-ONLY)

Per §−1, RL stays in paper indefinitely.

Acceptance:
- [ ] Supervised baseline has stable user-observed paper history
- [ ] RL beats baseline after fees in walk-forward (paper only)
- [ ] Drawdown not worse than supervised baseline (paper only)
- [ ] RL output goes into §11 daily report as a shadow column
      next to supervised baseline output, user reads both

## §6. §14.1 Asset-implicit Audit: Row-by-row Implementation Strategy

Per cx review #2: **no physical file moves in Phase 0**. Phase 0c only
adds adapters next to existing files; original production files stay
in place. Physical move (`paper/oms.py` → `ashare/oms.py`, etc.) is
deferred to **Phase G**.

| # | A-share-implicit code path | File:line | Phase 0c (adapter) | Phase G (physical move) |
|---|---|---|---|---|
| 1 | ST stock filter | `data/build_tradable_mask.py:~120` | Add `UniverseFilter` Protocol in `core/`; A-share impl `AShareUniverseFilter` lives at new path but original file delegates via thin wrapper | Move `build_tradable_mask.py` → `ashare/build_tradable_mask.py` |
| 2 | Limit-up/down logic | `factors/candidate_sanitizer.py:~80` | Add `SanitizerRule` interface in `core/`; original `CandidateSanitizer` keeps existing API, gets rule-injection seam | Split into `ashare/sanitizer.py` + new `crypto/sanitizer.py` |
| 3 | One-price (一字板) | `factors/candidate_sanitizer.py:~95` | A-share-only rule registered with sanitizer; no `core/` exposure | Stays in `ashare/sanitizer.py` |
| 4 | SH/SZ/BJ prefix | `data/instrument_id.py` (likely; confirm in 0a) | Add `core.instrument.Symbol`; A-share keeps `qlib_code` helpers, adds `to_symbol()` adapter | Replace `qlib_code` with `Symbol.canonical()` |
| 5 | Qlib CN data | `models/feature_pipeline.py:~50` | Add `MarketDataSource` Protocol; A-share gets `QlibAShareSource` wrapper, original code delegates | Move into `ashare/data_source.py` |
| 6 | 100-share lot | `paper/oms.py:~200` | Add `Instrument.lot_size`; `paper/oms.py` reads from `Instrument` if present, falls back to hard-coded 100 (zero behavior change) | Remove hard-coded 100; `Instrument` lookup is the only path |
| 7 | T+1 fills | `paper/oms.py:pending_target_weights` | Add `SettlementModel` Protocol; `paper/oms.py` instantiates `T1SettlementModel` internally with byte-identical state-machine semantics | Move `paper/oms.py` → `ashare/oms.py`; `crypto/oms.py` uses `InstantSettlementModel` |
| 8 | Stamp tax | `backtest/optimizer_v2.py:~150` (confirm in 0a) | Add `TaxModel` Protocol; optimizer reads `AShareStampTax` instance, behavior identical | Move into `ashare/optimizer.py` |
| 9 | A-share stop-loss defaults | `risk/risk_guard.py:~configs` | Config file lifted to `config/ashare_risk_config.py`; `risk_guard.py` reads from it (was hard-coded) | Move into `ashare/risk_guard.py` |
| 10 | A-share take-profit defaults | `risk/risk_guard.py:~configs` | Same as #9 | Same as #9 |
| 11 | Trading calendar (9:30-15:00 + holidays) | `scheduler/jobs.py` + Qlib | Add `TradingCalendar` Protocol; `AShareCalendar` wraps Qlib calendar with byte-identical behavior | Move into `ashare/calendar.py` |
| 12 | Cron 9:20/14:30/22:00 | `scripts/install_crontab.py` | Group A-share jobs into `ASHARE_JOBS = [...]` constant; crypto jobs in separate `CRYPTO_JOBS` (per §−0.5 Layer 3) | Optionally split into `scripts/install_ashare_crontab.py` + `scripts/install_crypto_crontab.py` |
| 13 | `change_pct / 10` heuristic | `scheduler/jobs.py:~180` | Document the A-share-specific intent in comment; no behavior change | Move into `ashare/crossmarket.py` |
| 14 | RMB account currency | `paper/oms.py:~init` | Add `account_currency` field on `paper/oms.py` init, defaults to "CNY" (zero behavior change) | Move into `ashare/oms.py` config |
| 15 | A-share IVOL sign | `factors/factor_registry.py` | Phase B sign config: add per-asset-class sign registry; A-share sign unchanged | Phase B work, stays in factor_registry |
| 16 | A-share MAX sign | `factors/factor_registry.py` | Same as #15 | Same as #15 |
| 17 | A-share crash model | `models/crash_predictor.py` | Phase D: do not reuse A-share crash model for crypto; crypto trains its own | Stays in `ashare/crash_predictor.py` post-G |
| 18 | A-share regime overlay | `factors/regime_*.py` | Phase F: crypto has its own regime definitions (BTC trend) | Stays in `ashare/regime/*.py` post-G |
| 19 | A-share LLM event prompts | `factors/llm_event_extractor_v2.py` | Phase F: crypto LLM prompts use different event taxonomy (ETF/listing/hack/depeg/unlock) | Stays in `ashare/llm_event_extractor.py` post-G; crypto in `crypto/llm_event_extractor.py` |
| 20 | A-share `qlib_code` format | Multiple call sites | Phase G work; no 0c action | Replace with `Symbol.canonical()` system-wide |

**Phase 0c rule (per cx review #2)**: For each row 1-14, the Phase 0c
work is **adapter wiring only — no file move**. The original production
file (e.g. `paper/oms.py`) stays at its original path. The new
`core/` Protocol is consumed by a small adapter inside the existing
file, preserving behavior. A regression test fixes behavior to
byte-identical.

**Phase G rule**: physical file moves happen only after A-share cron
has been stable for ≥30 days on the Phase 0c adapter wiring, and only
after explicit user sign-off.

Each Phase 0c row gets:
1. A regression smoke test in `tests/test_ashare_unchanged_after_crypto_split.py`
   committed FIRST
2. An adapter PR that adds the Protocol consumer, leaves the file in
   place
3. 3-day A-share cron GREEN gate before next row's PR (per §−0.5
   Layer 5)
4. CI gate pass per §−0.5 Layer 4

## §6.5. Legacy Crypto Quarantine (Phase 0a PREREQUISITE)

Per cx code audit 2026-05-30: the current repo **already has
crypto/watchlist logic mixed into the A-share daily scheduler**. The
§14.1 audit (asset-class-implicit A-share logic) did NOT capture this,
because §14.1 catalogued A-share-flavored code that needs lifting to
`core/`, while this section catalogues pre-existing crypto code that
needs **quarantining** before new crypto pipeline is built.

This is the most important addition between r2 and r3 of this spec.
**Phase 0a cannot start until the legacy crypto quarantine plan is
agreed and the audit table below is committed.**

### Legacy Crypto Coupling Audit (verified 2026-05-30)

| # | Location | What it does | Impact on A-share | Quarantine action |
|---|---|---|---|---|
| L1 | `scheduler/jobs.py:11` | `from data.collectors.crypto import CryptoCollector` (module-level import) | A-share scheduler imports crypto code at startup; ccxt import failure crashes A-share daily run | Move import inside the legacy market-context function; guard with `try/except`; default-off feature flag |
| L2 | `scheduler/jobs.py:71` | `self.crypto_collector = CryptoCollector()` in `DailyPipeline.__init__` | A-share pipeline init touches Binance ccxt; network/proxy failure during init affects A-share startup | Lazy-init behind feature flag `LEGACY_MARKET_CONTEXT_ENABLED` (default `False`) |
| L3 | `scheduler/jobs.py:102, 112` | `fetch_realtime` / `fetch_daily` dispatch for crypto symbols | Used by the same dispatcher path A-share equities use; coupled state | Split into separate `_legacy_crypto_dispatch` only callable under flag |
| L4 | `scheduler/jobs.py:1812-1815` | Morning recommendation: BTC/ETH added to candidates list | Crypto enters candidate pool; if LGB candidates short, crypto can leak into A-share recommendations | Disable behind flag; when flag off, BTC/ETH never enters candidates |
| L5 | `scheduler/jobs.py:1868` | "Non-LGB candidates (crypto / gold) keep raw short_score as ranked" | Confirms crypto/gold get ranked in same pipeline as A-share stocks | Same flag-gated removal as L4 |
| L6 | `scheduler/jobs.py:1394-1416` | `_format_crypto_forecast` — BTC/ETH forecast in evening report | Affects evening report text only | Move to `legacy_market_context.py`; flag-gated |
| L7 | `scheduler/jobs.py:1933-1937, 1958` | `crypto_data = {}` then fetch + pass to report | Evening report fetch; network failure here affects evening report timing | Flag-gated lazy fetch with timeout |
| L8 | `config/watchlist.py:1-10` | `MARKET_STOCK`/`MARKET_CRYPTO`/`MARKET_GOLD` enum + `WATCHLIST_CRYPTO` | Conceptual coupling: same watchlist abstraction handles all 3 markets | Add `WATCHLIST_CRYPTO_LEGACY` marker; new crypto quant uses `config/crypto_universe.py` per §2.2 instead |
| L9 | `data/collectors/crypto.py` (entire file) | `CryptoCollector` class with Binance ccxt + AKShare fallback. Used by L1-L7. | This is the legacy collector. New spec specifies `data/collectors/crypto_market.py` per §2.1 | **Keep `crypto.py` as legacy-only**; new pipeline does NOT touch it; new code goes to `crypto_market.py` + `crypto_derivatives.py` |

### Quarantine Strategy (FROZEN, single PR — strict precondition to Phase 0a)

The quarantine is a **single PR** completed BEFORE Phase 0a starts.
Per user direction 2026-05-30: **default = FALSE**, not the
preserve-current-behavior default. Goal is to immediately remove
legacy crypto from the A-share runtime, then verify 3 days of GREEN
production before any new crypto code is written.

The PR makes ZERO model training change (LGB/XGB unaffected) and ZERO
A-share recommendation logic change. It changes the import graph,
feature-flags the legacy crypto report data, and adds tests.

Steps:

1. **Create `config/feature_flags.py`**:
   ```python
   import os

   # Per user direction 2026-05-30: default FALSE. Goal is to keep
   # A-share runtime clean of legacy crypto. Only set TRUE manually
   # if you need the old BTC/ETH evening-report background context
   # for a specific day.
   LEGACY_MARKET_CONTEXT_ENABLED = (
       os.environ.get("LEGACY_MARKET_CONTEXT_ENABLED", "false").lower()
       in ("true", "1", "yes")
   )
   ```

2. **Lazy-import pattern in `scheduler/jobs.py`**. Remove the
   module-level `from data.collectors.crypto import CryptoCollector`
   (line 11). Replace with a private accessor:

   ```python
   class DailyPipeline:
       def __init__(self, ...):
           ...
           self._crypto_collector = None   # lazy, see _get_crypto_collector

       def _get_crypto_collector(self):
           """Returns CryptoCollector only if legacy flag is on.
           Import happens here so module load does not touch ccxt."""
           from config.feature_flags import LEGACY_MARKET_CONTEXT_ENABLED
           if not LEGACY_MARKET_CONTEXT_ENABLED:
               return None
           if self._crypto_collector is None:
               from data.collectors.crypto import CryptoCollector
               self._crypto_collector = CryptoCollector()
           return self._crypto_collector
   ```

   The module-level import is **removed**, not commented out. With the
   flag off (default), `data/collectors/crypto.py` is never loaded,
   ccxt is never imported, no network call is risked at startup.

3. **Flag-gate each L1-L7 call site**. Every reference to
   `self.crypto_collector` becomes `self._get_crypto_collector()` and
   handles the `None` return:

   - L4 (line 1812-1815): if collector is None, skip the BTC/ETH
     candidate-add loop entirely. Candidates list does not gain any
     `MARKET_CRYPTO` entries.
   - L6 (line 1394): `_format_crypto_forecast` returns "crypto context
     disabled" stub text when collector is None.
   - L7 (line 1933): `crypto_data = {}` stays empty when collector is
     None; the empty dict is passed through and report renders without
     the crypto section.

4. **Add tests in `tests/test_legacy_crypto_quarantine.py`** (NEW):

   ```python
   def test_pipeline_init_does_not_import_crypto_when_flag_false(monkeypatch):
       """With flag FALSE (default), DailyPipeline import + init must
       not import data.collectors.crypto."""
       monkeypatch.setenv("LEGACY_MARKET_CONTEXT_ENABLED", "false")
       # Force reload of feature_flags to pick up env
       import importlib
       import config.feature_flags
       importlib.reload(config.feature_flags)
       # Remove crypto module from sys.modules if loaded
       import sys
       sys.modules.pop("data.collectors.crypto", None)

       from scheduler.jobs import DailyPipeline
       p = DailyPipeline(...)  # init must not trigger crypto import

       assert "data.collectors.crypto" not in sys.modules
       assert p._crypto_collector is None

   def test_candidates_contain_no_crypto_when_flag_false(monkeypatch):
       """With flag FALSE, morning-recommendation candidate pool must
       contain no MARKET_CRYPTO entries."""
       monkeypatch.setenv("LEGACY_MARKET_CONTEXT_ENABLED", "false")
       # ... run morning_recommendation path ...
       candidates = pipeline.build_candidates(...)
       for c in candidates:
           assert c.market_type != "crypto"
           assert c.code not in ("BTC/USDT", "ETH/USDT")

   def test_no_crypto_network_call_during_ashare_run(monkeypatch):
       """Per cx system design review punch list #2: with flag FALSE,
       the A-share daily run must make ZERO network call to crypto
       exchange APIs (ccxt / binance / okx / bybit endpoints).

       Mechanism: monkeypatch the urllib3 / requests HTTP adapter to
       record all outbound URLs during the daily run. Assert none
       match crypto exchange host patterns."""
       monkeypatch.setenv("LEGACY_MARKET_CONTEXT_ENABLED", "false")
       outbound_hosts = []

       import requests
       orig_send = requests.adapters.HTTPAdapter.send
       def trace_send(self, request, **kwargs):
           outbound_hosts.append(request.url)
           return orig_send(self, request, **kwargs)
       monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", trace_send)

       # Run the A-share morning recommendation pipeline end-to-end on
       # a fixed test date with fixtures.
       from scheduler.jobs import DailyPipeline
       p = DailyPipeline(...)
       p.run_morning_recommendation(target_date="2026-05-29")

       CRYPTO_HOST_PATTERNS = ("binance.com", "okx.com", "bybit.com",
                                "kraken.com", "coinbase.com")
       for url in outbound_hosts:
           assert not any(p in url for p in CRYPTO_HOST_PATTERNS), (
               f"A-share run attempted crypto network call to {url} "
               f"with LEGACY_MARKET_CONTEXT_ENABLED=false"
           )
   ```

5. **CI lint** (added to `scripts/check_namespace_isolation.py`,
   per §−0.5 Layer 1):

   - `crypto/` (future namespace) forbidden to import
     `data.collectors.crypto`, `scheduler.jobs`, `config.watchlist`
   - `ashare/` forbidden to import `data.collectors.crypto` (it's
     legacy market context, not A-share quant)
   - `scheduler/jobs.py` may not have a module-level
     `from data.collectors.crypto import` line (enforced by AST scan)

6. **Update `data/collectors/crypto.py` docstring** to mark it
   LEGACY-ONLY:

   ```python
   """LEGACY market-context crypto collector.

   This module is retained only for the LEGACY_MARKET_CONTEXT_ENABLED
   evening-report BTC/ETH forecast. It is NOT used by the new crypto
   quant pipeline (data/collectors/crypto_market.py +
   crypto_derivatives.py per cc-crypto-implementation-spec-2026-05-30.md
   §2.1, 2.5).

   New crypto code must NOT import this module.
   """
   ```

### What This Quarantine Does NOT Do

- Does NOT delete the legacy `data/collectors/crypto.py` (kept for
  scheduler compat under the flag)
- Does NOT remove BTC/ETH from `WATCHLIST_CRYPTO` (kept for the legacy
  path)
- Does NOT change A-share recommendation logic (LGB candidates and
  ranking unaffected)
- Does NOT touch model training (`models/lgb_*.py`, training scripts)

### Strict Step Order (FROZEN — no parallel work permitted)

Per user direction 2026-05-30: Phase 0a CANNOT start until legacy
quarantine has been deployed AND observed clean for 3 days. New crypto
pipeline code must not be written in parallel with legacy quarantine.
Reason: two crypto code paths simultaneously touching the scheduler
makes regression attribution impossible.

```
Step 1: Legacy Crypto Quarantine PR
        ↓
Step 2: Merge — flag defaults FALSE (legacy crypto OFF in production)
        ↓
Step 3: A-share daily cron observed GREEN for 3 consecutive days
        - byte-identical recommendations vs pre-quarantine baseline
          (excluding the legacy crypto evening-report background text)
        - paper OMS unchanged
        - no init-time errors from missing CryptoCollector
        ↓
Step 4: ONLY NOW — Phase 0a measurement spike + schema work begins
        against the clean baseline
```

No new **crypto pipeline** file (`data/collectors/crypto_market.py`,
`data/collectors/crypto_derivatives.py`, `core/*.py`,
`config/crypto_universe.py`, `config/crypto_storage.py`,
`scripts/crypto_*.py`, etc.) may be created during Steps 1-3.

**Quarantine-PR support files ARE allowed and required** during Step 1:

- `config/feature_flags.py` (new)
- `tests/test_legacy_crypto_quarantine.py` (new)
- `scripts/check_namespace_isolation.py` (new, lint)
- `scheduler/legacy_market_context.py` (new, optional — destination
  for moved L6 `_format_crypto_forecast` if extracting helps readability)

These exist to **shrink** legacy crypto's reach, not to extend it.
They are different from new-crypto-pipeline files.

### Quarantine PR Merge Gate (cx review r6, 2026-05-30)

Per cx feedback after the r5 quarantine PR review, the merge gate is
5 items. The pre-existing `tests/test_scheduler.py` 6 failures are
**NOT** in scope — they are unrelated historical test rot and tracked
as a separate follow-up PR (`tests/test_scheduler.py` 前五/前十/snapshot
/LLM fallback 漂移修复).

Quarantine PR merge gate:

1. `python scripts/check_namespace_isolation.py` returns 0
2. `pytest tests/test_legacy_crypto_quarantine.py` all green
   (init-level proof: import does not load legacy crypto, accessor
   returns None when flag off, no crypto host network call at init)
3. `pytest tests/test_legacy_crypto_quarantine_fullpath.py` all green
   (full-path proof: dispatcher / fetch helper / forecast formatter
   all behave per-spec when flag off; importlib metapath finder
   actively bans `data.collectors.crypto` import during a full
   runtime-path exercise — sanity-checked to actually block when
   flag on)
4. `pytest tests/test_scheduler.py` failure set documented as
   identical to baseline commit (`git stash` proof) — quarantine
   demonstrably did not introduce or fix any of the 6 pre-existing
   failures
5. Production cron runs for 3 consecutive days with
   `LEGACY_MARKET_CONTEXT_ENABLED=false` and A-share GREEN throughout

NOT required inside this PR:

- Fixing the 6 pre-existing `tests/test_scheduler.py` failures
  (separate follow-up PR; out of quarantine scope)
- New crypto pipeline code (`crypto_market.py`, `core/*.py`, etc. —
  those are Phase 0a onwards)

Phase 0a cannot start until all 5 gate items above + the post-flag
behavior commitment below pass.

### Phase 0a Pre-Gate Checklist

Phase 0a cannot start until all the following pass:

- [ ] §6.5 audit table reviewed and any cx amendments incorporated
- [ ] Quarantine PR merged per the 5-item merge gate above
- [ ] Post-Phase-0a default remains FALSE permanently. The flag may
      only be flipped TRUE manually on demand for one-off legacy
      evening-report needs; it is never the production default again.
- [ ] Only after all of the above, Phase 0a measurement spike +
      schema work begins against the clean baseline

The reason: without quarantine, the §−0.5 Layer 1 import-isolation
rule cannot be enforced (since scheduler already imports
`data.collectors.crypto` at module level). Quarantine creates the
clean baseline that isolation rules then defend.

## §7. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| CCXT rate limit hits during backfill | High | Medium | Throttle + exponential backoff in `fetch_historical` |
| Binance API access blocked from China | Medium | High | Phase 0 spike measures access; OKX/Bybit fallback in config |
| §14.1 refactor breaks A-share cron silently | Medium | High | 20 regression tests in CI; A-share cron monitored for 3 days before each Phase 0 merge |
| Funding history depth shorter than 1y | Medium | Medium | Phase 0 spike measures; backfill what's available, document limit |
| Disk space for 5y 1h × top-30 universe | Low | Medium | Phase A is 5 symbols only; top-30 deferred to Phase B with explicit sizing |
| Universe expansion silently survivorship-biases | Medium | High | `delisted_coins.parquet` discipline enforced at `eligible_at` |
| IVOL/MAX sign-flip not caught | Low | High | Phase B acceptance requires explicit sign validation; no copy-paste from A-share |
| paper/oms.py refactor breaks T+1 logic | Medium | High | `T1SettlementModel` is identical semantics to existing pending_target_weights; regression test on existing A-share paper history |
| Funding-arb live trading attempted before evidence | Low | Critical | Hard gate: §5.2 acceptance requires all 4 stress simulations; no live capital without `[validated-on-local]` Sharpe |

## §8. Rollback Plan

Per cx review #3: **automatic `git reset --hard` is FORBIDDEN**. It
can destroy uncommitted work or parallel session state. All rollback
paths use feature flags, cron disable, or `git revert` (new commit,
preserves history).

### Rollback Tier 1: Feature Flag (fastest, no git operation)

For any §14.1 adapter that wraps a `core/` Protocol path, a feature
flag controls whether the adapter is active:

```python
# config/feature_flags.py
ENABLE_CORE_SETTLEMENT_MODEL = True   # row #7 adapter
ENABLE_CORE_INSTRUMENT_LOT = True     # row #6 adapter
# ... one per §14.1 row
```

Procedure if adapter causes regression:

1. Manual: edit `config/feature_flags.py`, set offending flag to `False`
2. Restart relevant A-share cron job
3. A-share reverts to pre-adapter code path (the original behavior is
   preserved as the fallback branch in each adapter)
4. Investigate offline; no git history change needed

This is the **default and preferred** rollback path for §14.1 work.

### Rollback Tier 2: Disable Crypto Cron (when crypto-side bug suspected)

If a crypto cron job is misbehaving (rate limit, exception loop, data
corruption in crypto storage):

```bash
python scripts/disable_crypto_cron.py
# Removes CRYPTO_JOBS entries from crontab. A-share untouched.
# Idempotent. Logs what was removed.
```

Procedure:

1. Run disable script
2. A-share cron continues normally (zero dependency on crypto, per §−0.5 Layer 3)
3. Investigate crypto issue offline
4. Re-enable crypto cron entries one job at a time after fix

### Rollback Tier 3: `git revert` (when code commit caused regression)

If the regression is rooted in a specific commit:

```bash
git log --oneline -10        # identify offending commit SHA
git revert <sha>             # creates NEW commit that undoes <sha>
git push                     # deploy the revert
```

Procedure:

1. Identify offending commit from monitoring (per §−0.5 Layer 6)
2. `git revert <sha>` — creates new commit, history preserved
3. CI runs full test suite + A-share regression
4. If green, deploy; if red, investigate further (do NOT use
   `--strategy=ours` or `git reset`)
5. Post-mortem in `plans/postmortems/<date>-<summary>.md`

### Explicit Prohibitions

The following operations are forbidden in any rollback procedure:

- `git reset --hard <sha>` — destroys uncommitted work
- `git push --force` — destroys upstream history that collaborators may have
- `git checkout <sha> .` — overwrites working tree silently
- Editing crontab to add/remove A-share entries during a crypto
  rollback (A-share entries are never touched during crypto rollback)
- Disabling A-share regression tests "temporarily" to merge a fix
- Reverting via direct file edit instead of `git revert`

### Failure Domain Independence

A-share rollback never requires crypto cooperation:

- A-share cron entries are not in `CRYPTO_JOBS`
- A-share imports never reference `crypto/`
- A-share parquet/jsonl paths are disjoint from crypto storage
- A-share feature flags are separate from crypto feature flags

If crypto is completely broken, A-share continues producing
recommendations normally. This is the structural guarantee §−0.5
provides; rollback procedures rely on it.

## §9. Open Questions For CX

### CX Review Round 1 Resolution (2026-05-30 morning)

CX reviewed the prior draft of this spec and raised 9 items. All
resolved in revision r1:

| CX r1 item | Resolution in this spec |
|---|---|
| 1. Phase 0 too heavy | Split into 0a/0b/0c per §1 |
| 2. §14.1 rows 1-14 Phase 0 refactor too risky | §6 table — adapter-only in 0c, physical moves deferred to Phase G |
| 3. `git reset --hard` auto-rollback | §8 forbids; uses feature flag / cron disable / `git revert` |
| 4. Cron `5,35 * * * *` duplicate + closed-bar buffer | §2.6 fixed to `5 * * * *`; `CLOSED_BUFFER_SEC=120` |
| 5. Binance-only Phase A | §2.2 adds OKX/Bybit fallback with health check |
| 6. `Symbol.filesystem_safe()` missing venue | §1.4 `canonical()` and `filesystem_safe()` both include venue + instrument_class |
| 7. Tests using testnet | §2.8 split into unit (fixture) + integration (public mainnet read-only, no keys) |
| 8. Phase B "prediction" concept error | §3.2 rule-based baselines only; "prediction" deferred to Phase C |
| 9. "User not read → stop retrain" unrealistic | §11 Reading Discipline rewritten: reports + retrain run unconditionally; only paper-policy promotion gates on user sign-off |

### Pre-Phase-0a Doc-Consistency Round (2026-05-30 soak, gate items 4 + 5)

During the 3-day production soak of the Quarantine PR (branch `crypto`,
2026-06-01 → 2026-06-03), all doc-consistency work required to start
Phase 0a was completed:

| Gate item | File | Resolution |
|---|---|---|
| #4. CC plan self-corrections (8 items: capital ladder → state-gated / DeFi out of pipeline / funding-arb framing / migration Phase G+ / Nautilus rationale / Frontier shadow-only / funding numbers retag / §14.1 SHA-pin) | `plans/cc-crypto-quant-integration-plan-2026-05-30.md` | New §0.5 "自纠错" section authoritative override of §1-§16; §14.1 gains `AUDIT-FROZEN-AT: acaedfa` header |
| #5.A CX roadmap Phase 0 duration 1-2d → 1-2w | `plans/crypto-quant-roadmap-2026-05-30.md` §"Phase Crypto-0" | Duration revised, rationale: scope grew over convergence |
| #5.B CC §14.1 SHA-pin | (overlap with #4 item 8) | Header added |
| #5.C Funding-arb 6 minimum-evidence items into Phase D/E acceptance | `plans/crypto-quant-roadmap-2026-05-30.md` §"Phase Crypto-D" + §"Phase Crypto-E" | Promoted from Addendum to acceptance gates; collateral ledger reconciliation prototype required at Phase D |
| #5.D CX review-of-CC self-retag | `plans/cx-review-cc-crypto-quant-integration-plan-2026-05-30.md` §6 | Funding arb Sharpe 2-3 / 92% positive / $10M capacity all tagged `[paper-reported]` or `[exchange-dashboard]` |

After soak completes GREEN and quarantine merges, Phase 0a may start
without further doc-consistency blocker.

### CX System Design Review Round 5 (2026-05-30 night)

cx delivered a system-design review (`plans/cx-crypto-system-design-review-2026-05-30.md`)
scoring 8.0/10. 7/7 areas accepted + 6 spec amendments applied:

| Item | Resolution |
|---|---|
| §1 Isolation designed not implemented | Already in §6.5 (legacy quarantine PR + 3-day GREEN); reaffirmed |
| §2 Cross-source sanity | §2.7 NEW — cross-source section with BTC/ETH/SOL primary-vs-fallback close spread / funding sign agreement / latency; YELLOW on disagreement |
| §3 Phase A universe too small for alpha | §4.2 — Phase C explicitly shadow-only no IC threshold; universe expansion gated by 7 pre-conditions (delisted handling / depeg filter / scam filter / volume floor / listed-days / exchange coverage / survivorship test) + user sign-off |
| §4 Paper OMS needs own spec | §5.1 — CC commits to write `plans/crypto-paper-oms-spec-YYYY-MM-DD.md` before Phase D starts |
| §5 RiskGuard crypto-native | §5.2 — 10 layers (stale primary / fallback unavail / cross-exchange spread / funding z / OI surge / liquidation cascade / depeg basket / withdrawal halt / BTC.D break / vol circuit breaker) + RYG action gating |
| §6 Daily report observation-focused | §11 — added What Changed Since Yesterday / Paper vs Shadow / Data Trust three sections |
| §7 Research freeze | §10 — explicit frozen list (Kronos/RD-Agent/GraphSAGE/RL/DeFi/Nautilus/live/testnet) + allowed list (OHLCV/funding/health/baselines/shadow) through Phase D |
| Punch #2 no-crypto-network-call test | §6.5 — new test_no_crypto_network_call_during_ashare_run patches requests adapter, asserts no binance/okx/bybit hosts during A-share daily run |

Plus 6 amendments from cx's r5 follow-up:

| cx r5 amendment | Resolution |
|---|---|
| 1. CRYPTO_STORAGE_ROOT hardcoded | §−0.4 — env-overridable: `Path(os.environ.get("CRYPTO_STORAGE_ROOT", "/Volumes/DATA/crypto"))` |
| 2. run_crypto_job vs run_network_job duplication | §−0.4 — `run_crypto_job.py` is the single entry; internally delegates to `run_network_job.py --network crypto_global`; direct `run_network_job.py` use in CRYPTO_JOBS forbidden |
| 3. spike failure opaque | §−0.3 — spike writes phase0_measurements.json even on failure with `step / reason / error_class / traceback`; 9 distinct failure reasons enumerated |
| 4. Quarantine default FALSE | already correct (§6.5) — confirmed |
| 5. "No new crypto file" overly strict | §6.5 — clarified: quarantine-PR support files (config/feature_flags.py, tests/, lint scripts, scheduler/legacy_market_context.py) ARE required + allowed |
| 6. "No RED-by-cross-source" impossible wording | §2.7 — fixed: "No primary RED events during soak; cross-source YELLOW allowed if not persistent (>50% window)" |

### User Direction + CX Review Round 4 (2026-05-30 late evening)

After r3 sign-off path was set, the user issued three additional hard
directives and cx flagged two refinements:

| Item | Source | Resolution |
|---|---|---|
| All crypto network traffic via ssproxy | User direction | §−0.3 NEW — `crypto_global` profile aliases existing `global`; cron uses `run_network_job.py --network crypto_global`; collector reads `HTTPS_PROXY` from env, asserts presence (no fake `_load_proxy()` profile-loader) |
| All crypto data on `/Volumes/DATA` external drive | User direction | §−0.4 NEW — `CRYPTO_STORAGE_ROOT = Path("/Volumes/DATA/crypto")` single source of truth in `config/crypto_storage.py`; `ensure_mounted_and_writable()` pre-flight every script; repo path `data/storage/crypto/` forbidden by CI lint |
| Legacy quarantine flag must default FALSE (not TRUE) | User direction | §6.5 — `LEGACY_MARKET_CONTEXT_ENABLED` defaults FALSE; lazy-import pattern in `scheduler/jobs.py` replaces module-level import; tests verify A-share recommendations contain no `MARKET_CRYPTO` and `data.collectors.crypto` not in `sys.modules` |
| Strict step order — no parallel new-crypto work | User direction | §6.5 + §10 — Step 1 quarantine → Step 2 flag FALSE → Step 3 3-day GREEN → Step 4 Phase 0a; new crypto files forbidden during Steps 1-3 |
| Lint must forbid legacy import paths from new crypto code | cx refinement | §−0.5 Layer 1 — added rules: `crypto/` may not import `data.collectors.crypto`, `scheduler.jobs`, `config.watchlist`; AST rule enforces no module-level legacy import in `scheduler/jobs.py` |
| Post-Phase-0a default must remain FALSE | cx refinement | §6.5 — explicit rule: flag may only be flipped TRUE manually on demand for one-off legacy evening-report needs, never the production default again |

### CX Review Round 3 Resolution (2026-05-30 evening)

CX reviewed revision r2 and raised 5 must-fix + 3 should-fix items
plus a code audit that surfaced legacy crypto coupling. All resolved
in revision r3:

| CX r3 item | Resolution |
|---|---|
| 1. Daily report cron `08:30 UTC` mislabeled | §11 cron comment now reads "08:30 Asia/Shanghai local = 00:30 UTC" |
| 2. Phase A 3-5d vs 7d GREEN contradiction | §2 split into Phase A.impl (3-5d) + Phase A.soak (7d); §2.9 has separate acceptance gates |
| 3. `fetch_historical` no end_ts filter | §2.1 — added `df = df[(timestamp_utc >= start) & (timestamp_utc < end)]` |
| 4. `_load_proxy` only catches ImportError | §2.1 — added `strict=True` default with explicit fail-fast on KeyError/AttributeError/ValueError + missing profile |
| 5. A-share health byte-identical includes mtime | §−0.5 Layer 6 — explicitly excludes mtime; compares normalized business outputs only (recommendations / OMS state / PnL / cron pattern) |
| 6. (should) `quote_volume` estimation not flagged | §1.2 schema + §2.1 collector — `quote_volume_estimated: bool` column added, default True for Phase A |
| 7. (should) Fallback unreachable should be YELLOW | §2.7 — explicit rule emphasized: fallback unreachable = YELLOW (operational but degraded resilience) |
| 8. (should) Boot TZ assertion needs tzname print | §2.6 — assertion now prints `time.tzname` + offset + abbreviations before raise |
| **Code audit: legacy crypto in scheduler** | **§6.5 NEW** — Legacy Crypto Quarantine audit (9 file:line locations) + Phase 0a pre-gate requirement |

### CX Review Round 2 Resolution (2026-05-30 afternoon)

CX reviewed revision r1 and raised 8 must-fix + 3 suggested items. All
resolved in revision r2:

| CX r2 item | Resolution in this spec |
|---|---|
| 1. cron UTC vs Asia/Shanghai mismatch (P0) | §2.6 — all schedules expressed in Asia/Shanghai local time with UTC event noted in comments; boot-time TZ assertion in `install_crontab.py` |
| 2. Code snippet vs CLOSED_BUFFER_SEC contradiction | §2.1 — `_is_closed_with_buffer()` is the single source of truth; both `fetch_recent` and `fetch_historical` call it |
| 3. `fetch_historical` hardcodes `is_closed_bar=True` | §2.1 — `fetch_historical` now uses `_is_closed_with_buffer()` for tail safety |
| 4. funding/OI cron missing `--network crypto_global` | §2.6 — all crypto cron commands carry `--network crypto_global` |
| 5. Phase A 30d OI impossible in 3-5 days | §2.9 — Phase A acceptance is **3 days of OI samples**; 30d moved to Phase B with `fetch_open_interest_history` |
| 6. funding `limit=1000` no pagination | §2.5 — `fetch_funding_history` paginates via cursor advance |
| 7. Phase 0a "zero code change" contradiction | §1.8.a — clarified to "zero A-share/core production code change; new isolated read-only scripts allowed" |
| 8. Phase G user sign-off front/back inconsistency | §5.4 — G.a physical migration ALSO requires user written sign-off (recorded in `phase_g_signoff_log.md`) |
| 9. Health JSON `"GREEN" \| "YELLOW" \| "RED"` not valid JSON | §2.7 — single string value + enum table |
| 10. Phase C N=5 RankIC threshold too noisy | §4.2 — Phase C redefined as shadow model, no hard IC/ICIR threshold; promotion gated to user sign-off |
| 11. "What CC Would Recommend Differently If Live" suggests live paths | §11 — renamed "Blocked Actions Under Paper-only Constraint", veto-only, no alternatives |

### Remaining Open Questions

Before Phase 0 starts, CC needs CX sign-off on:

1. **Phase 0 duration**: confirm 1-2 weeks (CC) over 1-2 days (CX
   roadmap). If CX prefers shorter, which deliverables can drop?

2. **§14.1 SHA-pinning**: confirm `AUDIT-FROZEN-AT` header is the
   right form. CX previously proposed "Phase Crypto-0 acceptance
   checklist" wording — is SHA pin too heavy or just right?

3. **Phase A universe**: 5 majors on Binance only. CX roadmap also
   mentioned OKX/Bybit fallback. Phase A or Phase B?

4. **Initial backfill depth**: 5y 1h, or shorter? Disk + rate-limit
   tradeoff. CC default: 5y 1h, all available 4h/1d.

5. **OI sampling cadence**: CC proposes 15-min via cron polling. CX
   roadmap §"Cron Plan" mentioned `*/30 1h, */4 4h` — does CX
   prefer 30-min OI?

6. **Funding-arb evidence list**: CX Addendum §1 lists 6 prerequisites
   for live funding arb. CC §5.2 acceptance commits to these.
   Confirm the list is the complete gate (no item missing).

7. **`core/` directory creation timing**: Phase 0 creates `core/`
   with 5 Protocol files. Does CX want this in a separate PR before
   §14.1 refactor PRs, or all-in-one Phase 0 PR?

8. **CCXT version pin**: latest stable is `ccxt==4.x`. CC defaults
   to this. CX preference?

9. **Test framework**: pytest with existing `tests/` layout. CX
   confirm?

10. **Phase 0 PR size discipline**: CC proposes splitting Phase 0
    into:
    - PR 1: `core/` Protocols + 2 impls each (no behavior change)
    - PR 2: §14.1 rows 1-7 refactor (A-share regression tests)
    - PR 3: §14.1 rows 8-14 refactor
    - PR 4: `crypto-data-contract.md` + `numeric_claims_audit.md`
    - PR 5: Phase 0 spike script + measurement results
    Each PR < 500 LOC. CX confirm structure?

## §11. Daily Paper Trading Report (per §−1)

Per user constraint §−1, a daily report is the operational deliverable
of Phase D onwards. File path (per §−0.4 external storage):

```
/Volumes/DATA/crypto/reports/daily/{YYYY-MM-DD}_paper_report.md
```

(Resolved via `CRYPTO_STORAGE_ROOT / "reports" / "daily" / ...` in
`config/crypto_storage.py`.)

Cron entry (added in Phase D):

```python
CronJob(
    name="crypto_daily_report",
    # 08:30 Asia/Shanghai local = 00:30 UTC. Fires after UTC 00:00
    # daily roll + 30min buffer, aligned with user morning routine.
    # All other CRYPTO_JOBS use same local-time convention (see §2.6).
    schedule="30 8 * * *",
    command="scripts/crypto_daily_report.py --network crypto_global",
    timeout_sec=300,
    enforce_deps=True,
    dep_wait_seconds=1800,
)
```

### Report Format (FROZEN)

```markdown
# Crypto Paper Trading Report — {YYYY-MM-DD}

LIVE_TRADING_ALLOWED: False
MAX_LEVERAGE: 1.0
Mode: PAPER-ONLY (user constraint 2026-05-30)

## What Changed Since Yesterday
{Per cx system design review §6: report must surface deltas, not just
snapshots. Each subsection is one line per change; "no change" rows
are omitted to keep the report scannable.}

- Position changes: {SYM} {old%} → {new%}, ...
- New signals: {SYM} moved from {old_decision} to {new_decision}
- Removed signals: {SYM} rejected by {sanitizer_rule / risk_rule}
- Data health: {exchange} {old_status} → {new_status}
- RiskGuard: {rule_name} triggered for {SYM}
- Model: {active_version_change_or_none}; shadow model {ver} differed on {SYM}

## Paper vs Shadow
{Per cx system design review §6: show user the divergence between
what paper portfolio did and what shadow models would have done. This
is the observation surface for promotion decisions.}

Paper Policy (active):
  {rule baseline name OR active model version}

Shadow Candidates:
  supervised model (XGB/LGB walk-forward latest)
  funding-arb shadow (if Phase E started)
  event/on-chain shadow (if Phase F started)

Decision Comparison:
  Paper portfolio did: {action summary}
  Shadow {name} would have done: {action summary}
  Divergence: {symbol-level diff list}

## Data Trust
{Per cx system design review §6: dedicated section so user can build
intuition about data reliability over time.}

- Primary exchange: {exchange_name} {status}
- Fallback exchanges: {okx_status}, {bybit_status}
- Latest closed 1h bar age (across primary universe): {min}-{max} min
- Gap rate 30d: {percent}
- Cross-source spread (BTC primary vs fallback): {bps} bps
- Funding sign agreement (primary vs fallback): {true/false}

## Portfolio Snapshot
- Account equity (notional USDT): {value}
- 24h PnL: {value} ({percent}%)
- 7d PnL: {value}
- 30d PnL: {value}
- Inception PnL: {value}
- Sharpe (since inception, fee-adjusted): {value} [validated-on-local]
- Max drawdown since inception: {value}

## Positions
| Symbol | Weight | Notional | Unrealized PnL | Entry Time |
|---|---|---|---|---|
| BTC/USDT | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |

## Today's Signals
| Symbol | Predicted ret_24h | Decile | Decision | Reason |
|---|---|---|---|---|
| BTC/USDT | +0.024 | 8 | BUY | Top decile, sanitizer pass |
| ... | ... | ... | ... | ... |

## Sanitizer Rejections (24h)
| Symbol | Rule Triggered | Detail |
|---|---|---|
| {SYM} | depeg | USDT off-peg 0.3% |
| ... | ... | ... |

## RiskGuard Triggers (24h)
| Rule | Symbol | Action | Detail |
|---|---|---|---|
| funding_extreme | BTC perp | reduce_weight 0.5x | funding > 0.05% |
| ... | ... | ... | ... |

## Funding-arb Shadow (Phase E onwards)
- Paper carry collected today: {value}
- Margin used (simulated): {value} / leverage 1.0
- Stress test: negative-funding window PnL: {value}

## Model Confidence (per coin)
| Symbol | Predict | Confidence | Last retrain |
|---|---|---|---|
| BTC/USDT | +0.024 | 0.71 | 2026-05-27 |
| ... | ... | ... | ... |

## Data Health
- All bars closed and fresh: GREEN/YELLOW/RED
- Gap rate 30d: {percent}
- Funding/OI age: {seconds}

## Anomalies
{Free-text section. Any exchange API errors, depeg events, news flags,
or strategy edge cases worth flagging to user.}

## Blocked Actions Under Paper-only Constraint
{Per cx review #11: this section ONLY lists actions blocked by the
§−1 paper-only constraint. It does NOT propose alternatives, recommend
leverage, or suggest live execution paths. Pure veto log.

Example entries:
- BLOCKED: increase position size beyond 1.0x notional (leverage
  forbidden per §−1)
- BLOCKED: connect to exchange API with live keys (no keys configured)
- BLOCKED: route order through Nautilus live engine (Phase G+
  user-sign-off-required)

The section exists for audit, not for inspiration.}
```

### Reading Discipline (FROZEN, per cx review #9)

System cannot reliably detect whether the user has read a report. So
reading discipline relies on explicit user actions, not inference.

- Reports are generated every day, unconditionally. No "user not read
  → stop generating" path.
- Model retrain runs on its weekly schedule, unconditionally. Retrain
  output, however, **only affects the shadow output column** in the
  daily report. It does NOT change the live paper-portfolio policy
  until the user signs off explicitly.
- Concretely: when retrain produces a new model, the daily report's
  "Today's Signals" table uses the OLD model; a new "Shadow Signals
  (post-retrain)" table shows what the NEW model would produce. User
  reads both, then either:
  - Manually edits `config/crypto_active_model_version.py` to promote
    the new model, OR
  - Lets retrain continue running in shadow indefinitely.
- Promotion paper → testnet → live ALWAYS requires explicit written
  user sign-off (per §−1). No time-based or metric-based auto-promotion.
- After 30 days of paper trading the system does NOT prompt the user.
  CC waits for the user to ask.

## §10. Sign-off Statement

When this document is signed off by CX:

1. CC commits to **DETAIL-FROZEN** items as written
2. CC may revise **DETAIL-STRUCTURAL** within the acceptance criteria
3. CC writes a new spec doc before each **DETAIL-ACCEPTANCE-ONLY**
   phase
4. CC plan §"Self-corrections Required" (8 rows from cc-review)
   merged before Phase 0a starts
5. CX roadmap Implementation Punch List items A/B/C/D resolved
   before Phase 0a starts

### Phase-by-phase Sign-off

Per cx review #1 and §−0.5 Layer 5, sign-off is incremental:

- **Pre-Phase-0a Prerequisites** (in order, no parallel work):
  1. **Legacy Crypto Quarantine PR** (§6.5) — single PR, flag
     defaults FALSE, lazy import, tests + CI lint
  2. **3 days of A-share GREEN** with `LEGACY_MARKET_CONTEXT_ENABLED=false`
     in production (Phase 0a cannot start before day 4)
  3. **Infra readiness**:
     - `crypto_global` network profile defined in `config/network_profiles.py` per §−0.3
     - `/Volumes/DATA` mounted, writable, `crypto/` subdirs created per §−0.4
     - `config/feature_flags.py` and `config/crypto_storage.py` exist
- **Phase 0a** (docs/schema/spike/audit): sign-off when §1.8.a passes.
  Zero A-share/core production code change; new isolated read-only
  scripts allowed. Lowest-risk start.
- **Phase 0b** (`core/` Protocol + A-share wrapper, greenfield): sign-off
  when §1.8.b passes. No A-share production code touched.
- **Phase 0c** (per-row adapter wiring + regression tests): sign-off
  when §1.8.c passes. Each §14.1 row is its own PR with 3-day A-share
  GREEN gate.
- **Phase A** (crypto OHLCV + funding/OI foundation): sign-off when
  §2.9 passes. Crypto-side only; A-share continues uninterrupted.
- **Phases B-H**: each has its own design doc + acceptance gate; no
  blanket pre-approval.

### Hard Gates That Cannot Be Bypassed

- **§−1 user paper-only constraint**: never bypassed without written
  user sign-off (not even by CX or this spec author)
- **§−0.5 A-share isolation guarantee**: any PR violating Layer 1
  (import isolation) or Layer 4 (CI gate) is non-mergeable; any
  deploy violating Layer 5 (3-day GREEN gate) is rolled back
- **§14.1 byte-identical regression**: failure halts the phase, no
  "ship and fix" allowed

If CX rejects any FROZEN item: discuss before that phase starts. If
CX accepts with revisions: the revisions land in this spec, not in
implementation surprises.

### Research Freeze (per cx system design review §7)

Through Pre-Phase-0a, Phase 0a/0b/0c, A, B, C — the following are
**explicitly frozen**:

- ❌ Kronos shadow predictor (deferred to Phase F earliest)
- ❌ RD-Agent factor research loops (Phase F earliest)
- ❌ GraphSAGE / on-chain GNN (Phase F earliest)
- ❌ RL allocation (Phase H earliest)
- ❌ DeFi yield mainline (permanently out-of-scope per CC §3-4 self-correction)
- ❌ NautilusTrader (Phase G prototype only, after process-model conflict acknowledged)
- ❌ Any live exchange key configuration (gated to §−1 user written sign-off)
- ❌ Any testnet connection (cx review r1 #7 — deferred to Phase G+ with sign-off)
- ❌ Any leverage simulation > 1.0x (§−1 hard rule, even paper)

Allowed through Phase D:

- ✅ OHLCV via CCXT public REST
- ✅ Funding / OI via CCXT public REST
- ✅ Data health monitoring + cross-source sanity
- ✅ Rule-based baselines (4 strategies per §3.2)
- ✅ Shadow XGB/LGB if Phase C starts (no production policy effect)
- ✅ Daily paper report

Reason: the next milestone is **trust** (paper runs clean for weeks
without touching A-share), not alpha. Allowing frontier research in
parallel would expand scope faster than the trust foundation can
support.

This document is the contract.
