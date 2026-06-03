# Crypto Data Contract (single source of truth)

Date: 2026-05-31
Status: **light draft (pre-Phase-0a)** — schema column + storage path
authority for the future crypto pipeline. Locking these here so Phase
0a code can `import` from a single document instead of bouncing
between `cc-crypto-implementation-spec` §1.2 / §2.1 / §2.5 / §−0.4.

## Authority

This document supersedes any scattered schema definitions in:

- `cc-crypto-implementation-spec-2026-05-30.md` (§1.2 / §2.1 / §2.5 / §2.7 / §−0.4)
- `crypto-quant-roadmap-2026-05-30.md` (§Data Plan / §Phase A Data: Spot OHLCV)
- `crypto-quant-literature-and-engineering-review-2026-05-30.md` (§Data Source Review)

When this doc disagrees with any other, **this doc wins**. Phase 0a
sign-off freezes this file at its sign-off SHA via
`AUDIT-FROZEN-AT: <sha>` header (mirroring §14.1 audit pattern).

## §1. Storage Root

Per user direction 2026-05-30: all crypto persistent data lives on
external volume `/Volumes/DATA/crypto/`, **not** in repo tree.

```python
# config/crypto_storage.py (to be created in Phase 0b)
import os
from pathlib import Path

CRYPTO_STORAGE_ROOT = Path(
    os.environ.get("CRYPTO_STORAGE_ROOT", "/Volumes/DATA/crypto")
)

REQUIRED_SUBDIRS = [
    "raw/ohlcv", "raw/funding", "raw/open_interest",
    "features", "predictions", "health", "audit", "paper",
    "reports/daily",
]
```

Every crypto entrypoint MUST call
`config.crypto_storage.ensure_mounted_and_writable()` as its first
action. Silent fallback to repo path is **forbidden** — CI lint
rejects `data/storage/crypto/` references.

## §1.5. Network Profile (ssproxy MANDATORY)

Per user direction 2026-05-30: **all crypto data fetches MUST traverse
`ssproxy`**. Direct egress to exchange / chain endpoints from the
mainland network is not allowed. This is a network-policy invariant,
not a "best effort" preference.

Enforcement (deliverable in Phase 0b):

1. **Cron wrapper**: every crypto job in `scripts/install_crontab.py`
   is wired through `run_network_job.py --network crypto`. The
   `crypto` profile (added 2026-06-03 in
   `scripts/run_network_job.py:apply_profile`) does:
   - `_ensure_proxy()` — verifies ssproxy is listening; starts it
     if not; aborts with `sys.exit(2)` if proxy doesn't come up
     within 10s
   - `_set_proxy(env)` — sets http_proxy / https_proxy / HTTP_PROXY
     / HTTPS_PROXY / ALL_PROXY in the child env
   - `env["CRYPTO_NETWORK_ACTIVE"] = "crypto"`
   - `env["CRYPTO_SSPROXY_VERIFIED"] = "1"`

   Failure → exit code 2 (distinct from generic profile errors); the
   cron `run_with_status.py` wrapper marks the job failed.  A-share
   cron continues unaffected per the isolation invariant.

2. **Collector contract**: every `data/collectors/crypto_*.py`
   module's entrypoint calls
   `config.crypto_network.assert_proxy_active()` as its first
   action. The function checks BOTH `CRYPTO_NETWORK_ACTIVE=="crypto"`
   AND `CRYPTO_SSPROXY_VERIFIED` truthy; if either is missing or
   the network value is wrong, the collector raises
   `CryptoProxyNotActiveError` immediately — no silent fallback, no
   partial fetch.

3. **Tests**: a namespace-isolation lint test asserts that no crypto
   module performs a direct `requests`/`urllib`/`ccxt` call without
   the wrapper env-check in front of it.

Forbidden:
- `requests.get(...exchange-url...)` directly in any crypto module
- `ccxt.binance().fetch_ohlcv(...)` without the wrapper's env check
- Reading from network at module-import time (must be lazy)

A-share modules are not affected by this section — they continue
using the existing `--network domestic` profile.

## §2. Symbol Identifier

Single canonical form across the entire system:

```
{venue}__{base}_{quote}__{instrument_class}
```

Examples:

| Symbol | canonical |
|---|---|
| Binance BTC/USDT spot | `binance__btc_usdt__spot` |
| Binance BTC/USDT perp | `binance__btc_usdt__perp` |
| OKX BTC/USDT spot | `okx__btc_usdt__spot` |

Rules:

- venue lowercase (`binance` / `okx` / `bybit`)
- base / quote **lowercase** (`btc` / `usdt` / `eth`) — chosen 2026-06-03
  per cx code review round 4 P2 #3: implementation was already
  lowercase and case-insensitive filesystems (macOS default) would
  silently merge case-different paths. Anyone reading the canonical
  out of a log MUST lowercase before constructing a parquet path /
  dict key / report column. The `to_canonical_symbol` and
  `to_canonical_perp_symbol` helpers do this automatically.
- instrument_class enum: `spot` / `perp` / `future` / `option`
- filesystem-safe by construction (no `/`, no `:`, no `@`)
- used as parquet partition key, dict key, report column, log identifier

The `core.instrument.Symbol.canonical()` method (Phase 0b) is the
only allowed producer of this string. Code must NOT format these
manually.

## §3. Spot OHLCV Schema

Storage path:

```
/Volumes/DATA/crypto/raw/ohlcv/{venue}/{symbol_canonical}/{timeframe}/year={Y}/month={M}/day={D}.parquet
```

(One parquet per UTC calendar day. Day-partitioned for efficient
PIT reads.)

Schema columns (FROZEN — change only via §14.2-style new section):

| Column | Type | Nullable | Description |
|---|---|---|---|
| `timestamp_utc` | int64 | NO | Bar OPEN timestamp, milliseconds since epoch UTC |
| `exchange` | string | NO | Lowercase: `binance` / `okx` / `bybit` |
| `symbol` | string | NO | CCXT pair form: `BTC/USDT` (NO venue suffix here — venue is column) |
| `timeframe` | string | NO | `1h` / `4h` / `1d` |
| `open` | float64 | NO | |
| `high` | float64 | NO | |
| `low` | float64 | NO | |
| `close` | float64 | NO | |
| `volume_base` | float64 | NO | In base currency (BTC for BTC/USDT) |
| `volume_quote` | float64 | NO | In quote currency (USDT for BTC/USDT) |
| `quote_volume_estimated` | bool | NO | True if `volume_quote = volume_base * mid_price` approximation; False if from exchange. Phase A always True until we add an exchange-volume-aware fetcher. |
| `trades` | int32 | YES | -1 if exchange doesn't report |
| `is_closed_bar` | bool | NO | True iff `bar_close_ts + CLOSED_BUFFER_SEC ≤ ingestion_ts`. CLOSED_BUFFER_SEC default 120s. |
| `ingested_at` | int64 | NO | Ingestion wall-clock UTC ms — PIT replay anchor |

Closed-bar gate: implemented via single helper
`_is_closed_with_buffer(bar_open_ms, tf_sec, now_ms)` in
`data/collectors/crypto_market.py`. Both `fetch_recent` and
`fetch_historical` call it. Direct `bar_open + tf_ms <= now_ms`
comparisons or unconditional `is_closed_bar=True` are forbidden.

Range bound: `fetch_historical(start_ts_ms, end_ts_ms)` drops rows
where `timestamp_utc >= end_ts_ms` (CCXT's `since=` has no `until=`
companion, so the last page may overshoot).

## §4. Funding Rate Schema

Storage path:

```
/Volumes/DATA/crypto/raw/funding/{venue}/{symbol_canonical}/year={Y}/month={M}.parquet
```

(One parquet per month — funding is event-driven, low volume.)

Schema columns:

| Column | Type | Nullable | Description |
|---|---|---|---|
| `timestamp_utc` | int64 | NO | Funding event timestamp UTC ms |
| `exchange` | string | NO | |
| `symbol` | string | NO | CCXT perp form: `BTC/USDT:USDT` |
| `funding_rate` | float64 | NO | Signed; 0.0001 = 1bp per funding interval (typically 8h) |
| `next_funding_ts` | int64 | YES | |
| `mark_price` | float64 | YES | |
| `index_price` | float64 | YES | |
| `ingested_at` | int64 | NO | |

Pagination: `fetch_funding_history(symbol, start, end)` MUST use
cursor advance. Single-call `limit=1000` is insufficient for 1-year
windows (1095 events). Defensive: break on `last_ts <= cursor` to
prevent infinite loop on malformed responses.

## §5. Open Interest Schema

Storage path:

```
/Volumes/DATA/crypto/raw/open_interest/{venue}/{symbol_canonical}/year={Y}/month={M}.parquet
```

Schema columns:

| Column | Type | Nullable | Description |
|---|---|---|---|
| `timestamp_utc` | int64 | NO | Sample UTC ms, aligned to 15-minute grid |
| `exchange` | string | NO | |
| `symbol` | string | NO | |
| `open_interest` | float64 | NO | In base currency |
| `oi_quote` | float64 | YES | In quote currency if exchange reports |
| `long_short_ratio` | float64 | YES | |
| `ingested_at` | int64 | NO | |

Phase A: cron-polled every 15 min, accumulating forward. No
historical backfill (3-day acceptance window per spec §2.9.b).
Phase B adds `fetch_open_interest_history` for deeper history.

## §6. Delisted-Coin Audit Schema (survivorship preservation)

Storage path:

```
/Volumes/DATA/crypto/audit/delisted_coins.parquet
```

Schema columns:

| Column | Type | Nullable | Description |
|---|---|---|---|
| `exchange` | string | NO | |
| `symbol` | string | NO | |
| `listed_at_utc` | int64 | YES | Best-effort listing date |
| `delisted_at_utc` | int64 | NO | Required — defines the survivorship boundary |
| `last_close` | float64 | YES | Last available close before delisting |
| `reason` | string | YES | `exchange_removed` / `scam` / `unknown` |
| `detected_at` | int64 | NO | When we noticed it disappeared |

**Production rule**: universe construction at any historical `T`
MUST read BOTH `raw/ohlcv/` AND `audit/delisted_coins.parquet` with
`detected_at <= T`. Backtests that read only `raw/ohlcv/`
survivorship-bias by +62% (Ammann 2023 `[paper-reported]`).

Empty in Phase A (only 5 majors, none delisted). Code path must
still exist so Phase B universe expansion (top-30) does not silently
survivorship-bias.

## §7. Health File Schema

Storage path:

```
/Volumes/DATA/crypto/health/crypto_data_health.json
```

JSON shape (FROZEN):

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
    "binance__btc_usdt__spot/1h": {
      "latest_bar_ts": 1748600000000,
      "latest_bar_age_sec": 3245,
      "stale": false,
      "stale_threshold_sec": 5400,
      "gap_count_30d": 0,
      "gap_rate_30d": 0.0
    }
  },
  "funding": {
    "binance__btc_usdt__perp": {
      "latest_funding_ts": 1748592000000,
      "latest_funding_age_sec": 12000,
      "stale": false
    }
  },
  "open_interest": {},
  "cross_source": {
    "binance__btc_usdt__spot/1h": {
      "primary_close": 65432.10,
      "okx_close": 65430.50,
      "bybit_close": 65433.20,
      "max_spread_bps": 4.1,
      "binance_latency_ms": 420,
      "okx_latency_ms": 510,
      "bybit_latency_ms": 480
    }
  },
  "overall_status": "GREEN"
}
```

Decision rules for `overall_status` (one of `"GREEN"` / `"YELLOW"` /
`"RED"`):

- **`RED`** if ANY 1h bar > 90 min stale, OR any 4h bar > 5h stale,
  OR any 1d bar > 26h stale, OR primary exchange unreachable
- **`YELLOW`** if any gap rate > 1% in last 30 days, OR any
  fallback exchange unreachable, OR cross-source close spread > 25 bps,
  OR funding sign disagrees primary-vs-fallback
- **`GREEN`** otherwise

Cross-source samples (BTC/ETH/SOL only) piggyback on
`crypto_data_health` cron (every 4h) — they add ~3 REST calls per
health run.

## §8. Daily Paper Report Path

Per §−1 paper-only constraint:

```
/Volumes/DATA/crypto/reports/daily/{YYYY-MM-DD}_paper_report.md
```

One file per UTC calendar day. Generated by Phase D cron at
`30 8 * * *` Asia/Shanghai (= 00:30 UTC, after midnight roll).
Format spec at `cc-crypto-implementation-spec-2026-05-30.md §11`.

## §9. PIT (Point-in-Time) Discipline

Every parquet row carries `ingested_at`. Backtests / replays MUST:

1. Filter `ingested_at <= asof_ts` to exclude data that arrived after
   the simulated decision time
2. Filter `is_closed_bar=True` to exclude any bar still stabilizing
   at `asof_ts`
3. Use `core.universe.UniverseFilter.eligible(asof_ts)` (Phase 0b)
   which reads both `raw/ohlcv/` index AND `audit/delisted_coins.parquet`

Skipping any of these is a **silent survivorship/lookahead bias**.
A regression test in Phase B will inject a fake delisting + a fake
future-arriving row and assert the eligible-at backtest excludes
both.

## §10. Universe Construction (Phase A Initial)

Phase A primary universe (5 USDT spot majors on Binance):

```python
# config/crypto_universe.py (Phase 0b deliverable)
PRIMARY_EXCHANGE = "binance"     # may be overridden if Phase 0a spike shows unreachable
FALLBACK_EXCHANGES = ["okx", "bybit"]

PHASE_A_SPOT_BASES = ["BTC", "ETH", "SOL", "BNB", "XRP"]
PHASE_A_PERP_BASES = ["BTC", "ETH", "SOL"]

PHASE_A_TIMEFRAMES = ["1h", "4h", "1d"]
```

Universe expansion to top-20 / top-30 gated on 7 hard prerequisites
(per cx system design review §3) + user written sign-off — see spec
§4.2 for the list.

## §11. Numeric Defaults (FROZEN, can be revised after Phase 0a spike)

| Constant | Default | Source / source-tag | Override path |
|---|---|---|---|
| `CLOSED_BUFFER_SEC` | 120 | conservative — exchanges revise within 30-90s, `[paper-reported]` rule-of-thumb | Tighten after Phase 0a spike measures per-exchange revision windows |
| Stale-1h `max_lag_sec` | 5400 (90 min) | conservative — allows for exchange API lag + 1 retry + closed_buffer | Phase A.soak data |
| Stale-4h `max_lag_sec` | 18000 (5h) | same logic | Phase A.soak data |
| Stale-1d `max_lag_sec` | 93600 (26h) | same | Phase A.soak data |
| Cross-source spread `YELLOW` threshold | 25 bps | conservative — normal venue spread ~1-5 bps `[exchange-dashboard]` | Phase B feature-IC measurement |
| Min listed-days for universe | 60 | Phase A only uses majors so n/a; matters for Phase B expansion | Phase B research |
| Min dollar-volume floor | 50M USD/day rolling | conservative | Phase B research |

All other numeric claims live in `plans/numeric_claims_audit.md` with
explicit evidence tags.

## §12. Schema Versioning

Each parquet write carries an implicit `extractor_version` style
column ONLY for derived data (events, predictions). Raw OHLCV /
funding / OI have no version column — schema changes here trigger a
full re-fetch with the new schema, kept in a separate path:

```
/Volumes/DATA/crypto/raw/ohlcv_v2/...   # if §3 columns ever change
```

The current path is `raw/ohlcv/` (implicit v1).

## §13. Open Questions

Before Phase A code work starts (= soak complete + Phase 0a spike
measurements):

1. CCXT version pin — `4.x` series breaks `since=`/`until=` semantics
   subtly across versions. Phase 0a spike measures behavior on the
   installed version, this contract pins to that version.
2. Binance funding interval — Binance is 8h; if Phase 0a spike adds
   OKX/Bybit primary fallback, funding interval per venue must be
   recorded.
3. Initial backfill depth — 5y for 1h bars vs 2y vs 1y. Disk +
   rate-limit tradeoff resolved by Phase 0a spike disk measurement
   on `/Volumes/DATA`.

## Sign-off

This file gets `AUDIT-FROZEN-AT: <commit-sha>` at Phase Crypto-0
sign-off (mirroring CC plan §14.1 lock). Future revisions create
§14-style new sections, do not in-place rewrite frozen sections.
