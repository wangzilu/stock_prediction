# Crypto Daemon Architecture — 2026-06-03

**Decision**: Phase Crypto-D shifts from a cron-driven loop to a 24/7
event-driven paper daemon. Cron is retained for **backfill / watchdog
/ daily report only**.

This document pins the seven sharp details that the daemon design
must satisfy. It supersedes the Phase D section of
`plans/crypto-dev-phases.md`. The roadmap doc itself is amended to
point here.

## Why the pivot

`plans/crypto-data-contract.md` §3 + §11 currently assume REST-polled
1h/4h/1d closed bars. After the 2026-06-03 architectural review:

- Crontab's process-launch jitter is incompatible with 1m bars
  (Python cold-start + ssproxy negotiate + parquet append per fire).
- Exchange-side data is **stream**, not poll: Binance spot kline
  pushes every 1-2s; USD-M futures depth pushes 100-500ms; Bybit WS
  enforces 20s heartbeat. Polling REST collapses stream into samples
  and loses microstructure.
- Order book + last-trade-id + bar-aggregation + pending paper
  orders are **continuous state**. Cron's "cold-start each fire"
  fights that.
- Binance / Bybit WS connections drop on a 24h cadence by spec; the
  reconnect-with-resync pattern is daemon-shaped, not cron-shaped.

The agent research bundle in this commit's session backs each claim
with primary sources (see attached PR self-review).

## The seven sharp details

### #1 — WS log is the replay anchor (byte-identical paper PnL)

The daemon writes every WS event it receives to an append-only log
`crypto_root/raw/ws_events/{venue}/{symbol_canonical}/{YYYYMMDD}.jsonl`.
That log is the **single source of truth** for both live paper and
offline backtest. Replay engine reads the same file and must produce
byte-identical fills, positions and PnL.

Log entry schema (FROZEN):

```json
{
  "seq_id": 17432101,             // exchange-provided monotonic per stream
  "stream": "btcusdt@bookTicker", // exchange's stream name verbatim
  "exchange": "binance",
  "symbol_canonical": "binance__btc_usdt__spot",
  "venue_ts_ms": 1717459200123,   // exchange's reported timestamp
  "recv_local_ts_ns": 1717459200124847215,  // nanosecond local recv
  "payload": { ... }              // verbatim, no transforms
}
```

Two clocks because: `recv_local_ts_ns` is needed for latency audit,
but **replay sort key MUST be `seq_id`** so a clock jump (NTP sync,
laptop sleep) cannot corrupt history.

### #2 — Sequence gap detection + REST snapshot rebuild

Per Binance depth-stream contract: each event has `U` (first update
id) and `u` (last update id); consecutive events must satisfy
`event.U == prev.u + 1`. If the daemon sees a gap (`event.U > prev.u
+ 1`), the order book state is **silently corrupt** unless rebuilt.

Daemon contract:
1. On gap detect → emit `{"event":"gap_detected", "gap_size": N}` to
   the WS log AND to a separate `health.jsonl`.
2. Drop in-memory book, request REST `depthSnapshot`, replay buffered
   diffs from snapshot's `lastUpdateId+1`.
3. Strategy code MUST refuse to emit signals while book is
   `rebuilding`. Risk guard fails-closed here.

This is non-negotiable because measured gap rate is 0.1-0.3% during
high vol on Binance majors — without the rebuild step, paper PnL
silently drifts from any reasonable replay.

### #3 — ssproxy long-connection liveness

The mainland → ssproxy → exchange path is not optimized for long-
lived TCP. Observed failure modes:

- Application-level keepalive (Binance's auto WS ping every 3 min)
  goes through proxy fine, BUT proxy may TCP-close an idle socket
  before that fires (default `idle_timeout` 60-300s on most ssproxy
  builds).
- RTT spikes from < 200ms to > 2s for several seconds during proxy
  hop-switches; messages backlog locally then arrive in a burst.

Daemon contract:
1. **Application-level ping every 30s** independent of the exchange's
   schedule, with timeout 3s; missed two consecutive pings →
   reconnect.
2. **Local RTT histogram** logged per minute. If P95 > 1500ms for 3
   consecutive minutes, daemon writes `degraded_network=true` to
   health and strategy code throttles signal emission to bar-close
   only (effectively degrades to 5m bar trading until network
   recovers).
3. **No auto-trade resumption after reconnect**: a 30s "cooling
   window" after reconnect during which only mark-to-market runs;
   strategy emission is paused. Prevents a stale-state burst signal
   from firing into the first reconnected book.

### #4 — Daemon resource boundary (A-share isolation)

Daemon must NEVER contend with A-share crons during their active
windows. Hard contract:

- Process is launched via launchd with `Nice=10`,
  `LowPriorityIO=true`, `ProcessType=Background`.
- Memory soft limit 1 GiB. Hard limit 1.5 GiB. Exceeding hard limit
  is a self-restart (not a kill that A-share might inherit).
- CPU governance window (Asia/Shanghai):
  - **09:25-09:31**: daemon throttled to 25% of one core (A-share
    morning_recommendation needs headroom).
  - **14:30-14:36**: same, for sell_check.
  - **18:00-18:55**: same, for evening_outlook + all batch jobs.
  - **22:00-22:05**: same, for evening_outlook completion.
  - Other times: 100% of one core max.
- Daemon refuses to start if `psutil` shows total system memory
  pressure > 80%.
- Daemon writes a `daemon_resource.jsonl` heartbeat every 60s for
  the A-share-isolation soak script to verify.

These limits are enforced inside the daemon (Python `resource`
module + a wall-clock-based self-throttle), not by an external
governor — that way the daemon can never be the cause of a missed
A-share signal even if launchd config drifts.

### #5 — Paper fill simulation (honest, not optimistic)

A-share paper OMS marks at next-day close. **Crypto has no "close"**,
so we need an explicit fill model:

1. **Market orders**: fill at `last_trade_price ± slippage_bps` where
   slippage is sqrt-of-fraction-of-1m-volume per
   `backtest/cost_model.py` sqrt_adv path. Bound: never better than
   the current `best_ask` (buy) or `best_bid` (sell).
2. **Limit orders**: emit `pending` state. Daemon ticks each pending
   order on every book update; fills when the appropriate side of
   the book crosses the limit price AND the same-side reported
   volume at that level is ≥ order size. Otherwise stays pending
   indefinitely OR is cancelled by strategy timeout.
3. **Stale-data block**: if last book update is > 2s old, all new
   orders are rejected with `reason=stale_book`. Pending orders
   remain pending; they do NOT auto-fill against stale books.
4. **Spread filter**: orders are rejected when
   `(ask - bid) / mid > spread_max_bps` (default 25 bps per data
   contract §11).

Each fill carries a `fill_quality` record:
`{"adverse_selection_bps": ..., "spread_at_fill_bps": ...,
"book_age_ms": ...}`. Backtest replay uses the SAME logic so paper
and backtest can be directly compared.

### #6 — Soak window: 21 days, at least one ≥5% day

The 30-day Phase D paper from `plans/crypto-dev-phases.md` was sized
for daily-cron strategies. The intraday daemon has more failure
modes (WS gaps, proxy flaps, daemon restart cycles), so:

- **Minimum 21 calendar days continuous uptime**, no daemon crashes
  beyond auto-recovery.
- During those 21 days, at least one calendar day with the primary
  symbol moving ≥ 5% in either direction. Without this, the risk-
  guard envelope hasn't been pressure-tested.
- Daemon writes a daily `intraday_soak_day.json`:
  `{n_ws_gaps, n_reconnects, p95_rtt_ms, daemon_restart_count,
  fills_count, fill_quality_summary, after_cost_pnl}`. The soak
  passes only when:
  - `n_ws_gaps_per_day` ≤ 5 (sustained higher = strategy disabled)
  - `p95_rtt_ms` ≤ 800
  - `daemon_restart_count` ≤ 1/day
  - At least one ≥5% volatility day witnessed
  - Cumulative after-cost PnL within ±50% of the same period's
    backtest replay (tight band proves replay determinism)

If any threshold fails, soak clock resets.

### #7 — Daemon collaborates with backfill, never duplicates it

Three roles, three processes:

1. **Daemon (24/7)**: receives WS, writes events log, runs strategy,
   maintains paper OMS state. Never calls REST `fetch_ohlcv` or
   `fetch_funding_rate_history` itself.
2. **Backfill cron (hourly)**: calls REST collectors
   (`crypto_market.fetch_recent`, `crypto_derivatives.fetch_funding_recent`)
   to fill any historical gaps in the closed-bar parquet store. It is
   THE source of truth for closed bars; daemon's bar aggregator
   writes only to a separate `live_aggregates/` path and is
   reconciled against backfill output once per day.
3. **Daily report cron (23:55 UTC)**: reads daemon health log +
   paper OMS state + backfill freshness; produces
   `crypto_root/reports/daily/{YYYYMMDD}.md` for the user.

The reconciliation step is what makes the daemon trustworthy: if
daemon's 1m bar differs from backfill's 1m bar for the same minute by
more than 1bp, the reconciliation job writes an alert and the daemon
restarts (forces a fresh WS subscription).

## What the daemon is NOT allowed to do

- Trade real money (paper-only, hard rule, no API key with trade
  permissions in any config).
- Hold leverage (no perp positions whose notional > 1× spot equity).
- Connect to a venue not in `config/crypto_universe.PRIMARY_EXCHANGE`
  plus `FALLBACK_EXCHANGES`.
- Touch `data/storage/` (A-share). All writes go to
  `crypto_root() / "live/"`.
- Run if the A-share isolation lint flags any new code path that
  could import A-share modules at hot-loop time.
- Continue trading if `LEGACY_MARKET_CONTEXT_ENABLED=True` is
  detected at startup (cross-checks the quarantine invariant).

## Components (named, not implemented)

```
scripts/run_crypto_daemon.py            ← entrypoint (launchd target)
scripts/install_crypto_daemon.py        ← installs launchd plist
crypto/market_stream.py                 ← WS subscribe + event log
crypto/order_book.py                    ← in-memory book + gap rebuild
crypto/bar_aggregator.py                ← live → 1m/5m/15m closed bars
crypto/feature_online.py                ← rolling-window features
crypto/risk_guard.py                    ← stale / spread / vol / gap gate
crypto/replay.py                        ← consume WS log → identical PnL
paper/crypto_oms.py                     ← T+0 paper OMS (own state file)
strategies/crypto_fast_bar.py           ← first 1m/5m baseline strategy
config/crypto_daemon.py                 ← all the knobs above
tests/test_crypto_daemon_*.py           ← unit + replay determinism + soak
```

`scripts/run_paper_trading.py` (A-share) is NOT touched.
`scripts/install_crontab.py` adds **three** crypto cron entries:

```
*/5 * * * *  crypto_daemon_watchdog
0 */1 * * *  crypto_backfill_ohlcv
55 23 * * *  crypto_daily_report
```

NONE of those three is a trading loop.

## Roadmap impact

`plans/crypto-dev-phases.md` is amended in this same commit:

- **Phase A**: existing CCXT REST collectors keep their role (backfill,
  health). Add step 4a (WS collector) and step 4b (replay engine)
  before the next-phase code.
- **Phase B**: feature pipeline timeframes expand to include
  1m/5m/15m alongside 1h/4h/1d. Online features (`feature_online.py`)
  are computed by daemon; offline batch features stay in the
  feature_cache path. Both must produce the same vector for the
  same (timestamp, symbol) — a contract test enforces this.
- **Phase C**: funding arb backtest stays as designed (event-cadence
  is exchange-controlled, unchanged by daemon switch). Add a
  daemon-side "online funding monitor" that records realised funding
  for replay sanity.
- **Phase D**: was `run_crypto_paper_trading.py` cron. Now
  `run_crypto_daemon.py` 24/7 + the three cron entries above. 30-day
  soak → 21-day soak with the §6 acceptance bands.
- **Phase E backlog**: order book L2 alpha (was implicitly in D)
  moves here. Daemon only uses L1 (best bid/ask + last trade) for
  the first 21-day soak.

## Decision points (for user / cx)

1. **Mac launchd vs systemd-style supervisor**: this proposal commits
   to launchd because the dev box is Mac. If we ever migrate to a
   Linux VPS, `scripts/install_crypto_daemon.py` ships a systemd
   unit fallback. Decision: defer to migration time.
2. **WS venue choice for first daemon**: Binance is primary per
   `config/crypto_universe.PRIMARY_EXCHANGE` BUT Hyperliquid SDK has
   the cleanest 2026 Python integration (per agent research). My
   recommendation is **Binance first** because the rest of the
   pipeline targets it; add Hyperliquid as Phase E.1.
3. **Strategy for the first 21-day soak**: minute-bar momentum with
   a spread filter is the safest starting point (published Sharpes
   exist, no inventory management problem). Funding arb is parallel
   but is a different daemon role and should land after the bar
   daemon proves stable.

## Sign-off gate

This document is `AUDIT-FROZEN-AT: <commit-sha>` at the time the user
signs off in writing. Future changes create a §-style new section
rather than rewriting in place (same protocol as
`crypto-data-contract.md`).
