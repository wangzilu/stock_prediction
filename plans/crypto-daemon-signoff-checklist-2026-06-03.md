# Crypto Daemon Architecture — Sign-off Review Checklist

**Purpose**: structured review of `plans/crypto-daemon-architecture-2026-06-03.md`.
After every box is yes / explicitly-deferred, write
`AUDIT-FROZEN-AT: <merge-sha>` in the architecture doc's §Sign-off line
and Crypto-A step 4a (WS collector) is cleared to start.

**How to use**: each box has three response options:
  - **OK** — section is acceptable as written
  - **CHANGE** — wants edit (provide one-line redline)
  - **BLOCK** — wants design discussion before any sign-off

If anyone (user OR cx) marks BLOCK on any box, no sign-off. CHANGE
boxes are gathered into a §-style amendment to the architecture doc.

---

## §1 — WS log = replay anchor

Pinpoints:
- log path: `crypto_root/raw/ws_events/{venue}/{symbol_canonical}/{YYYYMMDD}.jsonl`
- dual clock: `seq_id` + `recv_local_ts_ns`
- replay sort key MUST be `seq_id` (not `recv_ts`)
- payload is verbatim, no transforms

Reviewer questions:
- [ ] Is per-day JSONL the right rotation cadence (vs hourly / per-symbol-size)?
- [ ] Is `seq_id` always present in Binance / Bybit / Hyperliquid streams?
       (Binance bookTicker: `u`; depth diff: `u`; trade: `T`+`E`. Streams
       without monotonic ID need explicit fallback.)
- [ ] Is "byte-identical paper PnL" achievable when CPython hash
       randomization is on? (Bound to seq_id only mitigates partly.)

Response: **OK** / **CHANGE** ____ / **BLOCK** ____

---

## §2 — Sequence gap detection + REST snapshot rebuild

Pinpoints:
- gap detect → emit `{"event":"gap_detected"}` to log AND health.jsonl
- drop in-memory book, REST `depthSnapshot`, replay diffs from
  `lastUpdateId+1`
- strategy code REFUSES signal emission while `rebuilding`
- measured 0.1-0.3% gap rate on Binance majors during high vol

Reviewer questions:
- [ ] Is the in-process REST snapshot fetch (not via wrapper) acceptable
       given §1.5's "every fetch via wrapper" rule? (The wrapper is for
       cron-launched processes; the daemon is one such process and its
       REST calls inherit the same sentinels — but worth pinning.)
- [ ] What's the upper bound on rebuilds per minute before strategy is
       forced into "degraded" mode?

Response: **OK** / **CHANGE** ____ / **BLOCK** ____

---

## §3 — ssproxy long-connection liveness

Pinpoints:
- application-level ping every 30s, timeout 3s, two misses → reconnect
- local RTT histogram per minute; P95 > 1500ms × 3 min → throttle to
  bar-close-only emission
- 30s cooling window after reconnect; mark-to-market only

Reviewer questions:
- [ ] Is 30s ping interval defensible vs Binance/Bybit's spec
       (Binance: ping every 3 min from server; Bybit: 20s heartbeat)?
- [ ] Should the cooling window be a fixed 30s or "until first 5 bars
       close"? Argument for fixed: deterministic. Argument for bars:
       semantically aligned.

Response: **OK** / **CHANGE** ____ / **BLOCK** ____

---

## §4 — Daemon resource boundary (A-share isolation)

Pinpoints:
- launchd `Nice=10`, `LowPriorityIO=true`, `ProcessType=Background`
- memory soft 1 GiB / hard 1.5 GiB; hard limit → self-restart
- CPU throttle windows (Asia/Shanghai):
  09:25-09:31, 14:30-14:36, 18:00-18:55, 22:00-22:05 = 25% one core
- enforcement is in-process (Python `resource` + wall-clock self-throttle)
- daemon writes `daemon_resource.jsonl` heartbeat every 60s

Reviewer questions:
- [ ] Are the 4 throttle windows exhaustive? (What about weekend evenings
       when batch jobs may run? `nightly_train.py` Sat 04:00.)
- [ ] Is 1 GiB soft / 1.5 GiB hard enough headroom for L1 book +
       feature rolling buffers across 5 symbols × 3 timeframes?
       (rough calc: 5 × 3 × 100 bars × ~1 KB / bar ≈ 1.5 MB; book
       updates buffered ~10k events at peak ≈ 10 MB. 1 GiB seems
       comfortable, but pin a measurement after week-1.)

Response: **OK** / **CHANGE** ____ / **BLOCK** ____

---

## §5 — Honest paper fill simulation

Pinpoints:
- market orders: `last_trade ± sqrt_adv slippage` bounded by best
  ask/bid; never fill better than current quote
- limit orders: pending state; cross-side volume check at fill
- stale-data block (> 2s old book → reject)
- spread filter (> 25 bps reject)
- `fill_quality` record: adverse_selection_bps, spread_at_fill_bps,
  book_age_ms

Reviewer questions:
- [ ] Is `last_trade ± slippage` adequate for thin markets where last
       trade can be stale by seconds? Should fallback be `mid` instead
       of `last_trade` when book is fresher than last trade?
- [ ] Is 25 bps a per-symbol parameter or a global constant? (Different
       majors have different normal spreads.)

Response: **OK** / **CHANGE** ____ / **BLOCK** ____

---

## §6 — Soak: 21 days + ≥1 day ≥5% move

Pinpoints:
- 21 calendar days minimum continuous uptime
- at least one day with primary symbol ≥ 5% move
- daily `intraday_soak_day.json` written
- gates: WS gaps ≤ 5/day, P95 RTT ≤ 800ms, restart ≤ 1/day, replay
  PnL within ±50% of backtest replay
- any threshold fail → soak clock RESETS

Reviewer questions:
- [ ] Is 21 days right vs 14 / 30? (21 is a compromise: longer than the
       Hummingbot-style 48h "looks fine" trap, shorter than the
       30-day A-share equivalent. ≥1 high-vol day matters more than
       total days.)
- [ ] What if BTC stays flat for 21 days? Do we wait, or accept the
       soak and tag it "vol-untested"?
- [ ] Is ±50% PnL band too loose? Tighten to ±20% after week 1?

Response: **OK** / **CHANGE** ____ / **BLOCK** ____

---

## §7 — Daemon collaborates with backfill, never duplicates

Pinpoints:
- daemon: WS only, never calls REST `fetch_ohlcv` / `fetch_funding`
- backfill cron (hourly): REST collectors, owns closed-bar parquet truth
- daily report cron (23:55 UTC): consumes daemon health + paper OMS
- daily reconciliation: daemon 1m bar vs backfill 1m bar, > 1bp diff
  → alert + daemon restart

Reviewer questions:
- [ ] What if backfill REST is rate-limited and lags by 1-2 hours?
       Does reconciliation fail spuriously? (Backfill cadence hourly
       gives ample buffer for the previous hour's bars to settle.)
- [ ] Is "restart on reconciliation diff" too aggressive? Could a single
       bad bar from REST trigger needless restart. Maybe restart only
       on 3 consecutive diffs.

Response: **OK** / **CHANGE** ____ / **BLOCK** ____

---

## Forbidden actions (the daemon NOT-allowed list)

Pinpoints:
- no live trading (paper-only, no API key with trade perms)
- no leverage > 1×
- venue must be in `config/crypto_universe.{PRIMARY,FALLBACK}`
- no writes to `data/storage/` (A-share)
- isolation lint clean at startup
- refuse to start if `LEGACY_MARKET_CONTEXT_ENABLED=True`

Reviewer questions:
- [ ] Is "no leverage > 1×" enforced at order-emit time or at risk-guard?
       Both? (Defense in depth recommended.)
- [ ] Should `LEGACY_MARKET_CONTEXT_ENABLED=True` cause daemon to refuse
       start, or to start in observation-only mode? (Doc says refuse;
       reviewer to confirm.)

Response: **OK** / **CHANGE** ____ / **BLOCK** ____

---

## Component naming (named, not implemented)

```
scripts/run_crypto_daemon.py
scripts/install_crypto_daemon.py
crypto/market_stream.py
crypto/order_book.py
crypto/bar_aggregator.py
crypto/feature_online.py
crypto/risk_guard.py
crypto/replay.py
paper/crypto_oms.py
strategies/crypto_fast_bar.py
config/crypto_daemon.py
tests/test_crypto_daemon_*.py
```

Cron entries: `*/5 watchdog`, `0 */1 backfill`, `55 23 report`.

Reviewer questions:
- [ ] Is the `crypto/` top-level package the right home? (Currently no
       top-level `crypto/` exists; collectors live under
       `data/collectors/`. The architecture doc creates a new package.)
- [ ] Does `paper/crypto_oms.py` live alongside `paper/oms.py` or in
       `crypto/paper_oms.py`? Architecture doc says `paper/`; OK?

Response: **OK** / **CHANGE** ____ / **BLOCK** ____

---

## Open decision points (architecture doc lists 3)

1. **launchd vs systemd**: doc commits to launchd (Mac dev box).
   Reviewer agree to defer Linux to migration time?
2. **WS venue for first daemon**: doc recommends Binance first.
   Reviewer agree?
3. **First strategy for 21-day soak**: doc recommends 1m/5m momentum +
   spread filter. Reviewer agree, or funding arb first?

Response: 1 ____ / 2 ____ / 3 ____

---

## When all boxes are answered

If every box is **OK** or explicitly **CHANGE with one-line redline**:

1. Apply CHANGE redlines to `plans/crypto-daemon-architecture-2026-06-03.md`
   as `§N.1`-style amendments (do NOT rewrite frozen text in place).
2. Commit the amendments + this checklist (with responses filled in)
   to master.
3. Write `AUDIT-FROZEN-AT: <merge-sha>` into the architecture doc's
   final §Sign-off block.
4. Mark Task #82 (Crypto-A step 1) related downstream tasks
   "✅ unblocked".
5. Crypto-A step 4a (`crypto/market_stream.py`) is cleared to start.

If any box is **BLOCK**:

1. Document the blocker in `plans/crypto-daemon-blockers-<date>.md`.
2. Architecture sign-off is held; Crypto-A step 4a does NOT start.
3. cx and user discuss the blocker before any further movement.

---

## Reviewer credentials block

```
Reviewed by:
  user:       [name]   date: _________   status: [signed / changes / blocked]
  cx:         [name]   date: _________   status: [signed / changes / blocked]
  cc:         claude-code            date: 2026-06-03   status: drafted

Final merge SHA filled in at sign-off:
  AUDIT-FROZEN-AT: ____________________
```
