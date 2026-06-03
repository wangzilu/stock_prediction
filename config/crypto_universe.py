"""Crypto Phase A universe constants.

Per `plans/crypto-data-contract.md` §10 Universe Construction:

- Phase A spot universe: 5 USDT majors on Binance
- Phase A perp universe: 3 (subset of spot)
- Phase A timeframes: 1h / 4h / 1d (closed bars only)
- Phase A primary exchange: Binance, with OKX / Bybit as fallbacks
- Phase B universe expansion gated on 7 hard prerequisites
  (`plans/cc-crypto-implementation-spec-2026-05-30.md` §4.2) PLUS
  written user sign-off

The module is import-cheap (no I/O, no network).
"""
from __future__ import annotations

# ---- Exchanges -----------------------------------------------------------

PRIMARY_EXCHANGE: str = "binance"
"""Primary exchange for Phase A. May be overridden by Phase 0a spike
results if the contract author determines mainland reachability via
ssproxy is unreliable for Binance."""

FALLBACK_EXCHANGES: tuple[str, ...] = ("okx", "bybit")
"""Tried in order if PRIMARY_EXCHANGE is unreachable or rate-limited.
The contract requires the collector to record which venue was actually
used in the health file for each fetch."""

# ---- Symbols -------------------------------------------------------------

PHASE_A_SPOT_BASES: tuple[str, ...] = ("BTC", "ETH", "SOL", "BNB", "XRP")
"""5 USDT-quoted spot majors. Picked because: top-5 by 30-day spot
volume on Binance as of 2026-05-30, each has > 3 years of clean
history, each has direct perp counterpart on Binance + OKX + Bybit
(needed for funding-arb research downstream)."""

PHASE_A_PERP_BASES: tuple[str, ...] = ("BTC", "ETH", "SOL")
"""3 perp pairs subset. Funding arb research starts with the most
liquid 3. BNB / XRP perps deferred to Phase B due to thinner OI."""

QUOTE_CURRENCY: str = "USDT"
"""All Phase A pairs are USDT-quoted. USDC / FDUSD considered for
Phase B if cross-stablecoin spread becomes a research topic."""

# ---- Timeframes ----------------------------------------------------------

PHASE_A_TIMEFRAMES: tuple[str, ...] = ("1h", "4h", "1d")
"""Phase A trades closed bars only — no tick / order-book / sub-1h
work. The collector layer must align bar boundaries to the
exchange's reported close (UTC) and apply `CLOSED_BUFFER_SEC` from
the data contract §11 before considering a bar usable."""

# ---- Backfill depth ------------------------------------------------------

BACKFILL_DEPTH_DAYS_BY_TF: dict[str, int] = {
    "1h": 90,    # rolling 90d for 1h — first paper run only needs a few weeks
    "4h": 365,   # rolling 1y for 4h
    "1d": 1825,  # rolling 5y for 1d (Liu-Tsyvinski sample window)
}
"""Initial backfill window per timeframe. Final values pending Phase 0a
disk-budget spike (contract §13 open question)."""

# ---- Universe expansion gate (Phase B) -----------------------------------

PHASE_B_EXPANSION_PREREQS: tuple[str, ...] = (
    "phase_a_paper_oms_clean_30_days",
    "vol_adv_data_pipeline_live",
    "cost_model_calibrated_via_paired_backtest",
    "dead_coin_audit_table_built",
    "ic_decay_lambda_measured_for_phase_a_factors",
    "research_signed_off_by_user_in_writing",
    "external_volume_at_50pct_or_less_capacity",
)
"""Seven prerequisites that must ALL be satisfied before Phase B
universe expansion (top-20 / top-30) is allowed. See spec §4.2."""


# ---- Helpers (small + pure) ---------------------------------------------
#
# Two ticker conventions are used in the pipeline (cx code review round 4
# P2 #2 — distinct helpers so naive `for s in spot_symbols(): fetch(s)`
# cannot silently mismatch the collector's CCXT-form expectation):
#
#   - CCXT pair form (`BTC/USDT`, `BTC/USDT:USDT`) — what
#     `data/collectors/crypto_market.py` and `crypto_derivatives.py`
#     accept. Use these `*_symbols_ccxt()` helpers when feeding the
#     collector layer.
#
#   - Binance-native ticker (`BTCUSDT`) — what the Binance REST URL path
#     uses internally. Phase A primary venue. Use these
#     `*_symbols_binance_native()` helpers when constructing direct
#     Binance REST URLs (rare; CCXT handles the translation usually).
#
# Other venues' native ticker forms (OKX, Bybit) are the collector
# layer's job to translate from the CCXT form.

def spot_symbols_ccxt(quote: str = QUOTE_CURRENCY) -> list[str]:
    """CCXT spot ticker list (`BTC/USDT`-style). This is what
    `crypto_market.fetch_recent` and `fetch_historical` accept."""
    return [f"{base}/{quote}" for base in PHASE_A_SPOT_BASES]


def perp_symbols_ccxt(quote: str = QUOTE_CURRENCY,
                       settle: str | None = None) -> list[str]:
    """CCXT perp ticker list (`BTC/USDT:USDT`-style). Settle defaults
    to the same as quote (linear perps). This is what
    `crypto_derivatives.fetch_funding_recent` etc. accept."""
    if settle is None:
        settle = quote
    return [f"{base}/{quote}:{settle}" for base in PHASE_A_PERP_BASES]


def spot_symbols_binance_native(quote: str = QUOTE_CURRENCY) -> list[str]:
    """Binance-native spot ticker (`BTCUSDT`, no slash). For direct
    REST URL construction against Binance only."""
    return [f"{base}{quote}" for base in PHASE_A_SPOT_BASES]


def perp_symbols_binance_native(quote: str = QUOTE_CURRENCY) -> list[str]:
    """Binance-native perp ticker (`BTCUSDT`, no slash). For direct
    REST URL construction against Binance USD-M futures only."""
    return [f"{base}{quote}" for base in PHASE_A_PERP_BASES]
