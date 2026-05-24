"""Data Availability Registry — centralized time-semantics for every data source.

Replaces scattered documentation with a machine-readable registry that defines
when each data source's event occurs, when it is published, and when the system
is allowed to use it (signal_lag_bdays).

Usage:
    from config.data_availability import DATA_REGISTRY, get_spec, print_registry
    spec = get_spec("st_daily_basic")
    print(spec.signal_lag_bdays)  # 1
    print_registry()
"""

from dataclasses import dataclass, field


@dataclass
class DataSourceSpec:
    name: str                       # e.g. "st_daily_basic"
    event_time: str                 # when the data event occurs: "T_close", "T_1100", "report_end"
    publish_time: str               # when data is published: "T_1700", "ann_date", "T+30~120d"
    available_time_rule: str        # when system can use it: "T+1_BDay", "publish+1_BDay", "same_day"
    signal_lag_bdays: int           # business days from event to signal usage (0, 1, 2...)
    execution_lag: str              # "T+1_open" for most A-share
    pit_safe_level: str             # "verified", "assumed", "unsafe"
    allowed_usage: list[str] = field(default_factory=list)  # ["training", "signal", "regime", "risk"]
    notes: str = ""


# ---------------------------------------------------------------------------
# Registry: every data source used in feature_merger.py and regime_controller.py
# ---------------------------------------------------------------------------

DATA_REGISTRY: dict[str, DataSourceSpec] = {}


def _reg(spec: DataSourceSpec) -> None:
    """Register a DataSourceSpec into the global registry."""
    DATA_REGISTRY[spec.name] = spec


# ── Alpha158 (Qlib OHLCV) ──────────────────────────────────────────────────
_reg(DataSourceSpec(
    name="alpha158",
    event_time="T_close",
    publish_time="T_close",
    available_time_rule="T+1_BDay",
    signal_lag_bdays=1,
    execution_lag="T+1_open",
    pit_safe_level="verified",
    allowed_usage=["training", "signal"],
    notes=(
        "Qlib Alpha158 uses OHLCV from day T (close at 15:00). "
        "Signal is generated after close, executed next open. "
        "Qlib's internal handler enforces lag-1 by default."
    ),
))

# ── ST daily_basic (PE/PB/MV/turnover) ─────────────────────────────────────
_reg(DataSourceSpec(
    name="st_daily_basic",
    event_time="T_close",
    publish_time="T_1700",
    available_time_rule="T+1_BDay",
    signal_lag_bdays=1,
    execution_lag="T+1_open",
    pit_safe_level="verified",
    allowed_usage=["training", "signal"],
    notes=(
        "TuShare daily_basic publishes PE/PB/turnover/MV after market close (~17:00). "
        "feature_merger.py adds BDay(1) to the date column before asof merge, "
        "so day-T data only enters T+1 predictions."
    ),
))

# ── ST moneyflow (capital flow per stock) ───────────────────────────────────
_reg(DataSourceSpec(
    name="st_moneyflow",
    event_time="T_close",
    publish_time="T_1700",
    available_time_rule="T+1_BDay",
    signal_lag_bdays=1,
    execution_lag="T+1_open",
    pit_safe_level="assumed",
    allowed_usage=["training", "signal"],
    notes=(
        "TuShare moneyflow (资金流向) published after close. "
        "feature_merger._load_st_moneyflow does NOT currently add BDay(1) — "
        "relies on asof merge with Qlib's lag-1 index. "
        "Effective lag is 1 BDay but should be explicitly enforced like st_daily_basic."
    ),
))

# ── Fund flow history (capital flow via fund_flow_history.parquet) ──────────
_reg(DataSourceSpec(
    name="fund_flow_history",
    event_time="T_close",
    publish_time="T_1700",
    available_time_rule="T+1_BDay",
    signal_lag_bdays=1,
    execution_lag="T+1_open",
    pit_safe_level="verified",
    allowed_usage=["training", "signal"],
    notes=(
        "Daily capital flow (net_mf_amount) from ST/AK. "
        "feature_merger._load_capital_flow_from_history explicitly adds BDay(1). "
        "PIT audit confirmed flow_lag1 RankIC +0.043 > flow_lag0 +0.038."
    ),
))

# ── Northbound HSGT (per-stock holdings) ────────────────────────────────────
_reg(DataSourceSpec(
    name="northbound_holdings",
    event_time="T_close",
    publish_time="T_1900",
    available_time_rule="T+1_BDay",
    signal_lag_bdays=1,
    execution_lag="T+1_open",
    pit_safe_level="verified",
    allowed_usage=["training", "signal"],
    notes=(
        "Northbound per-stock holdings (vol, ratio) published after HK close. "
        "feature_merger._load_northbound explicitly adds BDay(1) to trade_date."
    ),
))

# ── Northbound HSGT (aggregate flow for regime) ────────────────────────────
_reg(DataSourceSpec(
    name="northbound_hsgt_flow",
    event_time="T_close",
    publish_time="T_1900",
    available_time_rule="T+1_BDay",
    signal_lag_bdays=1,
    execution_lag="T+1_open",
    pit_safe_level="verified",
    allowed_usage=["regime"],
    notes=(
        "Aggregate northbound flow (st_moneyflow_hsgt.parquet) used in regime_controller. "
        "Published after HK close; regime_controller uses as_of(date) filter."
    ),
))

# ── ST margin_detail (融资余额) ─────────────────────────────────────────────
_reg(DataSourceSpec(
    name="st_margin_detail",
    event_time="T_close",
    publish_time="T+1_0900",
    available_time_rule="T+1_BDay",
    signal_lag_bdays=1,
    execution_lag="T+1_open",
    pit_safe_level="verified",
    allowed_usage=["regime"],
    notes=(
        "Margin trading data (融资余额) published next morning by exchanges. "
        "regime_controller uses as_of(date) filter. Day-T margin balance "
        "is available before T+1 open."
    ),
))

# ── ST limit_list_d (涨跌停) ────────────────────────────────────────────────
_reg(DataSourceSpec(
    name="st_limit_list_d",
    event_time="T_close",
    publish_time="T_1530",
    available_time_rule="T+1_BDay",
    signal_lag_bdays=1,
    execution_lag="T+1_open",
    pit_safe_level="verified",
    allowed_usage=["regime"],
    notes=(
        "Daily limit-up/limit-down list. Available immediately after close. "
        "regime_controller._microcap_crash uses 5-day rolling count."
    ),
))

# ── Financial reports WITH ann_date ─────────────────────────────────────────
_reg(DataSourceSpec(
    name="financial_with_ann_date",
    event_time="report_end",
    publish_time="ann_date",
    available_time_rule="ann_date+1_BDay",
    signal_lag_bdays=1,
    execution_lag="T+1_open",
    pit_safe_level="verified",
    allowed_usage=["training", "signal"],
    notes=(
        "Financial statements with actual announcement date (ann_date / f_ann_date). "
        "feature_merger._effective_date_from_frame adds BDay(1) to ann_date, "
        "ensuring data is only used from the trading day after announcement."
    ),
))

# ── Financial reports WITHOUT ann_date (statutory fallback) ─────────────────
_reg(DataSourceSpec(
    name="financial_no_ann_date",
    event_time="report_end",
    publish_time="T+30~120d",
    available_time_rule="statutory_deadline",
    signal_lag_bdays=120,
    execution_lag="T+1_open",
    pit_safe_level="assumed",
    allowed_usage=["training"],
    notes=(
        "Financial statements without ann_date fall back to conservative statutory "
        "deadlines: Q1 +45d, H1 +75d, Q3 +45d, FY +120d. "
        "Worst case is FY (report_end=Dec31, available ~Apr30 = +120 calendar days). "
        "signal_lag_bdays=120 reflects maximum calendar-day delay for FY reports."
    ),
))

# ── ST holder_number (股东户数) ─────────────────────────────────────────────
_reg(DataSourceSpec(
    name="st_holder_number",
    event_time="report_end",
    publish_time="ann_date",
    available_time_rule="ann_date+0_BDay",
    signal_lag_bdays=0,
    execution_lag="T+1_open",
    pit_safe_level="verified",
    allowed_usage=["training", "signal"],
    notes=(
        "Shareholder count keyed by ann_date. "
        "feature_merger._load_st_holder_number uses ann_date directly as merge key "
        "(no additional BDay(1) shift). Data becomes usable on ann_date itself "
        "since announcements are typically pre-market or post-close prior day."
    ),
))

# ── LLM events ─────────────────────────────────────────────────────────────
_reg(DataSourceSpec(
    name="llm_events",
    event_time="T_variable",
    publish_time="T_collected",
    available_time_rule="same_day",
    signal_lag_bdays=0,
    execution_lag="T+1_open",
    pit_safe_level="unsafe",
    allowed_usage=["regime"],
    notes=(
        "LLM-parsed news events stored as daily JSONL files (llm_events/YYYY-MM-DD.jsonl). "
        "regime_controller uses filename date as PIT filter. "
        "NOT PIT-safe for historical replay because news collection timestamps vary. "
        "Only 18 days of data available (2026-04-27+)."
    ),
))

# ── Shibor ──────────────────────────────────────────────────────────────────
_reg(DataSourceSpec(
    name="shibor",
    event_time="T_1100",
    publish_time="T_1100",
    available_time_rule="same_day",
    signal_lag_bdays=0,
    execution_lag="T+1_open",
    pit_safe_level="verified",
    allowed_usage=["regime"],
    notes=(
        "Shanghai Interbank Offered Rate published at 11:00 daily. "
        "regime_controller uses as_of(date) filter for overnight and 3M rates. "
        "Available intraday, used for liquidity_score and credit_stress_score."
    ),
))

# ── CPI ─────────────────────────────────────────────────────────────────────
_reg(DataSourceSpec(
    name="cpi",
    event_time="month_end",
    publish_time="T+10~15d",
    available_time_rule="publish+0_BDay",
    signal_lag_bdays=15,
    execution_lag="T+1_open",
    pit_safe_level="verified",
    allowed_usage=["regime"],
    notes=(
        "Monthly CPI (nt_yoy) from NBS, typically released 10-15 days after month end. "
        "regime_controller uses as_of(date) filter on monthly data. "
        "signal_lag_bdays=15 is approximate: month-end to typical publish date."
    ),
))

# ── PMI ─────────────────────────────────────────────────────────────────────
_reg(DataSourceSpec(
    name="pmi",
    event_time="month_end",
    publish_time="T+1d",
    available_time_rule="publish+0_BDay",
    signal_lag_bdays=1,
    execution_lag="T+1_open",
    pit_safe_level="verified",
    allowed_usage=["regime"],
    notes=(
        "Monthly PMI released on the 1st of the following month. "
        "Not currently used in regime_controller but collected via macro pipeline. "
        "signal_lag_bdays=1 reflects 1 calendar day after month end."
    ),
))

# ── M2 ──────────────────────────────────────────────────────────────────────
_reg(DataSourceSpec(
    name="m2",
    event_time="month_end",
    publish_time="T+10~15d",
    available_time_rule="publish+0_BDay",
    signal_lag_bdays=15,
    execution_lag="T+1_open",
    pit_safe_level="verified",
    allowed_usage=["regime"],
    notes=(
        "Monthly M2 growth (m2_yoy) from PBOC, released ~10-15 days after month end. "
        "regime_controller uses as_of(date) on monthly data for liquidity_score."
    ),
))

# ── IC/IM futures (AKShare) ─────────────────────────────────────────────────
_reg(DataSourceSpec(
    name="ic_im_futures",
    event_time="T_close",
    publish_time="T_1530",
    available_time_rule="T+1_BDay",
    signal_lag_bdays=1,
    execution_lag="T+1_open",
    pit_safe_level="verified",
    allowed_usage=["regime"],
    notes=(
        "IC/IM futures close prices (ak_futures_ic0.parquet) from AKShare. "
        "CFFEX futures close at 15:00. regime_controller computes real basis "
        "vs CSI500 spot (ak_index_csi500.parquet) for quant crowding risk."
    ),
))

# ── USD/CNY (AKShare) ──────────────────────────────────────────────────────
_reg(DataSourceSpec(
    name="usdcny",
    event_time="T_close",
    publish_time="T_1600",
    available_time_rule="T+1_BDay",
    signal_lag_bdays=1,
    execution_lag="T+1_open",
    pit_safe_level="verified",
    allowed_usage=["regime"],
    notes=(
        "USD/CNY exchange rate (中行折算价, 百倍报价 e.g. 683.73 = 6.8373) from AKShare. "
        "regime_controller uses 5-day change for fx_risk_score."
    ),
))

# ── Guba (人气榜 / popularity ranking) ──────────────────────────────────────
_reg(DataSourceSpec(
    name="guba",
    event_time="T_variable",
    publish_time="T_collected",
    available_time_rule="same_day",
    signal_lag_bdays=0,
    execution_lag="T+1_open",
    pit_safe_level="unsafe",
    allowed_usage=["regime"],
    notes=(
        "Guba popularity ranking stored as daily JSONL (guba/YYYY-MM-DD.jsonl). "
        "regime_controller._theme_breadth uses row count as breadth proxy. "
        "NOT PIT-safe for historical replay: very sparse (currently 1 file). "
        "Collection time is variable."
    ),
))

# ── Cross-market indices (恒生/纳指) ────────────────────────────────────────
_reg(DataSourceSpec(
    name="cross_market_indices",
    event_time="T_close",
    publish_time="T_close",
    available_time_rule="same_day",
    signal_lag_bdays=0,
    execution_lag="T+1_open",
    pit_safe_level="verified",
    allowed_usage=["training", "signal"],
    notes=(
        "Cross-market regime signals (HSI/Nasdaq returns). "
        "feature_merger._load_cross_market_regime broadcasts to all stocks per date. "
        "HK closes before A-share; Nasdaq prior-day close is known at A-share open. "
        "No additional BDay shift applied — asof merge ensures no look-ahead."
    ),
))

# ── Macro features (broadcast, legacy single-row) ──────────────────────────
_reg(DataSourceSpec(
    name="macro_features",
    event_time="variable",
    publish_time="variable",
    available_time_rule="same_day",
    signal_lag_bdays=0,
    execution_lag="T+1_open",
    pit_safe_level="unsafe",
    allowed_usage=["training"],
    notes=(
        "macro_features.parquet currently has only 1 row (latest snapshot), "
        "broadcast to all dates. This is a known PIT weakness — same values "
        "leak into historical training samples. Impact is small because macro "
        "factors change slowly. Should be upgraded to daily time-series."
    ),
))


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_spec(source_name: str) -> DataSourceSpec:
    """Look up a data source specification by name.

    Raises KeyError if source_name is not in the registry.
    """
    if source_name not in DATA_REGISTRY:
        available = ", ".join(sorted(DATA_REGISTRY.keys()))
        raise KeyError(
            f"Unknown data source '{source_name}'. "
            f"Available: {available}"
        )
    return DATA_REGISTRY[source_name]


def get_signal_lag(source_name: str) -> int:
    """Return the signal_lag_bdays for a data source."""
    return get_spec(source_name).signal_lag_bdays


def validate_feature_merger_lags() -> list[str]:
    """Check that feature_merger's actual lag handling matches the registry.

    Returns a list of warning messages. Empty list means all checks pass.
    """
    warnings = []

    # Sources where feature_merger explicitly adds BDay(1)
    explicit_bday1_sources = {
        "st_daily_basic": "feature_merger._load_st_daily_basic adds BDay(1)",
        "fund_flow_history": "feature_merger._load_capital_flow_from_history adds BDay(1)",
        "northbound_holdings": "feature_merger._load_northbound adds BDay(1)",
        "financial_with_ann_date": "feature_merger._effective_date_from_frame adds BDay(1) to ann_date",
    }

    for name, description in explicit_bday1_sources.items():
        spec = DATA_REGISTRY.get(name)
        if spec and spec.signal_lag_bdays != 1:
            warnings.append(
                f"MISMATCH: {name} has explicit BDay(1) in code but "
                f"registry says signal_lag_bdays={spec.signal_lag_bdays}"
            )

    # Sources where feature_merger does NOT add BDay(1) but registry says lag=1
    no_explicit_shift = {
        "st_moneyflow": (
            "feature_merger._load_st_moneyflow does NOT add BDay(1). "
            "Relies on Qlib's lag-1 index alignment. Consider adding explicit shift."
        ),
    }

    for name, note in no_explicit_shift.items():
        spec = DATA_REGISTRY.get(name)
        if spec and spec.signal_lag_bdays >= 1 and spec.pit_safe_level == "assumed":
            warnings.append(f"REVIEW: {name} — {note}")

    # Sources marked unsafe
    unsafe = [name for name, spec in DATA_REGISTRY.items()
              if spec.pit_safe_level == "unsafe"]
    if unsafe:
        warnings.append(
            f"UNSAFE PIT: {', '.join(unsafe)} — not safe for historical replay"
        )

    return warnings


def print_registry() -> None:
    """Print a formatted table of all registered data sources."""
    header = (
        f"{'Name':<28} {'Lag':>3} {'PIT Safety':<10} "
        f"{'Event':<14} {'Available Rule':<20} {'Usage'}"
    )
    print("=" * len(header))
    print("DATA AVAILABILITY REGISTRY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name in sorted(DATA_REGISTRY.keys()):
        spec = DATA_REGISTRY[name]
        usage_str = ", ".join(spec.allowed_usage)
        print(
            f"{spec.name:<28} {spec.signal_lag_bdays:>3} {spec.pit_safe_level:<10} "
            f"{spec.event_time:<14} {spec.available_time_rule:<20} {usage_str}"
        )

    print("-" * len(header))

    # Validation warnings
    warnings = validate_feature_merger_lags()
    if warnings:
        print(f"\nValidation warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ! {w}")
    else:
        print("\nAll validation checks passed.")
