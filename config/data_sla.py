"""Per-source freshness SLA contract.

The daily / weekly / quarterly publication cadence of each upstream
collector differs by data type. Pre-2026-06-06 the gate code asked the
same question for every source ("latest_date == latest trading day?"),
which forced a binary choice between:

  * mark the slow source non-critical and stop checking → silent stale,
    the bug class A.6 just closed; or
  * leave the source critical → permanent red between disclosure
    windows.

Phase A.7 (`plans/ashare-phases-2026-06.md`) replaces that binary with
a per-source SLA: each source declares a frequency and a budget in
trading days. ``scheduler.data_health.is_fresh`` reads the budget from
this map and applies it instead of "latest_date >= today".

Schema:

    SLA_BY_SOURCE = {
        "<health_source_name>": SourceSLA(frequency, max_age_trading_days, notes),
        ...
    }

Frequencies:

  * ``daily``     — published every trading day; budget normally 1.
  * ``weekly``    — published once a week (e.g. Saturday refresh);
                    budget normally 5-7 trading days so the gate is
                    fresh on a Monday morning.
  * ``quarterly`` — published when issuers file (e.g. Q1 ~end-April,
                    Q2 ~end-July, etc); budget normally 65 trading
                    days (roughly one fiscal quarter) so the gate
                    stays fresh between disclosure windows.

Callers that want to demand strict-daily freshness can opt out per-call.
The default behaviour is "use this map" — there is no implicit fallback
to ``latest_date == today`` for any source registered here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Allowed frequency labels. The dataclass __post_init__ enforces this
# so a typo cannot turn a weekly source into a silently-strict daily.
Frequency = Literal["daily", "weekly", "quarterly", "intraday"]


@dataclass(frozen=True)
class SourceSLA:
    """Per-source freshness SLA.

    Attributes
    ----------
    frequency:
        Publication cadence label (see module docstring).
    max_age_trading_days:
        How many CN trading days the recorded ``latest_date`` is
        allowed to lag behind the target date before the source is
        considered stale. Inclusive: a budget of 1 means
        ``latest_date >= target - 1 trading day`` is fresh, anything
        older is stale.
    notes:
        Free-text reason for the budget. Goes into audit / sign-off
        reports.
    """

    frequency: Frequency
    max_age_trading_days: int
    notes: str = ""

    def __post_init__(self) -> None:
        valid = {"daily", "weekly", "quarterly", "intraday"}
        if self.frequency not in valid:
            raise ValueError(
                f"SourceSLA.frequency={self.frequency!r} not in {sorted(valid)}"
            )
        if self.max_age_trading_days < 0:
            raise ValueError(
                f"SourceSLA.max_age_trading_days must be >= 0, "
                f"got {self.max_age_trading_days}"
            )


# Health-source-name → SLA contract. Every health row that downstream
# code wants to gate on MUST appear here; ``is_fresh`` falls back to
# strict-daily semantics ONLY when the caller explicitly opts out.
#
# When you add a new collector, add it here too. The companion
# ``tests/test_data_sla.py`` enforces that every entry in
# scheduler.data_health.PRODUCTION_GROUP_TO_HEALTH_SOURCE has an SLA
# record so we cannot silently regress to a free-floating freshness
# rule.
SLA_BY_SOURCE: dict[str, SourceSLA] = {
    # ── Daily price / flow ──────────────────────────────────────────
    "qlib_data_update": SourceSLA(
        "daily", 1,
        "Daily K-line. Should publish on every trading day's after-close batch.",
    ),
    "fund_flow_update": SourceSLA(
        "daily", 1,
        "Daily fund-flow (ST_CLIENT money_flow). Published once per trading day.",
    ),
    "northbound_update": SourceSLA(
        "daily", 1,
        "Daily Northbound flow. Was previously piggybacking fund_flow; "
        "now a separate health source after A6-5.",
    ),
    "regime_daily_update": SourceSLA(
        "daily", 1,
        "Daily regime data (margin / limit / hsgt / futures / USDCNY). "
        "Critical sub-sources gate success per A6-3.",
    ),
    "valuation_update": SourceSLA(
        "daily", 1,
        "Daily PE/PB/PS valuation parquet via baostock.",
    ),
    "st_daily_basic_update": SourceSLA(
        "daily", 1,
        "Daily ST_CLIENT daily_basic factors.",
    ),
    "st_moneyflow_update": SourceSLA(
        "daily", 1,
        "Daily ST_CLIENT moneyflow factors.",
    ),
    # ── Weekly refresh ──────────────────────────────────────────────
    "fundamental_update": SourceSLA(
        "weekly", 7,
        "Weekly fundamental features parquet (scripts/fetch_fundamental_features.py "
        "cron Sat 05:00).",
    ),
    "quality_update": SourceSLA(
        "weekly", 7,
        "Weekly fundamental quality factors (cron Sat 05:30). Earnings only "
        "refresh on the weekly recompute cadence.",
    ),
    # ── Quarterly disclosure ────────────────────────────────────────
    "shareholder_update": SourceSLA(
        "quarterly", 65,
        "Shareholder count disclosure tracks quarterly filings. Tradable "
        "window is one fiscal quarter — staying fresh through July, "
        "October, January, April disclosure cycles.",
    ),
    "st_holder_number_update": SourceSLA(
        "quarterly", 65,
        "ST_CLIENT holder-number disclosure follows quarterly filings.",
    ),
    # ── Overlays (LLM, supply chain) ────────────────────────────────
    "llm_event_pipeline": SourceSLA(
        "daily", 2,
        "Daily LLM event factor pipeline. Budget of 2 trading days so a "
        "single-day pipeline failure does not immediately invalidate "
        "downstream overlay use; A.6 fixes ensure partial / failed "
        "runs are visible in the gate.",
    ),
    "global_chain_factors": SourceSLA(
        "daily", 2,
        "Daily global supply-chain factor build. Same 2-day budget so a "
        "single failed scrape does not immediately disable the overlay.",
    ),
    # ── PE-1 PBOC monetary policy chain (Phase E.1) ─────────────────
    "pbc_policy_texts": SourceSLA(
        "daily", 2,
        "PBOC OMO / LPR daily text collection. 2-day budget tolerates "
        "the occasional weekend / holiday gap without disabling the "
        "downstream LLM extraction.",
    ),
    "pbc_policy_events": SourceSLA(
        "daily", 2,
        "Daily LLM extraction of PBC policy texts → structured events. "
        "Same 2-day budget so a 24-hour LLM API outage does not "
        "immediately disable the factor build.",
    ),
    "pbc_liquidity_factors": SourceSLA(
        "daily", 2,
        "Daily PBC liquidity factor build from extracted policy events. "
        "Feeds the xgb_209_pbc candidate profile via the FeatureMerger.",
    ),
    # ── PE-2 State Council + ministry industry policy chain (Phase E.2) ──
    # State Council and ministry policy docs publish less frequently
    # than PBOC daily OMO/LPR, so the budget is 3 trading days rather
    # than 2 — a long weekend + Tuesday silent day must not paint the
    # gate red and disable the downstream industry-policy overlay.
    "state_council_policy_texts": SourceSLA(
        "daily", 3,
        "State Council + 3 ministry policy doc collection from gov.cn. "
        "3-day budget tolerates the sparse publish cadence (some days "
        "have zero relevant policy docs; the gate must not flip red).",
    ),
    "state_council_policy_events": SourceSLA(
        "daily", 3,
        "Daily LLM extraction of State Council / ministry policy texts "
        "→ structured industry-policy events. 3-day budget matches the "
        "upstream collector.",
    ),
    "state_council_policy_factors": SourceSLA(
        "daily", 3,
        "Daily State Council industry-policy factor build. Emits per-"
        "(industry, date) rows; FeatureMerger maps stocks to industries "
        "at execution time. 3-day budget tolerates sparse publish days.",
    ),
    # ── PE-3 NBS macro-statistics chain (Phase E.3) ─────────────────
    # NBS publishes monthly (CPI / PPI / PMI / 社零) so the budget is
    # 35 trading days — roughly one monthly release cycle. A daily
    # weekday with zero new releases is the steady-state expectation;
    # the gate must not flip red between monthly publish dates.
    "nbs_policy_texts": SourceSLA(
        "daily", 35,
        "NBS macro-statistics text collection from stats.gov.cn. "
        "35-day budget matches NBS's monthly publish cadence — a "
        "zero-row weekday between monthly releases must NOT paint "
        "the gate red.",
    ),
    "nbs_policy_events": SourceSLA(
        "daily", 35,
        "Daily LLM extraction of NBS macro statistics → structured "
        "surprise events. 35-day budget matches the upstream collector.",
    ),
    "nbs_macro_factors": SourceSLA(
        "daily", 35,
        "Daily NBS macro-surprise factor build (3-month rolling "
        "aggregates). MARKET-keyed; broadcast to every stock by the "
        "FeatureMerger. 35-day budget matches the monthly release cycle.",
    ),
    # ── PE-4 CCTV Xinwen Lianbo theme attention chain (Phase E.4) ───
    # XWLB airs DAILY (incl. weekends) but our cron only runs weekdays.
    # The 2-trading-day budget covers a single failed weekday scrape
    # that gets re-run on Monday without painting the gate red. The
    # underlying broadcast is a CNS state-media program — theme
    # attention is a fact about media coverage, NOT a price prediction.
    "xinwen_lianbo_policy_texts": SourceSLA(
        "daily", 2,
        "CCTV Xinwen Lianbo (新闻联播) daily transcript collection via "
        "Sina syndication mirror. 2-day budget tolerates a single "
        "failed weekday scrape that gets re-run the next morning.",
    ),
    "xinwen_lianbo_policy_events": SourceSLA(
        "daily", 2,
        "Daily LLM extraction of Xinwen Lianbo transcripts → structured "
        "theme-attention events. 2-day budget matches the upstream "
        "collector.",
    ),
    "xinwen_lianbo_theme_factors": SourceSLA(
        "daily", 2,
        "Daily Xinwen Lianbo theme-attention factor build. Emits per-"
        "(theme, date) rows keyed THEME_<NAME>; FeatureMerger will map "
        "themes → stock baskets at execution time (mapper is TBD; for "
        "now the parquet is consumed standalone).",
    ),
    # ── Auxiliary ───────────────────────────────────────────────────
    "weekly_mask_rebuild": SourceSLA(
        "weekly", 7,
        "Tradable-mask rebuild runs Saturday morning; downstream training "
        "consumes it through the week.",
    ),
}


def get_sla(source: str) -> SourceSLA | None:
    """Return the registered SLA for ``source`` or ``None`` if absent.

    Callers can decide whether unregistered sources should fall back to
    a strict-daily check or be treated as exempt; the policy is the
    caller's responsibility, not this module's.
    """
    return SLA_BY_SOURCE.get(source)


def required_max_age(source: str) -> int | None:
    """Convenience wrapper — just the trading-day budget, or None."""
    sla = get_sla(source)
    return None if sla is None else sla.max_age_trading_days
