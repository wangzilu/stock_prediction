"""Production feature contract for A-share model training/inference.

Production code must not consume every feature that FeatureMerger can load.
Research and shadow scripts may experiment with additional loaders, but the
live champion model should only change feature groups through this explicit
contract plus a retrain/promotion decision.

Why this file exists (2026-06-04):
- commit 95cd256 (2026-05-12) opened the
  ``scripts/train_lgb.py`` → ``FeatureMerger._load_supplementary()``
  injection path with no allowlist. From that point onward, any new
  loader added to FeatureMerger silently joined the production champion
  at the next weekly retrain — bypassing the shadow→promotion gate.
- The 6-3 22:00 0-recommendation incident exposed the path. This file
  is the single gate: the live champion may only consume loaders
  listed in ``PRODUCTION_SUPPLEMENTARY_GROUPS``. Adding to this tuple
  must be a deliberate edit that someone reviews.

Promotion workflow:
- New loader lands in :class:`models.feature_merger.FeatureMerger` AND in
  ``SHADOW_SUPPLEMENTARY_GROUPS``.
- Shadow backtests / hold-out IC compare must show net positive
  cost-adjusted improvement.
- Only then does the loader's group name move into
  ``PRODUCTION_SUPPLEMENTARY_GROUPS`` (this file), with a comment
  recording the promotion evidence.
"""


import os

# ---------------------------------------------------------------------------
# Model profile machinery (cx round 10, 2026-06-04)
# ---------------------------------------------------------------------------
# History the user established this evening:
#   - commit 95cd256 (2026-05-12) opened the暗道 in scripts/train_lgb.py
#     that auto-injected EVERY FeatureMerger loader into training. No
#     shadow→promote gate.
#   - The first weekly retrain after that (~2026-05-23) wrote a 242-dim
#     model.pkl over the previous 174 / xgb_174 champion binary.
#   - A 16-day held-out comparison (data/storage/pit_baseline_comparison.json)
#     shows 242 "less bad" than 158, but IC is still negative and the
#     sample is too thin to confirm 242 ≻ 174.
#   - Meanwhile xgb_174 evidence still exists: artifact metrics.json
#     (RankIC 0.05117, ICIR 0.646), phase4 backtest (cost-adjusted
#     Sharpe 1.79), and feature_cache_174_holder_regime_ma.parquet.
#
# Tonight's safest fix is to expose the profile choice EXPLICITLY so
# the team can flip it the moment 174 is retrained — but keep the
# runtime default at xgb_242 because the 174 model binary is gone and
# changing the default would make Monday's cron refuse to serve.
#
# DO NOT treat ``xgb_242`` as validated. It is grandfathered pending
# task #112 (retrain 174 + 24-split + cost-adjusted backtest →
# challenge gate → maybe flip default).
PRODUCTION_MODEL_PROFILE: str = os.environ.get(
    "PRODUCTION_MODEL_PROFILE", "xgb_209",
).strip().lower()
# 2026-06-06: default flipped xgb_242 → xgb_209 after Phase B.4
# 24-split verdict (docs/phase_b4_verdict_20260606.md) showed
# xgb_209 wins on every metric: RankIC +0.0072, ICIR +0.101 (+40%
# stability), Spread20 +38 bps (more than 2x). Retrain on
# end-date 2026-06-05 produced Sp20 81 bps. Phase B.5 confirmed
# Bucket B groups are neutral (no further drops). Phase B.6 showed
# the LLM event group hurts slightly (-0.0012 RankIC) so xgb_209_llm
# stays shadow. Rollback path: set
# ``PRODUCTION_MODEL_PROFILE=xgb_242`` in the cron env; the legacy
# binary lgb_model_xgb_242.pkl + contract are still on disk.


# Supplementary loader groups per profile. Adding a NEW loader to
# FeatureMerger does NOT automatically join any profile — the group
# must be listed here, and that change is the explicit promotion act.
SUPPLEMENTARY_GROUPS_BY_PROFILE: dict[str, tuple[str, ...]] = {
    "xgb_242": (
        "fundamental",
        "capital_flow",
        "macro_zero_baseline",
        "shareholder",
        "valuation",
        "northbound",
        "quality",
        "st_daily_basic",
        "st_moneyflow",
        "st_holder_number",
        "cross_market_regime",
    ),
    # xgb_209 — xgb_242 minus the Phase B Bucket A net-negative trio
    # (cross_market_regime, capital_flow, shareholder). Promoted by
    # docs/phase_b4_verdict_20260606.md after 24-split RankIC +0.0345
    # vs xgb_242's +0.0273 and Spread20 73 bps vs 35 bps; retrain on
    # end-date 2026-06-05 produced Spread20 81 bps. 8 supp groups,
    # 51 supp cols, 158 alpha158 → 209 total.
    "xgb_209": (
        "fundamental",
        "macro_zero_baseline",
        "valuation",
        "northbound",
        "quality",
        "st_daily_basic",
        "st_moneyflow",
        "st_holder_number",
    ),
    # xgb_209_chain — CANDIDATE profile, NOT YET PRODUCTION.
    # xgb_209 + global supply chain rule-based factors (6 numeric cols).
    # Shadow contract; promotion gated by ablation evidence.
    "xgb_209_chain": (
        "fundamental",
        "macro_zero_baseline",
        "valuation",
        "northbound",
        "quality",
        "st_daily_basic",
        "st_moneyflow",
        "st_holder_number",
        "global_chain",
    ),
    # xgb_209_chain_llm — CANDIDATE profile for Phase B.7 ablation.
    # xgb_209 + global supply chain LLM-extracted factors (same 6 cols
    # as xgb_209_chain but from extract_global_chain_llm.py output).
    # Differs from xgb_209_chain only in event source.
    "xgb_209_chain_llm": (
        "fundamental",
        "macro_zero_baseline",
        "valuation",
        "northbound",
        "quality",
        "st_daily_basic",
        "st_moneyflow",
        "st_holder_number",
        "global_chain_llm",
    ),
    # xgb_209_pbc — CANDIDATE profile for PE-1 PBOC liquidity input.
    # xgb_209 + 4 PBC liquidity cols (zscore_20d / easing_dummy /
    # tightening_dummy / short_rate_pressure). These are MARKET-level
    # signals broadcast to all stocks per date — same shape as
    # cross_market_regime. Shadow until B.7-style LOO confirms lift.
    "xgb_209_pbc": (
        "fundamental",
        "macro_zero_baseline",
        "valuation",
        "northbound",
        "quality",
        "st_daily_basic",
        "st_moneyflow",
        "st_holder_number",
        "pbc_liquidity",
    ),
    # xgb_209_guba — CANDIDATE profile, NOT YET PRODUCTION.
    # xgb_209 + 3 Eastmoney Guba popularity cols
    # (popularity_rank / rank_change / popularity_score). Same shadow
    # contract as xgb_209_llm: the loader is wired in feature_merger.py
    # so a Phase B-style LOO can ablate it, but production stays on
    # xgb_209 until ablation evidence shows ΔRankIC ≥ +0.005.
    "xgb_209_guba": (
        "fundamental",
        "macro_zero_baseline",
        "valuation",
        "northbound",
        "quality",
        "st_daily_basic",
        "st_moneyflow",
        "st_holder_number",
        "guba",
    ),
    # xgb_209_llm — CANDIDATE profile, next-champion under shadow.
    # xgb_209 plus the LLM event factor group (12 cols after the
    # 2026-06-07 L1 fact-count rebuild: 5 legacy impact/sentiment
    # + 7 fact-count positive_3d/negative_3d/price_sensitive_3d/
    # official_3d/count_3d/repeated_ratio_3d/event_intensity).
    # B.6.3 24-split verdict 2026-06-07: ΔRankIC +0.0044 (88% of
    # +0.005 gate), ΔSp20 +17.62 bps (+24%), ΔICIR +0.031. Below
    # strict RankIC gate but massive on Spread20 — operator chose
    # conservative path: shadow paper-trade 5+ trading days starting
    # Monday before flipping production default. See
    # docs/phase_b6_3_llm_24split_verdict_20260607.md.
    "xgb_209_llm": (
        "fundamental",
        "macro_zero_baseline",
        "valuation",
        "northbound",
        "quality",
        "st_daily_basic",
        "st_moneyflow",
        "st_holder_number",
        "llm_event",
    ),
    # xgb_209_xwlb — CANDIDATE profile for cx C.P1 #2 (2026-06-07).
    # xgb_209 + XWLB (新闻联播) theme factors broadcast to stock
    # baskets via config/xwlb_theme_baskets.yaml (C.P1 #3 mapper).
    # 4 supp cols: xwlb_theme_mention_count_1d /
    # xwlb_theme_mention_count_5d / xwlb_theme_consecutive_days /
    # xwlb_theme_priority_5d_max. Profile gated by candidate basket
    # coverage — production stays on xgb_209 until ablation evidence
    # confirms lift over the (partial) basket map.
    "xgb_209_xwlb": (
        "fundamental",
        "macro_zero_baseline",
        "valuation",
        "northbound",
        "quality",
        "st_daily_basic",
        "st_moneyflow",
        "st_holder_number",
        "xinwen_lianbo",
    ),
    # xgb_174 — Alpha158 (158) + capital_flow (3) + qlib-custom (13) =
    # 174 features. Both injection paths (FeatureMerger supplementary
    # for capital_flow + FeatureMerger.inject_qlib_custom_factors_into_handler
    # for the expression factors) ARE wired in scripts/train_lgb.py,
    # models/short_term.py and models/production_inference.py since
    # cx round 10 + round 16 (2026-06-04). The PROFILE_EXPECTED_COUNTS
    # block below pins the exact contract.
    "xgb_174": (
        "capital_flow",
    ),
}


if PRODUCTION_MODEL_PROFILE not in SUPPLEMENTARY_GROUPS_BY_PROFILE:
    raise RuntimeError(
        f"Unknown PRODUCTION_MODEL_PROFILE={PRODUCTION_MODEL_PROFILE!r}. "
        f"Allowed: {list(SUPPLEMENTARY_GROUPS_BY_PROFILE)}."
    )


# Resolved at import time. Downstream code should keep importing
# PRODUCTION_SUPPLEMENTARY_GROUPS as before; the profile mechanism
# selects which underlying tuple it points at.
PRODUCTION_SUPPLEMENTARY_GROUPS: tuple[str, ...] = (
    SUPPLEMENTARY_GROUPS_BY_PROFILE[PRODUCTION_MODEL_PROFILE]
)
# NOTE: every group above entered production via commit 95cd256 / its
# follow-ups WITHOUT a documented shadow→promotion record. Task #112
# tracks the 174-vs-242 24-split + cost-adjusted backtest that must
# retroactively justify the xgb_242 profile (or restore xgb_174 as
# default). Until then treat the current profile as
# "frozen-and-grandfathered", not "validated".


# cx batch D P2 #6 (2026-06-07): SHADOW_SUPPLEMENTARY_GROUPS is the
# explicit pool for loaders wired through FeatureMerger that are
# CURRENTLY consumed by one or more candidate xgb_209_* profiles but
# have NOT yet earned promotion into PRODUCTION_SUPPLEMENTARY_GROUPS.
# Pre-fix this was hardcoded ``()`` while six candidate profiles
# (xgb_209_chain / _chain_llm / _pbc / _guba / _llm) already consumed
# loader groups beyond the production tuple — the staging pool was
# effectively undocumented. New loaders MUST land here first;
# promotion to PRODUCTION_SUPPLEMENTARY_GROUPS only after ablation
# evidence (Phase B-style 24-split or hold-out LOO, ΔRankIC ≥ +0.005
# or net-positive cost-adjusted backtest).
#
# Derived programmatically: union of supp groups across every candidate
# xgb_209_* profile, minus what's already in PRODUCTION_SUPPLEMENTARY_GROUPS.
# Sorted for stable diffs in code review. xgb_174 / xgb_242 are
# grandfathered productions, not candidates, so excluded from the union.
_CANDIDATE_209_PROFILES: tuple[str, ...] = (
    "xgb_209_chain",
    "xgb_209_chain_llm",
    "xgb_209_pbc",
    "xgb_209_guba",
    "xgb_209_llm",
)


def _derive_shadow_supplementary_groups() -> tuple[str, ...]:
    """Union of candidate-profile supp groups minus production groups."""
    union: set[str] = set()
    for profile in _CANDIDATE_209_PROFILES:
        for group in SUPPLEMENTARY_GROUPS_BY_PROFILE.get(profile, ()):
            union.add(group)
    return tuple(sorted(union - set(PRODUCTION_SUPPLEMENTARY_GROUPS)))


SHADOW_SUPPLEMENTARY_GROUPS: tuple[str, ...] = _derive_shadow_supplementary_groups()
# Loader groups that exist on FeatureMerger and are consumed by at least
# one candidate xgb_209_* profile but have NOT yet earned a slot in
# production. New loaders MUST land here first; promotion to
# PRODUCTION_SUPPLEMENTARY_GROUPS only after ablation evidence.


# Explicit sentinel for research scripts that legitimately want every
# loader (ablation studies, factor scans). The sentinel exists so the
# default of ``_load_supplementary(groups=...)`` can stay STRICT (must be
# explicit) without forcing research code to hand-list 11 group names.
RESEARCH_ALL_LOADERS: str = "_research_all_"


# ---------------------------------------------------------------------------
# cx F.P2 #6 (2026-06-07): pin macro_zero_baseline column NAMES
# ---------------------------------------------------------------------------
# Pre-fix ``FeatureMerger._load_macro`` derived its emitted column list
# from whatever ``macro_features.parquet`` happened to have on disk. The
# parquet is a TEMPLATE — a single-row snapshot whose schema can drift
# as the collector script adds/renames columns. Any such drift would
# silently change the supplementary dimension (51 → 52 → 50 → …) under
# the production model, and the count gate at
# ``assert_profile_dimensions`` would refuse the contract.
#
# The fix is to take the dimension OUT of the template's hands: the
# loader reads ONLY the names below, emits zeros for them, and ignores
# whatever else the template has. The names match what the live
# production contract artifacts already encode (see
# data/storage/production_feature_contract_xgb_209.json), so this is
# strictly a freeze of current state — not a behaviour change today.
#
# To add or remove a macro column SAFELY: edit this tuple, regenerate
# the production contract via ``scripts/train_lgb.py``, and update
# ``PROFILE_EXPECTED_COUNTS[*]["supplementary"]`` for any affected
# profile in the same commit.
MACRO_ZERO_BASELINE_COLS: tuple[str, ...] = (
    "bond_10y",
    "bond_1y",
    "term_spread",
    "usdcny",
    "copper_close",
    "iron_ore_close",
    "crude_oil_close",
    "gold_futures_close",
    "pmi",
    "cpi",
)


# ---------------------------------------------------------------------------
# Qlib custom expression profile (cx round 10 follow-up, 2026-06-04)
# ---------------------------------------------------------------------------
# The xgb_174 profile needs not just FeatureMerger groups but ALSO a set
# of Qlib expression-language factors (PE / PB / Turn / amount + their
# momenta / vol). These come from ``D.features(instruments, exprs, ...)``
# at training and inference time, not from parquet loaders. Recording
# them here so train_lgb / production_inference can dispatch the
# correct injection.

QLIB_CUSTOM_FACTORS_BY_PROFILE: dict[str, tuple[tuple[str, str], ...]] = {
    "xgb_174": (
        ("pe",              "$pe"),
        ("pb",              "$pb"),
        ("turn_raw",        "$turn"),
        ("amount_raw",      "$amount"),
        ("pe_mom20",        "$pe / Ref($pe, 20) - 1"),
        ("pb_mom20",        "$pb / Ref($pb, 20) - 1"),
        ("turn_anom20",     "$turn / Mean($turn, 20)"),
        ("turn_anom60",     "$turn / Mean($turn, 60)"),
        ("amount_anom20",   "$amount / Mean($amount, 20)"),
        ("turn_vol20",      "Std($turn, 20)"),
        ("ep",              "1.0 / If(Abs($pe) > 0.01, $pe, 1.0)"),
        ("bp",              "1.0 / If(Abs($pb) > 0.01, $pb, 1.0)"),
        ("price_pos20",     "($close - Min($close, 20)) / (Max($close, 20) - Min($close, 20) + 1e-8)"),
    ),
    "xgb_242": (),  # 242 path is FeatureMerger-only
}


def current_profile_qlib_custom_factors() -> tuple[tuple[str, str], ...]:
    """Qlib expression factors for the currently-selected profile."""
    return QLIB_CUSTOM_FACTORS_BY_PROFILE.get(PRODUCTION_MODEL_PROFILE, ())


# Expected exact column counts per profile. Used by the
# ``assert_profile_dimensions`` helper so train/inference paths cannot
# silently produce a wrong-dim artifact.
PROFILE_EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "xgb_242": {
        "alpha158": 158,
        "supplementary": 84,
        "qlib_custom": 0,
        "total": 242,
    },
    # xgb_209 — Phase B drop set: cross_market_regime (27) +
    # capital_flow (3) + shareholder (3) = 33 cols dropped from 242.
    "xgb_209": {
        "alpha158": 158,
        "supplementary": 51,
        "qlib_custom": 0,
        "total": 209,
    },
    # xgb_209_pbc — xgb_209 + 4 PBC liquidity cols.
    "xgb_209_pbc": {
        "alpha158": 158,
        "supplementary": 55,    # 51 base + 4 PBC
        "qlib_custom": 0,
        "total": 213,
    },
    # xgb_209_guba — xgb_209 + 3 guba popularity cols.
    "xgb_209_guba": {
        "alpha158": 158,
        "supplementary": 54,    # 51 base + 3 guba
        "qlib_custom": 0,
        "total": 212,
    },
    # xgb_209_chain / xgb_209_chain_llm — 6 supply-chain alpha cols
    # (global_chain_alpha + event_count + pos_score + neg_score +
    # company_level_alpha + industry_level_alpha; "level" is text and
    # excluded by the loader).
    "xgb_209_chain": {
        "alpha158": 158,
        "supplementary": 57,    # 51 base + 6 chain
        "qlib_custom": 0,
        "total": 215,
    },
    "xgb_209_chain_llm": {
        "alpha158": 158,
        "supplementary": 57,    # same shape as chain rule-based
        "qlib_custom": 0,
        "total": 215,
    },
    # xgb_209_llm — xgb_209 + 12 LLM event factor cols. The L1
    # fact-count rebuild on 2026-06-06 21:25 grew the parquet from
    # 5 legacy cols (impact_1d/5d_decayed + sentiment + count_5d +
    # confidence) to 12 (the 5 legacy + 7 fact-count: positive_3d,
    # negative_3d, price_sensitive_3d, official_3d, count_3d,
    # repeated_ratio_3d, event_intensity). The candidate profile
    # tracks this schema; bump the count if a future schema change
    # adds more cols (the build_feature_cache_209_llm contract gate
    # will hard-fail unless this count matches).
    "xgb_209_llm": {
        "alpha158": 158,
        "supplementary": 63,    # 51 base + 12 LLM
        "qlib_custom": 0,
        "total": 221,
    },
    # xgb_209_xwlb — xgb_209 (51 base supp) + 4 XWLB theme factors
    # broadcast to stock baskets. cx C.P1 #2.
    "xgb_209_xwlb": {
        "alpha158": 158,
        "supplementary": 55,    # 51 base + 4 XWLB
        "qlib_custom": 0,
        "total": 213,
    },
    "xgb_174": {
        "alpha158": 158,
        "supplementary": 3,    # capital_flow only
        "qlib_custom": 13,     # PE/PB/Turn/amount + derivatives
        "total": 174,
    },
}


# 2026-06-04 cx round 10 Option B: profile-aware model file naming.
# Pre-fix every profile wrote to the same ``lgb_model.pkl`` so a
# retrain at one profile overwrote another profile's deployable
# binary (that's how the 5-23 weekly retrain destroyed xgb_174).
# Now: each profile lives at ``lgb_model_{profile}.pkl``. A legacy
# symlink ``lgb_model.pkl`` points at the currently-active profile
# so any hardcoded paths in tests / monitoring scripts keep working
# during migration.
LEGACY_MODEL_FILENAME: str = "lgb_model.pkl"


def production_model_filename(profile: str | None = None) -> str:
    """Filename (NOT path) for the production model of ``profile``.

    Args:
        profile: profile name. None → PRODUCTION_MODEL_PROFILE.

    Returns:
        e.g. ``"lgb_model_xgb_242.pkl"`` / ``"lgb_model_xgb_174.pkl"``.

    Raises:
        RuntimeError: when ``profile`` is not a known profile.
    """
    p = (profile or PRODUCTION_MODEL_PROFILE).strip().lower()
    if p not in SUPPLEMENTARY_GROUPS_BY_PROFILE:
        raise RuntimeError(
            f"Unknown profile {p!r}. Allowed: "
            f"{list(SUPPLEMENTARY_GROUPS_BY_PROFILE)}"
        )
    return f"lgb_model_{p}.pkl"


LEGACY_CONTRACT_FILENAME: str = "production_feature_contract.json"


def production_contract_filename(profile: str | None = None) -> str:
    """Filename (NOT path) for the contract artifact of ``profile``.

    2026-06-04 cx round 22 P0-1: model files were split per profile in
    the Option B refactor, but ``production_feature_contract.json``
    stayed single — training xgb_174 would overwrite the 242 contract
    even though the 242 model binary still existed alongside the new
    174 binary. Now both ARE profile-specific:
        production_feature_contract_xgb_242.json
        production_feature_contract_xgb_174.json
    The legacy ``production_feature_contract.json`` filename is
    maintained as a symlink to the active profile's contract.
    """
    p = (profile or PRODUCTION_MODEL_PROFILE).strip().lower()
    if p not in SUPPLEMENTARY_GROUPS_BY_PROFILE:
        raise RuntimeError(
            f"Unknown profile {p!r}. Allowed: "
            f"{list(SUPPLEMENTARY_GROUPS_BY_PROFILE)}"
        )
    return f"production_feature_contract_{p}.json"


def assert_profile_dimensions(
    *, alpha_count: int, supp_count: int, custom_count: int,
    profile: str | None = None,
) -> None:
    """Hard-fail unless the (alpha + supp + custom) totals match the
    currently-selected profile's contract. cx round 16 P1-3: previously
    train/serve paths only checked ``supp + custom > 0`` — a partial
    custom factor failure on xgb_174 would still pass that floor.
    """
    profile = (profile or PRODUCTION_MODEL_PROFILE).strip().lower()
    spec = PROFILE_EXPECTED_COUNTS.get(profile)
    if spec is None:
        raise RuntimeError(
            f"assert_profile_dimensions: unknown profile {profile!r}"
        )
    if alpha_count != spec["alpha158"]:
        raise RuntimeError(
            f"profile={profile} expected {spec['alpha158']} Alpha158 cols "
            f"but got {alpha_count}"
        )
    if supp_count != spec["supplementary"]:
        raise RuntimeError(
            f"profile={profile} expected {spec['supplementary']} supplementary "
            f"cols but got {supp_count}"
        )
    if custom_count != spec["qlib_custom"]:
        raise RuntimeError(
            f"profile={profile} expected {spec['qlib_custom']} qlib-custom "
            f"cols but got {custom_count}"
        )
    total = alpha_count + supp_count + custom_count
    if total != spec["total"]:
        raise RuntimeError(
            f"profile={profile} total dim mismatch: "
            f"{alpha_count}+{supp_count}+{custom_count}={total} "
            f"!= contract {spec['total']}"
        )

