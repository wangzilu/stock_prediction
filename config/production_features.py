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
    "PRODUCTION_MODEL_PROFILE", "xgb_242",
).strip().lower()


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


SHADOW_SUPPLEMENTARY_GROUPS: tuple[str, ...] = ()
# Loader groups that exist on FeatureMerger but have NOT yet earned
# a slot in production. Empty today — every existing loader is
# grandfathered into PRODUCTION_SUPPLEMENTARY_GROUPS pending #102.


# Explicit sentinel for research scripts that legitimately want every
# loader (ablation studies, factor scans). The sentinel exists so the
# default of ``_load_supplementary(groups=...)`` can stay STRICT (must be
# explicit) without forcing research code to hand-list 11 group names.
RESEARCH_ALL_LOADERS: str = "_research_all_"


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
    "xgb_174": {
        "alpha158": 158,
        "supplementary": 3,    # capital_flow only
        "qlib_custom": 13,     # PE/PB/Turn/amount + derivatives
        "total": 174,
    },
}


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

