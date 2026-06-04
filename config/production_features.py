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


PRODUCTION_SUPPLEMENTARY_GROUPS: tuple[str, ...] = (
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
)
# NOTE: every group above entered production via commit 95cd256 / its
# follow-ups WITHOUT a documented shadow→promotion record. Task #102
# tracks the 174-vs-242 hold-out backtest that must retroactively
# justify this list. Until that lands, treat this contract as
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

