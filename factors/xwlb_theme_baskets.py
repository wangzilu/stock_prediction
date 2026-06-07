"""XWLB (新闻联播) theme → A-share basket broadcast helper.

cx C.P1 #3 (2026-06-07):

Problem:
    ``scripts/extract_policy_events.py:1073`` writes XWLB events with
    ``stock_code = f"THEME_{theme.upper()}"`` and the downstream factor
    builder (``scripts/build_policy_factors.py``) writes
    ``xinwen_lianbo_theme_factors.parquet`` keyed by
    ``(datetime, "THEME_<UPPER>")``. Stock-level training samples
    NEVER match a ``THEME_<NAME>`` instrument code, so a naïve
    FeatureMerger reindex onto a stock-keyed MultiIndex misses every
    row — the XWLB factor is invisible to the model.

Fix:
    A starter theme→basket map at ``config/xwlb_theme_baskets.yaml``
    lists, for each high-confidence theme, the qlib instruments most
    closely tied to it. ``broadcast_theme_to_stocks`` replicates each
    (date, THEME_X) factor row onto every (date, STOCK_K) row where
    STOCK_K is in the basket. Themes not in the YAML are no-ops.

Limitations (intentionally explicit):
    - Basket coverage is partial (a dozen themes); expansion is a
      separate PR.
    - Membership is hand-curated, not derived from concept/industry
      tags. Quarterly refresh cadence is OK.
    - When a stock belongs to multiple themes mentioned the same day,
      this helper aggregates by ``max`` per factor — the strongest
      signal wins. This matches the underlying factor semantics:
      ``mention_count_5d`` / ``consecutive_days`` /
      ``priority_5d_max`` are all already "max-over-window" style
      signals.

API:
    - ``load_theme_baskets()`` → dict[theme_upper, list[stock]]
    - ``broadcast_theme_to_stocks(df, factor_cols, prefix="xwlb_")``
      → DataFrame indexed by (datetime, instrument=STOCK).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASKETS_YAML = PROJECT_ROOT / "config" / "xwlb_theme_baskets.yaml"
THEME_INSTRUMENT_PREFIX = "THEME_"


@lru_cache(maxsize=1)
def load_theme_baskets() -> dict[str, list[str]]:
    """Load the theme→basket map from YAML.

    Returns:
        ``{"THEME_<UPPER>": ["SH600519", "SZ000858", ...], ...}`` —
        the keys are already uppercase + prefixed to match the XWLB
        parquet's instrument codes. Empty dict if the YAML is missing
        or malformed (the loader logs a warning and the caller falls
        back to no broadcast, which leaves stock rows NaN → zero-fill).
    """
    if not BASKETS_YAML.exists():
        logger.warning("xwlb basket YAML missing: %s", BASKETS_YAML)
        return {}
    try:
        import yaml  # PyYAML
    except ImportError:
        logger.warning(
            "PyYAML not installed; cannot load xwlb theme baskets. "
            "Install pyyaml or skip xinwen_lianbo group.",
        )
        return {}
    try:
        raw = yaml.safe_load(BASKETS_YAML.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("xwlb basket YAML parse failed: %s", exc)
        return {}
    out: dict[str, list[str]] = {}
    for theme, stocks in raw.items():
        if not isinstance(stocks, list):
            continue
        key = f"{THEME_INSTRUMENT_PREFIX}{str(theme).upper()}"
        out[key] = [str(s).upper().strip() for s in stocks if s]
    logger.info(
        "xwlb theme baskets: %d themes, %d total stock-theme pairs",
        len(out), sum(len(v) for v in out.values()),
    )
    return out


def broadcast_theme_to_stocks(
    df: pd.DataFrame,
    factor_cols: list[str],
    prefix: str = "xwlb_",
) -> pd.DataFrame | None:
    """Replicate theme-keyed factor rows onto basket-member stocks.

    Args:
        df: long-form frame with at least ``datetime``, ``instrument``
            (``"THEME_<UPPER>"``) and the columns in ``factor_cols``.
        factor_cols: numeric columns to broadcast (do NOT pre-prefix;
            this function applies ``prefix`` so the output names are
            stable regardless of input).
        prefix: column-name prefix for the output frame
            (default ``"xwlb_"``).

    Returns:
        DataFrame indexed by (datetime, instrument=STOCK) with the
        prefixed factor columns. Stocks that belong to multiple themes
        on the same day get the per-column ``max`` across their themes
        (the strongest signal). Returns ``None`` if the basket map is
        empty (caller should fall back to NaN).
    """
    baskets = load_theme_baskets()
    if not baskets:
        return None
    if df is None or df.empty:
        return None
    keep_cols = ["datetime", "instrument"] + list(factor_cols)
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        logger.warning("xwlb broadcast: input missing cols %s", missing)
        return None
    work = df[keep_cols].copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    work = work.dropna(subset=["datetime"])
    if work.empty:
        return None

    # Explode each theme row into one row per basket stock.
    # Build a small helper frame keyed by theme instrument with a list
    # of basket stocks, then merge on instrument and explode.
    basket_df = pd.DataFrame(
        [
            {"instrument": theme, "_basket": stocks}
            for theme, stocks in baskets.items()
        ]
    )
    merged = work.merge(basket_df, on="instrument", how="inner")
    if merged.empty:
        # No themes in the parquet match the basket map's keys.
        return None
    merged = merged.explode("_basket")
    merged = merged.dropna(subset=["_basket"])
    merged = merged.rename(columns={"_basket": "stock"})

    # Aggregate per (date, stock) — max across multiple themes the
    # stock belongs to. Matches the factor semantics (all four
    # underlying cols are themselves max-over-window).
    rename_map = {c: f"{prefix}{c}" for c in factor_cols}
    agg_df = (
        merged.rename(columns=rename_map)
        .groupby(["datetime", "stock"], as_index=False)[list(rename_map.values())]
        .max()
    )
    agg_df = agg_df.rename(columns={"stock": "instrument"})
    agg_df = agg_df.set_index(["datetime", "instrument"])
    return agg_df
