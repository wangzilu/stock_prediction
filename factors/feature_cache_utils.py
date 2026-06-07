"""Shared utilities for offline feature-cache joiners.

2026-06-08 (post-B.8 case-bug): every ``scripts/build_feature_cache_209_*.py``
joiner reads a factor parquet and reindexes it onto the base 209
production cache's ``(datetime, instrument)`` MultiIndex. Two repeated
silent failure modes have bitten us:

1. **Case mismatch.** Some producers (LLM event pipeline, sentiment,
   guba post-F.P1 #3) write ``qlib_code`` as UPPERCASE ``SH600000``,
   while the base cache uses lowercase ``sh600000`` (Qlib's canonical
   form for the 158 Alpha158 handler). The reindex then matches zero
   rows and ``fillna(0.0)`` produces a constant-zero factor column.
   The model trains as if the factor didn't exist; the LOO ablation
   shows only PRNG drift, not real signal. This is exactly how the
   B.6.3 24-split LLM verdict (+0.0044 RankIC) turned out to be
   illusory — see docs/phase_b8_4way_candidate_ablation_20260607.md.

2. **Date-range miss.** Factor parquet covers a date window that
   doesn't overlap the base cache (e.g. guba collector only had
   2026-05-22 → 2026-06-05 but base cache ended 2026-05-19). Reindex
   produces 0% coverage. Same constant-zero outcome.

Both failures are silent at training time — the model trains
successfully on dead columns. The B.7 chain ablation's "bit-identical
per-split metrics" was the same fingerprint: trees don't split on
columns that are >99.99% constant, so the LOO delta is just PRNG.

This module provides the two safety primitives every joiner MUST use:

* :func:`normalize_instrument_index` — case-insensitive canonicalisation
  to lowercase, the convention the base cache uses. Logs a WARNING when
  the producer parquet diverged from canonical so the upstream writer
  gets noticed.
* :func:`assert_join_coverage` — post-reindex sanity check. If the
  source factor parquet had rows but the reindexed frame has near-zero
  non-null coverage, raise with a clear root-cause message instead of
  letting a constant-zero column ship into model training.
"""
from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


# The base 209 production cache (and the 174-family cache it inherits
# from) writes instruments lowercase. Any factor parquet that joins
# onto this index MUST be lowercased before the join.
CANONICAL_INSTRUMENT_CASE = "lower"


def normalize_instrument_index(
    df: pd.DataFrame,
    *,
    level: int | str = "instrument",
    source_name: str = "factor",
) -> pd.DataFrame:
    """Lowercase the instrument level of a (datetime, instrument)
    MultiIndex in place, returning the (possibly rebuilt) DataFrame.

    Logs a WARNING when any value in the level needed normalisation —
    the upstream writer should be fixed to write canonical lowercase
    so this normalisation eventually becomes a no-op.
    """
    if not isinstance(df.index, pd.MultiIndex):
        return df
    try:
        current = df.index.get_level_values(level)
    except KeyError:
        return df
    lowered = current.astype(str).str.lower()
    if (current.astype(str) == lowered).all():
        return df  # already canonical, nothing to do
    n_changed = int((current.astype(str) != lowered).sum())
    logger.warning(
        "[case-norm] %s: lowercased %d / %d instrument keys to match "
        "base-cache canonical case. Fix the upstream writer to emit "
        "lowercase qlib_code so this normalisation becomes a no-op.",
        source_name, n_changed, len(current),
    )
    if isinstance(level, str):
        level_idx = df.index.names.index(level)
    else:
        level_idx = level
    new_levels = list(df.index.levels)
    new_levels[level_idx] = pd.Index(
        [str(x).lower() for x in df.index.levels[level_idx]]
    )
    df.index = df.index.set_levels(new_levels[level_idx], level=level_idx)
    return df


def assert_join_coverage(
    *,
    source_df: pd.DataFrame,
    reindexed: pd.DataFrame,
    factor_cols: Iterable[str],
    source_name: str = "factor",
    min_match_rows: int = 1,
    min_match_pct: float = 1e-5,
) -> None:
    """Fail loudly when a non-empty factor parquet reindexed to (nearly)
    zero rows on the base cache index.

    The default ``min_match_rows=1`` AND ``min_match_pct=1e-5`` (i.e.
    one in 100 k) is intentionally low — it catches the "complete miss"
    case without crying wolf on legitimately sparse signals. The B.7
    chain pipeline (~0.01 % non-zero) passes; the B.6.3 LLM bug (zero
    matches despite 198 k events) raises.

    Raises
    ------
    RuntimeError
        When ``len(source_df) > 0`` but the reindexed frame has fewer
        than ``min_match_rows`` AND less than ``min_match_pct`` of base
        rows carrying a non-null in any factor column.
    """
    cols = [c for c in factor_cols if c in reindexed.columns]
    if not cols:
        raise RuntimeError(
            f"[coverage-gate] {source_name}: none of the requested "
            f"factor columns {list(factor_cols)} survived reindex; "
            f"present cols = {list(reindexed.columns)[:8]}..."
        )
    if len(source_df) == 0:
        logger.warning(
            "[coverage-gate] %s: source parquet is empty; skipping "
            "coverage check. Downstream cache will be all-zero — that "
            "is the right behaviour but means the ablation cannot "
            "show signal until the producer is backfilled.",
            source_name,
        )
        return
    matched = int(reindexed[cols].notna().any(axis=1).sum())
    pct = matched / max(1, len(reindexed))
    if matched < min_match_rows or pct < min_match_pct:
        raise RuntimeError(
            f"[coverage-gate] {source_name}: source parquet had "
            f"{len(source_df)} rows, but reindex onto base index "
            f"({len(reindexed)} rows) matched only {matched} rows "
            f"({100 * pct:.4f} %). Floor is "
            f"max({min_match_rows} rows, {100 * min_match_pct:.4f} %). "
            f"Most likely cause: case mismatch on the instrument key "
            f"(see factors/feature_cache_utils.normalize_instrument_index) "
            f"or date-range mismatch (factor covers a window the base "
            f"cache does not). Inspect: "
            f"source_df.index[:3]={list(source_df.index[:3])} vs "
            f"base_sample={list(reindexed.index[:3])}. "
            f"This gate fires loudly precisely BECAUSE the silent "
            f"zero-column outcome let B.6.3's LLM verdict (+0.0044 "
            f"RankIC) ship as PRNG drift not real signal — refuse "
            f"to repeat that."
        )
    logger.info(
        "[coverage-gate] %s: %d / %d base rows carry a non-null factor "
        "value (%.4f %%). Above floor — proceeding.",
        source_name, matched, len(reindexed), 100 * pct,
    )
