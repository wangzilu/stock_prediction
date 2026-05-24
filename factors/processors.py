"""Phase 4L — Unified cross-sectional factor processing pipeline.

All new factors must pass through this pipeline before entering the model.
Functions operate on pd.Series with (datetime, instrument) MultiIndex,
applying transformations cross-sectionally (per date).

Standard pipeline:  fillna -> winsorize -> zscore
Full pipeline:      fillna -> winsorize -> industry_neutralize -> size_neutralize -> zscore

Usage:
    from factors.processors import full_pipeline, compute_residual_ic

    processed = full_pipeline(raw_factor)
    processed_neutral = full_pipeline(
        raw_factor, industry=ind_labels, size=log_mcap,
        steps=["fillna", "winsorize", "industry_neutralize", "size_neutralize", "zscore"],
    )
    residual = compute_residual_ic(new_factor, champion_pred, fwd_returns)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

__all__ = [
    "fillna_forward",
    "winsorize_mad",
    "zscore",
    "rank_normalize",
    "industry_neutralize",
    "size_neutralize",
    "full_pipeline",
    "compute_residual_ic",
]

# ---------------------------------------------------------------------------
# 1. Forward fill NaN
# ---------------------------------------------------------------------------

def fillna_forward(factor: pd.Series, max_gap: int = 5) -> pd.Series:
    """Forward-fill missing values within each stock, up to *max_gap* days.

    Parameters
    ----------
    factor : pd.Series
        Series with (datetime, instrument) MultiIndex.
    max_gap : int
        Maximum number of consecutive NaNs to fill per instrument.

    Returns
    -------
    pd.Series
        Factor with forward-filled values.
    """
    if factor.empty:
        return factor.copy()

    inst_level = 1  # instrument is the second level
    result = factor.copy()

    # Unstack to (datetime x instrument), forward fill per column, restack
    unstacked = result.unstack(level=inst_level)
    filled = unstacked.ffill(limit=max_gap)
    try:
        restacked = filled.stack(dropna=False)
    except (ValueError, TypeError):
        # pandas >= 2.1 removed dropna param from stack
        restacked = filled.stack(future_stack=True)

    # Restore original index order and name
    restacked.index.names = factor.index.names
    return restacked.reindex(factor.index)


# ---------------------------------------------------------------------------
# 2. MAD winsorization
# ---------------------------------------------------------------------------

def winsorize_mad(factor: pd.Series, n_mad: float = 5.0) -> pd.Series:
    """Per-date cross-sectional MAD winsorization.

    Clips values to [median - n_mad * MAD_sigma, median + n_mad * MAD_sigma]
    where MAD_sigma = 1.4826 * MAD (consistency factor for normal distribution).

    Parameters
    ----------
    factor : pd.Series
        Series with (datetime, instrument) MultiIndex.
    n_mad : float
        Number of MAD-based standard deviations for clipping bounds.

    Returns
    -------
    pd.Series
        Winsorized factor.
    """
    if factor.empty:
        return factor.copy()

    date_level = factor.index.names[0] if factor.index.names[0] else 0

    def _clip_group(g: pd.Series) -> pd.Series:
        median = g.median()
        mad = (g - median).abs().median()
        sigma_mad = 1.4826 * mad
        if sigma_mad < 1e-12:
            return g
        lower = median - n_mad * sigma_mad
        upper = median + n_mad * sigma_mad
        return g.clip(lower=lower, upper=upper)

    return factor.groupby(level=date_level, group_keys=False).apply(_clip_group)


# ---------------------------------------------------------------------------
# 3. Z-score
# ---------------------------------------------------------------------------

def zscore(factor: pd.Series) -> pd.Series:
    """Per-date cross-sectional z-score normalization (mean=0, std=1).

    Parameters
    ----------
    factor : pd.Series
        Series with (datetime, instrument) MultiIndex.

    Returns
    -------
    pd.Series
        Z-scored factor. Dates with std=0 produce NaN.
    """
    if factor.empty:
        return factor.copy()

    date_level = factor.index.names[0] if factor.index.names[0] else 0

    def _zscore_group(g: pd.Series) -> pd.Series:
        mu = g.mean()
        sigma = g.std(ddof=1)
        if sigma is None or sigma < 1e-12:
            return pd.Series(np.nan, index=g.index)
        return (g - mu) / sigma

    return factor.groupby(level=date_level, group_keys=False).apply(_zscore_group)


# ---------------------------------------------------------------------------
# 4. Rank normalize
# ---------------------------------------------------------------------------

def rank_normalize(factor: pd.Series) -> pd.Series:
    """Per-date cross-sectional rank normalization, scaled to [-1, 1].

    Parameters
    ----------
    factor : pd.Series
        Series with (datetime, instrument) MultiIndex.

    Returns
    -------
    pd.Series
        Rank-normalized factor in [-1, 1].
    """
    if factor.empty:
        return factor.copy()

    date_level = factor.index.names[0] if factor.index.names[0] else 0

    def _rank_group(g: pd.Series) -> pd.Series:
        n = g.notna().sum()
        if n < 2:
            return pd.Series(np.nan, index=g.index)
        ranked = g.rank(method="average", na_option="keep")
        # Scale from [1, n] to [-1, 1]
        return 2.0 * (ranked - 1.0) / (n - 1.0) - 1.0

    return factor.groupby(level=date_level, group_keys=False).apply(_rank_group)


# ---------------------------------------------------------------------------
# 5. Industry neutralize
# ---------------------------------------------------------------------------

def industry_neutralize(factor: pd.Series, industry: pd.Series) -> pd.Series:
    """Regress out industry means (demean within each industry per date).

    Parameters
    ----------
    factor : pd.Series
        Series with (datetime, instrument) MultiIndex.
    industry : pd.Series
        Industry labels with the same index.

    Returns
    -------
    pd.Series
        Industry-neutralized factor (residual after removing industry means).
    """
    if factor.empty:
        return factor.copy()

    # Align indices
    common = factor.index.intersection(industry.index)
    f = factor.loc[common]
    ind = industry.loc[common]

    date_level = f.index.names[0] if f.index.names[0] else 0

    def _demean(g: pd.Series) -> pd.Series:
        idx = g.index
        g_ind = ind.loc[idx]
        # Group by industry within this date, subtract industry mean
        industry_means = g.groupby(g_ind).transform("mean")
        return g - industry_means

    result = f.groupby(level=date_level, group_keys=False).apply(_demean)
    # Reindex back to original factor index (fill missing with NaN)
    return result.reindex(factor.index)


# ---------------------------------------------------------------------------
# 6. Size neutralize
# ---------------------------------------------------------------------------

def size_neutralize(factor: pd.Series, size: pd.Series) -> pd.Series:
    """Regress out size (log market cap) effect per date, return residuals.

    Parameters
    ----------
    factor : pd.Series
        Series with (datetime, instrument) MultiIndex.
    size : pd.Series
        Log market cap with the same index.

    Returns
    -------
    pd.Series
        Size-neutralized factor (OLS residuals after regressing on size).
    """
    if factor.empty:
        return factor.copy()

    common = factor.index.intersection(size.index)
    f = factor.loc[common]
    s = size.loc[common]

    date_level = f.index.names[0] if f.index.names[0] else 0

    def _regress_out_size(g: pd.Series) -> pd.Series:
        idx = g.index
        g_size = s.loc[idx]
        valid = g.notna() & g_size.notna()
        if valid.sum() < 3:
            return g
        y = g[valid].values
        x = g_size[valid].values
        # OLS: y = a + b*x + residual
        x_mat = np.column_stack([np.ones(len(x)), x])
        try:
            beta, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
            residuals = y - x_mat @ beta
        except np.linalg.LinAlgError:
            return g
        out = g.copy()
        out[valid] = residuals
        return out

    result = f.groupby(level=date_level, group_keys=False).apply(_regress_out_size)
    return result.reindex(factor.index)


# ---------------------------------------------------------------------------
# 7. Full pipeline
# ---------------------------------------------------------------------------

# Mapping from step name to function (and required extra args)
_STEP_REGISTRY = {
    "fillna": (fillna_forward, []),
    "winsorize": (winsorize_mad, []),
    "zscore": (zscore, []),
    "rank_normalize": (rank_normalize, []),
    "industry_neutralize": (industry_neutralize, ["industry"]),
    "size_neutralize": (size_neutralize, ["size"]),
}

DEFAULT_STEPS = ["fillna", "winsorize", "zscore"]


def full_pipeline(
    factor: pd.Series,
    industry: Optional[pd.Series] = None,
    size: Optional[pd.Series] = None,
    steps: Optional[list[str]] = None,
) -> pd.Series:
    """Run the standard cross-sectional factor processing pipeline.

    Parameters
    ----------
    factor : pd.Series
        Raw factor with (datetime, instrument) MultiIndex.
    industry : pd.Series, optional
        Industry labels (required if "industry_neutralize" in steps).
    size : pd.Series, optional
        Log market cap (required if "size_neutralize" in steps).
    steps : list[str], optional
        Processing steps to apply in order.
        Default: ["fillna", "winsorize", "zscore"].

    Returns
    -------
    pd.Series
        Processed factor.
    """
    if steps is None:
        steps = DEFAULT_STEPS

    extras = {"industry": industry, "size": size}
    result = factor.copy()

    for step_name in steps:
        if step_name not in _STEP_REGISTRY:
            raise ValueError(
                f"Unknown step '{step_name}'. "
                f"Available: {list(_STEP_REGISTRY.keys())}"
            )
        func, required_extras = _STEP_REGISTRY[step_name]
        kwargs = {}
        for extra_name in required_extras:
            val = extras.get(extra_name)
            if val is None:
                raise ValueError(
                    f"Step '{step_name}' requires '{extra_name}' argument, "
                    f"but it was not provided."
                )
            kwargs[extra_name] = val

        result = func(result, **kwargs)
        logger.debug(f"Pipeline step '{step_name}': nan={result.isna().sum()}")

    return result


# ---------------------------------------------------------------------------
# 8. Residual IC
# ---------------------------------------------------------------------------

def compute_residual_ic(
    new_factor: pd.Series,
    champion_pred: pd.Series,
    returns: pd.Series,
    min_obs: int = 5,
) -> dict:
    """Compute IC of new_factor after regressing out champion predictions.

    Measures the marginal information content of a new factor beyond
    what the current champion model already captures.

    Parameters
    ----------
    new_factor : pd.Series
        Candidate factor with (datetime, instrument) MultiIndex.
    champion_pred : pd.Series
        Champion model predictions with same index.
    returns : pd.Series
        Forward returns with same index.
    min_obs : int
        Minimum observations per date for valid computation.

    Returns
    -------
    dict
        residual_ic : float — mean Pearson IC of residualized factor vs returns
        residual_rank_ic : float — mean Spearman IC of residualized factor vs returns
        raw_ic : float — mean Pearson IC of raw new_factor vs returns (for comparison)
        marginal_value : bool — True if residual_rank_ic > 0.005
    """
    # Align all three series on common non-NaN index
    common = (
        new_factor.dropna().index
        .intersection(champion_pred.dropna().index)
        .intersection(returns.dropna().index)
    )
    if len(common) == 0:
        return {
            "residual_ic": 0.0,
            "residual_rank_ic": 0.0,
            "raw_ic": 0.0,
            "marginal_value": False,
        }

    f = new_factor.loc[common]
    c = champion_pred.loc[common]
    r = returns.loc[common]

    date_level = f.index.names[0] if f.index.names[0] else 0
    dates = f.index.get_level_values(date_level).unique()

    residual_ics = []
    residual_rics = []
    raw_ics = []

    for dt in dates:
        try:
            f_day = f.xs(dt, level=date_level)
            c_day = c.xs(dt, level=date_level)
            r_day = r.xs(dt, level=date_level)
        except KeyError:
            continue

        valid = f_day.index.intersection(c_day.index).intersection(r_day.index)
        if len(valid) < min_obs:
            continue

        fv = f_day.loc[valid].values
        cv = c_day.loc[valid].values
        rv = r_day.loc[valid].values

        # Check for valid finite data
        mask = np.isfinite(fv) & np.isfinite(cv) & np.isfinite(rv)
        if mask.sum() < min_obs:
            continue
        fv, cv, rv = fv[mask], cv[mask], rv[mask]

        # Regress new_factor on champion to get residual
        x_mat = np.column_stack([np.ones(len(cv)), cv])
        try:
            beta, _, _, _ = np.linalg.lstsq(x_mat, fv, rcond=None)
            residual = fv - x_mat @ beta
        except np.linalg.LinAlgError:
            continue

        # Residual IC (Pearson)
        try:
            ric_p, _ = stats.pearsonr(residual, rv)
            if np.isfinite(ric_p):
                residual_ics.append(ric_p)
        except Exception:
            pass

        # Residual RankIC (Spearman)
        try:
            ric_s, _ = stats.spearmanr(residual, rv)
            if np.isfinite(ric_s):
                residual_rics.append(ric_s)
        except Exception:
            pass

        # Raw IC for comparison
        try:
            raw_p, _ = stats.pearsonr(fv, rv)
            if np.isfinite(raw_p):
                raw_ics.append(raw_p)
        except Exception:
            pass

    residual_ic = float(np.mean(residual_ics)) if residual_ics else 0.0
    residual_rank_ic = float(np.mean(residual_rics)) if residual_rics else 0.0
    raw_ic = float(np.mean(raw_ics)) if raw_ics else 0.0
    marginal_value = abs(residual_rank_ic) > 0.005

    return {
        "residual_ic": round(residual_ic, 6),
        "residual_rank_ic": round(residual_rank_ic, 6),
        "raw_ic": round(raw_ic, 6),
        "marginal_value": marginal_value,
    }
