"""Return attribution analysis for portfolio backtest.

Provides Brinson-style (allocation + selection + interaction) decomposition
and simple factor attribution.

Reference:
    Brinson, Hood & Beebower (1986) "Determinants of Portfolio Performance"
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def attribute_returns(
    daily_portfolio_returns: pd.DataFrame,
    daily_benchmark_returns: pd.DataFrame,
    portfolio_industry_weights: pd.DataFrame,
    benchmark_industry_weights: pd.DataFrame,
) -> dict:
    """Brinson-style attribution: allocation + selection + interaction.

    All inputs are DataFrames indexed by date with columns = industry names.

    Args:
        daily_portfolio_returns: (n_dates, n_industries) portfolio return
            per industry per day. Each cell = weighted-average return of
            portfolio stocks in that industry on that day.
        daily_benchmark_returns: (n_dates, n_industries) benchmark return
            per industry per day.
        portfolio_industry_weights: (n_dates, n_industries) portfolio weight
            in each industry per day (rows should sum to ~1).
        benchmark_industry_weights: (n_dates, n_industries) benchmark weight
            in each industry per day (rows should sum to ~1).

    Returns:
        dict with keys:
        - total_excess: float, cumulative portfolio - benchmark return
        - allocation_effect: float, sum over time of allocation effect
        - selection_effect: float, sum over time of selection effect
        - interaction_effect: float, sum over time of interaction effect
        - daily_allocation: pd.Series indexed by date
        - daily_selection: pd.Series indexed by date
        - daily_interaction: pd.Series indexed by date
        - industry_allocation: pd.Series indexed by industry (total contribution)
        - industry_selection: pd.Series indexed by industry (total contribution)
    """
    # Align all inputs to common dates and industries
    common_dates = (
        daily_portfolio_returns.index
        .intersection(daily_benchmark_returns.index)
        .intersection(portfolio_industry_weights.index)
        .intersection(benchmark_industry_weights.index)
    )
    common_industries = (
        daily_portfolio_returns.columns
        .intersection(daily_benchmark_returns.columns)
        .intersection(portfolio_industry_weights.columns)
        .intersection(benchmark_industry_weights.columns)
    )

    if len(common_dates) == 0 or len(common_industries) == 0:
        logger.warning(
            "No overlapping dates/industries for attribution. "
            f"dates={len(common_dates)}, industries={len(common_industries)}"
        )
        return {
            "total_excess": 0.0,
            "allocation_effect": 0.0,
            "selection_effect": 0.0,
            "interaction_effect": 0.0,
            "daily_allocation": pd.Series(dtype=float),
            "daily_selection": pd.Series(dtype=float),
            "daily_interaction": pd.Series(dtype=float),
            "industry_allocation": pd.Series(dtype=float),
            "industry_selection": pd.Series(dtype=float),
        }

    rp = daily_portfolio_returns.loc[common_dates, common_industries]
    rb = daily_benchmark_returns.loc[common_dates, common_industries]
    wp = portfolio_industry_weights.loc[common_dates, common_industries]
    wb = benchmark_industry_weights.loc[common_dates, common_industries]

    # Total benchmark return per day (scalar): rb_total = sum(wb_i * rb_i)
    rb_total = (wb * rb).sum(axis=1)  # Series indexed by date

    # Brinson decomposition (per date, per industry):
    #   allocation_i  = (wp_i - wb_i) * (rb_i - rb_total)
    #   selection_i   = wb_i * (rp_i - rb_i)
    #   interaction_i = (wp_i - wb_i) * (rp_i - rb_i)
    weight_diff = wp - wb  # (dates, industries)
    return_diff = rp - rb  # (dates, industries)
    # rb_i - rb_total: each industry benchmark return minus total benchmark
    rb_active = rb.sub(rb_total, axis=0)  # (dates, industries)

    alloc_matrix = weight_diff * rb_active          # (dates, industries)
    select_matrix = wb * return_diff                 # (dates, industries)
    interact_matrix = weight_diff * return_diff      # (dates, industries)

    # Daily totals (sum across industries)
    daily_allocation = alloc_matrix.sum(axis=1)
    daily_selection = select_matrix.sum(axis=1)
    daily_interaction = interact_matrix.sum(axis=1)

    # Industry totals (sum across dates)
    industry_allocation = alloc_matrix.sum(axis=0)
    industry_selection = select_matrix.sum(axis=0)

    # Portfolio and benchmark total returns
    rp_total = (wp * rp).sum(axis=1)
    total_excess = float(rp_total.sum() - rb_total.sum())

    return {
        "total_excess": total_excess,
        "allocation_effect": float(daily_allocation.sum()),
        "selection_effect": float(daily_selection.sum()),
        "interaction_effect": float(daily_interaction.sum()),
        "daily_allocation": daily_allocation,
        "daily_selection": daily_selection,
        "daily_interaction": daily_interaction,
        "industry_allocation": industry_allocation,
        "industry_selection": industry_selection,
    }


def simple_attribution(
    daily_returns: pd.Series,
    factor_exposures: pd.DataFrame,
    factor_returns: Optional[pd.DataFrame] = None,
) -> dict:
    """Factor attribution via cross-sectional regression.

    If factor_returns is provided, uses exposure * factor_return decomposition.
    Otherwise, estimates factor returns via OLS regression of daily_returns on
    factor_exposures for each date.

    Args:
        daily_returns: Series indexed by (date, instrument) with daily returns.
        factor_exposures: DataFrame indexed by (date, instrument) with columns
            = factor names, containing exposure/loading of each stock to each
            factor on that date.
        factor_returns: Optional DataFrame indexed by date with columns = factor
            names. If None, factor returns are estimated via OLS.

    Returns:
        dict with:
        - factor_contrib: pd.DataFrame (dates x factors) — daily contribution
          of each factor = mean_exposure * factor_return
        - alpha: pd.Series indexed by date — residual return (alpha)
        - total_factor: float — cumulative factor-explained return
        - total_alpha: float — cumulative alpha
        - factor_summary: pd.Series — per-factor cumulative contribution
    """
    dates = daily_returns.index.get_level_values(0).unique().sort_values()
    factors = factor_exposures.columns.tolist()

    daily_factor_contrib = []
    daily_alpha = []
    valid_dates = []

    for date in dates:
        if date not in daily_returns.index.get_level_values(0):
            continue
        if date not in factor_exposures.index.get_level_values(0):
            continue

        ret = daily_returns.loc[date]
        exp = factor_exposures.loc[date]

        # Align stocks
        common = ret.index.intersection(exp.index)
        if len(common) < max(len(factors) + 1, 5):
            # Not enough stocks for regression
            continue

        r = ret.loc[common].values  # (n_stocks,)
        X = exp.loc[common].values  # (n_stocks, n_factors)

        if factor_returns is not None and date in factor_returns.index:
            # Direct decomposition: contribution = mean_exposure * factor_return
            fr = factor_returns.loc[date].reindex(factors).values
            mean_exp = np.nanmean(X, axis=0)
            contrib = mean_exp * fr
            alpha_val = np.nanmean(r) - np.nansum(contrib)
        else:
            # OLS: r = X @ beta + epsilon
            # Add intercept
            X_with_const = np.column_stack([np.ones(len(r)), X])
            try:
                # Use lstsq for numerical stability
                beta, residuals, rank, sv = np.linalg.lstsq(
                    X_with_const, r, rcond=None
                )
            except np.linalg.LinAlgError:
                continue

            intercept = beta[0]
            factor_betas = beta[1:]
            mean_exp = np.nanmean(X, axis=0)
            contrib = mean_exp * factor_betas
            alpha_val = intercept + (np.nanmean(r) - np.nansum(contrib) - intercept)
            # Simplify: alpha_val = mean(r) - sum(mean_exp * beta)
            alpha_val = float(np.nanmean(r) - np.nansum(contrib))

        daily_factor_contrib.append(contrib)
        daily_alpha.append(alpha_val)
        valid_dates.append(date)

    if not valid_dates:
        logger.warning("No valid dates for factor attribution.")
        return {
            "factor_contrib": pd.DataFrame(columns=factors),
            "alpha": pd.Series(dtype=float),
            "total_factor": 0.0,
            "total_alpha": 0.0,
            "factor_summary": pd.Series(dtype=float),
        }

    factor_contrib_df = pd.DataFrame(
        daily_factor_contrib, index=valid_dates, columns=factors
    )
    alpha_series = pd.Series(daily_alpha, index=valid_dates)

    return {
        "factor_contrib": factor_contrib_df,
        "alpha": alpha_series,
        "total_factor": float(factor_contrib_df.sum().sum()),
        "total_alpha": float(alpha_series.sum()),
        "factor_summary": factor_contrib_df.sum(axis=0),
    }


def compute_industry_weights(
    portfolio_stocks: pd.DataFrame,
    industry_mapping: pd.DataFrame,
    industry_col: str = "industry",
    stock_col: str = "instrument",
) -> pd.DataFrame:
    """Compute portfolio industry weights from TopK stock holdings.

    Args:
        portfolio_stocks: DataFrame with at least columns [date, stock_col].
            Each row = one stock held on that date (equal weight assumed).
        industry_mapping: DataFrame with columns [stock_col, industry_col].
            Maps each stock to its industry. Can be indexed by stock or have
            stock as a column.
        industry_col: Name of the industry column in industry_mapping.
        stock_col: Name of the stock/instrument column.

    Returns:
        DataFrame indexed by date, columns = industry names, values = weight
        (fraction of portfolio in each industry). Rows sum to 1.
    """
    # Normalize industry_mapping: ensure stock_col is a column
    if stock_col not in industry_mapping.columns:
        if industry_mapping.index.name == stock_col:
            industry_mapping = industry_mapping.reset_index()
        else:
            # Assume index is the stock identifier
            industry_mapping = industry_mapping.copy()
            industry_mapping[stock_col] = industry_mapping.index

    # Merge portfolio with industry
    merged = portfolio_stocks.merge(
        industry_mapping[[stock_col, industry_col]],
        on=stock_col,
        how="left",
    )

    # Handle unmapped stocks
    n_unmapped = merged[industry_col].isna().sum()
    if n_unmapped > 0:
        logger.warning(
            f"{n_unmapped} stock-day entries have no industry mapping, "
            f"assigned to 'Unknown'"
        )
        merged[industry_col] = merged[industry_col].fillna("Unknown")

    # Equal-weight: count stocks per industry per date, normalize
    date_col = "date" if "date" in merged.columns else merged.columns[0]
    counts = merged.groupby([date_col, industry_col]).size().unstack(fill_value=0)
    weights = counts.div(counts.sum(axis=1), axis=0)

    return weights
