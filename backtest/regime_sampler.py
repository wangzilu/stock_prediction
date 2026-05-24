"""Regime-weighted training sampler.

Uses regime_controller scores to weight training samples so that the model
focuses on historical periods whose macro regime resembles the target date.

Usage:
    from backtest.regime_sampler import compute_regime_vectors, compute_sample_weights

    vectors = compute_regime_vectors(['2024-10-08', '2025-01-06'])
    weights = compute_sample_weights(['2024-10-08', '2025-01-06'], target_date='2026-05-22')
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "data" / "storage" / "regime_vectors_cache.parquet"

# The 11 individual regime score columns (excluding risk_on_score)
REGIME_COLS = [
    "liquidity_score",
    "credit_stress_score",
    "leverage_unwind_score",
    "microcap_crash_risk",
    "external_shock_score",
    "policy_support_score",
    "theme_breadth_score",
    "inflation_score",
    "northbound_score",
    "futures_basis_score",
    "fx_risk_score",
]


def _load_cache() -> pd.DataFrame:
    """Load cached regime vectors if available."""
    if CACHE_PATH.exists():
        try:
            df = pd.read_parquet(CACHE_PATH)
            df.index = df.index.astype(str)
            return df
        except Exception as e:
            logger.warning("Failed to load regime vector cache: %s", e)
    return pd.DataFrame(columns=REGIME_COLS)


def _save_cache(df: pd.DataFrame) -> None:
    """Persist regime vectors to parquet cache."""
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE_PATH)
    except Exception as e:
        logger.warning("Failed to save regime vector cache: %s", e)


def compute_regime_vectors(dates: list[str]) -> pd.DataFrame:
    """Compute regime score vectors for a list of dates.

    Parameters
    ----------
    dates : list[str]
        Dates in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date string with 11 regime score columns.
        Each score is in [-1, +1].
    """
    from signals.regime_controller import RegimeController

    dates_str = [str(d)[:10] for d in dates]

    # Load cache and find which dates need computation
    cache = _load_cache()
    cached_dates = set(cache.index)
    missing = [d for d in dates_str if d not in cached_dates]

    if missing:
        rc = RegimeController()
        new_rows = []
        for d in missing:
            try:
                scores = rc.compute(date=d)
                row = {col: scores.get(col, 0.0) for col in REGIME_COLS}
                row["date"] = d
                new_rows.append(row)
            except Exception as e:
                logger.warning("regime_controller failed for %s: %s", d, e)
                row = {col: 0.0 for col in REGIME_COLS}
                row["date"] = d
                new_rows.append(row)

        if new_rows:
            new_df = pd.DataFrame(new_rows).set_index("date")
            cache = pd.concat([cache, new_df])
            cache = cache[~cache.index.duplicated(keep="last")]
            _save_cache(cache)

    # Return only requested dates, in order
    result = cache.reindex(dates_str)
    # Fill any still-missing dates with zeros
    result = result.fillna(0.0)
    return result[REGIME_COLS]


def compute_sample_weights(
    training_dates: list[str],
    target_date: str,
    method: str = "cosine",
    temperature: float = 1.0,
) -> pd.Series:
    """Compute per-date sample weights based on regime similarity to target.

    Parameters
    ----------
    training_dates : list[str]
        Dates used for training.
    target_date : str
        The date whose regime we want to match.
    method : str
        Similarity method. Currently only 'cosine' is supported.
    temperature : float
        Softmax temperature. Lower values concentrate weight on similar dates.
        Default 1.0.

    Returns
    -------
    pd.Series
        Weights indexed by date, summing to 1.0.
    """
    all_dates = list(training_dates) + [target_date]
    vectors = compute_regime_vectors(all_dates)

    target_vec = vectors.loc[target_date].values.astype(float)
    train_vecs = vectors.loc[list(training_dates)].values.astype(float)

    if method == "cosine":
        target_norm = np.linalg.norm(target_vec)
        if target_norm < 1e-12:
            # Target vector is zero -- uniform weights
            n = len(training_dates)
            return pd.Series(np.ones(n) / n, index=training_dates)

        train_norms = np.linalg.norm(train_vecs, axis=1)
        # Avoid division by zero for any training date with zero vector
        train_norms = np.maximum(train_norms, 1e-12)

        similarities = train_vecs @ target_vec / (train_norms * target_norm)
    else:
        raise ValueError(f"Unknown similarity method: {method}")

    # Softmax with temperature
    logits = similarities / max(temperature, 1e-12)
    logits = logits - logits.max()  # numerical stability
    exp_logits = np.exp(logits)
    weights = exp_logits / exp_logits.sum()

    return pd.Series(weights, index=training_dates)


def compute_sample_weights_for_index(
    index: pd.MultiIndex,
    target_date: str,
    temperature: float = 1.0,
    method: str = "cosine",
) -> np.ndarray:
    """Expand date-level regime weights to per-row weights for a MultiIndex.

    Parameters
    ----------
    index : pd.MultiIndex
        Qlib-style (datetime, instrument) MultiIndex.
    target_date : str
        The date whose regime we want to match.
    temperature : float
        Softmax temperature for weight computation.
    method : str
        Similarity method.

    Returns
    -------
    np.ndarray
        1-D array of weights, one per row of the index, suitable for
        XGBoost/LightGBM sample_weight.
    """
    # Extract unique dates from the index
    date_level = index.get_level_values(0)
    unique_dates = sorted(set(
        pd.Timestamp(d).strftime("%Y-%m-%d") for d in date_level.unique()
    ))

    date_weights = compute_sample_weights(
        unique_dates, target_date, method=method, temperature=temperature
    )

    # Map each row to its date weight
    row_dates = pd.Series(
        [pd.Timestamp(d).strftime("%Y-%m-%d") for d in date_level],
        index=index,
    )
    row_weights = row_dates.map(date_weights).fillna(1.0 / len(unique_dates)).values

    return row_weights
