"""Phase 4E: Ensemble Fusion Module.

Provides multiple fusion strategies for combining predictions from
different models (experiment artifacts or LightGBM JSON predictions).

Fusion methods:
    - rank_mean: average of rank-normalized predictions
    - robust_z_mean: average of MAD-based z-scored predictions
    - rolling_ic_weighted: weight by rolling RankIC / IC volatility

Usage:
    from models.ensemble_fusion import EnsembleFusion

    ens = EnsembleFusion(["xgb_174_champion", "lgb_rolling_6split"])
    fused = ens.fuse(method="rank_mean")
    report = ens.report()
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "storage" / "experiments"
LGB_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "storage" / "lgb_latest_predictions.json"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_pred_pkl(experiment_id: str) -> Optional[pd.Series]:
    """Load pred.pkl from experiment artifact directory.

    Returns a Series with (datetime, instrument) MultiIndex, or None.
    """
    pred_path = EXPERIMENTS_DIR / experiment_id / "pred.pkl"
    if not pred_path.exists():
        return None
    import pickle
    with open(pred_path, "rb") as f:
        obj = pickle.load(f)
    # pred.pkl may be DataFrame (single column) or Series
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            s = obj.iloc[:, 0]
        else:
            # Use first column named 'score' or fall back to first column
            col = "score" if "score" in obj.columns else obj.columns[0]
            s = obj[col]
    elif isinstance(obj, pd.Series):
        s = obj
    else:
        logger.warning("pred.pkl for %s has unexpected type %s", experiment_id, type(obj))
        return None
    s.name = experiment_id
    return s


def _load_lgb_json() -> Optional[pd.Series]:
    """Load lgb_latest_predictions.json as a single-date prediction Series.

    Returns Series with (datetime, instrument) MultiIndex.

    2026-06-04 cx round 3 P1-8: routes through the validated loader
    so freshness + RED-distribution gates apply. Pre-fix this read
    the JSON raw, which means a poisoned cache (smoke RED, manual
    debug) would feed the production ensembler and the ensembler's
    output would silently inherit the rot.
    """
    from models.lgb_cache import load_prediction_cache
    from models.prediction_health import PredictionDistributionRed
    try:
        preds, data = load_prediction_cache(LGB_PREDICTIONS_PATH)
    except FileNotFoundError:
        return None
    except PredictionDistributionRed:
        logger.error(
            "ensemble_fusion refusing RED-distribution LGB cache — "
            "excluding lgb_latest from the fusion this cycle."
        )
        return None
    except RuntimeError as exc:
        logger.error("ensemble_fusion LGB cache load failed: %s", exc)
        return None
    if not preds:
        return None

    date_str = data.get("latest_date", "2024-01-01")
    dt = pd.Timestamp(date_str)

    # Build MultiIndex series
    instruments = list(preds.keys())
    values = [preds[k] for k in instruments]
    idx = pd.MultiIndex.from_arrays(
        [[dt] * len(instruments), instruments],
        names=["datetime", "instrument"],
    )
    s = pd.Series(values, index=idx, dtype=float, name="lgb_latest")
    return s


def load_model_predictions(experiment_ids: list[str]) -> dict[str, pd.Series]:
    """Load predictions from each experiment's artifact.

    Tries pred.pkl first; falls back to lgb_latest_predictions.json for
    experiment IDs that start with 'lgb'.

    Returns {model_id: prediction_series} where series has
    (datetime, instrument) MultiIndex.
    """
    results: dict[str, pd.Series] = {}
    lgb_loaded = False

    for eid in experiment_ids:
        # Try pred.pkl
        s = _load_pred_pkl(eid)
        if s is not None:
            results[eid] = s
            logger.info("Loaded pred.pkl for %s (%d predictions)", eid, len(s))
            continue

        # Fallback: lgb JSON (only load once, shared across lgb experiments)
        if not lgb_loaded and "lgb" in eid.lower():
            s = _load_lgb_json()
            if s is not None:
                s.name = eid
                results[eid] = s
                lgb_loaded = True
                logger.info("Loaded lgb JSON fallback for %s (%d predictions)", eid, len(s))
                continue

        logger.warning("No predictions found for %s — skipping", eid)

    return results


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def rank_normalize(pred: pd.Series) -> pd.Series:
    """Per-date cross-sectional rank, scaled to [0, 1] percentile.

    For each date, ranks all stocks and maps to [0, 1] using
    (rank - 1) / (n - 1).
    """
    def _rank_pctile(group: pd.Series) -> pd.Series:
        n = len(group)
        if n <= 1:
            return pd.Series(0.5, index=group.index)
        ranked = group.rank(method="average")
        return (ranked - 1) / (n - 1)

    return pred.groupby(level=0).transform(_rank_pctile)


def robust_zscore(pred: pd.Series) -> pd.Series:
    """Per-date MAD-based z-score (more robust than mean/std).

    z = (x - median) / (1.4826 * MAD)
    where MAD = median(|x - median|)
    """
    def _mad_zscore(group: pd.Series) -> pd.Series:
        med = group.median()
        mad = (group - med).abs().median()
        if mad < 1e-12:
            return pd.Series(0.0, index=group.index)
        return (group - med) / (1.4826 * mad)

    return pred.groupby(level=0).transform(_mad_zscore)


# ---------------------------------------------------------------------------
# Fusion methods
# ---------------------------------------------------------------------------

def fuse_rank_mean(
    predictions: dict[str, pd.Series], min_model_count: int = 2
) -> pd.Series:
    """Average of rank-normalized predictions across models.

    Args:
        min_model_count: minimum number of non-NaN model predictions per row.
            Rows with fewer models get NaN (prevents single-model masquerading
            as ensemble).
    """
    if not predictions:
        raise ValueError("No predictions to fuse")

    ranked = {k: rank_normalize(v) for k, v in predictions.items()}
    df = pd.DataFrame(ranked)
    valid_count = df.notna().sum(axis=1)
    fused = df.mean(axis=1)
    fused[valid_count < min_model_count] = np.nan
    fused.name = "fused_rank_mean"
    return fused


def fuse_robust_z_mean(
    predictions: dict[str, pd.Series], min_model_count: int = 2
) -> pd.Series:
    """Average of robust z-scored predictions across models.

    Args:
        min_model_count: minimum number of non-NaN model predictions per row.
    """
    if not predictions:
        raise ValueError("No predictions to fuse")

    zscored = {k: robust_zscore(v) for k, v in predictions.items()}
    df = pd.DataFrame(zscored)
    valid_count = df.notna().sum(axis=1)
    fused = df.mean(axis=1)
    fused[valid_count < min_model_count] = np.nan
    fused.name = "fused_robust_z_mean"
    return fused


def fuse_rolling_ic_weighted(
    predictions: dict[str, pd.Series],
    returns: pd.Series,
    lookback: int = 60,
) -> pd.Series:
    """Weight each model by rolling RankIC / IC volatility.

    Parameters
    ----------
    predictions : dict mapping model_id -> prediction Series
        Each Series has (datetime, instrument) MultiIndex.
    returns : Series
        Forward returns with the same (datetime, instrument) MultiIndex.
    lookback : int
        Number of trading days for rolling IC calculation.

    Constraints:
        - Single model weight <= 60%
        - Weights sum to 1
        - Recalculated daily
    """
    if not predictions:
        raise ValueError("No predictions to fuse")

    model_ids = list(predictions.keys())
    n_models = len(model_ids)

    # Compute daily RankIC for each model
    daily_ics: dict[str, pd.Series] = {}
    for mid in model_ids:
        pred = predictions[mid]
        # Align with returns
        common = pred.index.intersection(returns.index)
        p = pred.loc[common]
        r = returns.loc[common]

        # Per-date RankIC (Spearman correlation)
        dates = p.index.get_level_values(0).unique()
        ic_values = {}
        for dt in dates:
            try:
                p_dt = p.loc[dt]
                r_dt = r.loc[dt]
                # Align instruments
                common_inst = p_dt.index.intersection(r_dt.index)
                if len(common_inst) < 10:
                    continue
                corr, _ = stats.spearmanr(p_dt.loc[common_inst], r_dt.loc[common_inst])
                if np.isfinite(corr):
                    ic_values[dt] = corr
            except Exception:
                continue
        daily_ics[mid] = pd.Series(ic_values, dtype=float).sort_index()

    ic_df = pd.DataFrame(daily_ics)
    all_dates = ic_df.index.sort_values()

    # Rolling IC mean and std -> IC-IR ratio as weight signal
    ranked_preds = {k: rank_normalize(v) for k, v in predictions.items()}
    pred_df = pd.DataFrame(ranked_preds)
    all_pred_dates = pred_df.index.get_level_values(0).unique().sort_values()

    fused_parts = []
    for dt in all_pred_dates:
        # Get lookback window of ICs
        past_dates = all_dates[all_dates < dt]
        if len(past_dates) < 5:
            # Not enough history: equal weight
            weights = np.ones(n_models) / n_models
        else:
            window = past_dates[-lookback:]
            ic_window = ic_df.loc[window].dropna(how="all")
            if len(ic_window) < 5:
                weights = np.ones(n_models) / n_models
            else:
                ic_mean = ic_window.mean()
                ic_std = ic_window.std().replace(0, np.nan)
                ic_ir = (ic_mean / ic_std).fillna(0)
                # Only use positive IC-IR models
                ic_ir = ic_ir.clip(lower=0)
                if ic_ir.sum() < 1e-10:
                    weights = np.ones(n_models) / n_models
                else:
                    weights = ic_ir.values / ic_ir.sum()

        # Apply constraints: max 60% per model
        weights = _apply_weight_constraints(weights, max_single=0.6)

        # Weight predictions for this date
        try:
            day_preds = pred_df.loc[dt]
            if isinstance(day_preds, pd.DataFrame):
                weighted = (day_preds * weights).sum(axis=1)
                fused_parts.append(weighted)
        except KeyError:
            continue

    if not fused_parts:
        raise ValueError("No dates could be fused")

    fused = pd.concat(fused_parts)
    fused.name = "fused_rolling_ic_weighted"
    return fused


def _apply_weight_constraints(
    weights: np.ndarray,
    max_single: float = 0.6,
    tol: float = 1e-8,
) -> np.ndarray:
    """Clip individual weights to max_single and renormalize to sum=1."""
    weights = np.array(weights, dtype=float)
    for _ in range(20):  # iterate until converged
        excess = weights > max_single
        if not excess.any():
            break
        surplus = (weights[excess] - max_single).sum()
        weights[excess] = max_single
        below = ~excess
        if below.sum() == 0:
            break
        weights[below] += surplus * (weights[below] / (weights[below].sum() + tol))

    total = weights.sum()
    if total > tol:
        weights /= total
    else:
        weights = np.ones_like(weights) / len(weights)
    return weights


# ---------------------------------------------------------------------------
# Disagreement & Consensus
# ---------------------------------------------------------------------------

def compute_model_disagreement(predictions: dict[str, pd.Series]) -> pd.Series:
    """Per-date, per-stock std of rank-normalized predictions across models.

    High disagreement means models disagree about this stock's ranking.
    """
    if len(predictions) < 2:
        raise ValueError("Need at least 2 models for disagreement")

    ranked = {k: rank_normalize(v) for k, v in predictions.items()}
    df = pd.DataFrame(ranked)
    disagreement = df.std(axis=1)
    disagreement.name = "model_disagreement"
    return disagreement


def compute_model_consensus(
    predictions: dict[str, pd.Series],
    top_pct: float = 0.10,
) -> pd.Series:
    """Per-date, per-stock: how many models put this stock in top top_pct%.

    Values range from 0 to n_models.
    """
    if not predictions:
        raise ValueError("No predictions for consensus")

    def _is_top(group: pd.Series, pct: float) -> pd.Series:
        threshold = group.quantile(1 - pct)
        return (group >= threshold).astype(int)

    top_flags = {}
    for mid, pred in predictions.items():
        top_flags[mid] = pred.groupby(level=0).transform(_is_top, pct=top_pct)

    df = pd.DataFrame(top_flags)
    consensus = df.sum(axis=1)
    consensus.name = "model_consensus"
    return consensus


# ---------------------------------------------------------------------------
# EnsembleFusion class
# ---------------------------------------------------------------------------

class EnsembleFusion:
    """High-level ensemble fusion interface.

    Parameters
    ----------
    model_ids : list of experiment IDs to load predictions from.
    predictions : optional dict of pre-loaded predictions (bypasses loading).
    """

    METHODS = {
        "rank_mean": fuse_rank_mean,
        "robust_z_mean": fuse_robust_z_mean,
    }

    def __init__(
        self,
        model_ids: list[str] = None,
        predictions: dict[str, pd.Series] = None,
    ):
        if predictions is not None:
            self.predictions = predictions
        elif model_ids is not None:
            self.predictions = load_model_predictions(model_ids)
        else:
            raise ValueError("Provide model_ids or predictions")

        if not self.predictions:
            raise ValueError("No predictions loaded — check experiment IDs")

        self.model_ids = list(self.predictions.keys())
        logger.info(
            "EnsembleFusion initialized with %d models: %s",
            len(self.model_ids), self.model_ids,
        )

    def fuse(self, method: str = "rank_mean", **kwargs) -> pd.Series:
        """Fuse predictions using the specified method.

        Parameters
        ----------
        method : one of 'rank_mean', 'robust_z_mean', 'rolling_ic_weighted'
        **kwargs : additional args for the method (e.g. returns, lookback)
        """
        if method == "rolling_ic_weighted":
            returns = kwargs.get("returns")
            if returns is None:
                raise ValueError("rolling_ic_weighted requires 'returns' kwarg")
            lookback = kwargs.get("lookback", 60)
            return fuse_rolling_ic_weighted(
                self.predictions, returns, lookback=lookback,
            )

        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Available: {list(self.METHODS)}")

        return self.METHODS[method](self.predictions)

    def disagreement(self) -> pd.Series:
        """Compute model disagreement (std of rank-normalized preds)."""
        return compute_model_disagreement(self.predictions)

    def consensus(self, top_pct: float = 0.10) -> pd.Series:
        """Compute model consensus (count of models with stock in top%)."""
        return compute_model_consensus(self.predictions, top_pct=top_pct)

    def report(self, returns: pd.Series = None) -> dict:
        """Generate ensemble report.

        Parameters
        ----------
        returns : optional forward returns Series for IC computation.

        Returns dict with per-model stats and fusion improvement metrics.
        """
        result = {
            "n_models": len(self.model_ids),
            "model_ids": self.model_ids,
        }

        # Per-model stats
        model_stats = {}
        for mid in self.model_ids:
            pred = self.predictions[mid]
            model_stats[mid] = {
                "n_predictions": len(pred),
                "n_dates": pred.index.get_level_values(0).nunique(),
                "mean": float(pred.mean()),
                "std": float(pred.std()),
            }

        # If returns provided, compute IC
        if returns is not None:
            for mid in self.model_ids:
                pred = self.predictions[mid]
                common = pred.index.intersection(returns.index)
                if len(common) < 10:
                    model_stats[mid]["rank_ic"] = None
                    continue
                p = pred.loc[common]
                r = returns.loc[common]
                dates = p.index.get_level_values(0).unique()
                ics = []
                for dt in dates:
                    try:
                        p_dt = p.loc[dt]
                        r_dt = r.loc[dt]
                        ci = p_dt.index.intersection(r_dt.index)
                        if len(ci) < 10:
                            continue
                        corr, _ = stats.spearmanr(p_dt.loc[ci], r_dt.loc[ci])
                        if np.isfinite(corr):
                            ics.append(corr)
                    except Exception:
                        continue
                if ics:
                    model_stats[mid]["rank_ic"] = float(np.mean(ics))
                    model_stats[mid]["rank_icir"] = (
                        float(np.mean(ics) / np.std(ics)) if np.std(ics) > 0 else 0.0
                    )
                else:
                    model_stats[mid]["rank_ic"] = None

            # Fusion IC
            for method_name in ["rank_mean", "robust_z_mean"]:
                try:
                    fused = self.fuse(method=method_name)
                    common = fused.index.intersection(returns.index)
                    if len(common) < 10:
                        continue
                    f = fused.loc[common]
                    r = returns.loc[common]
                    dates = f.index.get_level_values(0).unique()
                    ics = []
                    for dt in dates:
                        try:
                            f_dt = f.loc[dt]
                            r_dt = r.loc[dt]
                            ci = f_dt.index.intersection(r_dt.index)
                            if len(ci) < 10:
                                continue
                            corr, _ = stats.spearmanr(f_dt.loc[ci], r_dt.loc[ci])
                            if np.isfinite(corr):
                                ics.append(corr)
                        except Exception:
                            continue
                    if ics:
                        result[f"fusion_{method_name}_rank_ic"] = float(np.mean(ics))
                except Exception as e:
                    logger.warning("Could not compute fusion IC for %s: %s", method_name, e)

            # Improvement over best single model
            single_ics = [
                model_stats[mid].get("rank_ic")
                for mid in self.model_ids
                if model_stats[mid].get("rank_ic") is not None
            ]
            if single_ics:
                best_single = max(single_ics)
                result["best_single_ic"] = best_single
                for method_name in ["rank_mean", "robust_z_mean"]:
                    key = f"fusion_{method_name}_rank_ic"
                    if key in result:
                        result[f"improvement_{method_name}"] = result[key] - best_single

        result["model_stats"] = model_stats

        # Disagreement summary
        if len(self.model_ids) >= 2:
            d = self.disagreement()
            result["disagreement_mean"] = float(d.mean())
            result["disagreement_std"] = float(d.std())

        return result
