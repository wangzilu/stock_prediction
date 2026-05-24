"""Alpha Factory Lite — automated candidate factor validation.

Register candidate factors, compute tearsheet metrics (IC, RankIC, spread,
coverage, negative control, autocorrelation), and gate-check before promotion.

The factory does NOT modify the main model.  It only validates candidates.

Usage:
    from tracker.alpha_factory import AlphaFactory, run_tearsheet_from_series

    factory = AlphaFactory()
    factory.register('momentum_20d', 'Price momentum 20-day', build_func)
    factory.run_tearsheet('momentum_20d', returns=fwd_returns)
    result = factory.check_gate('momentum_20d')
    print(result)

    # Standalone tearsheet from raw Series
    metrics = run_tearsheet_from_series(factor_values, forward_returns)
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = PROJECT_ROOT / "data" / "storage" / "candidate_factors"

# Gate thresholds for candidate factors (looser than full promotion gate)
DEFAULT_CANDIDATE_GATE = {
    "rank_ic_mean": 0.005,
    "coverage": 0.30,            # event factors can be sparse
    "negative_control_ic": 0.01,
    "rank_ic_pos_ratio": 0.50,
}


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy/pandas types in JSON serialization."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Standalone tearsheet computation
# ---------------------------------------------------------------------------

def run_tearsheet_from_series(
    factor_values: pd.Series,
    returns: pd.Series,
    n_quantiles: int = 5,
    seed: int = 42,
) -> dict:
    """Compute tearsheet metrics from raw factor and return Series.

    Args:
        factor_values: Series with (datetime, instrument) MultiIndex.
        returns: Series with same MultiIndex (forward returns).
        n_quantiles: Number of quantile buckets for spread calculation.
        seed: Random seed for negative control shuffle.

    Returns:
        dict with all tearsheet metrics.
    """
    # Align factor and returns on their common index
    common = factor_values.dropna().index.intersection(returns.dropna().index)
    if len(common) == 0:
        return {"error": "No overlapping non-NaN data between factor and returns"}

    f = factor_values.loc[common]
    r = returns.loc[common]

    # Get date level name
    date_level = f.index.names[0] if f.index.names[0] else 0

    # Group by date for cross-sectional computations
    dates = f.index.get_level_values(date_level).unique()

    rank_ics = []
    ics = []

    for dt in dates:
        try:
            f_day = f.xs(dt, level=date_level)
            r_day = r.xs(dt, level=date_level)
        except KeyError:
            continue

        # Need at least 5 stocks per day for meaningful correlation
        valid = f_day.dropna().index.intersection(r_day.dropna().index)
        if len(valid) < 5:
            continue

        fv = f_day.loc[valid]
        rv = r_day.loc[valid]

        # Spearman (RankIC)
        try:
            ric, _ = stats.spearmanr(fv.values, rv.values)
            if np.isfinite(ric):
                rank_ics.append(ric)
        except Exception:
            pass

        # Pearson (IC)
        try:
            ic, _ = stats.pearsonr(fv.values, rv.values)
            if np.isfinite(ic):
                ics.append(ic)
        except Exception:
            pass

    rank_ics = np.array(rank_ics)
    ics = np.array(ics)

    if len(rank_ics) == 0:
        return {"error": "No valid daily cross-sections found"}

    rank_ic_mean = float(np.mean(rank_ics))
    rank_ic_std = float(np.std(rank_ics, ddof=1)) if len(rank_ics) > 1 else 0.0
    rank_icir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0.0
    rank_ic_pos_ratio = float(np.mean(rank_ics > 0))

    ic_mean = float(np.mean(ics)) if len(ics) > 0 else 0.0
    ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
    icir = ic_mean / ic_std if ic_std > 0 else 0.0

    # --- Quintile spread ---
    try:
        spread = _compute_spread(f, r, date_level, n_quantiles)
    except Exception:
        spread = None

    # --- Coverage ---
    total_slots = len(factor_values)
    non_nan = factor_values.notna().sum()
    coverage = float(non_nan / total_slots) if total_slots > 0 else 0.0

    # --- Negative control (shuffle factor within each date, recompute RankIC) ---
    rng = np.random.RandomState(seed)
    shuffled_rics = []
    for dt in dates:
        try:
            f_day = f.xs(dt, level=date_level)
            r_day = r.xs(dt, level=date_level)
        except KeyError:
            continue
        valid = f_day.dropna().index.intersection(r_day.dropna().index)
        if len(valid) < 5:
            continue
        fv_shuf = f_day.loc[valid].values.copy()
        rng.shuffle(fv_shuf)
        rv = r_day.loc[valid].values
        try:
            sric, _ = stats.spearmanr(fv_shuf, rv)
            if np.isfinite(sric):
                shuffled_rics.append(sric)
        except Exception:
            pass

    negative_control_ic = float(np.abs(np.mean(shuffled_rics))) if shuffled_rics else 0.0

    # --- Autocorrelation (lag 1d and 5d) ---
    autocorr_1d = _compute_autocorrelation(f, date_level, lag=1)
    autocorr_5d = _compute_autocorrelation(f, date_level, lag=5)

    result = {
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_std": rank_ic_std,
        "rank_icir": rank_icir,
        "rank_ic_pos_ratio": rank_ic_pos_ratio,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "spread_q1_q5": spread,
        "coverage": coverage,
        "negative_control_ic": negative_control_ic,
        "autocorr_1d": autocorr_1d,
        "autocorr_5d": autocorr_5d,
        "n_days": len(rank_ics),
        "n_obs": len(common),
    }
    return result


def _compute_spread(
    factor: pd.Series,
    returns: pd.Series,
    date_level,
    n_quantiles: int,
) -> float:
    """Top vs bottom quintile average return spread."""
    dates = factor.index.get_level_values(date_level).unique()
    top_rets = []
    bot_rets = []

    for dt in dates:
        try:
            f_day = factor.xs(dt, level=date_level)
            r_day = returns.xs(dt, level=date_level)
        except KeyError:
            continue
        valid = f_day.dropna().index.intersection(r_day.dropna().index)
        if len(valid) < n_quantiles:
            continue
        fv = f_day.loc[valid]
        rv = r_day.loc[valid]
        try:
            q_labels = pd.qcut(fv.rank(method="first"), n_quantiles, labels=False)
        except ValueError:
            continue
        top_rets.append(rv[q_labels == n_quantiles - 1].mean())
        bot_rets.append(rv[q_labels == 0].mean())

    if not top_rets:
        return None
    return float(np.mean(top_rets) - np.mean(bot_rets))


def _compute_autocorrelation(
    factor: pd.Series,
    date_level,
    lag: int,
) -> Optional[float]:
    """Average cross-sectional autocorrelation of the factor at given lag."""
    dates = sorted(factor.index.get_level_values(date_level).unique())
    if len(dates) <= lag:
        return None

    autocorrs = []
    for i in range(lag, len(dates)):
        dt_now = dates[i]
        dt_prev = dates[i - lag]
        try:
            f_now = factor.xs(dt_now, level=date_level)
            f_prev = factor.xs(dt_prev, level=date_level)
        except KeyError:
            continue
        common_instr = f_now.dropna().index.intersection(f_prev.dropna().index)
        if len(common_instr) < 5:
            continue
        try:
            corr, _ = stats.spearmanr(
                f_now.loc[common_instr].values,
                f_prev.loc[common_instr].values,
            )
            if np.isfinite(corr):
                autocorrs.append(corr)
        except Exception:
            pass

    return float(np.mean(autocorrs)) if autocorrs else None


# ---------------------------------------------------------------------------
# CandidateFactor
# ---------------------------------------------------------------------------

class CandidateFactor:
    """Represents a single candidate factor under evaluation."""

    def __init__(
        self,
        name: str,
        description: str,
        build_func: Callable,
        base_dir: Path = None,
        **config,
    ):
        self.name = name
        self.description = description
        self.build_func = build_func
        self.config = config

        self.base_dir = base_dir or CANDIDATE_DIR
        self.artifact_dir = self.base_dir / name
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        # Persist config
        self._write_json("config.json", {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            **config,
        })
        # Initial verdict
        if not (self.artifact_dir / "verdict.json").exists():
            self._write_json("verdict.json", {"verdict": "pending"})

    @property
    def tearsheet_path(self) -> Path:
        return self.artifact_dir / "tearsheet.json"

    @property
    def verdict(self) -> str:
        v = self._read_json("verdict.json")
        return v.get("verdict", "pending")

    @verdict.setter
    def verdict(self, value: str):
        self._write_json("verdict.json", {
            "verdict": value,
            "updated_at": datetime.now().isoformat(),
        })

    def load_tearsheet(self) -> dict:
        return self._read_json("tearsheet.json")

    def save_tearsheet(self, metrics: dict):
        metrics["saved_at"] = datetime.now().isoformat()
        self._write_json("tearsheet.json", metrics)

    # --- internal helpers (mirror artifact_contract pattern) ---

    def _write_json(self, filename: str, data: dict):
        path = self.artifact_dir / filename
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        tmp.replace(path)

    def _read_json(self, filename: str) -> dict:
        path = self.artifact_dir / filename
        if not path.exists():
            return {}
        with open(path) as fh:
            return json.load(fh)


# ---------------------------------------------------------------------------
# AlphaFactory
# ---------------------------------------------------------------------------

class AlphaFactory:
    """Registry + validation pipeline for candidate factors."""

    def __init__(
        self,
        base_dir: Path = None,
        gate_thresholds: dict = None,
    ):
        self.base_dir = base_dir or CANDIDATE_DIR
        self.gate_thresholds = {**DEFAULT_CANDIDATE_GATE, **(gate_thresholds or {})}
        self._candidates: dict[str, CandidateFactor] = {}

        # Reload any previously-registered candidates from disk
        if self.base_dir.exists():
            for d in sorted(self.base_dir.iterdir()):
                if d.is_dir() and (d / "config.json").exists():
                    cfg = json.loads((d / "config.json").read_text())
                    # Only load metadata; build_func is not recoverable from disk
                    cf = CandidateFactor.__new__(CandidateFactor)
                    cf.name = cfg.get("name", d.name)
                    cf.description = cfg.get("description", "")
                    cf.build_func = None
                    cf.config = {
                        k: v for k, v in cfg.items()
                        if k not in ("name", "description", "created_at")
                    }
                    cf.base_dir = self.base_dir
                    cf.artifact_dir = d
                    self._candidates[cf.name] = cf

    # ----- public API -----

    def register(
        self,
        name: str,
        description: str,
        build_func: Callable,
        **config,
    ) -> CandidateFactor:
        """Register a new candidate factor."""
        cf = CandidateFactor(
            name=name,
            description=description,
            build_func=build_func,
            base_dir=self.base_dir,
            **config,
        )
        self._candidates[name] = cf
        logger.info(f"Registered candidate factor: {name}")
        return cf

    def run_tearsheet(
        self,
        name: str,
        returns: pd.Series = None,
        n_quantiles: int = 5,
    ) -> dict:
        """Compute and save tearsheet for a registered candidate.

        Args:
            name: Registered candidate name.
            returns: Forward returns Series with (datetime, instrument) MultiIndex.
                     Must be provided externally.
            n_quantiles: Quantile buckets for spread.

        Returns:
            Tearsheet metrics dict.
        """
        cf = self._get(name)

        if returns is None:
            raise ValueError("returns must be provided (forward return Series)")

        # Build factor values
        if cf.build_func is None:
            raise ValueError(
                f"build_func not available for '{name}' "
                f"(loaded from disk without callable)"
            )
        factor_values = cf.build_func()

        # Compute tearsheet
        metrics = run_tearsheet_from_series(
            factor_values, returns, n_quantiles=n_quantiles,
        )

        cf.save_tearsheet(metrics)
        logger.info(f"Tearsheet saved for {name}: RankIC={metrics.get('rank_ic_mean')}")
        return metrics

    def check_gate(self, name: str) -> dict:
        """Check if a candidate factor passes the promotion gate.

        Returns:
            dict with keys: name, pass, verdict, failures, metrics
        """
        cf = self._get(name)
        tearsheet = cf.load_tearsheet()

        if not tearsheet or "error" in tearsheet:
            cf.verdict = "fail"
            return {
                "name": name,
                "pass": False,
                "verdict": "fail",
                "failures": [tearsheet.get("error", "No tearsheet computed")],
                "metrics": tearsheet,
            }

        failures = []
        th = self.gate_thresholds

        # 1. RankIC mean
        ric = tearsheet.get("rank_ic_mean")
        if ric is not None and ric < th["rank_ic_mean"]:
            failures.append(
                f"rank_ic_mean={ric:.4f} < threshold={th['rank_ic_mean']}"
            )

        # 2. Coverage
        cov = tearsheet.get("coverage")
        if cov is not None and cov < th["coverage"]:
            failures.append(
                f"coverage={cov:.2f} < threshold={th['coverage']}"
            )

        # 3. Negative control
        nc = tearsheet.get("negative_control_ic")
        if nc is not None and nc > th["negative_control_ic"]:
            failures.append(
                f"negative_control_ic={nc:.4f} > threshold={th['negative_control_ic']}"
            )

        # 4. RankIC positive ratio
        pos = tearsheet.get("rank_ic_pos_ratio")
        if pos is not None and pos < th["rank_ic_pos_ratio"]:
            failures.append(
                f"rank_ic_pos_ratio={pos:.2f} < threshold={th['rank_ic_pos_ratio']}"
            )

        passed = len(failures) == 0
        verdict = "pass" if passed else "fail"
        cf.verdict = verdict

        result = {
            "name": name,
            "pass": passed,
            "verdict": verdict,
            "failures": failures,
            "metrics": tearsheet,
        }

        logger.info(
            f"Gate {'PASS' if passed else 'FAIL'}: {name} "
            f"(failures={len(failures)})"
        )
        return result

    def list_candidates(self) -> list[dict]:
        """List all registered candidates with status."""
        rows = []
        for name, cf in sorted(self._candidates.items()):
            ts = cf.load_tearsheet()
            rows.append({
                "name": name,
                "description": cf.description,
                "verdict": cf.verdict,
                "rank_ic_mean": ts.get("rank_ic_mean"),
                "coverage": ts.get("coverage"),
            })
        return rows

    def summary_table(self) -> pd.DataFrame:
        """DataFrame comparing all candidate factors."""
        rows = []
        for name, cf in sorted(self._candidates.items()):
            ts = cf.load_tearsheet()
            row = {
                "name": name,
                "description": cf.description,
                "verdict": cf.verdict,
            }
            for key in (
                "rank_ic_mean", "rank_ic_std", "rank_icir",
                "rank_ic_pos_ratio", "ic_mean", "ic_std", "icir",
                "spread_q1_q5", "coverage", "negative_control_ic",
                "autocorr_1d", "autocorr_5d", "n_days",
            ):
                row[key] = ts.get(key)
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        if "rank_ic_mean" in df.columns:
            df = df.sort_values("rank_ic_mean", ascending=False, na_position="last")
        return df

    # ----- internal -----

    def _get(self, name: str) -> CandidateFactor:
        if name not in self._candidates:
            raise KeyError(f"Candidate '{name}' not registered")
        return self._candidates[name]
