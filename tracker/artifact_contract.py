"""Unified experiment artifact contract.

Every training/backtest run must produce a standard set of artifacts.
This module defines the schema, handles serialization, and validates
completeness.

Artifact structure per experiment:
    data/storage/experiments/{experiment_id}/
        config.json        — hyperparams, data window, feature set, preprocessing
        pred.pkl           — predictions (datetime × instrument DataFrame)
        label.pkl          — actual labels (same index as pred)
        metrics.json       — IC/RankIC/ICIR/RankICIR/spread/cost_adjusted
        backtest.json      — Sharpe/annual_return/max_dd/turnover/cost_drag
        factor_health.json — coverage/freshness/autocorr/persistence
        exposure.json      — industry/style/size exposure

Usage:
    from tracker.artifact_contract import ExperimentArtifact

    art = ExperimentArtifact.create(
        experiment_id="xgb_174_rolling_24split_20260524",
        model_name="xgb_174",
        feature_set="FS-174",
    )
    art.save_config({...})
    art.save_predictions(pred_df, label_df)
    art.save_metrics({...})
    art.save_backtest({...})
    art.save_factor_health({...})
    art.save_exposure({...})

    # Validate completeness
    report = art.validate()
    print(report["complete"])  # True/False
    print(report["missing"])   # list of missing artifacts
"""
import json
import logging
import pickle
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "data" / "storage" / "experiments"

REQUIRED_ARTIFACTS = ["config.json", "metrics.json"]
OPTIONAL_ARTIFACTS = [
    "pred.pkl", "label.pkl", "backtest.json",
    "factor_health.json", "exposure.json",
]

# Minimum required fields in each artifact
METRICS_REQUIRED_FIELDS = [
    "rank_ic_mean", "rank_ic_std",
]
CONFIG_REQUIRED_FIELDS = [
    "model_name", "feature_set", "created_at",
]


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types that break json.dump."""

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


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


class ExperimentArtifact:
    """Manages artifacts for a single experiment run."""

    def __init__(self, experiment_id: str, base_dir: Path = None):
        self.experiment_id = experiment_id
        self.base_dir = base_dir or EXPERIMENTS_DIR
        self.artifact_dir = self.base_dir / experiment_id
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        experiment_id: str,
        model_name: str,
        feature_set: str,
        description: str = "",
        **extra_config,
    ) -> "ExperimentArtifact":
        """Create a new experiment with initial config."""
        art = cls(experiment_id)
        config = {
            "experiment_id": experiment_id,
            "model_name": model_name,
            "feature_set": feature_set,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "code_version": _git_hash(),
            **extra_config,
        }
        art.save_config(config)
        return art

    @classmethod
    def load(cls, experiment_id: str) -> "ExperimentArtifact":
        """Load an existing experiment."""
        art = cls(experiment_id)
        if not art.artifact_dir.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found")
        return art

    @classmethod
    def list_all(cls) -> list[str]:
        """List all experiment IDs."""
        if not EXPERIMENTS_DIR.exists():
            return []
        return sorted(
            d.name for d in EXPERIMENTS_DIR.iterdir()
            if d.is_dir() and (d / "config.json").exists()
        )

    # --- Save methods ---

    def save_config(self, config: dict):
        self._write_json("config.json", config)

    def save_predictions(self, pred: pd.DataFrame, label: pd.DataFrame = None):
        """Save prediction and label DataFrames as pickle."""
        self._write_pickle("pred.pkl", pred)
        if label is not None:
            self._write_pickle("label.pkl", label)

    def save_metrics(self, metrics: dict):
        """Save signal-level metrics.

        Expected fields (not all required):
            ic_mean, ic_std, icir,
            rank_ic_mean, rank_ic_std, rank_icir,
            rank_ic_pos_ratio,
            spread_top20, spread_top50, spread_top100,
            coverage, n_predictions, n_days,
            cost_adjusted_spread, cost_to_return_ratio
        """
        metrics["saved_at"] = datetime.now().isoformat()
        self._write_json("metrics.json", metrics)

    def save_backtest(self, backtest: dict):
        """Save portfolio backtest results.

        Expected fields:
            sharpe, annual_return, annual_vol, max_drawdown,
            avg_turnover, cost_drag, excess_return, excess_ir,
            win_rate, n_trades
        """
        backtest["saved_at"] = datetime.now().isoformat()
        self._write_json("backtest.json", backtest)

    def save_factor_health(self, health: dict):
        """Save factor health diagnostics.

        Expected fields:
            coverage, freshness_days,
            autocorr_1d, autocorr_5d,
            persistence_5d, persistence_20d,
            n_factors, stale_factor_count
        """
        health["saved_at"] = datetime.now().isoformat()
        self._write_json("factor_health.json", health)

    def save_exposure(self, exposure: dict):
        """Save industry/style/size exposure.

        Expected fields:
            industry_active_weights: {industry: weight},
            style_exposures: {size, beta, momentum, volatility, liquidity},
            max_single_name_weight, top5_concentration
        """
        exposure["saved_at"] = datetime.now().isoformat()
        self._write_json("exposure.json", exposure)

    # --- Load methods ---

    def load_config(self) -> dict:
        return self._read_json("config.json")

    def load_metrics(self) -> dict:
        return self._read_json("metrics.json")

    def load_backtest(self) -> dict:
        return self._read_json("backtest.json")

    def load_factor_health(self) -> dict:
        return self._read_json("factor_health.json")

    def load_exposure(self) -> dict:
        return self._read_json("exposure.json")

    def load_predictions(self) -> pd.DataFrame:
        return self._read_pickle("pred.pkl")

    def load_labels(self) -> pd.DataFrame:
        return self._read_pickle("label.pkl")

    # --- Validation ---

    def validate(self) -> dict:
        """Check artifact completeness and field validity."""
        missing = []
        warnings = []

        for f in REQUIRED_ARTIFACTS:
            if not (self.artifact_dir / f).exists():
                missing.append(f)

        # Check config fields
        if (self.artifact_dir / "config.json").exists():
            config = self.load_config()
            for field in CONFIG_REQUIRED_FIELDS:
                if field not in config:
                    warnings.append(f"config.json missing field: {field}")

        # Check metrics fields
        if (self.artifact_dir / "metrics.json").exists():
            metrics = self.load_metrics()
            for field in METRICS_REQUIRED_FIELDS:
                if field not in metrics:
                    warnings.append(f"metrics.json missing field: {field}")

        # Check optional artifacts
        optional_present = []
        for f in OPTIONAL_ARTIFACTS:
            if (self.artifact_dir / f).exists():
                optional_present.append(f)

        return {
            "experiment_id": self.experiment_id,
            "complete": len(missing) == 0,
            "missing_required": missing,
            "warnings": warnings,
            "optional_present": optional_present,
            "artifact_dir": str(self.artifact_dir),
        }

    def summary(self) -> dict:
        """One-line summary for comparison tables."""
        result = {"experiment_id": self.experiment_id}
        try:
            config = self.load_config()
            result["model_name"] = config.get("model_name", "?")
            result["feature_set"] = config.get("feature_set", "?")
            result["created_at"] = config.get("created_at", "?")
        except Exception:
            pass
        try:
            metrics = self.load_metrics()
            result["rank_ic"] = metrics.get("rank_ic_mean")
            result["rank_icir"] = metrics.get("rank_icir")
            result["spread_top20"] = metrics.get("spread_top20")
            result["spread_top100"] = metrics.get("spread_top100")
        except Exception:
            pass
        try:
            bt = self.load_backtest()
            result["sharpe"] = bt.get("sharpe")
            result["annual_return"] = bt.get("annual_return")
            result["max_drawdown"] = bt.get("max_drawdown")
            result["avg_turnover"] = bt.get("avg_turnover")
        except Exception:
            pass
        return result

    # --- Internal helpers ---

    def _write_json(self, filename: str, data: dict):
        path = self.artifact_dir / filename
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        tmp.replace(path)

    def _read_json(self, filename: str) -> dict:
        path = self.artifact_dir / filename
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    def _write_pickle(self, filename: str, obj: Any):
        path = self.artifact_dir / filename
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
        tmp.replace(path)

    def _read_pickle(self, filename: str) -> Any:
        path = self.artifact_dir / filename
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)


def compare_experiments(experiment_ids: list[str] = None) -> pd.DataFrame:
    """Build comparison table across experiments.

    Returns DataFrame with one row per experiment, columns from summary().
    """
    if experiment_ids is None:
        experiment_ids = ExperimentArtifact.list_all()

    rows = []
    for eid in experiment_ids:
        try:
            art = ExperimentArtifact.load(eid)
            rows.append(art.summary())
        except Exception as e:
            rows.append({"experiment_id": eid, "error": str(e)})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Sort by rank_ic descending
    if "rank_ic" in df.columns:
        df = df.sort_values("rank_ic", ascending=False, na_position="last")
    return df
