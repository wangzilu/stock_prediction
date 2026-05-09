"""Local model registry for experiment tracking.

Stores each training run's config, metrics, and model path.
Simple JSON-based, no MLflow dependency.

Usage:
    registry = ModelRegistry()
    run_id = registry.record_run(
        model_name="lgb",
        config={...},
        metrics={"ic_mean": 0.033, ...},
        model_path="data/storage/lgb_model.pkl",
    )
    best = registry.get_best("lgb", metric="ic_mean")
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_DIR = Path(__file__).resolve().parents[1] / "data" / "storage" / "model_registry"


class ModelRegistry:

    def __init__(self, registry_dir: Path = DEFAULT_REGISTRY_DIR):
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.registry_dir / "index.json"
        self._index = self._load_index()

    def _load_index(self) -> list:
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text())
            except Exception:
                return []
        return []

    def _save_index(self):
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._index, ensure_ascii=False, indent=2))
        os.replace(tmp, self.index_path)

    def record_run(
        self,
        model_name: str,
        config: dict,
        metrics: dict,
        model_path: str = "",
        data_info: dict = None,
        notes: str = "",
    ) -> str:
        """Record a training run.

        Returns:
            run_id string (timestamp-based)
        """
        run_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        entry = {
            "run_id": run_id,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": config,
            "metrics": metrics,
            "model_path": model_path,
            "data_info": data_info or {},
            "notes": notes,
        }

        self._index.append(entry)
        # Keep last 200 entries
        self._index = self._index[-200:]
        self._save_index()

        logger.info(f"Recorded run {run_id}: {metrics}")
        return run_id

    def get_runs(self, model_name: Optional[str] = None, limit: int = 20) -> list:
        """Get recent runs, optionally filtered by model name."""
        runs = self._index
        if model_name:
            runs = [r for r in runs if r["model_name"] == model_name]
        return runs[-limit:]

    def get_best(self, model_name: str, metric: str = "ic_mean", higher_is_better: bool = True) -> Optional[dict]:
        """Get the best run for a model by a metric."""
        runs = [r for r in self._index if r["model_name"] == model_name and metric in r.get("metrics", {})]
        if not runs:
            return None
        return max(runs, key=lambda r: r["metrics"][metric] * (1 if higher_is_better else -1))

    def compare_latest(self, limit: int = 10) -> list:
        """Get the latest run per model for comparison."""
        latest = {}
        for run in self._index:
            name = run["model_name"]
            if name not in latest or run["timestamp"] > latest[name]["timestamp"]:
                latest[name] = run
        return sorted(latest.values(), key=lambda r: r.get("metrics", {}).get("ic_mean", 0), reverse=True)
