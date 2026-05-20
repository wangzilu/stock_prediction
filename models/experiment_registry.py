"""Experiment Registry — unified tracking for all experiments.

Every experiment (factor test, model ablation, backtest, optimizer comparison)
gets a unique run_id, frozen params, and metrics. Stored as append-only JSONL.

Usage:
    from models.experiment_registry import ExperimentRegistry

    reg = ExperimentRegistry()
    run_id = reg.start("phase4k_optimizer", params={"top_k": 100, "max_turnover": 0.10})
    # ... run experiment ...
    reg.finish(run_id, metrics={"sharpe": 4.54, "turnover": 0.082}, status="success")

    # Query
    reg.list_runs(tag="phase4k")
    reg.get_run(run_id)
"""
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from utils.json_utils import json_default
from utils.versioning import get_code_version, get_data_version

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = PROJECT_ROOT / "data" / "storage" / "experiments"
REGISTRY_FILE = REGISTRY_DIR / "registry.jsonl"


class ExperimentRegistry:
    """Append-only experiment registry backed by JSONL."""

    def __init__(self, registry_path: Path = None):
        self.path = registry_path or REGISTRY_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def start(self, name: str, params: dict = None, tags: list = None,
              description: str = "") -> str:
        """Register a new experiment run. Returns run_id."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

        record = {
            "run_id": run_id,
            "name": name,
            "description": description,
            "tags": tags or [],
            "params": params or {},
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "metrics": {},
            "artifacts": [],
            "code_version": _safe_version(get_code_version),
            "data_version": _safe_version(get_data_version),
        }

        self._append(record)
        logger.info(f"Experiment started: {run_id} ({name})")
        return run_id

    def finish(self, run_id: str, metrics: dict = None, status: str = "success",
               artifacts: list = None, notes: str = ""):
        """Mark experiment as finished with metrics."""
        record = {
            "run_id": run_id,
            "event": "finish",
            "status": status,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "metrics": metrics or {},
            "artifacts": artifacts or [],
            "notes": notes,
        }
        self._append(record)
        logger.info(f"Experiment finished: {run_id} → {status}")

    def fail(self, run_id: str, error: str):
        """Mark experiment as failed."""
        self.finish(run_id, status="failed", notes=error)

    def get_run(self, run_id: str) -> dict:
        """Get latest state of a run by merging all its events."""
        events = [r for r in self._read_all() if r.get("run_id") == run_id]
        if not events:
            return {}
        # Merge: start record + finish record
        merged = {}
        for e in events:
            merged.update(e)
        return merged

    def list_runs(self, tag: str = None, name: str = None,
                  status: str = None, limit: int = 50) -> list:
        """List recent runs, optionally filtered."""
        all_records = self._read_all()

        # Group by run_id and merge
        runs = {}
        for r in all_records:
            rid = r.get("run_id")
            if rid not in runs:
                runs[rid] = {}
            runs[rid].update(r)

        result = list(runs.values())

        # Filter
        if tag:
            result = [r for r in result if tag in r.get("tags", [])]
        if name:
            result = [r for r in result if name in r.get("name", "")]
        if status:
            result = [r for r in result if r.get("status") == status]

        # Sort by start time descending
        result.sort(key=lambda r: r.get("started_at", ""), reverse=True)
        return result[:limit]

    def summary_table(self, tag: str = None, limit: int = 20) -> str:
        """Return a formatted summary table string."""
        runs = self.list_runs(tag=tag, limit=limit)
        if not runs:
            return "No experiments found."

        lines = [f"{'RunID':<28} {'Name':<30} {'Status':<10} {'Sharpe':>8} {'Key Metric':>15}"]
        lines.append("-" * 95)
        for r in runs:
            rid = r.get("run_id", "?")[:27]
            name = r.get("name", "?")[:29]
            status = r.get("status", "?")
            metrics = r.get("metrics", {})
            sharpe = metrics.get("sharpe", metrics.get("avg_sharpe", ""))
            sharpe_str = f"{sharpe:+.3f}" if isinstance(sharpe, (int, float)) else str(sharpe)
            # Pick first numeric metric as "key"
            key = ""
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and k != "sharpe":
                    key = f"{k}={v}"
                    break
            lines.append(f"{rid:<28} {name:<30} {status:<10} {sharpe_str:>8} {key:>15}")

        return "\n".join(lines)

    def _append(self, record: dict):
        """Append a record to the JSONL file."""
        with open(self.path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")

    def _read_all(self) -> list:
        """Read all records from the JSONL file."""
        if not self.path.exists():
            return []
        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records


def _safe_version(fn):
    """Call a version function, return None on error."""
    try:
        return fn()
    except Exception:
        return None
