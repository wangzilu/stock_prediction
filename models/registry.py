"""Model registry — single source of truth for champion/shadow/research models.

Tracks model lifecycle: research_only → shadow → champion → retired.
Shadow daily inference reads from here, not hardcoded paths.

Usage:
    from models.registry import ModelRegistry

    reg = ModelRegistry()
    reg.register("xgb_174", role="champion", feature_set="FS-174",
                 model_path="data/storage/models/xgb_174/champion_20260519.json",
                 metrics={"rank_ic": 0.051, "spread": 0.025})
    champion = reg.get_champion()
    shadow = reg.get_shadow()
"""
import json
import os
from datetime import datetime
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "storage"
MODELS_DIR = DATA_DIR / "models"
REGISTRY_PATH = DATA_DIR / "phase4" / "model_registry.json"


class ModelRegistry:

    def __init__(self, path: Path = REGISTRY_PATH):
        self.path = Path(path)
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"models": {}, "champion": None, "shadow": None, "updated_at": None}

    def _save(self):
        self._data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))
        os.replace(tmp, self.path)

    def register(self, model_id: str, *,
                 role: str = "research_only",
                 feature_set: str = "",
                 model_path: str = "",
                 train_start: str = "",
                 train_end: str = "",
                 n_features: int = 0,
                 metrics: dict = None,
                 ) -> dict:
        """Register or update a model entry."""
        entry = self._data["models"].get(model_id, {})
        entry.update({
            "model_id": model_id,
            "role": role,
            "feature_set": feature_set,
            "model_path": model_path,
            "train_start": train_start,
            "train_end": train_end,
            "n_features": n_features,
            "metrics": metrics or {},
            "registered_at": datetime.now().isoformat(timespec="seconds"),
        })
        self._data["models"][model_id] = entry

        if role == "champion":
            # Demote old champion to shadow
            old = self._data.get("champion")
            if old and old != model_id and old in self._data["models"]:
                self._data["models"][old]["role"] = "shadow"
            self._data["champion"] = model_id
        elif role == "shadow":
            self._data["shadow"] = model_id

        self._save()
        return entry

    def promote(self, model_id: str, to_role: str):
        """Promote model to new role."""
        if model_id not in self._data["models"]:
            raise ValueError(f"Model {model_id} not registered")
        return self.register(model_id, role=to_role,
                             **{k: v for k, v in self._data["models"][model_id].items()
                                if k not in ("model_id", "role", "registered_at")})

    def reject(self, model_id: str):
        """Mark model as rejected."""
        if model_id in self._data["models"]:
            self._data["models"][model_id]["role"] = "rejected"
            self._data["models"][model_id]["rejected_at"] = datetime.now().isoformat(timespec="seconds")
            if self._data.get("champion") == model_id:
                self._data["champion"] = None
            if self._data.get("shadow") == model_id:
                self._data["shadow"] = None
            self._save()

    def get_champion(self) -> dict | None:
        mid = self._data.get("champion")
        return self._data["models"].get(mid) if mid else None

    def get_shadow(self) -> dict | None:
        mid = self._data.get("shadow")
        return self._data["models"].get(mid) if mid else None

    def get_model(self, model_id: str) -> dict | None:
        return self._data["models"].get(model_id)

    def list_models(self) -> list[dict]:
        return list(self._data["models"].values())

    def set_execution_config(self, model_id: str, execution: dict):
        """Set execution strategy config for a model (optimizer params, etc.)."""
        if model_id in self._data["models"]:
            self._data["models"][model_id]["execution"] = execution
            self._save()

    def status(self) -> dict:
        champion = self._data.get("champion")
        shadow = self._data.get("shadow")
        ch_info = self._data["models"].get(champion, {}) if champion else {}
        sh_info = self._data["models"].get(shadow, {}) if shadow else {}
        return {
            "champion": champion,
            "champion_execution": ch_info.get("execution", {}),
            "champion_metrics": ch_info.get("metrics", {}),
            "shadow": shadow,
            "shadow_execution": sh_info.get("execution", {}),
            "shadow_metrics": sh_info.get("metrics", {}),
            "n_models": len(self._data["models"]),
            "updated_at": self._data.get("updated_at"),
        }
