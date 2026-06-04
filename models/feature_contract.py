"""Production feature contract — single source of truth for the
242-dim (or future N-dim) champion's feature shape.

Why this module exists (2026-06-04):
- cx round 2 P1-3 / P1-4: ``scripts/export_feature_contract.py``
  already writes ``data/storage/production_feature_contract.json``,
  but no code reads it. The "real" gate at train + inference time was
  still just ``booster.num_features() == X.shape[1]`` — a COUNT
  check, not a NAME / ORDER check. A loader silently reordering its
  columns, or two loaders swapping a column with the same dtype,
  would slip through the dim gate and produce silent garbage at
  serve time (same incident class as 6-3 22:00).

What this module pins:
- One read/write surface for the contract JSON.
- A ``verify_inference_dataset`` function that ``short_term`` calls
  after building the inference dataset. It compares the actual
  column names against the contract's stored names. For the
  Alpha158 segment we accept placeholder ``alpha158_f000…`` names
  because Qlib does not give us stable string names per col — the
  Alpha158 segment is gated by COUNT only. For the supplementary
  segment (positions ≥ 158) names + order are strict.

How it gets populated:
- ``scripts/train_lgb.py`` rewrites the artifact at the end of every
  successful production train, using the actual ``feature_cols`` it
  injected. So the contract refreshes on every retrain.
- ``scripts/export_feature_contract.py`` is the manual recovery
  tool — useful if the live artifact is lost and a retrain is not
  yet possible.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


class FeatureContractViolation(RuntimeError):
    """Raised when the inference dataset's feature shape / names do
    NOT match the production contract artifact. cx P0-d (2026-06-04):
    callers MUST NOT silently fall back to a cached prediction on
    this error — the cached prediction was almost certainly built
    from the same broken shape."""


# 2026-06-04 cx round 22 P0-1: contract filename moved to the
# profile-aware helper in config.production_features. The legacy
# constant stays only as the SYMLINK target so callers that still
# read ``production_feature_contract.json`` see the active profile's
# contract (via the legacy alias maintained by train_lgb).
CONTRACT_FILENAME = "production_feature_contract.json"  # legacy symlink target


def contract_path(data_dir: Path, profile: str | None = None) -> Path:
    """Path to the contract artifact for ``profile``.

    Pre-fix this returned the legacy single filename regardless of
    profile, so training xgb_174 would overwrite the 242 contract.
    Now resolves to ``production_feature_contract_{profile}.json``.
    """
    from config.production_features import production_contract_filename
    return Path(data_dir) / production_contract_filename(profile)


def legacy_contract_path(data_dir: Path) -> Path:
    """Legacy ``production_feature_contract.json`` alias (symlink)."""
    return Path(data_dir) / CONTRACT_FILENAME


def write_contract(
    data_dir: Path,
    *,
    model_pkl_path: str,
    feature_names: Iterable[str],
    alpha158_count: int,
    supplementary_count: int,
    production_groups: Iterable[str],
    booster_num_features: int | None = None,
    profile: str | None = None,
) -> Path:
    """Atomically write the production feature contract.

    Args:
        data_dir: where to land the JSON (production: ``data/storage``).
        model_pkl_path: path of the model artifact this contract pins.
        feature_names: ordered list of every column the trained model
            sees, starting with Alpha158 cols then supplementary cols.
        alpha158_count: number of Alpha158 columns (typically 158).
        supplementary_count: number of supplementary columns.
        production_groups: the PRODUCTION_SUPPLEMENTARY_GROUPS tuple
            used by this run — recorded for traceability.
        booster_num_features: if provided, recorded alongside the
            shape for an independent sanity column. Defaults to
            ``alpha158_count + supplementary_count``.

    Returns:
        Path of the written JSON.
    """
    feature_names = list(feature_names)
    expected = int(alpha158_count) + int(supplementary_count)
    if len(feature_names) != expected:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but "
            f"alpha158_count + supplementary_count = {expected}"
        )
    if booster_num_features is not None and int(booster_num_features) != expected:
        raise ValueError(
            f"booster_num_features={booster_num_features} disagrees with "
            f"alpha158_count + supplementary_count = {expected}"
        )

    features: list[dict] = []
    for i, name in enumerate(feature_names):
        group = "alpha158" if i < alpha158_count else "supplementary"
        features.append({
            "index": i,
            "name": str(name),
            "group": group,
        })

    contract = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "model_pkl_path": str(model_pkl_path),
        "booster_num_features": expected,
        "alpha158_count": int(alpha158_count),
        "supplementary_count": int(supplementary_count),
        "production_groups": list(production_groups),
        "features": features,
        "profile": (profile or "").strip().lower(),  # cx round 22 P0-1
        # cx round 2: write a flag so ``verify_inference_dataset``
        # knows this artifact has real-name granularity in the supp
        # segment (older / placeholder-Alpha158 artifacts can still be
        # loaded but with reduced strictness on the alpha segment).
        "schema_version": 2,
    }
    out = contract_path(data_dir, profile)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(contract, ensure_ascii=False, indent=2))
    tmp.replace(out)

    # cx round 22 P0-1: maintain the legacy symlink so consumers that
    # still read ``production_feature_contract.json`` see the active
    # profile's contract. Use atomic flip via .tmp + os.replace.
    import os as _os
    legacy = legacy_contract_path(data_dir)
    tmp_link = legacy.with_suffix(legacy.suffix + ".symlink.tmp")
    try:
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        _os.symlink(out.name, tmp_link)
        _os.replace(tmp_link, legacy)
    except OSError as link_exc:
        import shutil
        shutil.copy2(out, legacy)
        logger.warning("Contract symlink unsupported (%s); copied instead", link_exc)

    logger.info("Wrote feature contract: %s (%d features, schema v2, profile=%s)",
                out, expected, contract["profile"] or "<unset>")
    return out


def load_contract(data_dir: Path, profile: str | None = None) -> dict | None:
    """Return the parsed contract dict, or None if the artifact does
    not exist. Callers should treat None as "no gate yet — log and
    continue" rather than as a fatal error, because there is a
    bootstrap window between the first deploy of this module and the
    first retrain that writes a contract."""
    path = contract_path(data_dir, profile)
    if not path.exists():
        # Fall back to the legacy single-file location for back-compat
        # during migration. After train_lgb runs once with the new
        # writer, the profile-specific contract exists and this fallback
        # is no longer hit.
        legacy = legacy_contract_path(data_dir)
        if legacy.exists():
            logger.warning(
                "load_contract: profile contract %s missing; using legacy %s. "
                "Re-run scripts/train_lgb.py to populate the profile contract.",
                path.name, legacy.name,
            )
            return json.loads(legacy.read_text())
        return None
    return json.loads(path.read_text())


def verify_inference_dataset(
    contract: dict, actual_names: list[str],
) -> None:
    """Validate an inference dataset against the production contract.

    Strict for the supplementary segment (positions ≥ alpha158_count):
    name AND order must match exactly. For the Alpha158 segment, we
    accept ``alpha158_f000`` placeholders OR real Qlib names — the
    segment is gated by COUNT only.

    Raises:
        FeatureContractViolation: when count mismatches, or when any
            supplementary name / order is wrong.
    """
    expected_count = int(contract.get("booster_num_features", 0))
    if expected_count <= 0:
        raise FeatureContractViolation(
            "contract has no booster_num_features — corrupted artifact"
        )
    if len(actual_names) != expected_count:
        raise FeatureContractViolation(
            f"feature count drift: contract pins {expected_count}, "
            f"inference dataset has {len(actual_names)}. Refusing to "
            f"serve — silent default-leaf garbage is exactly what the "
            f"2026-06-03 22:00 incident produced."
        )

    alpha_count = int(contract.get("alpha158_count", 158))

    # Pull supplementary expected names in order
    features = contract.get("features") or []
    expected_supp = [
        str(f["name"]) for f in features if int(f.get("index", -1)) >= alpha_count
    ]
    actual_supp = list(actual_names[alpha_count:])

    if len(actual_supp) != len(expected_supp):
        raise FeatureContractViolation(
            f"supplementary segment length drift: contract has "
            f"{len(expected_supp)} supp cols, inference has "
            f"{len(actual_supp)}"
        )

    for i, (act, exp) in enumerate(zip(actual_supp, expected_supp)):
        if act != exp:
            raise FeatureContractViolation(
                f"supplementary feature drift at position "
                f"{alpha_count + i}: contract expects {exp!r}, "
                f"inference got {act!r}. This is the precise failure "
                f"mode the contract artifact exists to catch — a new "
                f"or reordered loader column slipping past the dim "
                f"check."
            )
