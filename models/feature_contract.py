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

    # cx round 25 P1-3: resolve profile ONCE so both the path AND the
    # JSON profile field reflect the same value. Pre-fix passing
    # profile=None used PRODUCTION_MODEL_PROFILE for the path but
    # wrote "" into the JSON.
    from config.production_features import PRODUCTION_MODEL_PROFILE
    resolved_profile = (profile or PRODUCTION_MODEL_PROFILE).strip().lower()

    contract = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "model_pkl_path": str(model_pkl_path),
        "booster_num_features": expected,
        "alpha158_count": int(alpha158_count),
        "supplementary_count": int(supplementary_count),
        "production_groups": list(production_groups),
        "features": features,
        "profile": resolved_profile,  # cx round 22 P0-1 + round 25 P1-3
        # cx round 2: write a flag so ``verify_inference_dataset``
        # knows this artifact has real-name granularity in the supp
        # segment (older / placeholder-Alpha158 artifacts can still be
        # loaded but with reduced strictness on the alpha segment).
        "schema_version": 2,
    }
    out = contract_path(data_dir, resolved_profile)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(contract, ensure_ascii=False, indent=2))
    tmp.replace(out)

    # cx round 25 P1-1: DO NOT update the legacy symlink here. Pre-fix
    # this function flipped the legacy alias to the new contract BEFORE
    # the new model.pkl was saved, creating a window where concurrent
    # inference would see new contract + old model and refuse to serve.
    # ``train_lgb`` now flips the legacy contract symlink AFTER the
    # model.pkl atomic save succeeds, paired with the model symlink
    # flip, so the legacy alias pair is always model-consistent.
    logger.info("Wrote feature contract: %s (%d features, schema v2, profile=%s)",
                out, expected, resolved_profile)
    return out


def update_legacy_contract_alias(data_dir: Path, profile: str | None = None) -> None:
    """Atomically flip ``production_feature_contract.json`` to point
    at the profile's contract. Called from train_lgb AFTER the model
    artifact is saved so the legacy alias pair (model + contract) is
    always consistent. cx round 25 P1-1 + P2-4 + round 27 P2-3.
    """
    import os as _os
    from config.production_features import PRODUCTION_MODEL_PROFILE
    resolved = (profile or PRODUCTION_MODEL_PROFILE).strip().lower()
    target = contract_path(data_dir, resolved)
    # cx round 27 P2-3: refuse to create a symlink to a missing target.
    # Pre-fix the helper would happily create a dangling symlink; the
    # next load_contract would fail with a confusing FileNotFoundError
    # instead of explicit "contract target missing".
    if not target.exists():
        raise FeatureContractViolation(
            f"update_legacy_contract_alias: target {target} does not "
            f"exist. Refusing to create a dangling legacy symlink. "
            f"This usually means write_contract was skipped or its "
            f"output was deleted."
        )
    legacy = legacy_contract_path(data_dir)
    tmp_link = legacy.with_suffix(legacy.suffix + ".symlink.tmp")
    try:
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        _os.symlink(target.name, tmp_link)
        _os.replace(tmp_link, legacy)
    except OSError as link_exc:
        import shutil
        # cx round 25 P2-4: copy via .tmp + atomic replace so the
        # legacy path is never absent during the swap.
        tmp_copy = legacy.with_suffix(legacy.suffix + ".copy.tmp")
        shutil.copy2(target, tmp_copy)
        _os.replace(tmp_copy, legacy)
        logger.warning(
            "Contract symlink unsupported (%s); used atomic copy. "
            "Reads from legacy alias will lag retrains until fixed.",
            link_exc,
        )


def load_contract(data_dir: Path, profile: str | None = None) -> dict | None:
    """Return the parsed contract dict, or None if the artifact does
    not exist.

    cx round 23 E.P2 #5: callers MUST treat missing contract as fatal;
    the bootstrap-warning path is decommissioned (cx round 22 P0-1). See
    ``models/short_term.py`` line ~432 (``raise FeatureContractViolation``
    when ``load_contract`` returns None) and
    ``models/production_inference.py`` line ~167 (same fatal raise) —
    both inference entry points refuse to serve without a contract.
    Returning None here is a signal to the caller, NOT permission to fall
    through to a count-only gate (which cannot catch loader reorder or
    silent column swap — the exact failure modes this artifact pins).
    The legacy "bootstrap window" language predates the round-22 lockdown
    and is retained ONLY in the call-graph audit, not as live guidance."""
    from config.production_features import PRODUCTION_MODEL_PROFILE
    resolved_profile = (profile or PRODUCTION_MODEL_PROFILE).strip().lower()

    def _validate_profile(data: dict, *, source_label: str) -> None:
        """cx round 27 P1-2: profile mismatch check applies to BOTH
        the profile-specific path AND the legacy fallback. Pre-fix
        only the legacy fallback validated, so a file named
        ``production_feature_contract_xgb_242.json`` whose JSON had
        ``profile: xgb_174`` would silently slip through."""
        embedded = str(data.get("profile") or "").strip().lower()
        if embedded and embedded != resolved_profile:
            raise FeatureContractViolation(
                f"load_contract: {source_label} carries "
                f"profile={embedded!r} but caller requested "
                f"{resolved_profile!r}. Refusing to serve a contract "
                f"from the wrong profile."
            )

    path = contract_path(data_dir, resolved_profile)
    if path.exists():
        data = json.loads(path.read_text())
        _validate_profile(data, source_label=path.name)
        return data

    # Fall back to the legacy single-file location for back-compat
    # during migration. cx round 25 P1-2 + round 27 P1-2: same
    # profile validation as the primary path; pre-profile (empty
    # ``profile``) artifacts get a one-shot warning.
    legacy = legacy_contract_path(data_dir)
    if not legacy.exists():
        return None
    legacy_data = json.loads(legacy.read_text())
    _validate_profile(legacy_data, source_label=f"legacy alias {legacy.name}")
    legacy_profile = str(legacy_data.get("profile") or "").strip().lower()
    logger.warning(
        "load_contract: profile contract %s missing; using legacy %s "
        "(legacy profile=%s). Re-run scripts/train_lgb.py.",
        path.name, legacy.name, legacy_profile or "<unset>",
    )
    return legacy_data


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
