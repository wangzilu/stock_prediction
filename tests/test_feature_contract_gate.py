"""Behavior pin: production_feature_contract.json is now a REAL gate
(cx round 2 P1-3 / P1-4), not just a report.

Pre-fix:
- ``scripts/export_feature_contract.py`` wrote the artifact, but no
  code path READ it. The "gate" at train and inference time was still
  just ``booster.num_features() == X.shape[1]`` — a COUNT check, not
  a NAME / ORDER check. A loader silently reordering its columns, or
  two loaders swapping a column with the same dtype, would slip past
  the dim gate and produce silent garbage at serve time (same incident
  class as 6-3 22:00).

Post-fix:
- ``models/feature_contract.py`` owns load/write/verify.
- ``scripts/train_lgb.py`` writes the contract after every successful
  production train, recording the real supplementary names.
- ``models/short_term.ShortTermModel.load_from_pickle`` reads the
  contract and refuses to serve when supplementary names / order
  drift. (Alpha158 segment is COUNT-only because Qlib does not give
  us stable string names per column.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.feature_contract import (
    FeatureContractViolation,
    load_contract,
    verify_inference_dataset,
    write_contract,
)


# A small synthetic contract covering both segments.
def _make_contract(tmp_path: Path) -> dict:
    alpha158_names = [f"alpha158_f{i:03d}" for i in range(158)]
    supp_names = [
        "flow_net_mf_latest", "flow_net_mf_5d", "flow_net_mf_20d_avg",
        "macro_cpi", "macro_ppi", "macro_m2_yoy",
    ]
    write_contract(
        tmp_path,
        model_pkl_path=str(tmp_path / "lgb_model.pkl"),
        feature_names=alpha158_names + supp_names,
        alpha158_count=158,
        supplementary_count=len(supp_names),
        production_groups=("capital_flow", "macro_zero_baseline"),
    )
    contract = load_contract(tmp_path)
    assert contract is not None
    return contract


# ---------------------------------------------------------------------------
# Round-trip + schema basics
# ---------------------------------------------------------------------------

def test_write_then_load_round_trip(tmp_path):
    contract = _make_contract(tmp_path)
    assert contract["booster_num_features"] == 164
    assert contract["alpha158_count"] == 158
    assert contract["supplementary_count"] == 6
    assert contract["schema_version"] == 2
    assert len(contract["features"]) == 164
    assert contract["features"][0]["group"] == "alpha158"
    assert contract["features"][158]["group"] == "supplementary"
    assert contract["features"][158]["name"] == "flow_net_mf_latest"


def test_write_contract_rejects_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match="entries but"):
        write_contract(
            tmp_path,
            model_pkl_path="x.pkl",
            feature_names=["a", "b", "c"],
            alpha158_count=2,
            supplementary_count=10,  # 2+10=12, but only 3 names
            production_groups=(),
        )


def test_load_contract_returns_none_when_missing(tmp_path):
    assert load_contract(tmp_path) is None


# ---------------------------------------------------------------------------
# verify_inference_dataset gate
# ---------------------------------------------------------------------------

def test_verify_passes_on_exact_match(tmp_path):
    contract = _make_contract(tmp_path)
    actual = [f["name"] for f in contract["features"]]
    verify_inference_dataset(contract, actual)  # no raise


def test_verify_fails_on_count_drift(tmp_path):
    contract = _make_contract(tmp_path)
    actual = [f["name"] for f in contract["features"]][:-1]  # drop one
    with pytest.raises(FeatureContractViolation, match="feature count drift"):
        verify_inference_dataset(contract, actual)


def test_verify_fails_on_supp_name_drift(tmp_path):
    """The headline P1-4 gate: contract pins ``flow_net_mf_latest`` at
    position 158, inference dataset has ``flow_net_mf_NEW`` instead.
    Same COUNT (164), different name → must raise. Before this gate
    the count check would have been satisfied and prediction would
    walk default-leaf garbage."""
    contract = _make_contract(tmp_path)
    actual = [f["name"] for f in contract["features"]]
    actual[158] = "flow_net_mf_NEW"  # silent rename / reorder
    with pytest.raises(FeatureContractViolation,
                       match="supplementary feature drift at position 158"):
        verify_inference_dataset(contract, actual)


def test_verify_fails_on_supp_segment_length_drift(tmp_path):
    """Pathological: contract supp segment has 6 cols, inference has
    5. The COUNT check would already catch this, but the explicit
    length-drift error message gives a clearer diagnosis."""
    contract = _make_contract(tmp_path)
    actual = [f["name"] for f in contract["features"]]
    actual.pop()  # drop last supp col
    # Count check fires FIRST — that's fine. Either error is
    # acceptable; pin that SOMETHING raises.
    with pytest.raises(FeatureContractViolation):
        verify_inference_dataset(contract, actual)


def test_verify_alpha158_segment_is_count_only(tmp_path):
    """Pin the design choice: real Qlib Alpha158 names (like KMID,
    KLEN, KMID2) are allowed even when the contract recorded
    placeholder ``alpha158_f000``. The Alpha158 segment names are
    inherently unstable per Qlib version, so the gate is COUNT-only
    there. The supplementary segment is the one that earns the
    strict name check."""
    contract = _make_contract(tmp_path)
    real_alpha = [
        "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2",
        "KSFT", "KSFT2", "OPEN0",
    ] + [f"alpha_real_{i}" for i in range(148)]
    supp = [f["name"] for f in contract["features"][158:]]
    actual = real_alpha + supp
    verify_inference_dataset(contract, actual)  # no raise


# ---------------------------------------------------------------------------
# Structural pin: train_lgb writes contract, short_term reads it
# ---------------------------------------------------------------------------

def test_train_lgb_writes_contract_after_save():
    src = (PROJECT_ROOT / "scripts" / "train_lgb.py").read_text()
    assert "from models.feature_contract import write_contract" in src, (
        "train_lgb.py no longer imports write_contract — P1-3 regressed."
    )
    assert "write_contract(" in src, (
        "train_lgb.py no longer calls write_contract."
    )


def test_short_term_reads_contract_and_verifies():
    src = (PROJECT_ROOT / "models" / "short_term.py").read_text()
    assert "load_contract" in src and "verify_inference_dataset" in src, (
        "short_term.py no longer wires the contract gate — P1-4 regressed."
    )
