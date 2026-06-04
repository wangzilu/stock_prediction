"""Lock down the production feature contract gate (P0-e, 2026-06-04).

These tests exist because commit 95cd256 (2026-05-12) opened the
``scripts/train_lgb.py`` → ``FeatureMerger._load_supplementary()``
injection path with NO allowlist — every new loader added to
FeatureMerger silently joined the production champion at the next
weekly retrain. The 6-3 22:00 0-recommendation incident exposed it.

If any of these tests starts failing without a deliberate edit to
``config/production_features.py``, someone re-opened the暗道.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.production_features import (
    PRODUCTION_SUPPLEMENTARY_GROUPS,
    RESEARCH_ALL_LOADERS,
    SHADOW_SUPPLEMENTARY_GROUPS,
)
from models.feature_merger import FeatureMerger


def _small_index():
    return pd.MultiIndex.from_product(
        [pd.date_range("2026-06-01", periods=2), ["SH600519"]],
        names=["datetime", "instrument"],
    )


def test_load_supplementary_rejects_none_groups():
    """groups=None used to mean 'load every loader' — the暗道. It must
    now raise."""
    merger = FeatureMerger()
    with pytest.raises(ValueError, match="groups=None is rejected"):
        merger._load_supplementary(_small_index(), groups=None)


def test_load_supplementary_rejects_unknown_type():
    merger = FeatureMerger()
    with pytest.raises(ValueError, match="must be a tuple"):
        merger._load_supplementary(_small_index(), groups=123)


def test_research_sentinel_loads_every_implemented_loader():
    """The opt-in sentinel must keep the historical 'load everything'
    behavior available for ablation / shadow scripts that legitimately
    want the full fan-out."""
    merger = FeatureMerger()
    supp_all = merger._load_supplementary(
        _small_index(), groups=RESEARCH_ALL_LOADERS,
    )
    supp_prod = merger._load_supplementary(
        _small_index(), groups=PRODUCTION_SUPPLEMENTARY_GROUPS,
    )
    # If the contract is the live champion, RESEARCH_ALL must always be
    # a strict superset (>= columns) of production. Equal is also fine
    # — that is the steady state when SHADOW_SUPPLEMENTARY_GROUPS is
    # empty and every loader is already promoted.
    n_all = 0 if supp_all is None else supp_all.shape[1]
    n_prod = 0 if supp_prod is None else supp_prod.shape[1]
    assert n_all >= n_prod, (
        f"RESEARCH_ALL produced {n_all} cols but production contract "
        f"produced {n_prod}. Either the sentinel regressed, or a "
        f"loader was added to PRODUCTION_SUPPLEMENTARY_GROUPS without "
        f"a matching loader method existing on FeatureMerger."
    )


def test_train_lgb_imports_production_contract():
    """The single production training script must import
    PRODUCTION_SUPPLEMENTARY_GROUPS — a structural guarantee that it
    cannot silently revert to the no-allowlist code path."""
    train_lgb = (PROJECT_ROOT / "scripts" / "train_lgb.py").read_text()
    assert "PRODUCTION_SUPPLEMENTARY_GROUPS" in train_lgb, (
        "scripts/train_lgb.py no longer references "
        "PRODUCTION_SUPPLEMENTARY_GROUPS. This is the暗道 reopening."
    )
    assert "inject_supplementary_into_handler" in train_lgb, (
        "scripts/train_lgb.py no longer uses "
        "inject_supplementary_into_handler — review whether the "
        "contract gate is still in place."
    )


def test_shadow_pool_disjoint_from_production():
    """A group cannot be in both shadow and production at the same
    time — shadow is by definition pre-promotion."""
    overlap = set(SHADOW_SUPPLEMENTARY_GROUPS) & set(
        PRODUCTION_SUPPLEMENTARY_GROUPS
    )
    assert overlap == set(), (
        f"SHADOW_SUPPLEMENTARY_GROUPS and PRODUCTION_SUPPLEMENTARY_GROUPS "
        f"overlap on {overlap}. Promote or demote — do not list in both."
    )
