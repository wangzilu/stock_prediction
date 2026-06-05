"""Unit tests for scripts/phase4e_24split_ensemble.py runner optimizations.

Covers the 2026-06-05 changes:
  - ``--models`` filter so xgb-only runs cut 24-split time ~3×.
  - Real ``early_stopping_rounds`` propagation into XGBRegressor params
    (the docstring claimed early stopping but ``fit()`` never passed
    the rounds, so XGB silently ran every 500 trees).
  - ``--preset`` plumbed to ``get_standard_splits`` (24split / 12split
    / 6split).
  - Autotag in checkpoint dir resolution so xgb-only checkpoints don't
    silently shadow the 5-25 3-model archive at
    ``data/storage/phase4e_24split/``.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _add_project_root_to_path():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    yield


def _import_runner():
    return importlib.import_module("scripts.phase4e_24split_ensemble")


# ── argparse plumbing ───────────────────────────────────────────────────

def test_argparse_defaults_match_legacy_run():
    """Default invocation must reproduce the historical 5-25 run:
    24-split, all 3 models, 500 estimators, early stop 30."""
    mod = _import_runner()
    p = mod._build_arg_parser()
    args = p.parse_args([])
    assert args.preset == "24split"
    assert args.models == ["xgb", "lgb", "catboost"]
    assert args.n_estimators == 500
    assert args.early_stopping_rounds == 30


def test_argparse_xgb_only_flag_drops_other_models():
    mod = _import_runner()
    args = mod._build_arg_parser().parse_args(["--models", "xgb"])
    assert args.models == ["xgb"]


def test_argparse_supports_12split_preset_user_originally_asked_about():
    """User in 2026-06-05 asked specifically about ``12-split`` data —
    confirm the preset is exposed end-to-end."""
    mod = _import_runner()
    args = mod._build_arg_parser().parse_args(["--preset", "12split"])
    assert args.preset == "12split"


def test_argparse_early_stop_zero_means_disable():
    mod = _import_runner()
    args = mod._build_arg_parser().parse_args(["--early-stopping-rounds", "0"])
    assert args.early_stopping_rounds == 0


# ── train_xgboost early-stopping wiring ────────────────────────────────

@pytest.fixture
def _synthetic_panel():
    """Tiny synthetic panel with a learnable signal so XGB actually
    converges before hitting n_estimators. Otherwise the test can't
    distinguish "early-stop fired" from "ran to the cap"."""
    rng = np.random.default_rng(0)
    n_train, n_valid = 500, 200
    n_feat = 8
    X_tr = pd.DataFrame(rng.normal(size=(n_train, n_feat)),
                        columns=[f"f{i}" for i in range(n_feat)])
    # signal is the sum of the first three features; rest is noise
    y_tr = pd.Series(X_tr.iloc[:, :3].sum(axis=1) + rng.normal(0, 0.1, n_train))
    X_va = pd.DataFrame(rng.normal(size=(n_valid, n_feat)),
                        columns=X_tr.columns)
    y_va = pd.Series(X_va.iloc[:, :3].sum(axis=1) + rng.normal(0, 0.1, n_valid))
    return X_tr, y_tr, X_va, y_va


def test_train_xgboost_passes_early_stopping_into_params(_synthetic_panel):
    """The returned ``params`` dict must record the early-stop rounds
    the runner passed in, proving the parameter actually reached the
    XGBRegressor init (and was not silently dropped like the pre-fix
    version)."""
    pytest.importorskip("xgboost")
    mod = _import_runner()
    X_tr, y_tr, X_va, y_va = _synthetic_panel
    _model, params = mod.train_xgboost(
        X_tr, y_tr, X_va, y_va, list(X_tr.columns),
        n_estimators=80, early_stopping_rounds=5,
    )
    assert params["n_estimators"] == 80
    assert params["early_stopping_rounds"] == 5


def test_train_xgboost_disabled_early_stop_omits_param(_synthetic_panel):
    """early_stopping_rounds=None reverts to the legacy fixed-rounds
    behaviour — XGBRegressor must NOT see an ``early_stopping_rounds``
    key, otherwise xgboost<1.6 would crash."""
    pytest.importorskip("xgboost")
    mod = _import_runner()
    X_tr, y_tr, X_va, y_va = _synthetic_panel
    _model, params = mod.train_xgboost(
        X_tr, y_tr, X_va, y_va, list(X_tr.columns),
        n_estimators=20, early_stopping_rounds=None,
    )
    assert "early_stopping_rounds" not in params
    assert params["n_estimators"] == 20


def test_train_xgboost_early_stop_actually_fires_before_cap(_synthetic_panel):
    """Smoke check that early stopping isn't only declared but actually
    halts training. With a learnable signal and patience=3, the best
    iteration must land well before n_estimators=300 — pre-fix this
    test would have always trained to exactly 300 trees."""
    pytest.importorskip("xgboost")
    mod = _import_runner()
    X_tr, y_tr, X_va, y_va = _synthetic_panel
    model, _ = mod.train_xgboost(
        X_tr, y_tr, X_va, y_va, list(X_tr.columns),
        n_estimators=300, early_stopping_rounds=3,
    )
    # ``best_iteration`` is set when early stopping fires.
    best = getattr(model, "best_iteration", None)
    assert best is not None, "best_iteration unset — early stop did not engage"
    assert best < 300, (
        f"best_iteration={best} hit the n_estimators cap; "
        f"early stopping likely did not actually halt training"
    )


# ── preset wiring ──────────────────────────────────────────────────────

def test_get_standard_splits_supports_all_runner_presets():
    """All presets the runner exposes must be valid in
    ``config.rolling_splits`` — guard against drift where the runner
    accepts a preset name the config does not."""
    from config.rolling_splits import get_standard_splits
    accepted = _import_runner()._build_arg_parser()._actions
    preset_choices = next(
        a.choices for a in accepted if a.dest == "preset"
    )
    for preset in preset_choices:
        splits = get_standard_splits(preset, end_date="2026-05-31")
        # contract: each split has the seven fields the runner reads
        assert splits, f"preset {preset} produced no splits"
        for s in splits:
            for key in ("split_id", "train_start", "train_end",
                         "valid_start", "valid_end",
                         "test_start", "test_end"):
                assert key in s, f"preset {preset} missing {key}"
