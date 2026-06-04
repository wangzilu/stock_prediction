"""Behavior pin: contract-violation gate stays fail-closed (cx round 2).

Round 1 of tonight's P0/P1 closeout shipped a
``except FeatureContractViolation: raise`` inside the inner block of
``scheduler/jobs.py::_load_lgb_predictions`` — but the OUTER
``except Exception: return _use_cache(str(e))`` swallowed it and
routed the request to the stale cache. So the "hard fail" gate was
actually fail-open.

cx round 2 P0-1 fixed it by adding an explicit
``except FeatureContractViolation: raise`` at the OUTER level before
the catch-all. This file pins that behavior so a future "let's clean
up exception handling" PR cannot silently re-open the暗道.

cx round 2 P0-2: ``scripts/train_lgb.py`` previously wrapped the
supplementary-feature injection in ``try/except Exception: print("skipped")``,
allowing the next ``model.fit()`` to run at Alpha158-only dim and the
new artifact to be saved as the production champion. The contract
test pins that the inject block is no longer in a try/except, so any
inject failure aborts training before save.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -----------------------------------------------------------------------------
# P0-1: jobs.py FeatureContractViolation must propagate, not route to cache
# -----------------------------------------------------------------------------

def test_lgb_predictions_propagates_contract_violation_not_silent_cache():
    """Mock ShortTermModel.load_from_pickle to raise
    FeatureContractViolation. _load_lgb_predictions must re-raise, NOT
    silently call _use_cache. A pre-fix run swallowed the exception
    via the outer ``except Exception`` and shipped yesterday's stale
    cache — exactly the 6-3 22:00 incident."""
    from scheduler.jobs import DailyPipeline
    from models.short_term import FeatureContractViolation

    pipeline = DailyPipeline()
    # Clear any cached predictions from prior tests
    if hasattr(pipeline, "_lgb_predictions"):
        delattr(pipeline, "_lgb_predictions")

    with patch(
        "models.short_term.ShortTermModel.load_from_pickle",
        side_effect=FeatureContractViolation(
            "test: trained 242, inference handler 158 → mismatch"
        ),
    ):
        with pytest.raises(FeatureContractViolation):
            pipeline._load_lgb_predictions()


def test_lgb_predictions_non_contract_failure_still_falls_back_to_cache():
    """Sanity contrast: a generic Exception (e.g. corrupt pickle,
    network blip, OOM) SHOULD still route to _use_cache. The hard-fail
    gate is specifically for FeatureContractViolation. This pin
    documents the intent — if someone "simplifies" both exceptions to
    the same handler later, the test fails and they revisit which
    failure modes are recoverable."""
    from scheduler.jobs import DailyPipeline

    pipeline = DailyPipeline()
    if hasattr(pipeline, "_lgb_predictions"):
        delattr(pipeline, "_lgb_predictions")

    with patch(
        "models.short_term.ShortTermModel.load_from_pickle",
        side_effect=RuntimeError("generic non-contract failure"),
    ), patch(
        "models.lgb_cache.load_prediction_cache",
        return_value=({"SH600519": 0.01}, {"latest_date": "2026-06-03"}),
    ):
        result = pipeline._load_lgb_predictions()
        assert result == {"SH600519": 0.01}, (
            "non-contract failures should still hit the cache fallback — "
            "the hard-fail gate is narrow, only FeatureContractViolation"
        )


# -----------------------------------------------------------------------------
# P0-2: train_lgb.py inject failure must abort training before save
# -----------------------------------------------------------------------------

TRAIN_LGB = (PROJECT_ROOT / "scripts" / "train_lgb.py").read_text()


def test_train_lgb_inject_not_in_try_except():
    """The supplementary injection block must NOT be wrapped in a
    catch-all try/except. Pre-fix it was:
        try:
            ... merger.inject_supplementary_into_handler(...) ...
        except Exception as e:
            print(f"Supplementary feature merge skipped: {e}")
            ...
    Any RuntimeError (loader missing, parquet corrupt, etc.) was
    silently swallowed and the training continued at Alpha158-only
    dim. The pin: after the inject call, an exception MUST propagate."""
    # Find the inject call site
    assert "merger.inject_supplementary_into_handler(" in TRAIN_LGB, (
        "scripts/train_lgb.py no longer calls "
        "inject_supplementary_into_handler — P0-c regressed?"
    )
    # Crude AST-free check: the file must NOT contain the pre-fix string
    # "Supplementary feature merge skipped" (the print that hid the
    # silent-fallback behavior).
    assert "Supplementary feature merge skipped" not in TRAIN_LGB, (
        "scripts/train_lgb.py still has the pre-fix silent-fallback "
        "'skipped' print. P0-2 regressed: inject failure will swallow."
    )


def test_train_lgb_raises_when_supp_cols_zero():
    """When inject_supplementary_into_handler returns 0, train_lgb
    MUST raise rather than silently train a 158-dim model and save
    it under the production artifact name. Pin via source-text check
    because the alternative (importing main() and forcing zero) needs
    full qlib data."""
    assert (
        "Refusing to save a 158-dim model under the 242-dim contract"
        in TRAIN_LGB
    ), (
        "P0-2 gate phrase missing — when inject returns 0 cols, "
        "train_lgb must raise RuntimeError before model.fit()."
    )
