"""cx round 3 P2 #84 follow-up — pin run_paper_trading.py actually
threads cost_model + vol_adv_snapshot into PaperOMS.

Previous PR (commit 02b74f7) added the chokepoint but production paper
continued using bare slippage_rate because the run_paper_trading entry
never instantiated a CostModel(impact_model="sqrt_adv") and never
loaded a snapshot. These tests fix that contract:

  1. With --impact-model=fixed (default), _build_cost_inputs returns
     (None, None) — no behaviour change vs pre-fix runs.
  2. With --impact-model=sqrt_adv, _build_cost_inputs builds a
     CostModel(impact_model='sqrt_adv') AND a vol/adv snapshot.
  3. The snapshot is loaded via paper.cost_inputs.load_or_build_snapshot,
     i.e. cached on disk for re-run efficiency.
  4. A snapshot-build failure degrades gracefully — production paper
     never aborts because qlib is offline; sqrt_adv just falls back
     per-fill to bare rate.

These tests are offline: they monkeypatch the qlib loader and never
read the real qlib provider URI.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# -----------------------------------------------------------------------------
# Default: --impact-model=fixed → no CostModel, no snapshot
# -----------------------------------------------------------------------------

def test_fixed_impact_model_returns_no_cost_inputs(monkeypatch):
    """Default behaviour: bare slippage_rate via PaperOMS.slippage_rate."""
    from scripts.run_paper_trading import _build_cost_inputs

    cm, snap = _build_cost_inputs(
        date="2026-06-03",
        impact_model="fixed",
        impact_coefficient=0.1,
        lookback_days=20,
    )
    assert cm is None, (
        "fixed impact model must produce no CostModel — PaperOMS would "
        "otherwise route through CostModel._slippage and change costs"
    )
    assert snap is None


# -----------------------------------------------------------------------------
# sqrt_adv: builds CostModel + tries to load snapshot
# -----------------------------------------------------------------------------

def test_sqrt_adv_impact_model_builds_cost_model_and_snapshot(monkeypatch):
    """With sqrt_adv requested, get a CostModel(impact_model='sqrt_adv')
    plus a snapshot dict built from the test loader."""
    from scripts.run_paper_trading import _build_cost_inputs

    # Patch qlib init + snapshot loader BEFORE the function runs them
    monkeypatch.setattr("config.qlib_runtime.init_qlib", lambda *a, **k: None)

    fake_snapshot = {"SH600519": {"vol": 0.02, "adv": 1e8}}
    mock_loader = MagicMock(return_value=fake_snapshot)
    monkeypatch.setattr("paper.cost_inputs.load_or_build_snapshot",
                         mock_loader)

    cm, snap = _build_cost_inputs(
        date="2026-06-03",
        impact_model="sqrt_adv",
        impact_coefficient=0.25,
        lookback_days=20,
    )
    assert cm is not None
    assert cm.impact_model == "sqrt_adv"
    assert cm.impact_coefficient == 0.25
    assert snap == fake_snapshot

    # And the loader was called with the right kwargs
    mock_loader.assert_called_once_with("2026-06-03", lookback_days=20)


def test_sqrt_adv_default_date_uses_today(monkeypatch):
    """When --date is not passed, sqrt_adv path uses today's date for
    the snapshot key. Pins the cron-fresh contract."""
    from scripts.run_paper_trading import _build_cost_inputs

    monkeypatch.setattr("config.qlib_runtime.init_qlib", lambda *a, **k: None)
    captured = {}

    def _stub_loader(asof, lookback_days=20):
        captured["asof"] = asof
        return {}

    monkeypatch.setattr("paper.cost_inputs.load_or_build_snapshot", _stub_loader)
    _build_cost_inputs(
        date=None, impact_model="sqrt_adv",
        impact_coefficient=0.1, lookback_days=20,
    )
    # Expect today's local date (we don't pin a specific value because
    # the test environment's clock isn't part of the contract).
    assert captured.get("asof"), "loader did not receive an asof date"
    # Has the YYYY-MM-DD shape
    import re
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", captured["asof"])


# -----------------------------------------------------------------------------
# Failure path: snapshot build fails → degrade gracefully
# -----------------------------------------------------------------------------

def test_snapshot_failure_returns_costmodel_with_none_snapshot(monkeypatch):
    """If snapshot construction raises (e.g. qlib data offline), the
    cost-model is still produced (sqrt_adv enabled) but snapshot is
    None — fill sites then fall back per-fill to bare rate. Production
    paper must NOT abort on this."""
    from scripts.run_paper_trading import _build_cost_inputs

    monkeypatch.setattr("config.qlib_runtime.init_qlib", lambda *a, **k: None)

    def _broken_loader(asof, lookback_days=20):
        raise RuntimeError("qlib data not available in test")

    monkeypatch.setattr("paper.cost_inputs.load_or_build_snapshot",
                         _broken_loader)

    cm, snap = _build_cost_inputs(
        date="2026-06-03", impact_model="sqrt_adv",
        impact_coefficient=0.1, lookback_days=20,
    )
    # CostModel still constructed — the loop downstream uses it; just
    # without a snapshot, _compute_slippage's CostModel._slippage will
    # fall back to bare rate.
    assert cm is not None
    assert cm.impact_model == "sqrt_adv"
    assert snap is None


# -----------------------------------------------------------------------------
# Source-level: PaperOMS is instantiated with the produced inputs
# -----------------------------------------------------------------------------

def test_main_passes_cost_model_and_snapshot_to_paper_oms_in_source():
    """Anti-regression: scripts/run_paper_trading.py source must
    construct PaperOMS with both cost_model AND vol_adv_snapshot kwargs.
    A refactor that drops either reverts the wiring."""
    from pathlib import Path
    src = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_paper_trading.py"
    ).read_text()

    assert "PaperOMS(" in src, "PaperOMS instantiation missing"
    # Find the PaperOMS(...) call and confirm both kwargs are inside it
    idx = src.find("PaperOMS(")
    # Allow up to 1000 chars between the open paren and matching close
    fragment = src[idx:idx + 1000]
    assert "cost_model=" in fragment, (
        "PaperOMS instantiation does NOT pass cost_model — sqrt_adv "
        "path is dead in production paper"
    )
    assert "vol_adv_snapshot=" in fragment, (
        "PaperOMS instantiation does NOT pass vol_adv_snapshot — "
        "sqrt_adv path falls back to bare rate at every fill"
    )


def test_argparse_exposes_impact_model_flag():
    """The new --impact-model flag must be in the source so cron entries
    can opt into sqrt_adv via cron command line."""
    from pathlib import Path
    src = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_paper_trading.py"
    ).read_text()
    assert "--impact-model" in src
    assert '"fixed"' in src and '"sqrt_adv"' in src
