"""Regression tests for Phase A.5 shadow containment hardfixes."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from factors.candidate_sanitizer import CandidateSanitizer
from scheduler.jobs import DailyPipeline


def _quote():
    return {"最新价": 10.0, "最高": 10.5, "最低": 9.8, "成交量": 1_000_000}


def test_chain_alpha_is_soft_tag_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(CandidateSanitizer, "_is_st_by_code", lambda self, code: False)
    monkeypatch.setattr(CandidateSanitizer, "_is_new_listing", lambda self, code: False)

    sanitizer = CandidateSanitizer(
        today="2026-06-05",
        st_list_path=tmp_path / "missing.json",
        chain_alpha={"SH600000": -99.0},
        require_quote=True,
    )

    ok, reason = sanitizer.check("SH600000", "浦发银行", quote=_quote())

    assert ok
    assert reason is None
    assert sanitizer.last_chain_alpha == -99.0
    assert "chain_negative" in sanitizer.last_soft_tags


def test_chain_alpha_can_be_hard_blocked_explicitly(monkeypatch, tmp_path):
    monkeypatch.setattr(CandidateSanitizer, "_is_st_by_code", lambda self, code: False)
    monkeypatch.setattr(CandidateSanitizer, "_is_new_listing", lambda self, code: False)

    sanitizer = CandidateSanitizer(
        today="2026-06-05",
        st_list_path=tmp_path / "missing.json",
        chain_alpha={"SH600000": -99.0},
        enable_chain_hard_block=True,
        require_quote=True,
    )

    ok, reason = sanitizer.check("SH600000", "浦发银行", quote=_quote())

    assert not ok
    assert reason == "chain_negative"


def test_chain_alpha_loader_refuses_future_only_rows(monkeypatch, tmp_path):
    data_dir = tmp_path / "storage"
    data_dir.mkdir()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-06-09"), "SH600000")],
        names=["datetime", "instrument"],
    )
    pd.DataFrame({"global_chain_alpha": [-1.0]}, index=idx).to_parquet(
        data_dir / "global_chain_factors.parquet"
    )

    import config.settings as settings
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)

    pipeline = DailyPipeline.__new__(DailyPipeline)

    assert pipeline._load_chain_alpha_for_sanitizer("2026-06-05") is None


def test_chain_alpha_loader_uses_asof_past_row(monkeypatch, tmp_path):
    data_dir = tmp_path / "storage"
    data_dir.mkdir()
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-06-04"), "SH600000")],
        names=["datetime", "instrument"],
    )
    pd.DataFrame({"global_chain_alpha": [-3.5]}, index=idx).to_parquet(
        data_dir / "global_chain_factors.parquet"
    )

    import config.settings as settings
    monkeypatch.setattr(settings, "DATA_DIR", data_dir)
    monkeypatch.setattr(
        "scheduler.data_health.trading_day_age",
        lambda older_date, reference_date=None: 1,
    )

    pipeline = DailyPipeline.__new__(DailyPipeline)

    assert pipeline._load_chain_alpha_for_sanitizer("2026-06-05") == {"SH600000": -3.5}
