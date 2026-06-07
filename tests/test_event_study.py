"""Tests for ``scripts/event_study.py`` — PE-6 (task #145).

Step 1 covers the CLI shape only:
  - --source must be one of the 7 known sources
  - --window parses 'lo,hi' with lo <= hi
  - defaults: --benchmark sh000300, --window -5,5, --out-dir under
    data/storage/event_study
"""
from __future__ import annotations

import pytest

from scripts import event_study as es


def test_parse_window_default_5_5():
    lo, hi = es.parse_window("-5,5")
    assert (lo, hi) == (-5, 5)


def test_parse_window_asymmetric():
    lo, hi = es.parse_window("-2,10")
    assert (lo, hi) == (-2, 10)


def test_parse_window_rejects_swapped():
    with pytest.raises(ValueError):
        es.parse_window("5,-5")


def test_parse_window_rejects_bad_format():
    with pytest.raises(ValueError):
        es.parse_window("not-a-window")


def test_parse_args_minimum():
    cfg = es.parse_args(
        ["--source", "llm", "--start", "2026-04-01", "--end", "2026-04-30"]
    )
    assert cfg.source == "llm"
    assert cfg.start == "2026-04-01"
    assert cfg.end == "2026-04-30"
    assert cfg.window_lo == -5
    assert cfg.window_hi == 5
    assert cfg.benchmark == es.DEFAULT_BENCHMARK
    assert cfg.top_n is None


def test_parse_args_window_override():
    # Argparse treats a leading "-" in an optional value as a new flag;
    # using "=" makes the assignment explicit.
    cfg = es.parse_args(
        [
            "--source", "pe1",
            "--start", "2024-01-01", "--end", "2024-12-31",
            "--window=-3,10",
        ]
    )
    assert (cfg.window_lo, cfg.window_hi) == (-3, 10)


def test_parse_args_rejects_unknown_source():
    with pytest.raises(SystemExit):
        es.parse_args(
            ["--source", "what", "--start", "2026-01-01", "--end", "2026-01-31"]
        )


def test_supported_sources_match_phase_doc():
    # The 7 sources called out in the task spec.
    assert set(es.SUPPORTED_SOURCES) == {
        "pe1", "pe2", "pe3", "pe4", "llm", "chain_rule", "chain_llm",
    }
