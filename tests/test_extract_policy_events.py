"""Tests for ``scripts/extract_policy_events.py`` — Phase E.1 step 2.

Covers:
  1. Schema validator downgrade — mismatched ``tool_type`` /
     ``policy_stance`` values demote to ``other`` / ``unknown`` rather
     than dropping the row.
  2. Field type enforcement — numeric fields coerce, missing → null,
     unexpectedness clamps to [0,1].
  3. LLM call is mocked end-to-end and the output JSONL matches the
     documented schema (publish_date / policy_type / title / url +
     7 extracted fields + extracted_at).
  4. Idempotent re-run on the same date overwrites atomically (no
     duplicates) and leaves no .tmp file.

All LLM calls are mocked. We never hit MiniMax/DeepSeek in tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from scripts import extract_policy_events as epe


# ─────────────────────────────────────────────────────────────────────
# Sample input row matching collect_policy_texts.py contract
# ─────────────────────────────────────────────────────────────────────
def _input_row(
    *,
    publish_date: str = "2026-06-05",
    policy_type: str = "omo",
    title: str = "公开市场业务交易公告",
    url: str = "http://www.pbc.gov.cn/.../2026/06/05/omo.html",
    content: str = (
        "为维护银行体系流动性合理充裕，人民银行以利率招标方式开展了"
        "1500亿元逆回购操作，中标利率1.40%。当日有800亿元逆回购到期，"
        "实现净投放700亿元。"
    ),
) -> dict:
    return {
        "publish_date": publish_date,
        "policy_type": policy_type,
        "title": title,
        "url": url,
        "content": content,
        "source": "pbc.gov.cn",
        "fetch_time": "2026-06-05T09:30:00Z",
    }


def _write_input_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────
# Test 1 — Schema validator downgrade (invalid tool_type / stance →
#          "other" / "unknown" rather than dropping the row)
# ─────────────────────────────────────────────────────────────────────
def test_schema_validator_downgrades_invalid_enums():
    """LLM emits an out-of-vocab enum -> validator downgrades, not drops."""
    bad_raw = {
        "policy_stance": "hawkish",       # not in {easing|tightening|neutral|unknown}
        "liquidity_injection_amount": None,
        "net_injection": 700.0,
        "repo_rate_change": None,
        "tool_type": "supercannon",       # not in tool_type enum
        "duration_days": 7,
        "unexpectedness": 0.4,
    }
    validated = epe.validate_event(bad_raw)
    assert validated is not None, "schema validator must downgrade, not drop"
    assert validated["tool_type"] == "other"
    assert validated["policy_stance"] == "unknown"
    # Numeric fields kept
    assert validated["net_injection"] == 700.0
    assert validated["duration_days"] == 7
    assert validated["unexpectedness"] == 0.4


# ─────────────────────────────────────────────────────────────────────
# Test 2 — Field type enforcement: coerce numerics, clamp ranges,
#          missing values become None
# ─────────────────────────────────────────────────────────────────────
def test_field_types_are_enforced_and_clamped():
    """Strings that look like numbers coerce. Out-of-range clamps. Missing → None."""
    raw = {
        "policy_stance": "easing",
        "liquidity_injection_amount": "1500",   # string → float
        "net_injection": "700.5",
        "repo_rate_change": "-10",              # string → int (bp)
        "tool_type": "omo",
        "duration_days": "7",
        "unexpectedness": 1.5,                  # out of [0,1] → clamp to 1.0
    }
    out = epe.validate_event(raw)
    assert out["policy_stance"] == "easing"
    assert out["liquidity_injection_amount"] == 1500.0
    assert out["net_injection"] == 700.5
    assert out["repo_rate_change"] == -10
    assert out["tool_type"] == "omo"
    assert out["duration_days"] == 7
    assert out["unexpectedness"] == 1.0

    # Missing-field row: all numerics → None, enums → "unknown"/"other"
    missing = {}
    out2 = epe.validate_event(missing)
    assert out2["policy_stance"] == "unknown"
    assert out2["tool_type"] == "other"
    assert out2["liquidity_injection_amount"] is None
    assert out2["net_injection"] is None
    assert out2["repo_rate_change"] is None
    assert out2["duration_days"] is None
    assert out2["unexpectedness"] is None

    # Negative unexpectedness clamps to 0.0
    raw3 = {"unexpectedness": -0.3}
    out3 = epe.validate_event(raw3)
    assert out3["unexpectedness"] == 0.0


# ─────────────────────────────────────────────────────────────────────
# Test 3 — LLM call mocked end-to-end. Output JSONL has the documented
#          schema and EventStore-bypass path is exercised.
# ─────────────────────────────────────────────────────────────────────
def test_llm_extract_produces_correct_jsonl_schema(tmp_path: Path):
    """Mock the LLM; assert output JSONL schema + extracted fields."""
    # Build a single-row input file
    target_date = "2026-06-05"
    input_root = tmp_path / "policy_texts" / "pbc"
    output_root = tmp_path / "policy_events" / "pbc"
    eventstore_dir = tmp_path / "events"

    _write_input_jsonl(input_root / f"{target_date}.jsonl", [_input_row()])

    # Mock the LLM to always return this canned response
    canned = {
        "policy_stance": "easing",
        "liquidity_injection_amount": 1500.0,
        "net_injection": 700.0,
        "repo_rate_change": None,
        "tool_type": "omo",
        "duration_days": 7,
        "unexpectedness": 0.3,
    }

    def fake_llm(_content: str) -> dict[str, Any] | None:
        return dict(canned)

    summary = epe.extract_pbc(
        target_date=target_date,
        input_dir=input_root,
        output_dir=output_root,
        eventstore_dir=eventstore_dir,
        llm_extract_fn=fake_llm,
    )

    # Output file exists and has one row
    out_path = output_root / f"{target_date}.jsonl"
    assert out_path.exists()
    lines = [
        json.loads(ln) for ln in out_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(lines) == 1
    row = lines[0]

    # Documented schema: source fields + 7 LLM fields + extracted_at
    expected_keys = {
        "publish_date", "policy_type", "title", "url",
        "policy_stance", "liquidity_injection_amount", "net_injection",
        "repo_rate_change", "tool_type", "duration_days", "unexpectedness",
        "extracted_at",
    }
    assert set(row.keys()) == expected_keys, sorted(row.keys())
    assert row["policy_stance"] == "easing"
    assert row["tool_type"] == "omo"
    assert row["net_injection"] == 700.0
    assert row["extracted_at"].endswith("Z")

    # Summary reports n_extracted=1
    assert summary["n_extracted"] == 1
    assert summary["n_input"] == 1
    assert summary["n_failed"] == 0


# ─────────────────────────────────────────────────────────────────────
# Test 4 — Idempotent re-run: same date, two runs, .tmp gone, no dupes
# ─────────────────────────────────────────────────────────────────────
def test_idempotent_rerun_overwrites_atomically(tmp_path: Path):
    """Two runs on the same date produce identical rows; no .tmp left over."""
    target_date = "2026-06-05"
    input_root = tmp_path / "policy_texts" / "pbc"
    output_root = tmp_path / "policy_events" / "pbc"
    eventstore_dir = tmp_path / "events"

    _write_input_jsonl(input_root / f"{target_date}.jsonl", [_input_row()])

    def fake_llm(_content: str):
        return {
            "policy_stance": "easing",
            "liquidity_injection_amount": 1500.0,
            "net_injection": 700.0,
            "repo_rate_change": None,
            "tool_type": "omo",
            "duration_days": 7,
            "unexpectedness": 0.3,
        }

    epe.extract_pbc(
        target_date=target_date,
        input_dir=input_root,
        output_dir=output_root,
        eventstore_dir=eventstore_dir,
        llm_extract_fn=fake_llm,
    )
    epe.extract_pbc(
        target_date=target_date,
        input_dir=input_root,
        output_dir=output_root,
        eventstore_dir=eventstore_dir,
        llm_extract_fn=fake_llm,
    )

    out_path = output_root / f"{target_date}.jsonl"
    assert out_path.exists()
    lines = [
        json.loads(ln) for ln in out_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    # Idempotent: still exactly 1 row, not 2
    assert len(lines) == 1
    # Atomic write: no .tmp leftover
    assert not list(output_root.glob("*.tmp"))


# ─────────────────────────────────────────────────────────────────────
# Test 5 — LLM call failure: do NOT silently write empty events.
#          Must raise / record failure, not write an empty rows file
#          successfully.
# ─────────────────────────────────────────────────────────────────────
def test_llm_call_failure_is_loud_not_silent(tmp_path: Path):
    """LLM raising on every input must surface in summary.n_failed."""
    target_date = "2026-06-05"
    input_root = tmp_path / "policy_texts" / "pbc"
    output_root = tmp_path / "policy_events" / "pbc"
    eventstore_dir = tmp_path / "events"

    _write_input_jsonl(
        input_root / f"{target_date}.jsonl",
        [_input_row(url="http://e.com/a"), _input_row(url="http://e.com/b")],
    )

    def fake_llm_fail(_content: str):
        raise RuntimeError("LLM API key absent")

    summary = epe.extract_pbc(
        target_date=target_date,
        input_dir=input_root,
        output_dir=output_root,
        eventstore_dir=eventstore_dir,
        llm_extract_fn=fake_llm_fail,
    )
    assert summary["n_input"] == 2
    assert summary["n_extracted"] == 0
    assert summary["n_failed"] == 2
