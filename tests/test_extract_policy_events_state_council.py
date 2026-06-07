"""Tests for scripts/extract_policy_events.py --source state_council.

Phase E.2 (PE-2) step 2 — mirror of test_extract_policy_events.py.
Covers:

  1. Schema validator downgrade — out-of-vocab policy_direction →
     "neutral", out-of-vocab subsidy_or_tax → "neither", never drop a row.
  2. Field types — fiscal_support coerces numerics, policy_strength
     clamps to [0,1], regulatory_tightening coerces to bool, deadline
     normalizes to YYYY-MM-DD or None.
  3. LLM end-to-end with mocked LLM, output JSONL schema is exactly
     the 4 source fields + 7 extracted fields + extracted_at.
  4. SYSTEM_PROMPT_SC threaded as a per-call kwarg to _call_llm — NO
     monkey patch of the module-global SYSTEM_PROMPT_V2.

All LLM calls are mocked. We never hit MiniMax / DeepSeek in tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import extract_policy_events as epe


# ─────────────────────────────────────────────────────────────────────
# Sample input row matching collect_policy_texts.py (PE-2) contract
# ─────────────────────────────────────────────────────────────────────
def _input_row(
    *,
    publish_date: str = "2026-06-05",
    policy_type: str = "state_council_doc",
    title: str = "国务院办公厅关于半导体产业升级的指导意见",
    url: str = "http://www.gov.cn/zhengce/zhengceku/2026-06/05/content_5800002.html",
    content: str = (
        "为推动半导体产业升级，国务院办公厅决定安排中央财政资金500亿元"
        "支持先进制程研发。对采购国产半导体设备的企业给予税收减免。"
        "本意见自发布之日起实施，2027年12月31日前有效。"
    ),
) -> dict:
    return {
        "publish_date": publish_date,
        "policy_type": policy_type,
        "title": title,
        "url": url,
        "content": content,
        "source": "gov.cn",
        "fetch_time": "2026-06-05T15:30:00Z",
    }


def _write_input_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────
# Test 1 — Schema validator downgrade for out-of-vocab enums.
# ─────────────────────────────────────────────────────────────────────
def test_sc_schema_validator_downgrades_invalid_enums():
    """LLM emits an out-of-vocab enum -> validator downgrades, not drops."""
    bad_raw = {
        "target_industries": ["Semiconductor", "renewable_energy"],
        "policy_direction": "veryGoodForStocks",  # not in supportive|restrictive|neutral
        "policy_strength": 0.7,
        "fiscal_support": 500,
        "subsidy_or_tax": "windfall",  # not in our vocab
        "regulatory_tightening": False,
        "implementation_deadline": "2027-12-31",
    }
    out = epe.validate_event_state_council(bad_raw)
    assert out is not None
    assert out["policy_direction"] == "neutral"
    assert out["subsidy_or_tax"] == "neither"
    # Industries lowercased + kept as a list (the validator does NOT
    # check membership against a closed vocabulary — too risky given
    # the taxonomy evolves).
    assert "semiconductor" in out["target_industries"]
    assert "renewable_energy" in out["target_industries"]
    assert out["fiscal_support"] == 500.0
    assert out["policy_strength"] == 0.7
    assert out["regulatory_tightening"] is False
    assert out["implementation_deadline"] == "2027-12-31"


# ─────────────────────────────────────────────────────────────────────
# Test 2 — Field type enforcement.
# ─────────────────────────────────────────────────────────────────────
def test_sc_field_types_are_enforced_and_clamped():
    """Numerics coerce; strengths clamp; deadline normalizes; missing → None."""
    raw = {
        "target_industries": "semiconductor, renewable_energy",  # str → list
        "policy_direction": "Supportive",
        "policy_strength": 1.7,  # clamp to 1.0
        "fiscal_support": "500亿元",  # strip unit
        "subsidy_or_tax": "tax_reduction",
        "regulatory_tightening": "true",  # string → bool
        "implementation_deadline": "2027-12-31",
    }
    out = epe.validate_event_state_council(raw)
    assert "semiconductor" in out["target_industries"]
    assert "renewable_energy" in out["target_industries"]
    assert out["policy_direction"] == "supportive"
    assert out["policy_strength"] == 1.0
    assert out["fiscal_support"] == 500.0
    assert out["subsidy_or_tax"] == "tax_reduction"
    assert out["regulatory_tightening"] is True
    assert out["implementation_deadline"] == "2027-12-31"

    # Missing-everything row: enums demote, numerics None, list empty,
    # bool False, deadline None.
    out2 = epe.validate_event_state_council({})
    assert out2["target_industries"] == []
    assert out2["policy_direction"] == "neutral"
    assert out2["policy_strength"] is None
    assert out2["fiscal_support"] is None
    assert out2["subsidy_or_tax"] == "neither"
    assert out2["regulatory_tightening"] is False
    assert out2["implementation_deadline"] is None

    # Negative strength clamps to 0.0.
    out3 = epe.validate_event_state_council({"policy_strength": -0.5})
    assert out3["policy_strength"] == 0.0

    # Garbage deadline → None.
    out4 = epe.validate_event_state_council({"implementation_deadline": "soon"})
    assert out4["implementation_deadline"] is None


# ─────────────────────────────────────────────────────────────────────
# Test 3 — LLM end-to-end mocked; output JSONL schema is documented set.
# ─────────────────────────────────────────────────────────────────────
def test_sc_llm_extract_produces_correct_jsonl_schema(tmp_path: Path):
    target_date = "2026-06-05"
    input_root = tmp_path / "policy_texts" / "state_council"
    output_root = tmp_path / "policy_events" / "state_council"
    eventstore_dir = tmp_path / "events"

    _write_input_jsonl(input_root / f"{target_date}.jsonl", [_input_row()])

    canned = {
        "target_industries": ["semiconductor"],
        "policy_direction": "supportive",
        "policy_strength": 0.8,
        "fiscal_support": 500.0,
        "subsidy_or_tax": "tax_reduction",
        "regulatory_tightening": False,
        "implementation_deadline": "2027-12-31",
    }

    def fake_llm(_content: str) -> dict[str, Any] | None:
        return dict(canned)

    summary = epe.extract_state_council(
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
    assert len(lines) == 1
    row = lines[0]

    expected_keys = {
        "publish_date", "policy_type", "title", "url",
        "target_industries", "policy_direction", "policy_strength",
        "fiscal_support", "subsidy_or_tax", "regulatory_tightening",
        "implementation_deadline", "extracted_at",
    }
    assert set(row.keys()) == expected_keys, sorted(row.keys())
    assert row["target_industries"] == ["semiconductor"]
    assert row["policy_direction"] == "supportive"
    assert row["policy_strength"] == 0.8
    assert row["fiscal_support"] == 500.0
    assert row["extracted_at"].endswith("Z")

    assert summary["n_extracted"] == 1
    assert summary["n_input"] == 1
    assert summary["n_failed"] == 0


# ─────────────────────────────────────────────────────────────────────
# Test 4 — Idempotent re-run: same date, two runs, no dupes, no .tmp.
# ─────────────────────────────────────────────────────────────────────
def test_sc_idempotent_rerun(tmp_path: Path):
    target_date = "2026-06-05"
    input_root = tmp_path / "policy_texts" / "state_council"
    output_root = tmp_path / "policy_events" / "state_council"

    _write_input_jsonl(input_root / f"{target_date}.jsonl", [_input_row()])

    def fake_llm(_content: str):
        return {
            "target_industries": ["semiconductor"],
            "policy_direction": "supportive",
            "policy_strength": 0.7,
            "fiscal_support": 500,
            "subsidy_or_tax": "subsidy",
            "regulatory_tightening": False,
            "implementation_deadline": "2027-12-31",
        }

    epe.extract_state_council(
        target_date=target_date,
        input_dir=input_root,
        output_dir=output_root,
        eventstore_dir=tmp_path / "events",
        llm_extract_fn=fake_llm,
    )
    epe.extract_state_council(
        target_date=target_date,
        input_dir=input_root,
        output_dir=output_root,
        eventstore_dir=tmp_path / "events",
        llm_extract_fn=fake_llm,
    )

    out_path = output_root / f"{target_date}.jsonl"
    lines = [
        json.loads(ln) for ln in out_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(lines) == 1
    assert not list(output_root.glob("*.tmp"))


# ─────────────────────────────────────────────────────────────────────
# Test 5 — SYSTEM_PROMPT_SC threaded through _call_llm as a kwarg.
#          MUST NOT mutate factors.llm_event_extractor_v2.SYSTEM_PROMPT_V2.
# ─────────────────────────────────────────────────────────────────────
def test_sc_system_prompt_threaded_through_call_llm(monkeypatch):
    """``_llm_extract_with_state_council_prompt`` must pass
    SYSTEM_PROMPT_SC to LLMEventExtractorV2._call_llm via the
    per-call kwarg. The module-global SYSTEM_PROMPT_V2 must not be
    touched (concurrency safety: the per-stock LLM pipeline runs in
    the same process).
    """
    from factors import llm_event_extractor_v2 as v2

    captured: dict[str, str] = {}
    original_global = v2.SYSTEM_PROMPT_V2

    def fake_call_llm(self, user_prompt, system_prompt=None):
        captured["user_prompt"] = user_prompt
        captured["system_prompt"] = system_prompt
        return (
            '{"target_industries": ["semiconductor"], '
            '"policy_direction": "supportive", "policy_strength": 0.7}',
            {"http_ok": True},
        )

    monkeypatch.setattr(v2.LLMEventExtractorV2, "_call_llm", fake_call_llm)
    monkeypatch.setenv("MINIMAX_API_KEY", "test")

    out = epe._llm_extract_with_state_council_prompt("test SC content body")

    assert out is not None
    assert captured["system_prompt"] is epe.SYSTEM_PROMPT_SC, (
        "_llm_extract_with_state_council_prompt must pass "
        "SYSTEM_PROMPT_SC; got %r" % (captured.get("system_prompt"),)
    )
    # Concurrency invariant: the module-global must NOT have been mutated.
    assert v2.SYSTEM_PROMPT_V2 == original_global, (
        "SYSTEM_PROMPT_V2 was mutated — monkey patch regression. "
        "PE-2 must use the per-call kwarg, not a module-global swap."
    )
    # And the SC prompt must NOT have leaked into the V2 global.
    assert v2.SYSTEM_PROMPT_V2 != epe.SYSTEM_PROMPT_SC


# ─────────────────────────────────────────────────────────────────────
# Test 6 — LLM call failure: n_failed reflects reality (no silent
#          empty-row write).
# ─────────────────────────────────────────────────────────────────────
def test_sc_llm_call_failure_is_loud(tmp_path: Path):
    target_date = "2026-06-05"
    input_root = tmp_path / "policy_texts" / "state_council"
    output_root = tmp_path / "policy_events" / "state_council"

    _write_input_jsonl(
        input_root / f"{target_date}.jsonl",
        [_input_row(url="http://e.com/a"), _input_row(url="http://e.com/b")],
    )

    def fake_fail(_content: str):
        raise RuntimeError("LLM API key absent")

    summary = epe.extract_state_council(
        target_date=target_date,
        input_dir=input_root,
        output_dir=output_root,
        eventstore_dir=tmp_path / "events",
        llm_extract_fn=fake_fail,
    )
    assert summary["n_input"] == 2
    assert summary["n_extracted"] == 0
    assert summary["n_failed"] == 2
