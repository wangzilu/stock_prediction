"""Tests for scripts/extract_policy_events.py --source nbs.

Phase E.3 (PE-3) step 2 — mirror of test_extract_policy_events_state_council.py
for the NBS macro-surprise branch. Covers:

  1. Schema validator downgrade — out-of-vocab series_name → "other",
     out-of-vocab surprise_direction → "unknown", never drop a row.
  2. Field types — headline / consensus / mom / yoy coerce numerics
     and strip units, release_period normalizes to YYYY-MM.
  3. LLM end-to-end with mocked LLM, output JSONL schema is exactly
     the 4 source fields + 8 extracted fields + extracted_at.
  4. Idempotent re-run: same date, two runs, no dupes, no .tmp.
  5. SYSTEM_PROMPT_NBS threaded as a per-call kwarg to _call_llm — NO
     monkey patch of the module-global SYSTEM_PROMPT_V2.
  6. LLM-failure loudness: LLM raising on every input → n_failed == n_input.

All LLM calls are mocked. We never hit MiniMax / DeepSeek in tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import extract_policy_events as epe


# ─────────────────────────────────────────────────────────────────────
# Sample input row matching collect_policy_texts.py (PE-3) contract
# ─────────────────────────────────────────────────────────────────────
def _input_row(
    *,
    publish_date: str = "2026-06-03",
    policy_type: str = "cpi_monthly",
    title: str = "2026年5月份居民消费价格变动情况",
    url: str = "http://www.stats.gov.cn/sj/zxfb/2026-06/03/content_7100001.html",
    content: str = (
        "2026年5月份，全国居民消费价格（CPI）同比上涨0.3%，"
        "环比上涨0.1%。市场普遍预期同比上涨0.5%，实际值低于预期。"
    ),
) -> dict:
    return {
        "publish_date": publish_date,
        "policy_type": policy_type,
        "title": title,
        "url": url,
        "content": content,
        "source": "stats.gov.cn",
        "fetch_time": "2026-06-03T15:40:00Z",
    }


def _write_input_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────
# Test 1 — Schema validator downgrade for out-of-vocab enums.
# ─────────────────────────────────────────────────────────────────────
def test_nbs_schema_validator_downgrades_invalid_enums():
    """LLM emits out-of-vocab enums → validator downgrades, not drops."""
    bad_raw = {
        "series_name": "inflation_index",  # not in cpi|ppi|pmi|retail_sales|other
        "release_period": "2026-05",
        "headline_value": 0.3,
        "prior_value": 0.4,
        "consensus_value": 0.5,
        "mom_change": 0.1,
        "yoy_change": 0.3,
        "surprise_direction": "bullish_for_bonds",  # not in our vocab
    }
    out = epe.validate_event_nbs(bad_raw)
    assert out is not None
    assert out["series_name"] == "other"
    assert out["surprise_direction"] == "unknown"
    # Numeric fields kept intact.
    assert out["release_period"] == "2026-05"
    assert out["headline_value"] == 0.3
    assert out["consensus_value"] == 0.5
    assert out["yoy_change"] == 0.3


# ─────────────────────────────────────────────────────────────────────
# Test 2 — Field type enforcement: numerics coerce / units stripped,
#          release_period normalizes, missing → None.
# ─────────────────────────────────────────────────────────────────────
def test_nbs_field_types_are_enforced_and_coerced():
    """Strings coerce; units stripped; missing → None; YYYY-MM-DD truncates
    to YYYY-MM."""
    raw = {
        "series_name": "CPI",
        "release_period": "2026-05-15",          # truncates to YYYY-MM
        "headline_value": "0.3%",                # strip % sign
        "prior_value": "0.4",
        "consensus_value": "0.5",
        "mom_change": "0.1个百分点",              # strip Chinese unit
        "yoy_change": "0.3%",
        "surprise_direction": "Downside",
    }
    out = epe.validate_event_nbs(raw)
    assert out["series_name"] == "cpi"
    assert out["release_period"] == "2026-05", out["release_period"]
    assert out["headline_value"] == 0.3
    assert out["prior_value"] == 0.4
    assert out["consensus_value"] == 0.5
    assert out["mom_change"] == 0.1
    assert out["yoy_change"] == 0.3
    assert out["surprise_direction"] == "downside"

    # Missing-everything row.
    out2 = epe.validate_event_nbs({})
    assert out2["series_name"] == "other"
    assert out2["release_period"] is None
    assert out2["headline_value"] is None
    assert out2["consensus_value"] is None
    assert out2["mom_change"] is None
    assert out2["yoy_change"] is None
    assert out2["surprise_direction"] == "unknown"

    # Garbage release_period → None (does not crash).
    out3 = epe.validate_event_nbs({"release_period": "soon"})
    assert out3["release_period"] is None

    # Period normalization: zero-pads month.
    out4 = epe.validate_event_nbs({"release_period": "2026-5"})
    assert out4["release_period"] == "2026-05"


# ─────────────────────────────────────────────────────────────────────
# Test 3 — LLM end-to-end mocked; output JSONL schema is documented set.
# ─────────────────────────────────────────────────────────────────────
def test_nbs_llm_extract_produces_correct_jsonl_schema(tmp_path: Path):
    target_date = "2026-06-03"
    input_root = tmp_path / "policy_texts" / "nbs"
    output_root = tmp_path / "policy_events" / "nbs"
    eventstore_dir = tmp_path / "events"

    _write_input_jsonl(input_root / f"{target_date}.jsonl", [_input_row()])

    canned = {
        "series_name": "cpi",
        "release_period": "2026-05",
        "headline_value": 0.3,
        "prior_value": 0.4,
        "consensus_value": 0.5,
        "mom_change": 0.1,
        "yoy_change": 0.3,
        "surprise_direction": "downside",
    }

    def fake_llm(_content: str) -> dict[str, Any] | None:
        return dict(canned)

    summary = epe.extract_nbs(
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
        "series_name", "release_period", "headline_value", "prior_value",
        "consensus_value", "mom_change", "yoy_change", "surprise_direction",
        "extracted_at",
    }
    assert set(row.keys()) == expected_keys, sorted(row.keys())
    assert row["series_name"] == "cpi"
    assert row["release_period"] == "2026-05"
    assert row["headline_value"] == 0.3
    assert row["surprise_direction"] == "downside"
    assert row["extracted_at"].endswith("Z")

    assert summary["n_extracted"] == 1
    assert summary["n_input"] == 1
    assert summary["n_failed"] == 0


# ─────────────────────────────────────────────────────────────────────
# Test 4 — Idempotent re-run.
# ─────────────────────────────────────────────────────────────────────
def test_nbs_idempotent_rerun(tmp_path: Path):
    target_date = "2026-06-03"
    input_root = tmp_path / "policy_texts" / "nbs"
    output_root = tmp_path / "policy_events" / "nbs"

    _write_input_jsonl(input_root / f"{target_date}.jsonl", [_input_row()])

    def fake_llm(_content: str):
        return {
            "series_name": "cpi",
            "release_period": "2026-05",
            "headline_value": 0.3,
            "prior_value": 0.4,
            "consensus_value": 0.5,
            "mom_change": 0.1,
            "yoy_change": 0.3,
            "surprise_direction": "downside",
        }

    epe.extract_nbs(
        target_date=target_date,
        input_dir=input_root, output_dir=output_root,
        eventstore_dir=tmp_path / "events",
        llm_extract_fn=fake_llm,
    )
    epe.extract_nbs(
        target_date=target_date,
        input_dir=input_root, output_dir=output_root,
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
# Test 5 — SYSTEM_PROMPT_NBS threaded through _call_llm as a kwarg.
#          MUST NOT mutate factors.llm_event_extractor_v2.SYSTEM_PROMPT_V2.
# ─────────────────────────────────────────────────────────────────────
def test_nbs_system_prompt_threaded_through_call_llm(monkeypatch):
    """``_llm_extract_with_nbs_prompt`` must pass SYSTEM_PROMPT_NBS to
    LLMEventExtractorV2._call_llm via the per-call kwarg. The module-
    global SYSTEM_PROMPT_V2 must NOT be touched (concurrency safety:
    the per-stock LLM pipeline runs in the same process).
    """
    from factors import llm_event_extractor_v2 as v2

    captured: dict[str, str] = {}
    original_global = v2.SYSTEM_PROMPT_V2

    def fake_call_llm(self, user_prompt, system_prompt=None):
        captured["user_prompt"] = user_prompt
        captured["system_prompt"] = system_prompt
        return (
            '{"series_name": "cpi", "release_period": "2026-05", '
            '"headline_value": 0.3, "surprise_direction": "downside"}',
            {"http_ok": True},
        )

    monkeypatch.setattr(v2.LLMEventExtractorV2, "_call_llm", fake_call_llm)
    monkeypatch.setenv("MINIMAX_API_KEY", "test")

    out = epe._llm_extract_with_nbs_prompt("test NBS content body")

    assert out is not None
    assert captured["system_prompt"] is epe.SYSTEM_PROMPT_NBS, (
        "_llm_extract_with_nbs_prompt must pass SYSTEM_PROMPT_NBS; got %r"
        % (captured.get("system_prompt"),)
    )
    # Concurrency invariant: the module-global must NOT have been mutated.
    assert v2.SYSTEM_PROMPT_V2 == original_global, (
        "SYSTEM_PROMPT_V2 was mutated — monkey patch regression. "
        "PE-3 must use the per-call kwarg, not a module-global swap."
    )
    # And the NBS prompt must NOT have leaked into the V2 global.
    assert v2.SYSTEM_PROMPT_V2 != epe.SYSTEM_PROMPT_NBS


# ─────────────────────────────────────────────────────────────────────
# Test 6 — LLM-failure loudness: n_failed reflects reality.
# ─────────────────────────────────────────────────────────────────────
def test_nbs_llm_call_failure_is_loud(tmp_path: Path):
    target_date = "2026-06-03"
    input_root = tmp_path / "policy_texts" / "nbs"
    output_root = tmp_path / "policy_events" / "nbs"

    _write_input_jsonl(
        input_root / f"{target_date}.jsonl",
        [_input_row(url="http://e.com/a"), _input_row(url="http://e.com/b")],
    )

    def fake_fail(_content: str):
        raise RuntimeError("LLM API key absent")

    summary = epe.extract_nbs(
        target_date=target_date,
        input_dir=input_root, output_dir=output_root,
        eventstore_dir=tmp_path / "events",
        llm_extract_fn=fake_fail,
    )
    assert summary["n_input"] == 2
    assert summary["n_extracted"] == 0
    assert summary["n_failed"] == 2
    # The errors list records the stage AND each URL so an operator can
    # find the offending rows without trawling the LLM logs.
    stages = {e["stage"] for e in summary["errors"]}
    assert "llm_call" in stages, summary["errors"]
