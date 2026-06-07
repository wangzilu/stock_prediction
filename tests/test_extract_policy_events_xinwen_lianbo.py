"""Tests for scripts/extract_policy_events.py --source xinwen_lianbo.

Phase E.4 (PE-4) step 2 — mirror of test_extract_policy_events_nbs.py
for the XWLB theme-attention branch. Covers:

  1. Free-form theme vocab — themes outside the documented set are
     kept (vocabulary evolves), counted via ``n_unknown_themes``.
  2. Field type coercion — themes lowercased & underscored, counts
     coerced to int, priority clamped to [0,1].
  3. LLM end-to-end with mocked LLM, output JSONL schema is the
     documented surface.
  4. Idempotent re-run: same date, two runs, no dupes, no .tmp.
  5. SYSTEM_PROMPT_XINWEN_LIANBO threaded as a per-call kwarg to
     _call_llm. NO monkey patch of v2.SYSTEM_PROMPT_V2.
  6. LLM-failure loudness: LLM raising on every input ⇒
     n_failed == n_input.

All LLM calls are mocked. We never hit MiniMax / DeepSeek in tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts import extract_policy_events as epe


# ─────────────────────────────────────────────────────────────────────
# Sample input row matching collect_policy_texts.py (PE-4) contract.
# ─────────────────────────────────────────────────────────────────────
def _input_row(
    *,
    publish_date: str = "2026-06-05",
    policy_type: str = "xinwen_lianbo_daily",
    title: str = "新闻联播 2026年6月5日 完整版",
    url: str = "https://news.sina.com.cn/zt_d/xwlb/2026-06-05/content_1001.html",
    content: str = (
        "习近平主持中央政治局会议，部署半导体自立自强工作。"
        "国务院常务会议研究扩大内需若干举措。"
        "工信部公布机器人产业发展规划。"
        "一带一路高峰论坛在京举行。"
    ),
) -> dict:
    return {
        "publish_date": publish_date,
        "policy_type": policy_type,
        "title": title,
        "url": url,
        "content": content,
        "source": "news.sina.com.cn",
        "fetch_time": "2026-06-05T15:45:00Z",
    }


def _write_input_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────
# Test 1 — Free-form theme vocab: unknown themes are KEPT, counted as
# unknown for operator visibility; valid known themes pass through.
# ─────────────────────────────────────────────────────────────────────
def test_xwlb_free_form_theme_vocab_keeps_unknown_themes():
    """LLM emits a mix of canonical and novel themes — validator must
    keep them all (lowercased / underscored) since the vocab is
    intentionally open."""
    raw = {
        "themes": [
            "semiconductor_self_reliance",        # known
            "Robotics AI",                         # known after normalize
            "quantum-supremacy",                   # novel — hyphen
            "deep_ocean_mining",                   # novel
        ],
        "theme_mention_counts": {
            "semiconductor_self_reliance": 3,
            "robotics_ai": 1,
            "quantum_supremacy": 1,
            "deep_ocean_mining": 1,
        },
        "policy_priority_signal": 0.7,
        "regions_mentioned": ["Shanghai", "广东"],
    }
    out = epe.validate_event_xinwen_lianbo(raw)
    assert "semiconductor_self_reliance" in out["themes"]
    assert "robotics_ai" in out["themes"]
    assert "quantum_supremacy" in out["themes"]
    assert "deep_ocean_mining" in out["themes"]
    # No dropping.
    assert len(out["themes"]) == 4
    # Counts coerced (and mapped onto the normalized theme tokens).
    assert out["theme_mention_counts"]["semiconductor_self_reliance"] == 3
    assert out["theme_mention_counts"]["robotics_ai"] == 1
    # Regions lowercased.
    assert "shanghai" in out["regions_mentioned"]
    assert "广东" in out["regions_mentioned"]
    # Priority preserved.
    assert out["policy_priority_signal"] == 0.7


# ─────────────────────────────────────────────────────────────────────
# Test 2 — Field type coercion.
# ─────────────────────────────────────────────────────────────────────
def test_xwlb_field_type_coercion():
    """Numerics coerce; priority clamps to [0,1]; missing → defaults."""
    # Out-of-range priority clamps; counts coerce from strings; themes
    # accept comma-joined fallback when LLM hands us a string.
    raw = {
        "themes": "real_estate, capital_markets",
        "theme_mention_counts": {"real_estate": "2", "capital_markets": "1"},
        "policy_priority_signal": 1.5,           # clamp to 1.0
        "regions_mentioned": "Beijing",
    }
    out = epe.validate_event_xinwen_lianbo(raw)
    assert out["themes"] == ["real_estate", "capital_markets"]
    assert out["theme_mention_counts"]["real_estate"] == 2
    assert out["theme_mention_counts"]["capital_markets"] == 1
    assert out["policy_priority_signal"] == 1.0
    assert out["regions_mentioned"] == ["beijing"]

    # Missing-everything row — sensible defaults, no crash.
    out2 = epe.validate_event_xinwen_lianbo({})
    assert out2["themes"] == []
    assert out2["theme_mention_counts"] == {}
    assert out2["policy_priority_signal"] == 0.0
    assert out2["regions_mentioned"] == []

    # Negative priority clamps to 0.
    out3 = epe.validate_event_xinwen_lianbo(
        {"themes": ["x"], "policy_priority_signal": -0.3}
    )
    assert out3["policy_priority_signal"] == 0.0

    # Count for theme not in `themes` list is DROPPED (consistency).
    out4 = epe.validate_event_xinwen_lianbo({
        "themes": ["a"],
        "theme_mention_counts": {"a": 4, "ghost": 5},
    })
    assert out4["theme_mention_counts"] == {"a": 4}, out4


# ─────────────────────────────────────────────────────────────────────
# Test 3 — LLM end-to-end mocked.
# ─────────────────────────────────────────────────────────────────────
def test_xwlb_llm_extract_produces_correct_jsonl_schema(tmp_path: Path):
    target_date = "2026-06-05"
    input_root = tmp_path / "policy_texts" / "xinwen_lianbo"
    output_root = tmp_path / "policy_events" / "xinwen_lianbo"
    eventstore_dir = tmp_path / "events"

    _write_input_jsonl(input_root / f"{target_date}.jsonl", [_input_row()])

    canned = {
        "themes": [
            "semiconductor_self_reliance",
            "domestic_consumption",
            "robotics_ai",
            "belt_and_road",
        ],
        "theme_mention_counts": {
            "semiconductor_self_reliance": 1,
            "domestic_consumption": 1,
            "robotics_ai": 1,
            "belt_and_road": 1,
        },
        "policy_priority_signal": 0.85,
        "regions_mentioned": [],
    }

    def fake_llm(_content: str) -> dict[str, Any] | None:
        return dict(canned)

    summary = epe.extract_xinwen_lianbo(
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
        "themes", "theme_mention_counts",
        "policy_priority_signal", "regions_mentioned",
        "extracted_at",
    }
    assert set(row.keys()) == expected_keys, sorted(row.keys())
    assert row["policy_type"] == "xinwen_lianbo_daily"
    assert "semiconductor_self_reliance" in row["themes"]
    assert row["theme_mention_counts"]["belt_and_road"] == 1
    assert row["policy_priority_signal"] == 0.85
    assert row["extracted_at"].endswith("Z")
    assert summary["n_extracted"] == 1
    assert summary["n_input"] == 1
    assert summary["n_failed"] == 0


# ─────────────────────────────────────────────────────────────────────
# Test 4 — Idempotent re-run.
# ─────────────────────────────────────────────────────────────────────
def test_xwlb_idempotent_rerun(tmp_path: Path):
    target_date = "2026-06-05"
    input_root = tmp_path / "policy_texts" / "xinwen_lianbo"
    output_root = tmp_path / "policy_events" / "xinwen_lianbo"

    _write_input_jsonl(input_root / f"{target_date}.jsonl", [_input_row()])

    def fake_llm(_content: str):
        return {
            "themes": ["semiconductor_self_reliance"],
            "theme_mention_counts": {"semiconductor_self_reliance": 2},
            "policy_priority_signal": 0.6,
            "regions_mentioned": [],
        }

    epe.extract_xinwen_lianbo(
        target_date=target_date,
        input_dir=input_root, output_dir=output_root,
        eventstore_dir=tmp_path / "events",
        llm_extract_fn=fake_llm,
    )
    epe.extract_xinwen_lianbo(
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
# Test 5 — SYSTEM_PROMPT_XINWEN_LIANBO threaded through _call_llm as a
# kwarg. MUST NOT mutate factors.llm_event_extractor_v2.SYSTEM_PROMPT_V2.
# ─────────────────────────────────────────────────────────────────────
def test_xwlb_system_prompt_threaded_through_call_llm(monkeypatch):
    """``_llm_extract_with_xinwen_lianbo_prompt`` must pass
    SYSTEM_PROMPT_XINWEN_LIANBO to LLMEventExtractorV2._call_llm via the
    per-call kwarg. The module-global SYSTEM_PROMPT_V2 must NOT be
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
            '{"themes": ["real_estate"], '
            '"theme_mention_counts": {"real_estate": 1}, '
            '"policy_priority_signal": 0.4, '
            '"regions_mentioned": []}',
            {"http_ok": True},
        )

    monkeypatch.setattr(v2.LLMEventExtractorV2, "_call_llm", fake_call_llm)
    monkeypatch.setenv("MINIMAX_API_KEY", "test")

    out = epe._llm_extract_with_xinwen_lianbo_prompt("test XWLB transcript body")
    assert out is not None
    assert captured["system_prompt"] is epe.SYSTEM_PROMPT_XINWEN_LIANBO, (
        "_llm_extract_with_xinwen_lianbo_prompt must pass "
        "SYSTEM_PROMPT_XINWEN_LIANBO; got %r" % (captured.get("system_prompt"),)
    )
    # Concurrency invariant: module-global must NOT have been mutated.
    assert v2.SYSTEM_PROMPT_V2 == original_global, (
        "SYSTEM_PROMPT_V2 was mutated — monkey patch regression. "
        "PE-4 must use the per-call kwarg, not a module-global swap."
    )
    assert v2.SYSTEM_PROMPT_V2 != epe.SYSTEM_PROMPT_XINWEN_LIANBO

    # And the PE-4 prompt itself must EXPLICITLY ban price prediction —
    # this is the L1 "facts not predictions" guard.
    assert "stock direction" in epe.SYSTEM_PROMPT_XINWEN_LIANBO.lower() or \
           "predict" in epe.SYSTEM_PROMPT_XINWEN_LIANBO.lower(), (
        "SYSTEM_PROMPT_XINWEN_LIANBO must explicitly forbid stock "
        "direction prediction — theme strength is a FACT about media "
        "attention, NOT a price forecast."
    )


# ─────────────────────────────────────────────────────────────────────
# Test 6 — LLM-failure loudness.
# ─────────────────────────────────────────────────────────────────────
def test_xwlb_llm_call_failure_is_loud(tmp_path: Path):
    target_date = "2026-06-05"
    input_root = tmp_path / "policy_texts" / "xinwen_lianbo"
    output_root = tmp_path / "policy_events" / "xinwen_lianbo"

    _write_input_jsonl(
        input_root / f"{target_date}.jsonl",
        [
            _input_row(url="http://e.com/a"),
            _input_row(url="http://e.com/b"),
        ],
    )

    def fake_fail(_content: str):
        raise RuntimeError("LLM API key absent")

    summary = epe.extract_xinwen_lianbo(
        target_date=target_date,
        input_dir=input_root, output_dir=output_root,
        eventstore_dir=tmp_path / "events",
        llm_extract_fn=fake_fail,
    )
    assert summary["n_input"] == 2
    assert summary["n_extracted"] == 0
    assert summary["n_failed"] == 2
    stages = {e["stage"] for e in summary["errors"]}
    assert "llm_call" in stages, summary["errors"]


# ─────────────────────────────────────────────────────────────────────
# Test 7 — Unknown theme counter ticks for vocabulary drift.
# ─────────────────────────────────────────────────────────────────────
def test_xwlb_unknown_theme_counter_ticks(tmp_path: Path):
    """Themes outside XINWEN_LIANBO_KNOWN_THEMES are KEPT but counted in
    ``n_unknown_themes`` so an operator can spot vocab drift."""
    target_date = "2026-06-05"
    input_root = tmp_path / "policy_texts" / "xinwen_lianbo"
    output_root = tmp_path / "policy_events" / "xinwen_lianbo"

    _write_input_jsonl(input_root / f"{target_date}.jsonl", [_input_row()])

    def fake_llm(_content: str):
        return {
            "themes": [
                "real_estate",               # known
                "fusion_power",              # unknown
                "smart_grid_2_0",            # unknown
            ],
            "theme_mention_counts": {
                "real_estate": 1,
                "fusion_power": 1,
                "smart_grid_2_0": 1,
            },
            "policy_priority_signal": 0.5,
            "regions_mentioned": [],
        }

    summary = epe.extract_xinwen_lianbo(
        target_date=target_date,
        input_dir=input_root, output_dir=output_root,
        eventstore_dir=tmp_path / "events",
        llm_extract_fn=fake_llm,
    )
    assert summary["n_extracted"] == 1
    # Two unknowns: fusion_power, smart_grid_2_0.
    assert summary["n_unknown_themes"] == 2, summary
