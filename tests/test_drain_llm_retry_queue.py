"""Unit tests for scripts/drain_llm_retry_queue.py.

Per cx code review 2026-05-31 [P3]: the drain script's core behavior —
rewrite the still-failed queue, append recovered events, run idempotent
closeout — was previously only verified on the empty-queue path. This
file covers the four substantive scenarios:

  1. All-success: every queue item extracts → queue deleted, events
     appended, no still-failed, exit code 0.
  2. Partial-failure: some items still fail → queue rewritten with
     only still-failed entries, exit code 2 (via main()).
  3. Queue self-duplicate: same item appears twice in the queue →
     extractor called only once; duplicate counter increments.
  4. Pre-existing-events dedup: a queue item whose key matches an
     already-written V2 jsonl record is skipped without retry.

Plus:

  5. Closeout-only: queue file absent but V2 jsonl exists → drain
     still runs EventStore sync + factor rebuild (idempotent
     half-completion recovery).

Strategy: monkeypatch RETRY_QUEUE_DIR / EVENTS_DIR to tmp_path,
stub LLMEventExtractorV2 with a scripted side_effect, stub the
closeout functions to assertable no-ops. No real LLM / EventStore /
parquet I/O touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(code: str, title: str, publish_time: str = "2026-05-30T10:00:00") -> dict:
    """Build a minimal retry-queue item shape."""
    return {
        "stock_code": code,
        "stock_name": f"name_{code}",
        "qlib_code": f"SH{code}" if code.startswith("6") else f"SZ{code}",
        "title": title,
        "content": f"content for {code} {title}",
        "source": "media_test",
        "publish_time": publish_time,
    }


def _write_queue(queue_dir: Path, target_date: str, items: list[dict]) -> Path:
    path = queue_dir / f"{target_date}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


def _read_events_path(events_dir: Path, target_date: str) -> list[dict]:
    path = events_dir / f"{target_date}.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _read_queue(queue_dir: Path, target_date: str) -> list[dict]:
    path = queue_dir / f"{target_date}.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Autouse fixture: redirect storage dirs + stub closeout
# ---------------------------------------------------------------------------


@pytest.fixture
def drain_module(tmp_path, monkeypatch):
    """Import drain module fresh with tmp dirs + stubbed closeout."""
    # Redirect storage paths BEFORE the module is (re)imported so the
    # module-level dir constants in factors.llm_event_extractor_v2 are
    # the only ones in play.
    queue_dir = tmp_path / "llm_retry_queue"
    events_dir = tmp_path / "llm_events_v2"
    queue_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)

    # Import here so monkeypatches below apply cleanly.
    import importlib
    import factors.llm_event_extractor_v2 as ext_mod
    import scripts.drain_llm_retry_queue as drain_mod

    # Patch directories
    monkeypatch.setattr(ext_mod, "RETRY_QUEUE_DIR", queue_dir)
    monkeypatch.setattr(ext_mod, "EVENTS_DIR", events_dir)
    monkeypatch.setattr(drain_mod, "RETRY_QUEUE_DIR", queue_dir)
    monkeypatch.setattr(drain_mod, "EVENTS_DIR", events_dir)

    # Stub closeout — record calls without touching real EventStore /
    # parquet builders. Tests can inspect these to assert closeout
    # actually ran (or didn't).
    sync_calls: list[tuple[Path, str]] = []
    rebuild_calls: list[str] = []
    monkeypatch.setattr(
        drain_mod, "_sync_to_eventstore",
        lambda events_path, target_date: sync_calls.append((events_path, target_date)),
    )
    monkeypatch.setattr(
        drain_mod, "_rebuild_factors",
        lambda target_date: rebuild_calls.append(target_date),
    )

    # Attach for test inspection
    drain_mod._test_queue_dir = queue_dir
    drain_mod._test_events_dir = events_dir
    drain_mod._test_sync_calls = sync_calls
    drain_mod._test_rebuild_calls = rebuild_calls
    return drain_mod


def _stub_extractor(monkeypatch, drain_mod, side_effect_extracts: list):
    """Replace LLMEventExtractorV2 with a stub that returns the given
    list of extract_single() results in order. None means 'still failed';
    a dict means 'extracted successfully'."""
    mock_instance = MagicMock()
    mock_instance.extract_single.side_effect = side_effect_extracts
    mock_instance._stats = {
        "calls": len(side_effect_extracts),
        "http_fail": sum(1 for v in side_effect_extracts if v is None),
        "rate_limited": 0,
        "parse_fail": 0,
    }
    mock_cls = MagicMock(return_value=mock_instance)
    monkeypatch.setattr(drain_mod, "LLMEventExtractorV2", mock_cls)
    return mock_instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_success_clears_queue_and_appends_events(drain_module, monkeypatch):
    """Scenario 1: every queue item extracts successfully → queue file
    deleted, V2 jsonl gains one record per item, still_failed=0."""
    target_date = "2026-05-30"
    items = [
        _make_item("600519", "Headline A"),
        _make_item("000001", "Headline B"),
        _make_item("300750", "Headline C"),
    ]
    _write_queue(drain_module._test_queue_dir, target_date, items)

    _stub_extractor(monkeypatch, drain_module, side_effect_extracts=[
        {"event_type": "positive", "severity": 0.5},
        {"event_type": "positive", "severity": 0.6},
        {"event_type": "negative", "severity": 0.4},
    ])

    result = drain_module.drain(target_date)

    assert result["items"] == 3
    assert result["recovered"] == 3
    assert result["still_failed"] == 0
    assert result["duplicates"] == 0
    assert result["closeout_ran"] is True

    # Queue file must be gone (fully drained)
    assert not (drain_module._test_queue_dir / f"{target_date}.jsonl").exists()
    # Events file has 3 records, each from v2_retry extractor
    events = _read_events_path(drain_module._test_events_dir, target_date)
    assert len(events) == 3
    assert all(e["extractor_version"] == "v2_retry" for e in events)
    # Closeout invoked
    assert len(drain_module._test_sync_calls) == 1
    assert drain_module._test_rebuild_calls == [target_date]


def test_partial_failure_rewrites_queue_and_returns_still_failed(drain_module, monkeypatch):
    """Scenario 2: 2 of 3 items recover, 1 still fails → queue rewritten
    to contain only the still-failed item, recovered=2, still_failed=1."""
    target_date = "2026-05-31"
    items = [
        _make_item("600519", "Headline A"),
        _make_item("000001", "Headline B (still failing)"),
        _make_item("300750", "Headline C"),
    ]
    _write_queue(drain_module._test_queue_dir, target_date, items)

    _stub_extractor(monkeypatch, drain_module, side_effect_extracts=[
        {"event_type": "positive", "severity": 0.5},
        None,  # still fails
        {"event_type": "negative", "severity": 0.4},
    ])

    result = drain_module.drain(target_date)

    assert result["items"] == 3
    assert result["recovered"] == 2
    assert result["still_failed"] == 1
    assert result["duplicates"] == 0
    assert result["closeout_ran"] is True

    # Queue rewritten to contain only the failed item
    remaining = _read_queue(drain_module._test_queue_dir, target_date)
    assert len(remaining) == 1
    assert remaining[0]["stock_code"] == "000001"
    assert remaining[0]["title"] == "Headline B (still failing)"

    # Events file has 2 records (the successes)
    events = _read_events_path(drain_module._test_events_dir, target_date)
    assert len(events) == 2
    assert {e["stock_code"] for e in events} == {"600519", "300750"}


def test_main_returns_exit_code_2_on_partial_failure(drain_module, monkeypatch):
    """Scenario 2 wrapper: main() returns 2 when still_failed > 0 so the
    cron job is marked failed in the daily health dashboard."""
    target_date = "2026-06-01"
    items = [_make_item("600519", "Headline still failing")]
    _write_queue(drain_module._test_queue_dir, target_date, items)
    _stub_extractor(monkeypatch, drain_module, side_effect_extracts=[None])

    monkeypatch.setattr("sys.argv", ["drain_llm_retry_queue.py", "--date", target_date])
    exit_code = drain_module.main()
    assert exit_code == 2

    # And exit 0 path for control: rerun with a successful extract
    _write_queue(drain_module._test_queue_dir, target_date, items)
    _stub_extractor(
        monkeypatch, drain_module,
        side_effect_extracts=[{"event_type": "neutral", "severity": 0.1}],
    )
    exit_code = drain_module.main()
    assert exit_code == 0


def test_queue_self_duplicate_processed_once(drain_module, monkeypatch):
    """Scenario 3: the same item appears twice in the queue → extractor
    called only once, duplicates counter == 1."""
    target_date = "2026-06-02"
    item = _make_item("600519", "Headline X", publish_time="2026-06-02T09:30:00")
    _write_queue(drain_module._test_queue_dir, target_date, [item, item])

    extractor = _stub_extractor(
        monkeypatch, drain_module,
        side_effect_extracts=[{"event_type": "positive", "severity": 0.5}],
    )

    result = drain_module.drain(target_date)

    assert result["items"] == 2  # raw queue had 2 lines
    assert result["recovered"] == 1  # but only 1 unique
    assert result["still_failed"] == 0
    assert result["duplicates"] == 1
    # extractor.extract_single called exactly once
    assert extractor.extract_single.call_count == 1


def test_skips_items_matching_existing_events(drain_module, monkeypatch):
    """Scenario 4: a queue item whose dedup key matches an already-
    written V2 jsonl record is skipped (no retry call)."""
    target_date = "2026-06-03"
    existing_item = _make_item("600519", "Already-extracted headline")
    new_item = _make_item("000001", "Genuinely new headline")
    # Write existing event to V2 jsonl using same dedup-key shape as
    # the queue item — drain._dedup_key reads stock_code/title/publish_time
    # so the same fields suffice.
    events_path = drain_module._test_events_dir / f"{target_date}.jsonl"
    with open(events_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(existing_item, ensure_ascii=False) + "\n")

    _write_queue(drain_module._test_queue_dir, target_date, [existing_item, new_item])

    extractor = _stub_extractor(
        monkeypatch, drain_module,
        side_effect_extracts=[{"event_type": "positive", "severity": 0.5}],
    )

    result = drain_module.drain(target_date)

    assert result["items"] == 2
    assert result["recovered"] == 1     # only new_item extracted
    assert result["still_failed"] == 0
    assert result["duplicates"] == 1    # existing_item skipped
    assert extractor.extract_single.call_count == 1

    # The events file now has the original existing_item + 1 new
    events = _read_events_path(drain_module._test_events_dir, target_date)
    assert len(events) == 2
    codes = [e["stock_code"] for e in events]
    assert "600519" in codes  # the existing one
    assert "000001" in codes  # the recovered new one


def test_closeout_only_when_queue_absent_but_events_exist(drain_module):
    """Scenario 5: queue file doesn't exist (no retry items) but V2 jsonl
    does → drain still runs EventStore sync + factor rebuild. Catches the
    half-completed-prior-run case where events were appended but closeout
    crashed before sync."""
    target_date = "2026-06-04"
    events_path = drain_module._test_events_dir / f"{target_date}.jsonl"
    with open(events_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_make_item("600519", "Some prior event")) + "\n")

    # No queue file written
    result = drain_module.drain(target_date)

    assert result["items"] == 0
    assert result["recovered"] == 0
    assert result["still_failed"] == 0
    assert result["duplicates"] == 0
    assert result["closeout_ran"] is True   # the key assertion

    # Closeout invoked
    assert len(drain_module._test_sync_calls) == 1
    assert drain_module._test_rebuild_calls == [target_date]


def test_empty_queue_and_no_events_skips_closeout(drain_module):
    """Negative control: queue absent AND events absent → closeout
    skipped (nothing to sync)."""
    target_date = "2026-06-05"

    result = drain_module.drain(target_date)

    assert result["items"] == 0
    assert result["recovered"] == 0
    assert result["closeout_ran"] is False
    # No closeout calls
    assert drain_module._test_sync_calls == []
    assert drain_module._test_rebuild_calls == []
