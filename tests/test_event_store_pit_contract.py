"""PE-5 (task #144) — strict PIT 4-time contract on the EventStore.

Pins:
  (a) ``EventStore.add_event`` REJECTS a write that's missing any of
      the four contract fields ``publish_time`` / ``available_time`` /
      ``signal_date`` / ``execution_date`` (or violates the date
      ordering).
  (b) A well-formed event — either fully pre-populated by the caller,
      OR with only ``publish_time`` supplied and the rest derived by
      EventStore — is accepted and persisted with all 4 fields on disk.
  (c) ``scripts.migrate_eventstore_pit_times`` derives the missing
      fields correctly on the legacy PE-1 (policy) and LLM-event row
      shapes, leaves already-compliant rows untouched, and skips rows
      that genuinely have no ``publish_time``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# Make ``factors`` and ``scripts`` importable without requiring an
# editable install. Mirrors the convention used by the surrounding
# tests in this directory.
@pytest.fixture(autouse=True)
def _add_project_root_to_path():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    yield


# ─────────────────────────────────────────────────────────────────────
# (a) write rejects missing fields
# ─────────────────────────────────────────────────────────────────────

def _base_well_formed_event(**overrides) -> dict:
    """A minimal contract-compliant event for the write-path tests."""
    ev = {
        "date": "2026-06-05",
        "stock_code": "000001",
        "source": "news",
        "event_type": "other",
        "direction": 0,
        "confidence": 0.5,
        "summary": "test event",
        "publish_time": "2026-06-05 14:00:00",
        "available_time": "2026-06-05 14:00:00",
        "signal_date": "2026-06-08",   # 2026-06-05 is Fri; next BDay = Mon
        "execution_date": "2026-06-08",  # pre-close, so same as signal_date
    }
    ev.update(overrides)
    return ev


def test_write_rejects_when_publish_time_missing(tmp_path: Path) -> None:
    """An event without ``publish_time`` cannot satisfy the PIT
    contract — derivation has no input, so the write must reject."""
    from factors.event_store import EventStore, PITContractError

    store = EventStore(store_dir=tmp_path)
    ev = _base_well_formed_event()
    del ev["publish_time"]
    del ev["available_time"]
    del ev["signal_date"]
    del ev["execution_date"]

    with pytest.raises(PITContractError) as exc:
        store.add_event(ev)
    # The error message must enumerate the offending field(s) so an
    # operator can diagnose the upstream bug.
    assert "publish_time" in str(exc.value)


def test_write_rejects_when_signal_date_before_available_time(
    tmp_path: Path,
) -> None:
    """Date-ordering check is part of the contract — flag the silent
    bug where someone hand-sets signal_date earlier than the day the
    data became consumable."""
    from factors.event_store import EventStore, PITContractError

    store = EventStore(store_dir=tmp_path)
    ev = _base_well_formed_event(
        publish_time="2026-06-05 14:00:00",
        available_time="2026-06-05 14:00:00",
        signal_date="2026-05-01",        # absurd: way before available_time
        execution_date="2026-05-01",
    )
    with pytest.raises(PITContractError):
        store.add_event(ev)


def test_write_rejects_when_execution_before_signal(tmp_path: Path) -> None:
    from factors.event_store import EventStore, PITContractError

    store = EventStore(store_dir=tmp_path)
    ev = _base_well_formed_event(
        signal_date="2026-06-08",
        execution_date="2026-06-05",  # before signal
    )
    with pytest.raises(PITContractError):
        store.add_event(ev)


def test_strict_pit_false_opt_out(tmp_path: Path) -> None:
    """The migration helper / legacy tests need an escape hatch.
    ``EventStore(strict_pit=False)`` must accept pre-contract rows."""
    from factors.event_store import EventStore

    store = EventStore(store_dir=tmp_path, strict_pit=False)
    ev = _base_well_formed_event()
    del ev["available_time"]
    del ev["signal_date"]
    del ev["execution_date"]
    # _validate_event will still try to derive from publish_time, so
    # the write should succeed and a sensible default is persisted.
    assert store.add_event(ev) is True


# ─────────────────────────────────────────────────────────────────────
# (b) write accepts well-formed / derivable events
# ─────────────────────────────────────────────────────────────────────

def test_write_accepts_fully_specified_event(tmp_path: Path) -> None:
    from factors.event_store import EventStore

    store = EventStore(store_dir=tmp_path)
    ev = _base_well_formed_event()
    assert store.add_event(ev) is True

    fp = tmp_path / "2026-06-05.jsonl"
    assert fp.exists()
    with open(fp, encoding="utf-8") as f:
        row = json.loads(f.readline())
    for field in (
        "publish_time", "available_time", "signal_date", "execution_date",
    ):
        assert row.get(field), f"persisted row is missing {field}: {row}"


def test_write_derives_missing_fields_from_publish_time(
    tmp_path: Path,
) -> None:
    """Caller supplies only ``publish_time`` — EventStore must derive
    the other three. Friday 14:00 publish (pre-close) must yield
    signal_date=Monday and execution_date=Monday (same day, pre-close
    rule)."""
    from factors.event_store import EventStore

    store = EventStore(store_dir=tmp_path)
    ev = _base_well_formed_event()
    del ev["available_time"]
    del ev["signal_date"]
    del ev["execution_date"]

    assert store.add_event(ev) is True

    fp = tmp_path / "2026-06-05.jsonl"
    with open(fp, encoding="utf-8") as f:
        row = json.loads(f.readline())
    assert row["available_time"] == "2026-06-05 14:00:00"
    assert row["signal_date"] == "2026-06-08"
    assert row["execution_date"] == "2026-06-08"


def test_write_post_close_event_rolls_execution_to_next_bday(
    tmp_path: Path,
) -> None:
    """Friday 16:30 publish (post-close) → signal_date=Monday,
    execution_date=Tuesday (post-close rule adds +1 BDay)."""
    from factors.event_store import EventStore

    store = EventStore(store_dir=tmp_path)
    ev = _base_well_formed_event(publish_time="2026-06-05 16:30:00")
    del ev["available_time"]
    del ev["signal_date"]
    del ev["execution_date"]

    assert store.add_event(ev) is True

    fp = tmp_path / "2026-06-05.jsonl"
    with open(fp, encoding="utf-8") as f:
        row = json.loads(f.readline())
    assert row["available_time"] == "2026-06-05 16:30:00"
    assert row["signal_date"] == "2026-06-08"
    assert row["execution_date"] == "2026-06-09"


# ─────────────────────────────────────────────────────────────────────
# (c) migration helper derives correct fields on standard shapes
# ─────────────────────────────────────────────────────────────────────

def _write_jsonl(fp: Path, rows: list[dict]) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(fp: Path) -> list[dict]:
    rows: list[dict] = []
    with open(fp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_migration_backfills_pe1_policy_row(tmp_path: Path) -> None:
    """PE-1 (PBC) shape: ``publish_time`` is date-only, no PIT fields."""
    from scripts.migrate_eventstore_pit_times import migrate

    pe1_row = {
        "date": "2026-06-05",
        "stock_code": "MARKET",
        "source": "policy",
        "event_type": "policy_support",
        "direction": 1,
        "confidence": 0.6,
        "summary": "公开市场操作净投放700亿元",
        "publish_time": "2026-06-05",
        "topic": "omo",
        "is_policy": True,
        "_hash": "deadbeef",
    }
    _write_jsonl(tmp_path / "2026-06-05.jsonl", [pe1_row])

    agg = migrate(tmp_path)
    assert agg["total"] == 1
    assert agg["backfilled"] == 1
    assert agg["skipped"] == 0

    out = _read_jsonl(tmp_path / "2026-06-05.jsonl")[0]
    assert out["publish_time"] == "2026-06-05"
    # Midnight publish → available_time has the ``00:00:00`` suffix
    assert out["available_time"].startswith("2026-06-05")
    # 2026-06-05 is Friday → next BDay is Monday
    assert out["signal_date"] == "2026-06-08"
    # Midnight is pre-close, so execution_date == signal_date
    assert out["execution_date"] == "2026-06-08"

    # Backup must exist
    assert (tmp_path / "2026-06-05.jsonl.pre_pe5.bak").exists()


def test_migration_backfills_llm_event_row(tmp_path: Path) -> None:
    """LLM-event shape: ``publish_time`` has full timestamp; an
    ``extract_date`` exists in the row but is metadata-only and must
    NOT become the available_time anchor."""
    from scripts.migrate_eventstore_pit_times import migrate

    llm_row = {
        "date": "2026-04-16",
        "stock_code": "002124",
        "source": "news",
        "event_type": "earnings_negative",
        "direction": 1,
        "confidence": 0.75,
        "summary": "天邦食品预亏",
        "magnitude": 0.333,
        "publish_time": "2026-04-15 19:06:00",   # post-close
        "llm_model": "minimax",
        "prompt_version": "v1",
        "extract_date": "2026-05-22",  # batch re-extract, not PIT
        "source_quality": 0.8,
        "_hash": "feedface",
    }
    _write_jsonl(tmp_path / "2026-04-16.jsonl", [llm_row])

    agg = migrate(tmp_path)
    assert agg["backfilled"] == 1
    assert agg["skipped"] == 0

    out = _read_jsonl(tmp_path / "2026-04-16.jsonl")[0]
    # publish_time preserved
    assert out["publish_time"] == "2026-04-15 19:06:00"
    # available_time = publish_time (parser_lag=0), NOT extract_date
    assert out["available_time"] == "2026-04-15 19:06:00"
    # 2026-04-15 is Wednesday; post-close → signal_date = Thursday
    assert out["signal_date"] == "2026-04-16"
    # Post-close → execution_date = signal + 1 BDay = Friday
    assert out["execution_date"] == "2026-04-17"
    # extract_date metadata preserved untouched
    assert out["extract_date"] == "2026-05-22"


def test_migration_skips_row_without_publish_time(tmp_path: Path) -> None:
    """Pre-PIT-era rows that genuinely lack publish_time cannot be
    derived; they are logged + skipped and left intact."""
    from scripts.migrate_eventstore_pit_times import migrate

    bad_row = {
        "date": "2026-05-01",
        "stock_code": "600519",
        "source": "news",
        "event_type": "other",
        "direction": 0,
        "confidence": 0.5,
        "summary": "no publish_time at all",
        "_hash": "abc",
        # no publish_time
    }
    _write_jsonl(tmp_path / "2026-05-01.jsonl", [bad_row])

    agg = migrate(tmp_path)
    assert agg["total"] == 1
    assert agg["backfilled"] == 0
    assert agg["skipped"] == 1
    assert "no publish_time" in agg["skip_reasons"]

    # File contents are unchanged (no backup needed)
    out = _read_jsonl(tmp_path / "2026-05-01.jsonl")[0]
    assert out == bad_row


def test_migration_is_idempotent(tmp_path: Path) -> None:
    """Re-running on a migrated corpus is a no-op."""
    from scripts.migrate_eventstore_pit_times import migrate

    well_formed = {
        "date": "2026-06-05",
        "stock_code": "000001",
        "source": "news",
        "event_type": "other",
        "direction": 0,
        "confidence": 0.5,
        "summary": "test",
        "publish_time": "2026-06-05 14:00:00",
        "available_time": "2026-06-05 14:00:00",
        "signal_date": "2026-06-08",
        "execution_date": "2026-06-08",
    }
    _write_jsonl(tmp_path / "2026-06-05.jsonl", [well_formed])

    agg1 = migrate(tmp_path)
    assert agg1["already_ok"] == 1
    assert agg1["backfilled"] == 0

    # Second run: still all-OK, no backup created
    agg2 = migrate(tmp_path)
    assert agg2["already_ok"] == 1
    assert agg2["backfilled"] == 0
    assert not (tmp_path / "2026-06-05.jsonl.pre_pe5.bak").exists()
