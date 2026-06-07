"""PE-5 (task #144) — backfill PIT 4-time fields on existing EventStore rows.

The strict PIT contract added in ``factors.event_store`` rejects writes
that don't carry ``publish_time`` / ``available_time`` / ``signal_date``
/ ``execution_date``. Existing JSONL files in ``data/storage/events/``
were produced before the contract existed and so a large fraction of
rows lack the derived fields (most of them have a parseable
``publish_time`` and therefore can be backfilled deterministically).

This helper:

  1. Walks ``data/storage/events/*.jsonl`` (path overrideable).
  2. For each row, fills any missing PIT field by invoking
     ``_compute_pit_times(publish_time)`` from the canonical helper.
  3. Validates the resulting row against ``_validate_pit_times`` — rows
     that still fail (truly missing publish_time, unparseable, etc.) are
     LOGGED + SKIPPED and the original row is preserved in the rewrite.
  4. Backs up each file to ``<file>.pre_pe5.bak`` before rewriting in
     place atomically (write to ``.tmp`` then ``rename``).

Idempotent: re-running over already-migrated files is a no-op (every
row already has the 4 fields → validator passes → no change is needed).

Usage:
    python scripts/migrate_eventstore_pit_times.py
    python scripts/migrate_eventstore_pit_times.py --dry-run
    python scripts/migrate_eventstore_pit_times.py --store-dir /custom/path

Exit code is 0 on success (regardless of skip count); non-zero only if
the walk itself failed (no files found, etc).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable

# Allow running as ``python scripts/migrate_eventstore_pit_times.py``
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from factors.event_store import (  # noqa: E402
    PITContractError,
    STORE_DIR,
    _compute_pit_times,
    _validate_pit_times,
)

logger = logging.getLogger("pe5_migrate")

# Suffix used for backup files. Hard-coded so a partial run is recoverable.
BACKUP_SUFFIX = ".pre_pe5.bak"


def _backfill_row(row: dict) -> tuple[dict, str | None]:
    """Add / repair the 4 PIT fields on ``row``.

    Returns ``(new_row, skip_reason)``. ``skip_reason is None`` means we
    either had nothing to do (row was already contract-compliant) or we
    successfully derived the missing fields. Otherwise the original row
    is preserved and the reason logged.

    Repair (not just fill) is necessary because pre-PE5 rows often set
    ``available_time`` to the migration / re-extract date (which is
    AFTER ``publish_time``) and ``signal_date`` to a value derived from
    ``publish_time`` alone — that combination violates
    ``signal_date >= available_time``. We recompute the 3 derived
    fields from ``publish_time`` whenever the existing values fail
    validation, treating parser lag as 0 (the original scrape latency
    isn't recoverable, and a later ``extract_date`` is just the day we
    re-ran extraction, not a PIT-meaningful availability anchor — it is
    preserved separately as the ``extract_date`` metadata field).
    """
    # Already-compliant rows: validator passes, nothing to do.
    try:
        _validate_pit_times(row)
        return row, None
    except PITContractError:
        pass  # fall through to backfill / repair attempt

    publish_time = (row.get("publish_time") or "").strip()
    if not publish_time:
        # No publish_time → cannot derive available_time / signal_date /
        # execution_date. Skip per task spec: "For rows that can't be
        # derived (truly missing publish_time), log + skip."
        return row, "no publish_time"

    try:
        pit = _compute_pit_times(publish_time)
    except PITContractError as e:
        return row, f"compute_pit_times failed: {e}"

    new_row = dict(row)  # copy so we don't mutate caller's dict
    new_row["publish_time"] = publish_time
    new_row["available_time"] = pit["available_time"]
    new_row["signal_date"] = pit["signal_date"]
    new_row["execution_date"] = pit["execution_date"]
    # event_time: not part of the contract but the wider schema expects
    # it and other downstream consumers read it.
    if not new_row.get("event_time"):
        new_row["event_time"] = publish_time

    # Final validation — defends against weird ordering bugs
    try:
        _validate_pit_times(new_row)
    except PITContractError as e:
        return row, f"post-backfill validation failed: {e}"

    return new_row, None


def _migrate_file(
    fp: Path, *, dry_run: bool, source_breakdown: dict[str, list[int]],
) -> dict:
    """Migrate one JSONL file.

    Returns per-file stats: ``{"total", "already_ok", "backfilled",
    "skipped", "skip_reasons"}``. ``source_breakdown`` is mutated to
    accumulate counts by ``source`` field for the final report.
    """
    stats = {
        "file": fp.name,
        "total": 0,
        "already_ok": 0,
        "backfilled": 0,
        "skipped": 0,
        "skip_reasons": {},  # reason -> count
    }
    rewritten_rows: list[dict] = []
    any_change = False

    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("%s: skipping malformed JSON line (%s)", fp.name, e)
                continue
            stats["total"] += 1
            src = row.get("source", "unknown")
            source_breakdown.setdefault(src, [0, 0, 0])  # ok / backfilled / skipped

            # Detect already-compliant vs needs-backfill before _backfill_row
            already = False
            try:
                _validate_pit_times(row)
                already = True
            except PITContractError:
                already = False

            if already:
                stats["already_ok"] += 1
                source_breakdown[src][0] += 1
                rewritten_rows.append(row)
                continue

            new_row, skip_reason = _backfill_row(row)
            if skip_reason is None:
                stats["backfilled"] += 1
                source_breakdown[src][1] += 1
                rewritten_rows.append(new_row)
                if new_row is not row:
                    any_change = True
            else:
                stats["skipped"] += 1
                source_breakdown[src][2] += 1
                stats["skip_reasons"][skip_reason] = (
                    stats["skip_reasons"].get(skip_reason, 0) + 1
                )
                rewritten_rows.append(row)  # preserve original

    if dry_run:
        return stats

    if not any_change:
        # Nothing to do — leave the file alone, no backup needed.
        return stats

    # Back up the original file then rewrite atomically.
    backup_path = fp.with_suffix(fp.suffix + BACKUP_SUFFIX)
    if not backup_path.exists():
        shutil.copy2(fp, backup_path)
    tmp = fp.with_suffix(fp.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rewritten_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(fp)
    return stats


def migrate(
    store_dir: Path,
    *,
    dry_run: bool = False,
) -> dict:
    """Run the migration over every JSONL file in ``store_dir``.

    Returns aggregated stats. Logs a per-file line and a final summary.
    """
    files = sorted(store_dir.glob("*.jsonl"))
    if not files:
        logger.warning("No JSONL files found in %s", store_dir)
        return {"files": 0, "total": 0, "already_ok": 0, "backfilled": 0, "skipped": 0}

    agg = {
        "files": 0,
        "total": 0,
        "already_ok": 0,
        "backfilled": 0,
        "skipped": 0,
        "skip_reasons": {},
        "by_source": {},  # source -> [ok, backfilled, skipped]
    }
    for fp in files:
        # Don't migrate our own backups.
        if fp.name.endswith(BACKUP_SUFFIX):
            continue
        per = _migrate_file(fp, dry_run=dry_run, source_breakdown=agg["by_source"])
        agg["files"] += 1
        agg["total"] += per["total"]
        agg["already_ok"] += per["already_ok"]
        agg["backfilled"] += per["backfilled"]
        agg["skipped"] += per["skipped"]
        for reason, count in per["skip_reasons"].items():
            agg["skip_reasons"][reason] = agg["skip_reasons"].get(reason, 0) + count
        logger.info(
            "  %s: total=%d already=%d backfilled=%d skipped=%d",
            per["file"], per["total"], per["already_ok"],
            per["backfilled"], per["skipped"],
        )
    return agg


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--store-dir", type=Path, default=STORE_DIR,
        help=f"EventStore directory to migrate (default: {STORE_DIR}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change but write nothing.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(
        "PE-5 migrate: store_dir=%s dry_run=%s",
        args.store_dir, args.dry_run,
    )
    if not args.store_dir.exists():
        logger.error("Store dir does not exist: %s", args.store_dir)
        return 2

    agg = migrate(args.store_dir, dry_run=args.dry_run)

    logger.info("=" * 60)
    logger.info(
        "PE-5 migration summary: files=%d total=%d already_ok=%d "
        "backfilled=%d skipped=%d",
        agg["files"], agg["total"], agg["already_ok"],
        agg["backfilled"], agg["skipped"],
    )
    if agg["by_source"]:
        logger.info("Per-source [ok / backfilled / skipped]:")
        for src in sorted(agg["by_source"]):
            ok, bf, sk = agg["by_source"][src]
            logger.info("  %-15s  ok=%d  backfilled=%d  skipped=%d", src, ok, bf, sk)
    if agg["skip_reasons"]:
        logger.info("Skip reasons:")
        for reason, count in sorted(
            agg["skip_reasons"].items(), key=lambda kv: -kv[1]
        ):
            logger.info("  %-40s  %d", reason, count)
    if args.dry_run:
        logger.info("DRY RUN — no files were modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
