"""Phase C.5 (L5): daily LLM event-factor quality report.

Reads the day's extracted events from
``data/storage/llm_events_v2/<YYYY-MM-DD>.jsonl`` and writes a
machine-readable quality report to
``data/storage/llm_factor_quality/<YYYY-MM-DD>.json``.

The report tracks the metrics the project lead's 2026-06-06 LLM critique
called out as missing: events_count / stock_coverage /
event_type_distribution / direction_distribution / repeated_ratio /
generic_drop_count / top_duplicate_titles / source_distribution /
PIT_invalid_count, plus the downgrade-rate counter introduced by the
L3 schema validator.

Usage::

    # default = today
    python scripts/llm_factor_quality_report.py
    # explicit date
    python scripts/llm_factor_quality_report.py --date 2026-06-05
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _load_events_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _read_prefilter_stats(target_date: str) -> dict:
    """Look up generic_drop_count + dedup_drop_count from the L0/L1
    prefilter stats file that ``run_llm_event_pipeline`` writes.

    Returns ``{}`` if the file is absent or unreadable — the rest of
    the report still works, the quality JSON just records this with
    null / 0 instead of dropping the whole row.
    """
    candidates = [
        DATA_DIR / "llm_prefilter_stats" / f"{target_date}.json",
        DATA_DIR / "llm_prefilter_stats.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                logger.warning("prefilter stats unreadable %s: %s", path, e)
    return {}


def _publish_time_valid(row: dict, target_date: str) -> bool:
    """A row's publish_time should be <= the target date. Reject anything
    that claims to have been published in the future (file-date drift /
    timezone bugs)."""
    pt = (row.get("publish_time") or "")[:10]
    if not pt:
        return False
    try:
        # str ordering works on ISO YYYY-MM-DD
        return pt <= target_date
    except Exception:
        return False


def _top_n(counter: Counter, n: int = 10) -> list[tuple[str, int]]:
    return counter.most_common(n)


def build_report(events: list[dict], target_date: str,
                  prefilter_stats: dict | None = None) -> dict:
    prefilter_stats = prefilter_stats or {}

    by_type = Counter(r.get("event_type", "?") for r in events)
    by_direction = Counter(int(r.get("direction", 0)) for r in events)
    by_source = Counter(r.get("source", "?") for r in events)
    by_title = Counter((r.get("title") or "").strip() for r in events)

    n_repeated = sum(1 for r in events if r.get("is_repeated_news"))
    n_price_sensitive = sum(1 for r in events if r.get("is_price_sensitive"))
    n_official = sum(1 for r in events if r.get("is_official_disclosure"))
    n_with_downgrade = sum(1 for r in events if r.get("event_type_original"))
    n_pit_invalid = sum(1 for r in events if not _publish_time_valid(r, target_date))

    stocks = {r.get("qlib_code") or r.get("stock_code") for r in events}
    stocks.discard(None)

    return {
        "target_date": target_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "events_count": len(events),
        "stock_coverage": len(stocks),
        "event_type_distribution": dict(by_type),
        "direction_distribution": {str(k): v for k, v in by_direction.items()},
        "source_distribution": dict(_top_n(by_source, 15)),
        "repeated_ratio": (n_repeated / len(events)) if events else 0.0,
        "price_sensitive_count": n_price_sensitive,
        "official_disclosure_count": n_official,
        "schema_downgrade_count": n_with_downgrade,
        "schema_downgrade_ratio": (n_with_downgrade / len(events)) if events else 0.0,
        "pit_invalid_count": n_pit_invalid,
        "pit_invalid_ratio": (n_pit_invalid / len(events)) if events else 0.0,
        "top_duplicate_titles": [
            {"title": t, "count": c} for t, c in _top_n(by_title, 10)
            if c >= 2
        ],
        # L0/L1 prefilter visibility — sourced from the pipeline's own
        # stats sink rather than recomputed (we do not have the raw
        # pre-LLM input here).
        "prefilter_generic_drop_count": prefilter_stats.get("generic_drop_count"),
        "prefilter_dedup_drop_count": prefilter_stats.get("dedup_drop_count"),
        "prefilter_l0_kept": prefilter_stats.get("l0_kept"),
        "prefilter_l1_kept": prefilter_stats.get("l1_kept"),
    }


def write_report(report: dict, target_date: str) -> Path:
    out_dir = DATA_DIR / "llm_factor_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{target_date}.json"
    # Atomic write via .tmp + replace so a parallel reader cannot see
    # a half-written file.
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    os.replace(tmp, out_path)
    logger.info("Quality report written: %s (events=%d, stocks=%d)",
                out_path, report["events_count"], report["stock_coverage"])
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--date", default=datetime.now().strftime("%Y-%m-%d"),
        help="Target date YYYY-MM-DD (default: today).",
    )
    p.add_argument(
        "--events-path", default=None,
        help="Override JSONL path. Default: data/storage/llm_events_v2/<date>.jsonl.",
    )
    p.add_argument(
        "--print-summary", action="store_true",
        help="Print a human-readable summary alongside the JSON file.",
    )
    args = p.parse_args()

    events_path = (
        Path(args.events_path) if args.events_path
        else DATA_DIR / "llm_events_v2" / f"{args.date}.jsonl"
    )
    # 2026-06-06 cx review (P1): the previous code logged a warning and
    # wrote a "0 events" report on success when the file was absent.
    # That painted the gate green while the upstream pipeline had
    # produced nothing. Fail loud now so the cron wrapper surfaces it.
    if not events_path.exists():
        logger.error(
            "Events file missing: %s. Refusing to write a 0-events "
            "report — that would look like a clean run while the "
            "llm_event_pipeline upstream is still in flight or has "
            "failed silently. cx review 2026-06-06 (P1).",
            events_path,
        )
        sys.exit(1)

    events = _load_events_jsonl(events_path)
    prefilter_stats = _read_prefilter_stats(args.date)
    report = build_report(events, args.date, prefilter_stats=prefilter_stats)
    write_report(report, args.date)
    # Also fail when the file exists but is empty — same blast radius
    # as a missing file: the cron wrapper must mark the day red.
    if report["events_count"] == 0:
        logger.error(
            "Events file %s exists but contains 0 events. Reporting "
            "this as failure so the gate does not paint a green flag "
            "over an empty pipeline.",
            events_path,
        )
        sys.exit(1)

    if args.print_summary:
        print(json.dumps({
            "events_count": report["events_count"],
            "stock_coverage": report["stock_coverage"],
            "repeated_ratio": round(report["repeated_ratio"], 3),
            "schema_downgrade_ratio": round(report["schema_downgrade_ratio"], 3),
            "pit_invalid_count": report["pit_invalid_count"],
            "top_event_types": _top_n(
                Counter(report["event_type_distribution"]), 5,
            ),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
