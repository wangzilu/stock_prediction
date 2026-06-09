"""SPIKE — Collect insider trading announcements (高管 / 股东 增减持公告).

LLM Channel 3 (insider trading) — separates insider 增减持 announcements
from the generic V2 event extractor so the signal is not diluted by
share_placement / share_unlock / management_change noise.

Design choice (2026-06-09): **filter from existing announcement pipeline**
rather than open a new collector. Rationale:

  1. ``scripts/collect_announcements.py`` already pulls Eastmoney's 公告 API
     for every A-share with manifest + atomic-replace + dedup. Adding a
     second source would double the rate-limit pressure and split
     idempotency state across two cron paths.
  2. The Eastmoney announcement title is sufficient to bucket insider
     events with high precision (verified manually on 2026-06-09:
     ~46 hits / 1,924 announcements, ~2.4 % daily volume).
  3. AKShare has primary sources (``stock_ggcg_em``, ``stock_share_hold_change_sse / _szse``,
     ``stock_hold_management_detail_em``) but they each carry their own
     rate-limit profile + ChunkedEncodingError patterns. They are
     reserved for the "Phase 2" backfill / cross-check pass — see
     ``docs/llm_channel_3_insider_trading_spike_20260609.md``.

Filter contract
---------------
INCLUDE if any keyword in title:
    减持 | 增持 | 权益变动 | 协议转让 | 新进股东
EXCLUDE if any keyword in title:
    回购 | 注销   (these are buyback / cancellation, NOT insider trades)
EXCLUDE if title contains 计划期满未实施 | 计划期届满未减持 — stale planned-
    but-unexecuted disclosures under 减持新规 are not actionable signals.

Output
------
    data/storage/insider_announcements/<YYYY-MM-DD>.jsonl
    data/storage/insider_announcements/<YYYY-MM-DD>.manifest.json

One row per (stock_code, ann_id) — same line format as the upstream
``announcements/`` jsonl plus a ``channel: "insider_trading"`` tag
so the LLM extractor can stream by channel.

Usage
-----
    # daily cron (today)
    python scripts/collect_insider_announcements.py

    # backfill 30 trading days
    python scripts/collect_insider_announcements.py --days 30

    # explicit date
    python scripts/collect_insider_announcements.py --date 2026-06-09

SPIKE status
------------
SCAFFOLD ONLY. The collect path is implemented (no LLM dep), but the
keyword filter is V0 — Channel 3 Phase 1 work (per the spike doc) is to
validate filter precision/recall on 30 sampled days vs hand-labelled
ground truth before turning the daily cron on.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
ANN_DIR = DATA_DIR / "announcements"
OUT_DIR = DATA_DIR / "insider_announcements"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Title filter (V0). Refine in Phase 1 against hand-labelled ground truth.
INCLUDE_KEYWORDS = ("减持", "增持", "权益变动", "协议转让", "新进股东")
# 回购 / 注销 are buyback paths — handled by share_buyback events, not insider.
# 业绩预告 / 业绩快报 sometimes co-occur with 增减持 but are own channel.
EXCLUDE_KEYWORDS = ("回购", "注销", "业绩预告", "业绩快报")
# Stale-planned-but-unexecuted disclosures under 减持新规 (2023). The text
# of the announcement reveals "持有股份未发生变动" — these are PIT-stale
# signals that already leaked when the plan was first announced 6 months
# ago. Filter on title; the LLM may also flag via ``is_committed_no_sell``.
STALE_PLAN_KEYWORDS = (
    "计划期满未实施",
    "计划期届满未减持",
    "计划期届满未实施",
    "减持期间届满未减持",
)


def _is_insider_title(title: str) -> bool:
    """V0 keyword filter. Return True when the announcement looks like an
    insider 增减持 event we want to send to the LLM.
    """
    if not title:
        return False
    if not any(k in title for k in INCLUDE_KEYWORDS):
        return False
    if any(k in title for k in EXCLUDE_KEYWORDS):
        return False
    if any(k in title for k in STALE_PLAN_KEYWORDS):
        return False
    return True


def filter_for_date(date: str) -> Path:
    """Read the upstream announcement jsonl for ``date`` and write the
    insider-only subset to ``OUT_DIR/<date>.jsonl``.

    Idempotent: if a sibling manifest exists with ``finished=True`` AND
    ``n_items >= 1`` AND ``upstream_mtime`` matches the current upstream
    file, the rebuild is skipped. Otherwise re-runs from scratch (the
    filter is pure on the upstream jsonl, no LLM cost).
    """
    src_path = ANN_DIR / f"{date}.jsonl"
    out_path = OUT_DIR / f"{date}.jsonl"
    manifest_path = OUT_DIR / f"{date}.manifest.json"

    if not src_path.exists():
        logger.warning("  %s: upstream announcement file missing (%s)", date, src_path)
        return out_path

    upstream_mtime = src_path.stat().st_mtime_ns

    # Idempotency check — match the pattern in collect_announcements.py.
    # SPIKE TODO: also bind on filter version so a filter change forces
    # rebuild without manual rm of manifest files.
    if out_path.exists() and manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text())
            if (
                m.get("finished")
                and int(m.get("n_items", 0)) >= 1
                and m.get("upstream_mtime_ns") == upstream_mtime
                and m.get("filter_version") == FILTER_VERSION
            ):
                logger.info(
                    "  %s: finished filter (%d insider items), skip",
                    date,
                    int(m["n_items"]),
                )
                return out_path
        except Exception:
            pass

    n_in, n_out = 0, 0
    rows: list[dict] = []
    with src_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _is_insider_title(item.get("title", "")):
                continue
            item["channel"] = "insider_trading"
            rows.append(item)
            n_out += 1

    # Atomic write — same pattern as collect_announcements.py.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_path)

    manifest = {
        "target_date": date,
        "finished": True,
        "n_input": n_in,
        "n_items": n_out,
        "n_unique_stocks": len(set(r.get("stock_code", "") for r in rows)),
        "upstream_mtime_ns": upstream_mtime,
        "filter_version": FILTER_VERSION,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "channel": "insider_trading",
    }
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    os.replace(manifest_tmp, manifest_path)

    logger.info(
        "  %s: %d / %d announcements matched insider filter (%.1f %%)",
        date,
        n_out,
        n_in,
        100 * n_out / max(1, n_in),
    )
    return out_path


# Bumped manually whenever INCLUDE/EXCLUDE/STALE keyword sets change so
# a filter revision forces idempotent rebuild without manual cache clear.
FILTER_VERSION = "v0.1-spike-20260609"


def main():
    parser = argparse.ArgumentParser(
        description="Filter Eastmoney announcements down to insider 增减持 events",
    )
    parser.add_argument("--date", default=None, help="YYYY-MM-DD; defaults to today")
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Backfill N trading days (default 1 = today only)",
    )
    args = parser.parse_args()

    if args.date:
        dates = [args.date]
    else:
        dates = []
        d = datetime.now()
        n = 0
        while n < args.days:
            if d.weekday() < 5:
                dates.append(d.strftime("%Y-%m-%d"))
                n += 1
            d -= timedelta(days=1)
        dates.reverse()

    logger.info("Filtering insider announcements for %d dates...", len(dates))
    total = 0
    for date in dates:
        p = filter_for_date(date)
        if p.exists():
            total += sum(1 for _ in open(p))
    logger.info("Total insider announcements: %d across %d dates", total, len(dates))


if __name__ == "__main__":
    main()
