"""Backfill historical LPR (Loan Prime Rate) via AKShare.

PBC's own ``pbc.gov.cn`` LPR list page only exposes ~4 announcements
(2026-03 to 2026-05). The web archive can't be relied upon. But
``akshare.macro_china_lpr()`` returns 1572 records going back to
1991-04 — the entire LPR history the production chain needs.

This script:
  1. Fetches the AKShare LPR series
  2. Converts each row into a synthetic policy_event-shaped JSONL
     so build_policy_factors.py can re-use the same factor build
     path as live PE-1 cron runs.
  3. Writes to ``data/storage/policy_events/pbc/<YYYY-MM-DD>.jsonl``
     (one row per announcement date)
  4. Also writes a sidecar manifest so the operator can see what
     was backfilled vs collected live.

After backfill:
  python scripts/build_policy_factors.py --source pbc \
      --start 1991-04-21 --end 2026-05-20
  → data/storage/pbc_liquidity_factors.parquet covering 35 years.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = DATA_DIR / "policy_events" / "pbc"
MANIFEST_PATH = OUTPUT_DIR / "_backfill_lpr_manifest.json"


def _lpr_row_to_event(date: str, lpr_1y: float, lpr_5y: float,
                       prev_1y: float | None, prev_5y: float | None) -> dict:
    """Convert one LPR row into our policy_event schema."""
    # Determine stance from rate movement
    d1 = (lpr_1y - prev_1y) if prev_1y is not None else 0.0
    d5 = (lpr_5y - prev_5y) if prev_5y is not None else 0.0
    if d1 < -0.0001 or d5 < -0.0001:
        stance = "easing"
    elif d1 > 0.0001 or d5 > 0.0001:
        stance = "tightening"
    else:
        stance = "neutral"

    # repo_rate_change in basis points (use 1Y change; treat 5Y as
    # secondary), positive = hike, negative = cut.
    repo_change_bps = int(round(d1 * 10000))
    return {
        "publish_date": date,
        "policy_type": "lpr",
        "title": f"LPR 公告 {date}",
        "policy_stance": stance,
        "tool_type": "lpr",
        "duration_days": 365,
        "liquidity_injection_amount": None,
        "net_injection": None,
        "repo_rate_change": repo_change_bps,
        "unexpectedness": 0.5,  # historical-derived; not surprise-tagged
        "url": "(akshare historical backfill)",
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
        "source": "akshare_macro_china_lpr",
        "backfilled": True,
        "lpr_1y": lpr_1y,
        "lpr_5y": lpr_5y,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't write files; just print stats.")
    args = ap.parse_args()

    logger.info("Fetching AKShare LPR history…")
    import akshare as ak
    df = ak.macro_china_lpr()
    if df.empty:
        raise SystemExit("AKShare returned empty LPR DataFrame")

    # AKShare returns descending date order; sort ascending for prev-row diff.
    df = df.sort_values("TRADE_DATE").reset_index(drop=True)
    logger.info("Got %d LPR rows: %s → %s", len(df),
                df.iloc[0]["TRADE_DATE"], df.iloc[-1]["TRADE_DATE"])

    written = []
    skipped_existing = 0
    prev_1y = None
    prev_5y = None
    for _, row in df.iterrows():
        date = pd.Timestamp(row["TRADE_DATE"]).strftime("%Y-%m-%d")
        # LPR1Y and LPR5Y may be NaN for very early rows; skip those.
        if pd.isna(row.get("LPR1Y")):
            continue
        l1 = float(row["LPR1Y"])
        l5 = float(row["LPR5Y"]) if not pd.isna(row.get("LPR5Y")) else l1
        event = _lpr_row_to_event(date, l1, l5, prev_1y, prev_5y)
        prev_1y, prev_5y = l1, l5

        out_path = OUTPUT_DIR / f"{date}.jsonl"
        if out_path.exists():
            # Don't clobber live-collected days; the manifest tracks this
            skipped_existing += 1
            continue
        if not args.dry_run:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(event, ensure_ascii=False) + "\n")
            tmp.replace(out_path)
        written.append(date)

    logger.info("Wrote %d new files, skipped %d existing.",
                len(written), skipped_existing)
    if not args.dry_run:
        MANIFEST_PATH.write_text(json.dumps({
            "source": "akshare_macro_china_lpr",
            "backfilled_at": datetime.now().isoformat(timespec="seconds"),
            "n_written": len(written),
            "n_skipped_existing": skipped_existing,
            "first_date": written[0] if written else None,
            "last_date": written[-1] if written else None,
        }, indent=2))
        logger.info("Wrote manifest: %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
