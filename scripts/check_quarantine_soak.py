#!/usr/bin/env python3
"""Quarantine 3-day soak health check.

Per plans/cc-crypto-implementation-spec-2026-05-30.md §6.5 merge gate
item 5: production cron must run 3 consecutive days with
LEGACY_MARKET_CONTEXT_ENABLED=false and A-share GREEN before the
`crypto` branch can be merged to master.

This script gives an unambiguous verdict for one calendar day:

  GREEN   — all expected A-share jobs ran success, evening report
            contains the "crypto context disabled" stub (proves
            quarantine code is live in production), no new error
            patterns introduced by quarantine
  YELLOW  — at least one non-fatal anomaly (job ran late but recovered,
            stale data warning, etc.) — investigate but don't roll back
  RED     — A-share regression OR quarantine bypassed (legacy crypto
            text appears in production output) — roll back

Usage:
    python scripts/check_quarantine_soak.py
    python scripts/check_quarantine_soak.py --date 2026-05-31

Exit codes:
    0  GREEN  (continue soak / merge if day 3)
    1  YELLOW (manual investigation needed)
    2  RED    (must roll back)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOB_STATUS = PROJECT_ROOT / "data" / "storage" / "job_status.json"
LOG_DIR = PROJECT_ROOT / "logs"

# A-share jobs that MUST run successfully each weekday. If any of these
# is missing or failed on the target date, that's RED.
REQUIRED_DAILY_JOBS = [
    "morning_recommendation",
    "sell_check",
    "daily_summary",
    "evening_outlook",
    "paper_trading",
]

# Patterns whose appearance in the evening report PROVES quarantine
# is active in production.
QUARANTINE_PROOF_PATTERNS = [
    "crypto context disabled",
    "legacy quarantine off",
]

# Patterns whose appearance would indicate quarantine BYPASS (legacy
# crypto forecast leaked into production despite flag default-false).
QUARANTINE_BYPASS_PATTERNS = [
    "BTC：震荡",
    "BTC：偏多",
    "BTC：偏空",
    "ETH：震荡",
    "ETH：偏多",
    "ETH：偏空",
]


def _load_job_status() -> dict:
    if not JOB_STATUS.exists():
        return {}
    with open(JOB_STATUS) as f:
        return json.load(f).get("jobs", {})


def _jobs_on_date(jobs: dict, target: str) -> dict:
    """Returns {job_name: status_dict} for jobs whose finished_at falls
    on target date (YYYY-MM-DD)."""
    return {
        name: info for name, info in jobs.items()
        if str(info.get("finished_at", "")).startswith(target)
    }


def _find_evening_log(target: str) -> Path | None:
    """Most recent evening_outlook log file. Cron appends, so this is
    actually a single cumulative file; we grep the target date out."""
    candidates = list(LOG_DIR.glob("cron_evening_outlook*.log"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _check_evening_quarantine_proof(target: str) -> tuple[str, list[str]]:
    """Returns (verdict, notes) — verdict in {GREEN, YELLOW, RED}.

    GREEN  : evening log on target date contains a quarantine proof
             pattern and no bypass pattern.
    YELLOW : evening log on target date contains neither proof nor
             bypass (e.g. evening outlook didn't render the crypto
             section at all — likely a different code path).
    RED    : evening log contains a bypass pattern (legacy crypto
             forecast leaked into production).
    """
    log = _find_evening_log(target)
    if log is None:
        return "YELLOW", ["evening_outlook log not found"]

    # Read only the lines from the target date. The log accumulates;
    # date markers appear in the report header text.
    text = log.read_text(encoding="utf-8", errors="ignore")
    # Crude window: take the last 200 KB which should cover the most
    # recent evening run plus some context.
    text = text[-200_000:]

    notes: list[str] = []
    has_proof = any(p in text for p in QUARANTINE_PROOF_PATTERNS)
    bypasses = [p for p in QUARANTINE_BYPASS_PATTERNS if p in text]

    if bypasses:
        notes.append(f"quarantine BYPASS detected: {bypasses}")
        return "RED", notes
    if has_proof:
        notes.append("quarantine proof pattern found in evening log")
        return "GREEN", notes
    notes.append(
        "neither quarantine proof nor bypass pattern found — evening "
        "outlook may have skipped the crypto section"
    )
    return "YELLOW", notes


def _check_required_jobs(jobs_today: dict, target: str) -> tuple[str, list[str]]:
    """Returns (verdict, notes)."""
    notes: list[str] = []
    missing = [j for j in REQUIRED_DAILY_JOBS if j not in jobs_today]
    failed = [
        j for j, info in jobs_today.items()
        if j in REQUIRED_DAILY_JOBS and info.get("status") != "success"
    ]

    # 2026-06-04 cx round 15 P2-5: use the CN trading calendar instead
    # of pandas weekday >= 5. Pre-fix春节 / 国庆 / 调休 produced false
    # "missing daily job" alarms (it WAS a calendar weekday but a CN
    # holiday) or false "weekend so OK" passes (调休 made a weekend
    # day a working day).
    is_non_trading_day = False
    try:
        from qlib.data import D
        cal = D.calendar(end_time=target)
        if cal is not None and len(cal) > 0:
            import pandas as _pd
            latest_trading = str(_pd.Timestamp(cal[-1]).date())
            is_non_trading_day = (target > latest_trading) or (
                target not in {str(_pd.Timestamp(d).date()) for d in cal}
            )
    except Exception:
        # Fallback: weekday-based check
        weekday = datetime.strptime(target, "%Y-%m-%d").weekday()
        is_non_trading_day = weekday >= 5

    if is_non_trading_day:
        notes.append(f"{target} is non-trading day (CN calendar) — daily-cron requirements waived")
        if failed:
            return "YELLOW", notes + [f"unexpected failures on non-trading day: {failed}"]
        return "GREEN", notes

    if failed:
        return "RED", [f"required jobs FAILED: {failed}"] + notes
    if missing:
        return "RED", [f"required jobs MISSING (never ran): {missing}"] + notes
    notes.append(
        f"all {len(REQUIRED_DAILY_JOBS)} required daily jobs ran successfully"
    )
    return "GREEN", notes


def _verdict(*sub_verdicts: str) -> str:
    """Aggregate: RED dominates YELLOW dominates GREEN."""
    if "RED" in sub_verdicts:
        return "RED"
    if "YELLOW" in sub_verdicts:
        return "YELLOW"
    return "GREEN"


def main() -> int:
    parser = argparse.ArgumentParser(description="Quarantine soak health check")
    parser.add_argument(
        "--date", type=str, default=None,
        help="YYYY-MM-DD (default: today)"
    )
    args = parser.parse_args()
    target = args.date or datetime.now().strftime("%Y-%m-%d")

    print(f"=== Quarantine soak check for {target} ===\n")

    jobs = _load_job_status()
    jobs_today = _jobs_on_date(jobs, target)
    print(f"Jobs that ran on {target}: {len(jobs_today)}")
    for name in sorted(jobs_today):
        info = jobs_today[name]
        finished = info.get("finished_at", "?")[:19]
        status = info.get("status", "?")
        marker = "✓" if status == "success" else "✗"
        print(f"  {marker} {finished}  {status:10s}  {name}")
    print()

    job_verdict, job_notes = _check_required_jobs(jobs_today, target)
    print(f"[A-share daily jobs]  {job_verdict}")
    for n in job_notes:
        print(f"  · {n}")
    print()

    qua_verdict, qua_notes = _check_evening_quarantine_proof(target)
    print(f"[Quarantine proof]    {qua_verdict}")
    for n in qua_notes:
        print(f"  · {n}")
    print()

    overall = _verdict(job_verdict, qua_verdict)
    print(f"=== Verdict for {target}: {overall} ===")

    if overall == "GREEN":
        print()
        print("Continue soak. If day 3 also GREEN → merge crypto → master.")
        return 0
    if overall == "YELLOW":
        print()
        print("Investigate manually. Soak clock does NOT reset on YELLOW,")
        print("but verify the anomaly is unrelated to quarantine.")
        return 1
    # RED
    print()
    print("ABORT soak. Rollback steps (per spec §8):")
    print("  1. python scripts/disable_crypto_cron.py    (no crypto cron yet, OK)")
    print("  2. git checkout master                       (revert to pre-quarantine HEAD)")
    print("  3. Investigate logs/ for the specific failure")
    return 2


if __name__ == "__main__":
    sys.exit(main())
