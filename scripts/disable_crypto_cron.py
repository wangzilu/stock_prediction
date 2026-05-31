#!/usr/bin/env python3
"""Disable crypto cron entries — SAFETY PLACEHOLDER.

Per `plans/cc-crypto-implementation-spec-2026-05-30.md` §8 rollback
Tier 2. Today (2026-05-31) there are NO crypto cron entries — Phase A
hasn't created any yet. This script exists so the rollback
documentation does not reference a missing tool, AND so when Phase A
later adds crypto entries the kill-switch exists from day one
(updating this script's logic is the Phase A deployment task).

By design this script:
  ✓ READS current crontab to identify any crypto entries
  ✓ PRINTS what would be removed
  ✗ DOES NOT touch A-share cron entries (independent failure domain)
  ✗ DOES NOT modify crontab in this placeholder version — it only
    reports. The actual removal logic is intentionally NOT wired up
    here. Phase A deployment will extend this to a `--apply` mode
    once there are crypto entries worth removing.

The narrow scope is deliberate per user direction 2026-05-31:
"只做安全占位，不接生产 crontab 逻辑太深，不碰 A 股 cron block."

Exit codes:
  0  no crypto cron entries found (current state — clean baseline)
  0  crypto entries found and listed (would require --apply to remove,
     which this placeholder does not implement)
  1  crontab unreadable (permission / not installed)
"""

from __future__ import annotations

import re
import subprocess
import sys

# Patterns that identify a crypto cron entry. Today the entire crypto
# code path is quarantined behind LEGACY_MARKET_CONTEXT_ENABLED so
# nothing here matches. Phase A adds entries like crypto_ohlcv_1h /
# crypto_funding / crypto_data_health — those will start matching.
CRYPTO_JOB_PATTERNS = [
    re.compile(r"\bcrypto_ohlcv_(?:1h|4h|1d)\b"),
    re.compile(r"\bcrypto_funding\b"),
    re.compile(r"\bcrypto_oi\b"),
    re.compile(r"\bcrypto_data_health\b"),
    re.compile(r"\bcrypto_update_market_data\.py\b"),
    re.compile(r"\bcrypto_update_derivatives\.py\b"),
    re.compile(r"\bcrypto_paper_trading\b"),
    re.compile(r"\bcrypto_daily_report\.py\b"),
    re.compile(r"\bcrypto_train_model\.py\b"),
    re.compile(r"\bcrypto_predict\.py\b"),
    re.compile(r"\bcrypto_build_features\.py\b"),
    re.compile(r"\bcrypto_backtest_baseline\.py\b"),
    re.compile(r"\brun_crypto_job\.py\b"),
]

# Explicit list of NON-crypto job-id substrings that this script must
# never touch even if a pattern false-matches. Defense-in-depth so
# accidental future regex broadening cannot kill A-share cron.
ASHARE_GUARD_PATTERNS = [
    re.compile(r"--job-id\s+(?!crypto)"),  # any job-id that doesn't start with "crypto"
]


def _read_crontab() -> str | None:
    """Return current crontab text. None on permission / not-installed
    failure. Empty string is a legitimate value (no crontab installed
    for this user)."""
    try:
        result = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True
        )
    except (FileNotFoundError, PermissionError) as e:
        print(f"ERROR: cannot read crontab ({e}). Is crontab installed?", file=sys.stderr)
        return None
    if result.returncode != 0:
        # Common case: no crontab for current user → returncode 1, stderr "no crontab for ..."
        msg = (result.stderr or "").strip()
        if "no crontab" in msg.lower():
            return ""
        print(f"ERROR: crontab -l exit {result.returncode}: {msg}", file=sys.stderr)
        return None
    return result.stdout


def _identify_crypto_lines(crontab_text: str) -> list[str]:
    """Return the cron lines that look like crypto entries."""
    matches = []
    for line in crontab_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if any(p.search(stripped) for p in CRYPTO_JOB_PATTERNS):
            matches.append(stripped)
    return matches


def main() -> int:
    crontab_text = _read_crontab()
    if crontab_text is None:
        return 1

    crypto_lines = _identify_crypto_lines(crontab_text)

    if not crypto_lines:
        print(
            "No crypto cron entries found. Crypto runtime is fully "
            "quarantined at the cron layer.\n"
            "(Phase A will add entries — at that point this script's "
            "--apply mode needs to be implemented.)"
        )
        return 0

    print(f"Found {len(crypto_lines)} crypto cron entries:")
    for line in crypto_lines:
        # Print first 120 chars to keep output scannable
        preview = line if len(line) <= 120 else line[:117] + "..."
        print(f"  - {preview}")
    print(
        "\nThis placeholder does not remove them. To implement removal "
        "(Phase A deployment task):\n"
        "  1. Strip matching lines from crontab text\n"
        "  2. Verify ALL surviving lines pass ASHARE_GUARD_PATTERNS\n"
        "  3. Pipe to `crontab -` to apply\n"
        "  4. Add --dry-run / --apply CLI flags\n"
        "  5. Add a test that asserts A-share entries are untouched"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
