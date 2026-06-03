#!/usr/bin/env python3
"""Unified network wrapper for cron jobs.

Usage:
    python run_network_job.py --network domestic|global|crypto|none|llm|push [--timeout N] -- command...
"""
import argparse
import os
import socket
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Allow importing config even when running as a script
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config.network_profiles import (
    PROXY_URL,
    PROXY_PORT,
    PROXY_START_CMD,
    LLM_NETWORK,
    PUSH_NETWORK,
)

# All proxy-related env vars we manage
PROXY_VARS = [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
    "no_proxy",
]


def _log(msg: str) -> None:
    print(f"[run_network_job] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Proxy helpers
# ---------------------------------------------------------------------------

def _unset_proxy(env: dict) -> None:
    for var in PROXY_VARS:
        env.pop(var, None)


def _set_proxy(env: dict) -> None:
    for var in PROXY_VARS:
        if var == "no_proxy":
            continue
        env[var] = PROXY_URL


def _proxy_listening() -> bool:
    """Check if proxy port is accepting connections (timeout 3s)."""
    try:
        with socket.create_connection(("127.0.0.1", PROXY_PORT), timeout=3):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def _ensure_proxy() -> bool:
    """Ensure the proxy is up; try starting it if not. Returns True if ready."""
    if _proxy_listening():
        return True

    _log(f"Proxy not listening on :{PROXY_PORT}, attempting to start ...")
    try:
        subprocess.Popen(
            PROXY_START_CMD,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        _log(f"Failed to launch proxy start command: {exc}")
        return False

    # Wait up to 10s, polling every 1s
    for i in range(10):
        time.sleep(1)
        if _proxy_listening():
            _log(f"Proxy came up after ~{i + 1}s")
            return True

    _log("Proxy did not come up within 10s — aborting (fail fast)")
    return False


# ---------------------------------------------------------------------------
# Network profile application
# ---------------------------------------------------------------------------

def apply_profile(profile: str, env: dict, timeout: int | None) -> int | None:
    """Apply network profile to *env* dict. Returns adjusted timeout."""
    if profile == "domestic":
        _unset_proxy(env)
        _log("Profile: domestic — proxy env vars cleared")

    elif profile == "global":
        if not _ensure_proxy():
            sys.exit(1)
        _set_proxy(env)
        _log(f"Profile: global — proxy set to {PROXY_URL}")

    elif profile == "crypto":
        # Per plans/crypto-data-contract.md §1.5 + config/crypto_network.py:
        # crypto data fetches MUST traverse ssproxy. The collector
        # entrypoints call assert_proxy_active() which checks the env
        # sentinels we set below. If ssproxy preflight fails, exit
        # network_unreachable; A-share cron is unaffected because each
        # job is invoked under its own wrapper instance.
        if not _ensure_proxy():
            _log("Profile: crypto — ssproxy preflight failed; aborting")
            sys.exit(2)
        _set_proxy(env)
        env["CRYPTO_NETWORK_ACTIVE"] = "crypto"
        env["CRYPTO_SSPROXY_VERIFIED"] = "1"
        _log(
            f"Profile: crypto — proxy set to {PROXY_URL}; sentinels "
            "CRYPTO_NETWORK_ACTIVE=crypto + CRYPTO_SSPROXY_VERIFIED=1"
        )

    elif profile == "none":
        _unset_proxy(env)
        _log("Profile: none — proxy env vars cleared")

    elif profile == "llm":
        # Resolve alias
        return apply_profile(LLM_NETWORK, env, timeout)

    elif profile == "push":
        _unset_proxy(env)
        if timeout is None or timeout > 60:
            timeout = 60
        _log(f"Profile: push — proxy cleared, timeout capped at {timeout}s")

    else:
        _log(f"Unknown network profile: {profile}")
        sys.exit(1)

    return timeout


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a command under a specific network profile.",
    )
    parser.add_argument(
        "--network",
        required=True,
        choices=["domestic", "global", "crypto", "none", "llm", "push"],
        help="Network profile to apply",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for the sub-command",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run (after '--')",
    )

    args = parser.parse_args()

    # Strip leading '--' from remainder
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        _log("No command specified")
        sys.exit(1)

    env = os.environ.copy()
    timeout = apply_profile(args.network, env, args.timeout)

    _log(f"Timeout: {timeout}s" if timeout else "Timeout: none")
    _log(f"Command: {' '.join(cmd)}")

    try:
        # start_new_session=True creates a new process group so we can
        # kill the entire tree (including child workers) on timeout.
        proc = subprocess.Popen(cmd, env=env, start_new_session=True)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill entire process group, not just the direct child
            import signal
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            _log(f"TIMEOUT — command exceeded {timeout}s, killed process group")
            sys.exit(124)
        sys.exit(proc.returncode)
    except Exception as exc:
        _log(f"ERROR — {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
