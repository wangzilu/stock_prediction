"""Crypto storage root resolver (Phase Crypto-A).

Per `plans/crypto-data-contract.md` §1 Storage Root:

- Real path is `/Volumes/DATA/crypto/` (external volume).
- Repo-tree path `data/storage/crypto/` is **forbidden** as a storage
  target. Only `data/storage/crypto/README.md` (sentinel) lives in
  repo.
- Every crypto entrypoint must call `crypto_root()` (or its alias
  `ensure_mounted_and_writable()`) as its first action. If the
  external volume is not mounted, the function raises
  `RuntimeError` immediately — no silent fallback.

The module performs no I/O at import time so it is cheap to import
even from contexts that will never write crypto data.
"""
from __future__ import annotations

import os
from pathlib import Path

# Environment override is allowed for tests and (eventually) for users
# who store the volume under a different mount path. Production cron
# does not set it; defaults to the contract path.
_DEFAULT_ROOT = Path("/Volumes/DATA/crypto")

# Required subdirectories that must be writable when the root is in use.
# Mirrors the contract §1 listing.
REQUIRED_SUBDIRS: tuple[str, ...] = (
    "raw/ohlcv",
    "raw/funding",
    "raw/open_interest",
    "features",
    "predictions",
    "health",
    "audit",
    "paper",
    "reports/daily",
)

# Forbidden in-repo path. crypto_root() must never return this prefix.
_REPO_FORBIDDEN_PREFIX = Path(__file__).resolve().parents[1] / "data" / "storage" / "crypto"


class CryptoStorageNotMountedError(RuntimeError):
    """Raised when /Volumes/DATA/crypto (or the env-override path) is
    not present at the moment a crypto entrypoint asks for the storage
    root. Distinct from generic RuntimeError so callers and tests can
    catch it precisely."""


def _resolve_root_from_env() -> Path:
    override = os.environ.get("CRYPTO_STORAGE_ROOT")
    if override:
        return Path(override)
    return _DEFAULT_ROOT


def crypto_root() -> Path:
    """Return the crypto storage root, after verifying it exists and
    is writable. Raise `CryptoStorageNotMountedError` if not.

    Caller contract:
      - Call this at the top of any crypto entrypoint.
      - Do NOT cache the result across process restarts — the volume
        may unmount between cron runs.
      - Do NOT fall back to a repo-tree path if this raises. The
        correct response is to exit with a clear error so the cron
        wrapper can mark the job failed (without affecting A-share).
    """
    root = _resolve_root_from_env()

    # Guard against accidentally pointing at the in-repo sentinel dir.
    try:
        root_resolved = root.resolve()
    except (OSError, RuntimeError):
        root_resolved = root
    if root_resolved == _REPO_FORBIDDEN_PREFIX:
        raise CryptoStorageNotMountedError(
            f"crypto_root() refuses to return the in-repo sentinel path "
            f"{_REPO_FORBIDDEN_PREFIX}. Set CRYPTO_STORAGE_ROOT to the "
            f"actual external volume, or mount /Volumes/DATA/crypto."
        )

    if not root.exists():
        raise CryptoStorageNotMountedError(
            f"Crypto storage volume not mounted: {root}. "
            "Mount the external volume or set CRYPTO_STORAGE_ROOT to a "
            "real path. Per data-contract §1 there is no silent "
            "fallback to a repo-tree location."
        )
    if not root.is_dir():
        raise CryptoStorageNotMountedError(
            f"Crypto storage root {root} exists but is not a directory."
        )
    # Cheap writability probe: try to create a transient hidden file.
    probe = root / ".crypto_root_probe"
    try:
        probe.touch(exist_ok=True)
        probe.unlink(missing_ok=True)
    except OSError as e:
        raise CryptoStorageNotMountedError(
            f"Crypto storage root {root} is not writable: {e}"
        ) from e

    return root


def ensure_mounted_and_writable() -> Path:
    """Alias used at entrypoint sites where readability is the priority.
    Returns the verified root; raises on failure (same as crypto_root)."""
    return crypto_root()


def required_subdirs(root: Path | None = None) -> list[Path]:
    """Return the list of required subdirectory paths under `root`.
    Does NOT create them — callers (typically Phase 0b's bootstrap
    script) decide when to materialise."""
    base = root if root is not None else crypto_root()
    return [base / sub for sub in REQUIRED_SUBDIRS]


def is_inside_crypto_root(path: Path) -> bool:
    """True if `path` is under the configured crypto root. Used by tests
    and by audit logs that want to verify a write target is policy-
    compliant. Does not raise if the root isn't mounted (returns False
    instead, since an unmounted path can't contain anything)."""
    try:
        root = _resolve_root_from_env().resolve()
    except (OSError, RuntimeError):
        return False
    try:
        return str(Path(path).resolve()).startswith(str(root) + os.sep) or \
            Path(path).resolve() == root
    except (OSError, RuntimeError):
        return False
