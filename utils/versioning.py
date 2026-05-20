"""Data version binding for experiment reproducibility.

Every experiment artifact must embed version metadata so results can be
traced back to the exact code, data, and config that produced them.

Usage:
    from utils.versioning import get_experiment_metadata
    meta = get_experiment_metadata()
    # Include `meta` in your JSON output
"""

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "experiment.py"
_DEFAULT_CACHE_PATH = (
    _PROJECT_ROOT / "data" / "storage" / "feature_cache_174_holder_regime_ma.parquet"
)
_QLIB_CALENDAR_PATH = (
    _PROJECT_ROOT / "data" / "storage" / "qlib_data" / "cn_data" / "calendars" / "day.txt"
)


def get_code_version() -> str:
    """Return the current git HEAD commit hash (short)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def get_data_version(cache_path: Path | str | None = None) -> str:
    """Fast fingerprint of the feature cache: mtime + file size hashed.

    We deliberately avoid sha256-ing the full 3+ GB parquet file.
    The mtime+size combo changes whenever the file is rebuilt.
    """
    path = Path(cache_path) if cache_path else _DEFAULT_CACHE_PATH
    if not path.exists():
        return "missing"
    stat = path.stat()
    fingerprint = f"{stat.st_mtime_ns}:{stat.st_size}"
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def get_config_version(config_path: Path | str | None = None) -> str:
    """SHA-256 (truncated) of config/experiment.py contents."""
    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.exists():
        return "missing"
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]


def get_qlib_data_date() -> str:
    """Read the latest date from the qlib calendar file."""
    if not _QLIB_CALENDAR_PATH.exists():
        return "unknown"
    try:
        # Calendar is one date per line, sorted ascending; read last line
        text = _QLIB_CALENDAR_PATH.read_text().strip()
        if text:
            return text.splitlines()[-1].strip()
    except Exception:
        pass
    return "unknown"


def get_experiment_metadata(
    cache_path: Path | str | None = None,
    model_version: str = "xgb_174",
) -> dict:
    """Return a complete version-binding dict for experiment artifacts.

    Parameters
    ----------
    cache_path : Path or str, optional
        Path to the feature cache parquet. Defaults to the standard
        feature_cache_174_holder_regime_ma.parquet.
    model_version : str
        Human-readable model identifier, e.g. "xgb_174", "xgb_175".

    Returns
    -------
    dict with keys: data_version, model_version, code_version,
    config_version, qlib_data_date, timestamp.
    """
    return {
        "data_version": get_data_version(cache_path),
        "model_version": model_version,
        "code_version": get_code_version(),
        "config_version": get_config_version(),
        "qlib_data_date": get_qlib_data_date(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
