"""Runtime helpers for initializing Qlib consistently.

Qlib's default joblib backend uses multiprocessing. That is fast for normal
script execution, but fragile when code is launched from stdin, notebooks, or
ad-hoc probes on macOS because worker processes cannot re-import ``<stdin>``.
"""
from __future__ import annotations

import os
from typing import Any

import qlib
from qlib.constant import REG_CN


TRUE_VALUES = {"1", "true", "yes", "on"}


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1, got {parsed}")
    return parsed


def build_qlib_init_kwargs(provider_uri: str, region: str = REG_CN, **overrides: Any) -> dict[str, Any]:
    """Build qlib.init kwargs with optional debug-safe multiprocessing controls."""
    debug_safe = os.environ.get("QLIB_DEBUG_SAFE", "").lower() in TRUE_VALUES
    backend = os.environ.get("QLIB_JOBLIB_BACKEND") or ("threading" if debug_safe else None)
    kernels = _optional_int_env("QLIB_KERNELS") or (1 if debug_safe else None)

    kwargs: dict[str, Any] = {
        "provider_uri": provider_uri,
        "region": region,
    }
    if backend:
        kwargs["joblib_backend"] = backend
    if kernels is not None:
        kwargs["kernels"] = kernels
    kwargs.update({key: value for key, value in overrides.items() if value is not None})
    return kwargs


def init_qlib(provider_uri: str, region: str = REG_CN, **overrides: Any) -> None:
    """Initialize Qlib using project-wide runtime safeguards."""
    qlib.init(**build_qlib_init_kwargs(provider_uri, region=region, **overrides))
