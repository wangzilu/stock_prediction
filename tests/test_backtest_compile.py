"""Compile + import smoke gate for every module under backtest/.

Per cx code review 2026-05-31 P0 + Step 0 of A-share tech debt sequence:
backtest/portfolio_backtest.py shipped with a syntax error that broke
every research script importing it. There was no CI check that would
have caught it. This test fills that gap — any module under backtest/
that fails py_compile or fails to import (transitive dependency missing,
ImportError chain, etc.) fails CI.

Cheap to run (< 0.5s per module), high-signal: catches typos, broken
refactors, accidental syntax errors, missing imports.
"""

from __future__ import annotations

import importlib
import py_compile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DIR = PROJECT_ROOT / "backtest"


def _discover_backtest_modules() -> list[tuple[str, Path]]:
    """Yield (module_dotted_name, file_path) for every .py file under
    backtest/ except __pycache__."""
    modules = []
    for py_file in BACKTEST_DIR.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        rel = py_file.relative_to(PROJECT_ROOT)
        dotted = ".".join(rel.with_suffix("").parts)
        # Drop trailing `.__init__` for packages
        if dotted.endswith(".__init__"):
            dotted = dotted[: -len(".__init__")]
        modules.append((dotted, py_file))
    return modules


MODULES = _discover_backtest_modules()


@pytest.mark.parametrize("module_name,path", MODULES, ids=[m for m, _ in MODULES])
def test_backtest_module_py_compiles(module_name: str, path: Path):
    """Every .py under backtest/ must pass py_compile (no syntax errors)."""
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"py_compile failed for {path}: {e}")


@pytest.mark.parametrize("module_name,path", MODULES, ids=[m for m, _ in MODULES])
def test_backtest_module_imports(module_name: str, path: Path):
    """Every module under backtest/ must import cleanly (no broken
    transitive deps, ImportError chains, etc.)."""
    try:
        importlib.import_module(module_name)
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"import {module_name} failed: {type(e).__name__}: {e}")
