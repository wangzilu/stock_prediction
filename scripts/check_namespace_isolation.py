#!/usr/bin/env python3
"""Namespace isolation lint.

Fails (exit 1) if any forbidden cross-namespace import is present.

Per `plans/cc-crypto-implementation-spec-2026-05-30.md` §−0.5 Layer 1.

Phase 0a / 0b / 0c will expand this lint with `core/`, `ashare/`,
`crypto/` namespace rules. The current version (Quarantine PR) covers
only what is needed today:

1. Legacy crypto module must not be re-imported at module level by
   scheduler/jobs.py (must remain lazy per §6.5).
2. New crypto pipeline files (when they later land under
   `data/collectors/crypto_market.py`, `data/collectors/crypto_derivatives.py`,
   or `crypto/` namespace) must NOT import legacy
   `data.collectors.crypto`, `scheduler.jobs`, or `config.watchlist`.

Usage:
    python scripts/check_namespace_isolation.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --- Rule definitions ------------------------------------------------------


def _walk_imports(tree: ast.AST) -> Iterable[tuple[str, ast.AST]]:
    """Yield (dotted_module_name, ast_node) for every import in the tree."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module, node


def _is_legacy_crypto_import(node: ast.AST) -> bool:
    """Detect any of:
       - `from data.collectors.crypto import X`
       - `import data.collectors.crypto`
       - `importlib.import_module("data.collectors.crypto")`
       - `__import__("data.collectors.crypto")`
    Returns True if `node` references data.collectors.crypto via any
    of these forms.
    """
    target = "data.collectors.crypto"
    if isinstance(node, ast.ImportFrom) and node.module == target:
        return True
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == target:
                return True
    if isinstance(node, ast.Call):
        func = node.func
        # importlib.import_module("data.collectors.crypto")
        is_import_module = (
            isinstance(func, ast.Attribute)
            and func.attr == "import_module"
            and isinstance(func.value, ast.Name)
            and func.value.id == "importlib"
        )
        # __import__("data.collectors.crypto")
        is_dunder_import = isinstance(func, ast.Name) and func.id == "__import__"
        if is_import_module or is_dunder_import:
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value == target:
                    return True
    return False


def check_no_module_level_legacy_crypto_import(path: Path) -> list[str]:
    """scheduler/jobs.py: only `_get_crypto_collector` is allowed to
    reference `data.collectors.crypto`. Any import / dynamic-import
    anywhere else — including other functions, method bodies, or the
    module top-level — is forbidden by quarantine §6.5."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    errors = []
    allowed_function = "_get_crypto_collector"

    def _walk_under_function(func_node: ast.FunctionDef) -> bool:
        """Returns True if this function's name matches the allowed
        accessor."""
        return func_node.name == allowed_function

    # Determine, for every line in the file, whether that line is inside
    # the allowed function. Build a set of "allowed line numbers" by
    # walking each FunctionDef whose name matches.
    allowed_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _walk_under_function(node):
            for descendant in ast.walk(node):
                if hasattr(descendant, "lineno"):
                    allowed_lines.add(descendant.lineno)

    # Walk every node and collect legacy-crypto references not inside
    # the allowed accessor.
    for node in ast.walk(tree):
        if _is_legacy_crypto_import(node):
            line = getattr(node, "lineno", 0)
            if line not in allowed_lines:
                form = type(node).__name__
                errors.append(
                    f"{path}:{line}: "
                    f"legacy crypto reference via {form} is forbidden "
                    f"outside `{allowed_function}`. Per quarantine §6.5 "
                    f"only that accessor may import data.collectors.crypto."
                )
    return errors


CRYPTO_PIPELINE_GLOBS = (
    "data/collectors/crypto_market.py",
    "data/collectors/crypto_derivatives.py",
    "crypto/**/*.py",
)


FORBIDDEN_FOR_CRYPTO_PIPELINE = (
    "data.collectors.crypto",
    "scheduler.jobs",
    "config.watchlist",
    "ashare",
)


def check_crypto_pipeline_no_legacy_imports(path: Path) -> list[str]:
    """New crypto pipeline files must not import legacy A-share / legacy
    crypto modules."""
    if not path.exists():
        return []
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        return [f"{path}: syntax error during lint: {e}"]
    errors = []
    for mod, node in _walk_imports(tree):
        for forbidden in FORBIDDEN_FOR_CRYPTO_PIPELINE:
            if mod == forbidden or mod.startswith(forbidden + "."):
                errors.append(
                    f"{path}:{node.lineno}: "
                    f"forbidden import '{mod}' in new crypto pipeline file. "
                    f"§−0.5 Layer 1 rules: new crypto code may not depend on "
                    f"legacy `data.collectors.crypto`, A-share scheduler, "
                    f"config.watchlist, or ashare/ namespace."
                )
    return errors


# --- Driver ----------------------------------------------------------------


def collect_crypto_pipeline_files() -> list[Path]:
    files: list[Path] = []
    for glob_pattern in CRYPTO_PIPELINE_GLOBS:
        if "**" in glob_pattern:
            files.extend(PROJECT_ROOT.glob(glob_pattern))
        else:
            p = PROJECT_ROOT / glob_pattern
            if p.exists():
                files.append(p)
    return files


def main() -> int:
    all_errors: list[str] = []

    scheduler_jobs = PROJECT_ROOT / "scheduler" / "jobs.py"
    if scheduler_jobs.exists():
        all_errors.extend(check_no_module_level_legacy_crypto_import(scheduler_jobs))

    for crypto_file in collect_crypto_pipeline_files():
        all_errors.extend(check_crypto_pipeline_no_legacy_imports(crypto_file))

    if all_errors:
        print("Namespace isolation lint FAILED:\n", file=sys.stderr)
        for err in all_errors:
            print(f"  {err}", file=sys.stderr)
        print(
            f"\n{len(all_errors)} violation(s). See "
            "plans/cc-crypto-implementation-spec-2026-05-30.md §−0.5 Layer 1.",
            file=sys.stderr,
        )
        return 1

    print("Namespace isolation lint OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
