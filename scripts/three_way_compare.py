"""Three-way same-exam comparator for xgb_174 / xgb175 / xgb_242.

Reads the experiments ledger (``data/storage/experiments_ledger.jsonl``),
filters to runs that share the same ``split_config`` and ``data_end``,
and renders a single table:

    Model       Code      n_feat   RankIC   ICIR   Spread20  Spread100  Days
    xgb_174     5b485df    174    +0.058   0.31      226        148    1215
    xgb175      5b485df    205    +0.078   0.39      ...
    xgb_242     5b485df    242     ...

This is the answer the project lead asked for: "who wins on the same
exam, when 'the same exam' is defined as identical code commit,
identical 24-split end date, identical cache provenance."

The comparator REFUSES to mix code commits or data_ends in one row
(prints a warning instead) — a silent mix would be exactly the
cross-time, cross-split confusion this whole framework is built to
prevent.

Usage:
    # head-to-head with the default standard_24split + auto-pick most
    # common data_end among recent runs
    python scripts/three_way_compare.py

    # pin both axes explicitly
    python scripts/three_way_compare.py \
        --split-config standard_24split --data-end 2026-05-19

    # also write a markdown report
    python scripts/three_way_compare.py \
        --markdown docs/three_way_compare_20260606.md
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracker.experiment_ledger import read_ledger, filter_runs


def _fmt_float(x, n=4):
    if x is None:
        return "—"
    try:
        return f"{float(x):+.{n}f}"
    except (TypeError, ValueError):
        return str(x)


def _fmt_int(x):
    if x is None:
        return "—"
    try:
        return f"{int(x)}"
    except (TypeError, ValueError):
        return str(x)


def _table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    """Render a small left-aligned ASCII table. ``columns`` is a list of
    (label, key) pairs; rows are flat dicts."""
    widths = {label: len(label) for label, _ in columns}
    for r in rows:
        for label, key in columns:
            widths[label] = max(widths[label], len(str(r.get(key, ""))))

    header = "  ".join(label.ljust(widths[label]) for label, _ in columns)
    sep = "  ".join("-" * widths[label] for label, _ in columns)
    lines = [header, sep]
    for r in rows:
        lines.append("  ".join(
            str(r.get(key, "")).ljust(widths[label])
            for label, key in columns
        ))
    return "\n".join(lines)


def _choose_data_end(runs: list[dict]) -> Optional[str]:
    """Pick the most common data_end if user did not pin one. Returns
    None when no runs are available."""
    if not runs:
        return None
    counts = Counter(r.get("data_end") for r in runs if r.get("data_end"))
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _build_rows(runs: list[dict]) -> list[dict]:
    """Project ledger entries onto the comparator table schema."""
    rows = []
    # Sort: highest RankIC first, NaNs / missing at the bottom
    def _ic(r):
        try:
            return -float(r.get("metrics", {}).get("rank_ic_mean") or float("-inf"))
        except Exception:
            return float("inf")
    for r in sorted(runs, key=_ic):
        m = r.get("metrics") or {}
        rows.append({
            "Model": r.get("model_profile", "?"),
            "Commit": (r.get("code_commit") or "")[:7] or "—",
            "Cache": Path(r.get("cache_path") or "").name or "—",
            "n_feat": _fmt_int(r.get("feature_count")),
            "RankIC": _fmt_float(m.get("rank_ic_mean")),
            "ICIR": _fmt_float(m.get("rank_icir"), 2),
            "Spread20": _fmt_int(m.get("spread_top20")),
            "Spread100": _fmt_int(m.get("spread_top100")),
            "Days": _fmt_int(m.get("n_days")),
            "data_end": r.get("data_end", "?"),
            "exp_id": r.get("experiment_id", "?"),
        })
    return rows


def _to_markdown(rows: list[dict], split_config: str, data_end: str) -> str:
    lines = [
        f"# Three-way head-to-head: 174 / 175 / 242",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Split config: `{split_config}`",
        f"- Pinned data_end: `{data_end}`",
        f"- Rows are from `data/storage/experiments_ledger.jsonl`.",
        "",
        "## Headline metrics",
        "",
        "| Model | Commit | Cache | n_feat | RankIC | ICIR | Spread20 | Spread100 | Days |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {Model} | `{Commit}` | `{Cache}` | {n_feat} | {RankIC} | {ICIR} | "
            "{Spread20} | {Spread100} | {Days} |".format(**r)
        )
    lines.extend([
        "",
        "## Verdict prerequisites",
        "",
        "Before reading anything into RankIC differences, verify:",
        "1. All rows share the same `Commit` and `data_end` columns.",
        "2. All rows used the same `split_config`.",
        "3. No row is missing metrics (`—` in a cell means metrics were absent or NaN).",
        "",
        "If any of those fails, the comparison is NOT apples-to-apples and",
        "the rankings can be misleading.",
        "",
        "## Provenance",
        "",
    ])
    for r in rows:
        lines.append(f"- `{r['exp_id']}` → `{r['Cache']}` (commit {r['Commit']})")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--split-config", default="24split",
        help="Pin split_config (default: 24split — matches phase4e default).",
    )
    p.add_argument(
        "--data-end", default=None,
        help="Pin data_end (YYYY-MM-DD). When omitted, the most common "
             "data_end among matching runs is used.",
    )
    p.add_argument(
        "--code-commit", default=None,
        help="Optional commit prefix filter — e.g. '5b485df' to compare "
             "only runs from the current tree.",
    )
    p.add_argument(
        "--markdown", default=None,
        help="Optional path to write a markdown report (alongside the "
             "ASCII table that always prints to stdout).",
    )
    args = p.parse_args()

    matching = filter_runs(
        split_config=args.split_config,
        data_end=args.data_end,
        code_commit=args.code_commit,
    )
    if not matching:
        # When the user didn't pin data_end, drop the data_end filter and
        # auto-pick the most common one.
        if args.data_end is None:
            all_runs = filter_runs(
                split_config=args.split_config,
                code_commit=args.code_commit,
            )
            chosen = _choose_data_end(all_runs)
            if chosen is not None:
                print(f"No runs at user-pinned data_end; auto-pick most common "
                      f"data_end={chosen} from {len(all_runs)} matching rows.")
                matching = [r for r in all_runs if r.get("data_end") == chosen]
                args.data_end = chosen

    if not matching:
        print(f"No ledger rows match split_config={args.split_config!r} "
              f"data_end={args.data_end!r} commit={args.code_commit!r}. "
              f"Either you have not run anything yet, or your filters are wrong.")
        sys.exit(1)

    # Refuse to mix code commits silently.
    distinct_commits = {r.get("code_commit", "")[:12] for r in matching}
    if args.code_commit is None and len(distinct_commits) > 1:
        print("WARN: the matching ledger rows span multiple code commits:")
        for c in sorted(distinct_commits):
            print(f"        - {c}")
        print("       Pass --code-commit <prefix> to lock the comparison to ONE tree.")

    # Refuse to mix data_ends silently.
    distinct_ends = {r.get("data_end") for r in matching}
    if args.data_end is None and len(distinct_ends) > 1:
        print("WARN: the matching ledger rows span multiple data_ends:")
        for d in sorted(distinct_ends):
            print(f"        - {d}")
        print("       Pass --data-end YYYY-MM-DD to lock the comparison.")

    rows = _build_rows(matching)

    columns = [
        ("Model", "Model"),
        ("Commit", "Commit"),
        ("Cache", "Cache"),
        ("n_feat", "n_feat"),
        ("RankIC", "RankIC"),
        ("ICIR", "ICIR"),
        ("Spread20", "Spread20"),
        ("Spread100", "Spread100"),
        ("Days", "Days"),
        ("data_end", "data_end"),
    ]
    print()
    print(f"split_config = {args.split_config}")
    if args.data_end:
        print(f"data_end     = {args.data_end}")
    if args.code_commit:
        print(f"code_commit  = {args.code_commit}")
    print()
    print(_table(rows, columns))
    print()

    if args.markdown:
        md = _to_markdown(rows, args.split_config, args.data_end or "<auto>")
        out_path = Path(args.markdown).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
        print(f"Markdown report written: {out_path}")


if __name__ == "__main__":
    main()
