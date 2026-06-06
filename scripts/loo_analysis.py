"""Phase B post-mortem — read the ledger LOO rows and rank groups.

Reads ``data/storage/experiments_ledger.jsonl`` for rows whose
``dropped_groups`` contains one element (LOO ablation runs) plus the
matching baseline (empty ``dropped_groups``), and prints the ranked
table by ΔRankIC = LOO_RankIC − baseline_RankIC.

Negative Δ means dropping the group hurt — the group was carrying
weight in the baseline. Positive Δ means dropping the group helped —
the group is net-negative and a candidate for removal from
PRODUCTION_SUPPLEMENTARY_GROUPS.

Usage::

    python scripts/loo_analysis.py
    python scripts/loo_analysis.py \\
        --split-config 6split --data-end 2026-05-19 \\
        --markdown docs/loo_ablation_20260606.md
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracker.experiment_ledger import filter_runs


def _ic(row: dict) -> float | None:
    m = row.get("metrics") or {}
    v = m.get("rank_ic_mean")
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def _split_baseline_vs_loo(rows: list[dict]) -> tuple[dict | None, list[dict]]:
    baseline = None
    loo = []
    for r in rows:
        if not r.get("dropped_groups"):
            # Empty list = baseline (no group dropped).
            if baseline is None or r.get("ts", "") > baseline.get("ts", ""):
                baseline = r
        elif len(r["dropped_groups"]) == 1:
            loo.append(r)
    return baseline, loo


def _table_md(baseline: dict, loo: list[dict]) -> str:
    base_ic = _ic(baseline)
    rows = []
    for r in loo:
        rows.append({
            "group": r["dropped_groups"][0],
            "rank_ic": _ic(r),
            "spread20": (r.get("metrics") or {}).get("spread_top20"),
            "n_days": (r.get("metrics") or {}).get("n_days"),
            "exp_id": r.get("experiment_id", ""),
        })
    # Δ = LOO_IC - baseline_IC. Sort with the BIGGEST positive Δ first
    # (= the LOO that helps the most = the noisiest group).
    for r in rows:
        ic = r["rank_ic"]
        r["delta"] = (ic - base_ic) if (ic is not None and base_ic is not None) else None
    rows.sort(key=lambda r: r["delta"] if r["delta"] is not None else -999, reverse=True)

    lines = [
        f"# LOO 6-split ablation — {len(rows)} groups",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Baseline (xgb_242 full, no drop) RankIC = **{_fmt(base_ic)}**",
        f"- Baseline experiment: `{baseline.get('experiment_id', '?')}`",
        f"- Baseline commit: `{baseline.get('code_commit', '?')[:7]}`",
        f"- Baseline cache: `{Path(baseline.get('cache_path', '?')).name}`",
        "",
        "## Ranked by ΔRankIC (best LOO first = group that hurts most when present)",
        "",
        "| Drop group | LOO RankIC | ΔRankIC vs baseline | Spread20 (bps) | Days | exp_id |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        d_sign = ""
        if r["delta"] is not None:
            d_sign = "✅" if r["delta"] > 0 else ("⚠️" if r["delta"] < -0.005 else "")
        lines.append(
            f"| `{r['group']}` | {_fmt(r['rank_ic'])} | {_fmt(r['delta'])} {d_sign} | "
            f"{_fmt(r['spread20'], 2)} | {r['n_days']} | `{r['exp_id'][:50]}` |"
        )
    lines.extend([
        "",
        "## Interpretation guide",
        "",
        "- **ΔRankIC > 0** (`✅`): dropping the group helped. The group is a "
        "net-negative loader and a candidate for removal from "
        "`PRODUCTION_SUPPLEMENTARY_GROUPS`.",
        "- **ΔRankIC ≈ 0** (no symbol): the group is essentially noise; "
        "dropping it costs nothing but does not help either. Phase B.2 "
        "should 24-split confirm before keeping it.",
        "- **ΔRankIC < −0.005** (`⚠️`): dropping the group hurt. The group "
        "carries real signal — keep.",
        "",
        "All ΔRankIC values are on a 6-split FAST screen. A larger 24-split "
        "is required before changing PRODUCTION_SUPPLEMENTARY_GROUPS for "
        "real (Phase B.2).",
    ])
    return "\n".join(lines)


def _fmt(v, dp=4):
    if v is None:
        return "—"
    try:
        return f"{float(v):+.{dp}f}" if isinstance(v, (int, float)) else str(v)
    except Exception:
        return str(v)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-config", default="6split")
    p.add_argument("--data-end", default="2026-05-19")
    p.add_argument("--code-commit", default=None,
                    help="Optional commit prefix filter.")
    p.add_argument("--markdown", default=None,
                    help="Optional path to write the markdown report.")
    args = p.parse_args()

    rows = filter_runs(
        split_config=args.split_config,
        data_end=args.data_end,
        code_commit=args.code_commit,
    )
    baseline, loo = _split_baseline_vs_loo(rows)
    if baseline is None:
        print(f"No baseline (empty dropped_groups) row found for split={args.split_config} "
              f"end={args.data_end}. Run the wrapper first.")
        sys.exit(1)
    if not loo:
        print(f"No LOO rows yet. Baseline at exp_id="
              f"{baseline.get('experiment_id', '?')}.")
        sys.exit(0)

    # cx review 2026-06-06 (P1): refuse to print a "complete" report
    # when the LOO row set is shorter than the ablation wrapper's group
    # list. A half-finished sweep can ship a wrong "drop this group"
    # conclusion. Use the SC-A3 production tier set as the canonical
    # expected group list. Override via env var when the operator
    # genuinely runs a narrower sweep.
    expected_groups_env = os.environ.get("LOO_EXPECTED_GROUPS", "").strip()
    if expected_groups_env:
        expected_groups = set(g.strip() for g in expected_groups_env.split(",") if g.strip())
    else:
        # Match the wrapper's LOO_GROUPS list. Hardcoded here to avoid
        # importing a bash array.
        expected_groups = {
            "capital_flow", "macro_zero_baseline", "shareholder",
            "valuation", "quality", "st_daily_basic", "st_moneyflow",
            "st_holder_number", "cross_market_regime",
        }
    seen_groups = {r["dropped_groups"][0] for r in loo if r.get("dropped_groups")}
    missing = sorted(expected_groups - seen_groups)
    if missing:
        print(
            f"PARTIAL SWEEP — {len(seen_groups)}/{len(expected_groups)} LOO "
            f"rows found. Missing groups: {missing}. The report cannot rank "
            f"net-negative loaders honestly until the sweep completes. "
            f"Set LOO_EXPECTED_GROUPS=<csv> to override if this is a "
            f"deliberate narrow sweep."
        )
        sys.exit(2)

    md = _table_md(baseline, loo)
    print(md)
    if args.markdown:
        out = Path(args.markdown).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        print(f"\nMarkdown report written: {out}")


if __name__ == "__main__":
    main()
