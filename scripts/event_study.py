"""Ex-post event study for LLM / chain / policy event factors (PE-6, task #145).

Motivation
----------
B.7 ablation (docs/phase_b7_verdict_20260607.md, commit 5ed34ee) found
chain factors at <0.01% density did not move xgb_209. That's a verdict
about the *model*, not about the signal. Before we throw away an event
stream entirely we need an offline tool that asks:

    "When an event of type X fires on date D, what does the average
     excess return curve from D-5 to D+5 look like? Is it statistically
     different from zero on D or D+1?"

This is the classical event study from finance. It validates the
signal independent of any downstream model.

CLI
---
::

    python scripts/event_study.py \\
        --source {pe1,pe2,pe3,pe4,llm,chain_rule,chain_llm} \\
        --start YYYY-MM-DD --end YYYY-MM-DD \\
        [--window -5,5] [--benchmark sh000300] [--top-n 20] \\
        [--out-dir data/storage/event_study]

Behavior per source
-------------------
- ``pe1`` (PBC monetary policy)        — market-keyed
- ``pe2`` (State Council / industries) — industry-keyed (theme broadcast)
- ``pe3`` (NBS macro)                  — market-keyed
- ``pe4`` (Xinwen Lianbo themes)       — theme-keyed (basket broadcast)
- ``llm`` (LLM company events)         — stock-keyed (qlib_code)
- ``chain_rule`` (rule-based chain)    — market-keyed (no A-share attribution)
- ``chain_llm``  (LLM chain)           — market-keyed (no A-share attribution)

For stock-keyed events we compute the per-stock excess return relative
to the benchmark over [D + offset_lo, D + offset_hi]. For market /
theme / industry-keyed events we use the benchmark return itself as the
"stock" return, so the study answers "did the market move after the
event" rather than picking specific names.

Outputs
-------
``<out_dir>/<source>_<start>_<end>.csv``
    Long-form DataFrame ``(event_id, event_date, instrument, event_type,
    offset_-5, ..., offset_+5)``.
``<out_dir>/<source>_<start>_<end>.png``
    Matplotlib figure: mean ± std band per event_type across offsets.
``<out_dir>/<source>_<start>_<end>.summary.json``
    Per-event-type aggregates: n, mean and t-stat / p-value of the
    abnormal return at offset=0 and offset=+1.

No model inference — this is purely a return-curve tool.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


SUPPORTED_SOURCES: tuple[str, ...] = (
    "pe1", "pe2", "pe3", "pe4", "llm", "chain_rule", "chain_llm",
)
DEFAULT_BENCHMARK = "sh000300"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "storage" / "event_study"
DATA_DIR = PROJECT_ROOT / "data" / "storage"

# Synthetic instrument code used when an event has no per-stock
# attribution. The benchmark return is used as the "stock" return for
# these events (see build_excess_return_panel).
MARKET_INSTRUMENT = "MARKET"

# Per-source on-disk layout. Each entry is (subdir under data/storage,
# date field name, instrument-key resolver, attribution kind).
#
# attribution_kind:
#   "stock"  — event keyed by a real qlib instrument
#   "market" — event keyed at the market level (use benchmark return)
#   "theme"  — event keyed by THEME_<UPPER>; broadcast via baskets
#              (step 4 wires that in; loader returns the raw THEME row)
SOURCE_SPEC: dict[str, dict] = {
    "pe1": {
        "subdir": "policy_events/pbc",
        "date_field": "publish_date",
        "event_type_field": "policy_stance",
        "instrument_kind": "market",
    },
    "pe2": {
        "subdir": "policy_events/state_council",
        "date_field": "publish_date",
        "event_type_field": "policy_direction",
        "instrument_kind": "market",
    },
    "pe3": {
        "subdir": "policy_events/nbs",
        "date_field": "publish_date",
        "event_type_field": "series_name",
        "instrument_kind": "market",
    },
    "pe4": {
        "subdir": "policy_events/xinwen_lianbo",
        "date_field": "publish_date",
        "event_type_field": "themes",
        "instrument_kind": "theme",
    },
    "llm": {
        "subdir": "llm_events_v2",
        "date_field": "extract_date",
        "event_type_field": "event_type",
        "instrument_kind": "stock",
    },
    "chain_rule": {
        "subdir": "global_chain_events",
        "date_field": "date",
        "event_type_field": "event_type",
        "instrument_kind": "market",
    },
    "chain_llm": {
        "subdir": "global_chain_events_llm",
        "date_field": "date",
        "event_type_field": "event_type",
        "instrument_kind": "market",
    },
}


@dataclass(frozen=True)
class EventStudyConfig:
    """Resolved CLI options for one event-study run."""

    source: str
    start: str
    end: str
    window_lo: int
    window_hi: int
    benchmark: str
    top_n: int | None
    out_dir: Path


def parse_window(spec: str) -> tuple[int, int]:
    """Parse ``"-5,5"`` style window spec into (lo, hi).

    Both bounds are inclusive; lo must be <= hi.
    """
    try:
        lo_str, hi_str = spec.split(",")
        lo = int(lo_str.strip())
        hi = int(hi_str.strip())
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"--window must be 'lo,hi' integers (got {spec!r})"
        ) from exc
    if lo > hi:
        raise ValueError(f"--window lo ({lo}) must be <= hi ({hi})")
    return lo, hi


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ex-post event study: average excess-return curve per event "
            "type across [-N, +M] trading days around the event date."
        ),
    )
    parser.add_argument(
        "--source", required=True, choices=list(SUPPORTED_SOURCES),
        help="Event source.",
    )
    parser.add_argument(
        "--start", required=True,
        help="Inclusive start date (YYYY-MM-DD) of the event window.",
    )
    parser.add_argument(
        "--end", required=True,
        help="Inclusive end date (YYYY-MM-DD) of the event window.",
    )
    parser.add_argument(
        "--window", default="-5,5",
        help=(
            "Offset window 'lo,hi' in trading days around event_date "
            "(default '-5,5'). Both bounds inclusive."
        ),
    )
    parser.add_argument(
        "--benchmark", default=DEFAULT_BENCHMARK,
        help=(
            "Qlib instrument code for the benchmark "
            f"(default {DEFAULT_BENCHMARK})."
        ),
    )
    parser.add_argument(
        "--top-n", type=int, default=None,
        help=(
            "If set, cap the number of events per event_type "
            "(useful for very dense LLM streams)."
        ),
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR),
        help=f"Output directory (default {DEFAULT_OUT_DIR}).",
    )
    return parser


EVENT_OUTPUT_COLUMNS = (
    "event_id", "event_date", "instrument", "event_type",
)


def _coerce_instrument_from_row(
    row: dict,
    instrument_kind: str,
) -> str | None:
    """Resolve the qlib-style instrument code for a single raw event row.

    Returns None if the row cannot be attributed (e.g. an LLM event with
    no qlib_code). The caller drops such rows from the panel.
    """
    if instrument_kind == "market":
        return MARKET_INSTRUMENT
    if instrument_kind == "stock":
        # LLM events carry qlib_code = "sh600519". Some legacy rows only
        # have stock_code = "600519" — derive the prefix from the first
        # digit when needed (6 → sh, 0/3 → sz, 8/4 → bj).
        q = row.get("qlib_code")
        if isinstance(q, str) and q.strip():
            return q.strip().upper()
        c = row.get("stock_code")
        if isinstance(c, str) and c.strip():
            c = c.strip()
            if c[0] == "6":
                return f"SH{c}"
            if c[0] in ("0", "3"):
                return f"SZ{c}"
            if c[0] in ("8", "4"):
                return f"BJ{c}"
        return None
    if instrument_kind == "theme":
        # XWLB events store a list under "themes"; one event row per
        # theme (we'll explode at the caller).
        theme = row.get("_theme_one")  # set by the explode pass
        if isinstance(theme, str) and theme.strip():
            return f"THEME_{theme.strip().upper()}"
        return None
    raise ValueError(f"unknown instrument_kind {instrument_kind!r}")


def _iter_jsonl_rows(events_root: Path):
    """Yield (path, parsed_dict) for every JSONL file under ``events_root``.

    Silently skips unreadable files / malformed lines so a single bad
    row doesn't poison the whole study.
    """
    if not events_root.exists():
        return
    for fp in sorted(events_root.glob("*.jsonl")):
        try:
            text = fp.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("event_study: cannot read %s (%s)", fp, exc)
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield fp, json.loads(line)
            except json.JSONDecodeError:
                continue


def load_events(
    *,
    source: str,
    start: str,
    end: str,
    events_root: Path | None = None,
) -> pd.DataFrame:
    """Load events for ``source`` whose date falls in [start, end].

    Returns a long-form DataFrame with columns
    ``event_id, event_date (datetime64[ns]), instrument, event_type``
    plus any source-specific extra columns (kept for downstream
    inspection / extension).

    For ``pe4`` (XWLB) the ``themes`` list is exploded: a single
    broadcast row with two themes becomes two event rows, each with
    instrument = ``THEME_<UPPER>``.

    Missing event directories return an empty frame with the standard
    columns (PIT-safe; never raises for missing data).
    """
    if source not in SOURCE_SPEC:
        raise ValueError(
            f"unknown source {source!r}; expected one of {SUPPORTED_SOURCES}"
        )
    spec = SOURCE_SPEC[source]
    root = events_root if events_root is not None else DATA_DIR / spec["subdir"]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError(f"end {end!r} must be >= start {start!r}")

    rows: list[dict] = []
    kind = spec["instrument_kind"]
    date_field = spec["date_field"]
    event_type_field = spec["event_type_field"]

    for fp, raw in _iter_jsonl_rows(Path(root)):
        date_raw = raw.get(date_field)
        date = pd.to_datetime(date_raw, errors="coerce")
        if pd.isna(date):
            continue
        if not (start_ts <= date <= end_ts):
            continue
        # Explode XWLB themes so each (date, theme) becomes its own row.
        if kind == "theme":
            themes = raw.get(event_type_field) or []
            if isinstance(themes, str):
                themes = [themes]
            if not isinstance(themes, list):
                continue
            for theme in themes:
                if not isinstance(theme, str) or not theme.strip():
                    continue
                row_copy = dict(raw)
                row_copy["_theme_one"] = theme.strip().lower()
                inst = _coerce_instrument_from_row(row_copy, kind)
                if inst is None:
                    continue
                rows.append({
                    "event_id": f"{fp.stem}:{len(rows)}",
                    "event_date": date.normalize(),
                    "instrument": inst,
                    "event_type": row_copy["_theme_one"],
                    "source_file": fp.name,
                })
            continue
        inst = _coerce_instrument_from_row(raw, kind)
        if inst is None:
            continue
        etype = raw.get(event_type_field) or "unknown"
        if isinstance(etype, list):  # defensive: pe3 series_name is scalar
            etype = etype[0] if etype else "unknown"
        rows.append({
            "event_id": f"{fp.stem}:{len(rows)}",
            "event_date": date.normalize(),
            "instrument": inst,
            "event_type": str(etype),
            "source_file": fp.name,
        })

    if not rows:
        return pd.DataFrame(columns=list(EVENT_OUTPUT_COLUMNS))
    df = pd.DataFrame(rows)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df.sort_values(["event_date", "instrument"]).reset_index(drop=True)


def parse_args(argv: list[str] | None = None) -> EventStudyConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    lo, hi = parse_window(args.window)
    return EventStudyConfig(
        source=args.source,
        start=args.start,
        end=args.end,
        window_lo=lo,
        window_hi=hi,
        benchmark=args.benchmark,
        top_n=args.top_n,
        out_dir=Path(args.out_dir),
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    cfg = parse_args(argv)
    # Step 1: only the CLI shape is implemented; later steps fill in
    # event loading, return alignment, aggregation, plotting.
    logger.info("PE-6 event_study config: %s", cfg)
    raise NotImplementedError(
        "PE-6 step 1 stub — event loader + return alignment land in "
        "subsequent commits."
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
