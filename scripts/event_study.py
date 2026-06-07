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
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


SUPPORTED_SOURCES: tuple[str, ...] = (
    "pe1", "pe2", "pe3", "pe4", "llm", "chain_rule", "chain_llm",
)
DEFAULT_BENCHMARK = "sh000300"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "storage" / "event_study"


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
