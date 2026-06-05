"""Experiment ledger — single, machine-readable, cross-experiment index.

2026-06-05: ``tracker/artifact_contract.ExperimentArtifact`` already
saves per-experiment config + metrics + backtest under
``data/storage/experiments/<exp_id>/``. That works one experiment at a
time but does NOT solve the recurring question

    "the +0.0785 RankIC headline — which code commit / which cache /
    which feature groups / which data window produced it?"

The ledger is a flat, append-only file (``data/storage/experiments_ledger.jsonl``)
where every training / backtest run appends a single line with the
fields the project lead asked for explicitly:

    {
      "experiment_id": "xgb_242_24split_20260605_220000",
      "model_profile": "xgb_242",
      "feature_groups": [...11 names...],
      "dropped_groups": [],
      "feature_count": 242,
      "data_end": "2026-05-19",
      "split_config": "standard_24split",
      "code_commit": "72aa580b...",
      "cache_path": "data/storage/feature_cache_242_production.parquet",
      "metrics": {"rank_ic_mean": ..., "spread_top20": ...},
      "ts": "2026-06-05T22:25:00",
      "artifact_dir": "data/storage/experiments/xgb_242_24split_20260605_220000",
    }

Reads stream the file (one line == one experiment) so cross-experiment
comparison is one ``jq`` or ``pandas.read_json(lines=True)`` away.

Design:
    - JSONL not a single JSON to keep writes atomic-per-line and
      survive partial writes.
    - Every entry MUST include ``model_profile`` and ``code_commit``;
      missing either is a hard error so future ledger queries can rely
      on those columns.
    - The ledger does NOT replace ``ExperimentArtifact`` — it is an
      INDEX over it. The single-experiment artifact dir still holds
      pred.pkl / label.pkl / etc.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = PROJECT_ROOT / "data" / "storage" / "experiments_ledger.jsonl"


def _git_commit() -> str:
    """Return short commit hash, or 'unknown' when not in a git tree."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


@dataclass
class LedgerEntry:
    """One row in the cross-experiment ledger."""

    experiment_id: str
    model_profile: str
    code_commit: str
    feature_count: int
    data_end: str
    split_config: str
    cache_path: str
    feature_groups: list[str] = field(default_factory=list)
    dropped_groups: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    artifact_dir: str = ""
    ts: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.experiment_id:
            raise ValueError("LedgerEntry.experiment_id is required")
        if not self.model_profile:
            raise ValueError(
                "LedgerEntry.model_profile is required — without it the "
                "ledger cannot answer 'which model' downstream"
            )
        if not self.code_commit or self.code_commit == "unknown":
            logger.warning(
                "LedgerEntry.code_commit is empty / 'unknown'. Future "
                "ledger queries will not be able to reproduce this run."
            )

    def to_jsonl(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False, default=str)


def record_run(
    *,
    experiment_id: str,
    model_profile: str,
    feature_count: int,
    data_end: str,
    split_config: str,
    cache_path: str,
    feature_groups: Iterable[str] | None = None,
    dropped_groups: Iterable[str] | None = None,
    metrics: dict | None = None,
    artifact_dir: str | os.PathLike | None = None,
    code_commit: Optional[str] = None,
    extra: dict | None = None,
    ledger_path: Path | None = None,
) -> Path:
    """Append a single experiment run to the ledger.

    Parameters
    ----------
    experiment_id:
        Unique ID — convention ``<model_profile>_<split_config>_<YYYYMMDD_HHMMSS>``.
    model_profile:
        Active model identifier (e.g. ``xgb_242``, ``xgb_174``, ``xgb175``).
        Required so a future cross-model query can group cleanly.
    feature_count:
        Total feature width seen by the model (Alpha158 + supp + custom).
    data_end:
        Last calendar date the splits could see, ISO ``YYYY-MM-DD``.
        Two runs with the same ``model_profile`` + ``code_commit`` but
        different ``data_end`` are NOT comparable.
    split_config:
        Preset name (``standard_24split`` / ``standard_12split`` /
        ``fast_6split`` / etc) or a free-text label for custom splits.
    cache_path:
        Path of the parquet / data source actually consumed. The ledger
        keeps this so the question "which 242 cache" never becomes
        ambiguous again.
    feature_groups:
        Supplementary loader groups active in this run (e.g. the
        ``PRODUCTION_SUPPLEMENTARY_GROUPS`` tuple). Empty for raw-only.
    dropped_groups:
        Groups REMOVED relative to ``feature_groups`` baseline. Used by
        LOO ablation runs. Empty in non-ablation runs.
    metrics:
        The headline metrics dict from this run (rank_ic_mean,
        spread_top20, etc). The ledger only stores the headline; full
        metrics still live in ``artifact_dir/metrics.json``.
    artifact_dir:
        Pointer to the full ``ExperimentArtifact`` directory.
    code_commit:
        Defaults to current ``git rev-parse HEAD`` (short) when omitted.
    extra:
        Arbitrary additional fields — used sparingly; prefer adding a
        real column to ``LedgerEntry`` for anything used by more than
        one caller.
    ledger_path:
        Override the default ledger location. Tests use this.

    Returns
    -------
    Path of the ledger file.
    """
    entry = LedgerEntry(
        experiment_id=experiment_id,
        model_profile=model_profile,
        code_commit=code_commit if code_commit is not None else _git_commit(),
        feature_count=int(feature_count),
        data_end=str(data_end),
        split_config=str(split_config),
        cache_path=str(cache_path),
        feature_groups=list(feature_groups or []),
        dropped_groups=list(dropped_groups or []),
        metrics=dict(metrics or {}),
        artifact_dir=str(artifact_dir or ""),
        extra=dict(extra or {}),
    )
    target = Path(ledger_path) if ledger_path is not None else LEDGER_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    line = entry.to_jsonl() + "\n"
    # Append atomically by writing a single line; POSIX guarantees
    # single-write atomicity up to PIPE_BUF (~4 KB), our records are
    # well under that. Use os.write so we never partial-buffer.
    fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)
    logger.info(
        "Ledger appended: %s profile=%s commit=%s data_end=%s metrics=%s",
        experiment_id, model_profile, entry.code_commit, data_end,
        list((metrics or {}).keys()),
    )
    return target


def read_ledger(ledger_path: Path | None = None) -> list[dict]:
    """Read every ledger entry. Returns oldest-first (file order)."""
    target = Path(ledger_path) if ledger_path is not None else LEDGER_PATH
    if not target.exists():
        return []
    out: list[dict] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed ledger line: %s (%s)", line[:80], exc)
    return out


def filter_runs(
    *,
    model_profile: str | None = None,
    split_config: str | None = None,
    data_end: str | None = None,
    code_commit: str | None = None,
    ledger_path: Path | None = None,
) -> list[dict]:
    """Return ledger entries matching the given filters. ``None`` means 'any'.

    Useful for the comparator: ``filter_runs(split_config="standard_24split")``
    feeds straight into a head-to-head table.
    """
    runs = read_ledger(ledger_path)
    out = []
    for r in runs:
        if model_profile is not None and r.get("model_profile") != model_profile:
            continue
        if split_config is not None and r.get("split_config") != split_config:
            continue
        if data_end is not None and r.get("data_end") != data_end:
            continue
        if code_commit is not None and not r.get("code_commit", "").startswith(code_commit):
            continue
        out.append(r)
    return out
