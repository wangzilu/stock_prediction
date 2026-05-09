"""Validate local Qlib daily data before training or inference.

This checks the file-level invariants that Qlib relies on:

- calendar exists and is non-empty
- feature bins use Qlib's `[start_index, values...]` format
- required OHLCV fields exist for enough instruments
- latest N calendar rows have acceptable close coverage

Usage:
    python scripts/check_qlib_data_health.py
    python scripts/check_qlib_data_health.py --qlib-dir data/storage/qlib_data/cn_data
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QLIB_DIR = PROJECT_ROOT / "data" / "storage" / "qlib_data" / "cn_data"
REQUIRED_FIELDS = ("open", "high", "low", "close", "volume", "amount")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class HealthReport:
    ok: bool
    qlib_dir: str
    calendar_count: int = 0
    latest_calendar_date: str = ""
    min_instruments: int = 0
    instruments_checked: int = 0
    instruments_with_required_fields: int = 0
    latest_close_coverage: float = 0.0
    malformed_bins: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    field_span_mismatches: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _read_calendar(qlib_dir: Path) -> list[str]:
    calendar_file = qlib_dir / "calendars" / "day.txt"
    if not calendar_file.exists():
        return []
    return [line.strip() for line in calendar_file.read_text().splitlines() if line.strip()]


def _normalize_instrument(code: str) -> str:
    code = code.strip()
    if not code:
        return code
    return code.lower()


def _load_instruments(qlib_dir: Path, universe: str) -> list[str]:
    inst_dir = qlib_dir / "instruments"
    candidates: list[str] = []

    files: list[Path]
    if universe == "csi800":
        files = [inst_dir / "csi300.txt", inst_dir / "csi500.txt"]
    elif universe == "all":
        files = [inst_dir / "all.txt"]
    else:
        files = [inst_dir / f"{universe}.txt"]

    for file_path in files:
        if not file_path.exists():
            continue
        for line in file_path.read_text().splitlines():
            parts = line.split()
            if parts:
                candidates.append(_normalize_instrument(parts[0]))

    if not candidates:
        feature_dir = qlib_dir / "features"
        candidates = sorted(p.name.lower() for p in feature_dir.iterdir() if p.is_dir()) if feature_dir.exists() else []

    return sorted(set(candidates))


def _feature_dir(qlib_dir: Path, instrument: str) -> Path:
    lower = qlib_dir / "features" / instrument.lower()
    upper = qlib_dir / "features" / instrument.upper()
    if lower.exists():
        return lower
    return upper


def _read_feature_bin(path: Path) -> np.ndarray:
    if not path.exists():
        return np.array([], dtype="<f4")
    return np.fromfile(path, dtype="<f4")


def _is_valid_start_index(value: float, calendar_count: int) -> bool:
    if not np.isfinite(value):
        return False
    index = int(value)
    return float(index) == float(value) and 0 <= index < calendar_count


def _bin_is_well_formed(arr: np.ndarray, calendar_count: int) -> bool:
    if arr.size < 2:
        return False
    if not _is_valid_start_index(float(arr[0]), calendar_count):
        return False
    start_index = int(arr[0])
    values_len = arr.size - 1
    return start_index + values_len <= calendar_count


def _bin_span(arr: np.ndarray, calendar_count: int) -> tuple[int, int] | None:
    if not _bin_is_well_formed(arr, calendar_count):
        return None
    start_index = int(arr[0])
    return start_index, start_index + arr.size - 2


def _latest_window_has_value(arr: np.ndarray, calendar_count: int, lookback_days: int) -> bool:
    if not _bin_is_well_formed(arr, calendar_count):
        return False
    start_index = int(arr[0])
    values = arr[1:]
    latest_start = max(0, calendar_count - lookback_days)
    rel_start = max(0, latest_start - start_index)
    if rel_start >= values.size:
        return False
    return bool(np.isfinite(values[rel_start:]).any())


def check_qlib_dir(
    qlib_dir: Path,
    universe: str = "csi800",
    min_coverage: float = 0.95,
    min_instruments: int = 0,
    lookback_days: int = 10,
    max_malformed: int = 20,
    max_missing: int = 20,
    max_calendar_age_days: int = 10,
) -> HealthReport:
    qlib_dir = qlib_dir.resolve()
    calendar = _read_calendar(qlib_dir)
    report = HealthReport(
        ok=False,
        qlib_dir=str(qlib_dir),
        calendar_count=len(calendar),
        min_instruments=min_instruments,
    )

    if not calendar:
        report.errors.append("calendar file is missing or empty")
        return report

    report.latest_calendar_date = calendar[-1]
    try:
        latest_dt = datetime.strptime(calendar[-1], "%Y-%m-%d").date()
        age_days = (datetime.now().date() - latest_dt).days
        if age_days > max_calendar_age_days:
            report.warnings.append(
                f"latest calendar date {calendar[-1]} is {age_days} calendar days old"
            )
    except ValueError:
        report.errors.append(f"invalid latest calendar date: {calendar[-1]}")

    instruments = _load_instruments(qlib_dir, universe)
    report.instruments_checked = len(instruments)
    if not instruments:
        report.errors.append(f"no instruments found for universe={universe}")
        return report
    if len(instruments) < min_instruments:
        report.errors.append(
            f"instrument count {len(instruments)} < required {min_instruments} for universe={universe}"
        )

    latest_close_ok = 0
    required_ok = 0

    for instrument in instruments:
        inst_dir = _feature_dir(qlib_dir, instrument)
        has_all_fields = True
        field_spans: dict[str, tuple[int, int]] = {}

        for field_name in REQUIRED_FIELDS:
            path = inst_dir / f"{field_name}.day.bin"
            if not path.exists():
                has_all_fields = False
                if len(report.missing_fields) < max_missing:
                    report.missing_fields.append(f"{instrument}/{field_name}")
                continue

            arr = _read_feature_bin(path)
            if not _bin_is_well_formed(arr, len(calendar)):
                has_all_fields = False
                if len(report.malformed_bins) < max_malformed:
                    first = arr[0].item() if arr.size else None
                    report.malformed_bins.append(f"{instrument}/{field_name}: first={first}")
            else:
                span = _bin_span(arr, len(calendar))
                if span is not None:
                    field_spans[field_name] = span

        if len(field_spans) == len(REQUIRED_FIELDS) and len(set(field_spans.values())) > 1:
            has_all_fields = False
            if len(report.field_span_mismatches) < max_malformed:
                report.field_span_mismatches.append(f"{instrument}: {field_spans}")

        close_arr = _read_feature_bin(inst_dir / "close.day.bin")
        if _latest_window_has_value(close_arr, len(calendar), lookback_days):
            latest_close_ok += 1
        if has_all_fields:
            required_ok += 1

    report.instruments_with_required_fields = required_ok
    report.latest_close_coverage = latest_close_ok / len(instruments)

    if report.malformed_bins:
        report.errors.append(
            f"found malformed feature bins; examples={report.malformed_bins[:3]}"
        )
    if report.field_span_mismatches:
        report.errors.append(
            f"found required-field span mismatches; examples={report.field_span_mismatches[:3]}"
        )
    if report.latest_close_coverage < min_coverage:
        report.errors.append(
            f"latest close coverage {report.latest_close_coverage:.1%} < {min_coverage:.1%}"
        )
    if required_ok / len(instruments) < min_coverage:
        report.errors.append(
            f"required field coverage {required_ok / len(instruments):.1%} < {min_coverage:.1%}"
        )

    report.ok = not report.errors
    return report


def _print_report(report: HealthReport, as_json: bool) -> None:
    payload = asdict(report)
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    level = logger.info if report.ok else logger.error
    level(
        "Qlib health: ok=%s instruments=%s required_ok=%s latest_close_coverage=%.1f%% latest=%s",
        report.ok,
        report.instruments_checked,
        report.instruments_with_required_fields,
        report.latest_close_coverage * 100,
        report.latest_calendar_date,
    )
    for warning in report.warnings:
        logger.warning(warning)
    for error in report.errors:
        logger.error(error)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qlib-dir", type=Path, default=DEFAULT_QLIB_DIR)
    parser.add_argument("--universe", default="csi800")
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--min-instruments", type=int, default=0)
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--max-calendar-age-days", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_qlib_dir(
        qlib_dir=args.qlib_dir,
        universe=args.universe,
        min_coverage=args.min_coverage,
        min_instruments=args.min_instruments,
        lookback_days=args.lookback_days,
        max_calendar_age_days=args.max_calendar_age_days,
    )
    _print_report(report, args.json)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
