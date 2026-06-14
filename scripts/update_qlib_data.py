"""Update Qlib data with latest A-share daily bars.

Default mode is incremental:

- infer each symbol's next required date from update_manifest or existing bins
- fetch only missing recent rows
- write to a staging Qlib directory
- repair legacy malformed bins when possible
- promote staging only after health checks pass

Full rebuild remains available with `--full`.

Usage:
    python scripts/update_qlib_data.py
    python scripts/update_qlib_data.py --repair-only
    python scripts/update_qlib_data.py --full --provider baostock
    python scripts/update_qlib_data.py --provider tushare
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@contextmanager
def progress_bar(total: int, desc: str, unit: str) -> Iterator[object | None]:
    """Show tqdm progress only for interactive terminal runs."""
    disabled = os.environ.get("UPDATE_QLIB_PROGRESS", "").lower() in {"0", "false", "no"}
    if disabled or not sys.stderr.isatty():
        yield None
        return
    try:
        from tqdm import tqdm
    except Exception:
        yield None
        return

    with tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True) as bar:
        yield bar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scheduler.data_health import write_health, HealthStatus  # noqa: E402
from scripts.check_qlib_data_health import check_qlib_dir  # noqa: E402
from config.settings import (  # noqa: E402
    QLIB_DATA_PROVIDER,
    QLIB_UNIVERSE_SOURCE,
    LGB_INFERENCE_UNIVERSE,
    LGB_MIN_DATA_INSTRUMENTS,
    LGB_MIN_PREDICTIONS,
    LGB_MODEL_PATH,
)


DATA_DIR = PROJECT_ROOT / "data" / "storage"
QLIB_DIR = DATA_DIR / "qlib_data" / "cn_data"
STAGING_QLIB_DIR = DATA_DIR / "qlib_data_staging" / "cn_data"
MANIFEST_PATH = DATA_DIR / "update_manifest.json"

QLIB_COLUMNS = {
    "open": "$open",
    "high": "$high",
    "low": "$low",
    "close": "$close",
    "volume": "$volume",
    "amount": "$amount",
    "turn": "$turn",
    "peTTM": "$pe",
    "pbMRQ": "$pb",
    "factor": "$factor",
}
REQUIRED_OUTPUT_COLUMNS = ["$open", "$high", "$low", "$close", "$volume", "$amount"]


@dataclass
class UniverseSelection:
    codes: set[str]
    groups: dict[str, set[str]] = field(default_factory=dict)
    source: str = "instruments"


def bs_to_qlib_code(bs_code: str) -> str:
    prefix, num = bs_code.split(".")
    return f"{prefix.lower()}{num}"


def qlib_to_bs_code(qlib_code: str) -> str:
    code = qlib_code.strip()
    prefix = code[:2].lower()
    num = code[-6:]
    return f"{prefix}.{num}"


def ts_code_to_bs_code(ts_code: str) -> str:
    num, suffix = ts_code.split(".")
    return f"{suffix.lower()}.{num}"


def bs_code_to_ts_code(bs_code: str) -> str:
    prefix, num = bs_code.split(".")
    return f"{num}.{prefix.upper()}"


def numeric_to_bs_code(code: str) -> str:
    code = str(code).strip().zfill(6)
    if code.startswith(("8", "4")):
        prefix = "bj"
    elif code.startswith(("6", "9")):
        prefix = "sh"
    else:
        prefix = "sz"
    return f"{prefix}.{code}"


def is_a_share_stock_code(bs_code: str) -> bool:
    code = bs_code.strip().lower()
    if len(code) != 9 or code[2] != ".":
        return False
    prefix, num = code.split(".")
    if not num.isdigit() or len(num) != 6:
        return False
    if prefix == "sh":
        return num.startswith(("60", "68"))
    if prefix == "sz":
        return num.startswith(("00", "30"))
    if prefix == "bj":
        return num.startswith(("4", "8"))
    return False


def _run_with_alarm(seconds: int, func, *, label: str):
    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Timeout while {label}")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        return func()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def read_calendar(qlib_dir: Path) -> list[str]:
    calendar_file = qlib_dir / "calendars" / "day.txt"
    if not calendar_file.exists():
        return []
    return [line.strip() for line in calendar_file.read_text().splitlines() if line.strip()]


def write_calendar(qlib_dir: Path, calendar_dates: list[str]) -> None:
    calendar_file = qlib_dir / "calendars" / "day.txt"
    calendar_file.parent.mkdir(parents=True, exist_ok=True)
    calendar_file.write_text("\n".join(calendar_dates) + "\n")


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    if not path.exists():
        return {"version": 1, "symbols": {}}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Manifest is invalid JSON; starting fresh: %s", path)
        return {"version": 1, "symbols": {}}


def save_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def load_codes_from_instruments(qlib_dir: Path, universe: str) -> set[str]:
    inst_dir = qlib_dir / "instruments"
    files: list[Path]
    if universe == "csi800":
        files = [inst_dir / "csi300.txt", inst_dir / "csi500.txt"]
    elif universe == "all":
        files = [inst_dir / "all.txt"]
    else:
        files = [inst_dir / f"{universe}.txt"]

    codes: set[str] = set()
    for file_path in files:
        if not file_path.exists():
            continue
        for line in file_path.read_text().splitlines():
            parts = line.split()
            if parts:
                codes.add(qlib_to_bs_code(parts[0]))

    return codes


def load_universe_from_instruments(qlib_dir: Path, universe: str) -> UniverseSelection:
    groups: dict[str, set[str]] = {}
    if universe in ("csi300", "csi800"):
        groups["csi300"] = load_codes_from_instruments(qlib_dir, "csi300")
    if universe in ("csi500", "csi800"):
        groups["csi500"] = load_codes_from_instruments(qlib_dir, "csi500")
    if universe == "all":
        groups["all"] = load_codes_from_instruments(qlib_dir, "all")
    if universe not in ("csi300", "csi500", "csi800", "all"):
        groups[universe] = load_codes_from_instruments(qlib_dir, universe)

    codes: set[str] = set()
    for group_codes in groups.values():
        codes |= group_codes
    if codes:
        groups.setdefault("all", set(codes))
    return UniverseSelection(codes=codes, groups=groups, source="instruments")


def get_all_stock_codes(manage_session: bool = True) -> list[str]:
    """Get all active A-share stock codes from baostock."""
    import baostock as bs

    if manage_session:
        bs.login()
    try:
        def _query_all_stock() -> list[str]:
            rs = bs.query_all_stock(day=datetime.now().strftime("%Y-%m-%d"))
            codes = []
            while rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                code = row[0].strip().lower()
                trade_status = row[1] if len(row) > 1 else "1"
                if trade_status == "1" and is_a_share_stock_code(code):
                    codes.append(code)
            return codes

        codes = _run_with_alarm(
            150,
            _query_all_stock,
            label="querying baostock all-stock universe",
        )
        if len(codes) >= 4500:
            return codes

        logger.warning(
            "baostock query_all_stock only returned %s A-share codes; trying stock_basic",
            len(codes),
        )

        def _query_stock_basic() -> list[str]:
            rs = bs.query_stock_basic()
            basic_codes = []
            while rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                code = row[0].strip().lower()
                status = row[4] if len(row) > 4 else "1"
                if status == "1" and is_a_share_stock_code(code):
                    basic_codes.append(code)
            return basic_codes

        return _run_with_alarm(
            150,
            _query_stock_basic,
            label="querying baostock stock_basic universe",
        )
    finally:
        if manage_session:
            bs.logout()


def _baostock_index_codes(query_fn) -> set[str]:
    rs = query_fn()
    codes: set[str] = set()
    while rs.error_code == "0" and rs.next():
        row = rs.get_row_data()
        if len(row) >= 2:
            codes.add(row[1])
    return codes


def get_universe_from_baostock(universe: str) -> UniverseSelection:
    import baostock as bs

    bs.login()
    try:
        groups: dict[str, set[str]] = {}
        if universe == "all":
            groups["all"] = set(get_all_stock_codes(manage_session=False))
        if universe in ("csi300", "csi800"):
            groups["csi300"] = _baostock_index_codes(bs.query_hs300_stocks)
        if universe in ("csi500", "csi800"):
            groups["csi500"] = _baostock_index_codes(bs.query_zz500_stocks)

        codes: set[str] = set()
        for group_codes in groups.values():
            codes |= group_codes

        if universe == "csi800" and len(codes) < 500:
            logger.warning(
                "CSI300+500 only got %s stocks, falling back to all A-shares",
                len(codes),
            )
            groups = {"all": set(get_all_stock_codes(manage_session=False))}
            codes = set(groups["all"])

        groups.setdefault("all", set(codes))
        return UniverseSelection(codes=codes, groups=groups, source="baostock")
    finally:
        bs.logout()


def _akshare_index_cons(symbol: str) -> set[str]:
    import akshare as ak

    df = ak.index_stock_cons(symbol=symbol)
    if df is None or df.empty:
        return set()

    code_col = None
    for candidate in ("品种代码", "成分券代码", "证券代码", "code"):
        if candidate in df.columns:
            code_col = candidate
            break
    if code_col is None:
        # Prefer the first column that looks like a six-digit stock code.
        for col in df.columns:
            sample = df[col].astype(str).str.extract(r"(\d{6})", expand=False).dropna()
            if not sample.empty:
                code_col = col
                break
    if code_col is None:
        return set()

    codes = set()
    for value in df[code_col].astype(str):
        match = "".join(ch for ch in value if ch.isdigit())
        if len(match) >= 6:
            codes.add(numeric_to_bs_code(match[-6:]))
    return codes


def get_universe_from_akshare(universe: str) -> UniverseSelection:
    groups: dict[str, set[str]] = {}
    if universe in ("csi300", "csi800"):
        groups["csi300"] = _akshare_index_cons("000300")
    if universe in ("csi500", "csi800"):
        groups["csi500"] = _akshare_index_cons("000905")
    if universe == "all":
        import akshare as ak

        df = ak.stock_info_a_code_name()
        code_col = "code" if "code" in df.columns else df.columns[0]
        groups["all"] = {numeric_to_bs_code(code) for code in df[code_col].astype(str)}

    codes: set[str] = set()
    for group_codes in groups.values():
        codes |= group_codes
    groups.setdefault("all", set(codes))
    return UniverseSelection(codes=codes, groups=groups, source="akshare")


def get_external_universe(universe: str, source: str) -> UniverseSelection:
    if source in ("auto", "akshare"):
        try:
            selection = get_universe_from_akshare(universe)
            if selection.codes:
                return selection
        except Exception as exc:
            if source == "akshare":
                raise
            logger.warning("AKShare universe unavailable: %s", exc)

    if source in ("auto", "baostock"):
        return get_universe_from_baostock(universe)

    raise ValueError(f"Unknown universe source: {source}")


def get_update_universe(
    qlib_dir: Path,
    universe: str,
    refresh_universe: bool,
    min_universe_size: int,
    universe_source: str,
) -> UniverseSelection:
    selection = load_universe_from_instruments(qlib_dir, universe)
    if selection.codes and not refresh_universe and len(selection.codes) >= min_universe_size:
        logger.info("Loaded %s %s codes from instruments", len(selection.codes), universe)
        return selection

    if selection.codes and not refresh_universe:
        logger.warning(
            "Instrument universe has only %s codes (<%s); refreshing from external source",
            len(selection.codes),
            min_universe_size,
        )
    else:
        logger.info("Refreshing %s universe from %s", universe, universe_source)

    external = get_external_universe(universe, universe_source)
    if not external.codes:
        if selection.codes:
            logger.warning("External universe empty; falling back to existing instruments")
            return selection
        raise RuntimeError(f"No symbols found for universe={universe}")
    logger.info(
        "Loaded %s %s codes from %s",
        len(external.codes),
        universe,
        external.source,
    )
    return external


def _feature_dir(qlib_dir: Path, bs_code: str) -> Path:
    qlib_code = bs_to_qlib_code(bs_code)
    lower_dir = qlib_dir / "features" / qlib_code.lower()
    upper_dir = qlib_dir / "features" / qlib_code.upper()
    if lower_dir.exists():
        return lower_dir
    if upper_dir.exists():
        return upper_dir
    return lower_dir


def _valid_start_index(value: float, calendar_count: int, arr_size: int) -> bool:
    if not np.isfinite(value):
        return False
    index = int(value)
    return float(index) == float(value) and 0 <= index < calendar_count and index + arr_size - 1 <= calendar_count


def read_feature_series(
    qlib_dir: Path,
    bs_code: str,
    field_name: str,
    calendar_dates: list[str],
) -> pd.Series:
    """Read either proper Qlib bins or the legacy full-calendar bins this repo wrote."""
    path = _feature_dir(qlib_dir, bs_code) / f"{field_name}.day.bin"
    if not path.exists() or not calendar_dates:
        return pd.Series(dtype=np.float32)

    arr = np.fromfile(path, dtype="<f4")
    if arr.size == 0:
        return pd.Series(dtype=np.float32)

    if arr.size >= 2 and _valid_start_index(float(arr[0]), len(calendar_dates), arr.size):
        start_index = int(arr[0])
        values = arr[1:]
        dates = calendar_dates[start_index:start_index + len(values)]
    else:
        values = arr[:len(calendar_dates)]
        dates = calendar_dates[:len(values)]

    return pd.Series(values, index=pd.to_datetime(dates), dtype=np.float32)


def infer_last_feature_date(qlib_dir: Path, bs_code: str, calendar_dates: list[str]) -> str | None:
    series = read_feature_series(qlib_dir, bs_code, "close", calendar_dates)
    if series.empty:
        return None
    series = series[np.isfinite(series)]
    if series.empty:
        return None
    return series.index.max().strftime("%Y-%m-%d")


def infer_feature_date_range(
    qlib_dir: Path,
    bs_code: str,
    calendar_dates: list[str],
) -> tuple[str, str] | None:
    series = read_feature_series(qlib_dir, bs_code, "close", calendar_dates)
    if series.empty:
        return None
    series = series[np.isfinite(series)]
    if series.empty:
        return None
    return (
        series.index.min().strftime("%Y-%m-%d"),
        series.index.max().strftime("%Y-%m-%d"),
    )


def next_date(date_str: str) -> str:
    return (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")


def build_start_dates(
    codes: Iterable[str],
    mode: str,
    end_date: str,
    full_start_date: str,
    incremental_lookback_days: int,
    new_symbol_days: int,
    qlib_dir: Path,
    calendar_dates: list[str],
    manifest: dict,
) -> dict[str, str]:
    starts: dict[str, str] = {}
    manifest_symbols = manifest.setdefault("symbols", {})

    if mode == "full":
        return {code: full_start_date for code in codes}

    fallback_recent = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=incremental_lookback_days)).strftime("%Y-%m-%d")
    fallback_new = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=new_symbol_days)).strftime("%Y-%m-%d")

    for code in codes:
        symbol_state = manifest_symbols.get(code, {})
        local_last_success = infer_last_feature_date(qlib_dir, code, calendar_dates)
        manifest_last_success = symbol_state.get("last_success_date")
        if local_last_success:
            last_success = local_last_success
        else:
            # The Qlib feature files are the source of truth.  A previous
            # failed staging run may have recorded manifest progress without
            # promoting those bins, so never let manifest-only progress skip a
            # missing local instrument.
            last_success = manifest_last_success if _feature_dir(qlib_dir, code).exists() else None

        if last_success:
            start_date = next_date(last_success)
            # Re-fetch a small overlap to repair late adjustments and partial bars.
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            overlap_dt = datetime.strptime(fallback_recent, "%Y-%m-%d")
            start_date = min(start_dt, overlap_dt).strftime("%Y-%m-%d")
        else:
            start_date = fallback_new

        if start_date <= end_date:
            starts[code] = start_date

    return starts


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df = df.sort_index()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    return df.rename(columns=QLIB_COLUMNS)


def fetch_stock_data_baostock(bs_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    import baostock as bs

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Timeout fetching {bs_code}")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(30)
    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount,turn,peTTM,pbMRQ",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2",
        )
        if rs.error_code != "0":
            raise RuntimeError(f"{rs.error_code}: {rs.error_msg}")
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "turn",
            "peTTM",
            "pbMRQ",
        ],
    )
    return normalize_frame(df)


def fetch_with_baostock(
    start_by_code: dict[str, str],
    end_date: str,
    cap_lookback_days: int | None = 30,
) -> dict[str, pd.DataFrame]:
    import baostock as bs

    # 2026-06-13 root-cause fix: 324 / 5527 stocks were `missing` in
    # the manifest (no last_success_date) — usually because manifest
    # was reset or a stock just got listed. build_start_dates then
    # defaulted them to `end_date - new_symbol_days (=365)`. baostock
    # is per-stock so it dutifully tried to pull a full year for each
    # of those 324, taking ~9h cumulatively vs the cron's 7200s budget.
    # Cap start_date at end_date - cap_lookback_days per stock here too.
    # Stocks that truly need deeper history get repaired by the weekly
    # ``--full`` run (Sat 04:00) — caller passes cap_lookback_days=None
    # from main() in that mode so the cap is skipped. Daily cron keeps
    # the already-warm 5198 stocks (last_success=06-09) only needing
    # 3-4 business days each — sub-30-minute total.
    # cx review 2026-06-13 C1 fix: cap is now opt-in via parameter so
    # ``--full --provider baostock`` actually returns full history.
    if cap_lookback_days is not None:
        cap_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=cap_lookback_days)
        cap_str = cap_dt.strftime("%Y-%m-%d")
        capped_count = 0
        capped_starts: dict[str, str] = {}
        for code, start in start_by_code.items():
            if start < cap_str:
                capped_starts[code] = cap_str
                capped_count += 1
            else:
                capped_starts[code] = start
        if capped_count > 0:
            logger.info(
                "baostock start-date clipped: %d/%d stocks had start_date "
                "older than %s — clipped to %s (deep history goes to --full).",
                capped_count, len(start_by_code), cap_str, cap_str,
            )
        start_by_code = capped_starts

    session_attempts = 0
    consecutive_errors = 0
    data: dict[str, pd.DataFrame] = {}

    def _login() -> None:
        login_result = bs.login()
        if login_result.error_code != "0":
            raise RuntimeError(f"baostock login failed: {login_result.error_msg}")

    def _logout() -> None:
        try:
            bs.logout()
        except Exception:
            pass

    def _reconnect(reason: str) -> None:
        nonlocal session_attempts, consecutive_errors
        logger.info("Reconnecting baostock session: %s", reason)
        _logout()
        time.sleep(2)
        _login()
        session_attempts = 0
        consecutive_errors = 0

    _login()
    try:
        total = len(start_by_code)
        with progress_bar(total, "baostock", "stock") as progress:
            for i, (code, start_date) in enumerate(sorted(start_by_code.items()), start=1):
                if session_attempts >= 80:
                    _reconnect("batch boundary")

                for attempt in range(2):
                    session_attempts += 1
                    try:
                        df = fetch_stock_data_baostock(code, start_date, end_date)
                        consecutive_errors = 0
                        if not df.empty:
                            data[code] = df
                        break
                    except Exception as exc:
                        consecutive_errors += 1
                        logger.warning(
                            "baostock failed for %s (attempt %s/2): %s",
                            code,
                            attempt + 1,
                            exc,
                        )
                        _reconnect(f"fetch failure for {code}")
                if consecutive_errors >= 5:
                    _reconnect("consecutive fetch failures")
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(ok=len(data), current=code)
                if i % 50 == 0:
                    logger.info("Fetched: %s/%s (%s ok)", i, total, len(data))
    finally:
        _logout()
    return data


def fetch_with_akshare(start_by_code: dict[str, str], end_date: str) -> dict[str, pd.DataFrame]:
    import akshare as ak
    import time as _time

    # 2026-06-14 fix A: 1-shot fail was too brittle. Today's 06-12 cron
    # observed AKShare returning RemoteDisconnected on the FIRST stock,
    # 1 retry exhausted, fall-through. But AKShare's eastmoney back-end
    # is known to drop the first connection of a session, then work after
    # a reconnect. Retry up to 3 times with 1s / 2s / 4s exponential
    # backoff. Also count consecutive failures: if >50 stocks in a row
    # ALL fail, the provider is dead, raise to fall through. Otherwise
    # we'd grind through all 5527 with 0% success burning ~20 minutes.
    MAX_RETRIES = 3
    MAX_CONSECUTIVE_FAILURES = 50
    consecutive_failures = 0

    data: dict[str, pd.DataFrame] = {}
    total = len(start_by_code)
    with progress_bar(total, "akshare", "stock") as progress:
        for i, (code, start_date) in enumerate(sorted(start_by_code.items()), start=1):
            symbol = code.split(".")[1]
            raw = None
            last_exc = None
            for attempt in range(MAX_RETRIES):
                try:
                    raw = ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date.replace("-", ""),
                        end_date=end_date.replace("-", ""),
                        adjust="qfq",
                    )
                    consecutive_failures = 0
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < MAX_RETRIES - 1:
                        _time.sleep(2 ** attempt)  # 1s, 2s, then bail
            if raw is None:
                logger.warning(
                    "AKShare failed for %s after %d retries: %s",
                    code, MAX_RETRIES, last_exc,
                )
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"AKShare hit {consecutive_failures} consecutive stock "
                        f"failures (latest={code}). Service appears down — "
                        f"bailing so the auto chain falls through to baostock."
                    )
            elif not raw.empty:
                df = raw.rename(
                    columns={
                        "日期": "date",
                        "开盘": "open",
                        "最高": "high",
                        "最低": "low",
                        "收盘": "close",
                        "成交量": "volume",
                        "成交额": "amount",
                        "换手率": "turn",
                    }
                )
                for missing in ("peTTM", "pbMRQ"):
                    if missing not in df.columns:
                        df[missing] = np.nan
                data[code] = normalize_frame(df)
            if progress is not None:
                progress.update(1)
                progress.set_postfix(ok=len(data), current=code)
            if i % 50 == 0:
                logger.info("Fetched: %s/%s (%s ok)", i, total, len(data))
            if i >= 100 and len(data) / i < 0.20:
                raise RuntimeError(
                    f"AKShare success rate too low: {len(data)}/{i}; "
                    "aborting provider so auto mode can fall back"
                )
    return data


def _tushare_trade_dates(pro, start_date: str, end_date: str) -> list[str]:
    cal = pro.trade_cal(
        exchange="",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        is_open="1",
    )
    if cal is None or cal.empty:
        return []
    return sorted(cal["cal_date"].astype(str).tolist())


def fetch_with_tushare(start_by_code: dict[str, str], end_date: str) -> dict[str, pd.DataFrame]:
    import tushare as ts

    token = os.environ.get("TUSHARE_TOKEN") or os.environ.get("TUSHARE_PRO_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not configured")

    ts.set_token(token)
    pro = ts.pro_api()

    requested_ts_codes = {bs_code_to_ts_code(code): code for code in start_by_code}
    min_start = min(start_by_code.values())
    trade_dates = _tushare_trade_dates(pro, min_start, end_date)
    if not trade_dates:
        return {}

    frames: list[pd.DataFrame] = []
    with progress_bar(len(trade_dates), "tushare", "day") as progress:
        for trade_date in trade_dates:
            daily = pro.daily(trade_date=trade_date)
            if daily is None or daily.empty:
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(frames=len(frames), current=trade_date)
                continue
            daily = daily[daily["ts_code"].isin(requested_ts_codes)]
            if daily.empty:
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(frames=len(frames), current=trade_date)
                continue

            try:
                basic = pro.daily_basic(
                    trade_date=trade_date,
                    fields="ts_code,trade_date,turnover_rate,pe_ttm,pb",
                )
                daily = daily.merge(basic, on=["ts_code", "trade_date"], how="left")
            except Exception as exc:
                logger.warning("Tushare daily_basic failed for %s: %s", trade_date, exc)

            try:
                factor = pro.adj_factor(trade_date=trade_date)
                daily = daily.merge(factor, on=["ts_code", "trade_date"], how="left")
            except Exception as exc:
                logger.warning("Tushare adj_factor failed for %s: %s", trade_date, exc)

            frames.append(daily)
            if progress is not None:
                progress.update(1)
                progress.set_postfix(frames=len(frames), current=trade_date)

    if not frames:
        return {}

    merged = pd.concat(frames, ignore_index=True)
    data: dict[str, pd.DataFrame] = {}
    for ts_code, part in merged.groupby("ts_code"):
        bs_code = requested_ts_codes.get(ts_code)
        if not bs_code:
            continue
        min_code_start = start_by_code[bs_code].replace("-", "")
        part = part[part["trade_date"].astype(str) >= min_code_start]
        if part.empty:
            continue
        df = pd.DataFrame({
            "date": pd.to_datetime(part["trade_date"].astype(str)),
            "open": part["open"],
            "high": part["high"],
            "low": part["low"],
            "close": part["close"],
            "volume": part["vol"],
            "amount": part["amount"],
            "turn": part.get("turnover_rate", np.nan),
            "peTTM": part.get("pe_ttm", np.nan),
            "pbMRQ": part.get("pb", np.nan),
            "factor": part.get("adj_factor", np.nan),
        })
        data[bs_code] = normalize_frame(df)
    return data


def _parse_stocktoday_response(resp) -> pd.DataFrame:
    """Parse StockToday/Tushare-compatible responses into a DataFrame."""
    if resp is None:
        return pd.DataFrame()
    if isinstance(resp, list):
        return pd.DataFrame(resp)
    if not isinstance(resp, dict):
        return pd.DataFrame()
    if "error" in resp:
        raise RuntimeError(str(resp["error"]))
    code = resp.get("code")
    if code not in (None, 0, "0"):
        raise RuntimeError(f"code={code}, msg={resp.get('msg', '')}")
    data = resp.get("data")
    if not data:
        return pd.DataFrame()
    if isinstance(data, dict):
        items = data.get("items")
        columns = data.get("fields") or data.get("columns")
        if items and columns:
            return pd.DataFrame(items, columns=columns)
        if items:
            return pd.DataFrame(items)
    if isinstance(data, list):
        return pd.DataFrame(data)
    return pd.DataFrame()


def _stocktoday_call(label: str, fn):
    return _run_with_alarm(45, fn, label=f"StockToday {label}")


def _stocktoday_trade_dates(st, start_date: str, end_date: str) -> list[str]:
    try:
        df = _parse_stocktoday_response(
            _stocktoday_call(
                "trade_cal",
                lambda: st.trade_cal(
                    exchange="SSE",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    is_open="1",
                ),
            )
        )
    except Exception as exc:
        logger.warning("StockToday trade_cal failed: %s; using weekday fallback", exc)
        df = pd.DataFrame()
    if df.empty or "cal_date" not in df.columns:
        return [d.strftime("%Y%m%d") for d in pd.bdate_range(start_date, end_date)]
    dates = sorted(set(df["cal_date"].astype(str).tolist()))
    end_yyyymmdd = end_date.replace("-", "")
    # ST trade_cal can lag intraday/early evening while daily/daily_basic
    # already has the latest trading day. Include a weekday end-date probe
    # and let the actual daily endpoint decide whether data exists.
    if pd.Timestamp(end_date).weekday() < 5 and end_yyyymmdd not in dates:
        dates.append(end_yyyymmdd)
        dates = sorted(dates)
    return dates


def fetch_with_stocktoday(
    start_by_code: dict[str, str],
    end_date: str,
    cap_lookback_days: int | None = 30,
) -> dict[str, pd.DataFrame]:
    from config.settings import ST_TOKEN
    from ST_CLIENT import StockToday

    if not ST_TOKEN:
        raise RuntimeError("ST_TOKEN is not configured")

    st = StockToday(token=ST_TOKEN)
    requested_ts_codes = {bs_code_to_ts_code(code): code for code in start_by_code}
    min_start = min(start_by_code.values())
    # 2026-06-11 root-cause fix D: cap min_start at 30 days back.
    # cx review 2026-06-13 C1 fix: cap is now opt-in via parameter so
    # ``--full --provider stocktoday`` actually iterates full history.
    # The 2026-06-11 investigation showed `--new-symbol-days=365`
    # causes every fresh / partial-history stock to fall back to
    # `end_date - 365d`. If even ONE stock in universe lacks history,
    # the WHOLE market-wide fetch loops 365 trade_dates × 3 endpoints —
    # that's the actual reason 4 days of qlib_data_update have
    # timed out. ST is market-wide per trade_date, so per-stock filter
    # later trims rows; capping the date span here only loses old
    # rows for new-listing stocks — acceptable since (a) ST returns
    # 400 on many of those old dates anyway and (b) a quarterly
    # ``--full`` run repairs deep history. baostock fallback still
    # honors original min_start for stocks that truly need it.
    if cap_lookback_days is not None:
        cap_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=cap_lookback_days)
        cap_str = cap_dt.strftime("%Y-%m-%d")
        if min_start < cap_str:
            logger.info(
                "StockToday min_start clipped from %s to %s (cap %d days) — "
                "deep-history backfill goes to --full or another provider.",
                min_start, cap_str, cap_lookback_days,
            )
            min_start = cap_str
    trade_dates = _stocktoday_trade_dates(st, min_start, end_date)
    if not trade_dates:
        return {}

    frames: list[pd.DataFrame] = []
    # 2026-06-13 fix A v3 — skip-and-continue.
    # Original "bail at 10 endpoint failures" was too aggressive.
    # 06-13 re-test: ST returned errors on 15/22 past trade_dates (mix
    # of HTTP 400 / "服务器异常状态" / "数据服务不可用"). The 06-12 run
    # bailed in 3 min having wasted the 7200s cron budget, leaving
    # akshare (dead) and baostock (7.7h) as the fallback. Even a
    # PARTIAL ST result (e.g. only today's date succeeds) is far more
    # useful than running baostock for hours and timing out.
    # New behaviour: log failures, skip the failed date+endpoint, keep
    # going. Whatever frames we collect get returned. The auto-chain
    # `len(data) >= min_ok` gate decides whether the partial result
    # is enough; if not, baostock fallback still runs but for fewer
    # stocks because partial ST data populated some of them.
    total_endpoint_failures = 0
    per_endpoint_failures = {"daily": 0, "daily_basic": 0, "adj_factor": 0}
    with progress_bar(len(trade_dates), "stocktoday", "day") as progress:
        for i, trade_date in enumerate(trade_dates, start=1):
            try:
                daily = _parse_stocktoday_response(
                    _stocktoday_call(
                        f"daily {trade_date}",
                        lambda td=trade_date: st.daily(trade_date=td),
                    )
                )
            except Exception as exc:
                logger.warning("StockToday daily failed for %s: %s", trade_date, exc)
                daily = pd.DataFrame()
                total_endpoint_failures += 1
                per_endpoint_failures["daily"] += 1
            if daily.empty or "ts_code" not in daily.columns:
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(frames=len(frames), current=trade_date)
                continue
            daily = daily[daily["ts_code"].astype(str).isin(requested_ts_codes)]
            if daily.empty:
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(frames=len(frames), current=trade_date)
                continue

            try:
                basic = _parse_stocktoday_response(
                    _stocktoday_call(
                        f"daily_basic {trade_date}",
                        lambda td=trade_date: st.daily_basic(
                            trade_date=td,
                            fields="ts_code,trade_date,turnover_rate,pe_ttm,pb",
                        ),
                    )
                )
                if not basic.empty:
                    daily = daily.merge(basic, on=["ts_code", "trade_date"], how="left")
            except Exception as exc:
                logger.warning("StockToday daily_basic failed for %s: %s", trade_date, exc)
                total_endpoint_failures += 1
                per_endpoint_failures["daily_basic"] += 1

            try:
                factor = _parse_stocktoday_response(
                    _stocktoday_call(
                        f"adj_factor {trade_date}",
                        lambda td=trade_date: st.adj_factor(trade_date=td),
                    )
                )
                if not factor.empty:
                    daily = daily.merge(factor, on=["ts_code", "trade_date"], how="left")
            except Exception as exc:
                logger.warning("StockToday adj_factor failed for %s: %s", trade_date, exc)
                total_endpoint_failures += 1
                per_endpoint_failures["adj_factor"] += 1

            frames.append(daily)
            if progress is not None:
                progress.update(1)
                progress.set_postfix(frames=len(frames), current=trade_date)
            if i % 10 == 0:
                time.sleep(0.5)

    n_dates = len(trade_dates)
    n_success = len(frames)
    logger.info(
        "StockToday loop done: %d/%d trade_dates returned data. "
        "Endpoint failure breakdown: daily=%d daily_basic=%d adj_factor=%d "
        "(total=%d). %s",
        n_success, n_dates,
        per_endpoint_failures["daily"],
        per_endpoint_failures["daily_basic"],
        per_endpoint_failures["adj_factor"],
        total_endpoint_failures,
        ("Partial data passed downstream — auto-chain min_ok gate decides "
         "whether to use it or fall through to next provider.") if n_success > 0
        else "No usable data — falling through to next provider.",
    )

    if not frames:
        return {}

    merged = pd.concat(frames, ignore_index=True)
    data: dict[str, pd.DataFrame] = {}
    for ts_code, part in merged.groupby("ts_code"):
        bs_code = requested_ts_codes.get(str(ts_code))
        if not bs_code:
            continue
        min_code_start = start_by_code[bs_code].replace("-", "")
        part = part[part["trade_date"].astype(str) >= min_code_start]
        if part.empty:
            continue
        df = pd.DataFrame({
            "date": pd.to_datetime(part["trade_date"].astype(str)),
            "open": part["open"],
            "high": part["high"],
            "low": part["low"],
            "close": part["close"],
            "volume": part["vol"],
            "amount": part["amount"],
            "turn": part.get("turnover_rate", np.nan),
            "peTTM": part.get("pe_ttm", np.nan),
            "pbMRQ": part.get("pb", np.nan),
            "factor": part.get("adj_factor", np.nan),
        })
        data[bs_code] = normalize_frame(df)
    return data


def fetch_data(
    provider: str,
    start_by_code: dict[str, str],
    end_date: str,
    cap_lookback_days: int | None = 30,
) -> tuple[str, dict[str, pd.DataFrame]]:
    # cx review 2026-06-13 C1 fix: ``cap_lookback_days=None`` means
    # "honor every per-stock start_date" — used by ``--full`` callers.
    # Default 30 matches the incremental-cron policy. Note: the
    # ``min_ok`` gate below is on stock-count coverage (width), not on
    # date freshness; the real freshness defense is
    # ``check_instrument_freshness`` + ``validate_qlib_health`` on the
    # staged output. See cx review I2.
    if not start_by_code:
        return provider, {}

    if provider == "auto":
        # 2026-06-14: drop the tushare → akshare → baostock fallback
        # chain. Today's evidence:
        #   - tushare: requires paid official token we don't have
        #   - akshare: backend RemoteDisconnected on every stock 06-12,
        #     looks like IP block or service outage
        #   - baostock: 5s/query × 5527 stocks = 7.7h, cron budget is
        #     7200s; cannot finish under any cap
        # ST is reliable for recent dates and the ONLY days it returns
        # 400 are specific past dates where the server-side ETL has a
        # gap — those days baostock can't recover either. Better to
        # accept a calendar gap than burn the cron budget on a chain
        # that will time out. Opt-in providers (--provider baostock
        # etc.) still work for manual deep-history backfill.
        logger.info("Trying StockToday provider (auto = ST-only since 2026-06-14)")
        data = fetch_with_stocktoday(start_by_code, end_date, cap_lookback_days)
        # No min_ok threshold gate: accept whatever ST returned. If it
        # returned 0 stocks (e.g. ST fully dark today), raise so the
        # cron registers a clean failure rather than falling through to
        # a multi-hour baostock attempt that also won't finish.
        if not data:
            raise RuntimeError(
                "StockToday returned no data (auto mode is ST-only "
                "since 2026-06-14). Calendar may have a gap for "
                f"end_date={end_date}; will retry on next cron tick. "
                "For manual deep-history backfill use --provider baostock."
            )
        return "stocktoday", data

    if provider == "stocktoday":
        return provider, fetch_with_stocktoday(start_by_code, end_date, cap_lookback_days)
    if provider == "tushare":
        return provider, fetch_with_tushare(start_by_code, end_date)
    if provider == "akshare":
        return provider, fetch_with_akshare(start_by_code, end_date)
    if provider == "baostock":
        return provider, fetch_with_baostock(start_by_code, end_date, cap_lookback_days)
    raise ValueError(f"Unknown provider: {provider}")


def _feature_span(values: np.ndarray) -> tuple[int, int] | None:
    finite = np.isfinite(values)
    if not finite.any():
        return None
    start_index = int(np.argmax(finite))
    end_index = len(finite) - 1 - int(np.argmax(finite[::-1]))
    return start_index, end_index


def write_feature_bin(
    path: Path,
    values: pd.Series,
    calendar_dates: list[str],
    span: tuple[int, int] | None = None,
) -> bool:
    numeric = pd.to_numeric(values, errors="coerce")
    aligned = numeric.reindex(pd.to_datetime(calendar_dates))
    aligned_values = aligned.to_numpy(dtype="<f4", na_value=np.nan)
    if span is None:
        span = _feature_span(aligned_values)
    if span is None:
        return False

    start_index, end_index = span
    payload = aligned_values[start_index:end_index + 1]
    out = np.hstack([[float(start_index)], payload]).astype("<f4")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.tofile(path)
    return True


def save_to_qlib_format(
    instrument: str,
    df: pd.DataFrame,
    qlib_dir: Path,
    calendar_dates: list[str],
) -> bool:
    """Merge new rows into Qlib bins using the proper `[start_index, values...]` format."""
    if df.empty:
        return False

    inst_dir = _feature_dir(qlib_dir, instrument)
    wrote_any = False
    calendar_index = pd.to_datetime(calendar_dates)
    merged_by_feature: dict[str, pd.Series] = {}

    for col in df.columns:
        if not col.startswith("$"):
            continue
        feature_name = col.replace("$", "")
        existing = read_feature_series(qlib_dir, instrument, feature_name, calendar_dates)
        if existing.empty:
            merged = pd.Series(index=calendar_index, dtype=np.float32)
        else:
            merged = pd.to_numeric(existing, errors="coerce").reindex(calendar_index)
        updates = pd.to_numeric(df[col], errors="coerce")
        updates.index = pd.to_datetime(updates.index)
        updates = updates.dropna()
        if updates.empty:
            continue
        merged.loc[updates.index] = updates.to_numpy(dtype=np.float32)
        merged_by_feature[feature_name] = merged

    core_spans = []
    for field_name in REQUIRED_OUTPUT_COLUMNS:
        feature_name = field_name.replace("$", "")
        series = merged_by_feature.get(feature_name)
        if series is None:
            continue
        span = _feature_span(series.to_numpy(dtype="<f4", na_value=np.nan))
        if span is not None:
            core_spans.append(span)
    common_core_span = None
    if core_spans:
        common_core_span = (
            min(span[0] for span in core_spans),
            max(span[1] for span in core_spans),
        )

    for feature_name, merged in merged_by_feature.items():
        span = common_core_span if f"${feature_name}" in REQUIRED_OUTPUT_COLUMNS else None
        path = inst_dir / f"{feature_name}.day.bin"
        wrote_any = write_feature_bin(path, merged, calendar_dates, span=span) or wrote_any

    return wrote_any


def repair_required_field_spans(qlib_dir: Path, calendar_dates: list[str]) -> int:
    """Rewrite required OHLCV/amount fields so each instrument has one shared span."""
    features_dir = qlib_dir / "features"
    if not features_dir.exists() or not calendar_dates:
        return 0

    calendar_index = pd.to_datetime(calendar_dates)
    repaired = 0
    for inst_dir in features_dir.iterdir():
        if not inst_dir.is_dir():
            continue

        series_by_field: dict[str, pd.Series] = {}
        spans: dict[str, tuple[int, int]] = {}
        for field_name in (col.replace("$", "") for col in REQUIRED_OUTPUT_COLUMNS):
            path = inst_dir / f"{field_name}.day.bin"
            if not path.exists():
                continue
            bs_code = qlib_to_bs_code(inst_dir.name)
            series = read_feature_series(qlib_dir, bs_code, field_name, calendar_dates)
            if series.empty:
                continue
            series = pd.to_numeric(series, errors="coerce").reindex(calendar_index)
            span = _feature_span(series.to_numpy(dtype="<f4", na_value=np.nan))
            if span is None:
                continue
            series_by_field[field_name] = series
            spans[field_name] = span

        if len(series_by_field) < len(REQUIRED_OUTPUT_COLUMNS):
            continue
        common_span = (
            min(span[0] for span in spans.values()),
            max(span[1] for span in spans.values()),
        )
        if all(span == common_span for span in spans.values()):
            continue

        for field_name, series in series_by_field.items():
            write_feature_bin(inst_dir / f"{field_name}.day.bin", series, calendar_dates, span=common_span)
        repaired += 1

    if repaired:
        logger.info("Repaired required-field spans for %s instruments", repaired)
    return repaired


def repair_legacy_bins(qlib_dir: Path, calendar_dates: list[str]) -> int:
    """Convert legacy full-calendar bins to Qlib header bins in-place."""
    features_dir = qlib_dir / "features"
    if not features_dir.exists() or not calendar_dates:
        return 0

    repaired = 0
    calendar_count = len(calendar_dates)
    for path in features_dir.glob("*/*.day.bin"):
        arr = np.fromfile(path, dtype="<f4")
        if arr.size == 0:
            continue
        if arr.size >= 2 and _valid_start_index(float(arr[0]), calendar_count, arr.size):
            continue
        values = arr[:calendar_count]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        start_index = int(np.argmax(finite))
        end_index = len(finite) - 1 - int(np.argmax(finite[::-1]))
        out = np.hstack([[float(start_index)], values[start_index:end_index + 1]]).astype("<f4")
        out.tofile(path)
        repaired += 1
    if repaired:
        logger.info("Repaired %s legacy Qlib feature bins", repaired)
    return repaired + repair_required_field_spans(qlib_dir, calendar_dates)


def update_instruments(qlib_dir: Path, stock_date_ranges: dict[str, tuple[str, str]]) -> None:
    inst_dir = qlib_dir / "instruments"
    if not inst_dir.exists() or not stock_date_ranges:
        return

    lookup = {}
    for code, date_range in stock_date_ranges.items():
        qlib_code = bs_to_qlib_code(code)
        lookup[qlib_code.lower()] = date_range
        lookup[qlib_code.upper()] = date_range

    for txt_file in inst_dir.glob("*.txt"):
        lines = txt_file.read_text().splitlines()
        updated = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                code = parts[0]
                if code in lookup:
                    parts[1], parts[2] = lookup[code]
                elif code.lower() in lookup:
                    parts[1], parts[2] = lookup[code.lower()]
                updated.append("\t".join(parts))
            elif line.strip():
                updated.append(line)
        txt_file.write_text("\n".join(updated) + "\n")
    logger.info("Updated instruments date ranges for %s stocks", len(stock_date_ranges))


def _load_existing_instrument_ranges(qlib_dir: Path) -> dict[str, tuple[str, str]]:
    ranges: dict[str, tuple[str, str]] = {}
    inst_dir = qlib_dir / "instruments"
    if not inst_dir.exists():
        return ranges

    for txt_file in inst_dir.glob("*.txt"):
        for line in txt_file.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                ranges[parts[0].lower()] = (parts[1], parts[2])
    return ranges


def _write_instrument_file(
    path: Path,
    codes: Iterable[str],
    date_ranges: dict[str, tuple[str, str]],
) -> int:
    rows = []
    for code in sorted(set(codes)):
        qlib_code = bs_to_qlib_code(code).lower()
        date_range = date_ranges.get(qlib_code)
        if not date_range:
            continue
        rows.append(f"{qlib_code}\t{date_range[0]}\t{date_range[1]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""))
    return len(rows)


def sync_instrument_files(
    qlib_dir: Path,
    universe_groups: dict[str, set[str]],
    calendar_dates: list[str],
) -> None:
    """Rewrite refreshed universe files using symbols with actual feature coverage."""
    if not universe_groups:
        return

    existing_ranges = _load_existing_instrument_ranges(qlib_dir)
    date_ranges = dict(existing_ranges)
    available_codes: set[str] = set()
    for codes in universe_groups.values():
        available_codes |= codes

    for code in sorted(available_codes):
        date_range = infer_feature_date_range(qlib_dir, code, calendar_dates)
        if date_range:
            date_ranges[bs_to_qlib_code(code).lower()] = date_range

    inst_dir = qlib_dir / "instruments"
    written: dict[str, int] = {}
    for group_name, codes in universe_groups.items():
        written[group_name] = _write_instrument_file(
            inst_dir / f"{group_name}.txt",
            codes,
            date_ranges,
        )

    if "all" not in universe_groups:
        all_codes: set[str] = set()
        for codes in universe_groups.values():
            all_codes |= codes
        written["all"] = _write_instrument_file(inst_dir / "all.txt", all_codes, date_ranges)

    logger.info(
        "Synced instrument files: %s",
        ", ".join(f"{name}={count}" for name, count in sorted(written.items())),
    )


def lgb_universe_needs_refresh(
    qlib_dir: Path,
    update_universe: str,
    lgb_universe: str,
    min_lgb_instruments: int,
) -> bool:
    """Return True when the update should refresh the universe for LGB coverage."""
    if update_universe not in (lgb_universe, "csi800"):
        return False

    local_count = len(load_codes_from_instruments(qlib_dir, lgb_universe))
    if local_count >= min_lgb_instruments:
        return False

    logger.warning(
        "LGB inference universe %s has only %s instruments (<%s); forcing universe refresh",
        lgb_universe,
        local_count,
        min_lgb_instruments,
    )
    return True


def validate_qlib_health(
    qlib_dir: Path,
    universe: str,
    min_coverage: float,
    lookback_days: int,
    min_instruments: int = 0,
) -> bool:
    report = check_qlib_dir(
        qlib_dir,
        universe=universe,
        min_coverage=min_coverage,
        min_instruments=min_instruments,
        lookback_days=lookback_days,
    )
    if not report.ok:
        logger.error(
            "Qlib data health check failed for universe=%s; refusing promotion",
            universe,
        )
        for error in report.errors:
            logger.error("  %s", error)
        return False

    logger.info(
        "Qlib data health check passed for universe=%s: instruments=%s coverage=%.1f%%",
        universe,
        report.instruments_checked,
        report.latest_close_coverage * 100,
    )
    return True


def validate_lgb_smoke(
    qlib_dir: Path,
    min_predictions: int,
) -> bool:
    if not LGB_MODEL_PATH.exists():
        logger.warning("LGB smoke check skipped because model file is missing: %s", LGB_MODEL_PATH)
        return True

    try:
        from scripts.smoke_lgb_predict import run_smoke

        result = run_smoke(
            model_path=LGB_MODEL_PATH,
            min_predictions=min_predictions,
            qlib_dir=qlib_dir,
        )
    except Exception as exc:
        logger.error("LGB smoke check failed before promotion: %s", exc)
        return False

    if not result.get("ok"):
        logger.error(
            "LGB smoke check failed before promotion: %s",
            result.get("error", "unknown error"),
        )
        return False

    logger.info(
        "LGB smoke check passed before promotion: finite=%s total=%s min=%s",
        result.get("finite_prediction_count"),
        result.get("prediction_count"),
        result.get("min_predictions"),
    )
    return True


def prepare_staging(source_dir: Path, staging_dir: Path, use_staging: bool) -> Path:
    if not use_staging:
        return source_dir

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    if source_dir.exists():
        shutil.copytree(source_dir, staging_dir)
    else:
        staging_dir.mkdir(parents=True)
    return staging_dir


def promote_staging(staging_dir: Path, target_dir: Path) -> None:
    backup_dir = target_dir.parent / f"{target_dir.name}.backup-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    if target_dir.exists():
        shutil.move(str(target_dir), str(backup_dir))
    shutil.move(str(staging_dir), str(target_dir))
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def update_manifest(
    manifest: dict,
    data_by_code: dict[str, pd.DataFrame],
    attempted_codes: set[str],
    source: str,
    end_date: str,
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    symbols = manifest.setdefault("symbols", {})

    for code, df in data_by_code.items():
        if df.empty:
            continue
        state = symbols.setdefault(code, {})
        state["last_success_date"] = df.index.max().strftime("%Y-%m-%d")
        state["last_attempt_date"] = end_date
        state["last_source"] = source
        state["fail_count"] = 0
        state["updated_at"] = now
        state.pop("last_error", None)

    for code in attempted_codes - set(data_by_code):
        state = symbols.setdefault(code, {})
        state["last_attempt_date"] = end_date
        state["fail_count"] = int(state.get("fail_count", 0)) + 1
        state["last_error"] = "no rows returned"
        state["updated_at"] = now

    manifest["last_run"] = {
        "updated_at": now,
        "source": source,
        "attempted": len(attempted_codes),
        "succeeded": len(data_by_code),
    }


def check_instrument_freshness(qlib_dir: Path, universe: str, expected_date: str) -> tuple[bool, list[str]]:
    """Check representative instrument end dates from Qlib instrument files."""
    inst_file = qlib_dir / "instruments" / f"{universe}.txt"
    if not inst_file.exists():
        inst_file = qlib_dir / "instruments" / "all.txt"

    sample_rows = []
    if inst_file.exists():
        watched = {"sh600000", "sz000001", "sh601398", "sz000002", "sh600519"}
        lines = inst_file.read_text().splitlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 3 and parts[0] in watched:
                sample_rows.append(parts[:3])
        if len(sample_rows) < 3:
            rows = [line.split()[:3] for line in lines if len(line.split()) >= 3]
            sample_rows.extend(rows[:5])
    if len(sample_rows) < 3:
        detail = [f"insufficient_sample:{len(sample_rows)}"]
        logger.error(
            "Freshness check could not inspect enough instrument rows "
            "(universe=%s, file=%s, sample=%s); failing closed",
            universe,
            inst_file,
            len(sample_rows),
        )
        return False, detail

    stale_detail = []
    for stock, _start, latest in sample_rows[:10]:
        if latest < expected_date:
            stale_detail.append(f"{stock}:{latest}")
            logger.warning(
                "  %s latest date = %s (expected=%s)",
                stock,
                latest,
                expected_date,
            )

    return len(stale_detail) < 3, stale_detail


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--incremental", action="store_true", help="Incremental daily update (default)")
    mode.add_argument("--full", action="store_true", help="Full historical rebuild/update")
    mode.add_argument("--repair-only", action="store_true", help="Repair legacy Qlib bin format without fetching data")
    parser.add_argument("--provider", choices=["auto", "stocktoday", "tushare", "akshare", "baostock"], default=QLIB_DATA_PROVIDER)
    parser.add_argument("--universe", default=LGB_INFERENCE_UNIVERSE, help="csi800, all, or instrument file stem")
    parser.add_argument("--refresh-universe", action="store_true", help="Refresh universe constituents from AKShare/baostock")
    parser.add_argument("--universe-source", choices=["auto", "akshare", "baostock"], default=QLIB_UNIVERSE_SOURCE)
    parser.add_argument(
        "--min-universe-size",
        type=int,
        default=LGB_MIN_DATA_INSTRUMENTS,
        help="Refresh universe when local instruments are smaller",
    )
    parser.add_argument("--qlib-dir", type=Path, default=QLIB_DIR)
    parser.add_argument("--staging-dir", type=Path, default=STAGING_QLIB_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--full-years", type=int, default=5)
    parser.add_argument("--incremental-lookback-days", type=int, default=10)
    parser.add_argument("--new-symbol-days", type=int, default=365)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--min-health-instruments", type=int, default=0)
    parser.add_argument("--health-lookback-days", type=int, default=10)
    parser.add_argument("--lgb-health-universe", default=LGB_INFERENCE_UNIVERSE)
    parser.add_argument("--min-lgb-data-instruments", type=int, default=LGB_MIN_DATA_INSTRUMENTS)
    parser.add_argument("--lgb-smoke-check", action="store_true")
    parser.add_argument("--no-staging", action="store_true", help="Write directly to qlib-dir")
    parser.add_argument("--no-repair-format", action="store_true", help="Skip legacy bin format repair")
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--check-today", action="store_true",
                        help="After update, verify latest data date matches today (exit 1 if stale)")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    # Global timeout: 45 minutes to prevent baostock infinite loops
    try:
        def _global_timeout(signum, frame):
            raise TimeoutError("Data update exceeded 45-minute global timeout")
        old = signal.signal(signal.SIGALRM, _global_timeout)
        signal.alarm(2700)  # 45 minutes
    except Exception:
        pass  # SIGALRM not available on Windows

    args = parse_args(argv)

    try:
        return _main_inner(args)
    except Exception as e:
        write_health("qlib_data_update", HealthStatus(
            success=False,
            error_type=type(e).__name__,
            error_message=str(e)[:200],
        ))
        raise


def _main_inner(args: argparse.Namespace) -> int:
    mode = "repair-only" if args.repair_only else "full" if args.full else "incremental"
    full_start_date = (
        datetime.strptime(args.end_date, "%Y-%m-%d") - timedelta(days=365 * args.full_years)
    ).strftime("%Y-%m-%d")

    logger.info(
        "Updating Qlib data: mode=%s provider=%s universe=%s end=%s",
        mode,
        args.provider,
        args.universe,
        args.end_date,
    )

    manifest = load_manifest(args.manifest)
    current_calendar = read_calendar(args.qlib_dir)

    if mode == "repair-only":
        output_dir = prepare_staging(args.qlib_dir, args.staging_dir, not args.no_staging)
        calendar_dates = read_calendar(output_dir)
        if not calendar_dates:
            logger.error("No calendar found; cannot repair Qlib data")
            return 1
        repaired = repair_legacy_bins(output_dir, calendar_dates)
        logger.info("Repair-only mode repaired %s bins", repaired)
        if not args.skip_health_check:
            if not validate_qlib_health(
                output_dir,
                universe=args.universe,
                min_coverage=args.min_coverage,
                min_instruments=args.min_health_instruments,
                lookback_days=args.health_lookback_days,
            ):
                return 1
        if not args.no_staging:
            promote_staging(output_dir, args.qlib_dir)
            logger.info("Promoted repaired Qlib data to %s", args.qlib_dir)
        return 0

    refresh_universe = args.refresh_universe or lgb_universe_needs_refresh(
        args.qlib_dir,
        update_universe=args.universe,
        lgb_universe=args.lgb_health_universe,
        min_lgb_instruments=args.min_lgb_data_instruments,
    )
    universe_selection = get_update_universe(
        qlib_dir=args.qlib_dir,
        universe=args.universe,
        refresh_universe=refresh_universe,
        min_universe_size=args.min_universe_size,
        universe_source=args.universe_source,
    )
    codes = universe_selection.codes
    logger.info("Found %s stocks to update", len(codes))

    start_by_code = build_start_dates(
        codes=codes,
        mode=mode,
        end_date=args.end_date,
        full_start_date=full_start_date,
        incremental_lookback_days=args.incremental_lookback_days,
        new_symbol_days=args.new_symbol_days,
        qlib_dir=args.qlib_dir,
        calendar_dates=current_calendar,
        manifest=manifest,
    )
    if not start_by_code:
        logger.info("No symbols need updating")
        latest_date = current_calendar[-1] if current_calendar else args.end_date
        if not args.skip_health_check:
            if not validate_qlib_health(
                args.qlib_dir,
                universe=args.universe,
                min_coverage=args.min_coverage,
                min_instruments=args.min_health_instruments,
                lookback_days=args.health_lookback_days,
            ):
                write_health("qlib_data_update", HealthStatus(
                    success=False,
                    error_type="HealthCheckFail",
                    error_message=f"no-op validation failed for universe={args.universe}",
                    n_items=0,
                    latest_date=latest_date,
                    network_profile="domestic",
                ))
                return 1
            if not validate_qlib_health(
                args.qlib_dir,
                universe=args.lgb_health_universe,
                min_coverage=args.min_coverage,
                min_instruments=args.min_lgb_data_instruments,
                lookback_days=args.health_lookback_days,
            ):
                write_health("qlib_data_update", HealthStatus(
                    success=False,
                    error_type="HealthCheckFail",
                    error_message=f"no-op validation failed for universe={args.lgb_health_universe}",
                    n_items=0,
                    latest_date=latest_date,
                    network_profile="domestic",
                ))
                return 1
        if args.lgb_smoke_check and not validate_lgb_smoke(
            args.qlib_dir,
            min_predictions=LGB_MIN_PREDICTIONS,
        ):
            write_health("qlib_data_update", HealthStatus(
                success=False,
                error_type="LGBSmokeFail",
                error_message="no-op LGB smoke check failed",
                n_items=0,
                latest_date=latest_date,
                network_profile="domestic",
            ))
            return 1
        if getattr(args, "check_today", False):
            from scheduler.data_health import _expected_latest_trading_date
            expected_date = _expected_latest_trading_date()
            fresh, stale_detail = check_instrument_freshness(
                args.qlib_dir,
                args.universe,
                expected_date,
            )
            if not fresh:
                write_health("qlib_data_update", HealthStatus(
                    success=False,
                    error_type="FreshnessFail",
                    error_message=(
                        f"no-op freshness failed expected={expected_date}; "
                        f"details={','.join(stale_detail)[:160]}"
                    ),
                    n_items=0,
                    latest_date=latest_date,
                    network_profile="domestic",
                ))
                return 1
        write_health("qlib_data_update", HealthStatus(
            success=True,
            n_items=len(codes),
            latest_date=latest_date,
            network_profile="domestic",
            extra={
                "mode": mode,
                "provider": args.provider,
                "universe": args.universe,
                "universe_source": universe_selection.source,
                "noop": True,
            },
        ))
        return 0
    logger.info("Symbols needing data: %s/%s", len(start_by_code), len(codes))

    # cx review 2026-06-13 C1 fix: ``--full`` mode disables the
    # 30-day lookback cap so a manual full backfill actually returns
    # the requested deep history. Incremental cron stays capped.
    cap_lookback_days = None if mode == "full" else 30
    source, data_by_code = fetch_data(
        args.provider, start_by_code, args.end_date, cap_lookback_days,
    )
    if not data_by_code:
        logger.error("No data fetched; refusing to update Qlib")
        update_manifest(manifest, data_by_code, set(start_by_code), source, args.end_date)
        save_manifest(manifest, args.manifest)
        return 1

    output_dir = prepare_staging(args.qlib_dir, args.staging_dir, not args.no_staging)

    all_dates = set(read_calendar(output_dir))
    for df in data_by_code.values():
        all_dates |= set(df.index.strftime("%Y-%m-%d"))
    calendar_dates = sorted(all_dates)
    write_calendar(output_dir, calendar_dates)
    logger.info(
        "Calendar: %s trading days (%s ~ %s)",
        len(calendar_dates),
        calendar_dates[0],
        calendar_dates[-1],
    )

    success = 0
    fail = 0
    stock_date_ranges: dict[str, tuple[str, str]] = {}
    for code, df in data_by_code.items():
        try:
            if save_to_qlib_format(code, df, output_dir, calendar_dates):
                success += 1
                date_range = infer_feature_date_range(output_dir, code, calendar_dates)
                if date_range:
                    stock_date_ranges[code] = date_range
            else:
                fail += 1
        except Exception as exc:
            fail += 1
            logger.warning("Failed writing %s: %s", code, exc)

    update_instruments(output_dir, stock_date_ranges)
    if universe_selection.source != "instruments" or args.refresh_universe:
        sync_instrument_files(output_dir, universe_selection.groups, calendar_dates)

    if not args.no_repair_format:
        repair_legacy_bins(output_dir, calendar_dates)

    if not args.skip_health_check:
        if not validate_qlib_health(
            output_dir,
            universe=args.universe,
            min_coverage=args.min_coverage,
            min_instruments=args.min_health_instruments,
            lookback_days=args.health_lookback_days,
        ):
            logger.error("Staged data was not promoted; manifest progress was not advanced")
            return 1
        if not validate_qlib_health(
            output_dir,
            universe=args.lgb_health_universe,
            min_coverage=args.min_coverage,
            min_instruments=args.min_lgb_data_instruments,
            lookback_days=args.health_lookback_days,
        ):
            logger.error("Staged data was not promoted; manifest progress was not advanced")
            return 1

    if args.lgb_smoke_check and not validate_lgb_smoke(
        output_dir,
        min_predictions=LGB_MIN_PREDICTIONS,
    ):
        logger.error("Staged data was not promoted; manifest progress was not advanced")
        return 1

    if getattr(args, "check_today", False):
        from scheduler.data_health import _expected_latest_trading_date
        expected_date = _expected_latest_trading_date()
        fresh, stale_detail = check_instrument_freshness(
            output_dir,
            args.universe,
            expected_date,
        )
        if not fresh:
            logger.error(
                "DATA STALE: %s sample stocks do not have expected date (%s). "
                "Staged data was not promoted; manifest progress was not advanced.",
                len(stale_detail),
                expected_date,
            )
            write_health("qlib_data_update", HealthStatus(
                success=False,
                error_type="FreshnessFail",
                error_message=(
                    f"{len(stale_detail)} sample stocks behind "
                    f"expected={expected_date}; details={','.join(stale_detail)[:160]}"
                ),
                n_items=success,
                latest_date=args.end_date,
                network_profile="domestic",
            ))
            return 1
        logger.info("--check-today: staged data freshness OK (expected=%s)", expected_date)

    update_manifest(manifest, data_by_code, set(start_by_code), source, args.end_date)
    save_manifest(manifest, args.manifest)

    if not args.no_staging:
        promote_staging(output_dir, args.qlib_dir)
        logger.info("Promoted staged Qlib data to %s", args.qlib_dir)

    logger.info("Update complete: %s stocks updated, %s failed, source=%s", success, fail, source)
    logger.info("Data stored in %s", args.qlib_dir)

    # Only NOW is it safe to declare success — both the update and
    # any requested freshness check have passed.
    write_health("qlib_data_update", HealthStatus(
        success=True,
        n_items=success,
        latest_date=args.end_date,
        network_profile="domestic",
    ))

    return 0 if success > 0 and fail == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
