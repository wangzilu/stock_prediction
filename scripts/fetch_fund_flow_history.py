"""Fetch fund flow + northbound history for all A-shares via StockToday API.

Default: batch-by-date mode (按日期批量拉全市场, ~120 requests for 60 days).
Fallback: per-stock mode (逐只拉取, 5000+ requests) or akshare.

Saves to:
  data/storage/fund_flow_history.parquet
  data/storage/northbound_history.parquet

Usage:
    # Batch mode (default, fast, ~120 requests for 60 days)
    python scripts/fetch_fund_flow_history.py                # 60 trading days
    python scripts/fetch_fund_flow_history.py --days 120     # 120 trading days
    python scripts/fetch_fund_flow_history.py --flow-only    # skip northbound
    python scripts/fetch_fund_flow_history.py --nb-only      # skip fund flow

    # Per-stock mode (old, slow, 5000+ requests)
    python scripts/fetch_fund_flow_history.py --per-stock
    python scripts/fetch_fund_flow_history.py --per-stock --top 500 --workers 3
    python scripts/fetch_fund_flow_history.py --per-stock --incremental

    # AKShare fallback (no ST_CLIENT needed)
    python scripts/fetch_fund_flow_history.py --source ak --per-stock
"""
import argparse
import logging
import os
import signal
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Bypass local proxy for domestic API requests — proxies cause timeouts
# on large data transfers from eastmoney / StockToday servers
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
            "ALL_PROXY", "all_proxy"):
    os.environ.pop(key, None)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
FLOW_PATH = DATA_DIR / "fund_flow_history.parquet"
NB_PATH = DATA_DIR / "northbound_history.parquet"

# ---------- retry / adaptive rate limiting ----------

MAX_RETRIES = 3
BACKOFF_BASE = 2.0
COOLDOWN_THRESHOLD = 5
COOLDOWN_SECONDS = 120
MAX_COOLDOWNS = 3
CHECKPOINT_INTERVAL = 25

_shutdown_requested = threading.Event()


class RateLimitExhausted(Exception):
    pass


class NoDataForStock(Exception):
    """Raised when a source returns a valid empty response for one stock."""


def _handle_shutdown(signum, _frame):
    _shutdown_requested.set()
    raise KeyboardInterrupt(f"received signal {signum}")


class AdaptiveThrottle:
    """Thread-safe adaptive rate limiter with cooldown."""

    def __init__(self, min_sleep: float = 0.3, max_sleep: float = 5.0):
        self.min_sleep = min_sleep
        self.max_sleep = max_sleep
        self._sleep = min_sleep
        self._lock = threading.Lock()
        self._consecutive_ok = 0
        self._consecutive_fail = 0
        self._cooldown_count = 0

    def on_success(self):
        with self._lock:
            self._consecutive_fail = 0
            self._cooldown_count = 0
            self._consecutive_ok += 1
            if self._consecutive_ok >= 10:
                self._sleep = max(self.min_sleep, self._sleep * 0.85)
                self._consecutive_ok = 0

    def on_failure(self):
        with self._lock:
            self._consecutive_ok = 0
            self._consecutive_fail += 1
            self._sleep = min(self.max_sleep, self._sleep * 1.3)
            if self._consecutive_fail >= COOLDOWN_THRESHOLD:
                self._cooldown_count += 1
                self._consecutive_fail = 0
                if self._cooldown_count >= MAX_COOLDOWNS:
                    raise RateLimitExhausted(
                        f"Blocked after {MAX_COOLDOWNS} cooldowns. "
                        f"Save and retry with --incremental."
                    )
                logger.warning(
                    f"  Cooldown {self._cooldown_count}/{MAX_COOLDOWNS} — "
                    f"pausing {COOLDOWN_SECONDS}s..."
                )
                self._lock.release()
                try:
                    time.sleep(COOLDOWN_SECONDS)
                finally:
                    self._lock.acquire()
                self._sleep = self.min_sleep

    def wait(self):
        with self._lock:
            t = self._sleep
        time.sleep(t)

    @property
    def current_sleep(self) -> float:
        with self._lock:
            return self._sleep


def fetch_with_retry(fn, *args, throttle: AdaptiveThrottle, **kwargs):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            result = fn(*args, **kwargs)
            throttle.on_success()
            return result
        except Exception as e:
            last_err = e
            throttle.on_failure()
            time.sleep(BACKOFF_BASE ** attempt)
    raise last_err


# ---------- ST_CLIENT setup ----------

_st_client = None
_st_lock = threading.Lock()


def get_st_client():
    global _st_client
    with _st_lock:
        if _st_client is None:
            from config.settings import ST_TOKEN
            from ST_CLIENT import StockToday
            _st_client = StockToday(
                token=ST_TOKEN,
                backup_url2="http://111.229.164.2:8083/",
            )
            logger.info("Initialized StockToday API client")
    return _st_client


def qlib_code_to_ts_code(code: str) -> str:
    """Convert Qlib code (SH600519) to Tushare code (600519.SH)."""
    num = code.replace("SH", "").replace("SZ", "").replace("BJ", "")
    if code.startswith("SH"):
        return f"{num}.SH"
    elif code.startswith("SZ"):
        return f"{num}.SZ"
    elif code.startswith("BJ"):
        return f"{num}.BJ"
    # guess from number
    if num.startswith("6"):
        return f"{num}.SH"
    elif num.startswith(("0", "3")):
        return f"{num}.SZ"
    elif num.startswith(("4", "8")):
        return f"{num}.BJ"
    return f"{num}.SZ"


# ---------- helpers ----------

def get_all_stock_codes(top_n: int = None) -> list:
    features_dir = DATA_DIR / "qlib_data" / "cn_data" / "features"
    codes = sorted([d.name.upper() for d in features_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(codes)} stocks in Qlib features")
    if top_n and top_n < len(codes):
        codes = codes[:top_n]
        logger.info(f"Using top {top_n} stocks")
    return codes


def load_existing_codes(path: Path, code_col: str = "qlib_code") -> set:
    if not path.exists():
        return set()
    try:
        df = pd.read_parquet(path, columns=[code_col])
        codes = set(df[code_col].unique())
        logger.info(f"Incremental: {len(codes)} stocks already in {path.name}")
        return codes
    except Exception:
        return set()


def save_checkpoint(all_dfs: list, path: Path, dedup_cols: list):
    if not all_dfs:
        return
    result = pd.concat(all_dfs, ignore_index=True)
    if path.exists():
        old = pd.read_parquet(path)
        result = pd.concat([old, result], ignore_index=True)
        if all(col in result.columns for col in dedup_cols):
            result.drop_duplicates(subset=dedup_cols, keep="last", inplace=True)
        else:
            logger.warning(f"  Skip dedup for {path.name}; missing columns: {dedup_cols}")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    logger.info(f"  Checkpoint: saved {len(result)} records to {path.name}")


# ---------- generic batch fetcher ----------

def batch_fetch(codes: list, fetch_one_fn, throttle: AdaptiveThrottle,
                label: str, workers: int, save_path: Path,
                dedup_cols: list) -> pd.DataFrame:
    all_dfs = []
    success = 0
    fail = 0
    lock = threading.Lock()
    last_checkpoint = 0

    logger.info(f"Fetching {label} for {len(codes)} stocks "
                f"({workers} workers, sleep={throttle.min_sleep:.1f}s)...")

    def _task(code):
        nonlocal success, fail, last_checkpoint
        try:
            throttle.wait()
            df = fetch_one_fn(code, throttle)
            with lock:
                if df is not None:
                    all_dfs.append(df)
                    success += 1
                    if success - last_checkpoint >= CHECKPOINT_INTERVAL:
                        save_checkpoint(list(all_dfs), save_path, dedup_cols)
                        last_checkpoint = success
                else:
                    fail += 1
        except RateLimitExhausted:
            raise
        except Exception as e:
            with lock:
                fail += 1
                if fail <= 10:
                    logger.warning(f"  Failed {label} {code}: {e}")

    aborted = False
    interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_task, c): c for c in codes}
            for i, future in enumerate(as_completed(futures), 1):
                if _shutdown_requested.is_set():
                    interrupted = True
                    for f in futures:
                        f.cancel()
                    break
                try:
                    future.result()
                except RateLimitExhausted as e:
                    logger.error(f"  {e}")
                    aborted = True
                    for f in futures:
                        f.cancel()
                    break
                if i % 50 == 0 or i == len(codes):
                    logger.info(f"  {label}: {i}/{len(codes)} "
                                f"({success} ok, {fail} fail, "
                                f"sleep={throttle.current_sleep:.2f}s)")
    except RateLimitExhausted as e:
        logger.error(f"  {e}")
        aborted = True
    except KeyboardInterrupt:
        logger.warning(f"  Interrupted while fetching {label}; saving checkpoint...")
        interrupted = True

    if (aborted or interrupted) and all_dfs:
        save_checkpoint(list(all_dfs), save_path, dedup_cols)
        logger.info(f"  Saved {success} stocks before stop. "
                    f"Re-run with --incremental to continue.")
    if interrupted:
        raise KeyboardInterrupt

    if not all_dfs:
        logger.error(f"No {label} data fetched!")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"{label}: {len(result)} records for {success} stocks, {fail} failed")
    return result


# ========== StockToday (ST_CLIENT) data sources ==========

def _parse_st_result(result) -> pd.DataFrame | None:
    """Parse StockToday API response into a DataFrame."""
    if result is None:
        return None
    if isinstance(result, dict):
        if "error" in result:
            raise RuntimeError(result["error"])
        data = result.get("data")
        if isinstance(data, dict):
            items = data.get("items")
            fields = data.get("fields")
            if items and fields:
                return pd.DataFrame(items, columns=fields)
        elif isinstance(data, list):
            if data:
                return pd.DataFrame(data)
        return None
    if isinstance(result, list):
        if not result:
            return None
        return pd.DataFrame(result)
    return None


def _fetch_st_frame(st, methods: list[str], ts_code: str) -> pd.DataFrame | None:
    errors = []
    for method_name in methods:
        try:
            result = getattr(st, method_name)(ts_code=ts_code)
            df = _parse_st_result(result)
        except RuntimeError as e:
            errors.append(f"{method_name}: {e}")
            continue
        if df is not None and not df.empty:
            return df
    if errors:
        raise RuntimeError("; ".join(errors))
    raise NoDataForStock(ts_code)


def _to_standard_flow_columns(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Normalize common ST/AK columns enough for safe concat + dedup."""
    result = df.copy()
    if "日期" in result.columns and "trade_date" not in result.columns:
        result["trade_date"] = pd.to_datetime(result["日期"], errors="coerce").dt.strftime("%Y%m%d")
    if "持股日期" in result.columns and "trade_date" not in result.columns:
        result["trade_date"] = pd.to_datetime(result["持股日期"], errors="coerce").dt.strftime("%Y%m%d")
    if "trade_date" in result.columns:
        result["trade_date"] = result["trade_date"].astype(str)
    result["qlib_code"] = code
    result["code"] = code.replace("SH", "").replace("SZ", "").replace("BJ", "")
    return result


def _fetch_one_flow_st(code: str, throttle: AdaptiveThrottle):
    """Fetch fund flow via StockToday API.

    ST_CLIENT already has internal retry + load balancing across 4 servers,
    so we do NOT wrap this in fetch_with_retry to avoid timeout stacking.
    """
    if code.startswith("BJ"):
        return None

    st = get_st_client()
    ts_code = qlib_code_to_ts_code(code)

    try:
        df = _fetch_st_frame(st, ["moneyflow", "moneyflow_dc", "moneyflow_ths"], ts_code)
    except NoDataForStock:
        return None
    except RuntimeError:
        throttle.on_failure()
        try:
            df = _fetch_one_flow_ak(code, throttle)
        except Exception:
            raise

    if df is None or df.empty:
        # Not a server error, just no data for this stock
        return None

    throttle.on_success()
    return _to_standard_flow_columns(df, code)


def _fetch_one_nb_st(code: str, throttle: AdaptiveThrottle):
    """Fetch northbound holdings via StockToday API.

    Tries hk_hold first (has holding quantity/value), then stock_hsgt (name only).
    Falls back to akshare if ST_CLIENT fails.
    """
    if code.startswith("BJ"):
        return None

    st = get_st_client()
    ts_code = qlib_code_to_ts_code(code)

    # Try hk_hold first — should have vol/ratio/holding detail
    try:
        result = st.hk_hold(ts_code=ts_code)
        df = _parse_st_result(result)
        if df is not None and not df.empty:
            throttle.on_success()
            return _to_standard_flow_columns(df, code)
    except Exception:
        pass  # fall through to stock_hsgt

    # Fallback to stock_hsgt (less detail but more reliable)
    try:
        df = _fetch_st_frame(st, ["stock_hsgt"], ts_code)
    except NoDataForStock:
        return None
    except RuntimeError:
        throttle.on_failure()
        try:
            df = _fetch_one_nb_ak(code, throttle)
        except Exception:
            raise

    if df is None or df.empty:
        return None

    throttle.on_success()
    return _to_standard_flow_columns(df, code)


# ========== Batch-by-date mode (ST_CLIENT, ~100x fewer requests) ==========

def _get_trade_dates(st, days: int = 60) -> list:
    """Get recent N trading dates from trade_cal."""
    from datetime import datetime, timedelta
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=int(days * 1.6))).strftime("%Y%m%d")
    try:
        resp = st.trade_cal(exchange="SSE", start_date=start, end_date=end, is_open="1")
        df = _parse_st_result(resp)
        if df is not None and not df.empty and "cal_date" in df.columns:
            dates = sorted(df["cal_date"].astype(str).tolist())
            return dates[-days:]
    except Exception as e:
        logger.warning(f"trade_cal failed: {e}")
    # Fallback: weekdays
    import pandas as _pd
    dates = _pd.bdate_range(start, end)
    return [d.strftime("%Y%m%d") for d in dates][-days:]


def _ts_to_qlib(ts_code: str) -> str:
    """600519.SH -> SH600519"""
    if not isinstance(ts_code, str) or "." not in ts_code:
        return str(ts_code)
    num, ex = ts_code.split(".", 1)
    return f"{ex}{num}"


def fetch_fund_flow_batch(days: int = 60) -> pd.DataFrame:
    """Fetch fund flow by trade_date (全市场一次请求).

    60 days = 60 requests instead of 5413.
    """
    st = get_st_client()
    trade_dates = _get_trade_dates(st, days)
    logger.info(f"Batch mode: {len(trade_dates)} trading days")

    all_dfs = []
    for i, td in enumerate(trade_dates):
        try:
            resp = st.moneyflow(trade_date=td)
            df = _parse_st_result(resp)
            if df is not None and not df.empty:
                df["trade_date"] = td
                if "ts_code" in df.columns:
                    df["qlib_code"] = df["ts_code"].apply(_ts_to_qlib)
                    df["code"] = df["ts_code"].str.split(".").str[0]
                all_dfs.append(df)
                logger.info(f"  flow {td}: {len(df)} stocks")
            else:
                logger.warning(f"  flow {td}: empty")
        except Exception as e:
            logger.warning(f"  flow {td} failed: {e}")

        if (i + 1) % 20 == 0:
            time.sleep(0.3)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Fund flow batch: {len(result)} rows, "
                f"{result['qlib_code'].nunique()} stocks, {len(trade_dates)} days")
    return result


def fetch_northbound_batch(days: int = 60) -> pd.DataFrame:
    """Fetch northbound flow by trade_date (全市场一次请求).

    Uses moneyflow_hsgt(trade_date=) for daily north/southbound totals,
    and hk_hold for per-stock holdings.
    """
    st = get_st_client()
    trade_dates = _get_trade_dates(st, days)
    logger.info(f"Batch northbound: {len(trade_dates)} trading days")

    all_dfs = []
    for i, td in enumerate(trade_dates):
        try:
            # moneyflow_hsgt gives per-stock northbound flow
            resp = st.moneyflow_hsgt(trade_date=td)
            df = _parse_st_result(resp)
            if df is not None and not df.empty:
                df["trade_date"] = td
                if "ts_code" in df.columns:
                    df["qlib_code"] = df["ts_code"].apply(_ts_to_qlib)
                    df["code"] = df["ts_code"].str.split(".").str[0]
                all_dfs.append(df)
                logger.info(f"  nb {td}: {len(df)} stocks")
            else:
                logger.warning(f"  nb {td}: empty")
        except Exception as e:
            logger.warning(f"  nb {td} failed: {e}")

        if (i + 1) % 20 == 0:
            time.sleep(0.3)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    n_stocks = result['qlib_code'].nunique() if 'qlib_code' in result.columns else 0
    logger.info(f"Northbound batch: {len(result)} rows, "
                f"{n_stocks} stocks, {len(trade_dates)} days")
    return result


# ========== AKShare data sources (fallback) ==========

def _fetch_one_flow_ak(code: str, throttle: AdaptiveThrottle):
    """Fetch fund flow via akshare (eastmoney)."""
    import akshare as ak

    num = code.replace("SH", "").replace("SZ", "").replace("BJ", "")
    market = "sh" if code.startswith("SH") or num.startswith("6") else "sz"

    df = fetch_with_retry(ak.stock_individual_fund_flow,
                          stock=num, market=market, throttle=throttle)
    if df is not None and not df.empty:
        return _to_standard_flow_columns(df, code)
    return None


def _fetch_one_nb_ak(code: str, throttle: AdaptiveThrottle):
    """Fetch northbound via akshare (eastmoney)."""
    import akshare as ak

    num = code.replace("SH", "").replace("SZ", "").replace("BJ", "")

    df = fetch_with_retry(ak.stock_hsgt_individual_em,
                          symbol=num, throttle=throttle)
    if df is not None and not df.empty:
        return _to_standard_flow_columns(df, code)
    return None


# ========== High-level fetch functions ==========

def fetch_fund_flow(codes: list, workers: int = 1,
                    existing_codes: set = None,
                    source: str = "st") -> pd.DataFrame:
    if existing_codes:
        before = len(codes)
        codes = [c for c in codes if c not in existing_codes]
        logger.info(f"Incremental: skipping {before - len(codes)} already-fetched, "
                    f"{len(codes)} remaining")
    if not codes:
        logger.info("Fund flow: nothing new to fetch")
        return pd.DataFrame()

    if source == "st":
        fetch_fn = _fetch_one_flow_st
        throttle = AdaptiveThrottle(min_sleep=0.3, max_sleep=5.0)
    else:
        fetch_fn = _fetch_one_flow_ak
        throttle = AdaptiveThrottle(min_sleep=1.0, max_sleep=8.0)

    return batch_fetch(codes, fetch_fn, throttle,
                       label="Fund flow", workers=workers,
                       save_path=FLOW_PATH,
                       dedup_cols=["qlib_code", "trade_date"])


def fetch_northbound(codes: list, workers: int = 1,
                     existing_codes: set = None,
                     source: str = "st") -> pd.DataFrame:
    if existing_codes:
        before = len(codes)
        codes = [c for c in codes if c not in existing_codes]
        logger.info(f"Incremental: skipping {before - len(codes)} already-fetched, "
                    f"{len(codes)} remaining")
    if not codes:
        logger.info("Northbound: nothing new to fetch")
        return pd.DataFrame()

    if source == "st":
        fetch_fn = _fetch_one_nb_st
        throttle = AdaptiveThrottle(min_sleep=0.3, max_sleep=5.0)
    else:
        fetch_fn = _fetch_one_nb_ak
        throttle = AdaptiveThrottle(min_sleep=1.5, max_sleep=10.0)

    result = batch_fetch(codes, fetch_fn, throttle,
                         label="Northbound", workers=workers,
                         save_path=NB_PATH,
                         dedup_cols=["qlib_code", "trade_date"])

    # staleness warning (akshare only — ST data is current)
    if source == "ak" and not result.empty and "持股日期" in result.columns:
        max_date = pd.to_datetime(result["持股日期"]).max()
        days_stale = (pd.Timestamp.now() - max_date).days
        if days_stale > 30:
            logger.warning(
                f"⚠ Northbound data is stale! Latest: {max_date.date()}, "
                f"{days_stale} days behind. Consider --source st."
            )

    return result


# ---------- main ----------

def _save_with_merge(new_df: pd.DataFrame, path: Path, dedup_cols: list, label: str):
    """Save new data, merging with existing parquet if present."""
    if new_df.empty:
        return
    if path.exists():
        old_df = pd.read_parquet(path)
        new_df = pd.concat([old_df, new_df], ignore_index=True)
        if all(c in new_df.columns for c in dedup_cols):
            new_df.drop_duplicates(subset=dedup_cols, keep="last", inplace=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_parquet(path, index=False)
    logger.info(f"Saved {label} to {path} ({len(new_df)} rows)")
    if "trade_date" in new_df.columns:
        logger.info(f"  Date range: {new_df['trade_date'].min()} ~ {new_df['trade_date'].max()}")


def main():
    from scheduler.data_health import write_health, HealthStatus

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    parser = argparse.ArgumentParser(description="Fetch fund flow + northbound history")
    parser.add_argument("--top", type=int, default=None, help="Only fetch top N stocks")
    parser.add_argument("--flow-only", action="store_true", help="Skip northbound")
    parser.add_argument("--nb-only", action="store_true", help="Skip fund flow")
    parser.add_argument("--incremental", action="store_true",
                        help="Skip stocks already in existing parquet")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1)")
    parser.add_argument("--source", choices=["st", "ak"], default="st",
                        help="Data source: st=StockToday (default), ak=akshare")
    parser.add_argument("--per-stock", action="store_true",
                        help="Use old per-stock mode (slow, 5000+ requests). "
                             "Default is batch-by-date (fast, ~120 requests)")
    parser.add_argument("--days", type=int, default=60,
                        help="Number of trading days for batch mode (default: 60)")
    args = parser.parse_args()

    n_items = 0
    latest_date = ""
    try:
        # ===== Batch-by-date mode (default, ~100x fewer requests) =====
        if args.source == "st" and not args.per_stock:
            logger.info(f"=== Batch mode: {args.days} trading days, ~{args.days * 2} requests ===")

            if not args.nb_only:
                flow_df = fetch_fund_flow_batch(days=args.days)
                _save_with_merge(flow_df, FLOW_PATH,
                                 dedup_cols=["qlib_code", "trade_date"], label="fund flow")
                n_items += len(flow_df)
                if not flow_df.empty and "trade_date" in flow_df.columns:
                    latest_date = str(flow_df["trade_date"].max())

            if not args.flow_only:
                nb_df = fetch_northbound_batch(days=args.days)
                _save_with_merge(nb_df, NB_PATH,
                                 dedup_cols=["qlib_code", "trade_date"], label="northbound")
                n_items += len(nb_df)

            logger.info("Done!")
            write_health("fund_flow_update", HealthStatus(
                success=True,
                n_items=n_items,
                latest_date=latest_date,
                network_profile="domestic",
            ))
            return

        # ===== Per-stock mode (old, slow) =====
        logger.info("=== Per-stock mode (slow) ===")
        codes = get_all_stock_codes(top_n=args.top)

        if not args.nb_only:
            existing = load_existing_codes(FLOW_PATH) if args.incremental else set()
            flow_df = fetch_fund_flow(codes, workers=args.workers,
                                      existing_codes=existing, source=args.source)
            _save_with_merge(flow_df, FLOW_PATH,
                             dedup_cols=["qlib_code", "trade_date"], label="fund flow")
            n_items += len(flow_df)
            if not flow_df.empty and "trade_date" in flow_df.columns:
                latest_date = str(flow_df["trade_date"].max())

        if not args.flow_only:
            existing = load_existing_codes(NB_PATH) if args.incremental else set()
            nb_df = fetch_northbound(codes, workers=args.workers,
                                     existing_codes=existing, source=args.source)
            _save_with_merge(nb_df, NB_PATH,
                             dedup_cols=["qlib_code", "trade_date"], label="northbound")
            n_items += len(nb_df)

        logger.info("Done!")
        write_health("fund_flow_update", HealthStatus(
            success=True,
            n_items=n_items,
            latest_date=latest_date,
            network_profile="domestic",
        ))
    except Exception as e:
        write_health("fund_flow_update", HealthStatus(
            success=False,
            error_type=type(e).__name__,
            error_message=str(e)[:200],
        ))
        raise


if __name__ == "__main__":
    main()
