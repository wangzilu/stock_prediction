"""Daily regime data refresh — pull day-frequency risk indicators.

Pulls data that changes daily (not monthly like PMI/CPI):
  - Margin detail (融资融券余额) — leverage signal      [CRITICAL]
  - Limit-down list (跌停统计) — microcap crash signal  [CRITICAL]
  - HSGT moneyflow (北向资金) — foreign flow signal     [CRITICAL]
  - IC/IM/IF futures via AKShare (no auth)               [non-critical]
  - USD/CNY via AKShare                                  [non-critical]

Monthly data (PMI/CPI/M2/Shibor) stays on weekly Saturday refresh.

Usage:
    python scripts/update_regime_daily.py

2026-06-05 (Phase A.6, fix A6-3): the previous version wrote
``HealthStatus(success=True, n_items=5)`` no matter what — even when
every ST_CLIENT sub-source raised and every AKShare call silently
exited. That made `regime_daily_update` a permanent green light and
every downstream freshness gate anchored to it lied. The new flow:

  - Each sub-collector returns ``{ok, n_rows, latest_date, error}``.
  - ``CRITICAL_SOURCES = {margin_detail, limit_list_d, moneyflow_hsgt}``
    — any one failing flips ``success`` to ``False`` and
    ``partial`` to ``True``.
  - Non-critical (futures, USD/CNY) failures flip ``partial`` to
    ``True`` but keep ``success`` truthful per critical health.
  - ``latest_date`` on the aggregate health row is the MIN of the
    critical sources' own latest_date (= worst-case freshness).
  - Each sub-source's individual freshness is recorded in
    ``extra.latest_date_<source>`` so downstream code can decide
    per-feature whether to downgrade.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"

# Sub-sources whose failure must flip the aggregate health row to
# ``success=False``. These are the ones that downstream
# ``cross_market_regime`` features and the production gate rely on.
CRITICAL_SOURCES = ("margin_detail", "limit_list_d", "moneyflow_hsgt")


@dataclass
class SubResult:
    """Per-sub-source health record."""
    source: str
    ok: bool = False
    n_rows: int = 0
    latest_date: str = ""
    error: str = ""

    def to_extra_keys(self) -> dict:
        return {
            f"latest_date_{self.source}": self.latest_date,
            f"ok_{self.source}": self.ok,
            f"n_rows_{self.source}": self.n_rows,
            f"error_{self.source}": self.error[:120],
        }


def get_st_client():
    token = Path(PROJECT_ROOT / ".st_token").read_text().strip()
    from ST_CLIENT import StockToday
    return StockToday(token=token)


def _extract_items(result) -> list | None:
    """Extract row list from ST_CLIENT TuShare-compatible envelope.

    Real shape is ``{"code": 0, "data": {"fields": [...], "items": [...]}}``.
    The previous version checked ``isinstance(data, list)`` against the wrapper
    dict, which is always False, so every call silently returned
    "no data" and the parquet stopped updating at 2026-05-12. Fixed
    2026-06-16 — see `docs/regime_t_plus_1_silent_fail_20260616.md`.
    """
    if not isinstance(result, dict):
        return None
    inner = result.get("data")
    # Direct list form (older API or already-unwrapped)
    if isinstance(inner, list):
        return inner
    # Envelope form {fields, items}
    if isinstance(inner, dict):
        items = inner.get("items")
        if isinstance(items, list):
            return items
    return None


def update_margin(st, date: str) -> SubResult:
    """Append today's margin data to existing parquet, return per-source health."""
    logger.info("  Updating margin_detail...")
    res = SubResult(source="margin_detail")
    try:
        result = st.margin_detail(trade_date=date.replace("-", ""))
        items = _extract_items(result)
        if items:
            new_df = pd.DataFrame(items)
            for col in new_df.columns:
                if new_df[col].dtype == object:
                    new_df[col] = new_df[col].astype(str)

            path = DATA_DIR / "st_margin_detail.parquet"
            if path.exists():
                old_df = pd.read_parquet(path)
                combined = pd.concat([old_df, new_df], ignore_index=True)
                if "trade_date" in combined.columns and "ts_code" in combined.columns:
                    combined = combined.drop_duplicates(
                        subset=["trade_date", "ts_code"], keep="last"
                    )
                combined.to_parquet(str(path), index=False)
            else:
                new_df.to_parquet(str(path), index=False)

            logger.info(f"    ✅ margin: +{len(new_df)} records for {date}")
            res.ok = True
            res.n_rows = len(new_df)
            res.latest_date = date
            return res
        logger.info(f"    margin: no data for {date}")
        res.error = f"no data for {date}"
    except Exception as e:
        logger.warning(f"    margin: {e}")
        res.error = str(e)
    return res


def update_limit_list(st, date: str) -> SubResult:
    """Append today's limit-down data, return per-source health."""
    logger.info("  Updating limit_list_d...")
    res = SubResult(source="limit_list_d")
    try:
        result = st.limit_list_d(trade_date=date.replace("-", ""))
        items = _extract_items(result)
        if items:
            new_df = pd.DataFrame(items)
            for col in new_df.columns:
                if new_df[col].dtype == object:
                    new_df[col] = new_df[col].astype(str)

            path = DATA_DIR / "st_limit_list_d.parquet"
            if path.exists():
                old_df = pd.read_parquet(path)
                combined = pd.concat([old_df, new_df], ignore_index=True)
                if "trade_date" in combined.columns and "ts_code" in combined.columns:
                    combined = combined.drop_duplicates(
                        subset=["trade_date", "ts_code"], keep="last"
                    )
                combined.to_parquet(str(path), index=False)
            else:
                new_df.to_parquet(str(path), index=False)

            logger.info(f"    ✅ limit_list: +{len(new_df)} records for {date}")
            res.ok = True
            res.n_rows = len(new_df)
            res.latest_date = date
            return res
        logger.info(f"    limit_list: no data for {date}")
        res.error = f"no data for {date}"
    except Exception as e:
        logger.warning(f"    limit_list: {e}")
        res.error = str(e)
    return res


def update_hsgt(st, date: str) -> SubResult:
    """Append today's northbound flow data, return per-source health."""
    logger.info("  Updating moneyflow_hsgt...")
    res = SubResult(source="moneyflow_hsgt")
    try:
        result = st.moneyflow_hsgt(trade_date=date.replace("-", ""))
        items = _extract_items(result)
        if items:
            new_df = pd.DataFrame(items)
            for col in new_df.columns:
                if new_df[col].dtype == object:
                    new_df[col] = new_df[col].astype(str)

            path = DATA_DIR / "st_moneyflow_hsgt.parquet"
            if path.exists():
                old_df = pd.read_parquet(path)
                combined = pd.concat([old_df, new_df], ignore_index=True)
                if "trade_date" in combined.columns:
                    combined = combined.drop_duplicates(
                        subset=["trade_date"], keep="last"
                    )
                combined.to_parquet(str(path), index=False)
            else:
                new_df.to_parquet(str(path), index=False)

            logger.info(f"    ✅ hsgt: +{len(new_df)} records for {date}")
            res.ok = True
            res.n_rows = len(new_df)
            res.latest_date = date
            return res
        logger.info(f"    hsgt: no data for {date}")
        res.error = f"no data for {date}"
    except Exception as e:
        logger.warning(f"    hsgt: {e}")
        res.error = str(e)
    return res


def update_futures_akshare() -> SubResult:
    """Update IC/IM/IF futures via AKShare (no auth needed)."""
    logger.info("  Updating futures (AKShare)...")
    res = SubResult(source="ak_futures")
    try:
        import akshare as ak
        import warnings; warnings.filterwarnings("ignore")
        any_ok = False
        latest_seen = ""
        for symbol, name in [("IC0", "IC"), ("IM0", "IM"), ("IF0", "IF")]:
            path = DATA_DIR / f"ak_futures_{symbol.lower()}.parquet"
            try:
                df = ak.futures_main_sina(symbol=symbol)
                if df is not None and not df.empty:
                    df.to_parquet(str(path), index=False)
                    logger.info(f"    ✅ {name}: {len(df)} rows")
                    any_ok = True
                    res.n_rows += len(df)
                    # AKShare uses '日期' column; try to capture latest
                    for col in ("date", "日期"):
                        if col in df.columns:
                            try:
                                latest_seen = max(latest_seen, str(df[col].max())[:10])
                            except Exception:
                                pass
                            break
            except Exception as e:
                logger.warning(f"    {name}: {e}")
                res.error = (res.error + f"; {name}: {e}").strip("; ")
        res.ok = any_ok
        res.latest_date = latest_seen
        if not any_ok and not res.error:
            res.error = "all 3 AKShare futures sub-calls returned empty"
    except Exception as e:
        logger.warning(f"    AKShare import failed: {e}")
        res.error = f"akshare import failed: {e}"
    return res


def update_usdcny_akshare() -> SubResult:
    """Update USD/CNY via AKShare."""
    logger.info("  Updating USD/CNY (AKShare)...")
    res = SubResult(source="ak_usdcny")
    try:
        import akshare as ak
        import warnings; warnings.filterwarnings("ignore")
        path = DATA_DIR / "ak_usdcny.parquet"
        df = ak.currency_boc_sina(symbol="美元", start_date="20210101", end_date="20261231")
        if df is not None and not df.empty:
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str)
            df.to_parquet(str(path), index=False)
            logger.info(f"    ✅ {len(df)} rows")
            res.ok = True
            res.n_rows = len(df)
            for col in ("date", "日期"):
                if col in df.columns:
                    try:
                        res.latest_date = str(df[col].max())[:10]
                    except Exception:
                        pass
                    break
        else:
            res.error = "AKShare currency_boc_sina returned empty"
    except Exception as e:
        logger.warning(f"    USD/CNY: {e}")
        res.error = str(e)
    return res


def _aggregate_health(results: list[SubResult], today: str) -> dict:
    """Build the aggregate health-status fields from per-sub-source results.

    Critical sources gate ``success``; non-critical only gate
    ``partial``. ``latest_date`` is the MIN among critical sources
    whose ok=True; if any critical source failed, the row is success
    False and latest_date is empty so downstream readers cannot fall
    back to a stale latest_date silently.
    """
    by_source = {r.source: r for r in results}
    critical_status = [by_source.get(s) for s in CRITICAL_SOURCES]
    critical_ok = [r for r in critical_status if r is not None and r.ok]
    all_critical_ok = len(critical_ok) == len(CRITICAL_SOURCES)

    non_critical_failures = [
        r for r in results
        if r.source not in CRITICAL_SOURCES and not r.ok
    ]

    success = all_critical_ok
    # ``partial`` = anything degraded but not full failure
    partial = (
        success and (len(non_critical_failures) > 0)
    ) or (
        not success and any(r.ok for r in critical_status if r is not None)
    )

    if all_critical_ok:
        latest_date = min(r.latest_date for r in critical_ok)
    else:
        latest_date = ""

    extra: dict = {}
    for r in results:
        extra.update(r.to_extra_keys())
    extra["critical_sources"] = list(CRITICAL_SOURCES)
    extra["non_critical_failures"] = [r.source for r in non_critical_failures]
    extra["aggregate_today"] = today

    n_items = sum(r.n_rows for r in results)
    error_message = ""
    if not success:
        failed_critical = [
            r.source for r in critical_status if r is None or not r.ok
        ]
        error_message = "critical sub-source(s) failed: " + ",".join(failed_critical)

    return {
        "success": success,
        "partial": partial,
        "n_items": n_items,
        "latest_date": latest_date,
        "error_message": error_message,
        "extra": extra,
    }


def _prev_trading_day(today_str: str) -> str:
    """Walk back from ``today_str`` to the prior trading day (Mon-Fri).

    Tushare publishes margin_detail/limit_list_d/moneyflow_hsgt **T+1** —
    on a given calendar day, the latest available trade_date is the
    previous trading day. Without this shift, the 18:05 cron tried to
    fetch ``today`` and got 0 rows every weekday, and the parquet
    silently stopped updating after 2026-05-12. See
    docs/regime_t_plus_1_silent_fail_20260616.md.

    Holidays not honored here — Tushare returns 0 rows for non-trading
    days and we fall back further; for now Mon-Fri is sufficient.
    """
    d = datetime.strptime(today_str, "%Y-%m-%d")
    # Walk back at least 1 day
    d = d - timedelta(days=1)
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d = d - timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def main():
    from scheduler.data_health import write_health, HealthStatus

    today = datetime.now().strftime("%Y-%m-%d")
    # 2026-06-16: critical ST sources are T+1 publish — fetch previous
    # trading day, not today. Aggregate health latest_date still reflects
    # the actual trade_date fetched (= previous trading day) so downstream
    # freshness gates see the truth.
    st_fetch_date = _prev_trading_day(today)
    logger.info(f"=== Daily Regime Data Update: today={today} st_fetch_date={st_fetch_date} (T+1 lag) ===")

    results: list[SubResult] = []

    # ST_CLIENT data — handle a top-level ST_CLIENT failure as failure
    # of all three critical sub-sources.
    try:
        st = get_st_client()
    except Exception as e:
        logger.error(f"ST_CLIENT init failed: {e}")
        for source in CRITICAL_SOURCES:
            results.append(SubResult(
                source=source, ok=False, n_rows=0, latest_date="",
                error=f"ST_CLIENT init failed: {e}",
            ))
        st = None

    if st is not None:
        results.append(update_margin(st, st_fetch_date))
        results.append(update_limit_list(st, st_fetch_date))
        results.append(update_hsgt(st, st_fetch_date))

    # AKShare data (no auth) — these stay on `today` semantics; AKShare
    # futures + USDCNY are intraday and return today's quote.
    results.append(update_futures_akshare())
    results.append(update_usdcny_akshare())

    logger.info("Done!")

    agg = _aggregate_health(results, today)
    write_health("regime_daily_update", HealthStatus(
        success=agg["success"],
        partial=agg["partial"],
        n_items=agg["n_items"],
        latest_date=agg["latest_date"],
        error_message=agg["error_message"],
        network_profile="domestic",
        extra=agg["extra"],
    ))

    # Surface failure via non-zero exit so the cron wrapper does not
    # mark job_status green while the data lies behind a False health.
    if not agg["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
