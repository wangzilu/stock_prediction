"""Collect A-share sell-side research reports (analyst rating + EPS revision).

SPIKE 2026-06-09 — LLM Channel #1 (research_rating).

Primary source: ``akshare.stock_research_report_em`` (Eastmoney 个股研报 page).
That call returns the broker's published rating, EPS forecasts for the next 3
fiscal years, the implied target P/E, the publishing organisation, and a
direct link to the report PDF — i.e. the structured-text we need is already
present in the table, so the LLM call is for *report-title summarisation*
and *target-price extraction from the PDF* (not for the rating itself, which
is already structured).

Backfill sources (NOT scoped here, documented for future work):
  * 慧博 (htbencharm.com) — broader broker coverage, scrape-heavy
  * 同花顺 i问财 — has a structured rating change API but rate-limited
  * 万得 / WindEDB — institutional only, out of scope

Output: ``data/storage/research_reports/YYYY-MM-DD.jsonl`` — one row per
(stock, report). Each row is the raw Eastmoney record with normalised
field names. The LLM extractor (``factors/research_rating_extractor.py``)
reads this file and emits per-report structured JSON to
``data/storage/research_rating_extracted/YYYY-MM-DD.jsonl``.

PIT discipline
--------------
``collected_at`` is the harvest timestamp (= the day this collector ran).
The signal date is ``collected_at + 1 BDay`` (T+1) — i.e. a report whose
``report_date`` was today gets folded into the factor cache for tomorrow's
session, never today's. See ``scripts/build_research_rating_factors.py`` for
the actual lag enforcement.

Usage
-----
    # daily cron mode (full portfolio sweep)
    python scripts/collect_research_reports.py

    # backfill for a specific date (note: Eastmoney returns historical
    # reports keyed by stock, so the --date flag only controls the
    # output filename, not the API filter)
    python scripts/collect_research_reports.py --date 2026-06-09

    # narrow to overnight portfolio (Top20) to save API calls
    python scripts/collect_research_reports.py --portfolio
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)

RESEARCH_DIR = DATA_DIR / "research_reports"
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

# Mirror collect_daily_news.py — only ingest reports published within
# this many days of target_date. Older reports are stale (the rating /
# EPS forecast has either been re-issued or the analyst quit covering).
_REPORT_RECENCY_DAYS = 30

# Eastmoney's stock_research_report_em returns up to ~750 historical
# rows per stock; we slice to the recency window in-script.
_MAX_REPORTS_PER_STOCK = 50

# Manifest schema version. Bump when the collector logic changes
# substantively (e.g. switch to 慧博 as primary). The skip-check in
# collect_research_reports() refuses to reuse a file whose manifest
# doesn't match.
COLLECTOR_VERSION = 1
FILTER_VERSION = 1


# Eastmoney 东财评级 → canonical English rating tag. We keep the
# raw Chinese string in the JSONL too (``raw_rating``) so the LLM
# extractor can reason about edge cases like "增持-A" / "持有".
RATING_MAP = {
    "买入": "buy",
    "增持": "outperform",
    "推荐": "buy",
    "强烈推荐": "strong_buy",
    "谨慎推荐": "outperform",
    "持有": "hold",
    "中性": "hold",
    "观望": "hold",
    "减持": "underperform",
    "卖出": "sell",
    "回避": "sell",
}


def _canonicalise_rating(raw: str) -> str:
    """Map a raw Eastmoney rating string to a canonical English tag.

    Defensive: rating cells frequently contain noise like "买入-A" /
    "增持(调高)" so we look for the *first* matching Chinese keyword
    rather than insisting on an exact match. Returns "unknown" when no
    keyword matches.
    """
    if not raw or not isinstance(raw, str):
        return "unknown"
    text = raw.strip()
    for k, v in RATING_MAP.items():
        if k in text:
            return v
    return "unknown"


def get_portfolio_stocks() -> list[dict]:
    """Get current Top20 portfolio stocks from the overnight snapshot.

    Mirrors ``scripts/collect_daily_news.py:get_portfolio_stocks`` —
    same cx round 4 P1-5 contract guard (overnight snapshot is a
    dict with an ``items`` list, NOT a top-level list). Hard-fails
    rather than silently expanding to full market.
    """
    snapshot_path = DATA_DIR / "overnight_stock_forecasts.json"
    if not snapshot_path.exists():
        raise RuntimeError(
            "No overnight snapshot — refuse to silently expand to full "
            "market (would fan out 5000+ AKShare calls)."
        )
    with open(snapshot_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        items = data.get("items")
        if not isinstance(items, list):
            raise RuntimeError(
                "overnight snapshot missing 'items' list (P1-5 guard)"
            )
    elif isinstance(data, list):
        items = data
    else:
        raise RuntimeError(
            f"overnight snapshot has unexpected type {type(data).__name__}"
        )
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        qlib_code = item.get("code", "")
        if len(qlib_code) < 8:
            continue
        code = qlib_code[2:]
        results.append({
            "code": code,
            "name": item.get("name", ""),
            "qlib_code": qlib_code.lower(),  # canonical lowercase
        })
    if not results:
        raise RuntimeError("portfolio snapshot parsed but 0 stocks")
    return results


def get_liquid_stocks(top_n: int = 300) -> list[dict]:
    """Get top-N liquid A-share stocks via ST_CLIENT (preferred) then AKShare.

    Mirrors ``collect_daily_news.get_liquid_stocks`` but defaults to a
    larger universe (300, not 100) because research-report coverage is
    skewed to mid/large cap and we want decent stock-level breadth for
    the rating-change factor. Backed by ST_CLIENT bak_basic (single-call
    full-market) per ``memory/feedback_st_over_baostock.md``.
    """
    try:
        from ST_CLIENT import StockToday
        token_file = PROJECT_ROOT / ".st_token"
        token = token_file.read_text().strip() if token_file.exists() else ""
        if token:
            st = StockToday(token=token)
            for days_back in range(0, 10):
                date_str = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
                resp = st.bak_basic(trade_date=date_str)
                if isinstance(resp, list) and len(resp) > 100:
                    break
            else:
                resp = None
            if isinstance(resp, list) and resp:
                df = pd.DataFrame(resp)
                if "name" in df.columns and "ts_code" in df.columns:
                    df = df[~df["name"].str.contains("ST|退", na=False)]
                    df = df.head(top_n)
                    out = []
                    for _, row in df.iterrows():
                        ts = str(row["ts_code"])
                        code = ts[:6]
                        prefix = "SH" if ts.endswith(".SH") else "SZ"
                        out.append({
                            "code": code,
                            "name": row["name"],
                            "qlib_code": f"{prefix}{code}".lower(),
                            "ts_code": ts,
                        })
                    if out:
                        return out
    except Exception as e:
        logger.warning("ST_CLIENT failed: %s", e)
    # Fallback: AKShare spot
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        df = df[~df["名称"].str.contains("ST|退", na=False)]
        df = df.sort_values("成交额", ascending=False).head(top_n)
        out = []
        for _, row in df.iterrows():
            code = str(row["代码"]).zfill(6)
            prefix = "SH" if code.startswith(("6", "9")) else "SZ"
            out.append({
                "code": code,
                "name": row["名称"],
                "qlib_code": f"{prefix}{code}".lower(),
            })
        return out
    except Exception as e:
        logger.error("Failed to get liquid stocks: %s", e)
        return []


def _is_recent_report(report_date: str, *, max_age_days: int,
                      now: datetime | None = None) -> bool:
    """Return True iff report_date is within max_age_days of now."""
    if not report_date:
        return False
    text = str(report_date).strip().replace("/", "-")
    head = text.split(" ", 1)[0]
    try:
        dt = datetime.strptime(head, "%Y-%m-%d")
    except ValueError:
        return False
    now = now or datetime.now()
    age = (now - dt).days
    if age < 0:
        return False
    return age <= max_age_days


def fetch_reports_for_stock(code: str, name: str, qlib_code: str,
                             target_date: str | None = None,
                             max_items: int = _MAX_REPORTS_PER_STOCK) -> list[dict]:
    """Fetch sell-side research reports for one stock via Eastmoney.

    Returns a list of normalised dicts. The columns out of
    ``ak.stock_research_report_em`` are Chinese; we rename to the
    English keys downstream extractors expect. SSL errors / API
    timeouts are caught — a single stock failure should not abort the
    whole sweep.

    PIT note: ``collected_at`` is the harvest moment (datetime.now()),
    NOT the report's publish date. The factor builder uses
    ``collected_at`` + 1 BDay as the signal date so a report whose
    publish-date is today is never used to predict today's return.
    """
    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare not installed")
        return []

    _now_ref = None
    if target_date:
        try:
            _now_ref = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            _now_ref = None

    try:
        df = ak.stock_research_report_em(symbol=code)
    except Exception as e:
        logger.warning("Eastmoney research_report fetch failed for %s (%s): %s",
                       code, name, str(e)[:120])
        return []

    if df is None or df.empty:
        return []

    records = []
    collected_at = datetime.now().isoformat(timespec="seconds")
    for _, row in df.iterrows():
        report_date = str(row.get("日期", ""))
        if not _is_recent_report(report_date, max_age_days=_REPORT_RECENCY_DAYS,
                                 now=_now_ref):
            continue
        raw_rating = str(row.get("东财评级", "")).strip()
        rec = {
            "stock_code": code,
            "stock_name": name,
            "qlib_code": qlib_code,
            "report_date": report_date,
            "report_title": str(row.get("报告名称", "")).strip(),
            "broker": str(row.get("机构", "")).strip(),
            "raw_rating": raw_rating,
            "canonical_rating": _canonicalise_rating(raw_rating),
            "industry": str(row.get("行业", "")).strip(),
            "report_pdf_url": str(row.get("报告PDF链接", "")).strip(),
            # EPS forecasts — the column names embed the fiscal-year
            # number; we read them by position to be year-agnostic.
            "eps_y1": _safe_float(row.get("2026-盈利预测-收益")),
            "pe_y1": _safe_float(row.get("2026-盈利预测-市盈率")),
            "eps_y2": _safe_float(row.get("2027-盈利预测-收益")),
            "pe_y2": _safe_float(row.get("2027-盈利预测-市盈率")),
            "eps_y3": _safe_float(row.get("2028-盈利预测-收益")),
            "pe_y3": _safe_float(row.get("2028-盈利预测-市盈率")),
            "n_reports_last_month": _safe_int(row.get("近一月个股研报数")),
            "collected_at": collected_at,
            "source": "eastmoney",
        }
        records.append(rec)
        if len(records) >= max_items:
            break
    return records


def _safe_float(v) -> float | None:
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> int | None:
    try:
        if v is None or pd.isna(v):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def collect_research_reports(target_date: str | None = None,
                              use_portfolio: bool = False,
                              top_n: int = 300) -> Path:
    """Collect research reports for target stocks and save as JSONL.

    Returns the JSONL path. A manifest sidecar (``<date>.manifest.json``)
    records collector_version / filter_version / recency_cutoff so the
    next-day skip-check can verify schema compatibility — mirrors the
    cx round 16 P1-3 fix in collect_daily_news.
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    output_path = RESEARCH_DIR / f"{target_date}.jsonl"
    manifest_path = RESEARCH_DIR / f"{target_date}.manifest.json"

    if output_path.exists():
        manifest_ok = False
        if manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text())
                manifest_ok = (
                    m.get("collector_version") == COLLECTOR_VERSION
                    and m.get("filter_version") == FILTER_VERSION
                    and int(m.get("recency_cutoff_days", -1)) == _REPORT_RECENCY_DAYS
                )
            except Exception:
                manifest_ok = False
        n_existing = sum(1 for _ in open(output_path))
        if manifest_ok and n_existing > 0:
            logger.info("Research reports already collected for %s (%d rows, manifest matches), skipping",
                        target_date, n_existing)
            return output_path
        logger.warning("Re-collecting %s (manifest mismatch or empty)", target_date)
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)

    if use_portfolio:
        stocks = get_portfolio_stocks()
    else:
        stocks = get_liquid_stocks(top_n)
    if not stocks:
        logger.error("No stocks to collect reports for")
        return output_path

    logger.info("Collecting research reports for %d stocks on %s", len(stocks), target_date)

    all_records: list[dict] = []
    lock = threading.Lock()

    def _fetch_one(stock):
        items = fetch_reports_for_stock(
            stock["code"], stock["name"], stock["qlib_code"],
            target_date=target_date,
        )
        time.sleep(0.15)  # gentle per-thread rate limit (Eastmoney has 429s)
        return items

    done = 0
    with ThreadPoolExecutor(max_workers=4) as executor:  # Eastmoney rate limits more aggressively than news API
        futures = {executor.submit(_fetch_one, s): s for s in stocks}
        for fut in as_completed(futures):
            done += 1
            items = fut.result()
            if items:
                with lock:
                    all_records.extend(items)
            if done % 50 == 0:
                logger.info("  Progress: %d/%d stocks, %d records", done, len(stocks), len(all_records))

    all_records.sort(key=lambda x: (x.get("stock_code", ""), x.get("report_date", "")))
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "target_date": target_date,
        "collector_version": COLLECTOR_VERSION,
        "filter_version": FILTER_VERSION,
        "recency_cutoff_days": _REPORT_RECENCY_DAYS,
        "n_records": len(all_records),
        "n_stocks_swept": len(stocks),
        "use_portfolio": bool(use_portfolio),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    logger.info("Collected %d research report records for %d stocks -> %s",
                len(all_records), len(stocks), output_path)
    return output_path


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Collect A-share sell-side research reports (rating + EPS).",
    )
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--portfolio", action="store_true",
                        help="Use overnight portfolio (Top20) instead of liquid universe")
    parser.add_argument("--top-n", type=int, default=300,
                        help="Number of liquid stocks (default: 300)")
    args = parser.parse_args()

    collect_research_reports(
        target_date=args.date,
        use_portfolio=args.portfolio,
        top_n=args.top_n,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
