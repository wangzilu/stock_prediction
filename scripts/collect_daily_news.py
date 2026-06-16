"""Collect daily stock news from ST_CLIENT (primary) or AKShare (fallback).

Usage:
    python scripts/collect_daily_news.py [--date 2024-01-15] [--portfolio]

By default collects news for the top 100 most liquid A-share stocks.
With --portfolio, collects only for stocks in the current Top20 portfolio.

Output: data/storage/daily_news/YYYY-MM-DD.jsonl
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.settings import DATA_DIR, ST_TOKEN

logger = logging.getLogger(__name__)

NEWS_DIR = DATA_DIR / "daily_news"
NEWS_DIR.mkdir(parents=True, exist_ok=True)


def _get_st_token() -> str:
    token_file = PROJECT_ROOT / ".st_token"
    if token_file.exists():
        return token_file.read_text().strip()
    return str(ST_TOKEN or "").strip()


def _parse_st_table(resp) -> list[dict]:
    """Normalize StockToday table responses into list-of-dicts.

    StockToday endpoints have returned both a raw list and the wrapped
    ``{"data": {"fields": [...], "items": [...]}}`` shape in production.
    The daily news liquid-universe path must support both; otherwise a
    valid ST response looks empty and the pipeline falls into the flaky
    AKShare branch.
    """
    if isinstance(resp, list):
        return [r for r in resp if isinstance(r, dict)]
    if not isinstance(resp, dict):
        return []
    data = resp.get("data", resp)
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if not isinstance(data, dict):
        return []
    rows = data.get("items") or data.get("data") or data.get("list") or []
    fields = data.get("fields") or data.get("columns") or resp.get("fields") or []
    if not isinstance(rows, list):
        return []
    if rows and isinstance(rows[0], dict):
        return [r for r in rows if isinstance(r, dict)]
    if fields and rows and isinstance(rows[0], (list, tuple)):
        return [dict(zip(fields, row)) for row in rows]
    return []


def _fallback_qlib_stocks(top_n: int) -> list[dict]:
    instruments_path = DATA_DIR / "qlib_data" / "cn_data" / "instruments" / "all.txt"
    if not instruments_path.exists():
        return []
    results: list[dict] = []
    with open(instruments_path, "r", encoding="utf-8") as f:
        for line in f:
            code = line.strip().split("\t", 1)[0].upper()
            if not code.startswith(("SH", "SZ")) or len(code) < 8:
                continue
            raw = code[2:]
            results.append({
                "code": raw,
                "name": raw,
                "qlib_code": code,
                "ts_code": f"{raw}.{'SH' if code.startswith('SH') else 'SZ'}",
            })
            if len(results) >= top_n:
                break
    if results:
        logger.warning("Using local Qlib instrument fallback for %d stocks", len(results))
    return results


def get_liquid_stocks(top_n: int = 100) -> list[dict]:
    """Get top N most liquid A-share stocks.

    Tries ST_CLIENT (reliable) first, AKShare as fallback.
    """
    # Try ST_CLIENT first
    try:
        from ST_CLIENT import StockToday
        token = _get_st_token()
        if token:
            st = StockToday(token=token)
            from datetime import timedelta
            result = None
            for days_back in range(0, 10):
                date_str = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
                resp = st.bak_basic(trade_date=date_str)
                rows = _parse_st_table(resp)
                logger.debug(
                    "bak_basic(%s): type=%s, rows=%s",
                    date_str, type(resp).__name__, len(rows),
                )
                if len(rows) > 100:
                    result = rows
                    logger.info(f"bak_basic found {len(result)} stocks for {date_str}")
                    break
            if isinstance(result, list) and result:
                df = pd.DataFrame(result)
                # Columns may be 'name'/'ts_code' — verify they exist
                if "name" not in df.columns or "ts_code" not in df.columns:
                    logger.warning(f"bak_basic unexpected columns: {list(df.columns)[:10]}")
                else:
                    df = df[~df["name"].str.contains("ST|退", na=False)]
                    df = df.head(top_n)
                    results = []
                    for _, row in df.iterrows():
                        ts = str(row["ts_code"])
                        code = ts[:6]
                        prefix = "SH" if ts.endswith(".SH") else "SZ"
                        results.append({
                            "code": code,
                            "name": row["name"],
                            "qlib_code": f"{prefix}{code}",
                            "ts_code": ts,
                        })
                    if results:
                        logger.info(f"Got {len(results)} stocks from ST_CLIENT bak_basic")
                        return results
            else:
                logger.warning(f"bak_basic returned no usable data (last response type: {type(resp).__name__})")
        else:
            logger.warning("No ST_CLIENT token found")
    except Exception as e:
        logger.warning(f"ST_CLIENT stock list failed: {e}")
        import traceback
        traceback.print_exc()

    # Fallback to AKShare
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        df = df[~df["名称"].str.contains("ST|退", na=False)]
        df = df.sort_values("成交额", ascending=False).head(top_n)
        results = []
        for _, row in df.iterrows():
            code = str(row["代码"]).zfill(6)
            prefix = "SH" if code.startswith(("6", "9")) else "SZ" if code.startswith(("0", "3")) else "BJ"
            results.append({"code": code, "name": row["名称"], "qlib_code": f"{prefix}{code}"})
        logger.info(f"Got {len(results)} liquid stocks from AKShare")
        return results
    except Exception as e:
        logger.error(f"Failed to get liquid stocks: {e}")
        return _fallback_qlib_stocks(top_n)


def get_portfolio_stocks() -> list[dict]:
    """Get stocks from the current overnight snapshot (Top20 portfolio).

    Returns:
        List of dicts with keys: code, name, qlib_code

    2026-06-04 cx round 4 P1-5: pre-fix this iterated ``data`` directly,
    assuming a top-level list. The actual payload (per
    scheduler/jobs.py overnight_stock_forecasts writer) is a dict
    ``{created_at, source_date, target_date, lgb_status, groups,
    items: [...]}``. Iterating a dict yields STRING KEYS, so
    ``item.get(...)`` blew up and the function silently fell through
    to ``get_liquid_stocks()`` — i.e. ``--portfolio`` mode was secretly
    collecting full-market news every cron tick.
    """
    snapshot_path = DATA_DIR / "overnight_stock_forecasts.json"
    if not snapshot_path.exists():
        logger.warning("No overnight snapshot found, falling back to liquid stocks")
        return get_liquid_stocks()

    try:
        with open(snapshot_path) as f:
            data = json.load(f)

        # Tolerate either the current dict format (with "items") or
        # the legacy top-level list format. Anything else is a hard
        # error — do NOT silently fall back to full-market.
        if isinstance(data, dict):
            items = data.get("items")
            if not isinstance(items, list):
                raise RuntimeError(
                    "overnight snapshot is a dict but has no 'items' list "
                    "(P1-5 contract guard)"
                )
        elif isinstance(data, list):
            items = data  # legacy format
        else:
            raise RuntimeError(
                f"overnight snapshot has unexpected top-level type "
                f"{type(data).__name__}"
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
                "qlib_code": qlib_code,
            })
        logger.info(f"Got {len(results)} portfolio stocks from snapshot")
        if not results:
            # Empty portfolio after parsing → still better to fail
            # than silently expand to full market.
            raise RuntimeError(
                "overnight snapshot parsed but yielded 0 portfolio stocks"
            )
        return results
    except Exception as e:
        # Hard fail: --portfolio means the caller wants only the
        # portfolio. Silently returning full market via
        # get_liquid_stocks() defeats the request AND drowns the LLM
        # pipeline in 5000-stock fan-out (cx round 4 P1-5 + P0-2).
        logger.error(
            f"Failed to load portfolio snapshot: {e}. "
            f"Refusing to silently expand to full-market liquidity."
        )
        raise


_NEWS_RECENCY_DAYS = 7  # cx round 4 P0-2: stale-news cutoff


def _is_recent_news(publish_time: str, *, max_age_days: int = _NEWS_RECENCY_DAYS,
                    now: datetime | None = None) -> bool:
    """Return True iff publish_time is within max_age_days of now.

    Acceptable formats: ``YYYY-MM-DD``, ``YYYY-MM-DD HH:MM:SS``,
    ``YYYY/MM/DD ...``, ``YYYY-MM-DDTHH:MM:SS``. Anything else (empty
    string, unparseable, future) is rejected — the safer side of
    feeding stale noise into LLM extractors that already get RPM
    rate-limited at 1002 (see 2026-06-04 incident, task #104).
    """
    if not publish_time:
        return False
    text = str(publish_time).strip().replace("/", "-").replace("T", " ")
    # Trim trailing time/timezone if present so we can parse the date.
    head = text.split(" ", 1)[0]
    try:
        dt = datetime.strptime(head, "%Y-%m-%d")
    except ValueError:
        return False
    now = now or datetime.now()
    age = (now - dt).days
    if age < 0:
        # Future-dated news is suspicious — reject.
        return False
    return age <= max_age_days


def collect_news_for_stock(code: str, name: str, max_items: int = 10,
                            target_date: str | None = None) -> list[dict]:
    """Collect recent news for a single stock via AKShare.

    Args:
        code: 6-digit stock code, e.g. '600519'
        name: stock name for logging
        max_items: max news items to return
        target_date: YYYY-MM-DD reference date for recency check.
            Default None → uses datetime.now(), which is correct for
            live cron runs but WRONG for historical backfills.
            2026-06-06 fix: backfills (``--date 2026-05-10``) now pass
            target_date so news within max_age_days of THAT date are
            kept, not within max_age_days of today.

    Returns:
        List of news dicts with standardized fields. Items older than
        ``_NEWS_RECENCY_DAYS`` are dropped at this layer (cx round 4
        P0-2) so the LLM never sees 60%+ stale news fan-out.
    """
    # 2026-06-06 P1 fix: thread the per-call reference date into the
    # recency check so historical backfills are not contaminated by
    # "today" - 7d cutoff.
    _now_ref = None
    if target_date:
        try:
            _now_ref = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            _now_ref = None
    # Try ST_CLIENT news first (anns_d for announcements)
    try:
        from ST_CLIENT import StockToday
        token_file = PROJECT_ROOT / ".st_token"
        token = token_file.read_text().strip() if token_file.exists() else ""
        if token:
            st = StockToday(token=token)
            ts_code = f"{code}.SH" if code.startswith("6") else f"{code}.SZ"
            result = st.anns_d(ts_code=ts_code)
            if isinstance(result, list) and result:
                records = []
                for item in result:
                    pub = str(item.get("ann_date", ""))
                    if not _is_recent_news(pub, now=_now_ref):
                        continue
                    records.append({
                        "stock_code": code,
                        "stock_name": name,
                        "title": str(item.get("title", "")),
                        "content_snippet": str(item.get("content", ""))[:500],
                        "source": "交易所公告",
                        "publish_time": pub,
                        "url": str(item.get("url", "")),
                    })
                    if len(records) >= max_items:
                        break
                if records:
                    return records
    except Exception:
        pass

    # Fallback: direct Eastmoney news API (bypasses AKShare regex bug)
    try:
        import requests
        url = "https://search-api-web.eastmoney.com/search/jsonp"
        params = {
            "cb": "jQuery_cb",
            "param": (
                f'{{"uid":"","keyword":"{code}","type":["cmsArticleWebOld"],'
                f'"client":"web","clientType":"web","clientVersion":"curr",'
                f'"param":{{"cmsArticleWebOld":{{"searchScope":"default",'
                f'"sort":"default","pageIndex":1,"pageSize":{max_items},'
                f'"preTag":"<em>","postTag":"</em>"}}}}}}'
            ),
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Referer": "https://so.eastmoney.com/",
        }
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        text = resp.text
        # Strip JSONP wrapper: jQuery_cb({...})
        if text.startswith("jQuery_cb("):
            text = text[len("jQuery_cb("):-1]
        import json as _json
        data = _json.loads(text)
        raw = data.get("result", {}).get("cmsArticleWebOld", [])
        # API returns either a list directly or a dict with "list" key
        if isinstance(raw, list):
            articles = raw
        elif isinstance(raw, dict):
            articles = raw.get("list", [])
        else:
            articles = []

        records = []
        for item in articles:
            pub = item.get("date", "")
            if not _is_recent_news(pub, now=_now_ref):
                continue
            title = item.get("title", "").replace("<em>", "").replace("</em>", "")
            content = item.get("content", "").replace("<em>", "").replace("</em>", "")
            records.append({
                "stock_code": code,
                "stock_name": name,
                "title": title,
                "content_snippet": content[:500],
                "source": item.get("mediaName", "eastmoney"),
                "publish_time": pub,
                "url": item.get("url", ""),
            })
            if len(records) >= max_items:
                break
        return records

    except Exception as e:
        logger.warning(f"Failed to collect news for {code} ({name}): {e}")
        return []


def collect_daily_news(
    target_date: str = None,
    use_portfolio: bool = False,
    top_n: int = 100,
) -> Path:
    """Collect news for all target stocks and save as JSONL.

    Args:
        target_date: YYYY-MM-DD, defaults to today
        use_portfolio: if True, use portfolio stocks instead of liquid stocks
        top_n: number of liquid stocks to use (ignored if use_portfolio=True)

    Returns:
        Path to the saved JSONL file
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    output_path = NEWS_DIR / f"{target_date}.jsonl"
    manifest_path = NEWS_DIR / f"{target_date}.manifest.json"

    # 2026-06-04 cx round 16 P1-3: pre-fix the skip rule was
    # "≥1000 lines" — pure row count, no version check. A file written
    # by a pre-fix collector (no recency cutoff, no portfolio-snapshot
    # contract guard) sat on disk with 1000+ rows and got accepted as
    # "already done", bypassing all the round 4 fixes. Now: a sibling
    # manifest records collector_version, filter_version, recency
    # cutoff. Mismatched manifest (or missing one) forces re-collection.
    COLLECTOR_VERSION = 4  # bump when collector logic changes substantively
    FILTER_VERSION = 4     # bump when filter logic changes substantively
    if output_path.exists():
        n_existing = sum(1 for _ in open(output_path))
        manifest_ok = False
        if manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text())
                manifest_ok = (
                    m.get("collector_version") == COLLECTOR_VERSION
                    and m.get("filter_version") == FILTER_VERSION
                    and int(m.get("recency_cutoff_days", -1)) == _NEWS_RECENCY_DAYS
                )
            except Exception:
                manifest_ok = False
        if manifest_ok and n_existing >= 1000:
            logger.info(f"News already collected for {target_date} ({n_existing} items, manifest matches), skipping")
            return output_path
        if not manifest_ok:
            logger.warning(
                f"Existing {output_path.name} has no/stale manifest "
                f"(collector_v={COLLECTOR_VERSION}, filter_v={FILTER_VERSION}, "
                f"recency={_NEWS_RECENCY_DAYS}d) — re-collecting."
            )
        else:
            logger.warning(f"Previous collection only got {n_existing} items, re-collecting")
        os.remove(str(output_path))
        if manifest_path.exists():
            os.remove(str(manifest_path))

    # Get target stocks
    if use_portfolio:
        stocks = get_portfolio_stocks()
    else:
        stocks = get_liquid_stocks(top_n)

    if not stocks:
        logger.error("No stocks to collect news for")
        return output_path

    logger.info(f"Collecting news for {len(stocks)} stocks on {target_date}")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    all_results = []
    results_lock = threading.Lock()

    def _fetch_one(stock):
        items = collect_news_for_stock(
            stock["code"], stock["name"], max_items=3,
            target_date=target_date,
        )
        for item in items:
            item["qlib_code"] = stock["qlib_code"]
            item["collect_date"] = target_date
        time.sleep(0.1)  # light rate limit per thread
        return items

    n_workers = 8
    done_count = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_fetch_one, s): s for s in stocks}
        for future in as_completed(futures):
            done_count += 1
            items = future.result()
            if items:
                with results_lock:
                    all_results.extend(items)
            if done_count % 100 == 0:
                logger.info(f"  Progress: {done_count}/{len(stocks)} stocks, {len(all_results)} news items")

    # Write all at once (sorted for reproducibility)
    all_results.sort(key=lambda x: x.get("stock_code", ""))
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # cx round 16 P1-3: manifest sidecar so the next-day skip-check
    # can verify schema/version compatibility before reusing this
    # file. See the top-of-function skip block.
    manifest = {
        "target_date": target_date,
        "collector_version": COLLECTOR_VERSION,
        "filter_version": FILTER_VERSION,
        "recency_cutoff_days": _NEWS_RECENCY_DAYS,
        "n_items": len(all_results),
        "use_portfolio": bool(use_portfolio),
        "collected_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    logger.info(f"Collected {len(all_results)} news items for {len(stocks)} stocks -> {output_path}")
    return output_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Collect daily stock news from AKShare")
    parser.add_argument("--date", type=str, default=None, help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--portfolio", action="store_true", help="Use portfolio stocks instead of top liquid")
    parser.add_argument("--top-n", type=int, default=100, help="Number of liquid stocks (default: 100)")
    args = parser.parse_args()

    collect_daily_news(
        target_date=args.date,
        use_portfolio=args.portfolio,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
