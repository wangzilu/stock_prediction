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

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

NEWS_DIR = DATA_DIR / "daily_news"
NEWS_DIR.mkdir(parents=True, exist_ok=True)


def get_liquid_stocks(top_n: int = 100) -> list[dict]:
    """Get top N most liquid A-share stocks.

    Tries ST_CLIENT (reliable) first, AKShare as fallback.
    """
    # Try ST_CLIENT first
    try:
        from ST_CLIENT import StockToday
        token_file = PROJECT_ROOT / ".st_token"
        token = token_file.read_text().strip() if token_file.exists() else ""
        if token:
            st = StockToday(token=token)
            from datetime import timedelta
            result = None
            for days_back in range(0, 10):
                date_str = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
                resp = st.bak_basic(trade_date=date_str)
                logger.debug(f"bak_basic({date_str}): type={type(resp).__name__}, len={len(resp) if isinstance(resp, list) else 'N/A'}")
                if isinstance(resp, list) and len(resp) > 100:
                    result = resp
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
            logger.warning("No ST_CLIENT token found at .st_token")
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
        return []


def get_portfolio_stocks() -> list[dict]:
    """Get stocks from the current overnight snapshot (Top20 portfolio).

    Returns:
        List of dicts with keys: code, name, qlib_code
    """
    snapshot_path = DATA_DIR / "overnight_stock_forecasts.json"
    if not snapshot_path.exists():
        logger.warning("No overnight snapshot found, falling back to liquid stocks")
        return get_liquid_stocks()

    try:
        with open(snapshot_path) as f:
            data = json.load(f)

        results = []
        for item in data:
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
        return results
    except Exception as e:
        logger.error(f"Failed to load portfolio snapshot: {e}")
        return get_liquid_stocks()


def collect_news_for_stock(code: str, name: str, max_items: int = 10) -> list[dict]:
    """Collect recent news for a single stock via AKShare.

    Args:
        code: 6-digit stock code, e.g. '600519'
        name: stock name for logging
        max_items: max news items to return

    Returns:
        List of news dicts with standardized fields
    """
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
                for item in result[:max_items]:
                    records.append({
                        "stock_code": code,
                        "stock_name": name,
                        "title": str(item.get("title", "")),
                        "content_snippet": str(item.get("content", ""))[:500],
                        "source": "交易所公告",
                        "publish_time": str(item.get("ann_date", "")),
                        "url": str(item.get("url", "")),
                    })
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
        for item in articles[:max_items]:
            title = item.get("title", "").replace("<em>", "").replace("</em>", "")
            content = item.get("content", "").replace("<em>", "").replace("</em>", "")
            records.append({
                "stock_code": code,
                "stock_name": name,
                "title": title,
                "content_snippet": content[:500],
                "source": item.get("mediaName", "eastmoney"),
                "publish_time": item.get("date", ""),
                "url": item.get("url", ""),
            })
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

    # Skip if already collected today with sufficient data
    if output_path.exists():
        n_existing = sum(1 for _ in open(output_path))
        if n_existing >= 1000:  # minimum viable for full-A coverage
            logger.info(f"News already collected for {target_date} ({n_existing} items), skipping")
            return output_path
        else:
            logger.warning(f"Previous collection only got {n_existing} items, re-collecting")
            os.remove(str(output_path))

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
        items = collect_news_for_stock(stock["code"], stock["name"], max_items=3)
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
