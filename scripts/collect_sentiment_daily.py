"""Daily batch sentiment collection from Xueqiu + Eastmoney Guba + 同花顺.

Collects hot-stock lists / trending discussions / concept rankings
from three retail-investor surfaces and writes them as raw JSONL.

Output:
  - data/storage/sentiment/<YYYY-MM-DD>.jsonl — one line per item
    with fields: stock_code, stock_name, source, heat, rank, date.
    Sources: xueqiu_hot, ths_hot, ths_concept, guba_hot.

2026-06-06 doc fix (P1 #4): the original header claimed
"saves to EventStore". That was untrue — this script only writes
JSONL. EventStore wiring + a derived sentiment_factors.parquet are
filed as task #164 follow-ups; until they ship, the sentiment chain
is collection-only and the factors are NOT in the production model.

Usage:
    python scripts/collect_sentiment_daily.py [--date YYYY-MM-DD] [--top-n 100]

Crontab: 16:40, network=domestic, timeout=600
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data" / "storage"
OUTPUT_DIR = DATA_DIR / "sentiment"


def fetch_xueqiu_hot(session, limit: int = 50) -> list[dict]:
    """Fetch Xueqiu hot stocks / trending discussions."""
    items = []
    try:
        # First get cookie
        session.get("https://xueqiu.com/", timeout=5)

        # Hot stocks
        url = "https://stock.xueqiu.com/v5/stock/hot_stock/list.json"
        params = {"size": limit, "type": "10"}  # 10 = A股
        resp = session.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            for item in data.get("items", []):
                code = item.get("code", "")
                name = item.get("name", "")
                if not code:
                    continue
                items.append({
                    "stock_code": code.replace("SH", "").replace("SZ", ""),
                    "stock_name": name,
                    "source": "xueqiu_hot",
                    "heat": item.get("current", 0),
                    "change_pct": item.get("percent", 0),
                })
            logger.info(f"Xueqiu hot: {len(items)} stocks")
    except Exception as e:
        logger.warning(f"Xueqiu hot failed: {e}")
    return items


def fetch_xueqiu_comments(session, stock_code: str, limit: int = 10) -> list[dict]:
    """Fetch recent Xueqiu comments for a stock."""
    items = []
    try:
        symbol = f"SH{stock_code}" if stock_code.startswith("6") else f"SZ{stock_code}"
        url = "https://xueqiu.com/query/v1/symbol/search/status.json"
        params = {"q": symbol, "count": limit, "symbol": symbol, "sort": "time"}
        resp = session.get(url, params=params, timeout=8)
        if resp.status_code == 200:
            for item in resp.json().get("list", [])[:limit]:
                text = item.get("text", "") or item.get("description", "")
                if "<" in text:
                    text = text.split("<")[0]
                if len(text) < 5:
                    continue
                items.append({
                    "stock_code": stock_code,
                    "text": text[:200],
                    "source": "xueqiu",
                    "timestamp": item.get("created_at", 0),
                    "retweet_count": item.get("retweet_count", 0),
                    "reply_count": item.get("reply_count", 0),
                })
    except Exception:
        pass
    return items


def fetch_ths_hot(session) -> list[dict]:
    """Fetch 同花顺 hot stocks / concept boards."""
    items = []

    # 同花顺人气排行
    try:
        url = "https://dq.10jqka.com.cn/fuyao/hot_list_data/out/hot_list/v1/stock"
        params = {"type": "stock", "market": "AB", "is_498": 1, "limit": 50}
        headers = {"Referer": "https://www.10jqka.com.cn/"}
        resp = session.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            for item in data.get("stock_list", []):
                code = item.get("code", "")
                name = item.get("name", "")
                if not code:
                    continue
                items.append({
                    "stock_code": code,
                    "stock_name": name,
                    "source": "ths_hot",
                    "heat": item.get("hot_num", 0),
                    "rank": item.get("order", 0),
                    "tag": item.get("tag", ""),
                })
            logger.info(f"THS hot: {len(items)} stocks")
    except Exception as e:
        logger.warning(f"THS hot failed: {e}")

    # 同花顺概念板块
    try:
        url = "https://dq.10jqka.com.cn/fuyao/hot_list_data/out/hot_list/v1/plate"
        params = {"type": "concept", "is_498": 1, "limit": 30}
        headers = {"Referer": "https://www.10jqka.com.cn/"}
        resp = session.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            for item in data.get("plate_list", []):
                items.append({
                    "stock_code": "",
                    "stock_name": item.get("name", ""),
                    "source": "ths_concept",
                    "heat": item.get("hot_num", 0),
                    "rank": item.get("order", 0),
                })
            logger.info(f"THS concepts: {len(data.get('plate_list', []))}")
    except Exception as e:
        logger.warning(f"THS concepts failed: {e}")

    return items


def fetch_eastmoney_guba_hot(session) -> list[dict]:
    """Fetch 东财股吧热帖."""
    items = []
    try:
        url = "https://gbapi.eastmoney.com/slist/search/hot.json"
        params = {"type": 1, "pageindex": 1, "pagesize": 50}
        resp = session.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("re", []):
                code = item.get("stockbar_code", "")
                title = item.get("post_title", "")
                if not title:
                    continue
                items.append({
                    "stock_code": code,
                    "text": title[:200],
                    "source": "guba_hot",
                    "read_count": item.get("post_click_count", 0),
                    "comment_count": item.get("post_comment_count", 0),
                })
            logger.info(f"Guba hot: {len(items)} posts")
    except Exception as e:
        logger.warning(f"Guba hot failed: {e}")
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--top-n", type=int, default=50, help="Top N hot stocks to get comments for")
    args = parser.parse_args()
    date = args.date

    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })
    session.trust_env = False  # Don't use system proxy

    all_items = []

    # 1. Xueqiu hot stocks
    xq_hot = fetch_xueqiu_hot(session)
    all_items.extend(xq_hot)

    # 2. 同花顺 hot stocks + concepts
    ths = fetch_ths_hot(session)
    all_items.extend(ths)

    # 3. 东财股吧热帖
    guba = fetch_eastmoney_guba_hot(session)
    all_items.extend(guba)

    # 4. Xueqiu comments for top hot stocks
    hot_codes = set()
    for item in xq_hot[:args.top_n]:
        code = item.get("stock_code", "")
        if code and code not in hot_codes:
            hot_codes.add(code)

    if hot_codes:
        logger.info(f"Fetching Xueqiu comments for {len(hot_codes)} hot stocks...")
        for code in list(hot_codes)[:20]:  # Limit to 20 to avoid rate limiting
            comments = fetch_xueqiu_comments(session, code, limit=5)
            all_items.extend(comments)
            time.sleep(0.3)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{date}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_items:
            item["date"] = date
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(all_items)} items to {output_path}")

    # Summary
    sources = {}
    for item in all_items:
        s = item.get("source", "?")
        sources[s] = sources.get(s, 0) + 1
    for s, n in sorted(sources.items(), key=lambda x: -x[1]):
        logger.info(f"  {s}: {n}")

    # Write health
    try:
        from scheduler.data_health import write_health, HealthStatus
        write_health("sentiment_daily", HealthStatus(
            success=True, n_items=len(all_items), latest_date=date,
            network_profile="domestic",
        ))
    except Exception:
        pass


if __name__ == "__main__":
    main()
