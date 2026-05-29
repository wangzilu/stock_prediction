"""Global Industry News Collector — Phase 4U Day2.

Collects industry-level news from GDELT DOC API and Google News RSS
for supply-chain topics (ai_server, apple_chain, tesla_robot,
semiconductor, ev_battery).

network=global — requires internet access to GDELT + Google RSS.

Usage:
    python -m scripts.collect_global_industry_news [--date 2026-05-25] [--retry]

Crontab (daily at 06:00 UTC, before A-share open):
    0 6 * * * cd /path/to/stockPrediction && python -m scripts.collect_global_industry_news
"""
import argparse
import hashlib
import json
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

STORAGE_DIR = PROJECT_ROOT / "data" / "storage" / "global_industry_news"
QUERIES_PATH = PROJECT_ROOT / "data" / "config" / "global_industry_queries.yaml"
DATA_HEALTH_DIR = PROJECT_ROOT / "data" / "storage" / "data_health"
JOB_STATUS_PATH = PROJECT_ROOT / "data" / "storage" / "job_status.json"

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE_RSS_URL = "https://news.google.com/rss/search"

REQUEST_TIMEOUT = 15  # seconds per request
INTER_REQUEST_DELAY = 0.5  # be polite to APIs


def _hash_title(title: str) -> str:
    """Create a stable dedup hash from title text."""
    normalized = title.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _load_queries() -> dict:
    """Load query config from YAML."""
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fetch_gdelt(query: str, topic: str, max_records: int = 50) -> list[dict]:
    """Fetch articles from GDELT DOC API for a query string.

    Returns list of news item dicts.
    """
    items = []
    try:
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": max_records,
            "format": "json",
        }
        resp = requests.get(
            GDELT_DOC_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "StockPrediction/1.0"},
        )
        if resp.status_code != 200:
            logger.warning("GDELT returned %d for query=%s", resp.status_code, query)
            return items

        data = resp.json()
        for article in data.get("articles", []):
            title = (article.get("title") or "").strip()
            if not title:
                continue
            url = article.get("url", "")
            domain = article.get("domain", "")
            seen_date = article.get("seendate", "")

            # Source quality heuristic based on domain
            quality = 0.5
            premium_domains = [
                "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
                "cnbc.com", "bbc.com", "nytimes.com", "apnews.com",
            ]
            if any(d in domain for d in premium_domains):
                quality = 0.9
            elif any(d in domain for d in ["yahoo.com", "businessinsider.com", "techcrunch.com"]):
                quality = 0.7

            items.append({
                "source_type": "gdelt",
                "domain": domain,
                "url": url,
                "title": title,
                "summary": "",
                "topic": topic,
                "query": query,
                "language": article.get("language", "English"),
                "source_quality": quality,
                "published_at": seen_date,
            })

    except requests.Timeout:
        logger.warning("GDELT timeout for query=%s", query)
    except Exception as e:
        logger.warning("GDELT fetch error for query=%s: %s", query, e)

    return items


def _fetch_google_rss(query: str, topic: str, max_items: int = 20) -> list[dict]:
    """Fetch articles from Google News RSS for a query string.

    Returns list of news item dicts.
    """
    items = []
    try:
        url = f"{GOOGLE_RSS_URL}?q={quote_plus(query)}&hl=en"
        resp = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "StockPrediction/1.0"},
        )
        if resp.status_code != 200:
            logger.warning("Google RSS returned %d for query=%s", resp.status_code, query)
            return items

        root = ET.fromstring(resp.content)
        count = 0
        for item_elem in root.iter("item"):
            if count >= max_items:
                break

            title_elem = item_elem.find("title")
            title = (title_elem.text or "").strip() if title_elem is not None else ""
            if not title:
                continue

            link_elem = item_elem.find("link")
            link = (link_elem.text or "").strip() if link_elem is not None else ""

            pub_elem = item_elem.find("pubDate")
            pub_date = (pub_elem.text or "").strip() if pub_elem is not None else ""

            source_elem = item_elem.find("source")
            domain = ""
            if source_elem is not None:
                domain = source_elem.get("url", "")
                # Extract domain from URL
                if domain:
                    from urllib.parse import urlparse
                    try:
                        domain = urlparse(domain).netloc
                    except Exception:
                        pass

            desc_elem = item_elem.find("description")
            description = (desc_elem.text or "").strip() if desc_elem is not None else ""

            quality = 0.6  # Google RSS aggregates, moderate default
            premium_domains = [
                "reuters.com", "bloomberg.com", "ft.com", "wsj.com",
                "cnbc.com", "bbc.com", "nytimes.com", "apnews.com",
            ]
            if any(d in domain for d in premium_domains):
                quality = 0.9
            elif any(d in domain for d in ["yahoo.com", "businessinsider.com"]):
                quality = 0.7

            items.append({
                "source_type": "google_rss",
                "domain": domain,
                "url": link,
                "title": title,
                "summary": description[:500] if description else "",
                "topic": topic,
                "query": query,
                "language": "en",
                "source_quality": quality,
                "published_at": pub_date,
            })
            count += 1

    except requests.Timeout:
        logger.warning("Google RSS timeout for query=%s", query)
    except Exception as e:
        logger.warning("Google RSS fetch error for query=%s: %s", query, e)

    return items


def collect_global_industry_news(
    target_date: str | None = None,
    retry: bool = False,
) -> Path:
    """Main collection routine.

    Args:
        target_date: YYYY-MM-DD (default: today)
        retry: if True, re-run even if output file exists

    Returns:
        Path to the output JSONL file.
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = STORAGE_DIR / f"{target_date}.jsonl"

    # Skip if already collected (unless --retry)
    if output_path.exists() and not retry:
        n_existing = sum(1 for _ in open(output_path))
        logger.info(
            "Output already exists (%d items): %s — use --retry to re-collect",
            n_existing, output_path,
        )
        return output_path

    queries_cfg = _load_queries()
    logger.info(
        "Collecting global industry news for %s — %d topics",
        target_date, len(queries_cfg),
    )

    all_items: list[dict] = []
    seen_hashes: set[str] = set()
    stats = {}  # topic -> {"gdelt": n, "rss": n, "deduped": n}

    for topic, cfg in queries_cfg.items():
        topic_items = []
        gdelt_count = 0
        rss_count = 0
        gdelt_max = cfg.get("gdelt_max", 50)
        rss_max = cfg.get("rss_max", 20)

        for query_str in cfg.get("queries", []):
            # GDELT
            gdelt_items = _fetch_gdelt(query_str, topic, max_records=gdelt_max)
            gdelt_count += len(gdelt_items)
            topic_items.extend(gdelt_items)
            time.sleep(INTER_REQUEST_DELAY)

            # Google RSS
            rss_items = _fetch_google_rss(query_str, topic, max_items=rss_max)
            rss_count += len(rss_items)
            topic_items.extend(rss_items)
            time.sleep(INTER_REQUEST_DELAY)

        # Deduplicate within topic by title hash
        dedup_items = []
        for item in topic_items:
            h = _hash_title(item["title"])
            if h not in seen_hashes:
                seen_hashes.add(h)
                item["dedup_hash"] = h
                item["id"] = h
                item["date"] = target_date
                dedup_items.append(item)

        all_items.extend(dedup_items)
        stats[topic] = {
            "gdelt_raw": gdelt_count,
            "rss_raw": rss_count,
            "after_dedup": len(dedup_items),
        }
        logger.info(
            "  %s: GDELT=%d, RSS=%d, deduped=%d",
            topic, gdelt_count, rss_count, len(dedup_items),
        )

        # Streaming write: save after EACH topic so timeout doesn't lose data
        with open(output_path, "w", encoding="utf-8") as f:
            for item in all_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("  (saved %d items so far)", len(all_items))

    logger.info(
        "Final: %d items to %s",
        len(all_items), output_path,
    )

    # Write data_health status
    _write_data_health(target_date, stats, len(all_items))

    return output_path


def _write_data_health(target_date: str, stats: dict, total: int) -> None:
    """Write data health via unified scheduler.data_health interface."""
    try:
        from scheduler.data_health import write_health, HealthStatus
        write_health("global_industry_news", HealthStatus(
            success=total > 0,
            n_items=total,
            latest_date=target_date,
            network_profile="global",
        ), date=target_date)
    except Exception:
        # Fallback: write custom health file
        DATA_HEALTH_DIR.mkdir(parents=True, exist_ok=True)
        health = {
            "job": "global_industry_news",
            "date": target_date,
            "status": "success" if total > 0 else "empty",
            "total_items": total,
        }
        health_path = DATA_HEALTH_DIR / f"global_industry_news_{target_date}.json"
        with open(health_path, "w", encoding="utf-8") as f:
            json.dump(health, f, ensure_ascii=False, indent=2)
    logger.info("Data health written for global_industry_news")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Collect global industry news (GDELT + Google RSS)"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Target date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--retry", action="store_true",
        help="Re-collect even if output already exists",
    )
    args = parser.parse_args()

    try:
        output = collect_global_industry_news(
            target_date=args.date,
            retry=args.retry,
        )
        logger.info("Done: %s", output)
    except Exception as e:
        logger.error("Collection failed: %s", e)
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
