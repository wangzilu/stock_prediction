import xml.etree.ElementTree as ET
import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# RSS feed sources — global mainstream media + policy + geopolitics
RSS_FEEDS = {
    # === Global mainstream media (direct RSS) ===
    "bbc_world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "bbc_business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "reuters_world": "https://news.google.com/rss/search?q=site:reuters.com+world&hl=en",
    "aljazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "france24": "https://www.france24.com/en/rss",
    "dw_news": "https://rss.dw.com/rdf/rss-en-world",
    "nyt_world": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "wsj_world": "https://news.google.com/rss/search?q=site:wsj.com+world+OR+economy&hl=en",

    # Central banks
    "fed": "https://www.federalreserve.gov/feeds/press_all.xml",

    # Geopolitics & conflicts
    "russia_ukraine": "https://news.google.com/rss/search?q=russia+ukraine+war+OR+zelensky+OR+putin+offensive&hl=en",
    "middle_east": "https://news.google.com/rss/search?q=iran+OR+israel+OR+hormuz+strait+OR+gaza+OR+hezbollah&hl=en",
    "taiwan_strait": "https://news.google.com/rss/search?q=taiwan+strait+OR+taiwan+military+OR+china+taiwan+tension&hl=en",
    "global_conflict": "https://news.google.com/rss/search?q=war+OR+military+strike+OR+missile+OR+nuclear+threat+OR+invasion&hl=en",

    # Trade & tariffs
    "trade_war": "https://news.google.com/rss/search?q=tariff+OR+trade+war+OR+EU+tariff+OR+trade+sanction+OR+export+ban&hl=en",
    "china_us": "https://news.google.com/rss/search?q=china+US+relations+OR+trump+china+OR+trump+xi+OR+decoupling&hl=en",

    # Macro & policy
    "central_banks": "https://news.google.com/rss/search?q=federal+reserve+OR+ECB+OR+BOJ+OR+PBOC+OR+interest+rate+decision&hl=en",
    "inflation": "https://news.google.com/rss/search?q=inflation+OR+CPI+OR+recession+OR+economic+slowdown&hl=en",
    "china_economy": "https://news.google.com/rss/search?q=china+economy+OR+china+GDP+OR+yuan+OR+A-shares+OR+CSI300&hl=en",

    # Safe haven & commodities
    "gold_oil": "https://news.google.com/rss/search?q=gold+price+OR+oil+price+OR+crude+brent+OR+safe+haven+OR+commodity&hl=en",
    "crypto_macro": "https://news.google.com/rss/search?q=bitcoin+OR+crypto+regulation+OR+stablecoin+OR+digital+currency&hl=en",

    # EU & global trade
    "eu_policy": "https://news.google.com/rss/search?q=EU+tariff+OR+european+union+trade+OR+EU+sanction+OR+euro+economy&hl=en",

    # Japan & Asia
    "japan_asia": "https://news.google.com/rss/search?q=japan+china+relations+OR+BOJ+OR+yen+OR+asia+pacific+trade&hl=en",

    # Global diplomacy
    "diplomacy": "https://news.google.com/rss/search?q=G7+OR+G20+OR+summit+OR+bilateral+OR+diplomatic+OR+UN+security+council&hl=en",

    # Global stock markets (impact on A-shares)
    "us_stocks": "https://news.google.com/rss/search?q=S%26P500+OR+nasdaq+OR+dow+jones+OR+wall+street+OR+US+stock+market&hl=en",
    "europe_stocks": "https://news.google.com/rss/search?q=FTSE+OR+DAX+OR+european+stocks+OR+stoxx&hl=en",
    "asia_stocks": "https://news.google.com/rss/search?q=nikkei+OR+hang+seng+OR+kospi+OR+asia+stocks+OR+asia+markets&hl=en",

    # Top economists & analysts commentary
    "economist_views": "https://news.google.com/rss/search?q=economist+forecast+OR+analyst+predict+OR+market+outlook+OR+goldman+sachs+OR+morgan+stanley&hl=en",
    "imf_worldbank": "https://news.google.com/rss/search?q=IMF+OR+world+bank+OR+global+growth+forecast+OR+OECD+outlook&hl=en",

    # Financial media with analyst commentary
    "bloomberg_views": "https://news.google.com/rss/search?q=site:bloomberg.com+analysis+OR+outlook+OR+forecast&hl=en",
    "ft_analysis": "https://news.google.com/rss/search?q=site:ft.com+analysis+OR+outlook+OR+economist&hl=en",
    "reuters_analysis": "https://news.google.com/rss/search?q=site:reuters.com+analysis+OR+forecast+OR+outlook&hl=en",
}


class MacroCollector:
    """Collects macroeconomic and central bank policy news via RSS."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "StockPrediction/1.0"
        })

    def fetch_rss(self, url: str, max_items: int = 20, timeout: int = 5) -> list:
        """Fetch and parse an RSS feed.

        Args:
            url: RSS feed URL
            max_items: Maximum items to return
            timeout: request timeout in seconds (default 5, reduced from 10
                to avoid blocking domestic jobs when proxy is unavailable)

        Returns:
            List of dicts with keys: title, link, published, description
            Returns [] on any failure (timeout, DNS, HTTP error).
        """
        try:
            resp = self.session.get(url, timeout=timeout)
            if resp.status_code != 200:
                logger.warning(f"RSS feed returned status {resp.status_code}: {url}")
                return []

            root = ET.fromstring(resp.content)

            items = []
            # Try standard RSS format
            for item in root.iter("item"):
                entry = {
                    "title": self._get_text(item, "title"),
                    "link": self._get_text(item, "link"),
                    "published": self._get_text(item, "pubDate"),
                    "description": self._get_text(item, "description"),
                }
                if entry["title"]:
                    items.append(entry)
                if len(items) >= max_items:
                    break

            # If no items found, try Atom format
            if not items:
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry_elem in root.findall(".//atom:entry", ns):
                    entry = {
                        "title": self._get_text_ns(entry_elem, "atom:title", ns),
                        "link": "",
                        "published": self._get_text_ns(entry_elem, "atom:updated", ns),
                        "description": self._get_text_ns(entry_elem, "atom:summary", ns),
                    }
                    link_elem = entry_elem.find("atom:link", ns)
                    if link_elem is not None:
                        entry["link"] = link_elem.get("href", "")
                    if entry["title"]:
                        items.append(entry)
                    if len(items) >= max_items:
                        break

            return items

        except Exception as e:
            logger.warning(f"RSS fetch failed for {url}: {e}")
            return []

    def _get_text(self, element, tag):
        """Get text content of a child element."""
        child = element.find(tag)
        return child.text.strip() if child is not None and child.text else ""

    def _get_text_ns(self, element, tag, ns):
        """Get text content with namespace."""
        child = element.find(tag, ns)
        return child.text.strip() if child is not None and child.text else ""

    def fetch_fed_news(self, max_items: int = 10) -> list:
        """Fetch Federal Reserve press releases.

        Returns:
            List of dicts with title, link, published, description
        """
        return self.fetch_rss(RSS_FEEDS["fed"], max_items)

    def fetch_market_news(self, max_items: int = 20) -> list:
        """Fetch global market/central bank news."""
        return self.fetch_rss(RSS_FEEDS["central_banks"], max_items)

    def fetch_china_macro_news(self, max_items: int = 20) -> list:
        """Fetch China macro economy news."""
        return self.fetch_rss(RSS_FEEDS["china_economy"], max_items)

    def fetch_all(self, max_per_source: int = 10, total_timeout: int = 60,
                  skip_global: bool = False) -> list:
        """Fetch news from all sources with total time budget.

        Args:
            max_per_source: max items per RSS feed
            total_timeout: max total seconds for all feeds (default 60s).
                If exceeded, returns whatever was collected so far.
            skip_global: if True, skip feeds that require VPN (Google/BBC/Reuters/etc.)
                Set True when running in domestic network profile.

        Returns:
            Combined list of news items, each with added 'source' key.
        """
        import os
        import time
        start = time.time()
        all_news = []
        n_failed = 0
        n_skipped = 0

        # Auto-detect: if no proxy env set, skip global feeds to avoid timeouts
        if skip_global is False:
            proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
            if not proxy:
                skip_global = True
                logger.info("MacroCollector: no proxy detected, skipping global RSS feeds")

        # ALL feeds in RSS_FEEDS are overseas sources requiring VPN.
        # In domestic mode, skip all of them and return empty.
        # Global news should be fetched by a separate global-network job.
        if skip_global:
            logger.info("MacroCollector: domestic mode, all RSS feeds are global — returning empty")
            return []

        GLOBAL_DOMAINS: set[str] = set()  # unused now but kept for future per-domain logic

        seen_titles = set()
        for source_name, url in RSS_FEEDS.items():
            # Check total time budget
            elapsed = time.time() - start
            if elapsed > total_timeout:
                remaining = len(RSS_FEEDS) - (len(all_news) + n_failed + n_skipped)
                logger.warning(
                    f"MacroCollector total timeout ({total_timeout}s) exceeded after "
                    f"{elapsed:.0f}s, skipping {remaining} remaining feeds"
                )
                break

            # Skip global feeds when no proxy
            if skip_global:
                from urllib.parse import urlparse
                domain = urlparse(url).hostname or ""
                if domain in GLOBAL_DOMAINS:
                    n_skipped += 1
                    continue

            items = self.fetch_rss(url, max_per_source, timeout=5)
            if not items:
                n_failed += 1
            for item in items:
                title = item.get("title", "").strip()
                if title and title not in seen_titles:
                    item["source"] = source_name
                    all_news.append(item)
                    seen_titles.add(title)

        logger.info(
            f"MacroCollector: {len(all_news)} headlines from "
            f"{len(RSS_FEEDS) - n_failed - n_skipped}/{len(RSS_FEEDS)} feeds "
            f"({n_skipped} skipped, {n_failed} failed) "
            f"in {time.time() - start:.0f}s"
        )
        return all_news
