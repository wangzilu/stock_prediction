import xml.etree.ElementTree as ET
import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# RSS feed sources — global mainstream media + policy
RSS_FEEDS = {
    # Central banks
    "fed": "https://www.federalreserve.gov/feeds/press_all.xml",
    # Google News aggregations (reliable, fast, multi-source)
    "geopolitics": "https://news.google.com/rss/search?q=war+OR+conflict+OR+military+OR+sanctions+OR+missile&hl=en",
    "china_us": "https://news.google.com/rss/search?q=china+US+OR+trump+china+OR+trade+war+OR+tariff&hl=en",
    "markets": "https://news.google.com/rss/search?q=federal+reserve+OR+central+bank+OR+interest+rate+OR+inflation&hl=en",
    "china_economy": "https://news.google.com/rss/search?q=china+economy+OR+PBOC+OR+yuan+OR+A-shares&hl=en",
    "middle_east": "https://news.google.com/rss/search?q=iran+OR+israel+OR+hormuz+OR+middle+east+conflict&hl=en",
    "russia_ukraine": "https://news.google.com/rss/search?q=russia+ukraine+war+OR+zelensky+OR+putin&hl=en",
    "gold_oil": "https://news.google.com/rss/search?q=gold+price+OR+oil+price+OR+crude+OR+safe+haven&hl=en",
    "crypto_macro": "https://news.google.com/rss/search?q=bitcoin+OR+crypto+regulation+OR+SEC+crypto&hl=en",
}


class MacroCollector:
    """Collects macroeconomic and central bank policy news via RSS."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "StockPrediction/1.0"
        })

    def fetch_rss(self, url: str, max_items: int = 20) -> list:
        """Fetch and parse an RSS feed.

        Args:
            url: RSS feed URL
            max_items: Maximum items to return

        Returns:
            List of dicts with keys: title, link, published, description
        """
        try:
            resp = self.session.get(url, timeout=10)
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

    def fetch_pboc_news(self, max_items: int = 10) -> list:
        """Fetch PBOC (People's Bank of China) policy news.

        Returns:
            List of dicts with title, link, published, description
        """
        return self.fetch_rss(RSS_FEEDS["pboc"], max_items)

    def fetch_market_news(self, max_items: int = 20) -> list:
        """Fetch global market/central bank news via Google News RSS.

        Returns:
            List of dicts with title, link, published, description
        """
        return self.fetch_rss(RSS_FEEDS["reuters_markets"], max_items)

    def fetch_china_macro_news(self, max_items: int = 20) -> list:
        """Fetch China macro economy news.

        Returns:
            List of dicts with title, link, published, description
        """
        return self.fetch_rss(RSS_FEEDS["reuters_china"], max_items)

    def fetch_all(self, max_per_source: int = 10) -> list:
        """Fetch news from all sources.

        Returns:
            Combined list of news items, each with added 'source' key.
        """
        all_news = []

        for source_name, url in RSS_FEEDS.items():
            items = self.fetch_rss(url, max_per_source)
            for item in items:
                item["source"] = source_name
            all_news.extend(items)

        return all_news
