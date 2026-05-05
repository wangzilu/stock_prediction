import requests
import time
import re
from datetime import datetime


class SentimentCollector:
    """Collects sentiment posts from Xueqiu and Eastmoney Guba."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        self.timeout = 10.0

    def fetch_xueqiu(self, qlib_code: str, limit: int = 20) -> list:
        """Fetch recent posts from Xueqiu for a stock.

        Args:
            qlib_code: Qlib format code, e.g. "SH600519"
            limit: Max posts to return

        Returns:
            List of dicts with keys: text, timestamp, source
        """
        try:
            symbol = qlib_code
            url = "https://xueqiu.com/query/v1/symbol/search/status.json"
            params = {
                "u": "",
                "q": symbol,
                "count": limit,
                "comment": "0",
                "symbol": symbol,
                "hl": "0",
                "source": "all",
                "sort": "time",
            }

            self.session.get("https://xueqiu.com/", timeout=self.timeout)
            resp = self.session.get(url, params=params, timeout=self.timeout)

            if resp.status_code != 200:
                return []

            data = resp.json()
            posts = []
            for item in data.get("list", [])[:limit]:
                text = item.get("text", "") or item.get("description", "")
                text = text.replace("<", " <").split("<")[0] if "<" in text else text
                created_at = item.get("created_at", 0)
                posts.append({
                    "text": text[:500],
                    "timestamp": datetime.fromtimestamp(created_at / 1000).isoformat()
                    if created_at
                    else "",
                    "source": "xueqiu",
                })
            return posts

        except Exception:
            return []

    def fetch_eastmoney(self, stock_code: str, limit: int = 20) -> list:
        """Fetch recent posts from Eastmoney Guba.

        Args:
            stock_code: Pure numeric code, e.g. "600519"
            limit: Max posts to return

        Returns:
            List of dicts with keys: text, timestamp, source
        """
        try:
            url = f"https://guba.eastmoney.com/list,{stock_code},1,f.html"
            resp = self.session.get(url, timeout=self.timeout)

            if resp.status_code != 200:
                return []

            titles = re.findall(
                r'title="([^"]+)"[^>]*>([^<]*)</a>', resp.text
            )

            posts = []
            seen = set()
            for title_attr, _ in titles:
                text = title_attr.strip()
                if not text or text in seen or len(text) < 4:
                    continue
                seen.add(text)
                posts.append({
                    "text": text[:500],
                    "timestamp": datetime.now().isoformat(),
                    "source": "eastmoney",
                })
                if len(posts) >= limit:
                    break

            return posts

        except Exception:
            return []

    def fetch_all(self, qlib_code: str, limit_per_source: int = 20) -> list:
        """Fetch sentiment from all sources for a stock.

        Args:
            qlib_code: Qlib format code, e.g. "SH600519"
            limit_per_source: Max posts per source

        Returns:
            Combined list of posts from all sources
        """
        stock_code = qlib_code[2:]

        xueqiu_posts = self.fetch_xueqiu(qlib_code, limit_per_source)
        time.sleep(0.5)
        eastmoney_posts = self.fetch_eastmoney(stock_code, limit_per_source)

        return xueqiu_posts + eastmoney_posts
