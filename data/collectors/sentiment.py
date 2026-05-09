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

    def fetch_akshare_comments(self, stock_code: str, limit: int = 20) -> list:
        """Fetch stock comments via AKShare structured API (more reliable than HTML scraping).

        Args:
            stock_code: Pure numeric code, e.g. "600519"
            limit: Max posts to return

        Returns:
            List of dicts with keys: text, timestamp, source
        """
        try:
            import akshare as ak
            # AKShare 个股评论情绪接口
            df = ak.stock_comment_detail_zlkp_jgcyd_em(symbol=stock_code)
            if df is None or df.empty:
                return []

            posts = []
            for _, row in df.head(limit).iterrows():
                text = str(row.get("用户评论", row.get("评论内容", "")))
                if not text or len(text) < 4:
                    continue
                posts.append({
                    "text": text[:500],
                    "timestamp": datetime.now().isoformat(),
                    "source": "akshare_comment",
                })
            return posts
        except Exception as e:
            # Fallback: try general comment sentiment
            try:
                import akshare as ak
                df = ak.stock_comment_em()
                if df is None or df.empty:
                    return []
                row = df[df["代码"] == stock_code]
                if row.empty:
                    return []
                # Extract sentiment summary as a single "post"
                r = row.iloc[0]
                summary = f"关注度:{r.get('关注指数', 'N/A')} 参与度:{r.get('参与指数', 'N/A')}"
                return [{
                    "text": summary,
                    "timestamp": datetime.now().isoformat(),
                    "source": "akshare_sentiment",
                }]
            except Exception:
                return []

    def fetch_all(self, qlib_code: str, limit_per_source: int = 20) -> list:
        """Fetch sentiment from all sources for a stock.

        Priority: AKShare API > Eastmoney HTML > Xueqiu (often blocked)

        Args:
            qlib_code: Qlib format code, e.g. "SH600519"
            limit_per_source: Max posts per source

        Returns:
            Combined list of posts from all sources
        """
        stock_code = qlib_code[2:]
        all_posts = []

        # Source 1: AKShare structured API (most reliable)
        ak_posts = self.fetch_akshare_comments(stock_code, limit_per_source)
        all_posts.extend(ak_posts)

        # Source 2: Eastmoney HTML (fragile but sometimes has more data)
        if len(all_posts) < limit_per_source:
            em_posts = self.fetch_eastmoney(stock_code, limit_per_source)
            all_posts.extend(em_posts)

        # Source 3: Xueqiu (often blocked, try as last resort)
        if len(all_posts) < 5:
            time.sleep(0.5)
            xq_posts = self.fetch_xueqiu(qlib_code, limit_per_source)
            all_posts.extend(xq_posts)

        return all_posts
