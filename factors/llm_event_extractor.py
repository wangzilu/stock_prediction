"""LLM-based structured event extraction from news.

.. deprecated::
    This module (V1) is deprecated. Use ``factors.llm_event_extractor_v2.LLMEventExtractorV2``
    instead. V1 asks the LLM to predict stock-price impacts (impact_1d/impact_5d), which is
    unreliable. V2 extracts structured facts only; impact estimation is done via historical
    calibration. This file is kept for backward compatibility and will be removed in a future
    release.

Uses MiniMax API to analyze stock news and extract structured event data
including event type, impact estimates, confidence, relevance, and novelty.

Usage (DEPRECATED — prefer V2):
    from factors.llm_event_extractor_v2 import LLMEventExtractorV2
    extractor = LLMEventExtractorV2()
    events = extractor.extract_from_news_file("data/storage/daily_news/2024-01-15.jsonl")
"""
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests

from config.settings import DATA_DIR, MINIMAX_API_KEY

logger = logging.getLogger(__name__)

EVENTS_DIR = DATA_DIR / "llm_events"
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

NEWS_DIR = DATA_DIR / "daily_news"


def _source_quality_score(source: str) -> float:
    """Source quality weight per CX: exchange > institutional > general > social."""
    s = source.lower() if source else ""
    if any(k in s for k in ["交易所", "上交所", "深交所", "公告", "巨潮"]):
        return 1.0
    if any(k in s for k in ["财联社", "证券时报", "中国证券报", "上海证券报", "券商"]):
        return 0.8
    if any(k in s for k in ["东方财富", "同花顺", "新浪财经", "网易财经", "每日经济"]):
        return 0.6
    if any(k in s for k in ["股吧", "雪球", "xueqiu", "论坛", "吧"]):
        return 0.3
    return 0.5  # default: general news


EXTRACTION_SYSTEM_PROMPT = """你是一位专业的金融事件分析师。你的任务是从股票新闻中提取结构化的事件信息。

请严格按照JSON格式输出，不要输出任何其他文字。"""

EXTRACTION_USER_PROMPT = """分析以下关于股票 {stock_code}（{stock_name}）的新闻，提取事件信息。

新闻标题：{title}
新闻摘要：{snippet}

请输出严格JSON格式（不要任何其他文字）：
{{
  "event_type": "事件类型，从以下选择：earnings_positive|earnings_negative|insider_sell|insider_buy|policy_positive|policy_negative|order_win|product_launch|lawsuit|restructure|dividend|buyback|analyst_upgrade|analyst_downgrade|management_change|industry_trend_positive|industry_trend_negative|other",
  "impact_1d": 预估该事件对股价的1日影响，-1到1之间的浮点数,
  "impact_5d": 预估该事件对股价的5日影响，-1到1之间的浮点数,
  "confidence": 你对这个判断的信心，0到1之间的浮点数,
  "relevance": 这条新闻与该股票的相关性，0到1之间的浮点数（0=完全无关，1=直接相关）,
  "novelty": 这条新闻的新颖性/信息量，0到1之间的浮点数（0=旧闻/水文，1=重大新信息）,
  "summary": "一句话总结事件核心内容"
}}

注意：
- impact值应该反映事件对股价的实际影响幅度（±0.01-0.05为小影响，±0.05-0.15为中等，±0.15+为重大）
- 如果新闻是水文、广告或与股票无关，relevance应该很低
- 如果新闻内容是已知信息的重复报道，novelty应该较低"""


class LLMEventExtractor:
    """Extract structured events from stock news using MiniMax LLM."""

    def __init__(
        self,
        api_key: str = None,
        model: str = "minimax-m2.5-highspeed",
        max_calls_per_minute: int = 60,
    ):
        self.api_key = api_key or MINIMAX_API_KEY
        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY is required for LLM event extraction")
        self.model = model
        self.api_url = "https://api.minimax.io/v1/chat/completions"
        self.max_calls_per_minute = max_calls_per_minute
        self._call_timestamps: list[float] = []
        import threading
        self._rate_lock = threading.Lock()

    def _rate_limit(self):
        """Enforce rate limit by waiting if necessary. Thread-safe."""
        with self._rate_lock:
            now = time.time()
            self._call_timestamps = [t for t in self._call_timestamps if now - t < 60]
            if len(self._call_timestamps) >= self.max_calls_per_minute:
                wait_time = 60 - (now - self._call_timestamps[0]) + 0.1
                if wait_time > 0:
                    logger.info(f"Rate limit: waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
            self._call_timestamps.append(time.time())

    def _call_llm(self, system: str, user: str) -> str:
        """Call MiniMax API with retry. Returns response text."""
        self._rate_limit()

        for attempt in range(2):
            try:
                resp = requests.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "max_tokens": 512,
                    },
                    timeout=(5, 15),  # (connect, read) timeout
                )

                if resp.status_code != 200:
                    logger.warning(f"MiniMax API attempt {attempt+1}: status {resp.status_code}")
                    if attempt == 0:
                        time.sleep(3)
                        continue
                    return ""

                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                # Strip think tags (MiniMax sometimes wraps in <think>)
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                return text

            except Exception as e:
                logger.warning(f"MiniMax API attempt {attempt+1} failed: {e}")
                if attempt == 0:
                    time.sleep(3)

        return ""

    def _parse_extraction(self, text: str) -> dict | None:
        """Parse LLM response into a structured event dict."""
        if not text:
            return None

        try:
            clean = text.strip()
            # Strip markdown code fences
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(
                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                )

            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start < 0 or end <= start:
                return None

            parsed = json.loads(clean[start:end])

            # Validate and clamp fields
            result = {
                "event_type": str(parsed.get("event_type", "other")),
                "impact_1d": max(-1.0, min(1.0, float(parsed.get("impact_1d", 0)))),
                "impact_5d": max(-1.0, min(1.0, float(parsed.get("impact_5d", 0)))),
                "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.5)))),
                "relevance": max(0.0, min(1.0, float(parsed.get("relevance", 0.5)))),
                "novelty": max(0.0, min(1.0, float(parsed.get("novelty", 0.5)))),
                "summary": str(parsed.get("summary", "")),
            }
            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM extraction: {e}")
            return None

    def extract_single(self, stock_code: str, stock_name: str, title: str, snippet: str) -> dict | None:
        """Extract structured event from a single news item.

        Args:
            stock_code: 6-digit stock code
            stock_name: stock name
            title: news title
            snippet: news content snippet

        Returns:
            Structured event dict or None if extraction failed
        """
        prompt = EXTRACTION_USER_PROMPT.format(
            stock_code=stock_code,
            stock_name=stock_name,
            title=title,
            snippet=snippet[:400],  # Limit snippet length
        )

        text = self._call_llm(EXTRACTION_SYSTEM_PROMPT, prompt)
        return self._parse_extraction(text)

    def extract_from_news_file(
        self,
        news_path: str | Path,
        max_news_per_stock: int = 3,
        target_date: str = None,
    ) -> Path:
        """Process a daily news JSONL file and extract events.

        For each stock, processes at most max_news_per_stock most recent items.

        Args:
            news_path: path to daily_news JSONL file
            max_news_per_stock: max news items to process per stock
            target_date: YYYY-MM-DD for output filename (inferred from news_path if None)

        Returns:
            Path to the saved LLM events JSONL file
        """
        news_path = Path(news_path)
        if not news_path.exists():
            raise FileNotFoundError(f"News file not found: {news_path}")

        if target_date is None:
            # Infer from filename: daily_news/2024-01-15.jsonl
            target_date = news_path.stem

        output_path = EVENTS_DIR / f"{target_date}.jsonl"

        # Skip if already processed with sufficient data
        if output_path.exists():
            n_existing = sum(1 for _ in open(output_path))
            if n_existing >= 500:  # minimum: 500 events for full-A coverage
                logger.info(f"Events already extracted for {target_date} ({n_existing} items), skipping")
                return output_path
            else:
                logger.warning(f"Previous extraction only got {n_existing} events, re-extracting")
                os.remove(str(output_path))

        # Load news grouped by stock
        stock_news: dict[str, list[dict]] = {}
        with open(news_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                code = item.get("stock_code", "")
                if code not in stock_news:
                    stock_news[code] = []
                stock_news[code].append(item)

        logger.info(f"Processing news for {len(stock_news)} stocks from {news_path.name}")

        # Build task list: (code, name, title, snippet, news_item) for each news
        tasks = []
        seen_titles = set()  # dedup by title
        for code, news_list in stock_news.items():
            selected = news_list[:max_news_per_stock]
            for news_item in selected:
                title = news_item.get("title", "").strip()
                if not title:
                    continue
                # Title dedup: per-stock, not global (different stocks can have similar announcements)
                title_key = f"{code}_{title[:30]}"
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)
                tasks.append((code, news_item))

        logger.info(f"  {len(tasks)} unique news items to process (after dedup from {sum(len(v[:max_news_per_stock]) for v in stock_news.values())})")

        # Concurrent LLM extraction
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        total_extracted = 0
        total_failed = 0
        results_lock = threading.Lock()
        all_records = []

        def _process_one(task):
            """Process a single news item. Thread-safe."""
            code, news_item = task
            title = news_item.get("title", "")
            snippet = news_item.get("content_snippet", "")
            name = news_item.get("stock_name", "")
            try:
                event = self.extract_single(code, name, title, snippet)
                if event is not None:
                    source = news_item.get("source", "unknown")
                    source_quality = _source_quality_score(source)
                    record = {
                        "stock_code": code,
                        "stock_name": name,
                        "qlib_code": news_item.get("qlib_code", ""),
                        "publish_time": news_item.get("publish_time", ""),
                        "news_title": title,
                        "raw_text": snippet[:500],
                        "source": source,
                        "source_quality": source_quality,
                        "model_version": "minimax-m2.5-highspeed",
                        "prompt_version": "v1",
                        "extract_date": target_date,
                        **event,
                    }
                    return ("ok", record)
                return ("fail", None)
            except Exception as e:
                return ("fail", None)

        # Stream results to file as they complete (survives timeout/kill)
        n_workers = 16
        with open(output_path, "w", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_process_one, t): i for i, t in enumerate(tasks)}
                done_count = 0
                for future in as_completed(futures):
                    done_count += 1
                    status, record = future.result()
                    if status == "ok" and record:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()  # ensure written to disk immediately
                        total_extracted += 1
                    else:
                        total_failed += 1
                    if done_count % 50 == 0:
                        logger.info(
                            f"  Progress: {done_count}/{len(tasks)} items, "
                            f"{total_extracted} extracted, {total_failed} failed"
                        )

        logger.info(
            f"Extraction complete: {total_extracted} events extracted, "
            f"{total_failed} failed -> {output_path}"
        )
        return output_path


def main():
    """CLI entry point for event extraction."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Extract structured events from daily news via LLM")
    parser.add_argument("--date", type=str, default=None, help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--max-per-stock", type=int, default=3, help="Max news per stock (default: 3)")
    args = parser.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    news_path = NEWS_DIR / f"{target_date}.jsonl"

    if not news_path.exists():
        logger.error(f"No news file for {target_date}. Run collect_daily_news first.")
        return

    extractor = LLMEventExtractor()
    extractor.extract_from_news_file(news_path, max_news_per_stock=args.max_per_stock)


if __name__ == "__main__":
    main()
