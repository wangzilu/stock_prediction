"""LLM Event Extractor V2 — extracts FACTS, not impact predictions.

CX design principle: LLM only extracts structured facts from news/announcements.
Impact estimation comes from historical return calibration, NOT from LLM.

V1 → V2 changes:
  - Removed: impact_1d, impact_5d (LLM shouldn't predict returns)
  - Added: magnitude fields, official/new/repeated flags, quality signals
  - Three-layer source classification: official > media > social

Usage:
    from factors.llm_event_extractor_v2 import LLMEventExtractorV2
    ext = LLMEventExtractorV2()
    event = ext.extract_single("600519", "贵州茅台", "title", "content")
"""
import json
import logging
import os
import re
import time
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from config.settings import DATA_DIR, MINIMAX_API_KEY

logger = logging.getLogger(__name__)

EVENTS_DIR = DATA_DIR / "llm_events_v2"
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

# V2 Event types (more granular than v1)
EVENT_TYPES = [
    "earnings_beat", "earnings_miss", "earnings_inline",
    "revenue_growth", "revenue_decline",
    "order_win", "major_contract",
    "product_launch", "tech_breakthrough",
    "market_share_gain", "market_share_loss",
    "share_buyback", "dividend_increase",
    "insider_buy", "insider_sell",
    "share_placement", "share_unlock",
    "analyst_upgrade", "analyst_downgrade",
    "regulatory_approval", "regulatory_penalty",
    "lawsuit_filed", "lawsuit_settled",
    "management_change", "restructuring",
    "strategic_cooperation", "joint_venture",
    "government_subsidy", "tax_benefit",
    "debt_issue", "credit_rating_change",
    "routine_announcement",  # 常规公告（无实质影响）
    "other",
]

# Source quality tiers
SOURCE_TIERS = {
    "交易所公告": {"tier": "official", "quality": 1.0},
    "巨潮资讯": {"tier": "official", "quality": 0.95},
    "上交所": {"tier": "official", "quality": 1.0},
    "深交所": {"tier": "official", "quality": 1.0},
    "证券时报": {"tier": "media", "quality": 0.8},
    "证券时报网": {"tier": "media", "quality": 0.8},
    "上海证券报": {"tier": "media", "quality": 0.8},
    "中国证券报": {"tier": "media", "quality": 0.8},
    "财联社": {"tier": "media", "quality": 0.75},
    "21世纪经济报道": {"tier": "media", "quality": 0.7},
    "界面新闻": {"tier": "media", "quality": 0.65},
    "红星资本局": {"tier": "media", "quality": 0.6},
    "eastmoney": {"tier": "media", "quality": 0.5},
    "雪球": {"tier": "social", "quality": 0.3},
    "股吧": {"tier": "social", "quality": 0.2},
}

SYSTEM_PROMPT_V2 = """你是A股事件结构化助手。从新闻/公告中提取事实信息，不预测股价。

输出严格JSON格式：
{
  "event_type": "从以下选择: earnings_beat, earnings_miss, earnings_inline, revenue_growth, revenue_decline, order_win, major_contract, product_launch, tech_breakthrough, market_share_gain, share_buyback, dividend_increase, insider_buy, insider_sell, share_placement, share_unlock, analyst_upgrade, analyst_downgrade, regulatory_approval, regulatory_penalty, lawsuit_filed, management_change, restructuring, strategic_cooperation, government_subsidy, routine_announcement, other",
  "direction": 1或-1或0,
  "is_official_disclosure": true或false,
  "is_new_information": true或false,
  "is_repeated_news": true或false,
  "is_price_sensitive": true或false,
  "magnitude_description": "金额或规模的文字描述，如'合同金额3.2亿元'，无则为空",
  "magnitude_value_wan": 提到的金额(万元)，无则为0,
  "confidence": 0到1之间的数字,
  "summary": "一句话概括事实（不超过50字）"
}

规则：
- 只提取新闻中明确陈述的事实，不推断
- is_official_disclosure: 来自交易所公告为true，媒体报道为false
- is_new_information: 首次披露为true，已知信息重复报道为false
- is_price_sensitive: 可能引起显著股价反应为true
- magnitude_value_wan: 提到具体金额就转换成万元填入，没提到就填0
- routine_announcement: 常规信息披露（独董声明、日常关联交易等）
- confidence: 信息确定性，公告=0.9+，权威媒体=0.7-0.8，传闻=0.3-0.5
- 无法判断时 confidence 设低，不要编造"""


class LLMEventExtractorV2:
    """V2: Extract structured FACTS from news, not impact predictions."""

    def __init__(self,
                 api_key: str = None,
                 model: str = "minimax-m2.5-highspeed",
                 max_calls_per_minute: int = 60):
        self.api_key = api_key or MINIMAX_API_KEY
        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY required")
        self.model = model
        self.api_url = "https://api.minimax.io/v1/chat/completions"
        self.max_calls_per_minute = max_calls_per_minute
        self._call_timestamps = []
        self._rate_lock = threading.Lock()

    def _rate_limit(self):
        with self._rate_lock:
            now = time.time()
            self._call_timestamps = [t for t in self._call_timestamps if now - t < 60]
            if len(self._call_timestamps) >= self.max_calls_per_minute:
                wait = 60 - (now - self._call_timestamps[0]) + 0.1
                if wait > 0:
                    time.sleep(wait)
            self._call_timestamps.append(time.time())

    def _call_llm(self, user_prompt: str) -> str:
        self._rate_limit()
        for attempt in range(2):
            try:
                resp = requests.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json={"model": self.model,
                          "messages": [{"role": "system", "content": SYSTEM_PROMPT_V2},
                                       {"role": "user", "content": user_prompt}],
                          "max_tokens": 512},
                    timeout=(5, 15),
                )
                if resp.status_code != 200:
                    if attempt == 0:
                        time.sleep(3)
                        continue
                    return ""
                text = resp.json()["choices"][0]["message"]["content"]
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                return text
            except Exception as e:
                if attempt == 0:
                    time.sleep(3)
        return ""

    def _parse_response(self, text: str) -> dict | None:
        if not text:
            return None

        clean = text.strip()
        # Strip markdown code fences if present
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )

        # Robust JSON extraction: first { to last } (handles nested braces)
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(clean[start:end])
        except json.JSONDecodeError:
            # Fallback: try simple non-nested regex (original V2 approach)
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None

        # Validate required fields
        event_type = data.get("event_type", "other")
        if event_type not in EVENT_TYPES:
            event_type = "other"

        return {
            "event_type": event_type,
            "direction": int(data.get("direction", 0)),
            "is_official_disclosure": bool(data.get("is_official_disclosure", False)),
            "is_new_information": bool(data.get("is_new_information", True)),
            "is_repeated_news": bool(data.get("is_repeated_news", False)),
            "is_price_sensitive": bool(data.get("is_price_sensitive", False)),
            "magnitude_description": str(data.get("magnitude_description", "")),
            "magnitude_value_wan": float(data.get("magnitude_value_wan", 0)),
            "confidence": max(0, min(1, float(data.get("confidence", 0.5)))),
            "summary": str(data.get("summary", ""))[:100],
        }

    def extract_single(self, code: str, name: str, title: str, content: str = "") -> dict | None:
        prompt = f"股票: {code} {name}\n标题: {title}"
        if content and content != title:
            prompt += f"\n内容: {content[:300]}"
        text = self._call_llm(prompt)
        return self._parse_response(text)

    def extract_from_news_file(self, news_path: Path, max_news_per_stock: int = 1,
                                target_date: str = None) -> Path:
        """Process news file and extract v2 events."""
        news_path = Path(news_path)
        if target_date is None:
            target_date = news_path.stem

        output_path = EVENTS_DIR / f"{target_date}.jsonl"

        if output_path.exists():
            n = sum(1 for _ in open(output_path))
            if n >= 500:
                logger.info(f"V2 events already extracted for {target_date} ({n}), skip")
                return output_path
            else:
                os.remove(str(output_path))

        # Load and dedup news
        stock_news = {}
        with open(news_path) as f:
            for line in f:
                item = json.loads(line)
                code = item.get("stock_code", item.get("qlib_code", "")[-6:])
                if not code:
                    continue
                stock_news.setdefault(code, []).append(item)

        tasks = []
        seen = set()
        for code, news_list in stock_news.items():
            for item in news_list[:max_news_per_stock]:
                title = item.get("title", "").strip()
                if not title:
                    continue
                key = f"{code}_{title[:30]}"
                if key in seen:
                    continue
                seen.add(key)
                tasks.append((code, item))

        logger.info(f"V2 extracting {len(tasks)} items for {target_date}")

        total_ok, total_fail = 0, 0

        def _process(task):
            code, item = task
            title = item.get("title", "")
            content = item.get("content_snippet", "")
            name = item.get("stock_name", "")
            source = item.get("source", "unknown")
            source_info = SOURCE_TIERS.get(source, {"tier": "media", "quality": 0.5})

            event = self.extract_single(code, name, title, content)
            if event:
                record = {
                    "stock_code": code,
                    "stock_name": name,
                    "qlib_code": item.get("qlib_code", ""),
                    "publish_time": item.get("publish_time", ""),
                    "title": title,
                    "source": source,
                    "source_tier": source_info["tier"],
                    "source_quality": source_info["quality"],
                    "extract_date": target_date,
                    "extractor_version": "v2",
                    **event,
                }
                return ("ok", record)
            return ("fail", None)

        with open(output_path, "w", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = {executor.submit(_process, t): i for i, t in enumerate(tasks)}
                done = 0
                for future in as_completed(futures):
                    done += 1
                    status, record = future.result()
                    if status == "ok" and record:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                        total_ok += 1
                    else:
                        total_fail += 1
                    if done % 100 == 0:
                        logger.info(f"  V2 progress: {done}/{len(tasks)}, {total_ok} ok, {total_fail} fail")

        logger.info(f"V2 extraction: {total_ok} events, {total_fail} failed -> {output_path}")
        return output_path
