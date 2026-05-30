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

# Persistent retry queue: items that fail with 429 even after exponential
# backoff get appended here so a later cron job can re-attempt them.
RETRY_QUEUE_DIR = DATA_DIR / "llm_retry_queue"
RETRY_QUEUE_DIR.mkdir(parents=True, exist_ok=True)

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

SYSTEM_PROMPT_V2 = """从A股新闻/公告提取结构化事实。

只输出JSON，禁止解释、思考、前后缀文字。第一个字符必须是 { 。
direction 必须是整数 1/-1/0（不是字符串）。
event_type 必须从枚举中选一个。

Schema:
{"event_type": "<one of: earnings_beat|earnings_miss|earnings_inline|revenue_growth|revenue_decline|order_win|major_contract|product_launch|tech_breakthrough|market_share_gain|market_share_loss|share_buyback|dividend_increase|insider_buy|insider_sell|share_placement|share_unlock|analyst_upgrade|analyst_downgrade|regulatory_approval|regulatory_penalty|lawsuit_filed|lawsuit_settled|management_change|restructuring|strategic_cooperation|joint_venture|government_subsidy|tax_benefit|debt_issue|credit_rating_change|routine_announcement|other>",
"direction": 1|-1|0,
"is_official_disclosure": true|false,
"is_new_information": true|false,
"is_repeated_news": true|false,
"is_price_sensitive": true|false,
"magnitude_description": "<≤30字, 无则空字符串>",
"magnitude_value_wan": <number 万元, 无则0>,
"confidence": <0-1: 交易所公告 0.9+, 权威媒体 0.7, 传闻 0.3-0.5>,
"summary": "<≤30字事实概述, 不要分析>"}

只提取明示事实, 不推断。无法判断时 confidence 设低。"""


_DIRECTION_STR_MAP = {
    "positive": 1, "up": 1, "bullish": 1, "bull": 1, "+1": 1, "1": 1,
    "negative": -1, "down": -1, "bearish": -1, "bear": -1, "-1": -1,
    "neutral": 0, "flat": 0, "0": 0, "": 0,
}


class LLMEventExtractorV2:
    """V2: Extract structured FACTS from news, not impact predictions.

    Uses MiniMax-Text-01 (non-reasoning) by default — extraction is not a
    reasoning task, and reasoning models burn 95% of completion_tokens on
    <think> blocks. See memory/feedback_llm_pipeline_arch.md.
    """

    def __init__(self,
                 api_key: str = None,
                 model: str = "MiniMax-Text-01",
                 max_calls_per_minute: int = 60):
        self.api_key = api_key or MINIMAX_API_KEY
        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY required")
        self.model = model
        self.api_url = "https://api.minimax.io/v1/chat/completions"
        self.max_calls_per_minute = max_calls_per_minute
        self._call_timestamps = []
        self._rate_lock = threading.Lock()
        # Accounting (thread-safe via _stats_lock)
        self._stats_lock = threading.Lock()
        self._stats = {
            "calls": 0, "http_fail": 0, "parse_fail": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
        }

    def _record_stats(self, http_ok: bool, parse_ok: bool, usage: dict, rate_limited: bool = False):
        with self._stats_lock:
            self._stats["calls"] += 1
            if not http_ok:
                self._stats["http_fail"] += 1
            elif not parse_ok:
                self._stats["parse_fail"] += 1
            if rate_limited:
                self._stats["rate_limited"] = self._stats.get("rate_limited", 0) + 1
            self._stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self._stats["completion_tokens"] += usage.get("completion_tokens", 0)
            self._stats["reasoning_tokens"] += usage.get("reasoning_tokens", 0)

    def _rate_limit(self):
        with self._rate_lock:
            now = time.time()
            self._call_timestamps = [t for t in self._call_timestamps if now - t < 60]
            if len(self._call_timestamps) >= self.max_calls_per_minute:
                wait = 60 - (now - self._call_timestamps[0]) + 0.1
                if wait > 0:
                    time.sleep(wait)
            self._call_timestamps.append(time.time())

    def _call_llm(self, user_prompt: str) -> tuple[str, dict]:
        """Returns (text, usage). usage has http_ok + token counts.

        Retry policy: 4 attempts. 429 (RPM exceeded) gets exponential backoff
        with jitter (5s, 15s, 45s, +/- 30% random); other transient errors
        get a flat 3s wait between attempts. The previous policy (2 attempts,
        flat 3s) dropped 273 of 425 calls today on the rerun because the
        4-thread burst pattern hit MiniMax-Text-01's RPM cap and each thread
        only waited 3s before giving up.
        """
        import random
        self._rate_limit()
        last_err = None
        last_was_429 = False
        backoffs_429 = [5.0, 15.0, 45.0]
        for attempt in range(4):
            try:
                resp = requests.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json={"model": self.model,
                          "messages": [{"role": "system", "content": SYSTEM_PROMPT_V2},
                                       {"role": "user", "content": user_prompt}],
                          "max_tokens": 512},
                    timeout=(5, 20),
                )
                if resp.status_code != 200:
                    last_err = f"HTTP {resp.status_code} {resp.text[:200]}"
                    last_was_429 = resp.status_code == 429
                    if attempt < 3:
                        if last_was_429 and attempt < len(backoffs_429):
                            base = backoffs_429[attempt]
                            wait = base * random.uniform(0.7, 1.3)
                        else:
                            wait = 3.0
                        time.sleep(wait)
                        continue
                    logger.warning("V2 LLM call failed after 4 attempts: %s", last_err)
                    return "", {"http_ok": False, "rate_limited": last_was_429}
                body = resp.json()
                text = body["choices"][0]["message"]["content"]
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                u = body.get("usage", {}) or {}
                details = u.get("completion_tokens_details", {}) or {}
                return text, {
                    "http_ok": True,
                    "prompt_tokens": u.get("prompt_tokens", 0),
                    "completion_tokens": u.get("completion_tokens", 0),
                    "reasoning_tokens": details.get("reasoning_tokens", 0),
                }
            except Exception as e:
                last_err = repr(e)
                if attempt < 3:
                    time.sleep(3.0 * random.uniform(0.7, 1.3))
        if last_err:
            logger.warning("V2 LLM exception after 4 attempts: %s", last_err)
        return "", {"http_ok": False}

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

        # direction may arrive as int, float, or string ("positive"/"+1"/etc.)
        d_raw = data.get("direction", 0)
        if isinstance(d_raw, str):
            direction = _DIRECTION_STR_MAP.get(d_raw.strip().lower(), 0)
        else:
            try:
                direction = max(-1, min(1, int(d_raw)))
            except (TypeError, ValueError):
                direction = 0

        try:
            mag = float(data.get("magnitude_value_wan", 0) or 0)
        except (TypeError, ValueError):
            mag = 0.0
        try:
            conf = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5

        return {
            "event_type": event_type,
            "direction": direction,
            "is_official_disclosure": bool(data.get("is_official_disclosure", False)),
            "is_new_information": bool(data.get("is_new_information", True)),
            "is_repeated_news": bool(data.get("is_repeated_news", False)),
            "is_price_sensitive": bool(data.get("is_price_sensitive", False)),
            "magnitude_description": str(data.get("magnitude_description", ""))[:60],
            "magnitude_value_wan": mag,
            "confidence": conf,
            "summary": str(data.get("summary", ""))[:100],
        }

    def extract_single(self, code: str, name: str, title: str, content: str = "",
                       *, source: str = "", publish_time: str = "",
                       qlib_code: str = "", target_date: str = "") -> dict | None:
        prompt = f"股票: {code} {name}\n标题: {title}"
        if content and content != title:
            prompt += f"\n内容: {content[:300]}"
        text, usage = self._call_llm(prompt)
        event = self._parse_response(text)
        rate_limited = usage.get("rate_limited", False)
        self._record_stats(
            usage.get("http_ok", False),
            event is not None,
            usage,
            rate_limited=rate_limited,
        )
        # Persistent retry queue: only enqueue when the final failure mode
        # was 429 (transient API rate limit), not parse error or invalid
        # input. Parse failures retry would just fail again — only rate-
        # limited items benefit from a delayed second attempt.
        if event is None and rate_limited and target_date:
            self._enqueue_retry(
                target_date,
                {
                    "stock_code": code, "stock_name": name,
                    "title": title, "content": content,
                    "source": source, "publish_time": publish_time,
                    "qlib_code": qlib_code,
                },
            )
        return event

    def _enqueue_retry(self, target_date: str, item: dict) -> None:
        """Append an item to today's retry queue file."""
        path = RETRY_QUEUE_DIR / f"{target_date}.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("retry queue append failed (%s): %s", path, e)

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

            event = self.extract_single(
                code, name, title, content,
                source=source,
                publish_time=item.get("publish_time", ""),
                qlib_code=item.get("qlib_code", ""),
                target_date=target_date,
            )
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

        s = self._stats
        if s["calls"]:
            avg_tok = (s["prompt_tokens"] + s["completion_tokens"]) / s["calls"]
            logger.info(
                "V2 LLM stats: model=%s calls=%d http_fail=%d (rate_limited=%d) "
                "parse_fail=%d prompt_tok=%d completion_tok=%d reason_tok=%d avg_tok/call=%.0f",
                self.model, s["calls"], s["http_fail"], s.get("rate_limited", 0),
                s["parse_fail"], s["prompt_tokens"], s["completion_tokens"],
                s["reasoning_tokens"], avg_tok,
            )
        return output_path
