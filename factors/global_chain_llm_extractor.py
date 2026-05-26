"""Global Supply Chain LLM Extractor — Phase 4U-2.

Takes pre-filtered candidate news (100-200 items) and uses LLM to extract
structured supply chain events. LLM does NOT predict returns — only extracts
facts, direction, confidence, and affected products.

Pipeline position:
  raw news (1400+) → prefilter (100-200) → THIS (30-80 events) → chain factors

Usage:
    from factors.global_chain_llm_extractor import extract_chain_events_llm
    events = extract_chain_events_llm(candidate_news)
"""
import json
import logging
import os
import re
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# LLM config — uses MiniMax (domestic, cheap)
MINIMAX_API_URL = "https://api.minimax.chat/v1/text/chatcompletion_v2"

SYSTEM_PROMPT = """You are a supply chain analyst. Given a news headline, determine if it describes a supply chain event that could affect A-share companies.

Output ONLY a JSON object with these fields:
{
  "is_supply_chain_event": true/false,
  "global_entity": "Nvidia/Apple/Tesla/TSMC/...",
  "industry": "AI server/semiconductor/EV/Apple chain/...",
  "event_type": "demand_increase/order_cut/capex_increase/supply_shortage/export_control/guidance_up/guidance_down/price_cut/price_increase/partnership/other",
  "direction": 1 or -1 or 0,
  "affected_products": ["optical module", "HBM", "..."],
  "time_horizon_days": 1-30,
  "evidence_level": "official/major_media/rumor",
  "is_new_information": true/false,
  "risk_flags": ["export_control", "inventory_buildup", "price_war", "..."],
  "summary": "one sentence in English"
}

Rules:
- Set is_supply_chain_event=false for generic market commentary without specific supply chain facts
- direction: 1=positive for the entity's suppliers, -1=negative, 0=neutral
- Do NOT predict stock prices or returns
- Do NOT include magnitude_value or impact percentages
- Keep summary factual, not opinative"""

USER_PROMPT_TEMPLATE = "News headline: {title}"


class GlobalChainLLMExtractor:
    """LLM-based supply chain event extractor using MiniMax."""

    def __init__(self, api_key: str = None, model: str = "MiniMax-Text-01"):
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        if not self.api_key:
            try:
                from config.settings import MINIMAX_API_KEY
                self.api_key = MINIMAX_API_KEY
            except (ImportError, AttributeError):
                pass
        self.model = model
        self.timeout = 15  # per-request timeout
        self.max_retries = 1

    def extract_one(self, title: str) -> Optional[dict]:
        """Extract supply chain event from a single news title."""
        if not self.api_key or not title:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(title=title)},
            ],
            "temperature": 0.1,
            "max_tokens": 300,
        }

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    MINIMAX_API_URL, headers=headers,
                    json=payload, timeout=self.timeout,
                )
                if resp.status_code == 429:
                    time.sleep(2)
                    continue
                if resp.status_code != 200:
                    return None

                text = resp.json()["choices"][0]["message"]["content"]
                # Remove think tags if present
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                return self._parse_response(text)

            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(1)
                    continue
                logger.debug(f"LLM extract failed: {e}")
                return None

        return None

    def _parse_response(self, text: str) -> Optional[dict]:
        """Parse LLM response into structured event."""
        if not text:
            return None

        # Strip markdown fences
        clean = text.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        # Find JSON
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start < 0 or end <= start:
            return None

        try:
            data = json.loads(clean[start:end])
        except json.JSONDecodeError:
            return None

        # Filter: only supply chain events
        if not data.get("is_supply_chain_event", False):
            return None

        return {
            "global_entity": data.get("global_entity", ""),
            "industry": data.get("industry", ""),
            "event_type": data.get("event_type", "other"),
            "direction": int(data.get("direction", 0)),
            "affected_products": data.get("affected_products", []),
            "time_horizon_days": int(data.get("time_horizon_days", 5)),
            "evidence_level": data.get("evidence_level", "major_media"),
            "is_new_information": bool(data.get("is_new_information", True)),
            "risk_flags": data.get("risk_flags", []),
            "summary": data.get("summary", ""),
            "source": "llm",
        }


def extract_chain_events_llm(
    candidates: list[dict],
    max_extract: int = 80,
    sleep_between: float = 0.3,
) -> list[dict]:
    """Extract supply chain events from pre-filtered candidates via LLM.

    Args:
        candidates: pre-filtered news items (from global_chain_prefilter)
        max_extract: max items to send to LLM
        sleep_between: seconds between LLM calls

    Returns:
        List of structured chain events (only those LLM confirmed as supply chain)
    """
    extractor = GlobalChainLLMExtractor()
    if not extractor.api_key:
        logger.warning("No MINIMAX_API_KEY — falling back to rule-based extraction")
        from factors.global_supply_chain_extractor import batch_extract
        return batch_extract(candidates)

    events = []
    items_to_process = candidates[:max_extract]

    logger.info(f"LLM extracting {len(items_to_process)}/{len(candidates)} candidates...")

    for i, item in enumerate(items_to_process):
        title = item.get("title", "")
        if not title:
            continue

        event = extractor.extract_one(title)
        if event:
            # Merge metadata from the news item
            event["date"] = item.get("date", "")
            event["news_title"] = title
            event["news_url"] = item.get("url", "")
            event["news_source"] = item.get("domain", item.get("source_type", ""))
            event["chain_relevance_score"] = item.get("chain_relevance_score", 0)
            events.append(event)

        if sleep_between > 0:
            time.sleep(sleep_between)

        if (i + 1) % 20 == 0:
            logger.info(f"  Progress: {i+1}/{len(items_to_process)}, {len(events)} events extracted")

    logger.info(f"LLM extraction: {len(items_to_process)} candidates → {len(events)} supply chain events")
    return events
