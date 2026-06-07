"""Global Supply Chain LLM Extractor — Phase 4U-2 + Phase D / SC-A2.

Takes pre-filtered candidate news (100-200 items) and uses an LLM to extract
structured supply chain facts.

Two schemas coexist:

* **v1 (legacy)** — per-event ``direction`` / ``confidence`` / ``event_type``.
  Conflates "X happened" with "X is bullish for stock Y", which the
  2026-06-06 project-lead critique flagged as the LLM's weakness.
  Still emitted by ``extract_chain_events_llm`` so the in-flight
  ``global_chain_events_llm/`` backfill (PID 3026) keeps writing the
  shape its downstream parquet builder expects.

* **v2 (SC-A2, this file's new path)** — per-event ``src_entity`` and a
  list of ``relations``. The LLM extracts RELATIONS only. It never
  predicts whether the A-share supplier rallies or falls. Whether a
  relation implies upside or downside is a downstream weighting step
  (news polarity × relation type), separated from extraction so the
  LLM's known weakness (stance prediction) cannot leak into the
  factor.

The v2 schema is COMPATIBLE with ``data/config/supply_chain_edges.yaml``:
``src_entity`` maps to the YAML's ``src_entity`` field, ``relations[i].target_entity``
maps to a YAML row's downstream entity, and ``evidence_strength`` aligns
with the SC-A3 A/B/C/D tier on each YAML row.

Usage::

    from factors.global_chain_llm_extractor import (
        extract_chain_events_llm,          # v1 — legacy
        extract_chain_events_llm_v2,       # v2 — relations
    )
    events_v2 = extract_chain_events_llm_v2(candidate_news)

Pipeline position::

    raw news (1400+) → prefilter (100-200) → THIS (30-80 events) → chain factors
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
MINIMAX_API_URL = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"

# ---------------------------------------------------------------------------
# v1 schema (legacy — kept for the in-flight global_chain_events_llm backfill)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# v2 schema (SC-A2 — relations only, no direction)
# ---------------------------------------------------------------------------

# Allowed relation types. These line up with the categories
# ``data/config/supply_chain_edges.yaml`` already uses (``customer_supplier``,
# ``peer_readthrough``, …) but expressed in the simpler vocabulary the
# project-lead's SC-A2 spec demands.
ALLOWED_RELATION_TYPES = frozenset({
    "supplier",
    "customer",
    "competitor",
    "joint_venture",
    "regulatory_target",
    "theme_co_member",
})

# Evidence strength tiers — aligned with SC-A3.
#   A — 公告 / 年报 (official)
#   B — 订单 / 合作 (company-interaction)
#   C — 研报 / 调研 (research)
#   D — 主题 / 逻辑 (theme / narrative)  ← default when uncertain
ALLOWED_EVIDENCE_STRENGTHS = frozenset({"A", "B", "C", "D"})

# Factuality buckets.
ALLOWED_FACTUALITIES = frozenset({"confirmed", "speculation", "rumor"})

# Default vocabulary loaded from the supply_chain_edges.yaml so the LLM is
# nudged toward names the downstream propagation graph already knows about.
# Kept tiny on purpose — the prompt only lists the canonical names; the LLM
# can still emit other names and they pass through.
_VOCAB_HINT = (
    "Nvidia, AMD, Intel, TSMC, ASML, Apple, Microsoft, Google, Meta, "
    "Amazon, Tesla, Samsung, SK_Hynix, Micron, CATL, BYD, Qualcomm, "
    "MediaTek, Pfizer, Roche, Novartis, AstraZeneca, Lockheed_Martin, "
    "Raytheon, BAE, US_Export_Control, China_Export_Control, "
    "Lithium_Price, Cobalt_Price, Copper_Price, Rare_Earth_Price, "
    "Oil_Price, Polysilicon_Price, Solar_Tariff"
)

SYSTEM_PROMPT_V2 = (
    "You are a supply-chain RELATION extractor.\n"
    "\n"
    "Given a news headline you EXTRACT RELATIONS between entities. You do "
    "NOT predict whether any stock will go up or down. You do NOT output a "
    "direction field. You do NOT output a confidence number for a price "
    "move. A relation existing is the fact; whether it implies upside or "
    "downside is decided downstream by combining the relation type with the "
    "news polarity.\n"
    "\n"
    "Output ONLY a JSON object with these fields:\n"
    "{\n"
    '  "is_supply_chain_event": true/false,\n'
    '  "src_entity": "<the news subject, e.g. Nvidia>",\n'
    '  "relations": [\n'
    '    {"target_entity": "<other entity>",\n'
    '     "relation_type": "supplier|customer|competitor|joint_venture|regulatory_target|theme_co_member",\n'
    '     "evidence_strength": "A|B|C|D"}\n'
    "  ],\n"
    '  "topic": "<thematic tag, optional, e.g. AI_server>",\n'
    '  "factuality": "confirmed|speculation|rumor",\n'
    '  "summary": "<one short factual sentence, no stance>"\n'
    "}\n"
    "\n"
    "Rules:\n"
    "- Set is_supply_chain_event=false for generic market commentary that "
    "does NOT name a real relation between two entities.\n"
    "- src_entity is MANDATORY whenever is_supply_chain_event=true. It is "
    "the news subject, not the affected A-share.\n"
    "- relations is MANDATORY whenever is_supply_chain_event=true; emit at "
    "least one {target_entity, relation_type, evidence_strength}.\n"
    "- relation_type MUST be one of: supplier, customer, competitor, "
    "joint_venture, regulatory_target, theme_co_member. Any other label "
    "will be rejected.\n"
    "- evidence_strength MUST be one of: A (official 公告/年报), "
    "B (公司互动 订单/合作), C (研报/调研), D (主题/行业逻辑). When you "
    "cannot judge the strength, assign D — never invent strength.\n"
    "- Prefer entity names from this vocabulary when the news clearly "
    f"refers to them: {_VOCAB_HINT}.\n"
    "- Do NOT predict stock direction. Do NOT include a 'direction' field. "
    "Do NOT include a confidence score. Do NOT include impact percentages.\n"
    "- Keep summary factual; no opinion, no stance, no price target."
)

USER_PROMPT_TEMPLATE_V2 = "News headline: {title}"


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

    # ------------------------------------------------------------------
    # Shared transport
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        title: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        """Single LLM request returning the raw response text.

        ``system_prompt`` is passed per-call (NOT via module-global
        monkey patch — that pattern was the PE-1 fix on 2026-06-07 for
        the V2 extractor). Defaults to the v1 ``SYSTEM_PROMPT`` so
        existing v1 callers are unchanged.
        """
        if not self.api_key or not title:
            return None
        sys_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(title=title)},
            ],
            "temperature": 0.1,
            "max_tokens": 400,
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
                return text
            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(1)
                    continue
                logger.debug(f"LLM extract failed: {e}")
                return None
        return None

    # ------------------------------------------------------------------
    # v1 path (legacy — direction / confidence)
    # ------------------------------------------------------------------

    def extract_one(self, title: str) -> Optional[dict]:
        """Extract supply chain event from a single news title (v1 schema)."""
        text = self._call_llm(title, system_prompt=SYSTEM_PROMPT)
        if not text:
            return None
        return self._parse_response(text)

    def _parse_response(self, text: str) -> Optional[dict]:
        """Parse LLM response into structured v1 event."""
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

    # ------------------------------------------------------------------
    # v2 path (SC-A2 — relations, no direction)
    # ------------------------------------------------------------------

    def extract_one_v2(self, title: str) -> Optional[dict]:
        """Extract supply chain relations from a single news title (v2 schema).

        The returned dict shape is what callers will write to JSONL::

            {
              "src_entity": str,
              "relations": [{"target_entity": str,
                             "relation_type": str,
                             "evidence_strength": "A"|"B"|"C"|"D"}, ...],
              "topic": str,
              "factuality": "confirmed"|"speculation"|"rumor",
              "summary": str,
              "schema_version": "v2",
              "source": "llm",
            }

        Returns ``None`` when the LLM call fails, the JSON is unparseable,
        ``is_supply_chain_event`` is false, ``src_entity`` is missing, or
        no valid relations survive normalisation.
        """
        text = self._call_llm(title, system_prompt=SYSTEM_PROMPT_V2)
        if not text:
            return None
        return parse_response_v2(text)


def _normalize_relation(raw: dict) -> Optional[dict]:
    """Validate one relation. Unknown ``evidence_strength`` downgrades
    to ``"D"``. Unknown ``relation_type`` rejects the relation entirely
    (we will not invent an enum value)."""
    if not isinstance(raw, dict):
        return None
    target = (raw.get("target_entity") or "").strip()
    if not target:
        return None
    rel_type = (raw.get("relation_type") or "").strip()
    if rel_type not in ALLOWED_RELATION_TYPES:
        return None
    strength = (raw.get("evidence_strength") or "").strip().upper()
    if strength not in ALLOWED_EVIDENCE_STRENGTHS:
        strength = "D"  # spec rule (c): unknown evidence → D
    return {
        "target_entity": target,
        "relation_type": rel_type,
        "evidence_strength": strength,
    }


def parse_response_v2(text: str) -> Optional[dict]:
    """Parse a raw LLM response into the v2 schema. Returns ``None``
    when the spec's mandatory invariants don't hold.

    Spec rule (d): refuse to emit a row when ``src_entity`` or
    ``relations`` is missing / empty.
    """
    if not text:
        return None

    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(clean[start:end])
    except json.JSONDecodeError:
        return None

    # Filter out non-supply-chain or unflagged commentary.
    if not data.get("is_supply_chain_event", False):
        return None

    src_entity = (data.get("src_entity") or "").strip()
    raw_relations = data.get("relations")
    # Mandatory-field check (spec rule d).
    if not src_entity:
        return None
    if not isinstance(raw_relations, list) or not raw_relations:
        return None

    relations = []
    for r in raw_relations:
        norm = _normalize_relation(r)
        if norm is not None:
            relations.append(norm)

    # If LLM emitted only invalid relations, refuse the row.
    if not relations:
        return None

    factuality = (data.get("factuality") or "").strip().lower()
    if factuality not in ALLOWED_FACTUALITIES:
        factuality = "speculation"  # conservative default

    return {
        "src_entity": src_entity,
        "relations": relations,
        "topic": (data.get("topic") or "").strip(),
        "factuality": factuality,
        "summary": (data.get("summary") or "").strip(),
        "schema_version": "v2",
        "source": "llm",
    }


# ---------------------------------------------------------------------------
# Batch entry points
# ---------------------------------------------------------------------------

def extract_chain_events_llm(
    candidates: list[dict],
    max_extract: int = 80,
    sleep_between: float = 0.3,
) -> list[dict]:
    """Extract v1 supply chain events (legacy direction/confidence schema).

    Kept unchanged so the in-flight backfill writing to
    ``data/storage/global_chain_events_llm/`` keeps producing the same
    JSONL shape its downstream parquet builder reads under
    ``--schema v1``.

    Args:
        candidates: pre-filtered news items (from global_chain_prefilter)
        max_extract: max items to send to LLM
        sleep_between: seconds between LLM calls
    """
    extractor = GlobalChainLLMExtractor()
    if not extractor.api_key:
        logger.warning("No MINIMAX_API_KEY — falling back to rule-based extraction")
        from factors.global_supply_chain_extractor import batch_extract
        return batch_extract(candidates)

    events = []
    items_to_process = candidates[:max_extract]

    logger.info(f"LLM extracting {len(items_to_process)}/{len(candidates)} candidates (v1)...")

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

    logger.info(f"LLM extraction (v1): {len(items_to_process)} candidates → {len(events)} events")
    return events


def extract_chain_events_llm_v2(
    candidates: list[dict],
    max_extract: int = 80,
    sleep_between: float = 0.3,
) -> list[dict]:
    """Extract v2 supply chain RELATIONS (no direction).

    Same pre-filter input shape as ``extract_chain_events_llm`` but
    every event carries an ``src_entity`` and a ``relations`` list
    instead of a ``direction``/``confidence`` pair. The downstream
    propagation step computes upside/downside from
    ``news polarity × relation_type``; this function never does.
    """
    extractor = GlobalChainLLMExtractor()
    if not extractor.api_key:
        logger.warning(
            "No MINIMAX_API_KEY — v2 schema requires LLM. Returning empty list "
            "(rule-based fallback is direction-shaped and incompatible)."
        )
        return []

    events: list[dict] = []
    items_to_process = candidates[:max_extract]

    logger.info(
        f"LLM extracting {len(items_to_process)}/{len(candidates)} candidates (v2 relations)..."
    )

    for i, item in enumerate(items_to_process):
        title = item.get("title", "")
        if not title:
            continue
        event = extractor.extract_one_v2(title)
        if event:
            event["date"] = item.get("date", "")
            event["news_title"] = title
            event["news_url"] = item.get("url", "")
            event["news_source"] = item.get("domain", item.get("source_type", ""))
            event["chain_relevance_score"] = item.get("chain_relevance_score", 0)
            events.append(event)

        if sleep_between > 0:
            time.sleep(sleep_between)
        if (i + 1) % 20 == 0:
            logger.info(
                f"  Progress: {i+1}/{len(items_to_process)}, "
                f"{len(events)} relations rows extracted"
            )

    logger.info(
        f"LLM extraction (v2): {len(items_to_process)} candidates → {len(events)} rows"
    )
    return events
