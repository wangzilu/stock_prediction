"""SPIKE — Insider trading announcement extractor (LLM Channel 3).

Dedicated LLM extractor for A-share 高管/股东 增减持 announcements. The
generic ``factors/llm_event_extractor_v2.py`` carries an enum
``insider_buy / insider_sell`` but lumps them with 27 other event_types,
yielding a per-day insider event count of ~5-10 with no holder-type
breakdown — insufficient granularity for the alpha the literature claims.

Channel 3 schema (per-announcement)
-----------------------------------
    ts_code               : "SZ002444" (qlib upper-case format)
    announce_date         : YYYY-MM-DD (公告日, NOT execution-completion date)
    action                : "增持" | "减持" | "新进" | "协议转让出让"
                            | "协议转让受让" | "被动稀释"
    holder_type           : "实控人" | "控股股东" | "董监高"
                            | "持股5%以上股东" | "战略投资者" | "普通股东"
                            | "外部机构" | "其他"
    holder_name           : str (公告中披露的股东全称, ≤ 60 字)
    shares_changed        : float (万股; +增持 / -减持 / 0 不可考)
    pct_of_company        : float (本次变动占总股本%, e.g. 0.5 for 0.5%)
    pct_of_holder_position_change : float
                            ("减持30% of own holding" → 30.0; 增持 0)
    is_committed_no_sell  : bool (本公告是否包含未来 6/12 个月不减持承诺)
    reason_disclosed      : "个人资金需求" | "股权激励" | "补充流动性"
                            | "战略调整" | "财务投资退出" | "其他" | ""
    price_band            : str ("≥¥X.XX" / "区间¥X-Y" / "" 无披露)
    summary_sentence      : str (≤ 50 字, 客观陈述)
    confidence            : float ∈ [0, 1]
    is_official_disclosure: bool (always True for exchange announcements)

PIT contract
------------
- ``announce_date`` is the **signal day**. Execution date is +1 BDay
  (post-15:00 announcements get auto-shifted by the upstream cache
  joiner — see ``factors/llm_event_extractor_v2.py`` `publish_hour >= 15`
  branch and the build script's `BDay(1)` lag.
- Insider announcements at the exchanges have publish timestamps that
  are almost always 19:00-22:00 (after close, batched), so the +1 BDay
  lag is effectively the default path. The +0 fast-path exists for the
  rare pre-09:30 disclosure (typically 公告补正 / 自愿性披露).

Cost ceiling
------------
Daily volume estimate: ~50-100 insider announcements / day post-filter
(see ``scripts/collect_insider_announcements.py``). With MiniMax-Text-01
@ ~400 tokens / call and ¥0.001 / 1k tokens, daily budget ≈ ¥0.05.
Backfill 1y ≈ 250 * ¥0.05 ≈ ¥12.50 — well inside the LLM monthly cap.

Reuse pattern
-------------
- LLM transport reuses ``LLMEventExtractorV2._call_llm`` (rate-limit,
  retry, 429 backoff, persistent retry queue). The system prompt is
  threaded as a kwarg, NOT via module-global monkey-patch (the
  2026-06-07 cx review P1 #2 fix already exposed this hook).
- Output JSONL writes via ``.tmp`` + ``os.replace`` for atomicity.
- ``confidence`` calibration mirrors V2: exchange disclosure 0.9+,
  off-exchange media reposting 0.5-0.7. We hard-cap at 1.0 because
  exchanges occasionally make typos in numerics.

SPIKE status
------------
SCAFFOLD ONLY. The ``extract_single`` path is wired but NOT executed —
running it would burn LLM tokens before the Phase 1 filter precision
audit and the IC backtest validation gate.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import DATA_DIR
from factors.llm_event_extractor_v2 import LLMEventExtractorV2

logger = logging.getLogger(__name__)

EVENTS_DIR = DATA_DIR / "llm_events_insider"
EVENTS_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_DIR_DLQ = DATA_DIR / "llm_retry_queue_insider"
EVENTS_DIR_DLQ.mkdir(parents=True, exist_ok=True)

# ── Enum sets (mirror the schema docstring; runtime-validated) ───────
ACTIONS = (
    "增持", "减持", "新进",
    "协议转让出让", "协议转让受让",
    "被动稀释",
    "其他",
)
HOLDER_TYPES = (
    "实控人",
    "控股股东",
    "董监高",
    "持股5%以上股东",
    "战略投资者",
    "普通股东",
    "外部机构",
    "其他",
)
REASONS = (
    "个人资金需求",
    "股权激励",
    "补充流动性",
    "战略调整",
    "财务投资退出",
    "其他",
    "",
)

EXTRACTOR_VERSION = "insider-v0.1-spike-20260609"


SYSTEM_PROMPT_INSIDER = """从A股高管/股东增减持公告中提取结构化事实。仅事实, 不做预测/不评估影响。

只输出JSON, 第一个字符必须是 { 。所有数值必须是 number, 不是字符串。

Schema (必须包含全部字段; 无法判断时设为 null/"" 并降低 confidence):
{
"action": "增持|减持|新进|协议转让出让|协议转让受让|被动稀释|其他",
"holder_type": "实控人|控股股东|董监高|持股5%以上股东|战略投资者|普通股东|外部机构|其他",
"holder_name": "<股东全称, ≤60字>",
"shares_changed": <万股, 增持正数, 减持负数, 无法判断 0>,
"pct_of_company": <%, e.g. 0.5 表示 0.5%, 无法判断 0>,
"pct_of_holder_position_change": <%, e.g. 30.0 表示该股东减持自己30%持仓, 增持/无法判断 0>,
"is_committed_no_sell": true|false,
"reason_disclosed": "个人资金需求|股权激励|补充流动性|战略调整|财务投资退出|其他|",
"price_band": "<≤30字, 无则空字符串>",
"summary_sentence": "<≤50字事实概述, 客观, 不要分析>",
"confidence": <0-1: 交易所原文 0.85+, 媒体转载 0.5-0.7>
}

规则:
- action="减持"时 shares_changed 必须为负数 (e.g. -120.5)。
- 公告标题含"被动稀释"/"持股比例被动稀释"时 action="被动稀释"。
- 公告内含"未来6个月不减持"等承诺时 is_committed_no_sell=true。
- 减持新规下的"预披露公告"action仍然是"减持"(意图明确)。
- 无法确定的数字字段一律 0, 不要猜测。"""


class InsiderTradingExtractor:
    """LLM extractor for A-share insider 增减持 announcements.

    Reuses LLMEventExtractorV2 for transport (rate-limit, retry, 429
    backoff). Only the prompt + parse contract differ.

    SPIKE: not yet exercised end-to-end. The ``extract_single`` path
    raises if MINIMAX_API_KEY is unset, same as V2.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "MiniMax-Text-01",
        max_calls_per_minute: int = 60,
    ):
        # Compose, don't inherit — V2's extract_single / extract_from_news_file
        # carry V2-specific schema parsing that would silently drop our
        # Channel-3 fields.
        self._transport = LLMEventExtractorV2(
            api_key=api_key,
            model=model,
            max_calls_per_minute=max_calls_per_minute,
        )

    # ── Internal helpers ────────────────────────────────────────────
    def _parse_response(self, text: str) -> dict | None:
        if not text:
            return None
        clean = text.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(clean[start:end])
        except json.JSONDecodeError:
            return None

        # Validate enums (downgrade unknown → "其他" rather than drop —
        # match the V2 / policy event validator philosophy: dropping
        # silently lies about n_extracted).
        action = data.get("action", "其他")
        if action not in ACTIONS:
            action = "其他"
        holder_type = data.get("holder_type", "其他")
        if holder_type not in HOLDER_TYPES:
            holder_type = "其他"
        reason = data.get("reason_disclosed", "")
        if reason not in REASONS:
            reason = "其他"

        def _num(field: str, default: float = 0.0) -> float:
            try:
                v = float(data.get(field, default) or default)
            except (TypeError, ValueError):
                return default
            return v

        shares_changed = _num("shares_changed", 0.0)
        # Schema rule: 减持 must be negative. If the LLM emits positive,
        # flip sign (it's overwhelmingly more likely a sign error than
        # the announcement actually meaning "increase".)
        if action == "减持" and shares_changed > 0:
            shares_changed = -shares_changed

        try:
            conf = max(0.0, min(1.0, float(data.get("confidence", 0.7))))
        except (TypeError, ValueError):
            conf = 0.7

        return {
            "action": action,
            "holder_type": holder_type,
            "holder_name": str(data.get("holder_name", ""))[:80],
            "shares_changed": shares_changed,
            "pct_of_company": _num("pct_of_company", 0.0),
            "pct_of_holder_position_change": _num(
                "pct_of_holder_position_change", 0.0
            ),
            "is_committed_no_sell": bool(data.get("is_committed_no_sell", False)),
            "reason_disclosed": reason,
            "price_band": str(data.get("price_band", ""))[:40],
            "summary_sentence": str(data.get("summary_sentence", ""))[:100],
            "confidence": conf,
            "is_official_disclosure": True,  # filter source = exchange announcement
        }

    def extract_single(
        self,
        ts_code: str,
        stock_name: str,
        title: str,
        content: str = "",
        *,
        announce_date: str = "",
        target_date: str = "",
    ) -> dict | None:
        """Extract one insider event. Returns dict or None.

        ``ts_code`` here uses qlib UPPER-CASE (e.g. ``SZ002444``); the
        downstream feature-cache joiner MUST run ``normalize_instrument_index``
        (see ``factors/feature_cache_utils.py``) before reindexing.
        """
        prompt = f"股票: {ts_code} {stock_name}\n公告日期: {announce_date}\n标题: {title}"
        if content and content != title:
            prompt += f"\n内容: {content[:400]}"
        text, usage = self._transport._call_llm(
            prompt, system_prompt=SYSTEM_PROMPT_INSIDER,
        )
        event = self._parse_response(text)
        # Defer rate-limited retry to the V2 retry queue; we tag the
        # retry record with ``channel`` so the cron retry job routes it
        # back through Channel 3 rather than the V2 prompt.
        if event is None and usage.get("rate_limited") and target_date:
            self._enqueue_retry(
                target_date,
                {
                    "ts_code": ts_code,
                    "stock_name": stock_name,
                    "title": title,
                    "content": content,
                    "announce_date": announce_date,
                    "channel": "insider_trading",
                },
            )
        return event

    def _enqueue_retry(self, target_date: str, item: dict) -> None:
        path = EVENTS_DIR_DLQ / f"{target_date}.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("insider retry queue append failed (%s): %s", path, e)

    def extract_from_announcements_file(
        self,
        ann_path: Path,
        target_date: str | None = None,
        max_workers: int = 8,
    ) -> Path:
        """Process a filtered insider announcement JSONL file.

        Expects ``scripts/collect_insider_announcements.py`` output:
            {stock_code, stock_name, title, qlib_code, notice_date, ...}

        Emits one event per (stock, ann_id) to
            data/storage/llm_events_insider/<YYYY-MM-DD>.jsonl
        """
        ann_path = Path(ann_path)
        if target_date is None:
            target_date = ann_path.stem
        out_path = EVENTS_DIR / f"{target_date}.jsonl"

        # Idempotency: a non-empty existing file is treated as final.
        # The Phase 1 backfill MAY need to overwrite — call sites can
        # ``rm`` before re-running, same as the V2 extractor.
        if out_path.exists():
            n = sum(1 for _ in open(out_path))
            if n >= 1:
                logger.info(
                    "insider events already extracted for %s (%d), skip",
                    target_date,
                    n,
                )
                return out_path

        tasks = []
        with open(ann_path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                tasks.append(item)

        logger.info(
            "Channel 3 LLM extracting %d insider items for %s",
            len(tasks),
            target_date,
        )

        ok, fail = 0, 0

        def _process(item):
            qlib = item.get("qlib_code", "").upper()
            stock_code = item.get("stock_code", "")
            ts_code = qlib or stock_code
            name = item.get("stock_name", "")
            title = item.get("title", "")
            content = item.get("content_snippet", "")
            announce_date = item.get("notice_date", "") or item.get(
                "publish_time", ""
            )
            event = self.extract_single(
                ts_code,
                name,
                title,
                content,
                announce_date=announce_date,
                target_date=target_date,
            )
            if event:
                record = {
                    "ts_code": ts_code,
                    "stock_code": stock_code,
                    "stock_name": name,
                    "qlib_code": qlib or "",
                    "announce_date": announce_date,
                    "title": title,
                    "source": "eastmoney_announcement",
                    "extract_date": target_date,
                    "extractor_version": EXTRACTOR_VERSION,
                    "channel": "insider_trading",
                    **event,
                }
                return ("ok", record)
            return ("fail", None)

        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_process, t): i for i, t in enumerate(tasks)}
                for fut in as_completed(futures):
                    status, record = fut.result()
                    if status == "ok" and record:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                        ok += 1
                    else:
                        fail += 1
        os.replace(tmp_path, out_path)

        logger.info(
            "Channel 3 extraction: %d events, %d failed → %s",
            ok, fail, out_path,
        )
        return out_path
