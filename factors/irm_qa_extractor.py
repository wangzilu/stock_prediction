"""LLM extractor for investor-interaction Q&A from IRM cninfo / SSE e-interact.

SPIKE STATUS (2026-06-09): scaffold only. The class is wired (mirrors
``LLMEventExtractorV2``'s rate-limit + retry-queue plumbing) but the
caller / cron / EventStore wiring is intentionally deferred. See
docs/llm_channel_2_irm_qa_spike_20260609.md for the go/no-go criteria.

Design principles (carried over from V2)
----------------------------------------
1. LLM extracts FACTS, never impact predictions. The 1-to-5
   ``information_value_score`` is the LLM's assessment of substance
   (does the answer contain a specific number / date / product /
   commitment?), NOT a price-impact forecast.
2. ``forward_signal_direction`` is the SIGN of management's
   communication, not the sign of expected return. Example: management
   says "Q2 orders are down YoY"; direction = -1 because that's the
   sign of their forward statement. The return-impact mapping is
   calibrated downstream against historical returns, NOT here.
3. Use non-reasoning model (MiniMax-Text-01 by default). Per
   memory/feedback_llm_pipeline_arch.md, reasoning models burn ~95% of
   completion tokens on <think> blocks for an extraction task.
4. Post-LLM keyword gate. The LLM cheerfully tags every answer that
   mentions "订单" as ``orders`` topic. ``IRM_TOPIC_GATES`` below
   provides a sanity rail mirroring the event_schema_validator pattern.

Schema (output JSON, validated row-level)
-----------------------------------------
::

    question_topic:               <enum, see IRM_TOPIC_TYPES>
    is_substantive:               true | false
    information_value_score:      1 | 2 | 3 | 4 | 5
    forward_signal_direction:     1 | -1 | 0
    contains_guidance_change:     true | false
    contains_specific_number:     true | false
    summary_sentence:             "<≤40字 单句事实概述>"
    confidence:                   0.0-1.0

Score guide (kept in the system prompt, repeated here for reviewers)
--------------------------------------------------------------------
* 5 — explicit guidance, hard number, dated commitment
       ("Q3 revenue guidance raised from X to Y", "投产时间Q4")
* 4 — qualitative but specific firm-level fact
       ("某项目订单已签订", "新产品10月上市")
* 3 — generic confirmation of public info, no new detail
       ("正在按计划推进", "请关注定期报告")
* 2 — IR template ("感谢关注", "建议查阅公司公告")
* 1 — no reply or boilerplate / off-topic

The 1-2 buckets are by design the LOW info path. Downstream factor
construction filters substantive=True before aggregating, but the raw
1-2 rows are still PERSISTED — they're useful as a denominator (how
many questions did the company DODGE?).
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

from config.settings import DATA_DIR, MINIMAX_API_KEY

logger = logging.getLogger(__name__)

IRM_EVENTS_DIR = DATA_DIR / "irm_qa_extracted"
IRM_EVENTS_DIR.mkdir(parents=True, exist_ok=True)

# Shared retry queue with the event V2 pipeline. A 429 from MiniMax on
# the IRM channel deserves the same delayed retry treatment as on the
# news channel — they share the per-account RPM cap.
RETRY_QUEUE_DIR = DATA_DIR / "llm_retry_queue"
RETRY_QUEUE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Topic enum + keyword gates
# ─────────────────────────────────────────────────────────────────────
IRM_TOPIC_TYPES = [
    "guidance",        # 业绩指引 / forward outlook
    "orders",          # 在手订单 / new orders
    "capex",           # 资本支出 / 新项目投产
    "product",         # 新产品 / 技术 / R&D progress
    "management",      # 高管变动 / 战略变化
    "regulation",      # 监管 / 合规
    "ma_event",        # 并购重组 / M&A
    "dividend",        # 分红 / 回购
    "shareholding",    # 股东 / 解禁
    "operations",      # 日常经营 / 产能利用率
    "esg",             # ESG / 社会责任
    "other",           # catch-all
]

# Conservative keyword gates: when the LLM tags a question with a
# specific topic, we expect at least one of these tokens to appear in
# either the question OR the answer. If not, downgrade to "other". The
# gate is intentionally low-recall — false negatives at gate time are
# preferable to false positives biasing the factor distribution.
IRM_TOPIC_GATES: dict[str, tuple[str, ...]] = {
    "guidance":     ("指引", "预测", "预期", "目标", "全年", "Q1", "Q2", "Q3", "Q4", "季度", "营收", "净利", "毛利率"),
    "orders":       ("订单", "在手", "签订", "合同", "中标", "客户"),
    "capex":        ("投资", "产能", "扩产", "投产", "建设", "项目", "工厂"),
    "product":      ("产品", "技术", "研发", "新品", "上市", "量产"),
    "management":   ("高管", "董事", "总经理", "战略", "任命", "离任", "聘任"),
    "regulation":   ("监管", "合规", "处罚", "审批", "许可", "证监会"),
    "ma_event":     ("并购", "重组", "收购", "出售", "资产", "标的"),
    "dividend":     ("分红", "股息", "回购", "派息", "分配"),
    "shareholding": ("股东", "解禁", "减持", "增持", "持股", "限售"),
    "operations":   ("产能利用", "开工", "出货", "出库", "库存", "毛利", "日常"),
    "esg":          ("ESG", "环保", "排放", "社会责任", "可持续"),
}

_DIRECTION_STR_MAP = {
    "positive": 1, "up": 1, "bullish": 1, "+1": 1, "1": 1,
    "negative": -1, "down": -1, "bearish": -1, "-1": -1,
    "neutral": 0, "flat": 0, "0": 0, "": 0,
}

SYSTEM_PROMPT = """从A股投资者关系平台问答提取结构化事实。

只输出JSON,禁止解释/思考/前后缀。第一个字符必须是 { 。
direction 是整数 1/-1/0(不是字符串)。
question_topic 必须从枚举选一个。

Schema:
{"question_topic": "<one of: guidance|orders|capex|product|management|regulation|ma_event|dividend|shareholding|operations|esg|other>",
"is_substantive": true|false,
"information_value_score": 1|2|3|4|5,
"forward_signal_direction": 1|-1|0,
"contains_guidance_change": true|false,
"contains_specific_number": true|false,
"summary_sentence": "<≤40字事实概述>",
"confidence": <0-1>}

打分基准:
5 - 明示指引/硬数字/带日期承诺 (如 "Q3营收上调至X亿")
4 - 具体公司层面事实 (如 "某项目订单已签订","新品10月上市")
3 - 通用确认,无新细节 (如 "按计划推进","请关注定期报告")
2 - IR模板套话 (如 "感谢关注","建议查阅公告")
1 - 无回答/明显答非所问

is_substantive 仅当 information_value_score >= 3 时为 true。
forward_signal_direction: 管理层表达的方向,正面=1,负面=-1,中性=0。
contains_specific_number: 回答中是否包含具体数字、日期、百分比、金额。

只提取明示事实,不推断。无法判断时 confidence 设低。"""


# ─────────────────────────────────────────────────────────────────────
# Extractor
# ─────────────────────────────────────────────────────────────────────
class IRMQAExtractor:
    """LLM extractor for investor-interaction Q&A.

    Threading + retry semantics mirror ``LLMEventExtractorV2``:
    rate-limited token bucket, 4-attempt 429-aware retry with
    exponential backoff, per-thread accounting, persistent 429
    retry queue.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "MiniMax-Text-01",
        max_calls_per_minute: int = 60,
    ):
        self.api_key = api_key or MINIMAX_API_KEY
        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY required")
        self.model = model
        self.api_url = "https://api.minimax.io/v1/chat/completions"
        self.max_calls_per_minute = max_calls_per_minute
        self._call_timestamps: list[float] = []
        self._rate_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats: dict = {
            "calls": 0, "http_fail": 0, "parse_fail": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
        }

    # ── plumbing copied 1:1 from LLMEventExtractorV2 ─────────────────
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
        """Same retry policy as V2: 4 attempts, exponential backoff on 429."""
        self._rate_limit()
        last_was_429 = False
        backoffs_429 = [5.0, 15.0, 45.0]
        for attempt in range(4):
            try:
                resp = requests.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json={"model": self.model,
                          "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                       {"role": "user", "content": user_prompt}],
                          "max_tokens": 512},
                    timeout=(5, 20),
                )
                if resp.status_code != 200:
                    last_was_429 = resp.status_code == 429
                    if attempt < 3:
                        if last_was_429 and attempt < len(backoffs_429):
                            wait = backoffs_429[attempt] * random.uniform(0.7, 1.3)
                        else:
                            wait = 3.0
                        time.sleep(wait)
                        continue
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
            except Exception:
                if attempt < 3:
                    time.sleep(3.0 * random.uniform(0.7, 1.3))
        return "", {"http_ok": False}

    # ── parsing + validation ──────────────────────────────────────────
    def _parse_response(self, text: str) -> dict | None:
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
            m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if not m:
                return None
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return None

        topic = data.get("question_topic", "other")
        if topic not in IRM_TOPIC_TYPES:
            topic = "other"

        d_raw = data.get("forward_signal_direction", 0)
        if isinstance(d_raw, str):
            direction = _DIRECTION_STR_MAP.get(d_raw.strip().lower(), 0)
        else:
            try:
                direction = max(-1, min(1, int(d_raw)))
            except (TypeError, ValueError):
                direction = 0

        try:
            score = int(data.get("information_value_score", 1))
            score = max(1, min(5, score))
        except (TypeError, ValueError):
            score = 1

        try:
            conf = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5

        return {
            "question_topic": topic,
            "is_substantive": bool(data.get("is_substantive", score >= 3)),
            "information_value_score": score,
            "forward_signal_direction": direction,
            "contains_guidance_change": bool(data.get("contains_guidance_change", False)),
            "contains_specific_number": bool(data.get("contains_specific_number", False)),
            "summary_sentence": str(data.get("summary_sentence", ""))[:80],
            "confidence": conf,
        }

    @staticmethod
    def _topic_gate(topic: str, question: str, answer: str) -> tuple[str, str]:
        """Mirror event_schema_validator's keyword gate: if a topic-specific
        keyword does NOT appear in either question or answer, downgrade to
        ``other`` with a reason string.

        Returns ``(final_topic, downgrade_reason)``. When no downgrade,
        reason is the empty string.
        """
        if topic == "other":
            return topic, ""
        gate = IRM_TOPIC_GATES.get(topic)
        if not gate:
            return topic, ""
        haystack = f"{question}\n{answer}"
        if any(kw in haystack for kw in gate):
            return topic, ""
        return "other", f"no_{topic}_keyword_in_qa"

    # ── public surface ────────────────────────────────────────────────
    def extract_single(
        self,
        stock_code: str,
        stock_name: str,
        question: str,
        answer: str,
        *,
        answer_date: str = "",
    ) -> dict | None:
        """Extract one Q&A. Returns the parsed dict (no row metadata) or None
        on parse failure. Caller is responsible for joining stock_code /
        qlib_code / dates back onto the result.
        """
        # Truncate both fields hard — long-tail rants from retail investors
        # blow up token cost and rarely add signal past the first 300 chars.
        q = (question or "").strip()[:300]
        a = (answer or "").strip()[:500]
        if not q:
            return None
        prompt = (
            f"股票: {stock_code} {stock_name}\n"
            f"提问: {q}\n"
            f"回答: {a if a else '(无回答)'}"
        )
        text, usage = self._call_llm(prompt)
        parsed = self._parse_response(text)

        if parsed is not None and parsed["question_topic"] != "other":
            final, reason = self._topic_gate(parsed["question_topic"], q, a)
            if final != parsed["question_topic"]:
                parsed["question_topic_original"] = parsed["question_topic"]
                parsed["question_topic_downgrade_reason"] = reason
                parsed["question_topic"] = final

        rate_limited = usage.get("rate_limited", False)
        self._record_stats(
            usage.get("http_ok", False),
            parsed is not None,
            usage,
            rate_limited=rate_limited,
        )
        if parsed is None and rate_limited and answer_date:
            self._enqueue_retry(answer_date, {
                "stock_code": stock_code, "stock_name": stock_name,
                "question": q, "answer": a,
                "answer_date": answer_date,
            })
        return parsed

    def _enqueue_retry(self, answer_date: str, item: dict) -> None:
        path = RETRY_QUEUE_DIR / f"irm_qa_{answer_date}.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("retry queue append failed (%s): %s", path, e)

    def extract_from_file(
        self,
        qa_path: Path,
        *,
        target_date: str | None = None,
        max_workers: int = 8,
    ) -> Path:
        """Process a day's collected Q&A JSONL and emit an extracted JSONL
        with one row per Q&A.

        Output filename matches input filename (date-keyed). If output
        already has substantial coverage, skip — same protection as the
        V2 extractor.
        """
        qa_path = Path(qa_path)
        if target_date is None:
            target_date = qa_path.stem
        out_path = IRM_EVENTS_DIR / f"{target_date}.jsonl"

        if out_path.exists():
            n = sum(1 for _ in open(out_path, encoding="utf-8"))
            # Heuristic: 100 extracted rows is "enough" — IRM Q&A volume is
            # smaller than news, so the 500-threshold from V2 doesn't fit.
            if n >= 100:
                logger.info("IRM Q&A already extracted for %s (%d), skip", target_date, n)
                return out_path
            os.remove(str(out_path))

        rows: list[dict] = []
        with open(qa_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # Optional pre-filter: extracting un-answered questions is mostly
        # wasted tokens (LLM will return is_substantive=False with low
        # confidence). But we KEEP them so the denominator is intact —
        # downstream "dodge rate" factor needs total Q count.
        logger.info("Extracting %d Q&A rows for %s", len(rows), target_date)

        total_ok, total_fail = 0, 0

        def _process(row: dict):
            parsed = self.extract_single(
                row["stock_code"], row.get("stock_name", ""),
                row.get("question", ""), row.get("answer", ""),
                answer_date=row.get("answer_date", target_date),
            )
            if parsed is None:
                return ("fail", None)
            out = {
                "stock_code": row["stock_code"],
                "stock_name": row.get("stock_name", ""),
                "qlib_code": row.get("qlib_code", ""),
                "venue": row.get("venue", ""),
                "question_id": row.get("question_id", ""),
                "industry": row.get("industry", ""),
                "ask_date": row.get("ask_date", ""),
                "answer_date": row.get("answer_date", ""),
                "is_answered": row.get("is_answered", False),
                "extract_date": target_date,
                "extractor_version": "irm_qa_v1",
                **parsed,
            }
            return ("ok", out)

        with open(out_path, "w", encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_process, r): i for i, r in enumerate(rows)}
                done = 0
                for fut in as_completed(futures):
                    done += 1
                    status, rec = fut.result()
                    if status == "ok" and rec:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        total_ok += 1
                    else:
                        total_fail += 1
                    if done % 100 == 0:
                        logger.info(
                            "  progress: %d/%d, ok=%d, fail=%d",
                            done, len(rows), total_ok, total_fail,
                        )

        logger.info(
            "IRM Q&A extraction: %d ok, %d failed → %s",
            total_ok, total_fail, out_path,
        )
        s = self._stats
        if s["calls"]:
            avg_tok = (s["prompt_tokens"] + s["completion_tokens"]) / s["calls"]
            logger.info(
                "IRM Q&A LLM stats: model=%s calls=%d http_fail=%d "
                "(rate_limited=%d) parse_fail=%d prompt_tok=%d "
                "completion_tok=%d reason_tok=%d avg_tok/call=%.0f",
                self.model, s["calls"], s["http_fail"],
                s.get("rate_limited", 0), s["parse_fail"],
                s["prompt_tokens"], s["completion_tokens"],
                s["reasoning_tokens"], avg_tok,
            )
        return out_path
