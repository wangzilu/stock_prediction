"""Research-Rating LLM Extractor — sell-side analyst report parser.

SPIKE 2026-06-09 — LLM Channel #1.

Design principle (CX): the LLM extracts STRUCTURED FACTS the AKShare
Eastmoney row does NOT already carry — specifically:

  * rating_change_direction (upgrade / downgrade / reiterate / initiate)
    — Eastmoney gives the CURRENT rating only; we need to know whether
    it changed vs. the broker's last report on this name. We provide the
    LLM with the latest *previous* rating from the same broker as
    context and ask it to classify the move.
  * target_price (RMB) — Eastmoney does NOT include this; it's only in
    the PDF body or the report-title abstract. The LLM reads the title
    and (in a follow-up) the PDF first page.
  * summary_sentence (≤30字) — used for downstream news-cluster dedup
    and the manual review UI.

The LLM does NOT extract the rating itself (already in Eastmoney's
table), the broker (already structured), the report_date (already
structured), or the EPS forecasts (already columns). Those are passed
THROUGH from the collector record. Saves ~60% of MiniMax tokens vs.
the naive "extract everything" approach.

Output (JSONL row schema)
-------------------------
::

    {
        "stock_code": "600519",
        "qlib_code": "sh600519",
        "report_date": "2026-06-09",
        "broker": "中信证券",
        "rating_current": "buy",                # canonical, from collector
        "rating_previous": "outperform" | null,  # last rating from this broker
        "rating_change": "upgrade" | "downgrade" | "reiterate" | "initiate",
        "target_price": 1850.0 | null,           # RMB, LLM-extracted from title/PDF
        "eps_current_year": 68.71,               # passthrough from collector eps_y1
        "eps_next_year": 73.01,                  # passthrough eps_y2
        "eps_revision_pct": 0.025 | null,        # vs. previous report's eps_y1
        "summary_sentence": "i茅台改革见效, 调高目标价",
        "confidence": 0.85,
        "extracted_at": "2026-06-09T16:30:00",
        "extractor_version": "research_rating_v1"
    }

PIT discipline
--------------
Signal date := ``collected_at`` + 1 BDay, NOT ``report_date``. The
collector tags every row with ``collected_at`` (the harvest moment).
A report published at 09:00 today must NOT predict today's close
because the publish-time→data-pipeline lag is real (we typically
collect at the EOD 16:30 cron) — same-day signals would be lookahead.
The factor builder (``scripts/build_research_rating_factors.py``)
enforces the +1 BDay shift.

LLM model selection
-------------------
MiniMax-Text-01 (non-reasoning), same as LLM event extractor V2 — this
is extraction not reasoning. See ``memory/feedback_llm_pipeline_arch.md``
for the family/family rationale.

Tokens budget
-------------
Each call sends ~250 prompt tokens + receives ~150 completion tokens.
At ~200 reports/day production volume and MiniMax-Text-01 price of
~$1/1M input + $1/1M output, daily cost is ~$0.08. Backfill of one
year ≈ $30. Well under any reasonable budget — no rate-limit risk
provided we reuse the V2 ``_rate_limit()``  60-RPM throttle.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from config.settings import DATA_DIR
from factors.llm_event_extractor_v2 import LLMEventExtractorV2

logger = logging.getLogger(__name__)

EXTRACTED_DIR = DATA_DIR / "research_rating_extracted"
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

EXTRACTOR_VERSION = "research_rating_v1"

# Canonical rating ordering — used to compute the upgrade/downgrade
# direction when both current + previous ratings are known.
RATING_RANK = {
    "sell": -2,
    "underperform": -1,
    "hold": 0,
    "outperform": 1,
    "buy": 2,
    "strong_buy": 3,
    "unknown": None,
}


def _classify_rating_change(curr: str, prev: Optional[str]) -> str:
    """Classify the rating delta from broker's prior verdict to current.

    Returns one of: ``upgrade``, ``downgrade``, ``reiterate``, ``initiate``.
    "initiate" when the broker has no prior rating on file (i.e. first
    coverage). "reiterate" when the rank is identical (e.g. 买入 → 买入).
    """
    if prev is None or prev == "" or prev == "unknown":
        return "initiate"
    cr = RATING_RANK.get(curr)
    pr = RATING_RANK.get(prev)
    if cr is None or pr is None:
        return "reiterate"  # safe default — we don't know enough to call it a delta
    if cr > pr:
        return "upgrade"
    if cr < pr:
        return "downgrade"
    return "reiterate"


def _compute_eps_revision_pct(curr_eps: Optional[float],
                              prev_eps: Optional[float]) -> Optional[float]:
    """Percentage revision of the current-year EPS forecast vs. the same
    broker's previous report on the same name. None when either side is
    missing or the prior is zero (cannot divide).
    """
    if curr_eps is None or prev_eps is None:
        return None
    try:
        if abs(prev_eps) < 1e-9:
            return None
        return float((curr_eps - prev_eps) / abs(prev_eps))
    except (TypeError, ValueError):
        return None


SYSTEM_PROMPT_RESEARCH_RATING = """你是A股卖方研报分析助手。给定一份研报的标题 + 元数据,
提取目标价 (人民币元) 和一句话事实摘要 (≤30字)。

只输出JSON, 禁止解释、思考、前后缀。第一个字符必须是 { 。
target_price 是数字 (元), 无法判断时输出 null。
summary 必须是中文事实陈述, 不要分析或预测。

Schema:
{"target_price": <number 元 or null>,
 "summary": "<≤30字事实摘要>",
 "confidence": <0-1, 标题清晰则0.8+, 模糊则0.3-0.5>}

只从给定文本提取明示信息, 不推断。"""


class ResearchRatingExtractor:
    """Wraps ``LLMEventExtractorV2._call_llm`` with the research-rating prompt.

    Reuses the V2 instance for rate-limiting, retry/backoff, token
    accounting — we ONLY need to swap the system prompt and post-process
    the response. The V2 ``_call_llm(prompt, system_prompt=...)`` API
    accepts a per-call prompt override (cx P1 #2 thread-safe path) so
    we never touch the V2 module-global SYSTEM_PROMPT_V2.
    """

    def __init__(self, model: str = "MiniMax-Text-01"):
        self._llm = LLMEventExtractorV2(model=model)

    def extract_single(self, report: dict, previous_report: Optional[dict] = None) -> dict | None:
        """Extract LLM-derived fields for ONE research report row.

        Args:
            report: collector record (one line of
                ``data/storage/research_reports/<date>.jsonl``).
            previous_report: same broker's most-recent prior report on
                this stock, or None if first coverage. Used to compute
                rating_change + eps_revision_pct.

        Returns the merged dict (passthrough fields + LLM-extracted
        fields) or None if the LLM call hard-failed (HTTP, parse, etc.).
        """
        title = report.get("report_title", "")
        broker = report.get("broker", "")
        rating_cn = report.get("raw_rating", "")
        prev_rating = (previous_report or {}).get("canonical_rating")
        prev_rating_str = (previous_report or {}).get("raw_rating", "无")

        prompt = (
            f"股票: {report.get('stock_code')} {report.get('stock_name')}\n"
            f"机构: {broker}\n"
            f"评级: {rating_cn}\n"
            f"上次评级 (同机构): {prev_rating_str}\n"
            f"标题: {title}"
        )

        text, usage = self._llm._call_llm(
            prompt, system_prompt=SYSTEM_PROMPT_RESEARCH_RATING,
        )
        if not usage.get("http_ok"):
            return None

        parsed = self._parse_response(text)
        if parsed is None:
            return None

        rating_change = _classify_rating_change(
            report.get("canonical_rating", "unknown"),
            prev_rating,
        )
        eps_rev_pct = _compute_eps_revision_pct(
            report.get("eps_y1"),
            (previous_report or {}).get("eps_y1"),
        )

        return {
            "stock_code": report.get("stock_code"),
            "qlib_code": str(report.get("qlib_code", "")).lower(),  # canonical case
            "report_date": report.get("report_date"),
            "broker": broker,
            "rating_current": report.get("canonical_rating"),
            "rating_previous": prev_rating,
            "rating_change": rating_change,
            "target_price": parsed.get("target_price"),
            "eps_current_year": report.get("eps_y1"),
            "eps_next_year": report.get("eps_y2"),
            "eps_revision_pct": eps_rev_pct,
            "summary_sentence": parsed.get("summary", ""),
            "confidence": parsed.get("confidence", 0.5),
            "collected_at": report.get("collected_at", ""),
            "extracted_at": datetime.now().isoformat(timespec="seconds"),
            "extractor_version": EXTRACTOR_VERSION,
        }

    @staticmethod
    def _parse_response(text: str) -> Optional[dict]:
        """Reuse the V2 robust JSON extraction (first { to last })."""
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
        # Coerce target_price to float or None.
        tp = data.get("target_price")
        if tp is not None:
            try:
                tp = float(tp)
            except (TypeError, ValueError):
                tp = None
        try:
            conf = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        return {
            "target_price": tp,
            "summary": str(data.get("summary", ""))[:60],
            "confidence": conf,
        }

    def extract_from_file(self, reports_path: Path,
                          target_date: Optional[str] = None) -> Path:
        """Process a per-day collector JSONL → extracted JSONL.

        Iterates rows, looking up the SAME-BROKER prior report from
        prior rows in the same file (sorted by report_date ascending
        per ``collect_research_reports``). Production-grade: a future
        revision will widen the lookback to the previous N days of
        per-day files (so we can detect cross-day upgrades). The spike
        scaffolds the in-file path first because the per-day JSONL
        already contains 30 days of rolling-window history per stock.
        """
        reports_path = Path(reports_path)
        if target_date is None:
            target_date = reports_path.stem
        output_path = EXTRACTED_DIR / f"{target_date}.jsonl"

        if output_path.exists():
            n = sum(1 for _ in open(output_path))
            if n > 0:
                logger.info("Research rating already extracted for %s (%d rows), skip",
                            target_date, n)
                return output_path

        # Load all rows, group by (stock_code, broker), sort ascending
        # by report_date so the "previous" lookup is just the prior row.
        rows: list[dict] = []
        with open(reports_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not rows:
            logger.warning("No research-report rows in %s", reports_path)
            output_path.write_text("")
            return output_path

        # Group by (stock, broker), ascending date.
        from collections import defaultdict
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            key = (r.get("stock_code", ""), r.get("broker", ""))
            grouped[key].append(r)
        for k in grouped:
            grouped[k].sort(key=lambda x: x.get("report_date", ""))

        # Extract. Per (stock, broker), the i-th row's "previous" is the
        # (i-1)-th row from the same group.
        n_ok = 0
        n_fail = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for key, group in grouped.items():
                prev = None
                for report in group:
                    extracted = self.extract_single(report, previous_report=prev)
                    if extracted is None:
                        n_fail += 1
                        prev = report  # still update so next "prev" is correct
                        continue
                    f.write(json.dumps(extracted, ensure_ascii=False) + "\n")
                    n_ok += 1
                    prev = report
        logger.info("Research rating extraction: %d ok, %d fail -> %s",
                    n_ok, n_fail, output_path)
        return output_path


def iter_extracted_jsonl(path: Path) -> Iterable[dict]:
    """Yield extracted records from a per-day JSONL — utility for the
    factor builder's loader.
    """
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
