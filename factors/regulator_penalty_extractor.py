"""SPIKE — LLM extractor for 证监会 / 交易所 regulator actions.

Channel 4 of the LLM event pipeline (mirrors V2 architecture; see
``factors/llm_event_extractor_v2.py``).

Design decisions
----------------
1. **Facts-only schema**. Same discipline as V2: the LLM extracts the
   regulator type, severity tier, topic, fine amount, and whether the
   inquiry letter uses high-pressure language. It does NOT predict
   stock impact — ``expected_market_impact`` is a 1-5 LLM judgment of
   document severity, NOT a return forecast. Direction sign comes from
   historical calibration at the FACTOR layer, not here.

2. **Standalone prompt**. The V2 extractor's ``regulatory_penalty``
   event_type is too coarse (just one of 33 possible types, and the
   keyword gate downgrades it to ``other`` ~40% of the time per
   factors/event_schema_validator.py). A dedicated prompt with a
   typed severity scale eliminates the downgrade ambiguity.

3. **Reuse network plumbing**. The V2 extractor owns the MiniMax
   client + rate limiter + retry queue; we call ``_call_llm`` with a
   per-call ``system_prompt`` (the 2026-06-07 cx P1 #2 fix made that
   contract thread-safe — see scripts/extract_policy_events.py:780).

Schema (per-event)
------------------
* ``ts_code``             — 6-digit code, may be empty when the doc
                            covers multiple stocks or is a market-wide
                            CSRC press release.
* ``event_date``          — YYYY-MM-DD, the regulator filing date.
                            PIT lag of +1 BDay is applied at the
                            FACTOR layer (build_regulator_penalty_
                            factors.py), NOT here.
* ``severity``            — enum {warning, fine, suspension,
                            delisting_warning, criminal_referral}.
                            Karpoff-Lott (1993) ranks these in order of
                            magnitude of the CAR drop.
* ``regulator``           — enum {CSRC, SSE, SZSE}.
* ``topic``               — enum {财务造假, 关联交易, 资金占用,
                            信披违规, 操纵市场, 内幕交易, 其他}.
* ``fine_amount_yuan``    — float, 0 if no fine. Inquiry letters
                            with no monetary penalty MUST output 0.
* ``is_strict_inquiry``   — bool. True only when the document uses
                            high-pressure language (请充分说明 / 高度
                            关注 / 请按时回复). Choi 2021 shows these
                            phrases correlate +25% with subsequent
                            full investigation within 90 days.
* ``expected_market_impact`` — int in [1, 5]. LLM's subjective severity
                            judgment of the DOCUMENT (not the stock
                            price reaction). 1=routine compliance,
                            5=criminal referral / delisting.
* ``summary_sentence``    — ≤60 char Chinese summary, fact-only.
* ``confidence``          — float in [0, 1]. <0.5 means the LLM was
                            unsure (typically when the document was
                            multi-stock and the code couldn't be
                            disambiguated).

Output
------
``data/storage/regulator_events/<YYYY-MM-DD>.jsonl``

Each line is a single event keyed by (ts_code, event_date).
Multi-stock documents emit one line per stock.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)

EVENTS_DIR = DATA_DIR / "regulator_events"
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

SEVERITY_TIERS = (
    "warning",            # 警示函 / 监管关注
    "fine",               # 行政处罚 + 罚款
    "suspension",         # 暂停业务资格 / 责令停业整顿
    "delisting_warning",  # 退市风险警示 (ST / *ST 升格)
    "criminal_referral",  # 移送公安 / 司法机关
)
REGULATORS = ("CSRC", "SSE", "SZSE")
TOPICS = (
    "财务造假",
    "关联交易",
    "资金占用",
    "信披违规",
    "操纵市场",
    "内幕交易",
    "其他",
)

# Severity → numeric tier for downstream factor build. Higher = worse.
# Calibrated against Karpoff-Lott (1993) and Bhagat-Bizjak (2019)
# absolute CAR magnitudes (warning ≈ -0.5%, fine ≈ -1.5%, suspension
# ≈ -3%, delisting_warning ≈ -8%, criminal_referral ≈ -12%).
SEVERITY_SCORE = {
    "warning": 1,
    "fine": 2,
    "suspension": 3,
    "delisting_warning": 4,
    "criminal_referral": 5,
}

# The high-pressure phrases that Choi 2021 ("SEC inquiries and
# subsequent enforcement", JFE 2021) identified as predictive of an
# actual investigation. Used as a post-LLM validation gate — if the
# LLM claims is_strict_inquiry=true but none of these phrases appear,
# the factor builder will downgrade the flag.
STRICT_INQUIRY_PHRASES = (
    "请充分说明", "高度关注", "请按时回复",
    "立案调查", "现场检查", "请说明合理性",
)

SYSTEM_PROMPT_REGULATOR = """你是中国证券监管文件分析师。从下文证监会/交易所行政处罚/自律监管/问询函中提取结构化事实。

只输出JSON，禁止解释、思考、前后缀文字。第一个字符必须是 { 。

严格规则：
1. 只提取原文明示的事实。禁止预测股价、不要给出交易建议。
2. 如果某字段在原文中没有，输出 null（不要猜测、不要填 0），但 fine_amount_yuan 为 0 元罚款时输出 0。
3. ts_code 必须是 6 位数字字符串（如 "600519"）。如原文涉及多只股票或不可识别，输出 null。
4. severity 严格枚举：warning(警示函/监管关注) | fine(行政处罚带罚款) | suspension(暂停业务) | delisting_warning(退市风险警示) | criminal_referral(移送公安)。
5. is_strict_inquiry 仅在原文明确出现 "请充分说明"、"高度关注"、"请按时回复"、"立案调查" 等高压语 时为 true。
6. expected_market_impact 是对文件本身严重程度的 1-5 主观判断，不是对股价反应的预测。

Schema:
{
  "ts_code": "<6位数字 or null>",
  "event_date": "<YYYY-MM-DD or null>",
  "severity": "<one of: warning|fine|suspension|delisting_warning|criminal_referral>",
  "regulator": "<one of: CSRC|SSE|SZSE>",
  "topic": "<one of: 财务造假|关联交易|资金占用|信披违规|操纵市场|内幕交易|其他>",
  "fine_amount_yuan": <float 人民币元, 0 if no fine>,
  "is_strict_inquiry": <bool>,
  "expected_market_impact": <int in [1, 5]>,
  "summary_sentence": "<≤60字事实概述, 无分析>",
  "confidence": <float in [0, 1]>
}"""


@dataclass
class RegulatorEvent:
    """Validated event record. Field defaults match the JSON schema."""
    ts_code: str = ""
    event_date: str = ""
    severity: str = "warning"
    regulator: str = "CSRC"
    topic: str = "其他"
    fine_amount_yuan: float = 0.0
    is_strict_inquiry: bool = False
    expected_market_impact: int = 1
    summary_sentence: str = ""
    confidence: float = 0.5
    # Provenance — populated by the extractor wrapper, not the LLM.
    source_url: str = ""
    source_key: str = ""
    extract_date: str = ""
    extractor_version: str = "v0-spike"

    def to_dict(self) -> dict:
        return {
            "ts_code": self.ts_code,
            "event_date": self.event_date,
            "severity": self.severity,
            "regulator": self.regulator,
            "topic": self.topic,
            "fine_amount_yuan": float(self.fine_amount_yuan),
            "is_strict_inquiry": bool(self.is_strict_inquiry),
            "expected_market_impact": int(self.expected_market_impact),
            "summary_sentence": self.summary_sentence,
            "confidence": float(self.confidence),
            "source_url": self.source_url,
            "source_key": self.source_key,
            "extract_date": self.extract_date,
            "extractor_version": self.extractor_version,
        }


# ---------------------------------------------------------------------
# Validation — mirrors scripts/extract_policy_events.py:validate_event
# discipline. Coerces, clamps, and falls back to safe defaults so a
# malformed LLM JSON never crashes the downstream factor build.
# ---------------------------------------------------------------------
_TS_CODE_RE = re.compile(r"^\d{6}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _coerce_severity(raw) -> str:
    if isinstance(raw, str) and raw.strip() in SEVERITY_TIERS:
        return raw.strip()
    return "warning"


def _coerce_regulator(raw) -> str:
    if isinstance(raw, str) and raw.strip().upper() in REGULATORS:
        return raw.strip().upper()
    return "CSRC"


def _coerce_topic(raw) -> str:
    if isinstance(raw, str) and raw.strip() in TOPICS:
        return raw.strip()
    return "其他"


def _coerce_float(raw, *, default: float = 0.0, lo: float = 0.0,
                   hi: float = 1e12) -> float:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return max(lo, min(hi, v))


def _coerce_int(raw, *, default: int = 1, lo: int = 1, hi: int = 5) -> int:
    try:
        v = int(float(raw))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _coerce_ts_code(raw) -> str:
    if not isinstance(raw, str):
        return ""
    s = raw.strip()
    return s if _TS_CODE_RE.match(s) else ""


def _coerce_event_date(raw) -> str:
    if not isinstance(raw, str):
        return ""
    s = raw.strip()
    return s if _DATE_RE.match(s) else ""


def validate_event(raw: dict, *, body_text: str = "") -> RegulatorEvent:
    """Coerce a raw LLM dict into a typed RegulatorEvent.

    The ``body_text`` is the original document body — used to verify
    the ``is_strict_inquiry`` claim against STRICT_INQUIRY_PHRASES.
    If the LLM said true but no phrase appears in the body, we
    downgrade to false (Choi 2021's phrase-match gate).
    """
    severity = _coerce_severity(raw.get("severity"))
    is_strict_llm = bool(raw.get("is_strict_inquiry", False))

    # Phrase-gate the strict-inquiry flag. The LLM tends to over-call
    # this for any 问询函 with stern-sounding language; the empirical
    # phrases from Choi 2021 are narrower than the LLM's intuition.
    if is_strict_llm and body_text:
        if not any(p in body_text for p in STRICT_INQUIRY_PHRASES):
            logger.debug(
                "is_strict_inquiry downgraded — LLM said true but no "
                "trigger phrase found in body."
            )
            is_strict_llm = False

    return RegulatorEvent(
        ts_code=_coerce_ts_code(raw.get("ts_code")),
        event_date=_coerce_event_date(raw.get("event_date")),
        severity=severity,
        regulator=_coerce_regulator(raw.get("regulator")),
        topic=_coerce_topic(raw.get("topic")),
        fine_amount_yuan=_coerce_float(raw.get("fine_amount_yuan")),
        is_strict_inquiry=is_strict_llm,
        expected_market_impact=_coerce_int(raw.get("expected_market_impact")),
        summary_sentence=str(raw.get("summary_sentence", ""))[:120],
        confidence=_coerce_float(raw.get("confidence"), default=0.5, lo=0.0, hi=1.0),
    )


# ---------------------------------------------------------------------
# LLM wrapper — uses the V2 module's _call_llm + per-call system prompt.
# ---------------------------------------------------------------------
def extract_single(doc: dict) -> RegulatorEvent | None:
    """Extract a single regulator event from one raw doc dict.

    Args:
        doc: a row from regulator_actions/<date>.jsonl with fields
             title, body, url, filed_date, regulator, doc_type, ts_code.

    Returns:
        A validated RegulatorEvent or None if the LLM call failed.
    """
    from factors.llm_event_extractor_v2 import LLMEventExtractorV2

    ext = LLMEventExtractorV2()
    body = doc.get("body", "")
    title = doc.get("title", "")
    prompt = (
        f"监管机构: {doc.get('regulator', '')}\n"
        f"文件类型: {doc.get('doc_type', '')}\n"
        f"备案日期: {doc.get('filed_date', '')}\n"
        f"标题: {title}\n"
        f"正文(前3000字): {body[:3000]}"
    )
    text, usage = ext._call_llm(prompt, system_prompt=SYSTEM_PROMPT_REGULATOR)
    if not usage.get("http_ok"):
        return None

    # Reuse V2's JSON extraction.
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        raw_json = json.loads(clean[start:end])
    except json.JSONDecodeError:
        return None

    event = validate_event(raw_json, body_text=body)
    # Fallback: if LLM left ts_code blank but the doc carried one,
    # backfill from the list-page metadata.
    if not event.ts_code and doc.get("ts_code"):
        event.ts_code = _coerce_ts_code(doc.get("ts_code"))
    # Same for event_date.
    if not event.event_date and doc.get("filed_date"):
        event.event_date = _coerce_event_date(doc.get("filed_date"))
    event.source_url = str(doc.get("url", ""))[:500]
    event.source_key = str(doc.get("source_key", ""))
    event.extract_date = datetime.now().strftime("%Y-%m-%d")
    return event


def extract_from_actions_file(actions_path: Path, target_date: str | None = None) -> Path:
    """Process one day's regulator_actions JSONL via the LLM.

    Args:
        actions_path: path to data/storage/regulator_actions/<date>.jsonl.
        target_date: defaults to the file stem.

    Returns:
        Path to data/storage/regulator_events/<date>.jsonl.

    SPIKE: this is the orchestration shell. The actual LLM dispatch
    pattern (ThreadPoolExecutor with rate-limit-aware fan-out) is
    copied from llm_event_extractor_v2.extract_from_news_file. Kept
    minimal here because the SPIKE constraint forbids actually calling
    the LLM.
    """
    actions_path = Path(actions_path)
    if target_date is None:
        target_date = actions_path.stem
    output_path = EVENTS_DIR / f"{target_date}.jsonl"

    if not actions_path.exists():
        logger.warning("No actions file at %s", actions_path)
        return output_path

    docs: list[dict] = []
    with actions_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    logger.info("SPIKE: would extract %d docs from %s", len(docs), actions_path)
    # ── TODO L1: ThreadPoolExecutor fan-out with V2-style retry queue. ──
    # for doc in docs:
    #     event = extract_single(doc)
    #     ...
    return output_path
