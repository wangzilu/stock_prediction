"""Extract structured policy events from PBC + State Council + NBS +
Xinwen Lianbo texts —
Phase E.1 step 2 (PBC) + Phase E.2 step 2 (State Council / ministry) +
Phase E.3 step 2 (NBS macro surprise) + Phase E.4 step 2 (Xinwen Lianbo
theme attention).

PE-1 (PBC) reads ``data/storage/policy_texts/pbc/<YYYY-MM-DD>.jsonl``
and writes ``data/storage/policy_events/pbc/<YYYY-MM-DD>.jsonl`` —
extracts monetary-policy stance / liquidity injection / rate change.

PE-2 (State Council) reads
``data/storage/policy_texts/state_council/<YYYY-MM-DD>.jsonl`` and
writes ``data/storage/policy_events/state_council/<YYYY-MM-DD>.jsonl``
— extracts target industries / fiscal support / regulatory direction.

PE-3 (NBS) reads
``data/storage/policy_texts/nbs/<YYYY-MM-DD>.jsonl`` and writes
``data/storage/policy_events/nbs/<YYYY-MM-DD>.jsonl`` — extracts
series name (CPI / PPI / PMI / retail sales) / headline value /
consensus / mom & yoy change / surprise direction.

The same events are also appended to the unified ``EventStore`` so
downstream factor builders / overlays can query them by ``signal_date``.

Design principles
-----------------
- **Facts, not predictions.** The LLM extracts what the PBOC text
  literally says — net injection in 亿元, repo-rate change in bp, tool
  type, etc. It does NOT predict stock returns, and the prompt makes
  this explicit. This is the L1 lesson from the 2026-06-06 LLM critique.
- **Validator downgrades, never drops.** When the LLM emits an enum
  outside the documented set (e.g. ``tool_type="supercannon"``), the
  validator demotes to ``"other"``/``"unknown"`` and keeps the row.
  Dropping silently would make the n_extracted count lie about how
  many texts actually went through the LLM.
- **Idempotent.** Same date re-run overwrites atomically via ``.tmp``
  + ``replace``. Re-running an EventStore append also dedups via the
  ``_hash`` content key built into ``EventStore.add_event``.
- **Fail loud.** If the LLM raises on every row (e.g. API key missing
  at the helper layer), ``summary["n_failed"]`` records it. The CLI
  refuses to write health=success when zero rows extracted.

Usage
-----
    # single-day (production cron mode)
    python scripts/extract_policy_events.py --source pbc --date 2026-06-05

    # backfill window
    python scripts/extract_policy_events.py --source pbc \\
        --start 2026-04-01 --end 2026-06-05

Output schema (one JSON object per line)
----------------------------------------
    {
      "publish_date":               "YYYY-MM-DD",
      "policy_type":                "omo|lpr|mlf|rrr|quarterly_report|...",
      "title":                      str,
      "url":                        str,
      "policy_stance":              "easing|tightening|neutral|unknown",
      "liquidity_injection_amount": float | null,  # 亿元
      "net_injection":              float | null,  # 亿元
      "repo_rate_change":           int   | null,  # basis points
      "tool_type":                  "omo|mlf|slf|rrr|lpr|"
                                     "quarterly_report|press_conference|other",
      "duration_days":              int   | null,
      "unexpectedness":             float | null,  # in [0, 1]
      "extracted_at":               "YYYY-MM-DDTHH:MM:SSZ"
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)

# ── Storage layout ───────────────────────────────────────────────────
INPUT_DIR = DATA_DIR / "policy_texts" / "pbc"
OUTPUT_DIR = DATA_DIR / "policy_events" / "pbc"

# Phase E.2 (PE-2) — State Council + ministry chain.
INPUT_DIR_SC = DATA_DIR / "policy_texts" / "state_council"
OUTPUT_DIR_SC = DATA_DIR / "policy_events" / "state_council"

# Phase E.3 (PE-3) — NBS macro surprise chain.
INPUT_DIR_NBS = DATA_DIR / "policy_texts" / "nbs"
OUTPUT_DIR_NBS = DATA_DIR / "policy_events" / "nbs"

# Phase E.4 (PE-4) — Xinwen Lianbo (新闻联播) theme attention chain.
INPUT_DIR_XWLB = DATA_DIR / "policy_texts" / "xinwen_lianbo"
OUTPUT_DIR_XWLB = DATA_DIR / "policy_events" / "xinwen_lianbo"

# Health source name matches the convention used by collect_policy_texts.
HEALTH_SOURCE_NAME = "pbc_policy_events"
HEALTH_SOURCE_NAME_SC = "state_council_policy_events"
HEALTH_SOURCE_NAME_NBS = "nbs_policy_events"
HEALTH_SOURCE_NAME_XWLB = "xinwen_lianbo_policy_events"

# ─────────────────────────────────────────────────────────────────────
# Schema enumerations — the validator gate.
#
# The LLM is asked to choose from these closed sets; mismatches downgrade
# to the fallback rather than dropping the row. This mirrors the L3
# pattern in ``factors/event_schema_validator.py`` (post-LLM keyword
# gate that demotes earnings_* → routine_announcement on miss).
# ─────────────────────────────────────────────────────────────────────
POLICY_STANCES: frozenset[str] = frozenset({
    "easing", "tightening", "neutral", "unknown",
})
TOOL_TYPES: frozenset[str] = frozenset({
    "omo", "mlf", "slf", "rrr", "lpr",
    "quarterly_report", "press_conference", "other",
})

# Phase E.2 (PE-2) — State Council / ministry industry policy enums.
# Downgrade rules: out-of-vocab policy_direction → "neutral",
# out-of-vocab subsidy_or_tax → "neither". target_industries is a free
# list (we do NOT lock the vocabulary because the industry taxonomy
# evolves; the downstream mapper is the gate, not this validator).
POLICY_DIRECTIONS: frozenset[str] = frozenset({
    "supportive", "restrictive", "neutral",
})
SUBSIDY_TAX_TYPES: frozenset[str] = frozenset({
    "subsidy", "tax_reduction", "tax_increase", "neither",
})

# Phase E.3 (PE-3) — NBS macro surprise enums. The LLM extracts which
# series the article covers and the direction of the surprise vs
# consensus. Downgrade rules: out-of-vocab series_name → "other",
# out-of-vocab surprise_direction → "unknown".
NBS_SERIES_NAMES: frozenset[str] = frozenset({
    "cpi", "ppi", "pmi", "retail_sales", "other",
})
NBS_SURPRISE_DIRECTIONS: frozenset[str] = frozenset({
    "upside", "downside", "inline", "unknown",
})

# Phase E.4 (PE-4) — Xinwen Lianbo theme attention. The phase plan
# lists 9 canonical themes; we accept any lowercase_with_underscores
# string from the LLM (themes evolve faster than we can re-prompt) but
# anchor a "known set" so downgrades stay measurable. Themes outside
# this set are KEPT (free-form vocab), just flagged via
# ``n_unknown_themes`` for an operator review.
XINWEN_LIANBO_KNOWN_THEMES: frozenset[str] = frozenset({
    "semiconductor_self_reliance",
    "domestic_consumption",
    "real_estate",
    "private_economy",
    "capital_markets",
    "robotics_ai",
    "renewable_energy",
    "military_security",
    "belt_and_road",
    "rural_revitalization",
    "carbon_neutrality",
})

# Fact-only system prompt. Explicitly bans price prediction and trading
# advice. The "if a field is absent, output null" line is required by
# spec to prevent the LLM from confabulating numbers it didn't see.
SYSTEM_PROMPT_PBC = """你是央行货币政策分析师。从下文人民银行公告/新闻稿中提取结构化事实。

只输出JSON，禁止解释、思考、前后缀文字。第一个字符必须是 { 。

严格规则：
1. 只提取原文明示的事实，不要推断股票或市场走向，不要给出交易建议。
2. 如果某字段在原文中没有，输出 null（不要猜测、不要填 0）。
3. 数值字段统一使用整数或浮点数，不要带单位。

Schema:
{
  "policy_stance": "<one of: easing|tightening|neutral|unknown>",
  "liquidity_injection_amount": <float 亿元, positive=投放, negative=回笼, null if absent>,
  "net_injection": <float 亿元 after netting out maturity, null if absent>,
  "repo_rate_change": <int basis points, positive=hike, negative=cut, null if no rate change>,
  "tool_type": "<one of: omo|mlf|slf|rrr|lpr|quarterly_report|press_conference|other>",
  "duration_days": <int e.g. 7 for 7-day reverse repo, 365 for 1-year MLF, null if N/A>,
  "unexpectedness": <float in [0, 1]: 0=announced beforehand or routine; 1=surprise>
}"""


# Phase E.2 (PE-2) State Council / ministry industry-policy prompt.
# Fact-only. The "DO NOT predict stock impact" line is explicit because
# the LLM is being asked for a subjective ``policy_strength`` score and
# would otherwise drift into "this benefits semi stocks +5%" hallucination.
SYSTEM_PROMPT_SC = """你是中国产业政策分析师。从下文国务院/部委政策文件中提取结构化事实。

只输出JSON，禁止解释、思考、前后缀文字。第一个字符必须是 { 。

严格规则：
1. 只提取原文明示的事实。禁止预测股价、不要给出交易建议、不要推断市场反应。
2. 如果某字段在原文中没有，输出 null（不要猜测、不要填 0）。
3. policy_strength 是文件本身力度的主观判断（资金量、覆盖面、强制度），不是股票影响预测。
4. target_industries 使用英文小写下划线名（semiconductor / renewable_energy / real_estate / 等）。

Schema:
{
  "target_industries": [<list of industry names>],
  "policy_direction": "<one of: supportive|restrictive|neutral>",
  "policy_strength": <float in [0, 1]: 0=泛泛表态; 1=明确大额、强制、即时>,
  "fiscal_support": <float 亿元 of central/local fiscal money pledged, null if absent>,
  "subsidy_or_tax": "<one of: subsidy|tax_reduction|tax_increase|neither>",
  "regulatory_tightening": <bool: true 仅当原文明确加强监管/限制/禁止/处罚>,
  "implementation_deadline": "<YYYY-MM-DD or null>"
}"""


# Phase E.3 (PE-3) NBS macro statistics prompt. Fact-only: extract the
# headline number / prior / consensus / surprise vs expectations. The
# "DO NOT predict stock direction" line is critical because macro
# releases are reflexively associated with market reactions in training
# data — without the guard the LLM would output "CPI undershoots →
# stocks rally" which is exactly the prediction we forbid.
SYSTEM_PROMPT_NBS = """You are a Chinese macro-statistics analyst. Extract
structured facts from the National Bureau of Statistics (NBS) release
text below.

Output ONLY one JSON object. No prose, no explanation, no preamble.
The first character must be { .

Strict rules:
1. Extract ONLY facts literally stated in the text. DO NOT predict
   stock prices, DO NOT give trading advice, DO NOT infer market
   reactions.
2. If a field is absent or not stated in the text, output null. Do not
   guess. Do not fill 0.
3. Numeric fields are plain numbers (no units, no % sign, no 亿元).
4. release_period is the YYYY-MM of the DATA POINT (e.g. "April CPI
   released in May" => release_period="2026-04"), NOT the release date.
5. surprise_direction is "upside" / "downside" / "inline" only when
   the text quotes a consensus or expectation; otherwise "unknown".

Schema:
{
  "series_name": "<one of: cpi|ppi|pmi|retail_sales|other>",
  "release_period": "<YYYY-MM of the data point, or null>",
  "headline_value": <float, the headline number reported, or null>,
  "prior_value": <float, the previous month's value if quoted, or null>,
  "consensus_value": <float, analyst consensus / market expectation if quoted, or null>,
  "mom_change": <float, month-on-month change in percentage points / %, or null>,
  "yoy_change": <float, year-on-year change in percentage points / %, or null>,
  "surprise_direction": "<one of: upside|downside|inline|unknown>"
}"""


# Phase E.4 (PE-4) Xinwen Lianbo theme attention prompt. The LLM scores
# the MEDIA ATTENTION pattern of the daily broadcast — what themes were
# covered, how many news items touched each, where the broadcast led.
# It is critical that the prompt explicitly forbids price prediction:
# theme strength is a fact about state-media attention, NOT a stock
# direction forecast. This is the L1 "facts not predictions" lesson from
# the 2026-06-06 LLM critique; without it the LLM cheerfully says
# "carbon_neutrality strong → buy renewable stocks" which is exactly
# what we forbid.
SYSTEM_PROMPT_XINWEN_LIANBO = """You are a Chinese state-media attention
analyst. Extract structured facts about which themes the CCTV 新闻联播
broadcast covered today.

Output ONLY one JSON object. No prose, no explanation, no preamble.
The first character must be { .

Strict rules:
1. Do NOT predict stock direction. Theme strength is a FACT about
   media attention, NOT a price prediction. Do NOT give trading
   advice. Do NOT infer market reactions.
2. Extract ONLY themes literally covered in the transcript. Do not
   guess or add themes that "should" be present.
3. Use lowercase_with_underscores for theme names. Prefer the
   canonical set when applicable:
     semiconductor_self_reliance / domestic_consumption / real_estate /
     private_economy / capital_markets / robotics_ai / renewable_energy /
     military_security / belt_and_road / rural_revitalization /
     carbon_neutrality.
   Themes outside this list ARE allowed — coin a new lowercase token
   if the transcript covers something not in the canonical set.
4. theme_mention_counts must be an integer count of DISTINCT news
   items in the broadcast that touched the theme. 0 means the theme
   was not mentioned and should NOT appear in either list.
5. policy_priority_signal is a float in [0, 1]. 0 = filler / sports /
   weather only. 1 = the lead story occupied 5+ minutes and the
   anchor explicitly tied it to top-level political priority.
6. If a field is genuinely absent from the broadcast, output null /
   empty list. Do not invent.

Schema:
{
  "themes": [<lowercase_with_underscores theme tokens>],
  "theme_mention_counts": {<theme>: <int count>, ...},
  "policy_priority_signal": <float in [0, 1]>,
  "regions_mentioned": [<province / region names if a regional initiative was highlighted, lowercased>]
}"""


def _now_utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _date_range(start: str, end: str) -> list[str]:
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    if e < s:
        raise ValueError(f"--end ({end}) must be >= --start ({start})")
    out: list[str] = []
    cur = s
    while cur <= e:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def _atomic_write_jsonl(rows: list[dict], path: Path) -> None:
    """Write rows to ``path`` atomically (``.tmp`` + ``replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _coerce_number(value: Any, *, as_int: bool = False) -> int | float | None:
    """Coerce ``value`` to int or float. Return None if not parseable.

    Accepts ``None`` / empty string / "null" → None. Strips trailing
    "亿元", "bp", whitespace. Bool is intentionally NOT accepted as a
    number — passing ``True`` returns None so the LLM cannot pollute
    numeric fields with booleans.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if as_int else float(value)
    s = str(value).strip()
    if not s or s.lower() in ("null", "none", "nan", "n/a"):
        return None
    # Strip common Chinese unit suffixes the LLM tends to add despite
    # the schema. "1500亿元" → "1500", "-10bp" → "-10".
    s = re.sub(r"(亿元|亿|bp|BP|个百分点|%)", "", s).strip()
    try:
        if as_int:
            return int(float(s))
        return float(s)
    except (ValueError, TypeError):
        return None


def _coerce_unexpectedness(value: Any) -> float | None:
    """Coerce + clamp to [0, 1]. Out-of-range -> clip; non-numeric -> None."""
    n = _coerce_number(value)
    if n is None:
        return None
    return float(max(0.0, min(1.0, n)))


def validate_event(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate LLM output against the schema, downgrading on miss.

    Mirrors the spec rule: when the LLM emits an out-of-vocab enum
    (e.g. ``tool_type="supercannon"``), downgrade to ``other``/
    ``unknown`` rather than drop the row. Missing numeric fields are
    represented as ``None`` so downstream factor builders can safely
    ``fillna``.

    Returns a fresh dict with every documented field present (so
    consumers don't have to ``.get(...)`` with defaults).
    """
    stance = raw.get("policy_stance")
    if not isinstance(stance, str) or stance.lower() not in POLICY_STANCES:
        stance = "unknown"
    else:
        stance = stance.lower()

    tool = raw.get("tool_type")
    if not isinstance(tool, str) or tool.lower() not in TOOL_TYPES:
        tool = "other"
    else:
        tool = tool.lower()

    return {
        "policy_stance": stance,
        "liquidity_injection_amount": _coerce_number(
            raw.get("liquidity_injection_amount")
        ),
        "net_injection": _coerce_number(raw.get("net_injection")),
        "repo_rate_change": _coerce_number(
            raw.get("repo_rate_change"), as_int=True,
        ),
        "tool_type": tool,
        "duration_days": _coerce_number(
            raw.get("duration_days"), as_int=True,
        ),
        "unexpectedness": _coerce_unexpectedness(raw.get("unexpectedness")),
    }


# ─────────────────────────────────────────────────────────────────────
# Phase E.2 (PE-2) — State Council validator. Same downgrade discipline
# as PBC: bad enums demote, bad numerics → None, never drop a row.
# ─────────────────────────────────────────────────────────────────────
_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")


def _coerce_industries(value: Any) -> list[str]:
    """Coerce ``target_industries`` to a list of lowercased strings.

    Accepts list / single string / None. Empty / non-list inputs become
    an empty list rather than failing the row.
    """
    if value is None:
        return []
    if isinstance(value, str):
        # Sometimes the LLM returns a comma-joined string instead of a list.
        parts = [p.strip() for p in re.split(r"[,，;；/、]", value) if p.strip()]
        return [p.lower() for p in parts]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            s = item.strip().lower()
            if s:
                out.append(s)
        return out
    return []


def _coerce_iso_date(value: Any) -> str | None:
    """Normalize a YYYY-MM-DD-ish string. None / unparseable → None."""
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or s.lower() in ("null", "none", "n/a"):
        return None
    m = _DATE_RE.match(s)
    if not m:
        return None
    try:
        y, mo, d = (int(g) for g in m.groups())
        # quick sanity bounds; don't bother with calendar correctness.
        if not (1900 < y < 2200 and 1 <= mo <= 12 and 1 <= d <= 31):
            return None
        return f"{y:04d}-{mo:02d}-{d:02d}"
    except ValueError:
        return None


def _coerce_bool(value: Any) -> bool:
    """Coerce to bool. None / unparseable → False (the conservative
    side: don't claim regulatory tightening if the LLM was silent)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in ("true", "1", "yes", "y", "是")


def validate_event_state_council(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate State Council LLM output, downgrading on out-of-vocab.

    Mirrors ``validate_event`` discipline for PE-1: missing → None,
    bad enums demote rather than drop. Returns a fresh dict with every
    documented field present.
    """
    direction = raw.get("policy_direction")
    if (
        not isinstance(direction, str)
        or direction.lower() not in POLICY_DIRECTIONS
    ):
        direction = "neutral"
    else:
        direction = direction.lower()

    subsidy = raw.get("subsidy_or_tax")
    if (
        not isinstance(subsidy, str)
        or subsidy.lower() not in SUBSIDY_TAX_TYPES
    ):
        subsidy = "neither"
    else:
        subsidy = subsidy.lower()

    # policy_strength uses the same [0,1] clamp as unexpectedness.
    strength = _coerce_unexpectedness(raw.get("policy_strength"))

    return {
        "target_industries": _coerce_industries(raw.get("target_industries")),
        "policy_direction": direction,
        "policy_strength": strength,
        "fiscal_support": _coerce_number(raw.get("fiscal_support")),
        "subsidy_or_tax": subsidy,
        "regulatory_tightening": _coerce_bool(raw.get("regulatory_tightening")),
        "implementation_deadline": _coerce_iso_date(
            raw.get("implementation_deadline")
        ),
    }


# ─────────────────────────────────────────────────────────────────────
# Phase E.3 (PE-3) — NBS macro-surprise validator. Same downgrade
# discipline as PBC / PE-2: bad enums demote, bad numerics → None,
# never drop a row.
# ─────────────────────────────────────────────────────────────────────
_PERIOD_RE = re.compile(r"^(\d{4})-(\d{1,2})$")


def _coerce_period(value: Any) -> str | None:
    """Normalize a YYYY-MM-ish release period. None / unparseable → None.

    Tolerates trailing day component (``YYYY-MM-DD`` → ``YYYY-MM``) so a
    LLM that confuses publish-date with release-period yields something
    rather than null.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or s.lower() in ("null", "none", "n/a"):
        return None
    # Accept YYYY-MM-DD by truncating to YYYY-MM.
    if _DATE_RE.match(s):
        s = s[:7]
    m = _PERIOD_RE.match(s)
    if not m:
        return None
    try:
        y, mo = (int(g) for g in m.groups())
        if not (1900 < y < 2200 and 1 <= mo <= 12):
            return None
        return f"{y:04d}-{mo:02d}"
    except ValueError:
        return None


def validate_event_nbs(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate NBS LLM output, downgrading on out-of-vocab enums.

    Mirrors ``validate_event`` / ``validate_event_state_council``
    discipline: missing → None, bad enums demote rather than drop the
    row. Returns a fresh dict with every documented field present.
    """
    series = raw.get("series_name")
    if (
        not isinstance(series, str)
        or series.lower() not in NBS_SERIES_NAMES
    ):
        series = "other"
    else:
        series = series.lower()

    direction = raw.get("surprise_direction")
    if (
        not isinstance(direction, str)
        or direction.lower() not in NBS_SURPRISE_DIRECTIONS
    ):
        direction = "unknown"
    else:
        direction = direction.lower()

    return {
        "series_name": series,
        "release_period": _coerce_period(raw.get("release_period")),
        "headline_value": _coerce_number(raw.get("headline_value")),
        "prior_value": _coerce_number(raw.get("prior_value")),
        "consensus_value": _coerce_number(raw.get("consensus_value")),
        "mom_change": _coerce_number(raw.get("mom_change")),
        "yoy_change": _coerce_number(raw.get("yoy_change")),
        "surprise_direction": direction,
    }


# ─────────────────────────────────────────────────────────────────────
# Phase E.4 (PE-4) — Xinwen Lianbo theme attention validator.
#
# Free-form theme vocabulary: unlike NBS series_name which has a closed
# 5-element enum, XWLB themes are a moving target. The validator keeps
# whatever the LLM emits (lowercased, underscored, non-empty) but
# additionally counts how many themes fell outside the documented
# XINWEN_LIANBO_KNOWN_THEMES set so an operator can spot taxonomy
# drift in the summary.
# ─────────────────────────────────────────────────────────────────────
def _coerce_theme_token(value: Any) -> str | None:
    """Coerce a single value into a lowercase_with_underscores theme.

    Strips whitespace, lowercases, replaces internal spaces / dashes
    with underscores, collapses repeated underscores. Returns None for
    None / non-string / empty after coercion.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    if not s or s in ("null", "none", "n/a"):
        return None
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or None


def _coerce_theme_list(value: Any) -> list[str]:
    """Coerce ``themes`` to a list of theme tokens.

    Accepts list / single string / None. Returns a stable de-duplicated
    list (preserves first-seen order). Empty / non-list inputs become
    an empty list rather than failing the row.
    """
    if value is None:
        return []
    items: list[Any]
    if isinstance(value, str):
        items = [p for p in re.split(r"[,，;；/、]", value) if p.strip()]
    elif isinstance(value, list):
        items = list(value)
    else:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        tok = _coerce_theme_token(it)
        if not tok or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _coerce_mention_counts(value: Any, *, valid_themes: set[str]) -> dict[str, int]:
    """Coerce theme_mention_counts dict; drop entries whose theme is
    not in ``valid_themes`` (so the count list cannot contradict the
    themes list — keeps downstream tabulation consistent).
    """
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in value.items():
        tok = _coerce_theme_token(k)
        if not tok or tok not in valid_themes:
            continue
        n = _coerce_number(v, as_int=True)
        if n is None or n < 0:
            continue
        out[tok] = int(n)
    return out


def _coerce_region_list(value: Any) -> list[str]:
    """Coerce ``regions_mentioned`` to a list of lowercase region names.

    Identical discipline as theme list but does NOT collapse hyphens /
    spaces (region names tend to be 2-3 char Chinese tokens).
    """
    if value is None:
        return []
    items: list[Any]
    if isinstance(value, str):
        items = [p for p in re.split(r"[,，;；/、]", value) if p.strip()]
    elif isinstance(value, list):
        items = list(value)
    else:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if not isinstance(it, str):
            continue
        s = it.strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def validate_event_xinwen_lianbo(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate XWLB LLM output, downgrading on miss; never drop a row.

    Mirrors validate_event / validate_event_state_council / validate_event_nbs
    discipline. Returns a fresh dict with every documented field present.
    Numeric fields outside [0,1] clamp; missing → 0.0 for the priority
    signal (so the factor build does not have to .fillna).
    """
    themes = _coerce_theme_list(raw.get("themes"))
    valid_set = set(themes)
    counts = _coerce_mention_counts(
        raw.get("theme_mention_counts"), valid_themes=valid_set,
    )
    # Make sure every theme has a count (default 1 if missing).
    for t in themes:
        counts.setdefault(t, 1)
    priority = _coerce_unexpectedness(raw.get("policy_priority_signal"))
    if priority is None:
        priority = 0.0
    regions = _coerce_region_list(raw.get("regions_mentioned"))
    return {
        "themes": themes,
        "theme_mention_counts": counts,
        "policy_priority_signal": float(priority),
        "regions_mentioned": regions,
    }


# ─────────────────────────────────────────────────────────────────────
# LLM wrapper — uses the same MiniMax client wired into
# ``factors/llm_event_extractor_v2.py``. The wrapper here owns the
# PBOC-specific system prompt; the network plumbing (auth header,
# retry policy, rate limiter) is reused from the V2 extractor.
# ─────────────────────────────────────────────────────────────────────
def _llm_extract_via_minimax(content: str,
                              system_prompt: str | None = None) -> dict[str, Any] | None:
    """Call MiniMax with an explicit system prompt; return parsed JSON.

    2026-06-07 (cx P1 #2 fix): the caller passes ``system_prompt``
    explicitly so we no longer rely on monkey-patching the V2 module's
    global. The V2 extractor's internal retry/backoff applies inside
    ``_call_llm``.

    Raises on hard failure (network down, auth invalid after retries)
    so the caller's ``n_failed`` counter records it correctly.
    """
    from factors.llm_event_extractor_v2 import LLMEventExtractorV2

    # Construct on first use so test mocks can bypass entirely.
    ext = LLMEventExtractorV2()
    text, usage = ext._call_llm(content[:3000],
                                  system_prompt=system_prompt)
    if not usage.get("http_ok"):
        raise RuntimeError(
            f"MiniMax call failed (rate_limited={usage.get('rate_limited')})"
        )

    # Reuse the V2 JSON parser's "find first { … last }" trick.
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
        return json.loads(clean[start:end])
    except json.JSONDecodeError:
        return None


def _llm_extract_with_pbc_prompt(content: str) -> dict[str, Any] | None:
    """Production LLM call with the PBC system prompt passed explicitly.

    2026-06-07 cx P1 #3 fix (round 2): the previous version monkey-
    patched the V2 module-global SYSTEM_PROMPT_V2 which is unsafe
    under concurrent callers — the per-stock LLM pipeline and the
    PBC extraction can both run in-process and would scramble each
    other's system prompts. The patched _call_llm now accepts a
    per-call ``system_prompt`` arg; this wrapper threads SYSTEM_PROMPT_PBC
    through ``_llm_extract_via_minimax(..., system_prompt=...)`` so the
    PBC prompt never touches the module global.
    """
    return _llm_extract_via_minimax(content, system_prompt=SYSTEM_PROMPT_PBC)


def _llm_extract_with_state_council_prompt(
    content: str,
) -> dict[str, Any] | None:
    """Production LLM call with the SC prompt threaded as a kwarg.

    Same thread-safety constraint as ``_llm_extract_with_pbc_prompt``:
    we must NEVER mutate ``factors.llm_event_extractor_v2.SYSTEM_PROMPT_V2``
    or any other module-global. The per-call ``system_prompt`` arg is
    the only safe channel because the per-stock LLM pipeline can be
    running concurrently.
    """
    return _llm_extract_via_minimax(content, system_prompt=SYSTEM_PROMPT_SC)


def _llm_extract_with_nbs_prompt(
    content: str,
) -> dict[str, Any] | None:
    """Production LLM call with SYSTEM_PROMPT_NBS threaded as a kwarg.

    Same thread-safety contract as the PBC / SC wrappers: we MUST NEVER
    monkey-patch ``factors.llm_event_extractor_v2.SYSTEM_PROMPT_V2``.
    The per-call ``system_prompt`` arg is the only safe channel — the
    per-stock LLM pipeline and the macro-stats extraction can both be
    running in-process at 16:00 (PE-3 sits between the 15:50 PE-1
    extract and the 16:10 PE-1 factor build).
    """
    return _llm_extract_via_minimax(content, system_prompt=SYSTEM_PROMPT_NBS)


def _llm_extract_with_xinwen_lianbo_prompt(
    content: str,
) -> dict[str, Any] | None:
    """Production LLM call with SYSTEM_PROMPT_XINWEN_LIANBO threaded as
    a kwarg.

    Same thread-safety contract as the PBC / SC / NBS wrappers: we MUST
    NEVER monkey-patch ``factors.llm_event_extractor_v2.SYSTEM_PROMPT_V2``.
    The per-call ``system_prompt`` arg is the only safe channel — the
    per-stock LLM pipeline and the XWLB extraction run in the same
    process (PE-4 sits between the 15:45 PE-3 collect and the 16:25
    PE-4 build, and the V2 per-stock job is the L4 16:30 pipeline).
    """
    return _llm_extract_via_minimax(
        content, system_prompt=SYSTEM_PROMPT_XINWEN_LIANBO,
    )


# ─────────────────────────────────────────────────────────────────────
# EventStore append — best-effort. The unit tests pass an empty dir
# and we want them to pass even when EventStore isn't fully bootstrapped.
# ─────────────────────────────────────────────────────────────────────
def _append_to_event_store(
    row: dict, *, source_row: dict, eventstore_dir: Path | None,
) -> None:
    """Append one extracted event to the unified EventStore.

    Maps PBOC fields onto the EventStore schema:
      - source = "policy" (matches EventStore's category vocab)
      - event_type = "policy_support" / "policy_negative" / "other"
        derived from policy_stance
      - direction = +1 / -1 / 0 derived from policy_stance
      - confidence = unexpectedness when present, else 0.5
      - stock_code = "MARKET" (PBOC events are market-level, not per-stock)
    """
    try:
        from factors.event_store import EventStore
    except Exception as e:  # pragma: no cover — broken install
        logger.warning("EventStore import failed (%s); skipping append", e)
        return

    try:
        store = EventStore(store_dir=eventstore_dir) if eventstore_dir else EventStore()
    except Exception as e:
        logger.warning("EventStore init failed (%s); skipping append", e)
        return

    stance = row["policy_stance"]
    direction = {"easing": 1, "tightening": -1}.get(stance, 0)
    event_type = {
        "easing": "policy_support",
        "tightening": "policy_negative",
    }.get(stance, "other")
    conf_raw = row.get("unexpectedness")
    confidence = float(conf_raw) if conf_raw is not None else 0.5

    publish_time = source_row.get("publish_date", "")
    summary_text = (source_row.get("title") or "")[:200]
    # PE-5 (task #144): we pass publish_time only; EventStore.add_event's
    # strict PIT contract derives available_time / signal_date /
    # execution_date for us. If publish_time is empty the write will be
    # rejected with PITContractError, surfaced as a warning below — a
    # missing publish_date upstream is itself a data-quality bug.
    try:
        store.add_event({
            "date": publish_time,
            "stock_code": "MARKET",
            "source": "policy",
            "event_type": event_type,
            "direction": direction,
            "confidence": confidence,
            "summary": summary_text,
            "publish_time": publish_time,
            "event_time": publish_time,
            "topic": row.get("tool_type", "other"),
            "is_policy": True,
        })
    except Exception as e:
        logger.warning("EventStore.add_event failed: %s", e)


def _append_to_event_store_sc(
    row: dict, *, source_row: dict, eventstore_dir: Path | None,
) -> None:
    """Append a PE-2 State Council event to the unified EventStore.

    Schema choices:
      - source = "policy" (same category as PBC; the EventStore vocab
        does not split monetary / industrial — the consumer can split
        on event_type or topic).
      - event_type = "industry_policy_support" / "industry_policy_negative"
        / "other" derived from policy_direction.
      - direction = +1 / -1 / 0.
      - confidence = policy_strength when present, else 0.5.
      - One event per target_industry: the LLM hands us a list of
        industries and we emit one EventStore row per (industry, doc).
        Industries are keyed as synthetic instruments
        ``INDUSTRY_<UPPER>`` so the downstream factor builder can
        broadcast without a stock-list join.
    """
    try:
        from factors.event_store import EventStore
    except Exception as e:  # pragma: no cover — broken install
        logger.warning("EventStore import failed (%s); skipping append", e)
        return

    try:
        store = EventStore(store_dir=eventstore_dir) if eventstore_dir else EventStore()
    except Exception as e:
        logger.warning("EventStore init failed (%s); skipping append", e)
        return

    direction_str = row.get("policy_direction", "neutral")
    dir_int = {"supportive": 1, "restrictive": -1}.get(direction_str, 0)
    event_type = {
        "supportive": "industry_policy_support",
        "restrictive": "industry_policy_negative",
    }.get(direction_str, "other")
    conf_raw = row.get("policy_strength")
    confidence = float(conf_raw) if conf_raw is not None else 0.5

    publish_time = source_row.get("publish_date", "")
    summary_text = (source_row.get("title") or "")[:200]
    industries = row.get("target_industries") or []
    if not industries:
        # No target industry — keep a single MARKET-keyed event so the
        # text isn't silently lost.
        industries = ["__market__"]
    for industry in industries:
        stock_code = (
            "MARKET" if industry == "__market__"
            else f"INDUSTRY_{industry.upper()}"
        )
        # PE-5: publish_time is required by the strict PIT contract;
        # EventStore derives the other 3 time fields.
        try:
            store.add_event({
                "date": publish_time,
                "stock_code": stock_code,
                "source": "policy",
                "event_type": event_type,
                "direction": dir_int,
                "confidence": confidence,
                "summary": summary_text,
                "publish_time": publish_time,
                "event_time": publish_time,
                "topic": industry,
                "is_policy": True,
            })
        except Exception as e:
            logger.warning("EventStore.add_event failed (%s): %s", industry, e)


def _append_to_event_store_nbs(
    row: dict, *, source_row: dict, eventstore_dir: Path | None,
) -> None:
    """Append a PE-3 NBS macro-surprise event to the unified EventStore.

    Schema choices:
      - source = "policy" (same category as PBC; macro releases are
        market-wide signals).
      - event_type = "macro_surprise_<series>" (e.g. "macro_surprise_cpi").
      - direction = +1 for upside, -1 for downside, 0 for inline / unknown.
      - confidence = 0.7 when surprise_direction is known, 0.4 otherwise.
        (Lower default than the PBC unexpectedness=0.5 floor because a
        macro release with no consensus quoted is genuinely lower-info.)
      - stock_code = "MARKET" — macro stats affect the whole market.
    """
    try:
        from factors.event_store import EventStore
    except Exception as e:  # pragma: no cover — broken install
        logger.warning("EventStore import failed (%s); skipping append", e)
        return

    try:
        store = EventStore(store_dir=eventstore_dir) if eventstore_dir else EventStore()
    except Exception as e:
        logger.warning("EventStore init failed (%s); skipping append", e)
        return

    direction_str = row.get("surprise_direction", "unknown")
    dir_int = {"upside": 1, "downside": -1}.get(direction_str, 0)
    series = row.get("series_name", "other")
    event_type = f"macro_surprise_{series}"
    confidence = 0.7 if direction_str in ("upside", "downside") else 0.4

    publish_time = source_row.get("publish_date", "")
    summary_text = (source_row.get("title") or "")[:200]
    # PE-5: only publish_time is supplied; EventStore derives the rest.
    try:
        store.add_event({
            "date": publish_time,
            "stock_code": "MARKET",
            "source": "policy",
            "event_type": event_type,
            "direction": dir_int,
            "confidence": confidence,
            "summary": summary_text,
            "publish_time": publish_time,
            "event_time": publish_time,
            "topic": series,
            "is_policy": True,
        })
    except Exception as e:
        logger.warning("EventStore.add_event failed: %s", e)


def _append_to_event_store_xinwen_lianbo(
    row: dict, *, source_row: dict, eventstore_dir: Path | None,
) -> None:
    """Append a PE-4 XWLB theme-attention event to the unified EventStore.

    Schema choices mirror PE-2's per-industry split, but keyed per theme:
      - source = "policy" (same category as PBC / SC / NBS).
      - event_type = "media_attention_<theme>".
      - direction = 0 — state-media attention is NOT a directional
        forecast; the downstream factor builder counts mentions, the
        sign of the price move is left entirely to the model.
      - confidence = policy_priority_signal (0..1).
      - stock_code = ``THEME_<UPPER>`` per theme so the FeatureMerger
        can broadcast a theme to its stock basket via a downstream
        theme→stock mapper (analogous to the PE-2 industry mapper).
        Rows with NO themes get a single MARKET-keyed event so the
        broadcast is not silently dropped.
    """
    try:
        from factors.event_store import EventStore
    except Exception as e:  # pragma: no cover — broken install
        logger.warning("EventStore import failed (%s); skipping append", e)
        return

    try:
        store = EventStore(store_dir=eventstore_dir) if eventstore_dir else EventStore()
    except Exception as e:
        logger.warning("EventStore init failed (%s); skipping append", e)
        return

    themes = row.get("themes") or []
    counts = row.get("theme_mention_counts") or {}
    priority = row.get("policy_priority_signal")
    confidence = float(priority) if priority is not None else 0.5

    publish_time = source_row.get("publish_date", "")
    summary_text = (source_row.get("title") or "")[:200]

    if not themes:
        # No theme — keep a single MARKET-keyed event so the broadcast
        # isn't silently lost. event_type stays generic.
        try:
            store.add_event({
                "date": publish_time,
                "stock_code": "MARKET",
                "source": "policy",
                "event_type": "media_attention_none",
                "direction": 0,
                "confidence": confidence,
                "summary": summary_text,
                "publish_time": publish_time,
                "topic": "xinwen_lianbo",
                "is_policy": True,
            })
        except Exception as e:
            logger.warning("EventStore.add_event (XWLB MARKET) failed: %s", e)
        return

    for theme in themes:
        try:
            store.add_event({
                "date": publish_time,
                "stock_code": f"THEME_{theme.upper()}",
                "source": "policy",
                "event_type": f"media_attention_{theme}",
                "direction": 0,
                "confidence": confidence,
                "summary": summary_text,
                "publish_time": publish_time,
                "topic": theme,
                "is_policy": True,
                # Carry mention count + priority in `extra` so the
                # downstream factor builder doesn't have to reach
                # back into the JSONL.
                "extra": {
                    "mention_count": int(counts.get(theme, 1)),
                    "policy_priority_signal": confidence,
                },
            })
        except Exception as e:
            logger.warning("EventStore.add_event failed (%s): %s", theme, e)


# ─────────────────────────────────────────────────────────────────────
# Core extraction loop — operates on ONE date at a time, injectable
# LLM function for tests.
# ─────────────────────────────────────────────────────────────────────
def extract_pbc(
    *,
    target_date: str,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    eventstore_dir: Path | None = None,
    llm_extract_fn: Callable[[str], dict[str, Any] | None] | None = None,
) -> dict:
    """Extract policy events for a single date.

    Parameters
    ----------
    target_date:
        ``YYYY-MM-DD``. Reads ``<input_dir>/<target_date>.jsonl``.
    input_dir / output_dir / eventstore_dir:
        Override storage roots for tests. ``None`` uses production paths.
    llm_extract_fn:
        Injectable LLM call. Must return parsed-JSON dict or None on
        parse failure, and raise on hard failure. The default
        production path uses ``_llm_extract_with_pbc_prompt``.

    Returns
    -------
    summary dict with keys::

        {"target_date", "n_input", "n_extracted", "n_failed",
         "n_downgraded", "output_path", "errors"}
    """
    input_root = input_dir or INPUT_DIR
    output_root = output_dir or OUTPUT_DIR
    llm_fn = llm_extract_fn or _llm_extract_with_pbc_prompt

    input_path = input_root / f"{target_date}.jsonl"
    output_path = output_root / f"{target_date}.jsonl"

    summary = {
        "target_date": target_date,
        "n_input": 0,
        "n_extracted": 0,
        "n_failed": 0,
        "n_downgraded": 0,
        "output_path": str(output_path),
        "errors": [],
    }

    if not input_path.exists():
        logger.info("No input file for %s at %s — nothing to extract", target_date, input_path)
        # Still write an empty output file so re-runs are deterministic.
        _atomic_write_jsonl([], output_path)
        return summary

    input_rows: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                input_rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                summary["errors"].append({"stage": "input_parse", "msg": str(e)})

    summary["n_input"] = len(input_rows)
    output_rows: list[dict] = []

    for src in input_rows:
        content = (src.get("content") or "").strip()
        if not content:
            summary["n_failed"] += 1
            summary["errors"].append({
                "stage": "empty_content",
                "url": src.get("url", ""),
            })
            continue

        try:
            raw = llm_fn(content)
        except Exception as e:
            # Hard LLM failure — record and continue. We do NOT silently
            # write an empty row; n_failed reflects reality.
            summary["n_failed"] += 1
            summary["errors"].append({
                "stage": "llm_call",
                "url": src.get("url", ""),
                "msg": str(e)[:200],
            })
            continue

        if raw is None:
            summary["n_failed"] += 1
            summary["errors"].append({
                "stage": "llm_parse",
                "url": src.get("url", ""),
            })
            continue

        validated = validate_event(raw)
        if (
            validated["policy_stance"] == "unknown"
            and isinstance(raw.get("policy_stance"), str)
            and raw.get("policy_stance", "").lower() not in POLICY_STANCES
        ):
            summary["n_downgraded"] += 1
        if (
            validated["tool_type"] == "other"
            and isinstance(raw.get("tool_type"), str)
            and raw.get("tool_type", "").lower() not in TOOL_TYPES
        ):
            summary["n_downgraded"] += 1

        row = {
            "publish_date": src.get("publish_date", target_date),
            "policy_type": src.get("policy_type", "other"),
            "title": src.get("title", ""),
            "url": src.get("url", ""),
            **validated,
            "extracted_at": _now_utc_iso(),
        }
        output_rows.append(row)
        summary["n_extracted"] += 1

        # Append to EventStore (best-effort).
        _append_to_event_store(
            row, source_row=src, eventstore_dir=eventstore_dir,
        )

    # Atomic write — overwrites any previous run for this date.
    _atomic_write_jsonl(output_rows, output_path)
    logger.info(
        "Extracted %d/%d events for %s (failed=%d, downgraded=%d) → %s",
        summary["n_extracted"], summary["n_input"], target_date,
        summary["n_failed"], summary["n_downgraded"], output_path,
    )
    return summary


# ─────────────────────────────────────────────────────────────────────
# Phase E.2 (PE-2) — State Council extract loop. Same shape as
# extract_pbc but uses SC validator + SC LLM wrapper + SC EventStore
# append. Kept as a separate function so tests can drive each
# independently and a future refactor can collapse them once the
# field surface stabilises.
# ─────────────────────────────────────────────────────────────────────
def extract_state_council(
    *,
    target_date: str,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    eventstore_dir: Path | None = None,
    llm_extract_fn: Callable[[str], dict[str, Any] | None] | None = None,
) -> dict:
    """Extract State Council / ministry events for a single date.

    See ``extract_pbc`` for the parameter shape. The only differences:

      - default input is ``policy_texts/state_council/<date>.jsonl``
      - default output is ``policy_events/state_council/<date>.jsonl``
      - default LLM wrapper is ``_llm_extract_with_state_council_prompt``
      - validator is ``validate_event_state_council``
      - EventStore append uses the SC variant (one event per industry)
    """
    input_root = input_dir or INPUT_DIR_SC
    output_root = output_dir or OUTPUT_DIR_SC
    llm_fn = llm_extract_fn or _llm_extract_with_state_council_prompt

    input_path = input_root / f"{target_date}.jsonl"
    output_path = output_root / f"{target_date}.jsonl"

    summary = {
        "target_date": target_date,
        "n_input": 0,
        "n_extracted": 0,
        "n_failed": 0,
        "n_downgraded": 0,
        "output_path": str(output_path),
        "errors": [],
    }

    if not input_path.exists():
        logger.info(
            "No SC input file for %s at %s — writing empty output",
            target_date, input_path,
        )
        _atomic_write_jsonl([], output_path)
        return summary

    input_rows: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                input_rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                summary["errors"].append({"stage": "input_parse", "msg": str(e)})

    summary["n_input"] = len(input_rows)
    output_rows: list[dict] = []

    for src in input_rows:
        content = (src.get("content") or "").strip()
        if not content:
            summary["n_failed"] += 1
            summary["errors"].append({
                "stage": "empty_content",
                "url": src.get("url", ""),
            })
            continue

        try:
            raw = llm_fn(content)
        except Exception as e:
            summary["n_failed"] += 1
            summary["errors"].append({
                "stage": "llm_call",
                "url": src.get("url", ""),
                "msg": str(e)[:200],
            })
            continue

        if raw is None:
            summary["n_failed"] += 1
            summary["errors"].append({
                "stage": "llm_parse",
                "url": src.get("url", ""),
            })
            continue

        validated = validate_event_state_council(raw)
        # Track downgrades for visibility.
        if (
            validated["policy_direction"] == "neutral"
            and isinstance(raw.get("policy_direction"), str)
            and raw.get("policy_direction", "").lower() not in POLICY_DIRECTIONS
        ):
            summary["n_downgraded"] += 1
        if (
            validated["subsidy_or_tax"] == "neither"
            and isinstance(raw.get("subsidy_or_tax"), str)
            and raw.get("subsidy_or_tax", "").lower() not in SUBSIDY_TAX_TYPES
        ):
            summary["n_downgraded"] += 1

        row = {
            "publish_date": src.get("publish_date", target_date),
            "policy_type": src.get("policy_type", "other"),
            "title": src.get("title", ""),
            "url": src.get("url", ""),
            **validated,
            "extracted_at": _now_utc_iso(),
        }
        output_rows.append(row)
        summary["n_extracted"] += 1

        _append_to_event_store_sc(
            row, source_row=src, eventstore_dir=eventstore_dir,
        )

    _atomic_write_jsonl(output_rows, output_path)
    logger.info(
        "SC extracted %d/%d events for %s (failed=%d, downgraded=%d) → %s",
        summary["n_extracted"], summary["n_input"], target_date,
        summary["n_failed"], summary["n_downgraded"], output_path,
    )
    return summary


# ─────────────────────────────────────────────────────────────────────
# Phase E.3 (PE-3) — NBS macro-surprise extract loop. Same shape as
# extract_pbc / extract_state_council but uses NBS validator + NBS LLM
# wrapper + NBS EventStore append. stock_code is MARKET (same as PBC):
# macro releases are market-wide signals.
# ─────────────────────────────────────────────────────────────────────
def extract_nbs(
    *,
    target_date: str,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    eventstore_dir: Path | None = None,
    llm_extract_fn: Callable[[str], dict[str, Any] | None] | None = None,
) -> dict:
    """Extract NBS macro-surprise events for a single date.

    See ``extract_pbc`` for the parameter shape. NBS-specific differences:

      - default input is ``policy_texts/nbs/<date>.jsonl``
      - default output is ``policy_events/nbs/<date>.jsonl``
      - default LLM wrapper is ``_llm_extract_with_nbs_prompt``
      - validator is ``validate_event_nbs``
      - EventStore append uses the NBS variant (MARKET-keyed)
    """
    input_root = input_dir or INPUT_DIR_NBS
    output_root = output_dir or OUTPUT_DIR_NBS
    llm_fn = llm_extract_fn or _llm_extract_with_nbs_prompt

    input_path = input_root / f"{target_date}.jsonl"
    output_path = output_root / f"{target_date}.jsonl"

    summary = {
        "target_date": target_date,
        "n_input": 0,
        "n_extracted": 0,
        "n_failed": 0,
        "n_downgraded": 0,
        "output_path": str(output_path),
        "errors": [],
    }

    if not input_path.exists():
        logger.info(
            "No NBS input file for %s at %s — writing empty output",
            target_date, input_path,
        )
        _atomic_write_jsonl([], output_path)
        return summary

    input_rows: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                input_rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                summary["errors"].append({"stage": "input_parse", "msg": str(e)})

    summary["n_input"] = len(input_rows)
    output_rows: list[dict] = []

    for src in input_rows:
        content = (src.get("content") or "").strip()
        if not content:
            summary["n_failed"] += 1
            summary["errors"].append({
                "stage": "empty_content",
                "url": src.get("url", ""),
            })
            continue

        try:
            raw = llm_fn(content)
        except Exception as e:
            summary["n_failed"] += 1
            summary["errors"].append({
                "stage": "llm_call",
                "url": src.get("url", ""),
                "msg": str(e)[:200],
            })
            continue

        if raw is None:
            summary["n_failed"] += 1
            summary["errors"].append({
                "stage": "llm_parse",
                "url": src.get("url", ""),
            })
            continue

        validated = validate_event_nbs(raw)
        # Track downgrades for visibility (out-of-vocab enum → fallback).
        if (
            validated["series_name"] == "other"
            and isinstance(raw.get("series_name"), str)
            and raw.get("series_name", "").lower() not in NBS_SERIES_NAMES
        ):
            summary["n_downgraded"] += 1
        if (
            validated["surprise_direction"] == "unknown"
            and isinstance(raw.get("surprise_direction"), str)
            and raw.get("surprise_direction", "").lower()
            not in NBS_SURPRISE_DIRECTIONS
        ):
            summary["n_downgraded"] += 1

        row = {
            "publish_date": src.get("publish_date", target_date),
            "policy_type": src.get("policy_type", "other"),
            "title": src.get("title", ""),
            "url": src.get("url", ""),
            **validated,
            "extracted_at": _now_utc_iso(),
        }
        output_rows.append(row)
        summary["n_extracted"] += 1

        _append_to_event_store_nbs(
            row, source_row=src, eventstore_dir=eventstore_dir,
        )

    _atomic_write_jsonl(output_rows, output_path)
    logger.info(
        "NBS extracted %d/%d events for %s (failed=%d, downgraded=%d) → %s",
        summary["n_extracted"], summary["n_input"], target_date,
        summary["n_failed"], summary["n_downgraded"], output_path,
    )
    return summary


# ─────────────────────────────────────────────────────────────────────
# Phase E.4 (PE-4) — Xinwen Lianbo theme attention extract loop.
# Same shape as extract_pbc but uses the XWLB validator + XWLB LLM
# wrapper + XWLB EventStore append (one event per theme). Each
# extracted row carries a list of themes; the factor builder explodes
# on themes to produce per-(date, THEME_<NAME>) factors.
# ─────────────────────────────────────────────────────────────────────
def extract_xinwen_lianbo(
    *,
    target_date: str,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    eventstore_dir: Path | None = None,
    llm_extract_fn: Callable[[str], dict[str, Any] | None] | None = None,
) -> dict:
    """Extract XWLB theme-attention events for a single date.

    See ``extract_pbc`` for the parameter shape. XWLB-specific
    differences:

      - default input is ``policy_texts/xinwen_lianbo/<date>.jsonl``
      - default output is ``policy_events/xinwen_lianbo/<date>.jsonl``
      - default LLM wrapper is ``_llm_extract_with_xinwen_lianbo_prompt``
      - validator is ``validate_event_xinwen_lianbo``
      - EventStore append uses the XWLB variant (one event per theme,
        keyed ``THEME_<UPPER>``).
    """
    input_root = input_dir or INPUT_DIR_XWLB
    output_root = output_dir or OUTPUT_DIR_XWLB
    llm_fn = llm_extract_fn or _llm_extract_with_xinwen_lianbo_prompt

    input_path = input_root / f"{target_date}.jsonl"
    output_path = output_root / f"{target_date}.jsonl"

    summary = {
        "target_date": target_date,
        "n_input": 0,
        "n_extracted": 0,
        "n_failed": 0,
        "n_downgraded": 0,
        "n_unknown_themes": 0,
        "output_path": str(output_path),
        "errors": [],
    }

    if not input_path.exists():
        logger.info(
            "No XWLB input file for %s at %s — writing empty output",
            target_date, input_path,
        )
        _atomic_write_jsonl([], output_path)
        return summary

    input_rows: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                input_rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                summary["errors"].append({"stage": "input_parse", "msg": str(e)})

    summary["n_input"] = len(input_rows)
    output_rows: list[dict] = []

    for src in input_rows:
        content = (src.get("content") or "").strip()
        if not content:
            summary["n_failed"] += 1
            summary["errors"].append({
                "stage": "empty_content",
                "url": src.get("url", ""),
            })
            continue

        try:
            raw = llm_fn(content)
        except Exception as e:
            summary["n_failed"] += 1
            summary["errors"].append({
                "stage": "llm_call",
                "url": src.get("url", ""),
                "msg": str(e)[:200],
            })
            continue

        if raw is None:
            summary["n_failed"] += 1
            summary["errors"].append({
                "stage": "llm_parse",
                "url": src.get("url", ""),
            })
            continue

        validated = validate_event_xinwen_lianbo(raw)
        # Track how many themes fell outside the documented set so an
        # operator can spot vocab drift. Themes are KEPT either way
        # (free-form vocab — this is just a measurement).
        unknown = [
            t for t in validated["themes"]
            if t not in XINWEN_LIANBO_KNOWN_THEMES
        ]
        if unknown:
            summary["n_unknown_themes"] += len(unknown)

        row = {
            "publish_date": src.get("publish_date", target_date),
            "policy_type": src.get("policy_type", "xinwen_lianbo_daily"),
            "title": src.get("title", ""),
            "url": src.get("url", ""),
            **validated,
            "extracted_at": _now_utc_iso(),
        }
        output_rows.append(row)
        summary["n_extracted"] += 1

        _append_to_event_store_xinwen_lianbo(
            row, source_row=src, eventstore_dir=eventstore_dir,
        )

    _atomic_write_jsonl(output_rows, output_path)
    logger.info(
        "XWLB extracted %d/%d events for %s (failed=%d, unknown_themes=%d) → %s",
        summary["n_extracted"], summary["n_input"], target_date,
        summary["n_failed"], summary["n_unknown_themes"], output_path,
    )
    return summary


# ─────────────────────────────────────────────────────────────────────
# Health publishing
# ─────────────────────────────────────────────────────────────────────
def publish_health(
    summary: dict,
    *,
    target_date: str,
    health_source: str = HEALTH_SOURCE_NAME,
    sparse_steady: bool = False,
) -> None:
    """``sparse_steady=True`` mirrors collect_policy_texts.publish_health —
    when the upstream texts source is sparse-by-design (e.g. state_council
    while gov.cn list URLs are broken), 0 extracted events is the steady
    state and must not flip the SLA gate red. Added 2026-06-16.
    """
    try:
        from scheduler.data_health import HealthStatus, write_health
    except Exception as e:  # pragma: no cover — broken install
        logger.warning("Cannot import scheduler.data_health (%s)", e)
        return

    n_total = int(summary.get("n_extracted", 0))
    n_failed = int(summary.get("n_failed", 0))
    is_success = (n_total > 0) or sparse_steady
    status = HealthStatus(
        success=is_success,
        n_items=n_total,
        latest_date=target_date,
        partial=(n_total > 0 and n_failed > 0),
        error_type="" if is_success else "no_extracted",
        error_message=(
            "; ".join(
                f"{e.get('stage')}:{e.get('url', '')}"
                for e in summary.get("errors", [])[:3]
            )
            if n_failed else
            "sparse_by_design: upstream texts sparse_steady"
            if sparse_steady and n_total == 0
            else ""
        ),
        network_profile="ashare",
        extra={
            "n_input": summary.get("n_input", 0),
            "n_failed": n_failed,
            "n_downgraded": summary.get("n_downgraded", 0),
            "sparse_steady": sparse_steady,
        },
    )
    write_health(health_source, status, date=target_date)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Extract structured policy events from PBC (E.1) or "
            "State Council / ministry (E.2) texts."
        )
    )
    parser.add_argument(
        "--source", default="pbc",
        choices=["pbc", "state_council", "nbs", "xinwen_lianbo"],
        help=(
            "Policy source. 'pbc' = monetary policy (Phase E.1). "
            "'state_council' = State Council + 3 ministries (Phase E.2). "
            "'nbs' = NBS macro statistics CPI/PPI/PMI/retail sales "
            "(Phase E.3). 'xinwen_lianbo' = CCTV 新闻联播 theme "
            "attention (Phase E.4)."
        ),
    )
    parser.add_argument(
        "--date", default=None,
        help="Single date YYYY-MM-DD (default: today).",
    )
    parser.add_argument(
        "--start", default=None,
        help="Backfill start date YYYY-MM-DD (default: --date).",
    )
    parser.add_argument(
        "--end", default=None,
        help="Backfill end date YYYY-MM-DD (default: --date).",
    )
    args = parser.parse_args(argv)

    today = datetime.now().strftime("%Y-%m-%d")
    if args.start or args.end:
        start = args.start or args.end or today
        end = args.end or args.start or today
    else:
        start = args.date or today
        end = args.date or today

    if args.source == "pbc":
        extract_fn = extract_pbc
        health_source = HEALTH_SOURCE_NAME
    elif args.source == "state_council":
        extract_fn = extract_state_council
        health_source = HEALTH_SOURCE_NAME_SC
    elif args.source == "nbs":
        extract_fn = extract_nbs
        health_source = HEALTH_SOURCE_NAME_NBS
    elif args.source == "xinwen_lianbo":
        extract_fn = extract_xinwen_lianbo
        health_source = HEALTH_SOURCE_NAME_XWLB
    else:
        logger.error("Unsupported --source %s", args.source)
        return 2

    dates = _date_range(start, end)
    overall = {"n_input": 0, "n_extracted": 0, "n_failed": 0, "n_downgraded": 0}
    last_date = end
    # 2026-06-16: state_council is sparse_steady (see collect_policy_texts);
    # propagate that downstream so the extracted=0 case doesn't flip the
    # SLA gate red. Other sources stay strict.
    is_sparse_steady = args.source == "state_council"
    for d in dates:
        s = extract_fn(target_date=d)
        for k in overall:
            overall[k] += s.get(k, 0)
        publish_health(
            s, target_date=d, health_source=health_source,
            sparse_steady=is_sparse_steady,
        )
        last_date = d

    logger.info(
        "Done. n_input=%d, n_extracted=%d, n_failed=%d, n_downgraded=%d",
        overall["n_input"], overall["n_extracted"],
        overall["n_failed"], overall["n_downgraded"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
