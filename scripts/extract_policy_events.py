"""Extract structured policy events from PBOC texts — Phase E.1 step 2.

Reads ``data/storage/policy_texts/pbc/<YYYY-MM-DD>.jsonl`` (produced by
``scripts/collect_policy_texts.py``), calls a non-reasoning LLM on each
row's ``content`` field, and emits structured policy events to:

    data/storage/policy_events/pbc/<YYYY-MM-DD>.jsonl

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

# Health source name matches the convention used by collect_policy_texts.
HEALTH_SOURCE_NAME = "pbc_policy_events"

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
            "topic": row.get("tool_type", "other"),
            "is_policy": True,
        })
    except Exception as e:
        logger.warning("EventStore.add_event failed: %s", e)


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
# Health publishing
# ─────────────────────────────────────────────────────────────────────
def publish_health(summary: dict, *, target_date: str) -> None:
    try:
        from scheduler.data_health import HealthStatus, write_health
    except Exception as e:  # pragma: no cover — broken install
        logger.warning("Cannot import scheduler.data_health (%s)", e)
        return

    n_total = int(summary.get("n_extracted", 0))
    n_failed = int(summary.get("n_failed", 0))
    status = HealthStatus(
        success=n_total > 0,
        n_items=n_total,
        latest_date=target_date,
        partial=(n_total > 0 and n_failed > 0),
        error_type="" if n_total > 0 else "no_extracted",
        error_message=(
            "; ".join(
                f"{e.get('stage')}:{e.get('url', '')}"
                for e in summary.get("errors", [])[:3]
            )
            if n_failed else ""
        ),
        network_profile="ashare",
        extra={
            "n_input": summary.get("n_input", 0),
            "n_failed": n_failed,
            "n_downgraded": summary.get("n_downgraded", 0),
        },
    )
    write_health(HEALTH_SOURCE_NAME, status, date=target_date)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Extract structured policy events from PBOC texts."
    )
    parser.add_argument(
        "--source", default="pbc", choices=["pbc"],
        help="Policy source. Only 'pbc' is implemented today.",
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

    if args.source != "pbc":
        logger.error("Unsupported --source %s", args.source)
        return 2

    dates = _date_range(start, end)
    overall = {"n_input": 0, "n_extracted": 0, "n_failed": 0, "n_downgraded": 0}
    last_date = end
    for d in dates:
        s = extract_pbc(target_date=d)
        for k in overall:
            overall[k] += s.get(k, 0)
        publish_health(s, target_date=d)
        last_date = d

    logger.info(
        "Done. n_input=%d, n_extracted=%d, n_failed=%d, n_downgraded=%d",
        overall["n_input"], overall["n_extracted"],
        overall["n_failed"], overall["n_downgraded"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
