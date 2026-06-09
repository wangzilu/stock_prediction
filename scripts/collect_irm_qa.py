"""Collect investor-interaction Q&A from 深交所互动易 (irm.cninfo.com.cn) and
上证 e 互动 (sns.sseinfo.com).

SPIKE STATUS (2026-06-09): scaffold only. Wired for daily incremental writes
but NOT yet hooked into the cron — see
docs/llm_channel_2_irm_qa_spike_20260609.md for the go/no-go criteria before
flipping to production.

Data sources
------------
* AKShare ``stock_irm_cninfo(symbol=<6-digit-code>)`` returns a 14-column
  DataFrame: ['股票代码', '公司简称', '行业', '行业代码', '问题', '提问者',
  '来源', '提问时间', '更新时间', '提问者编号', '问题编号', '回答ID',
  '回答内容', '回答者']. The ``回答内容`` field is the management/IR
  response; ``更新时间`` ≈ answer_date (when management replied).
  Crucially the API returns BOTH question and answer in one DataFrame, so
  we do NOT need to chain ``stock_irm_ans_cninfo`` for every question.
* AKShare ``stock_sns_sseinfo(symbol=<6-digit-code>)`` returns the same
  semantic surface (question, answer, timestamps) for SSE-listed stocks.
  Schema columns (per AKShare docs): ['股票代码', '公司简称', '问题',
  '回答', '问题时间', '回答时间', ...]. Endpoint can be slow / timeout;
  caller MUST tolerate per-symbol failures.

Output layout
-------------
``data/storage/irm_qa/<YYYY-MM-DD>.jsonl`` — one row per Q&A, deduped by
``(stock_code, question_id)``. The signal date in the filename is the
ANSWER date, not the ask date — see "PIT discipline" below.

Schema (per JSONL row)::

    stock_code:    "002594"
    stock_name:    "比亚迪"
    qlib_code:     "sz002594"               # lowercase, matches base cache
    industry:      "制造业"
    venue:         "irm_cninfo" | "sns_sseinfo"
    question_id:   "<API question id, dedup key>"
    question:      <raw text>
    answer:        <raw text, may be empty if unanswered>
    ask_time:      "2026-06-08 19:27:51"
    answer_time:   "2026-06-09 16:06:15"    # may equal ask_time if no answer
    ask_date:      "2026-06-08"
    answer_date:   "2026-06-09"             # = signal_date input (lag +1 BDay)
    is_answered:   true | false
    collected_at:  "<UTC ISO>"

PIT discipline
--------------
The SIGNAL DAY for downstream factor construction is the ANSWER DATE +1
business day, NOT the ask date. Rationale: a retail investor's QUESTION
carries no firm-side information; the public response from management /
IR officer is when new information enters the market. Post-15:00 answers
shift to the next business day exactly as the LLM event pipeline does
(see ``scripts/build_llm_event_factors.py`` ``mask_postclose`` block).

We collect on ``answer_date`` and key the JSONL file by ``answer_date``.
The downstream factor builder is responsible for the +1 BDay shift; it
mirrors policy/event pipeline convention.

Stock universe
--------------
Two modes (mirrors ``scripts/collect_daily_news.py``):
    * ``--portfolio``: only the current overnight Top20 snapshot.
    * default: top-N most liquid A-share stocks (ST_CLIENT bak_basic
      primary, AKShare fallback). Default N=300 — we pre-filter for
      liquidity because small-cap coverage on these platforms is sparse
      (see spike doc § Risks).

Usage
-----
::

    python scripts/collect_irm_qa.py                    # today
    python scripts/collect_irm_qa.py --date 2026-06-09  # specific day
    python scripts/collect_irm_qa.py --portfolio        # Top20 only
    python scripts/collect_irm_qa.py --backfill-days 7  # last 7 days

Rate limiting
-------------
Both endpoints are public + unauthenticated. Conservative: 4 worker threads,
~0.5s sleep between symbol calls per worker, matches the announcement
collector's load profile. AKShare wraps both endpoints with a per-call
tqdm bar; the cninfo endpoint returned 320 rows in ~4.5s for 002594 on
2026-06-09, so 300 stocks ≈ 25-30 min wall-clock.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from config.settings import DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)

IRM_QA_DIR = DATA_DIR / "irm_qa"
IRM_QA_DIR.mkdir(parents=True, exist_ok=True)

# Default universe size when --portfolio not passed. 300 is a deliberate
# trade-off: covers all liquid large + mid caps where IR teams actively
# respond on these platforms. Going wider (500-1000) inflates API calls
# without much marginal Q&A volume — most small-caps see < 1 Q/week.
DEFAULT_UNIVERSE_SIZE = 300

# Per-worker pacing. Keep gentle — both endpoints serve public retail
# traffic and we don't want to be the reason they rate-limit at the IP.
WORKER_THREADS = 4
INTER_CALL_SLEEP_SEC = 0.5

# Per-row column maps. AKShare returns Chinese column names; we
# project to a canonical English-ish schema so downstream code does not
# carry Chinese identifiers.
CNINFO_COL_MAP = {
    "股票代码": "stock_code",
    "公司简称": "stock_name",
    "行业": "industry",
    "问题": "question",
    "回答内容": "answer",
    "提问时间": "ask_time",
    "更新时间": "answer_time",
    "问题编号": "question_id",
}
SSEINFO_COL_MAP = {
    "股票代码": "stock_code",
    "公司简称": "stock_name",
    "问题": "question",
    "回答": "answer",
    "问题时间": "ask_time",
    "回答时间": "answer_time",
}


def _to_qlib_code(code: str) -> str:
    """Lowercase qlib_code convention (matches base 209 production cache).

    See factors/feature_cache_utils.py for why this MUST be lowercase.
    """
    code = str(code).strip().zfill(6)
    if not code.isdigit() or len(code) != 6:
        return ""
    if code.startswith(("60", "68", "9")):
        return f"sh{code}"
    if code.startswith(("00", "30", "20")):
        return f"sz{code}"
    if code.startswith(("4", "8")):
        return f"bj{code}"
    return ""


def _normalize_ts(value) -> str:
    """Coerce AKShare timestamp value to ISO ``YYYY-MM-DD HH:MM:SS``.

    AKShare sometimes returns ``pd.Timestamp``, ``str``, or ``None``.
    Empty / unparseable values become an empty string — downstream
    code MUST handle that as "unknown time".
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        ts = pd.to_datetime(value)
        if pd.isna(ts):
            return ""
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return ""


def _date_from_ts(iso_ts: str) -> str:
    return iso_ts[:10] if iso_ts else ""


def get_universe(portfolio: bool, top_n: int) -> list[dict]:
    """Resolve the stock universe for collection.

    Returns a list of ``{code, name, qlib_code}`` dicts. Same data
    contract as ``scripts/collect_daily_news.get_liquid_stocks``.

    NOTE: keeping this as a thin wrapper so a single canonical source
    of universe ranking lives in one place — when the daily-news
    collector's logic evolves we want this caller to inherit it.
    """
    if portfolio:
        from scripts.collect_daily_news import get_portfolio_stocks
        items = get_portfolio_stocks()
        if items:
            return items
        logger.warning("Portfolio snapshot empty — falling back to liquid universe")
    from scripts.collect_daily_news import get_liquid_stocks
    return get_liquid_stocks(top_n=top_n)


def fetch_one_cninfo(code: str) -> pd.DataFrame | None:
    """Fetch ``stock_irm_cninfo`` for one symbol. Returns canonical-column
    DataFrame, or None on failure / empty payload.

    Per 2026-06-09 probe: returns ~320 rows for an active large cap (BYD)
    with both question and answer in a single call. NO chained call to
    ``stock_irm_ans_cninfo`` needed — that helper exists for the SINGLE-
    question detail view; we want the bulk listing.
    """
    import akshare as ak
    try:
        df = ak.stock_irm_cninfo(symbol=str(code))
    except Exception as e:
        logger.debug("cninfo fetch failed for %s: %s", code, repr(e)[:160])
        return None
    if df is None or df.empty:
        return None
    df = df.rename(columns=CNINFO_COL_MAP)
    df["venue"] = "irm_cninfo"
    return df


def fetch_one_sseinfo(code: str) -> pd.DataFrame | None:
    """Fetch ``stock_sns_sseinfo`` for one symbol.

    The endpoint is timeout-prone (see probe note in spike doc).
    Returns None on any failure — caller logs but does NOT retry, this is
    a soft data source.
    """
    import akshare as ak
    try:
        df = ak.stock_sns_sseinfo(symbol=str(code))
    except Exception as e:
        logger.debug("sseinfo fetch failed for %s: %s", code, repr(e)[:160])
        return None
    if df is None or df.empty:
        return None
    df = df.rename(columns=SSEINFO_COL_MAP)
    df["venue"] = "sns_sseinfo"
    return df


def fetch_one_stock(item: dict, target_answer_date: str) -> list[dict]:
    """Fetch IRM + SNS Q&A for one stock, filter to rows whose answer_date
    matches ``target_answer_date``.

    PIT note: we filter on answer_date NOT ask_date. A question asked
    weeks ago but answered today is a TODAY signal.
    """
    code = item["code"]
    name = item.get("name", "")
    qlib = item.get("qlib_code") or _to_qlib_code(code)
    rows: list[dict] = []

    # 6xx/9xx → SSE → try sseinfo first; 0xx/3xx → SZSE → try cninfo first.
    if code.startswith(("60", "68", "9")):
        primary_fn, secondary_fn = fetch_one_sseinfo, fetch_one_cninfo
    else:
        primary_fn, secondary_fn = fetch_one_cninfo, fetch_one_sseinfo

    for fetch_fn in (primary_fn, secondary_fn):
        df = fetch_fn(code)
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            ask_iso = _normalize_ts(r.get("ask_time"))
            ans_iso = _normalize_ts(r.get("answer_time"))
            ask_date = _date_from_ts(ask_iso)
            answer_date = _date_from_ts(ans_iso) or ask_date
            if target_answer_date and answer_date != target_answer_date:
                continue
            question = str(r.get("question") or "").strip()
            answer = str(r.get("answer") or "").strip()
            if not question:
                continue
            qid = str(r.get("question_id") or r.get("提问者编号") or "")
            rows.append({
                "stock_code": code,
                "stock_name": name or str(r.get("stock_name") or ""),
                "qlib_code": qlib,
                "industry": str(r.get("industry") or ""),
                "venue": r.get("venue", ""),
                "question_id": qid,
                "question": question,
                "answer": answer,
                "ask_time": ask_iso,
                "answer_time": ans_iso,
                "ask_date": ask_date,
                "answer_date": answer_date,
                "is_answered": bool(answer),
                "collected_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            })
        time.sleep(INTER_CALL_SLEEP_SEC)

    return rows


def dedup_by_question_id(rows: list[dict]) -> list[dict]:
    """Two venues can theoretically duplicate (rare — different stock sets).
    Dedup by ``(stock_code, question_id)`` keeping the first (which is the
    primary venue per code prefix).
    """
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for r in rows:
        key = (r["stock_code"], r["question_id"])
        if not r["question_id"]:
            # Without an ID, fall back to (code, question text hash). This is
            # a defensive path — the AKShare schema includes question_id, so
            # missing ID means the upstream API changed and we want a loud
            # log but no hard failure.
            key = (r["stock_code"], r["question"][:80])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def write_jsonl(rows: list[dict], date_str: str) -> Path:
    """Atomic write to ``irm_qa/<date>.jsonl``. Overwrites — caller is
    expected to pass a complete day's snapshot.
    """
    out_path = IRM_QA_DIR / f"{date_str}.jsonl"
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out_path)
    return out_path


def collect_for_date(
    target_answer_date: str,
    universe: list[dict],
    workers: int = WORKER_THREADS,
) -> Path:
    """Drive the collection for a single answer_date across the universe."""
    logger.info(
        "Collecting IRM Q&A for answer_date=%s, universe=%d stocks",
        target_answer_date, len(universe),
    )
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(fetch_one_stock, item, target_answer_date): item
            for item in universe
        }
        for i, fut in enumerate(as_completed(futures), 1):
            item = futures[fut]
            try:
                rows.extend(fut.result())
            except Exception as e:
                logger.warning("fetch failed for %s: %s", item.get("code"), repr(e)[:160])
            if i % 50 == 0:
                logger.info("  progress: %d/%d, rows so far=%d", i, len(universe), len(rows))

    rows = dedup_by_question_id(rows)
    out = write_jsonl(rows, target_answer_date)
    logger.info(
        "Wrote %d Q&A rows (%d unique stocks) → %s",
        len(rows), len({r["stock_code"] for r in rows}), out,
    )
    return out


def _publish_health(date_str: str, n_rows: int, n_stocks: int) -> None:
    """Publish freshness signal for the data_health gate.

    NOT YET wired into ``cron_critical_sources`` — the spike doc lists
    this as a go-live checkbox.
    """
    try:
        from scheduler.data_health import HealthStatus, write_health
    except Exception as e:  # pragma: no cover
        logger.debug("data_health import failed (%s)", e)
        return
    status = HealthStatus(
        success=n_rows > 0,
        n_items=n_rows,
        latest_date=date_str,
        error_type="" if n_rows > 0 else "no_qa_rows",
        network_profile="ashare",
        extra={"n_unique_stocks": n_stocks},
    )
    write_health("irm_qa", status, date=date_str)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Collect investor-interaction Q&A.")
    p.add_argument("--date", default=None,
                   help="Answer-date YYYY-MM-DD to filter on (default: today).")
    p.add_argument("--backfill-days", type=int, default=0,
                   help="Backfill the last N days ending --date (default 0 = single day).")
    p.add_argument("--portfolio", action="store_true",
                   help="Only collect for current Top20 snapshot.")
    p.add_argument("--top-n", type=int, default=DEFAULT_UNIVERSE_SIZE,
                   help=f"Universe size when not --portfolio (default {DEFAULT_UNIVERSE_SIZE}).")
    p.add_argument("--workers", type=int, default=WORKER_THREADS)
    args = p.parse_args(argv)

    end_date = args.date or datetime.now().strftime("%Y-%m-%d")
    universe = get_universe(portfolio=args.portfolio, top_n=args.top_n)
    if not universe:
        logger.error("Empty universe — refusing to run")
        return 2

    dates: list[str] = []
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    for d in range(args.backfill_days + 1):
        dates.append((end_dt - timedelta(days=d)).strftime("%Y-%m-%d"))

    rc = 0
    for date_str in sorted(dates):
        try:
            out = collect_for_date(date_str, universe, workers=args.workers)
            n_rows = sum(1 for _ in open(out, encoding="utf-8"))
            n_stocks = len({json.loads(l)["stock_code"]
                            for l in open(out, encoding="utf-8") if l.strip()})
            _publish_health(date_str, n_rows, n_stocks)
        except Exception as e:
            logger.error("collect_for_date(%s) failed: %s", date_str, repr(e)[:200])
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
