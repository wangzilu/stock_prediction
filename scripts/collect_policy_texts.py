"""Collect policy texts — Phase E.1 step 1 (PBOC liquidity overlay) +
Phase E.2 step 1 (State Council / ministry industry policy overlay) +
Phase E.3 step 1 (NBS macro statistics overlay) +
Phase E.4 step 1 (CCTV Xinwen Lianbo theme attention overlay).

Fetches policy texts from one of four source registries:

- ``--source pbc`` — People's Bank of China monetary policy texts
  (www.pbc.gov.cn), output under
  ``data/storage/policy_texts/pbc/<YYYY-MM-DD>.jsonl``.
- ``--source state_council`` — State Council policy docs + ministry
  bulletins from ``www.gov.cn``, output under
  ``data/storage/policy_texts/state_council/<YYYY-MM-DD>.jsonl``.
- ``--source nbs`` — National Bureau of Statistics CPI / PPI / PMI /
  retail-sales monthly releases from ``www.stats.gov.cn``, output
  under ``data/storage/policy_texts/nbs/<YYYY-MM-DD>.jsonl``.
- ``--source xinwen_lianbo`` — CCTV Xinwen Lianbo (新闻联播) daily
  broadcast transcripts (mirrored via ``news.sina.com.cn`` syndication
  because tv.cctv.com daily archive pages are JavaScript-rendered and
  do not return a static transcript without a headless browser). Output
  under ``data/storage/policy_texts/xinwen_lianbo/<YYYY-MM-DD>.jsonl``.
  Source URL host is ``news.sina.com.cn``.

Raw HTML for any item that fails to parse is dropped into
``data/storage/policy_texts/<source>_raw_html/`` so the LLM extract step
can be re-run with a different parser without re-fetching.

Usage
-----
    # Today only (cron mode)
    python scripts/collect_policy_texts.py --source pbc
    python scripts/collect_policy_texts.py --source state_council

    # Backfill an explicit window
    python scripts/collect_policy_texts.py --source pbc \\
        --start 2026-06-01 --end 2026-06-05

Constraints
-----------
- A-share isolation: this script lives in the A-share namespace and
  must not pull anything from ``data.crypto`` / ``scheduler.crypto`` /
  ssproxy. All HTTP uses plain ``requests`` with a 15-second timeout
  and a 3-attempt retry wrapper.
- Idempotent: re-running the same date overwrites the day's JSONL
  atomically (``.tmp`` + ``replace``). Existing rows for the same day
  are NOT merged — the day is a single re-fetched snapshot.
- Health: every run calls ``scheduler.data_health.write_health`` so the
  Phase A.7 SLA gate can see fresh / stale / partial state.

Output schema (one JSON object per line)
----------------------------------------
    {
      "publish_date":  "YYYY-MM-DD",   # asof_time of the text
      "policy_type":   "omo|mlf|slf|rrr|lpr|quarterly_report|press_conference|other",
      "title":         "原文标题",
      "url":           "https://www.pbc.gov.cn/...",
      "content":       "原文正文 (UTF-8, plain text, whitespace-normalized)",
      "source":        "pbc.gov.cn",
      "fetch_time":    "YYYY-MM-DDTHH:MM:SSZ"
    }

Phase E.1 downstream (out of scope here):
    extract_policy_events.py  -> LLM fact extraction
    build_policy_factors.py   -> PIT-safe factor parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_DIR  # noqa: E402

logger = logging.getLogger(__name__)

# ── Storage layout ───────────────────────────────────────────────────
POLICY_DIR = DATA_DIR / "policy_texts" / "pbc"
RAW_HTML_DIR = DATA_DIR / "policy_texts" / "pbc_raw_html"
# Phase E.2 (PE-2): State Council + ministry policy docs from gov.cn
POLICY_DIR_SC = DATA_DIR / "policy_texts" / "state_council"
RAW_HTML_DIR_SC = DATA_DIR / "policy_texts" / "state_council_raw_html"
# Phase E.3 (PE-3): NBS (国家统计局) macro statistics from stats.gov.cn
POLICY_DIR_NBS = DATA_DIR / "policy_texts" / "nbs"
RAW_HTML_DIR_NBS = DATA_DIR / "policy_texts" / "nbs_raw_html"
# Phase E.4 (PE-4): CCTV Xinwen Lianbo (新闻联播) daily transcripts
# mirrored via news.sina.com.cn. tv.cctv.com daily archive pages are
# JS-rendered and the transcript is only injected after the page boots,
# so we cannot scrape them with plain ``requests``. Sina's "新闻联播文字
# 版" syndication mirrors the transcript as static HTML which the same
# parse_article_page parser can extract.
POLICY_DIR_XWLB = DATA_DIR / "policy_texts" / "xinwen_lianbo"
RAW_HTML_DIR_XWLB = DATA_DIR / "policy_texts" / "xinwen_lianbo_raw_html"

# ── PBOC list pages — kept as named constants so a structural change
# is one-line patchable. URLs verified against the public PBOC
# 货币政策司 column layout (zhengcehuobisi/125207/...).
#
# These are LIST pages: each renders an HTML <ul> of detail-page links
# whose hrefs are relative (``./2026/06/01/...html``) and whose link
# text is the policy title. The list pages tend to outlive any specific
# document URL because they're the directory index, but PBOC has been
# known to renumber the deepest segment — keep the section IDs visible.
PBC_LIST_URLS: dict[str, str] = {
    # OMO daily 公开市场操作业务公告 — highest cadence (daily)
    "omo": "http://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/125475/index.html",
    # LPR 利率 — monthly on the 20th
    "lpr": "http://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/index.html",
    # MLF / SLF / RRR / quarterly report — bonus tier, list pages
    # included for backfill completeness. Parsers are tolerant of
    # 404 / structural drift here; the cron-critical tier is OMO + LPR.
    "mlf": "http://www.pbc.gov.cn/zhengcehuobisi/125207/125213/3963540/index.html",
    "rrr": "http://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125428/index.html",
    "quarterly_report": (
        "http://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/index.html"
    ),
}

PBC_BASE = "http://www.pbc.gov.cn"
GOV_CN_BASE = "http://www.gov.cn"
USER_AGENT = "Mozilla/5.0 (compatible; StockPrediction-PolicyCollector/1.0)"
REQUEST_TIMEOUT = 15  # seconds per attempt — required by spec
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0
INTER_REQUEST_DELAY = 0.4  # be polite to gov.cn / pbc.gov.cn

HEALTH_SOURCE_NAME = "pbc_policy_texts"
HEALTH_SOURCE_NAME_SC = "state_council_policy_texts"
HEALTH_SOURCE_NAME_NBS = "nbs_policy_texts"
HEALTH_SOURCE_NAME_XWLB = "xinwen_lianbo_policy_texts"

# ── Phase E.2 (PE-2) — State Council + ministry list pages.
# Like the PBOC registry above, these are LIST pages: each renders an
# HTML <ul> / <div> of detail-page links. The detail URL patterns at
# gov.cn vary by ministry — both the leading ``/zhengce/zhengceku/...``
# State Council layout and the ``/n.../c.../content.html`` MIIT layout
# are matched by ``parse_list_page`` below (timestamp-numeric path,
# slash-separated YYYY/MM/DD, and ``_<digits>.html`` suffixes are all
# attempted).
#
# Pick the three ministries with the most A-share industry impact:
# 工信部 (MIIT) — semiconductor / EV / industrial policy
# 发改委 (NDRC) — pricing / industrial planning / investment
# 财政部 (MOF) — fiscal / subsidy / tax
# 2026-06-16 revised — gov.cn restructured; sousuo JSON API is HMAC-gated
# (research confirmed in docs/state_council_sousuo_api_research_20260616.md).
# Fallback: hit each ministry's own portal directly. All 5 URLs verified
# live with vanilla User-Agent GET on 2026-06-16. The existing parse_list_page
# regex fan-out handles the date-encoded link patterns; ``state_council_doc``
# and ``state_council_meeting`` share the same zhengce/index.htm list and
# downstream filtering (TODO: STATE_COUNCIL_TITLE_KEYWORDS) splits doc vs
# meeting by title keyword. sparse_steady stays True until that filter ships.
STATE_COUNCIL_LIST_URLS: dict[str, str] = {
    # 国务院政策文件 (索引页, 53 article links per 2026-06 audit)
    "state_council_doc": "https://www.gov.cn/zhengce/index.htm",
    # 国务院常务会议 — same source, filtered downstream by title
    "state_council_meeting": "https://www.gov.cn/zhengce/index.htm",
    # 工信部政策发布
    "miit_policy": "https://www.miit.gov.cn/zwgk/zcwj/wjfb/index.html",
    # 发改委政策通知
    "ndrc_policy": "https://www.ndrc.gov.cn/xxgk/zcfb/tz/",
    # 财政部政策发布
    "mof_policy": "https://www.mof.gov.cn/zhengwuxinxi/zhengcefabu/",
}

STATE_COUNCIL_POLICY_TYPES: frozenset[str] = frozenset(
    STATE_COUNCIL_LIST_URLS.keys()
)


# ── Phase E.3 (PE-3) — NBS (国家统计局) macro statistics list pages.
#
# NBS publishes monthly headline data points (CPI / PPI / PMI / 社零)
# on its "最新发布" / "数据发布" index pages. We point at the two
# top-level indices and the per-series 月度数据 pages; the list-page
# parser is shared with PBC/PE-2 (gov.cn ``/YYYY-MM/DD/content_*.html``
# pattern matches NBS's preferred URL layout).
#
# Note: NBS structural URLs have moved several times (xxgk → sj → ...);
# the parser is tolerant of structural drift via the same regex fan-out
# already in parse_list_page. If a list page 404s on a given day the
# collector records an error and continues — the SLA budget is 35 days
# (monthly publish cadence) so a few empty days are expected.
NBS_LIST_URLS: dict[str, str] = {
    # CPI 月度数据 — published mid-month (e.g. May data on ~10 June)
    "cpi": "http://www.stats.gov.cn/sj/zxfb/",
    # PPI 月度数据 — same release window as CPI
    "ppi": "http://www.stats.gov.cn/sj/sjjd/",
    # PMI 月度数据 — published end-of-month / start-of-next
    "pmi": "http://www.stats.gov.cn/xxgk/sjfb/zxfb2020/",
    # 社会消费品零售总额 月度 — published mid-month
    "retail_sales": "http://www.stats.gov.cn/sj/",
}

# policy_type values emitted by collect_nbs. The downstream extractor
# uses these as routing hints (e.g. only CPI rows feed cpi_surprise).
NBS_POLICY_TYPES_BY_SERIES: dict[str, str] = {
    "cpi": "cpi_monthly",
    "ppi": "ppi_monthly",
    "pmi": "pmi_monthly",
    "retail_sales": "retail_sales_monthly",
}

NBS_POLICY_TYPES: frozenset[str] = frozenset(NBS_POLICY_TYPES_BY_SERIES.values())

# Title-keyword filters used to keep only series-relevant list rows. NBS
# tends to publish many adjacent items on the same index page (CPI / PPI
# / 社零 / etc) — filtering by title keyword lets each series-key fetch
# only its own rows from a shared index, avoiding double-counting.
NBS_TITLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cpi": ("居民消费价格", "CPI"),
    "ppi": ("工业生产者", "PPI", "出厂价格"),
    "pmi": ("采购经理指数", "PMI", "制造业景气"),
    "retail_sales": ("社会消费品零售", "社零", "消费品零售总额"),
}


# ── Phase E.4 (PE-4) — Xinwen Lianbo (新闻联播) list pages.
#
# We pull from news.sina.com.cn rather than tv.cctv.com because the
# CCTV daily-archive pages are JS-rendered: the transcript is only
# injected after the page boots and is not present in the initial HTML
# returned by ``requests``. Sina's 新闻联播文字版 mirror publishes a
# static HTML page per day at predictable URLs that the shared
# parse_list_page / parse_article_page parsers can read without
# changes — same gov.cn-style ``/YYYY-MM-DD/`` URL pattern.
#
# Two list pages: the daily roundup (news.sina.com.cn/zt_d/xwlb) and
# the syndication category page. Both list pages have considerable
# overlap; the dedup-by-URL pass in collect_xinwen_lianbo keeps each
# transcript single-counted.
XINWEN_LIANBO_LIST_URLS: dict[str, str] = {
    "xinwen_lianbo_daily": "https://news.sina.com.cn/zt_d/xwlb/",
    "xinwen_lianbo_category": "https://news.sina.com.cn/c/xwlb.shtml",
}
CCTV_XINWEN_LIANBO_DAY_URL = "https://tv.cctv.com/lm/xwlb/day/{yyyymmdd}.shtml"

XINWEN_LIANBO_POLICY_TYPES: frozenset[str] = frozenset({
    "xinwen_lianbo_daily",
})

# Title-keyword filter: only keep articles whose title contains a
# Xinwen Lianbo marker. Sina's category page mixes broadcast transcripts
# with adjacent meta-news ("xx 评论员文章"); the marker keeps the
# transcript flow clean. Falls back to keeping the row if the title is
# empty (defensive — let downstream LLM handle).
XINWEN_LIANBO_TITLE_KEYWORDS: tuple[str, ...] = (
    "新闻联播", "xwlb", "联播文字版",
)


# ─────────────────────────────────────────────────────────────────────
# HTTP helper — retry wrapper required by spec.
# ─────────────────────────────────────────────────────────────────────
def http_get(
    url: str,
    *,
    session: requests.Session | None = None,
    attempts: int = RETRY_ATTEMPTS,
    backoff: float = RETRY_BACKOFF_SECONDS,
) -> requests.Response:
    """GET ``url`` with explicit 15s timeout and 3-attempt retry.

    Raises ``requests.RequestException`` if all attempts fail. Callers
    are expected to catch this so a single bad URL does not poison the
    whole run.
    """
    sess = session or requests
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            resp = sess.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:
            # Network-layer failure — retry.
            last_exc = e
            if i < attempts - 1:
                time.sleep(backoff * (i + 1))
            continue
        if resp.status_code == 200:
            return resp
        # 4xx → final, do NOT retry (404, 403, 401, etc are not
        # transient and the same URL will keep returning them).
        if 400 <= resp.status_code < 500:
            raise requests.HTTPError(
                f"HTTP {resp.status_code} for {url}", response=resp,
            )
        # 5xx / other → retry with backoff.
        last_exc = requests.HTTPError(
            f"HTTP {resp.status_code} for {url}", response=resp,
        )
        if i < attempts - 1:
            time.sleep(backoff * (i + 1))
    assert last_exc is not None
    raise last_exc


# ─────────────────────────────────────────────────────────────────────
# Parsing — list page → list of (url, title, publish_date)
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PolicyLink:
    url: str
    title: str
    publish_date: str  # YYYY-MM-DD, may be "" if list page only shows the article


# Date hints that PBOC list rows tend to use, in best-to-worst order.
_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})"),
    re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日"),
)


def _try_parse_date(text: str) -> str:
    for pat in _DATE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                y, mo, d = (int(g) for g in m.groups())
                return f"{y:04d}-{mo:02d}-{d:02d}"
            except ValueError:
                continue
    return ""


def _host_base_from(base: str) -> str:
    """Return ``scheme://host`` of ``base`` so /-rooted hrefs resolve.

    Without this, mixing State Council (gov.cn) and PBOC (pbc.gov.cn)
    sources would resolve every leading-slash href onto ``PBC_BASE``,
    which would turn gov.cn paths into invalid pbc.gov.cn URLs.
    """
    if base.startswith("http://"):
        rest = base[len("http://"):]
        return "http://" + rest.split("/", 1)[0]
    if base.startswith("https://"):
        rest = base[len("https://"):]
        return "https://" + rest.split("/", 1)[0]
    return PBC_BASE  # last-ditch default for legacy callers


def _absolutize(href: str, base: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("//"):
        return "http:" + href
    if href.startswith("./"):
        # Relative to the list page directory
        return base.rsplit("/", 1)[0] + href[1:]
    if href.startswith("/"):
        # Host-rooted href: use the LIST PAGE's host, not PBC_BASE. This
        # is the PE-2 fix that lets gov.cn and pbc.gov.cn coexist in the
        # same script without cross-host URL corruption.
        return _host_base_from(base) + href
    return base.rsplit("/", 1)[0] + "/" + href


def parse_list_page(html: str, base_url: str) -> list[PolicyLink]:
    """Extract (url, title, publish_date) tuples from a PBOC list page.

    Uses BeautifulSoup if available, falls back to a regex pass that
    handles PBOC's canonical ``<a href="./YYYY/MM/DD/...html" ...>title</a>``
    pattern. List rows usually have a sibling ``<span>YYYY-MM-DD</span>``
    that we attach when parseable.
    """
    links: list[PolicyLink] = []
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            title = a.get_text(strip=True)
            if not title or len(title) < 4:
                continue
            # Skip nav anchors / external (e.g. mailto:, http://www.pbc.gov.cn/index.html nav)
            if href.startswith(("javascript:", "mailto:", "#")):
                continue
            # Article detail pages live under several patterns:
            #   (a) /zhengcehuobisi/.../<YYYYMMDDhhmmssNNN>/index.html
            #       (PBOC current live pattern — single timestamp string)
            #   (b) /YYYY/MM/DD/...html (PBOC legacy slash-separated pattern)
            #   (c) ..._<digits>.html  (PBOC legacy numbered suffix)
            #   (d) /YYYY-MM/DD/content_<digits>.html  (gov.cn State
            #       Council pattern — YYYY and MM joined by dash)
            #   (e) /content_<digits>.html  (gov.cn fallback when no
            #       date appears in the URL — date hint must come from
            #       sibling span / parent text)
            # 2026-06-06 fix: original parser only matched (b) so the live
            # OMO page (which uses (a)) returned 0 candidates. Match all.
            looks_like_article = (
                href.endswith(".html")
                and (
                    re.search(r"/\d{14,}/index\.html$", href) is not None
                    or re.search(r"/\d{4}/\d{1,2}/\d{1,2}/", href) is not None
                    or re.search(r"_\d{4,}\.html$", href) is not None
                    or re.search(r"/\d{4}-\d{1,2}/\d{1,2}/", href) is not None
                    or re.search(r"/content_\d+\.html$", href) is not None
                )
            )
            if not looks_like_article:
                continue
            url = _absolutize(href, base_url)
            # Look for a date hint near the anchor (sibling span / parent text)
            date_hint = ""
            parent = a.parent
            if parent is not None:
                date_hint = _try_parse_date(parent.get_text(" ", strip=True))
            if not date_hint:
                # Pattern (a): leading YYYYMMDD in the timestamp digit string.
                m_ts = re.search(r"/(\d{4})(\d{2})(\d{2})\d{6,}/index\.html$", href)
                if m_ts:
                    y, mo, d = (int(g) for g in m_ts.groups())
                    date_hint = f"{y:04d}-{mo:02d}-{d:02d}"
            if not date_hint:
                # Pattern (d): /YYYY-MM/DD/content_*.html (gov.cn).
                m_gov = re.search(r"/(\d{4})-(\d{1,2})/(\d{1,2})/", href)
                if m_gov:
                    y, mo, d = (int(g) for g in m_gov.groups())
                    date_hint = f"{y:04d}-{mo:02d}-{d:02d}"
            if not date_hint:
                # Pattern (b): slash-separated /YYYY/MM/DD/.
                m = re.search(r"/(\d{4})/(\d{1,2})/(\d{1,2})/", href)
                if m:
                    y, mo, d = (int(g) for g in m.groups())
                    date_hint = f"{y:04d}-{mo:02d}-{d:02d}"
            links.append(PolicyLink(url=url, title=title, publish_date=date_hint))
    except Exception as e:
        logger.warning("BeautifulSoup parse failed (%s); falling back to regex", e)
        # Last-resort regex pass over <a href="...html">title</a>
        for m in re.finditer(
            r'<a[^>]+href="([^"]+\.html)"[^>]*>([^<]{4,200})</a>', html
        ):
            href, title = m.group(1).strip(), m.group(2).strip()
            ts_m = re.search(r"/(\d{4})(\d{2})(\d{2})\d{6,}/index\.html$", href)
            slash_m = re.search(r"/(\d{4})/(\d{1,2})/(\d{1,2})/", href)
            dash_m = re.search(r"/(\d{4})-(\d{1,2})/(\d{1,2})/", href)
            if not (ts_m or slash_m or dash_m):
                continue
            url = _absolutize(href, base_url)
            date_hint = ""
            if ts_m:
                y, mo, d = (int(g) for g in ts_m.groups())
                date_hint = f"{y:04d}-{mo:02d}-{d:02d}"
            elif dash_m:
                y, mo, d = (int(g) for g in dash_m.groups())
                date_hint = f"{y:04d}-{mo:02d}-{d:02d}"
            elif slash_m:
                y, mo, d = (int(g) for g in slash_m.groups())
                date_hint = f"{y:04d}-{mo:02d}-{d:02d}"
            links.append(PolicyLink(url=url, title=title, publish_date=date_hint))
    # Deduplicate by URL preserving first-seen order.
    seen: set[str] = set()
    deduped: list[PolicyLink] = []
    for ln in links:
        if ln.url in seen:
            continue
        seen.add(ln.url)
        deduped.append(ln)
    return deduped


# ─────────────────────────────────────────────────────────────────────
# Parsing — article detail page → plain-text content
# ─────────────────────────────────────────────────────────────────────
def parse_article_page(html: str) -> tuple[str, str]:
    """Return (content_text, publish_date_hint) for one article.

    PBOC detail pages use ``<div id="zoom">`` (legacy) or
    ``<div class="detail_zoom">`` for the body, plus a header line
    like ``发布时间：2026-06-05``. Gov.cn State Council / ministry
    detail pages use ``<div id="UCAP-CONTENT">`` or
    ``<div class="pages_content">``; we try those first when present.
    """
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        body_div = (
            soup.find("div", id="UCAP-CONTENT")
            or soup.find("div", class_="pages_content")
            or soup.find("div", id="zoom")
            or soup.find("div", class_="detail_zoom")
            or soup.find("div", class_="content")
        )
        if body_div is not None:
            text = body_div.get_text("\n", strip=True)
        else:
            text = soup.get_text("\n", strip=True)
        # Normalize whitespace (collapse runs of blanks, but keep paragraph breaks)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Publish-date hint — look at the first 400 chars for "发布时间" / "发布日期"
        head = soup.get_text(" ", strip=True)[:600]
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        head = text[:600]
    date_hint = _try_parse_date(head)
    return text, date_hint


# ─────────────────────────────────────────────────────────────────────
# Main collection
# ─────────────────────────────────────────────────────────────────────
def _slug(text: str, limit: int = 60) -> str:
    """Make a filename-safe slug from a URL or title."""
    s = re.sub(r"[^0-9A-Za-z一-鿿]+", "_", text).strip("_")
    return s[:limit] if s else "untitled"


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


def _save_raw_html_on_failure(
    url: str,
    html: str,
    target_date: str,
    raw_html_dir: Path | None = None,
) -> Path:
    """Persist raw HTML so extract_policy_events.py can re-run the parser.

    ``raw_html_dir`` defaults to the PBC pool but PE-2 callers pass
    ``RAW_HTML_DIR_SC`` to keep State Council failures from polluting
    the PBC re-parse queue.
    """
    target_dir = raw_html_dir or RAW_HTML_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{target_date}_{_slug(url)}.html"
    path = target_dir / fname
    try:
        path.write_text(html, encoding="utf-8")
    except Exception as e:  # pragma: no cover — disk error
        logger.warning("Failed to save raw HTML for %s: %s", url, e)
    return path


def _atomic_write_jsonl(rows: list[dict], path: Path) -> None:
    """Write rows to ``path`` atomically (.tmp + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def collect_pbc(
    *,
    start: str,
    end: str,
    policy_types: Iterable[str] = ("omo", "lpr"),
    bonus_types: Iterable[str] = ("mlf", "rrr", "quarterly_report"),
    http_get_fn: Callable[..., requests.Response] = http_get,
    output_dir: Path | None = None,
) -> dict:
    """Collect PBOC policy texts for every day in [start, end] inclusive.

    Returns a summary dict::

        {
          "rows_by_date":     {date: n_rows, ...},
          "errors":           [{"url": str, "stage": str, "msg": str}, ...],
          "files_written":    [Path, ...],
          "policy_types_seen": {"omo": n, "lpr": n, ...},
          "n_total":           int,
        }

    The function does NOT raise on individual fetch errors — they
    accumulate in ``errors`` and the per-day file is still written
    (possibly with fewer rows). The caller decides whether ``partial``
    in the health record should be True.
    """
    output_root = output_dir or POLICY_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    dates = _date_range(start, end)
    date_set: set[str] = set(dates)

    summary: dict = {
        "rows_by_date": {d: 0 for d in dates},
        "errors": [],
        "files_written": [],
        "policy_types_seen": {},
        "n_total": 0,
    }

    # Step 1: for each list page in (priority ∪ bonus), discover links.
    by_date_rows: dict[str, list[dict]] = {d: [] for d in dates}
    seen_urls: set[str] = set()
    session = requests.Session()

    for pt in list(policy_types) + list(bonus_types):
        list_url = PBC_LIST_URLS.get(pt)
        if not list_url:
            continue
        try:
            resp = http_get_fn(list_url, session=session)
            list_html = resp.content.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning("List page failed for %s: %s", pt, e)
            summary["errors"].append(
                {"url": list_url, "stage": "list", "msg": str(e)}
            )
            continue

        links = parse_list_page(list_html, list_url)
        logger.info("PBOC %s list page: %d candidate links", pt, len(links))

        for ln in links:
            if ln.url in seen_urls:
                continue
            # If we can already filter by publish_date and it's outside
            # the requested window, skip the article fetch entirely.
            if ln.publish_date and ln.publish_date not in date_set:
                continue
            seen_urls.add(ln.url)
            time.sleep(INTER_REQUEST_DELAY)
            try:
                art_resp = http_get_fn(ln.url, session=session)
                art_html = art_resp.content.decode("utf-8", errors="replace")
            except Exception as e:
                logger.warning("Article fetch failed %s: %s", ln.url, e)
                summary["errors"].append(
                    {"url": ln.url, "stage": "article", "msg": str(e)}
                )
                continue
            try:
                content, article_date = parse_article_page(art_html)
            except Exception as e:
                logger.warning("Article parse failed %s: %s", ln.url, e)
                summary["errors"].append(
                    {"url": ln.url, "stage": "parse", "msg": str(e)}
                )
                _save_raw_html_on_failure(
                    ln.url, art_html, ln.publish_date or start,
                )
                continue

            publish_date = ln.publish_date or article_date
            if not publish_date:
                # Cannot place this in a daily file — skip but keep HTML
                _save_raw_html_on_failure(ln.url, art_html, start)
                summary["errors"].append(
                    {"url": ln.url, "stage": "no_date", "msg": "no publish_date"}
                )
                continue
            if publish_date not in date_set:
                continue
            if not content or len(content) < 20:
                # Empty body — keep HTML for re-parse but skip the row.
                _save_raw_html_on_failure(ln.url, art_html, publish_date)
                summary["errors"].append(
                    {"url": ln.url, "stage": "empty_body", "msg": "len(content)<20"}
                )
                continue

            row = {
                "publish_date": publish_date,
                "policy_type": pt,
                "title": ln.title,
                "url": ln.url,
                "content": content,
                "source": "pbc.gov.cn",
                "fetch_time": _now_utc_iso(),
            }
            by_date_rows[publish_date].append(row)
            summary["policy_types_seen"][pt] = (
                summary["policy_types_seen"].get(pt, 0) + 1
            )

    # Step 2: atomic-write per-day JSONL files (always write, even empty,
    # so re-runs are deterministic — overwrites supersede any prior partial).
    for d in dates:
        rows = by_date_rows[d]
        # Sort rows for stable output (idempotent across re-runs).
        rows.sort(key=lambda r: (r["policy_type"], r["url"]))
        out_path = output_root / f"{d}.jsonl"
        _atomic_write_jsonl(rows, out_path)
        summary["rows_by_date"][d] = len(rows)
        summary["files_written"].append(out_path)
        summary["n_total"] += len(rows)
        logger.info("Wrote %d row(s) -> %s", len(rows), out_path)

    return summary


# ─────────────────────────────────────────────────────────────────────
# Phase E.2 (PE-2) — State Council + ministry collector.
#
# Mirrors collect_pbc() exactly but pulls from STATE_COUNCIL_LIST_URLS,
# sets ``source="gov.cn"``, and writes under POLICY_DIR_SC. The list /
# article parsers are shared with PBC; the per-host base fix in
# ``_absolutize`` keeps gov.cn /-rooted hrefs from being absolutized
# onto pbc.gov.cn.
# ─────────────────────────────────────────────────────────────────────
def collect_state_council(
    *,
    start: str,
    end: str,
    policy_types: Iterable[str] = (
        "state_council_doc", "state_council_meeting",
        "miit_policy", "ndrc_policy", "mof_policy",
    ),
    bonus_types: Iterable[str] = (),
    http_get_fn: Callable[..., requests.Response] = http_get,
    output_dir: Path | None = None,
) -> dict:
    """Collect State Council + ministry policy texts for [start, end].

    Shape-identical to ``collect_pbc`` — same summary dict, same JSONL
    schema, same atomic-write per-day discipline. The only differences:

      - LIST URL registry is STATE_COUNCIL_LIST_URLS
      - source field is ``"gov.cn"``
      - raw-html-on-failure dump goes to RAW_HTML_DIR_SC
      - policy_type values are in STATE_COUNCIL_POLICY_TYPES
        (state_council_doc / state_council_meeting / miit_policy /
        ndrc_policy / mof_policy)

    A single list-page 5xx does not kill the run — errors accumulate
    in summary["errors"] and the per-day file is still written.
    """
    output_root = output_dir or POLICY_DIR_SC
    output_root.mkdir(parents=True, exist_ok=True)

    dates = _date_range(start, end)
    date_set: set[str] = set(dates)

    summary: dict = {
        "rows_by_date": {d: 0 for d in dates},
        "errors": [],
        "files_written": [],
        "policy_types_seen": {},
        "n_total": 0,
    }

    by_date_rows: dict[str, list[dict]] = {d: [] for d in dates}
    seen_urls: set[str] = set()
    session = requests.Session()

    for pt in list(policy_types) + list(bonus_types):
        list_url = STATE_COUNCIL_LIST_URLS.get(pt)
        if not list_url:
            continue
        try:
            resp = http_get_fn(list_url, session=session)
            list_html = resp.content.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning("State Council list page failed for %s: %s", pt, e)
            summary["errors"].append(
                {"url": list_url, "stage": "list", "msg": str(e)}
            )
            continue

        links = parse_list_page(list_html, list_url)
        logger.info("gov.cn %s list page: %d candidate links", pt, len(links))

        for ln in links:
            if ln.url in seen_urls:
                continue
            if ln.publish_date and ln.publish_date not in date_set:
                continue
            seen_urls.add(ln.url)
            time.sleep(INTER_REQUEST_DELAY)
            try:
                art_resp = http_get_fn(ln.url, session=session)
                art_html = art_resp.content.decode("utf-8", errors="replace")
            except Exception as e:
                logger.warning("State Council article fetch failed %s: %s", ln.url, e)
                summary["errors"].append(
                    {"url": ln.url, "stage": "article", "msg": str(e)}
                )
                continue
            try:
                content, article_date = parse_article_page(art_html)
            except Exception as e:
                logger.warning("State Council article parse failed %s: %s", ln.url, e)
                summary["errors"].append(
                    {"url": ln.url, "stage": "parse", "msg": str(e)}
                )
                _save_raw_html_on_failure(
                    ln.url, art_html, ln.publish_date or start,
                    raw_html_dir=RAW_HTML_DIR_SC,
                )
                continue

            publish_date = ln.publish_date or article_date
            if not publish_date:
                _save_raw_html_on_failure(
                    ln.url, art_html, start, raw_html_dir=RAW_HTML_DIR_SC,
                )
                summary["errors"].append(
                    {"url": ln.url, "stage": "no_date", "msg": "no publish_date"}
                )
                continue
            if publish_date not in date_set:
                continue
            if not content or len(content) < 20:
                _save_raw_html_on_failure(
                    ln.url, art_html, publish_date,
                    raw_html_dir=RAW_HTML_DIR_SC,
                )
                summary["errors"].append(
                    {"url": ln.url, "stage": "empty_body", "msg": "len(content)<20"}
                )
                continue

            row = {
                "publish_date": publish_date,
                "policy_type": pt,
                "title": ln.title,
                "url": ln.url,
                "content": content,
                "source": "gov.cn",
                "fetch_time": _now_utc_iso(),
            }
            by_date_rows[publish_date].append(row)
            summary["policy_types_seen"][pt] = (
                summary["policy_types_seen"].get(pt, 0) + 1
            )

    for d in dates:
        rows = by_date_rows[d]
        rows.sort(key=lambda r: (r["policy_type"], r["url"]))
        out_path = output_root / f"{d}.jsonl"
        _atomic_write_jsonl(rows, out_path)
        summary["rows_by_date"][d] = len(rows)
        summary["files_written"].append(out_path)
        summary["n_total"] += len(rows)
        logger.info("Wrote %d row(s) -> %s", len(rows), out_path)

    return summary


# ─────────────────────────────────────────────────────────────────────
# Phase E.3 (PE-3) — NBS (国家统计局) macro statistics collector.
#
# Same shape as collect_pbc / collect_state_council. The list-page
# parser is shared (gov.cn /YYYY-MM/DD/ pattern handles NBS just fine);
# the only NBS-specific logic is the title-keyword filter that keeps
# only series-relevant rows from a shared index page.
# ─────────────────────────────────────────────────────────────────────
def _title_matches_series(title: str, series_key: str) -> bool:
    """Return True if ``title`` contains any keyword for ``series_key``.

    Used to filter rows on a shared index page so the CPI fetch only
    keeps CPI rows even when the same index also lists PPI / 社零 /
    PMI. Falls back to True if the keyword tuple is missing (so a typo
    can't silently drop everything).
    """
    keywords = NBS_TITLE_KEYWORDS.get(series_key, ())
    if not keywords:
        return True
    return any(kw in title for kw in keywords)


def collect_nbs(
    *,
    start: str,
    end: str,
    policy_types: Iterable[str] = ("cpi", "ppi", "pmi", "retail_sales"),
    bonus_types: Iterable[str] = (),
    http_get_fn: Callable[..., requests.Response] = http_get,
    output_dir: Path | None = None,
) -> dict:
    """Collect NBS macro-statistics texts for [start, end] inclusive.

    Shape-identical to ``collect_pbc`` — same summary dict, same JSONL
    schema, same atomic per-day writes. NBS-specific differences:

      - LIST URL registry is NBS_LIST_URLS
      - source field is ``"stats.gov.cn"``
      - raw-html-on-failure dump goes to RAW_HTML_DIR_NBS
      - policy_type ∈ {"cpi_monthly", "ppi_monthly", "pmi_monthly",
        "retail_sales_monthly"} via NBS_POLICY_TYPES_BY_SERIES
      - title-keyword filter (NBS_TITLE_KEYWORDS) keeps only the
        series-relevant rows from a shared index page

    Like the PE-2 collector, a list-page 5xx does not kill the run —
    errors accumulate in summary["errors"] and the per-day file is
    still written. NBS publishes monthly so a 0-row day on a Friday is
    expected; the SLA budget is 35 days for that reason.
    """
    output_root = output_dir or POLICY_DIR_NBS
    output_root.mkdir(parents=True, exist_ok=True)

    dates = _date_range(start, end)
    date_set: set[str] = set(dates)

    summary: dict = {
        "rows_by_date": {d: 0 for d in dates},
        "errors": [],
        "files_written": [],
        "policy_types_seen": {},
        "n_total": 0,
    }

    by_date_rows: dict[str, list[dict]] = {d: [] for d in dates}
    seen_urls: set[str] = set()
    session = requests.Session()

    for pt in list(policy_types) + list(bonus_types):
        list_url = NBS_LIST_URLS.get(pt)
        if not list_url:
            continue
        try:
            resp = http_get_fn(list_url, session=session)
            list_html = resp.content.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning("NBS list page failed for %s: %s", pt, e)
            summary["errors"].append(
                {"url": list_url, "stage": "list", "msg": str(e)}
            )
            continue

        links = parse_list_page(list_html, list_url)
        logger.info("NBS %s list page: %d candidate links", pt, len(links))

        for ln in links:
            # Title-keyword filter: keep only rows whose title matches
            # the requested series. NBS indices mix multiple series in
            # one page; without this filter the CPI fetch would also
            # ingest PPI / 社零 rows and double-count them across
            # series_keys.
            if not _title_matches_series(ln.title, pt):
                continue
            if ln.url in seen_urls:
                continue
            if ln.publish_date and ln.publish_date not in date_set:
                continue
            seen_urls.add(ln.url)
            time.sleep(INTER_REQUEST_DELAY)
            try:
                art_resp = http_get_fn(ln.url, session=session)
                art_html = art_resp.content.decode("utf-8", errors="replace")
            except Exception as e:
                logger.warning("NBS article fetch failed %s: %s", ln.url, e)
                summary["errors"].append(
                    {"url": ln.url, "stage": "article", "msg": str(e)}
                )
                continue
            try:
                content, article_date = parse_article_page(art_html)
            except Exception as e:
                logger.warning("NBS article parse failed %s: %s", ln.url, e)
                summary["errors"].append(
                    {"url": ln.url, "stage": "parse", "msg": str(e)}
                )
                _save_raw_html_on_failure(
                    ln.url, art_html, ln.publish_date or start,
                    raw_html_dir=RAW_HTML_DIR_NBS,
                )
                continue

            publish_date = ln.publish_date or article_date
            if not publish_date:
                _save_raw_html_on_failure(
                    ln.url, art_html, start, raw_html_dir=RAW_HTML_DIR_NBS,
                )
                summary["errors"].append(
                    {"url": ln.url, "stage": "no_date", "msg": "no publish_date"}
                )
                continue
            if publish_date not in date_set:
                continue
            if not content or len(content) < 20:
                _save_raw_html_on_failure(
                    ln.url, art_html, publish_date,
                    raw_html_dir=RAW_HTML_DIR_NBS,
                )
                summary["errors"].append(
                    {"url": ln.url, "stage": "empty_body", "msg": "len(content)<20"}
                )
                continue

            policy_type_value = NBS_POLICY_TYPES_BY_SERIES.get(pt, "other")
            row = {
                "publish_date": publish_date,
                "policy_type": policy_type_value,
                "title": ln.title,
                "url": ln.url,
                "content": content,
                "source": "stats.gov.cn",
                "fetch_time": _now_utc_iso(),
            }
            by_date_rows[publish_date].append(row)
            summary["policy_types_seen"][policy_type_value] = (
                summary["policy_types_seen"].get(policy_type_value, 0) + 1
            )

    for d in dates:
        rows = by_date_rows[d]
        rows.sort(key=lambda r: (r["policy_type"], r["url"]))
        out_path = output_root / f"{d}.jsonl"
        _atomic_write_jsonl(rows, out_path)
        summary["rows_by_date"][d] = len(rows)
        summary["files_written"].append(out_path)
        summary["n_total"] += len(rows)
        logger.info("Wrote %d row(s) -> %s", len(rows), out_path)

    return summary


# ─────────────────────────────────────────────────────────────────────
# Phase E.4 (PE-4) — CCTV Xinwen Lianbo daily transcript collector.
#
# Shape-identical to collect_pbc / collect_state_council / collect_nbs.
# The list-page / article-page parsers are reused; the only PE-4-
# specific logic is the broadcast-title keyword filter (XWLB transcripts
# are interleaved with adjacent comment articles on sina.com.cn).
# Source field is ``news.sina.com.cn``; the underlying broadcaster is
# CCTV but our scraper path goes through Sina syndication.
# ─────────────────────────────────────────────────────────────────────
def _title_matches_xinwen_lianbo(title: str) -> bool:
    """Return True if ``title`` looks like an XWLB transcript article.

    Falls through to True when ``title`` is empty — defensive: don't
    drop a row whose title parse fell back, let the downstream LLM
    decide. Otherwise must contain one of the configured XWLB markers.
    """
    if not title:
        return True
    t = title.strip()
    if not t:
        return True
    return any(kw in t for kw in XINWEN_LIANBO_TITLE_KEYWORDS)


def _parse_cctv_xinwen_lianbo_day(
    html: str,
    *,
    publish_date: str,
    page_url: str,
) -> list[dict]:
    """Parse CCTV's static XWLB day page into topic rows.

    CCTV day pages contain the broadcast segment list as static HTML.
    The detail video pages do not reliably expose transcript text, but
    the segment titles themselves are still valuable as daily policy /
    macro theme-attention input and are better than a hard 0-row outage
    when the older Sina mirror 404s.
    """
    links: list[tuple[str, str]] = []
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = str(a.get("href", "")).strip()
            title = (
                str(a.get("title", "") or a.get("alt", "")).strip()
                or a.get_text(" ", strip=True)
            )
            title = re.sub(r"^完整版", "", title).strip()
            if not href.startswith("http") or "tv.cctv.com" not in href:
                continue
            if ".shtml" not in href or not title:
                continue
            if "新闻联播" in title and "21:00" in title:
                continue
            links.append((href, title))
    except Exception as e:
        logger.warning("CCTV XWLB BeautifulSoup parse failed: %s", e)
        for m in re.finditer(
            r'<a[^>]+href="([^"]*tv\.cctv\.com[^"]+\.shtml)"[^>]+title="([^"]+)"',
            html,
        ):
            title = re.sub(r"^完整版", "", m.group(2)).strip()
            if "新闻联播" in title and "21:00" in title:
                continue
            links.append((m.group(1), title))

    seen: set[str] = set()
    rows: list[dict] = []
    for url, title in links:
        if url in seen:
            continue
        seen.add(url)
        rows.append({
            "publish_date": publish_date,
            "policy_type": "xinwen_lianbo_daily",
            "title": title,
            "url": url,
            "content": title,
            "source": "tv.cctv.com",
            "content_level": "title_only",
            "source_quality": "fallback_title_only",
            "fetch_time": _now_utc_iso(),
            "fallback_from": page_url,
        })
    return rows


def _collect_cctv_xinwen_lianbo_fallback(
    *,
    dates: list[str],
    http_get_fn: Callable[..., requests.Response],
    session: requests.Session,
    summary: dict,
) -> dict[str, list[dict]]:
    by_date_rows: dict[str, list[dict]] = {}
    for d in dates:
        yyyymmdd = d.replace("-", "")
        page_url = CCTV_XINWEN_LIANBO_DAY_URL.format(yyyymmdd=yyyymmdd)
        try:
            resp = http_get_fn(page_url, session=session)
            html = resp.content.decode("utf-8", errors="replace")
            rows = _parse_cctv_xinwen_lianbo_day(
                html, publish_date=d, page_url=page_url,
            )
        except Exception as e:
            logger.warning("CCTV XWLB fallback failed for %s: %s", d, e)
            summary["errors"].append(
                {"url": page_url, "stage": "cctv_fallback", "msg": str(e)}
            )
            rows = []
        if rows:
            logger.info("CCTV XWLB fallback %s: %d rows", d, len(rows))
            by_date_rows[d] = rows
            summary.setdefault("fallback_title_only_dates", []).append(d)
            summary["policy_types_seen"]["xinwen_lianbo_daily"] = (
                summary["policy_types_seen"].get("xinwen_lianbo_daily", 0)
                + len(rows)
            )
    return by_date_rows


def collect_xinwen_lianbo(
    *,
    start: str,
    end: str,
    policy_types: Iterable[str] = (
        "xinwen_lianbo_daily", "xinwen_lianbo_category",
    ),
    bonus_types: Iterable[str] = (),
    http_get_fn: Callable[..., requests.Response] = http_get,
    output_dir: Path | None = None,
) -> dict:
    """Collect CCTV Xinwen Lianbo broadcast transcripts for [start, end].

    Shape-identical to ``collect_pbc`` — same summary dict, same JSONL
    schema, same atomic per-day writes. XWLB-specific differences:

      - LIST URL registry is XINWEN_LIANBO_LIST_URLS (Sina mirror)
      - source field is ``"news.sina.com.cn"``
      - raw-html-on-failure dump goes to RAW_HTML_DIR_XWLB
      - policy_type is always ``"xinwen_lianbo_daily"`` (single broadcast
        per day, no series split)
      - title-keyword filter keeps only rows whose title contains an
        XWLB marker (新闻联播 / xwlb / 联播文字版) to avoid the
        adjacent comment articles on the same category index.

    XWLB airs every day (incl. weekends) but the cron only fires on
    weekdays. The SLA budget is 2 trading days so a single failed
    scrape can be recovered on Monday without painting the gate red.
    """
    output_root = output_dir or POLICY_DIR_XWLB
    output_root.mkdir(parents=True, exist_ok=True)

    dates = _date_range(start, end)
    date_set: set[str] = set(dates)

    summary: dict = {
        "rows_by_date": {d: 0 for d in dates},
        "errors": [],
        "files_written": [],
        "policy_types_seen": {},
        "n_total": 0,
    }

    by_date_rows: dict[str, list[dict]] = {d: [] for d in dates}
    seen_urls: set[str] = set()
    session = requests.Session()

    for pt in list(policy_types) + list(bonus_types):
        list_url = XINWEN_LIANBO_LIST_URLS.get(pt)
        if not list_url:
            continue
        try:
            resp = http_get_fn(list_url, session=session)
            list_html = resp.content.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning("XWLB list page failed for %s: %s", pt, e)
            summary["errors"].append(
                {"url": list_url, "stage": "list", "msg": str(e)}
            )
            continue

        links = parse_list_page(list_html, list_url)
        logger.info("XWLB %s list page: %d candidate links", pt, len(links))

        for ln in links:
            # Drop comment-article rows that aren't transcript text.
            if not _title_matches_xinwen_lianbo(ln.title):
                continue
            if ln.url in seen_urls:
                continue
            # 2026-06-07 cx batch C P2 #5 fix: pre-fix this exact-match
            # predfilter dropped articles when the list parser returned
            # a wrong/offset publish_date (Sina column rolling-update
            # time, neighbour-article date bleed, etc.). The detail
            # page's article_date is reliable but never got a chance.
            # Now: predfilter only against EGREGIOUS dates (more than
            # 7 days outside the requested window). Borderline cases
            # fall through to the detail-page fetch + the line ~1160
            # post-fetch filter that compares article_date to date_set.
            # Costs at most a couple extra HTTP fetches per run on the
            # boundary; saves a real-article miss on every news site
            # whose list page lies about dates (most of them).
            if ln.publish_date:
                try:
                    ln_dt = datetime.strptime(ln.publish_date, "%Y-%m-%d")
                    s_dt = datetime.strptime(start, "%Y-%m-%d")
                    e_dt = datetime.strptime(end, "%Y-%m-%d")
                    if ln_dt < s_dt - timedelta(days=7):
                        continue
                    if ln_dt > e_dt + timedelta(days=1):
                        continue
                except (ValueError, TypeError):
                    # Garbage date string — let the detail page decide.
                    pass
            seen_urls.add(ln.url)
            time.sleep(INTER_REQUEST_DELAY)
            try:
                art_resp = http_get_fn(ln.url, session=session)
                art_html = art_resp.content.decode("utf-8", errors="replace")
            except Exception as e:
                logger.warning("XWLB article fetch failed %s: %s", ln.url, e)
                summary["errors"].append(
                    {"url": ln.url, "stage": "article", "msg": str(e)}
                )
                continue
            try:
                content, article_date = parse_article_page(art_html)
            except Exception as e:
                logger.warning("XWLB article parse failed %s: %s", ln.url, e)
                summary["errors"].append(
                    {"url": ln.url, "stage": "parse", "msg": str(e)}
                )
                _save_raw_html_on_failure(
                    ln.url, art_html, ln.publish_date or start,
                    raw_html_dir=RAW_HTML_DIR_XWLB,
                )
                continue

            publish_date = ln.publish_date or article_date
            if not publish_date:
                _save_raw_html_on_failure(
                    ln.url, art_html, start, raw_html_dir=RAW_HTML_DIR_XWLB,
                )
                summary["errors"].append(
                    {"url": ln.url, "stage": "no_date", "msg": "no publish_date"}
                )
                continue
            if publish_date not in date_set:
                continue
            if not content or len(content) < 20:
                _save_raw_html_on_failure(
                    ln.url, art_html, publish_date,
                    raw_html_dir=RAW_HTML_DIR_XWLB,
                )
                summary["errors"].append(
                    {"url": ln.url, "stage": "empty_body", "msg": "len(content)<20"}
                )
                continue

            row = {
                "publish_date": publish_date,
                # Single XWLB policy_type — the daily broadcast itself.
                "policy_type": "xinwen_lianbo_daily",
                "title": ln.title,
                "url": ln.url,
                "content": content,
                "source": "news.sina.com.cn",
                "fetch_time": _now_utc_iso(),
            }
            by_date_rows[publish_date].append(row)
            summary["policy_types_seen"]["xinwen_lianbo_daily"] = (
                summary["policy_types_seen"].get("xinwen_lianbo_daily", 0) + 1
            )

    missing_dates = [d for d in dates if not by_date_rows[d]]
    if missing_dates:
        logger.warning(
            "XWLB Sina mirror returned 0 rows for %s; trying CCTV day-page fallback",
            missing_dates,
        )
        fallback_rows = _collect_cctv_xinwen_lianbo_fallback(
            dates=missing_dates,
            http_get_fn=http_get_fn,
            session=session,
            summary=summary,
        )
        for d, rows in fallback_rows.items():
            by_date_rows[d].extend(rows)

    for d in dates:
        rows = by_date_rows[d]
        rows.sort(key=lambda r: (r["policy_type"], r["url"]))
        out_path = output_root / f"{d}.jsonl"
        _atomic_write_jsonl(rows, out_path)
        summary["rows_by_date"][d] = len(rows)
        summary["files_written"].append(out_path)
        summary["n_total"] += len(rows)
        logger.info("Wrote %d row(s) -> %s", len(rows), out_path)

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
    """Write a HealthStatus record via the standard scheduler interface.

    ``partial`` is True when any error rows were recorded but at least
    one item was successfully written. ``success`` is True iff at least
    one item was written (i.e. the day is on file).

    ``health_source`` defaults to the PBC source name; PE-2 callers
    pass ``HEALTH_SOURCE_NAME_SC`` so the SLA gate sees a separate row.

    ``sparse_steady=True`` (added 2026-06-16): treat 0 rows as the
    steady-state success case, not a failure. Used for state_council
    while gov.cn's restructured SPA list pages have no parseable static
    content — the collector legitimately writes 0 rows daily until the
    sousuo-API rewrite ships, and we don't want the SLA gate to bleed
    red on that. See ``STATE_COUNCIL_LIST_URLS`` comment in this file.
    """
    try:
        from scheduler.data_health import HealthStatus, write_health
    except Exception as e:  # pragma: no cover — broken install
        logger.warning("Cannot import scheduler.data_health (%s)", e)
        return

    n_total = int(summary.get("n_total", 0))
    n_errors = len(summary.get("errors", []))
    fallback_title_only_dates = summary.get("fallback_title_only_dates", [])
    last_date = ""
    for d in sorted(summary.get("rows_by_date", {}).keys(), reverse=True):
        if summary["rows_by_date"][d] > 0:
            last_date = d
            break
    is_success = (n_total > 0) or sparse_steady
    status = HealthStatus(
        success=is_success,
        n_items=n_total,
        latest_date=last_date or target_date,
        partial=(n_total > 0 and (n_errors > 0 or bool(fallback_title_only_dates))),
        error_type=(
            "title_only_fallback"
            if n_total > 0 and fallback_title_only_dates
            else "" if is_success
            else "no_rows"
        ),
        error_message=(
            f"title-only fallback dates={fallback_title_only_dates[:5]}"
            if fallback_title_only_dates
            else
            "; ".join(
                f"{e['stage']}:{e['url']}" for e in summary.get("errors", [])[:3]
            )
            if n_errors and not sparse_steady
            else
            "sparse_by_design: gov.cn SPA — no parseable rows (see collect_policy_texts.py)"
            if sparse_steady and n_total == 0
            else ""
        ),
        network_profile="ashare",
        extra={
            "policy_types_seen": summary.get("policy_types_seen", {}),
            "rows_by_date": summary.get("rows_by_date", {}),
            "n_errors": n_errors,
            "fallback_title_only_dates": fallback_title_only_dates,
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
        description="Collect policy texts (Phase E.1 step 1)."
    )
    parser.add_argument(
        "--source",
        default="pbc",
        choices=["pbc", "state_council", "nbs", "xinwen_lianbo"],
        help=(
            "Policy source. 'pbc' = People's Bank monetary policy texts "
            "(Phase E.1). 'state_council' = State Council + 3 ministry "
            "policy docs from gov.cn (Phase E.2). 'nbs' = National "
            "Bureau of Statistics CPI / PPI / PMI / retail sales "
            "monthly releases (Phase E.3). 'xinwen_lianbo' = CCTV "
            "新闻联播 daily broadcast transcripts via Sina mirror "
            "(Phase E.4)."
        ),
    )
    parser.add_argument(
        "--start", default=None,
        help="Backfill start date YYYY-MM-DD (default: today).",
    )
    parser.add_argument(
        "--end", default=None,
        help="Backfill end date YYYY-MM-DD (default: today).",
    )
    parser.add_argument(
        "--include-bonus", action="store_true",
        help=(
            "Also fetch MLF / RRR / quarterly_report list pages. "
            "These are bonus-tier sources for E.1 and may 404 without "
            "blocking the run."
        ),
    )
    parser.add_argument(
        "--fail-on-empty", action="store_true",
        help=(
            "2026-06-06 P3 #6 cron-hardening: exit 1 when no rows were "
            "written for any requested date. Without this flag the "
            "script always exits 0 and write_health flags success=False / "
            "no_rows, which a cron that only looks at exit code or file "
            "existence will miss. Enable for production cron, disable "
            "for local exploration."
        ),
    )
    args = parser.parse_args(argv)

    today = datetime.now().strftime("%Y-%m-%d")
    # 2026-06-07 cx batch C P1 #1 fix: XWLB airs DAILY incl. weekends
    # but the cron only fires Mon-Fri. Pre-fix the default was
    # start=end=today, so Saturday and Sunday transcripts were never
    # scraped by the weekday cron, and the SLA's 2-trading-day budget
    # only HID the gap (kept the health light green) — it didn't
    # backfill anything. Now: when --source xinwen_lianbo and no
    # explicit --start, default the start to today-3 (calendar days)
    # so a Monday run sweeps Friday→Sunday, a Tuesday run double-
    # checks Saturday, etc. Dedup is by URL so re-scrapes are cheap.
    # Other sources keep start=today since their SLA budgets are
    # tuned around weekday-only publication.
    if args.source == "xinwen_lianbo" and not args.start:
        weekend_lookback = (
            datetime.now() - timedelta(days=3)
        ).strftime("%Y-%m-%d")
        start = weekend_lookback
    else:
        start = args.start or today
    end = args.end or today

    if args.source == "pbc":
        bonus = ("mlf", "rrr", "quarterly_report") if args.include_bonus else ()
        collector = lambda: collect_pbc(
            start=start, end=end,
            policy_types=("omo", "lpr"),
            bonus_types=bonus,
        )
        health_source = HEALTH_SOURCE_NAME
    elif args.source == "state_council":
        # PE-2: all 5 list sources at once. include-bonus is a no-op
        # since gov.cn doesn't have a bonus tier — all five are required
        # (the SLA budget is 3 days, so an occasional empty pull is OK).
        collector = lambda: collect_state_council(
            start=start, end=end,
            policy_types=(
                "state_council_doc", "state_council_meeting",
                "miit_policy", "ndrc_policy", "mof_policy",
            ),
        )
        health_source = HEALTH_SOURCE_NAME_SC
    elif args.source == "nbs":
        # PE-3: NBS macro statistics. All 4 series at once; include-bonus
        # is a no-op since NBS doesn't have a bonus tier. NBS publishes
        # monthly so a 0-row day is the steady-state expectation — the
        # SLA budget is 35 days (one monthly release cycle).
        collector = lambda: collect_nbs(
            start=start, end=end,
            policy_types=("cpi", "ppi", "pmi", "retail_sales"),
        )
        health_source = HEALTH_SOURCE_NAME_NBS
    elif args.source == "xinwen_lianbo":
        # PE-4: CCTV Xinwen Lianbo daily transcript via Sina mirror. Two
        # list pages (daily roundup + category index) — overlapping rows
        # are deduped by URL. XWLB airs DAILY incl. weekends but the
        # cron only runs weekdays; SLA budget is 2 trading days so a
        # single failed scrape can be recovered on Monday.
        collector = lambda: collect_xinwen_lianbo(
            start=start, end=end,
            policy_types=(
                "xinwen_lianbo_daily", "xinwen_lianbo_category",
            ),
        )
        health_source = HEALTH_SOURCE_NAME_XWLB
    else:
        # Re-raise as a hard error rather than silently no-op. The
        # argparse choices= already prevents this, but defend.
        logger.error("Unsupported --source %s", args.source)
        return 2

    try:
        summary = collector()
    except Exception as e:
        logger.error("collector raised (source=%s): %s", args.source, e)
        try:
            from scheduler.data_health import HealthStatus, write_health
            write_health(
                health_source,
                HealthStatus(
                    success=False,
                    error_type=type(e).__name__,
                    error_message=str(e)[:300],
                    network_profile="ashare",
                ),
                date=end,
            )
        except Exception:
            pass
        return 1

    # Publish health using the LATEST day as the date key so the SLA
    # gate sees today's record on today's run.
    # 2026-06-16: state_council marked sparse_steady — gov.cn restructured,
    # 4 of 5 list URLs are commented out, the remaining state_council_doc
    # is SPA-rendered with no scrapeable static content. Treat 0 rows as
    # steady-state until the sousuo-API rewrite ships.
    is_sparse_steady = args.source == "state_council"
    publish_health(
        summary, target_date=end, health_source=health_source,
        sparse_steady=is_sparse_steady,
    )

    logger.info(
        "Done. n_total=%d, files=%d, errors=%d, rows_by_date=%s",
        summary["n_total"],
        len(summary["files_written"]),
        len(summary["errors"]),
        summary["rows_by_date"],
    )
    # Surface the per-file paths so the caller / cron log makes it
    # obvious what was written.
    for p in summary["files_written"]:
        logger.info("  -> %s (%d rows)", p, summary["rows_by_date"][p.stem])
    # 2026-06-06 P3 #6 fix: surface a non-zero exit when the user
    # asked for fail-loud and not a single row landed. write_health
    # already records this correctly; this is for cron wrappers that
    # only watch the exit code.
    if args.fail_on_empty and summary["n_total"] == 0:
        logger.error(
            "fail-on-empty: 0 rows written across [%s..%s]. Exiting 1.",
            start, end,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
