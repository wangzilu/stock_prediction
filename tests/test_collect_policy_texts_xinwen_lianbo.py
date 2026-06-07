"""Tests for scripts/collect_policy_texts.py --source xinwen_lianbo.

Phase E.4 (PE-4) step 1 — mirror of test_collect_policy_texts_nbs.py for
the CCTV Xinwen Lianbo (新闻联播) theme-attention branch. Covers:

  1. URL pattern fan-out: both registered list pages (daily roundup +
     category index) are fetched, deduped by URL, and the gov.cn-style
     ``/YYYY-MM-DD/`` URL pattern is parsed by the shared
     parse_list_page.
  2. JSONL schema lock: same 7 documented keys as PBC/PE-2/PE-3; source
     field is ``news.sina.com.cn``; policy_type is exactly
     ``xinwen_lianbo_daily``.
  3. Idempotent re-run: two runs over the same date produce identical
     rows; no .tmp leftover.
  4. /-rooted href cross-host safety: a Sina /-prefixed href must
     absolutize onto news.sina.com.cn, NOT pbc.gov.cn / gov.cn.
  5. Transcript title filter unit test — XWLB markers keep transcript
     rows, adjacent comment-article rows are dropped.

All tests are HTTP-mocked. We never depend on Sina being reachable.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import requests

from scripts import collect_policy_texts as ctp


# ─────────────────────────────────────────────────────────────────────
# Fixtures — minimal valid XWLB list / detail HTML.
# Each list page mixes a transcript row with an off-topic comment row
# so we can also exercise the title-keyword filter.
# ─────────────────────────────────────────────────────────────────────
DAILY_LIST_HTML = """
<html><body>
<ul class="list">
  <li>
    <a href="/zt_d/xwlb/2026-06-05/content_1001.html">
      新闻联播文字版 2026年6月5日
    </a>
    <span>2026-06-05</span>
  </li>
  <li>
    <a href="/zt_d/xwlb/2026-06-04/content_1002.html">
      新闻联播文字版 2026年6月4日
    </a>
    <span>2026-06-04</span>
  </li>
  <li>
    <a href="/zt_d/comment/2026-06-05/content_9999.html">
      央视评论员文章 (NOT a transcript)
    </a>
    <span>2026-06-05</span>
  </li>
  <li><a href="javascript:void(0)">nav anchor</a></li>
</ul>
</body></html>
"""

CATEGORY_LIST_HTML = """
<html><body>
<a href="/c/xwlb/2026-06-05/content_2001.html">
  新闻联播 2026年6月5日 完整版
</a>
<span>2026-06-05</span>
<a href="/c/comment/2026-06-05/content_9998.html">
  时评：央视访谈节选
</a>
<span>2026-06-05</span>
</body></html>
"""

DAILY_DETAIL_HTML = """
<html><body>
<div class="header">发布时间：2026-06-05 19:30</div>
<div class="pages_content">
  <p>各位观众晚上好，今天是2026年6月5日，星期五。</p>
  <p>1. 习近平主持中央政治局会议，部署半导体自立自强工作。</p>
  <p>2. 国务院常务会议研究扩大内需若干举措。</p>
  <p>3. 工信部公布机器人产业发展规划。</p>
  <p>4. 一带一路高峰论坛在京举行。</p>
</div>
</body></html>
"""

DETAIL_2026_06_04_HTML = """
<html><body>
<div id="UCAP-CONTENT">
  <p>各位观众晚上好。今天联播主要内容：央行宣布支持民营经济若干措施。
  另外，新能源装机突破历史新高。</p>
</div>
</body></html>
"""

CATEGORY_DETAIL_HTML = """
<html><body>
<div class="pages_content">
  <p>新闻联播 6月5日完整文字稿：习近平主持中央政治局会议……
  央行降准消息引发关注。</p>
</div>
</body></html>
"""


def _mock_response(status: int, body: str) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.content = body.encode("utf-8")
    resp.text = body
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            f"HTTP {status}", response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _make_http_get(url_to_response: dict[str, MagicMock]):
    """Build a fake ``http_get`` that returns canned responses by URL."""

    def fake_http_get(url, *, session=None, attempts=3, backoff=0.0):
        if url not in url_to_response:
            raise requests.HTTPError(f"unmocked URL: {url}")
        resp = url_to_response[url]
        if resp.status_code >= 400:
            raise requests.HTTPError(f"HTTP {resp.status_code} for {url}")
        return resp

    return fake_http_get


def _abs(list_url: str, href: str) -> str:
    """Helper to absolutize a /-rooted href via the production helper."""
    return ctp._absolutize(href, list_url)


# ─────────────────────────────────────────────────────────────────────
# Test 1 — URL pattern fan-out across both list pages.
# ─────────────────────────────────────────────────────────────────────
def test_url_pattern_fan_out_across_both_list_pages(tmp_path: Path):
    """Both registered LIST URLs are queried; transcript URLs from each
    are absolutized correctly and the transcripts are written under
    their publish_date.
    """
    daily_url = ctp.XINWEN_LIANBO_LIST_URLS["xinwen_lianbo_daily"]
    cat_url = ctp.XINWEN_LIANBO_LIST_URLS["xinwen_lianbo_category"]

    url_map = {
        daily_url: _mock_response(200, DAILY_LIST_HTML),
        cat_url: _mock_response(200, CATEGORY_LIST_HTML),
        # Daily list transcripts
        _abs(daily_url, "/zt_d/xwlb/2026-06-05/content_1001.html"):
            _mock_response(200, DAILY_DETAIL_HTML),
        _abs(daily_url, "/zt_d/xwlb/2026-06-04/content_1002.html"):
            _mock_response(200, DETAIL_2026_06_04_HTML),
        # Category list transcript — different URL, same broadcast day.
        _abs(cat_url, "/c/xwlb/2026-06-05/content_2001.html"):
            _mock_response(200, CATEGORY_DETAIL_HTML),
    }

    summary = ctp.collect_xinwen_lianbo(
        start="2026-06-04", end="2026-06-05",
        http_get_fn=_make_http_get(url_map),
        output_dir=tmp_path,
    )

    assert summary["n_total"] >= 2, summary
    assert "xinwen_lianbo_daily" in summary["policy_types_seen"]
    # Each window day has a file written.
    for d in ("2026-06-04", "2026-06-05"):
        assert (tmp_path / f"{d}.jsonl").exists(), d


# ─────────────────────────────────────────────────────────────────────
# Test 2 — JSONL schema lock.
# ─────────────────────────────────────────────────────────────────────
def test_jsonl_schema_matches_contract_with_sina_source(tmp_path: Path):
    """Every JSONL row has the 7 documented keys and nothing else.
    source is ``news.sina.com.cn`` and policy_type is
    ``xinwen_lianbo_daily``.
    """
    daily_url = ctp.XINWEN_LIANBO_LIST_URLS["xinwen_lianbo_daily"]
    cat_url = ctp.XINWEN_LIANBO_LIST_URLS["xinwen_lianbo_category"]
    url_map = {
        daily_url: _mock_response(200, DAILY_LIST_HTML),
        cat_url: _mock_response(200, "<html></html>"),
        _abs(daily_url, "/zt_d/xwlb/2026-06-05/content_1001.html"):
            _mock_response(200, DAILY_DETAIL_HTML),
    }
    summary = ctp.collect_xinwen_lianbo(
        start="2026-06-05", end="2026-06-05",
        http_get_fn=_make_http_get(url_map),
        output_dir=tmp_path,
    )
    assert summary["n_total"] >= 1, summary
    expected_keys = {
        "publish_date", "policy_type", "title", "url",
        "content", "source", "fetch_time",
    }
    rows = [
        json.loads(ln)
        for ln in (tmp_path / "2026-06-05.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if ln.strip()
    ]
    assert rows, "expected at least one row on 2026-06-05"
    for row in rows:
        assert set(row.keys()) == expected_keys, sorted(row.keys())
        assert row["source"] == "news.sina.com.cn", (
            "PE-4 rows must source=news.sina.com.cn so downstream can "
            "split PBC vs SC vs NBS vs XWLB by source field."
        )
        assert row["policy_type"] == "xinwen_lianbo_daily", row["policy_type"]
        assert row["fetch_time"].endswith("Z")
        assert len(row["content"]) >= 20


# ─────────────────────────────────────────────────────────────────────
# Test 3 — Idempotent re-run does not duplicate rows.
# ─────────────────────────────────────────────────────────────────────
def test_idempotent_rerun_does_not_duplicate_rows(tmp_path: Path):
    """Two runs over the same date produce identical rows. No append."""
    daily_url = ctp.XINWEN_LIANBO_LIST_URLS["xinwen_lianbo_daily"]
    cat_url = ctp.XINWEN_LIANBO_LIST_URLS["xinwen_lianbo_category"]
    url_map = {
        daily_url: _mock_response(200, DAILY_LIST_HTML),
        cat_url: _mock_response(200, "<html></html>"),
        _abs(daily_url, "/zt_d/xwlb/2026-06-05/content_1001.html"):
            _mock_response(200, DAILY_DETAIL_HTML),
    }
    fake = _make_http_get(url_map)

    s1 = ctp.collect_xinwen_lianbo(
        start="2026-06-05", end="2026-06-05",
        http_get_fn=fake, output_dir=tmp_path,
    )
    out = tmp_path / "2026-06-05.jsonl"
    assert out.exists()
    lines1 = [
        json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    s2 = ctp.collect_xinwen_lianbo(
        start="2026-06-05", end="2026-06-05",
        http_get_fn=fake, output_dir=tmp_path,
    )
    lines2 = [
        json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    def _strip(rows: list[dict]) -> list[dict]:
        return [{k: v for k, v in r.items() if k != "fetch_time"} for r in rows]

    assert _strip(lines1) == _strip(lines2)
    assert s1["n_total"] == s2["n_total"] == len(lines1)
    assert not list(tmp_path.glob("*.tmp"))


# ─────────────────────────────────────────────────────────────────────
# Test 4 — /-rooted href cross-host safety.
# ─────────────────────────────────────────────────────────────────────
def test_slash_rooted_href_resolves_to_sina_host_not_pbc_or_govcn():
    """A /-rooted href from a Sina list page must absolutize onto
    news.sina.com.cn, not www.pbc.gov.cn or www.gov.cn. Regression
    guard for the legacy ``_absolutize`` that hardcoded PBC_BASE.
    """
    base = "https://news.sina.com.cn/zt_d/xwlb/"
    href = "/zt_d/xwlb/2026-06-05/content_1001.html"
    resolved = ctp._absolutize(href, base)
    assert resolved.startswith("https://news.sina.com.cn/"), (
        f"/-rooted href must resolve onto the LIST PAGE host. got {resolved}"
    )
    assert "pbc.gov.cn" not in resolved
    assert "www.gov.cn/" not in resolved
    # PBC code path still works (no regression for PE-1).
    pbc_base = "http://www.pbc.gov.cn/zhengcehuobisi/x/y/index.html"
    pbc_resolved = ctp._absolutize(
        "/zhengcehuobisi/something/index.html", pbc_base,
    )
    assert pbc_resolved.startswith("http://www.pbc.gov.cn/")


# ─────────────────────────────────────────────────────────────────────
# Test 5 — Transcript title filter unit test.
# ─────────────────────────────────────────────────────────────────────
def test_title_matches_xinwen_lianbo_filter_keeps_only_transcripts():
    """The XWLB title filter keeps transcript rows (新闻联播, 联播文字版,
    xwlb) and drops adjacent comment-article rows. Empty / missing
    titles fall through to True so a defensive parse fallback is not
    silently dropped.
    """
    assert ctp._title_matches_xinwen_lianbo(
        "新闻联播文字版 2026年6月5日"
    )
    assert ctp._title_matches_xinwen_lianbo("新闻联播 6月5日完整版")
    assert ctp._title_matches_xinwen_lianbo("xwlb 20260605")
    assert ctp._title_matches_xinwen_lianbo("联播文字版")
    # Comment article — drop.
    assert not ctp._title_matches_xinwen_lianbo("央视评论员文章")
    assert not ctp._title_matches_xinwen_lianbo("时评：央视访谈节选")
    # Empty / blank title — fall through (defensive).
    assert ctp._title_matches_xinwen_lianbo("")
    assert ctp._title_matches_xinwen_lianbo("   ")


# ─────────────────────────────────────────────────────────────────────
# Test 6 — Comment-article rows are dropped from the collector output.
# Integration-level coverage of the filter inside collect_xinwen_lianbo.
# ─────────────────────────────────────────────────────────────────────
def test_comment_rows_are_dropped_by_collector(tmp_path: Path):
    """The collector's title-keyword filter must keep the transcript row
    and drop the adjacent comment row, so the comment URL is never even
    fetched (we deliberately don't mock it — an attempted fetch would
    raise requests.HTTPError and crash the test).
    """
    daily_url = ctp.XINWEN_LIANBO_LIST_URLS["xinwen_lianbo_daily"]
    cat_url = ctp.XINWEN_LIANBO_LIST_URLS["xinwen_lianbo_category"]
    url_map = {
        daily_url: _mock_response(200, DAILY_LIST_HTML),
        cat_url: _mock_response(200, "<html></html>"),
        _abs(daily_url, "/zt_d/xwlb/2026-06-05/content_1001.html"):
            _mock_response(200, DAILY_DETAIL_HTML),
        _abs(daily_url, "/zt_d/xwlb/2026-06-04/content_1002.html"):
            _mock_response(200, DETAIL_2026_06_04_HTML),
        # The comment URL is INTENTIONALLY NOT mocked — if the filter
        # leaks, _make_http_get raises and this test fails.
    }
    summary = ctp.collect_xinwen_lianbo(
        start="2026-06-04", end="2026-06-05",
        http_get_fn=_make_http_get(url_map),
        output_dir=tmp_path,
    )
    # No errors caused by the filter; comment URL never fetched.
    fetched_urls = {
        e.get("url", "") for e in summary["errors"]
        if e.get("stage") == "article"
    }
    assert not any("comment" in u for u in fetched_urls), summary["errors"]
    # And the comment URL did not land in any output JSONL row.
    out_5 = tmp_path / "2026-06-05.jsonl"
    rows = [
        json.loads(ln)
        for ln in out_5.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    for row in rows:
        assert "comment" not in row["url"], row
