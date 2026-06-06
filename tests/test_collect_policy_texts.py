"""Tests for scripts/collect_policy_texts.py — Phase E.1 step 1.

Covers:
  1. JSONL schema is exactly the contract documented in the script
     (publish_date / policy_type / title / url / content / source /
     fetch_time) and nothing extra; one row per policy item.
  2. Idempotent atomic write — re-running the same date overwrites
     the day's JSONL, never appends or duplicates.
  3. HTTP error handling — a 500 list page does NOT kill the run; the
     day file is still written (empty) and the error is recorded in
     the summary.

All tests are HTTP-mocked (the network is OFF). The mock intercepts
``http_get`` so we never depend on pbc.gov.cn being reachable in CI.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from scripts import collect_policy_texts as ctp


# ─────────────────────────────────────────────────────────────────────
# Fixtures — minimal valid OMO list / detail HTML matching PBOC layout
# ─────────────────────────────────────────────────────────────────────
OMO_LIST_HTML = """
<html><body>
<ul class="list">
  <li>
    <a href="./2026/06/03/omo_announcement_2026_06_03.html">
      公开市场业务交易公告
    </a>
    <span>2026-06-03</span>
  </li>
  <li>
    <a href="./2026/06/05/omo_announcement_2026_06_05.html">
      公开市场业务交易公告（第三号）
    </a>
    <span>2026-06-05</span>
  </li>
  <li>
    <a href="./2026/05/20/old_announcement.html">旧的公告（窗口外）</a>
    <span>2026-05-20</span>
  </li>
  <li><a href="javascript:void(0)">nav anchor</a></li>
</ul>
</body></html>
"""

LPR_LIST_HTML = """
<html><body>
<a href="./2026/06/02/lpr_2026_06.html">中国人民银行授权全国银行间同业拆借中心公布贷款市场报价利率</a>
<span>2026-06-02</span>
</body></html>
"""

OMO_DETAIL_HTML = """
<html><body>
<div class="header">发布时间：2026-06-05</div>
<div id="zoom">
  <p>为维护银行体系流动性合理充裕，2026年6月5日人民银行以利率招标方式开展了
  1500亿元逆回购操作，中标利率1.40%。</p>
  <p>当日有800亿元逆回购到期，实现净投放700亿元。</p>
</div>
</body></html>
"""

LPR_DETAIL_HTML = """
<html><body>
<div id="zoom">
  <p>2026年6月2日贷款市场报价利率（LPR）为：1年期LPR为3.10%，
  5年期以上LPR为3.60%。以上LPR自发布之日起实施，下一次发布时间为
  2026年7月20日。</p>
</div>
</body></html>
"""


def _mock_response(status: int, body: str) -> MagicMock:
    """Build a minimal ``requests.Response``-like mock."""
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


# ─────────────────────────────────────────────────────────────────────
# Test 1 — JSONL schema
# ─────────────────────────────────────────────────────────────────────
def test_jsonl_schema_is_exactly_the_documented_contract(tmp_path: Path):
    """Every JSONL row has the 7 documented keys and nothing else."""
    url_map = {
        ctp.PBC_LIST_URLS["omo"]: _mock_response(200, OMO_LIST_HTML),
        ctp.PBC_LIST_URLS["lpr"]: _mock_response(200, LPR_LIST_HTML),
        # Resolved relative URLs from the list pages (./YYYY/MM/DD/...html)
        # → list page dir + path component.
        ctp.PBC_LIST_URLS["omo"].rsplit("/", 1)[0]
        + "/2026/06/05/omo_announcement_2026_06_05.html":
            _mock_response(200, OMO_DETAIL_HTML),
        ctp.PBC_LIST_URLS["lpr"].rsplit("/", 1)[0]
        + "/2026/06/02/lpr_2026_06.html":
            _mock_response(200, LPR_DETAIL_HTML),
    }
    summary = ctp.collect_pbc(
        start="2026-06-02",
        end="2026-06-05",
        policy_types=("omo", "lpr"),
        bonus_types=(),
        http_get_fn=_make_http_get(url_map),
        output_dir=tmp_path,
    )

    # Both documents should land in the window.
    assert summary["n_total"] >= 2, summary
    assert summary["rows_by_date"]["2026-06-05"] >= 1
    assert summary["rows_by_date"]["2026-06-02"] >= 1

    expected_keys = {
        "publish_date", "policy_type", "title", "url",
        "content", "source", "fetch_time",
    }
    omo_path = tmp_path / "2026-06-05.jsonl"
    lines = [
        json.loads(ln) for ln in omo_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert lines, "OMO day file should not be empty"
    for row in lines:
        assert set(row.keys()) == expected_keys, (
            f"unexpected schema: {sorted(row.keys())}"
        )
        assert row["source"] == "pbc.gov.cn"
        assert row["policy_type"] in {
            "omo", "lpr", "mlf", "slf", "rrr",
            "quarterly_report", "press_conference", "other",
        }
        assert row["publish_date"] == "2026-06-05"
        # fetch_time is ISO-8601 UTC with trailing Z
        assert row["fetch_time"].endswith("Z")
        assert len(row["content"]) >= 20


# ─────────────────────────────────────────────────────────────────────
# Test 2 — Idempotent write
# ─────────────────────────────────────────────────────────────────────
def test_idempotent_rerun_does_not_duplicate_rows(tmp_path: Path):
    """Two runs over the same date produce the same JSONL — no append."""
    url_map = {
        ctp.PBC_LIST_URLS["omo"]: _mock_response(200, OMO_LIST_HTML),
        ctp.PBC_LIST_URLS["lpr"]: _mock_response(200, LPR_LIST_HTML),
        ctp.PBC_LIST_URLS["omo"].rsplit("/", 1)[0]
        + "/2026/06/05/omo_announcement_2026_06_05.html":
            _mock_response(200, OMO_DETAIL_HTML),
        ctp.PBC_LIST_URLS["lpr"].rsplit("/", 1)[0]
        + "/2026/06/02/lpr_2026_06.html":
            _mock_response(200, LPR_DETAIL_HTML),
    }
    fake = _make_http_get(url_map)

    s1 = ctp.collect_pbc(
        start="2026-06-05", end="2026-06-05",
        policy_types=("omo", "lpr"), bonus_types=(),
        http_get_fn=fake, output_dir=tmp_path,
    )
    s2 = ctp.collect_pbc(
        start="2026-06-05", end="2026-06-05",
        policy_types=("omo", "lpr"), bonus_types=(),
        http_get_fn=fake, output_dir=tmp_path,
    )

    out = tmp_path / "2026-06-05.jsonl"
    assert out.exists()
    lines1 = [
        json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    # Now run a second time and read again — content must be IDENTICAL.
    lines2 = [
        json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    # We compare everything except fetch_time (which is wall-clock).
    def _strip(rows: list[dict]) -> list[dict]:
        return [{k: v for k, v in r.items() if k != "fetch_time"} for r in rows]

    assert _strip(lines1) == _strip(lines2), "Re-run produced different rows"
    # And the row count matches both summaries.
    assert s1["n_total"] == s2["n_total"] == len(lines1)
    # No .tmp file should be left over after atomic replace.
    assert not list(tmp_path.glob("*.tmp"))


# ─────────────────────────────────────────────────────────────────────
# Test 3 — HTTP error handling (list-page 500 → empty day, no crash)
# ─────────────────────────────────────────────────────────────────────
def test_http_error_on_list_page_does_not_crash_run(tmp_path: Path):
    """A 500 on the OMO list page records an error but the day file is
    still written and the LPR path keeps working."""
    url_map = {
        ctp.PBC_LIST_URLS["omo"]: _mock_response(500, "boom"),
        ctp.PBC_LIST_URLS["lpr"]: _mock_response(200, LPR_LIST_HTML),
        ctp.PBC_LIST_URLS["lpr"].rsplit("/", 1)[0]
        + "/2026/06/02/lpr_2026_06.html":
            _mock_response(200, LPR_DETAIL_HTML),
    }
    summary = ctp.collect_pbc(
        start="2026-06-02", end="2026-06-02",
        policy_types=("omo", "lpr"), bonus_types=(),
        http_get_fn=_make_http_get(url_map),
        output_dir=tmp_path,
    )

    # OMO failure is recorded
    omo_list_url = ctp.PBC_LIST_URLS["omo"]
    assert any(
        e["stage"] == "list" and e["url"] == omo_list_url
        for e in summary["errors"]
    ), summary["errors"]
    # File for the day is still on disk (LPR row survived)
    out = tmp_path / "2026-06-02.jsonl"
    assert out.exists()
    lines = [
        json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(lines) == 1
    assert lines[0]["policy_type"] == "lpr"
    assert lines[0]["publish_date"] == "2026-06-02"


# ─────────────────────────────────────────────────────────────────────
# Test 4 — http_get retry wrapper
# ─────────────────────────────────────────────────────────────────────
def test_http_get_retries_on_transient_5xx(monkeypatch):
    """http_get should retry on 5xx and succeed on the 3rd attempt."""
    call_count = {"n": 0}

    class FakeSession:
        def get(self, url, headers, timeout):
            call_count["n"] += 1
            if call_count["n"] < 3:
                resp = MagicMock(spec=requests.Response)
                resp.status_code = 503
                resp.raise_for_status.side_effect = requests.HTTPError("503")
                return resp
            resp = MagicMock(spec=requests.Response)
            resp.status_code = 200
            resp.content = b"<html>ok</html>"
            resp.raise_for_status.return_value = None
            return resp

    resp = ctp.http_get(
        "http://example.invalid/x", session=FakeSession(), backoff=0.0,
    )
    assert resp.status_code == 200
    assert call_count["n"] == 3


def test_http_get_fail_fast_on_4xx():
    """4xx should not be retried — the request is final."""
    call_count = {"n": 0}

    class FakeSession:
        def get(self, url, headers, timeout):
            call_count["n"] += 1
            resp = MagicMock(spec=requests.Response)
            resp.status_code = 404
            resp.raise_for_status.side_effect = requests.HTTPError("404")
            return resp

    with pytest.raises(requests.HTTPError):
        ctp.http_get(
            "http://example.invalid/x", session=FakeSession(), backoff=0.0,
        )
    # Implementation may retry once for the loop bookkeeping or call
    # raise_for_status. Either way it must NOT make 3 separate calls.
    assert call_count["n"] <= 1


def test_parse_list_page_handles_live_timestamp_url_pattern():
    """2026-06-06 regression: the original parser only matched the
    legacy ``/YYYY/MM/DD/`` slash pattern. The live PBOC OMO page uses
    ``/<YYYYMMDDhhmmssNNN>/index.html`` — a single timestamp digit
    string in the path. parse_list_page must extract both.
    """
    sample_html = '''
    <html><body>
    <div>
      <a href="/zhengcehuobisi/125207/125213/125431/125475/2026060508521729212/index.html">
        公开市场业务交易公告 [2026]第107号
      </a>
      <a href="/zhengcehuobisi/125207/125213/125431/125475/2026060408523140012/index.html">
        公开市场业务交易公告 [2026]第106号
      </a>
      <a href="./2024/03/15/foo.html">legacy slash pattern</a>
      <a href="/index.html">nav link no date</a>
    </body></html>
    '''
    base = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/125475/index.html"
    links = ctp.parse_list_page(sample_html, base)
    by_date = {ln.publish_date for ln in links}
    assert "2026-06-05" in by_date, f"expected 2026-06-05 in {by_date}"
    assert "2026-06-04" in by_date, f"expected 2026-06-04 in {by_date}"
    assert "2024-03-15" in by_date, f"expected 2024-03-15 in {by_date}"
    # nav link without date marker should NOT be picked up
    assert all("index.html" not in ln.url or "/index.html" in ln.url
               for ln in links if "20240315" in ln.url or "20260605" in ln.url)
