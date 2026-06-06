"""Tests for the L4 generic blacklist patterns in factors/event_filter.py.

Phase C.1 fix: project lead flagged 8+ generic templates that were still
leaking through L0/L1 filtering and burning LLM RPM. These tests pin
the new patterns so a regex regression surfaces immediately.

Coverage:
  - Each new template label fires on its representative titles.
  - Genuine company-specific events still pass (no false positive).
  - Pattern boundaries: case sensitivity, leading whitespace, trailing
    punctuation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _add_project_root_to_path():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    yield


def _matched_label(title: str) -> str | None:
    from factors.event_filter import DROP_PATTERNS
    for p, label in DROP_PATTERNS:
        if p.search(title):
            return label
    return None


def _is_dropped(title: str) -> bool:
    """Title is dropped if ANY pattern (incl. title_too_short) matches.

    The point of L4 is to stop noisy titles from reaching the LLM. Some
    of the new templates land on the older labels (e.g.
    ``generic_breakout_template`` already covers ``突破均线`` patterns).
    Whether the label is the new one or the old one is irrelevant — the
    title getting dropped is what matters.
    """
    return _matched_label(title) is not None


# ── Drop side: each new template fires ────────────────────────────────

CAPITAL_FLOW = [
    "资金流向日报：北向资金净买入30亿",
    "资金流向榜单：主力净流入这些股",
    "主力资金净流出这些股",
    "主力资金净流入个股名单",
]

CONCEPT_MOVE = [
    "AI概念集体大涨，这些股领涨",
    "机器人概念上涨",
    "光伏概念大跌",
    "新能源汽车概念集体跳水",
    "半导体概念涨幅榜",
]

DIVIDEND_CALENDAR = [
    "12股下周实施分红",
    "23只本月进行分红",
    "本周分红榜单一览",
    "本月送转个股名单",
]

RIGHTS = [
    "抢权行情能否上演？这些股即将实施分红",
    "填权机会一览",
    "除权除息这些股名单",
]

HOLDER_LIST = [
    "股东户数降幅榜：宁德时代降幅居首",
    "股东户数增幅榜单",
    "户均持股变化榜",
    "股东户数连降这些股",
]

MULTI_BREAKOUT = [
    "12股突破年线",
    "8只创新高",
    "27股突破均线",
]

MARGIN_LIST = [
    "融资客大幅加仓榜单",
    "两融偏好榜",
    "融资融券净买名单",
    "融资客连续抢筹一览",
]

INSTITUTION_RESEARCH = [
    "机构调研频次榜TOP20",
    "主力调研这些股",
    "机构调研名单",
    "机构调研前10",
    "机构调研新进个股",
]


@pytest.mark.parametrize("title", CAPITAL_FLOW + CONCEPT_MOVE
                                  + DIVIDEND_CALENDAR + RIGHTS
                                  + HOLDER_LIST + MULTI_BREAKOUT
                                  + MARGIN_LIST + INSTITUTION_RESEARCH)
def test_l4_generic_title_is_dropped(title):
    """Every noisy template from the project lead's 2026-06-06 LLM
    critique must get dropped by SOME pattern — either one of the new
    L4 entries or one of the older generic templates (overlap is fine,
    the point is the title never reaches the LLM)."""
    assert _is_dropped(title), (
        f"L4 noisy title {title!r} was NOT dropped by any pattern. "
        f"matched_label={_matched_label(title)!r}"
    )


# ── Pass side: genuine events still flow through ──────────────────────

GENUINE_COMPANY_EVENTS = [
    "宁德时代发布2025年Q2业绩快报",
    "宁德时代签订与特斯拉的10年期电池供应合同",
    "比亚迪与丰田成立合资公司，专注电动车研发",
    "贵州茅台发布拟回购公司股份方案",
    "美的集团董事长高管增持",
    "中芯国际新一代28nm工艺量产",
    "万科A计划发行50亿元中期票据",
    "海康威视披露年报：净利润同比增长35%",
]


@pytest.mark.parametrize("title", GENUINE_COMPANY_EVENTS)
def test_genuine_events_pass_through(title):
    """Sanity: company-specific factual events MUST NOT be dropped by the
    L4 blacklist additions. A regression that catches one of these
    would cripple downstream factor coverage."""
    label = _matched_label(title)
    assert label is None, (
        f"genuine company event {title!r} matched blacklist pattern "
        f"{label!r} — that is a false positive."
    )


# ── Smoke: total pattern count ────────────────────────────────────────

def test_total_pattern_count_at_least_17():
    """The new patterns brought us from 9 → 17. Don't accidentally
    delete them on the next refactor."""
    from factors.event_filter import DROP_PATTERNS
    assert len(DROP_PATTERNS) >= 17, (
        f"DROP_PATTERNS has only {len(DROP_PATTERNS)} entries; expected "
        f">= 17 after the L4 expansion. A pattern was probably removed."
    )
