"""Post-LLM event-type validator (Phase C.2, L3).

LLMs cheerfully classify a headline mentioning the word "业绩" as
``earnings_*`` even when the article is "中信证券业绩预测" (a sell-side
forecast), or tag any "回购" word as ``share_buyback`` regardless of
whether the article is the actual buyback announcement vs commentary.

This module post-checks the LLM-assigned ``event_type`` against
hand-curated keyword gates per event family. When the gate fails, the
event is downgraded to ``other`` or ``routine_announcement`` instead
of being silently accepted into the factor stream.

Usage::

    from factors.event_schema_validator import validate_event_type

    final_type, validation_reason = validate_event_type(
        proposed_type="earnings_beat",
        title="中信证券：宁德时代Q1业绩预增超预期",
        content="...",
    )
    # final_type may be the input or a downgraded label.

The gate is intentionally simple — a fixed regex set, no machine
learning. The point is to catch the obvious overfit cases.
"""
from __future__ import annotations

import re
from typing import Iterable

# ─────────────────────────────────────────────────────────────────────
# Keyword gates per event family.
#
# An event_type whose family appears here MUST have at least one of the
# regexes hit somewhere in the title or content. When the gate fails
# the type downgrades to the fallback declared in DOWNGRADE_TARGET.
# Types not listed here are unconstrained (insider_buy / sell etc are
# already specific enough at the LLM level).
# ─────────────────────────────────────────────────────────────────────
KEYWORD_GATES: dict[str, list[re.Pattern]] = {
    # Earnings family. The LLM tags ANYTHING that says 业绩 as some
    # earnings_*; the regex set restricts to actual disclosure language.
    "earnings_beat":  [],     # filled by EARNINGS_KEYWORDS below
    "earnings_miss":  [],
    "earnings_inline": [],
    "revenue_growth": [],
    "revenue_decline": [],
    # Capital-action family.
    "share_buyback": [
        re.compile(r"(回购|拟回购|实施回购|股份回购|完成回购)"),
    ],
    "dividend_increase": [
        re.compile(r"(分红|派息|送转|每10股派|每股派|股息|权益分派)"),
    ],
    "share_placement": [
        re.compile(r"(定增|定向增发|配股|公开发行|向特定对象发行|非公开发行)"),
    ],
    "share_unlock": [
        re.compile(r"(解禁|解除限售|限售股解禁|首发限售解禁|限售解禁)"),
    ],
    "insider_buy": [
        re.compile(r"(高管|董事|监事|股东).{0,8}(增持|买入|认购)"),
    ],
    "insider_sell": [
        re.compile(r"(高管|董事|监事|股东).{0,8}(减持|卖出|套现|出售)"),
    ],
    # Regulatory family.
    "regulatory_approval": [
        re.compile(r"(批准|核准|许可|通过|获批|获得.{0,8}(认证|许可|批文))"),
    ],
    "regulatory_penalty": [
        re.compile(r"(处罚|立案|调查|警示函|监管措施|责令改正|罚款|没收|"
                   r"违规|违法|失信|处分)"),
    ],
    "lawsuit_filed": [
        re.compile(r"(诉讼|被诉|起诉|提起诉讼|仲裁|被起诉|被告|应诉)"),
    ],
    "lawsuit_settled": [
        re.compile(r"(和解|调解|结案|终审|败诉|胜诉|裁定|判决)"),
    ],
    # Capital structure family.
    "debt_issue": [
        re.compile(r"(发行.{0,4}(债券|中票|短融|超短融|可转债|公司债)|"
                   r"债券发行|拟发行.{0,4}债)"),
    ],
    "credit_rating_change": [
        re.compile(r"(信用评级|评级.{0,4}(上调|下调|调整|展望|确认|维持))"),
    ],
    # Management / restructuring.
    "management_change": [
        re.compile(r"(辞职|辞任|换届|聘任|聘请|离任|任命|高管变动|"
                    r"董事长.{0,4}变|总经理.{0,4}变|CEO.{0,4}变|"
                    r"董事辞|监事辞|高管辞)"),
    ],
    "restructuring": [
        re.compile(r"(重组|资产重组|破产重整|预重整|资产置换|借壳)"),
    ],
    # Business-event family.
    "strategic_cooperation": [
        re.compile(r"(战略合作|签署.{0,4}(协议|备忘录|合作)|建立战略|"
                    r"战略联盟|战略伙伴)"),
    ],
    "joint_venture": [
        re.compile(r"(合资|合资公司|合营|成立.{0,8}(子公司|公司)|"
                    r"联合.{0,4}成立)"),
    ],
    # Subsidy / tax.
    "government_subsidy": [
        re.compile(r"(政府补助|财政补贴|专项资金|拨款|奖励资金|"
                    r"产业扶持|项目补助)"),
    ],
    "tax_benefit": [
        re.compile(r"(税收优惠|税率优惠|增值税.{0,4}(优惠|退税)|"
                    r"减免税|所得税.{0,4}(优惠|减免)|"
                    r"享受.{0,4}税收)"),
    ],
}

# Earnings keywords shared across the earnings_* family.
EARNINGS_KEYWORDS = [
    re.compile(r"(年报|半年报|季报|一季报|三季报|"
                r"业绩(预告|快报|公告|预增|预减|预盈|预亏|预喜|预降)|"
                r"净利润|营业收入|营业总收入|归母净利|扣非净利|EPS|"
                r"每股收益|营收|毛利率)"),
]
for k in ("earnings_beat", "earnings_miss", "earnings_inline",
          "revenue_growth", "revenue_decline"):
    KEYWORD_GATES[k] = EARNINGS_KEYWORDS


# Default downgrade targets per family. ``other`` is fine for most;
# routine_announcement is reserved for "this was a generic disclosure
# the LLM over-classified".
DOWNGRADE_TARGET: dict[str, str] = {
    "earnings_beat":   "routine_announcement",
    "earnings_miss":   "routine_announcement",
    "earnings_inline": "routine_announcement",
    "revenue_growth":  "routine_announcement",
    "revenue_decline": "routine_announcement",
    "share_buyback":   "routine_announcement",
    "dividend_increase": "routine_announcement",
}


def validate_event_type(
    *,
    proposed_type: str,
    title: str = "",
    content: str = "",
) -> tuple[str, str]:
    """Validate the LLM-assigned ``proposed_type`` against keyword gates.

    Parameters
    ----------
    proposed_type:
        The ``event_type`` the LLM emitted.
    title:
        Article title (most signal-rich; checked first).
    content:
        Optional article body.

    Returns
    -------
    (final_type, reason):
        ``final_type`` is the original type when the gate passes (or
        when no gate exists for this type), or the downgrade target
        when it fails. ``reason`` is one of ``"passed"``,
        ``"no_gate"``, or ``"downgraded: <reason>"``.
    """
    if not proposed_type:
        return ("other", "downgraded: empty_proposed_type")
    if proposed_type not in KEYWORD_GATES:
        return (proposed_type, "no_gate")
    patterns = KEYWORD_GATES[proposed_type]
    haystack = f"{title}\n{content}"
    if any(p.search(haystack) for p in patterns):
        return (proposed_type, "passed")
    fallback = DOWNGRADE_TARGET.get(proposed_type, "other")
    return (
        fallback,
        f"downgraded: no keyword in {proposed_type} family matched",
    )


def gated_event_types() -> Iterable[str]:
    """Return the event_types this module enforces a gate on.

    Useful for tests that want to assert coverage of the families the
    project lead's audit flagged."""
    return tuple(KEYWORD_GATES.keys())
