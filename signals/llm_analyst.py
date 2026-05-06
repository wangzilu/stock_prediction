"""LLM-powered market analyst using MiniMax API.

Generates professional-grade market analysis reports in the style of
top financial analysts, combining global news, market data, and
geopolitical factors into actionable insights.
"""
import re
import json
import logging
import requests
from datetime import datetime
from config.settings import MINIMAX_API_KEY

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一位顶级金融分析师和地缘政治分析专家，风格参考"振海听风"。你的任务是基于提供的全球新闻头条、市场数据和地缘政治信息，撰写专业的每日市场研判报告。

写作要求：
1. 语言风格：专业但通俗易懂，有观点有态度，像大V发的深度短文
2. 结构清晰：分为"全球局势"、"大盘研判"、"今日推荐"三个板块
3. 有深度：不只是罗列新闻，要分析背后的逻辑链条和因果关系
4. 有前瞻性：基于当前局势推断短期走势
5. 控制在400-600字
6. 每个部分要有清晰的结论和操作建议

你必须基于提供的数据进行分析，不要编造不存在的事件。"""

GEO_ANALYSIS_PROMPT = """基于以下全球新闻头条，给出地缘政治和宏观经济评分（JSON格式）。

新闻头条：
{headlines}

请输出严格JSON格式（不要任何其他文字）：
{{
  "geo_risk_index": 数值（-1到1，-1=极端风险，1=非常安全），
  "china_us_temperature": 数值（-1到1，-1=极度对抗，1=友好合作），
  "policy_signal": 数值（-1到1，-1=强烈收紧，1=强烈宽松），
  "safe_haven_signal": 数值（0到1，0=无避险需求，1=极强避险需求），
  "market_direction": 数值（-1到1，-1=强烈看空A股，1=强烈看多A股），
  "reasoning": {{
    "geo_risk": "一句话理由",
    "china_us": "一句话理由",
    "policy": "一句话理由",
    "safe_haven": "一句话理由",
    "market": "一句话理由"
  }}
}}"""


class LLMAnalyst:
    """LLM-powered market analyst using MiniMax API."""

    def __init__(self, api_key: str = None, model: str = "minimax-m2.5-highspeed"):
        self.api_key = api_key or MINIMAX_API_KEY
        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY is required")
        self.model = model
        self.api_url = "https://api.minimax.io/v1/chat/completions"

    def _call_llm(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """Call MiniMax API and return cleaned response text."""
        try:
            resp = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )

            if resp.status_code != 200:
                logger.warning(f"MiniMax API returned {resp.status_code}: {resp.text[:200]}")
                return ""

            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            # Strip think tags
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text

        except Exception as e:
            logger.warning(f"MiniMax API call failed: {e}")
            return ""

    def analyze_geopolitics(self, headlines: list) -> dict:
        """Use LLM to analyze geopolitical risk from news headlines.

        Args:
            headlines: List of news headline strings

        Returns:
            Dict with geo_risk_index, china_us_temperature, policy_signal,
            safe_haven_signal, market_direction, reasoning
        """
        if not headlines:
            return self._default_geo_result()

        headline_text = "\n".join(f"- {h}" for h in headlines[:80])
        prompt = GEO_ANALYSIS_PROMPT.format(headlines=headline_text)

        text = self._call_llm("你是一个地缘政治和宏观经济分析专家。只输出JSON，不要其他内容。", prompt, max_tokens=1024)

        if not text:
            return self._default_geo_result()

        try:
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                result = {
                    "geo_risk_index": max(-1, min(1, float(parsed.get("geo_risk_index", 0)))),
                    "china_us_temperature": max(-1, min(1, float(parsed.get("china_us_temperature", 0)))),
                    "policy_signal": max(-1, min(1, float(parsed.get("policy_signal", 0)))),
                    "safe_haven_signal": max(0, min(1, float(parsed.get("safe_haven_signal", 0)))),
                    "market_direction": max(-1, min(1, float(parsed.get("market_direction", 0)))),
                    "reasoning": parsed.get("reasoning", {}),
                }
                return result
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM geo response: {e}")

        return self._default_geo_result()

    def generate_report(
        self,
        headlines: list,
        market_judgment: dict,
        recommendations: list,
        geo_factors: dict,
    ) -> str:
        """Generate a professional market analysis report.

        Args:
            headlines: Global news headlines
            market_judgment: Dict with direction, score, reason, index_change
            recommendations: List of Recommendation objects
            geo_factors: Dict with geo scores and reasoning

        Returns:
            Formatted report string for push notification
        """
        # Build context for LLM
        headline_text = "\n".join(f"- {h}" for h in headlines[:30])

        rec_text = ""
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                display_code = rec.code[2:] if rec.code[:2] in ("SH", "SZ") else rec.code
                rec_text += f"{i}. {rec.name}({display_code}) | 评分{(rec.final_score+1)*5:.1f} | {rec.reason}\n"

        reasoning_text = ""
        if geo_factors.get("reasoning"):
            for key, reason in geo_factors["reasoning"].items():
                reasoning_text += f"- {key}: {reason}\n"

        user_prompt = f"""请基于以下信息撰写今日市场研判报告：

日期：{datetime.now().strftime("%Y年%m月%d日")}

【全球新闻头条】
{headline_text}

【大盘数据】
A股大盘方向：{market_judgment.get('direction', '未知')}
沪深300涨跌幅：{market_judgment.get('index_change', 0):+.2f}%
研判理由：{market_judgment.get('reason', '')}

【地缘评估】
地缘风险指数：{geo_factors.get('geo_risk_index', 0):+.2f}
中美关系温度：{geo_factors.get('china_us_temperature', 0):+.2f}
政策方向：{geo_factors.get('policy_signal', 0):+.2f}
避险需求：{geo_factors.get('safe_haven_signal', 0):.2f}
{reasoning_text}

【今日推荐标的】
{rec_text if rec_text else '暂无明确推荐信号'}

请撰写研判报告，包含：
1. 📌 全球局势速览（100-150字）
2. 📊 A股大盘研判（100-150字，含仓位建议）
3. 💡 今日操作建议（100-150字，如有推荐标的则点评）

注意：如果没有推荐标的，操作建议部分给出观望理由和关注方向。"""

        report = self._call_llm(SYSTEM_PROMPT, user_prompt, max_tokens=2048)

        if not report:
            # Fallback to simple format
            return self._fallback_report(market_judgment, recommendations, geo_factors)

        return report

    def _fallback_report(self, market_judgment, recommendations, geo_factors):
        """Generate simple report when LLM is unavailable."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"📊 市场研判 ({now})",
            f"大盘：{market_judgment.get('direction', '中性')}（{market_judgment.get('reason', '')}）",
            f"建议仓位：{market_judgment.get('suggested_position', '5成')}",
            "─────────────",
        ]
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                display_code = rec.code[2:] if rec.code[:2] in ("SH", "SZ") else rec.code
                score = round((rec.final_score + 1) * 5, 1)
                lines.append(f"{i}. {rec.name}({display_code}) | {rec.signal} | {score}")
                lines.append(f"   {rec.reason}")
        else:
            lines.append("暂无明确推荐信号，建议观望")
        return "\n".join(lines)

    def _default_geo_result(self):
        return {
            "geo_risk_index": 0.0,
            "china_us_temperature": 0.0,
            "policy_signal": 0.0,
            "safe_haven_signal": 0.0,
            "market_direction": 0.0,
            "reasoning": {},
        }
