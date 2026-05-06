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

核心写作原则：
1. 因果逻辑链条是灵魂：不是罗列事件，而是分析"A导致B，B导致C，所以影响D"
2. 事件有层次结构：大事件（如美伊战争）包含子事件（如霍尔木兹封锁），不要把子事件和父事件并列，要体现从属关系
3. 每个判断都要有"因为...所以..."的逻辑支撑
4. 有观点有态度：不要两边讨好，明确给出方向判断
5. 落脚到投资：每个地缘分析最终都要回答"这对我的钱包意味着什么"

写作要求：
- 语言风格：专业但通俗易懂，像顶级分析师的深度短文
- 控制在500-800字
- 每个部分要有清晰的结论

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
        """Call MiniMax API with retry and return cleaned response text."""
        import time

        for attempt in range(2):
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
                    logger.warning(f"MiniMax API attempt {attempt+1} returned {resp.status_code}: {resp.text[:200]}")
                    if attempt == 0:
                        time.sleep(3)
                        continue
                    return ""

                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                # Strip think tags
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                return text

            except Exception as e:
                logger.warning(f"MiniMax API attempt {attempt+1} failed: {e}")
                if attempt == 0:
                    time.sleep(3)

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

        headline_text = "\n".join(f"- {h}" for h in headlines[:150])
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
        crypto_data: dict = None,
        gold_data: dict = None,
    ) -> str:
        """Generate a professional market analysis report.

        Args:
            headlines: Global news headlines
            market_judgment: Dict with direction, score, reason, index_change
            recommendations: List of Recommendation objects
            geo_factors: Dict with geo scores and reasoning
            crypto_data: Dict with BTC/ETH prices and change_pct
            gold_data: Dict with gold price and change_pct

        Returns:
            Formatted report string for push notification
        """
        headline_text = "\n".join(f"- {h}" for h in headlines[:100])

        rec_text = ""
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                display_code = rec.code[2:] if rec.code[:2] in ("SH", "SZ") else rec.code
                rec_text += f"{i}. {rec.name}({display_code}) | 评分{(rec.final_score+1)*5:.1f} | {rec.reason}\n"

        reasoning_text = ""
        if geo_factors.get("reasoning"):
            for key, reason in geo_factors["reasoning"].items():
                reasoning_text += f"- {key}: {reason}\n"

        # Build crypto/gold context
        crypto_text = ""
        if crypto_data:
            for symbol, data in crypto_data.items():
                name = "比特币" if "BTC" in symbol else "以太坊" if "ETH" in symbol else symbol
                crypto_text += f"  {name}: ${data.get('price', 0):,.0f} ({data.get('change_pct', 0):+.1f}%)\n"

        gold_text = ""
        if gold_data:
            gold_text = f"  黄金: ¥{gold_data.get('price', 0):,.1f} ({gold_data.get('change_pct', 0):+.1f}%)"

        user_prompt = f"""请基于以下信息撰写今日市场研判报告：

日期：{datetime.now().strftime("%Y年%m月%d日")}

【今日全球新闻头条（请全面分析所有重要事件，不要遗漏）】
{headline_text}

【A股大盘数据】
大盘方向：{market_judgment.get('direction', '未知')}
沪深300涨跌幅：{market_judgment.get('index_change', 0):+.2f}%
研判理由：{market_judgment.get('reason', '')}

【加密货币】
{crypto_text if crypto_text else '  数据暂无'}

【黄金】
{gold_text if gold_text else '  数据暂无'}

【地缘评估】
地缘风险指数：{geo_factors.get('geo_risk_index', 0):+.2f}（-1极端风险，+1安全）
中美关系温度：{geo_factors.get('china_us_temperature', 0):+.2f}（-1对抗，+1合作）
政策方向：{geo_factors.get('policy_signal', 0):+.2f}（-1紧缩，+1宽松）
避险需求：{geo_factors.get('safe_haven_signal', 0):.2f}（0无需求，1极强）
{reasoning_text}

【今日推荐标的】
{rec_text if rec_text else '暂无明确推荐信号'}

请撰写研判报告，严格包含以下5个板块：

1. 📌 **全球局势**（150-200字）
   用因果链条串联当前全球主要矛盾，不要罗列零散事件。
   例如：美伊战争→霍尔木兹海峡受威胁→油价上涨→全球通胀压力→央行政策两难
   注意事件的从属关系：子事件（如海峡封锁）是大事件（如美伊战争）的一部分，不要并列。
   最后落脚到：这对全球金融市场意味着什么？

2. 📊 **A股大盘复盘与明日预判**（200-250字）
   今天A股为什么涨/跌？不要只描述数据，要给出因果链：
   "因为...（消息面/资金面/情绪面），所以...（哪些板块涨/跌），导致...（大盘方向）"
   基于今天的逻辑链，推导明天的走势方向。
   给出明确的仓位建议（X成仓位）和操作策略。

3. 🪙 **加密货币**（100-150字）
   BTC/ETH的走势受什么驱动？（宏观流动性？监管？避险？技术面？）
   给出短期方向判断和操作建议。

4. 🏆 **黄金**（100-150字）
   黄金涨跌的核心逻辑是什么？（避险需求？美元走势？实际利率？央行购金？）
   当前该买入还是观望？

5. 💡 **个股推荐与操作建议**（100-150字）
   如有推荐标的则逐个说明推荐逻辑（不是罗列指标）。
   如无推荐则说明观望理由和应该关注的方向/板块。

关键要求：
- 每个分析都必须有"因为A→所以B→导致C"的逻辑链
- 不要把子事件和父事件并列（如"美伊战争"和"霍尔木兹封锁"不能并列）
- 落脚到投资决策：每个分析最后都回答"所以我该怎么做"
- 有鲜明观点，不要模棱两可"""

        report = self._call_llm(SYSTEM_PROMPT, user_prompt, max_tokens=3000)

        if not report:
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
