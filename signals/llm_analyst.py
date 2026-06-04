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

SYSTEM_PROMPT = """你是一位顶级金融分析师和地缘政治分析专家，结构参考"震海听风"式的清晰时政财经短评，但不要模仿具体措辞。你的任务是基于提供的全球新闻头条、市场数据和地缘政治信息，撰写专业的每日市场研判报告。

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

GEO_ANALYSIS_PROMPT = """你是一个独立的地缘政治和宏观经济分析师。请认真阅读以下全球新闻头条，自行判断当前最重要的地缘政治和宏观经济议题，然后给出评分。

重要：你必须基于新闻内容独立判断，不要预设任何立场。什么事件重要、什么关系紧张、什么政策在变——全部由你从新闻中读出来。

新闻头条：
{headlines}

请输出严格JSON格式（不要任何其他文字）：
{{
  "geo_risk_index": 数值（-1到1，-1=极端风险如大规模战争，1=非常安全平静），
  "china_us_temperature": 数值（-1到1，-1=极度对抗，1=友好合作），
  "policy_signal": 数值（-1到1，-1=全球央行强烈收紧，1=强烈宽松），
  "safe_haven_signal": 数值（0到1，0=无避险需求，1=极强避险需求），
  "market_direction": 数值（-1到1，-1=强烈看空A股，1=强烈看多A股），
  "key_events": [
    "你认为当前最重要的事件1（一句话描述+对市场的影响）",
    "你认为当前最重要的事件2",
    "你认为当前最重要的事件3",
    "..."
  ],
  "reasoning": {{
    "geo_risk": "基于你读到的新闻，为什么给这个分",
    "china_us": "基于你读到的新闻，为什么给这个分",
    "policy": "基于你读到的新闻，为什么给这个分",
    "safe_haven": "基于你读到的新闻，为什么给这个分",
    "market": "基于你读到的新闻，为什么给这个分，特别是对A股的影响逻辑"
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
            logger.warning(
                "analyze_geopolitics: no headlines passed in — returning default zero result "
                "(check upstream news fetch in domestic mode)"
            )
            return self._default_geo_result()

        # Limit headlines to fit within model context (estimate ~4 chars per token)
        selected = []
        char_count = 0
        for h in headlines:
            if char_count + len(h) > 12000:  # ~3000 tokens for headlines
                break
            selected.append(h)
            char_count += len(h)
        headline_text = "\n".join(f"- {h}" for h in selected)
        logger.info(f"Geo analysis input: {len(selected)} headlines, ~{char_count} chars")
        prompt = GEO_ANALYSIS_PROMPT.format(headlines=headline_text)

        text = self._call_llm("你是一个地缘政治和宏观经济分析专家。只输出JSON，不要其他内容。", prompt, max_tokens=2048)

        if not text:
            logger.warning("LLM geo analysis returned empty response")
            return self._default_geo_result()

        try:
            # Find JSON in response (may be wrapped in markdown code block)
            clean = text.strip()
            if clean.startswith("```"):
                # Strip markdown code fences
                lines = clean.split("\n")
                clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = clean[start:end]
                parsed = json.loads(json_str)
                result = {
                    "geo_risk_index": max(-1, min(1, float(parsed.get("geo_risk_index", 0)))),
                    "china_us_temperature": max(-1, min(1, float(parsed.get("china_us_temperature", 0)))),
                    "policy_signal": max(-1, min(1, float(parsed.get("policy_signal", 0)))),
                    "safe_haven_signal": max(0, min(1, float(parsed.get("safe_haven_signal", 0)))),
                    "market_direction": max(-1, min(1, float(parsed.get("market_direction", 0)))),
                    "key_events": parsed.get("key_events", []),
                    "reasoning": parsed.get("reasoning", {}),
                }
                return result
            else:
                logger.warning(f"No JSON found in LLM geo response: {text[:200]}")
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM geo response: {e}\nRaw: {text[:300]}")

        return self._default_geo_result()

    def generate_report(
        self,
        headlines: list,
        market_judgment: dict,
        recommendations: list,
        geo_factors: dict,
        crypto_data: dict = None,
        gold_data: dict = None,
        global_indices_text: str = "",
        horizon_recommendations_text: str = "",
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
                display_code = rec.code[2:] if rec.code[:2] in ("SH", "SZ", "BJ") else rec.code
                next_day = ""
                # cx round 11 P1-1: relabel from "明日预测" to "5日均/日".
                # Underlying value is 5-day model_score / 5, not a
                # next-day forecast. ``next_day_change_pct`` is now a
                # back-compat property aliasing ``horizon_dailyized_return_pct``.
                if getattr(rec, "horizon", "") == "短线" and getattr(rec, "horizon_dailyized_return_pct", None) is not None:
                    next_day = f" | 5日均/日{rec.horizon_dailyized_return_pct:+.2f}%"
                horizon = f" | {rec.horizon}" if getattr(rec, "horizon", "") else ""
                rec_text += (
                    f"{i}. {rec.name}({display_code}){horizon}{next_day} | "
                    f"评分{(rec.final_score+1)*5:.1f} | {rec.reason}\n"
                )

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

【全球股市行情（实时数据）】
{global_indices_text if global_indices_text else '  数据暂无'}

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

【长中短线分类推荐】
{horizon_recommendations_text if horizon_recommendations_text else '暂无分组推荐'}

请撰写研判报告，严格包含以下5个板块：

1. 📌 **全球局势**（200-300字）
   请认真阅读所有新闻头条，自行判断当前全球最重要的议题有哪些。
   不要预设——中东、台海、中美、俄乌、欧盟、各大央行、华尔街关注的焦点...
   一切由新闻内容决定。
   用因果链条串联：大事件→子事件→对市场的传导路径。
   子事件是大事件的一部分，不要并列。
   同时告诉读者：华尔街和全球投资者今天最关注什么？为什么？
   最后落脚到：这对全球金融市场和A股意味着什么？

2. 📊 **全球市场与A股大盘**（250-300字）
   先快速点评美股、港股等主要市场的表现和驱动因素（基于实时数据）。
   然后重点分析：
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
   必须按短线、中线、长线分别说明推荐逻辑（不是罗列指标）。
   短线标的必须引用 5 日预测均/日 (即 5 日横截面模型分数的日均近似，不是真正的明日收益)。
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
                display_code = rec.code[2:] if rec.code[:2] in ("SH", "SZ", "BJ") else rec.code
                score = round((rec.final_score + 1) * 5, 1)
                lines.append(f"{i}. {rec.name}({display_code}) | {rec.signal} | {score}")
                lines.append(f"   {rec.reason}")
        else:
            lines.append("暂无明确推荐信号，建议观望")
        return "\n".join(lines)

    def generate_summary(self, data: dict) -> str:
        """Generate daily market close summary."""
        user_prompt = f"""请基于以下数据撰写今日收盘市场总结（300-500字）：

全球指数：
{data.get('global_indices', '无数据')}

加密货币：{json.dumps(data.get('crypto_data', {}), ensure_ascii=False)}
黄金：{json.dumps(data.get('gold_data', {}), ensure_ascii=False)}

地缘政治因素：{json.dumps(data.get('geo_factors', {}), ensure_ascii=False)}

今日重要新闻：
{chr(10).join(data.get('headlines', [])[:20])}

要求：
1. 总结今日A股、港股、美股期货走势
2. 分析主要驱动因素
3. 点评板块轮动
4. 给出明日开盘预判"""

        return self._call_llm(SYSTEM_PROMPT, user_prompt, max_tokens=2048)

    def generate_outlook(self, data: dict) -> str:
        """Generate evening outlook for next trading day."""
        top_bull = data.get("top_bullish", [])
        top_bear = data.get("top_bearish", [])

        bull_text = "\n".join([f"  {b['code']}: {b['score']:.4f}" for b in top_bull])
        bear_text = "\n".join([f"  {b['code']}: {b['score']:.4f}" for b in top_bear])

        user_prompt = f"""请撰写明日市场展望的前三段宏观主线（350-650字）。

重要：你只输出以下三段，不要输出大盘预测表、个股列表、黄金、加密货币，因为这些会由程序在后面统一拼接。不要写标题之外的开场白。

固定结构：
一、世界大事
二、对世界格局的影响
三、对投资的影响

写法要求：
- 每段只抓1-2个真正重要的主线，不要罗列新闻
- 用"事件 → 格局变化 → 资金/风险偏好 → 投资动作"的因果链
- 观点明确，语言克制，不要重复数字表

结构化大盘预测：
{data.get('market_prediction_text', '未生成')}

上证/深证/北证/科创预测：
{data.get('a_share_forecast_text', '未生成')}

个股预测：
{data.get('short_candidates_text', '暂无')}

黄金预测：
{data.get('gold_forecast_text', '暂无')}

加密货币预测：
{data.get('crypto_forecast_text', '暂无')}

模型看多前10:
{bull_text}

模型看空后5:
{bear_text}

全球指数：
{data.get('global_indices', '无数据')}

加密货币：{json.dumps(data.get('crypto_data', {}), ensure_ascii=False)}
黄金：{json.dumps(data.get('gold_data', {}), ensure_ascii=False)}

地缘因素：{json.dumps(data.get('geo_factors', {}), ensure_ascii=False)}

夜间新闻：
{chr(10).join(data.get('headlines', [])[:20])}

再次强调：最终输出只包含"一、世界大事""二、对世界格局的影响""三、对投资的影响"三段。"""

        return self._call_llm(SYSTEM_PROMPT, user_prompt, max_tokens=2048)

    def _default_geo_result(self):
        return {
            "geo_risk_index": 0.0,
            "china_us_temperature": 0.0,
            "policy_signal": 0.0,
            "safe_haven_signal": 0.0,
            "market_direction": 0.0,
            "reasoning": {},
        }
