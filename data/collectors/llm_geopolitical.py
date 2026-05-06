"""LLM-based geopolitical analysis using Claude API.

Replaces GDELT with direct LLM reasoning about current geopolitical events.
Claude has up-to-date knowledge and can reason about complex geopolitical dynamics.
"""
import json
import logging
import requests
ANTHROPIC_API_KEY = ""  # Deprecated: use signals/llm_analyst.py with MiniMax instead

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个专业的地缘政治和宏观经济分析师。你的任务是评估当前全球局势对金融市场的影响。

请基于你所知道的最新国际局势，输出以下四个评分（JSON格式）：

1. geo_risk_index: 地缘政治风险指数，范围[-1, 1]
   - -1 = 极端风险（大规模战争、核威胁等）
   - 0 = 中性
   - 1 = 非常安全平静

2. china_us_temperature: 中美关系温度，范围[-1, 1]
   - -1 = 极度对抗（贸易战全面升级、军事对峙）
   - 0 = 中性
   - 1 = 友好合作

3. policy_signal: 全球央行政策方向，范围[-1, 1]
   - -1 = 强烈收紧/鹰派（加息、缩表）
   - 0 = 中性
   - 1 = 强烈宽松/鸽派（降息、放水）

4. safe_haven_signal: 避险需求信号，范围[0, 1]
   - 0 = 无避险需求
   - 1 = 极强避险需求（应买入黄金）

同时提供简短的分析理由（每个因子一句话）。

输出格式必须是严格的JSON：
{
  "geo_risk_index": 数值,
  "china_us_temperature": 数值,
  "policy_signal": 数值,
  "safe_haven_signal": 数值,
  "reasoning": {
    "geo_risk": "一句话理由",
    "china_us": "一句话理由",
    "policy": "一句话理由",
    "safe_haven": "一句话理由"
  }
}"""


class LLMGeopoliticalCollector:
    """Uses Claude API to analyze current geopolitical situation."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        self.api_url = "https://api.anthropic.com/v1/messages"

    def analyze(self, additional_context: str = "") -> dict:
        """Ask Claude to assess current geopolitical risks.

        Args:
            additional_context: Optional extra context (e.g., today's headlines)

        Returns:
            Dict with geo_risk_index, china_us_temperature, policy_signal,
            safe_haven_signal, and reasoning. Returns zeros on failure.
        """
        user_message = "请评估当前（今天）的全球地缘政治和宏观经济形势，给出四个评分。"
        if additional_context:
            user_message += f"\n\n补充信息：\n{additional_context}"

        try:
            resp = requests.post(
                self.api_url,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1024,
                    "system": SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": user_message}
                    ],
                },
                timeout=30,
            )

            if resp.status_code != 200:
                logger.warning(f"Claude API returned {resp.status_code}: {resp.text[:200]}")
                return self._default_result()

            data = resp.json()
            text = data["content"][0]["text"]

            # Parse JSON from response
            result = self._parse_response(text)
            logger.info(f"LLM geo analysis: {result}")
            return result

        except Exception as e:
            logger.warning(f"LLM geopolitical analysis failed: {e}")
            return self._default_result()

    def _parse_response(self, text: str) -> dict:
        """Parse Claude's JSON response."""
        try:
            # Try to find JSON in the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                parsed = json.loads(json_str)

                # Validate and clamp values
                result = {
                    "geo_risk_index": max(-1, min(1, float(parsed.get("geo_risk_index", 0)))),
                    "china_us_temperature": max(-1, min(1, float(parsed.get("china_us_temperature", 0)))),
                    "policy_signal": max(-1, min(1, float(parsed.get("policy_signal", 0)))),
                    "safe_haven_signal": max(0, min(1, float(parsed.get("safe_haven_signal", 0)))),
                    "reasoning": parsed.get("reasoning", {}),
                }
                return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        return self._default_result()

    def _default_result(self) -> dict:
        """Return neutral defaults when analysis fails."""
        return {
            "geo_risk_index": 0.0,
            "china_us_temperature": 0.0,
            "policy_signal": 0.0,
            "safe_haven_signal": 0.0,
            "reasoning": {},
        }
