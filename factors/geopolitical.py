import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class GeopoliticalScorer:
    """Computes geopolitical and macroeconomic risk factors.

    Factors produced:
    - geo_risk_index: Overall geopolitical risk level [-1, 1] (negative = high risk)
    - china_us_temperature: China-US relations temperature [-1, 1]
    - policy_signal: Central bank policy direction [-1, 1] (negative = hawkish/tightening)
    - safe_haven_signal: Safe haven demand signal [0, 1] (high = buy gold)
    """

    # Keywords for policy direction detection
    HAWKISH_KEYWORDS = {
        "rate hike", "tightening", "inflation", "hawkish", "raise rates",
        "tapering", "restrictive", "higher for longer",
        "加息", "收紧", "通胀", "紧缩", "上调",
    }
    DOVISH_KEYWORDS = {
        "rate cut", "easing", "dovish", "stimulus", "lower rates",
        "accommodative", "quantitative easing", "support growth",
        "降息", "宽松", "刺激", "降准", "支持经济",
    }
    RISK_KEYWORDS = {
        "war", "conflict", "attack", "missile", "nuclear", "invasion",
        "sanction", "escalation", "crisis", "threat", "tension",
        "战争", "冲突", "制裁", "危机", "紧张",
    }
    SAFE_HAVEN_KEYWORDS = {
        "gold", "safe haven", "risk aversion", "flight to safety",
        "uncertainty", "recession", "crash",
        "黄金", "避险", "衰退",
    }

    def compute_geo_risk_index(self, conflict_articles: list) -> float:
        """Compute geopolitical risk index from conflict articles.

        Args:
            conflict_articles: List of dicts with 'title' and 'tone' keys
                              (from GDELTCollector.fetch_geopolitical_conflicts)

        Returns:
            Float from -1 (extreme risk) to 1 (very calm).
            0 means neutral.
        """
        if not conflict_articles:
            return 0.0

        tones = []
        risk_count = 0

        for article in conflict_articles:
            tone = article.get("tone", 0)
            if isinstance(tone, (int, float)) and tone != 0:
                tones.append(tone)

            title = str(article.get("title", "")).lower()
            if any(kw in title for kw in self.RISK_KEYWORDS):
                risk_count += 1

        # Risk keyword density (0 to 1, higher = more risk)
        risk_density = min(risk_count / max(len(conflict_articles), 1), 1.0)

        if tones:
            # If we have tone data (GDELT), use weighted combo
            avg_tone = np.mean(tones)
            tone_score = float(np.clip(avg_tone / 10.0, -1.0, 1.0))
            score = tone_score * 0.6 + (1.0 - risk_density * 2) * 0.4
        else:
            # No tone data (RSS): rely purely on keyword density
            # More conflict keywords = more risk = more negative
            score = 1.0 - risk_density * 2  # density=0 → 1.0, density=1 → -1.0

        return float(np.clip(score, -1.0, 1.0))

    # Keywords for China-US sentiment analysis
    CHINA_US_HOSTILE = {
        "tariff", "trade war", "sanction", "ban", "restrict", "decouple",
        "military", "tension", "threat", "retaliate", "blacklist",
        "制裁", "贸易战", "脱钩", "对抗", "威胁",
    }
    CHINA_US_FRIENDLY = {
        "deal", "agreement", "cooperat", "talk", "meeting", "visit",
        "dialogue", "negotiat", "progress", "partner", "resume",
        "合作", "会谈", "访问", "对话", "协议", "缓和",
    }

    def compute_china_us_temperature(self, relation_articles: list) -> float:
        """Compute China-US relations temperature.

        Args:
            relation_articles: List of dicts with 'title' and 'tone' keys

        Returns:
            Float from -1 (very hostile) to 1 (very cooperative).
        """
        if not relation_articles:
            return 0.0

        # If tone data is available and non-zero, use it
        tones = [a.get("tone", 0) for a in relation_articles
                 if isinstance(a.get("tone"), (int, float)) and a.get("tone") != 0]
        if tones:
            avg_tone = np.mean(tones)
            return float(np.clip(avg_tone / 10.0, -1.0, 1.0))

        # Otherwise: keyword-based analysis
        hostile_count = 0
        friendly_count = 0
        for article in relation_articles:
            title = str(article.get("title", "")).lower()
            desc = str(article.get("description", "")).lower()
            text = title + " " + desc
            if any(kw in text for kw in self.CHINA_US_HOSTILE):
                hostile_count += 1
            if any(kw in text for kw in self.CHINA_US_FRIENDLY):
                friendly_count += 1

        total = hostile_count + friendly_count
        if total == 0:
            return 0.0
        return float(np.clip((friendly_count - hostile_count) / total, -1.0, 1.0))

    def compute_policy_signal(self, macro_news: list) -> float:
        """Compute central bank policy direction signal.

        Args:
            macro_news: List of dicts with 'title' and 'description' keys
                       (from MacroCollector.fetch_all)

        Returns:
            Float from -1 (hawkish/tightening) to 1 (dovish/easing).
        """
        if not macro_news:
            return 0.0

        hawkish_count = 0
        dovish_count = 0

        for item in macro_news:
            text = (
                str(item.get("title", "")).lower() + " " +
                str(item.get("description", "")).lower()
            )
            if any(kw in text for kw in self.HAWKISH_KEYWORDS):
                hawkish_count += 1
            if any(kw in text for kw in self.DOVISH_KEYWORDS):
                dovish_count += 1

        total = hawkish_count + dovish_count
        if total == 0:
            return 0.0

        return float(np.clip((dovish_count - hawkish_count) / total, -1.0, 1.0))

    def compute_safe_haven_signal(
        self, conflict_articles: list, macro_news: list
    ) -> float:
        """Compute safe haven demand signal (relevant for gold).

        Args:
            conflict_articles: From GDELT conflict fetch
            macro_news: From macro RSS fetch

        Returns:
            Float from 0 (no safe haven demand) to 1 (strong safe haven demand).
        """
        # Factor 1: geopolitical risk
        geo_risk = self.compute_geo_risk_index(conflict_articles)
        risk_component = max(0, -geo_risk)  # More negative = more risk = more demand

        # Factor 2: safe haven keyword density in news
        safe_count = 0
        total_items = len(macro_news) + len(conflict_articles)

        for item in macro_news + conflict_articles:
            text = (
                str(item.get("title", "")).lower() + " " +
                str(item.get("description", "")).lower()
            )
            if any(kw in text for kw in self.SAFE_HAVEN_KEYWORDS):
                safe_count += 1

        keyword_density = safe_count / max(total_items, 1)

        # Combine
        signal = risk_component * 0.6 + min(keyword_density * 5, 1.0) * 0.4
        return float(np.clip(signal, 0.0, 1.0))

    def compute_all_factors(
        self,
        conflict_articles: list,
        relation_articles: list,
        macro_news: list,
    ) -> dict:
        """Compute all geopolitical factors at once.

        Args:
            conflict_articles: From GDELTCollector.fetch_geopolitical_conflicts
            relation_articles: From GDELTCollector.fetch_china_us_relations
            macro_news: From MacroCollector.fetch_all

        Returns:
            Dict with all factor scores.
        """
        geo_risk = self.compute_geo_risk_index(conflict_articles)
        china_us = self.compute_china_us_temperature(relation_articles)
        policy = self.compute_policy_signal(macro_news)
        safe_haven = self.compute_safe_haven_signal(conflict_articles, macro_news)

        return {
            "geo_risk_index": round(geo_risk, 4),
            "china_us_temperature": round(china_us, 4),
            "policy_signal": round(policy, 4),
            "safe_haven_signal": round(safe_haven, 4),
        }
