import logging
import numpy as np

logger = logging.getLogger(__name__)


class SentimentScorer:
    """Scores financial text sentiment using SnowNLP + keyword fallback.

    Primary: SnowNLP (machine learning, trained on Chinese text)
    Fallback: keyword matching (when SnowNLP unavailable)
    """

    def __init__(self):
        self._snownlp_available = None
        self._keywords_positive = {"看涨", "利好", "突破", "强势", "涨停", "暴涨", "看好", "牛", "反弹", "新高", "业绩", "增长", "盈利", "分红"}
        self._keywords_negative = {"暴跌", "崩盘", "利空", "跌停", "做空", "下跌", "割肉", "套牢", "风险", "亏损", "不看好", "减持", "爆雷", "退市"}

    def _check_snownlp(self):
        if self._snownlp_available is None:
            try:
                from snownlp import SnowNLP
                SnowNLP("测试")
                self._snownlp_available = True
            except Exception:
                self._snownlp_available = False
                logger.info("SnowNLP not available, using keyword fallback")
        return self._snownlp_available

    def score_text(self, text: str) -> float:
        """Score a single text for sentiment.

        Args:
            text: Chinese financial text

        Returns:
            Float from -1.0 (very negative) to 1.0 (very positive)
        """
        if not text or len(text.strip()) < 2:
            return 0.0

        if self._check_snownlp():
            try:
                from snownlp import SnowNLP
                s = SnowNLP(text)
                # SnowNLP outputs 0~1, convert to -1~1
                score = s.sentiments * 2 - 1
                # Blend with keyword score for financial domain correction
                kw_score = self._score_with_keywords(text)
                if kw_score != 0:
                    # If keywords present, blend 60% SnowNLP + 40% keywords
                    return float(np.clip(score * 0.6 + kw_score * 0.4, -1.0, 1.0))
                return float(np.clip(score, -1.0, 1.0))
            except Exception:
                pass

        return self._score_with_keywords(text)

    def _score_with_keywords(self, text: str) -> float:
        """Keyword-based scoring fallback."""
        pos_count = sum(1 for kw in self._keywords_positive if kw in text)
        neg_count = sum(1 for kw in self._keywords_negative if kw in text)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return float(np.clip((pos_count - neg_count) / total, -1.0, 1.0))

    def score_batch(self, posts: list[dict]) -> dict:
        """Score a batch of posts and return aggregate metrics.

        Args:
            posts: List of dicts with "text", "timestamp", "source" keys

        Returns:
            Dict with keys:
                - sentiment_score: average sentiment [-1, 1]
                - heat: post count normalized (log scale)
                - post_count: raw number of posts
        """
        if not posts:
            return {"sentiment_score": 0.0, "heat": 0.0, "post_count": 0}

        scores = [self.score_text(p["text"]) for p in posts]
        avg_score = float(np.mean(scores))

        # Heat: log-normalized post count (0-1 scale, 100 posts = 1.0)
        heat = float(np.clip(np.log1p(len(posts)) / np.log1p(100), 0.0, 1.0))

        return {
            "sentiment_score": round(avg_score, 4),
            "heat": round(heat, 4),
            "post_count": len(posts),
        }
