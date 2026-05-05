import numpy as np


class SentimentScorer:
    """Scores financial text sentiment using keyword-based approach.

    For MVP, uses keyword matching. Will be upgraded to FinGPT in Phase 2.
    """

    def __init__(self):
        self._keywords_positive = {"看涨", "利好", "突破", "强势", "涨停", "暴涨", "看好", "牛", "反弹", "新高", "业绩", "增长", "盈利", "分红"}
        self._keywords_negative = {"暴跌", "崩盘", "利空", "跌停", "做空", "下跌", "割肉", "套牢", "风险", "亏损", "不看好", "减持", "爆雷", "退市"}

    def score_text(self, text: str) -> float:
        """Score a single text for sentiment.

        Args:
            text: Chinese financial text

        Returns:
            Float from -1.0 (very negative) to 1.0 (very positive)
        """
        return self._score_with_keywords(text)

    def _score_with_keywords(self, text: str) -> float:
        """Keyword-based scoring."""
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
