"""FinBERT-based news sentiment analysis for geopolitical context.

Uses ProsusAI/finbert to understand sentiment of news headlines,
combined with context rules to interpret geopolitical implications.
"""
import logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """Analyzes news headline sentiment using FinBERT.

    FinBERT understands financial language context:
    - "Trump visits Beijing for talks" → neutral/positive (diplomacy)
    - "US sanctions China" → negative (conflict)
    - "Fed cuts rates" → positive (easing)
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self._tokenizer = None
        self._model = None
        self._model_name = model_name
        self._loaded = False

    def _load_model(self):
        """Lazy-load FinBERT model."""
        if self._loaded:
            return
        try:
            logger.info("Loading FinBERT model...")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            self._model.eval()
            self._loaded = True
            logger.info("FinBERT loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}")
            self._loaded = False

    def score_headline(self, headline: str) -> float:
        """Score a single headline sentiment.

        Args:
            headline: News headline text

        Returns:
            Float from -1.0 (very negative) to 1.0 (very positive)
        """
        self._load_model()
        if not self._loaded:
            return 0.0

        try:
            inputs = self._tokenizer(
                headline, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]

            # FinBERT labels: 0=positive, 1=negative, 2=neutral
            score = float(probs[0] - probs[1])  # positive - negative
            return max(-1.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"Headline scoring failed: {e}")
            return 0.0

    def score_headlines_batch(self, headlines: list) -> list:
        """Score multiple headlines efficiently.

        Args:
            headlines: List of headline strings

        Returns:
            List of float scores [-1, 1]
        """
        self._load_model()
        if not self._loaded or not headlines:
            return [0.0] * len(headlines)

        try:
            inputs = self._tokenizer(
                headlines, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            )
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            scores = (probs[:, 0] - probs[:, 1]).tolist()
            return [max(-1.0, min(1.0, s)) for s in scores]

        except Exception as e:
            logger.warning(f"Batch scoring failed: {e}")
            return [0.0] * len(headlines)

    def analyze_geopolitical_news(self, news_items: list) -> dict:
        """Analyze a collection of news for geopolitical sentiment.

        Groups news by topic and computes per-topic sentiment.

        Args:
            news_items: List of dicts with 'title' and optionally 'source' keys

        Returns:
            Dict with:
                - overall_sentiment: average across all news [-1, 1]
                - conflict_sentiment: sentiment of conflict-related news
                - china_us_sentiment: sentiment of China-US news
                - policy_sentiment: sentiment of monetary policy news
                - market_sentiment: sentiment of stock market news
                - num_analyzed: total headlines analyzed
        """
        if not news_items:
            return self._empty_result()

        # Categorize headlines
        conflict_kw = {"war", "attack", "missile", "strike", "bomb", "military",
                       "iran", "russia", "ukraine", "israel", "gaza", "hormuz",
                       "taiwan", "invasion", "escalat", "conflict"}
        china_us_kw = {"china", "chinese", "trump", "beijing", "tariff", "trade war",
                       "xi", "us-china", "sanction", "decoupl"}
        policy_kw = {"fed", "rate", "inflation", "central bank", "ecb", "boj",
                     "pboc", "monetary", "easing", "tightening", "cpi"}
        market_kw = {"stock", "market", "nasdaq", "s&p", "dow", "nikkei",
                     "hang seng", "rally", "crash", "bull", "bear"}

        headlines = [item.get("title", "") for item in news_items if item.get("title")]
        if not headlines:
            return self._empty_result()

        # Score all headlines in batch
        scores = self.score_headlines_batch(headlines)

        # Group scores by category
        conflict_scores = []
        china_us_scores = []
        policy_scores = []
        market_scores = []

        for headline, score in zip(headlines, scores):
            hl = headline.lower()
            if any(kw in hl for kw in conflict_kw):
                conflict_scores.append(score)
            if any(kw in hl for kw in china_us_kw):
                china_us_scores.append(score)
            if any(kw in hl for kw in policy_kw):
                policy_scores.append(score)
            if any(kw in hl for kw in market_kw):
                market_scores.append(score)

        return {
            "overall_sentiment": round(float(np.mean(scores)), 4) if scores else 0.0,
            "conflict_sentiment": round(float(np.mean(conflict_scores)), 4) if conflict_scores else 0.0,
            "china_us_sentiment": round(float(np.mean(china_us_scores)), 4) if china_us_scores else 0.0,
            "policy_sentiment": round(float(np.mean(policy_scores)), 4) if policy_scores else 0.0,
            "market_sentiment": round(float(np.mean(market_scores)), 4) if market_scores else 0.0,
            "num_analyzed": len(headlines),
            "num_conflict": len(conflict_scores),
            "num_china_us": len(china_us_scores),
        }

    def _empty_result(self):
        return {
            "overall_sentiment": 0.0,
            "conflict_sentiment": 0.0,
            "china_us_sentiment": 0.0,
            "policy_sentiment": 0.0,
            "market_sentiment": 0.0,
            "num_analyzed": 0,
            "num_conflict": 0,
            "num_china_us": 0,
        }
