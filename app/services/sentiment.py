"""
Sentiment analysis service using RoBERTa.
Uses cardiffnlp/twitter-roberta-base-sentiment-latest for news text sentiment.
"""

import logging
from typing import Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SentimentScores(BaseModel):
    """Sentiment analysis scores."""
    negative: float  # Probability of negative sentiment (0-1)
    neutral: float   # Probability of neutral sentiment (0-1)
    positive: float  # Probability of positive sentiment (0-1)
    compound: float  # Synthetic compound score (-1 to 1): positive_prob - negative_prob
    label: str       # "positive", "negative", or "neutral"


class SentimentAnalyzer:
    """
    Sentiment analyzer using RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest).

    A RoBERTa-base model fine-tuned on ~124M tweets for sentiment analysis.
    Outputs three-class probabilities (negative, neutral, positive).
    """

    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # Label mapping for this specific model
    LABEL_MAP = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
        # Some versions use text labels directly
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
    }

    def __init__(self):
        self._pipeline = None
        self._initialized = False
        self._initialization_error: Optional[str] = None

    def _ensure_initialized(self):
        """Lazy initialization - loads RoBERTa model on first use."""
        if self._initialized:
            return

        if self._initialization_error:
            return

        try:
            from transformers import pipeline
            import torch

            device = 0 if torch.cuda.is_available() else -1
            logger.info("Loading RoBERTa sentiment model: %s", self.MODEL_NAME)

            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.MODEL_NAME,
                device=device,
                max_length=512,
                truncation=True,
                top_k=None,  # Get all class probabilities
            )

            self._initialized = True
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("RoBERTa sentiment model loaded on %s", device_name)

        except Exception as e:
            self._initialization_error = str(e)
            logger.error("Failed to initialize RoBERTa sentiment model: %s", e)

    def unload(self) -> None:
        """Release the model from memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        self._initialized = False
        self._initialization_error = None

    def _parse_scores(self, results: list[dict]) -> SentimentScores:
        """Parse pipeline output into SentimentScores."""
        scores = {}
        for r in results:
            label = self.LABEL_MAP.get(r["label"], r["label"].lower())
            scores[label] = r["score"]

        negative = scores.get("negative", 0.0)
        neutral = scores.get("neutral", 0.0)
        positive = scores.get("positive", 0.0)

        # Synthetic compound: positive - negative, range [-1, 1]
        compound = positive - negative

        # Label is the highest probability class
        if positive >= negative and positive >= neutral:
            label = "positive"
        elif negative >= positive and negative >= neutral:
            label = "negative"
        else:
            label = "neutral"

        return SentimentScores(
            negative=negative,
            neutral=neutral,
            positive=positive,
            compound=compound,
            label=label,
        )

    def analyze(self, text: str) -> SentimentScores:
        """
        Analyze sentiment of the given text.

        Args:
            text: The text to analyze

        Returns:
            SentimentScores with detailed sentiment breakdown
        """
        self._ensure_initialized()

        if not text or not text.strip():
            return SentimentScores(
                negative=0.0,
                neutral=1.0,
                positive=0.0,
                compound=0.0,
                label="neutral",
            )

        if not self._initialized:
            return SentimentScores(
                negative=0.0,
                neutral=1.0,
                positive=0.0,
                compound=0.0,
                label="neutral",
            )

        results = self._pipeline(text)

        # results is [[{'label': ..., 'score': ...}, ...]] when top_k=None
        return self._parse_scores(results[0] if results else [])

    def analyze_batch(self, texts: list[str]) -> list[SentimentScores]:
        """
        Analyze sentiment for multiple texts using batch inference.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentScores
        """
        self._ensure_initialized()

        if not texts:
            return []

        if not self._initialized:
            return [
                SentimentScores(
                    negative=0.0,
                    neutral=1.0,
                    positive=0.0,
                    compound=0.0,
                    label="neutral",
                )
                for _ in texts
            ]

        # Filter empty texts, track indices
        processed = []
        index_map = {}
        for i, text in enumerate(texts):
            if text and text.strip():
                index_map[len(processed)] = i
                processed.append(text)

        if not processed:
            return [
                SentimentScores(
                    negative=0.0,
                    neutral=1.0,
                    positive=0.0,
                    compound=0.0,
                    label="neutral",
                )
                for _ in texts
            ]

        # Batch inference
        batch_results = self._pipeline(processed)

        # Build results, inserting neutral for empty texts
        all_scores = [
            SentimentScores(
                negative=0.0,
                neutral=1.0,
                positive=0.0,
                compound=0.0,
                label="neutral",
            )
            for _ in texts
        ]

        for proc_idx, result in enumerate(batch_results):
            orig_idx = index_map[proc_idx]
            all_scores[orig_idx] = self._parse_scores(result)

        return all_scores

    def get_average_sentiment(self, texts: list[str]) -> SentimentScores:
        """
        Get average sentiment across multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            Averaged SentimentScores
        """
        if not texts:
            return SentimentScores(
                negative=0.0,
                neutral=1.0,
                positive=0.0,
                compound=0.0,
                label="neutral",
            )

        scores = self.analyze_batch(texts)
        n = len(scores)

        avg_neg = sum(s.negative for s in scores) / n
        avg_neu = sum(s.neutral for s in scores) / n
        avg_pos = sum(s.positive for s in scores) / n
        avg_compound = sum(s.compound for s in scores) / n

        if avg_compound >= 0.05:
            label = "positive"
        elif avg_compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return SentimentScores(
            negative=avg_neg,
            neutral=avg_neu,
            positive=avg_pos,
            compound=avg_compound,
            label=label,
        )


# Global analyzer instance
_sentiment_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create the global sentiment analyzer instance."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer
