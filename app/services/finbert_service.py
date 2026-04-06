"""
Financial sentiment analysis service using FinBERT.
Analyzes sentiment of financial/business text with specialized model.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FinancialSentiment(BaseModel):
    """Financial sentiment analysis result."""
    text: str               # The analyzed text
    sentiment: str          # "positive", "negative", "neutral"
    score: float            # Confidence score (0-1)
    positive_prob: float    # Probability of positive
    negative_prob: float    # Probability of negative
    neutral_prob: float     # Probability of neutral


class FinBERTService:
    """
    Financial sentiment analysis using ProsusAI/finbert.

    FinBERT is a BERT model fine-tuned on financial text for sentiment analysis.
    It understands financial jargon and context better than general sentiment models.

    Uses lazy initialization to avoid loading the model at startup.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._initialized = False
        self._initialization_error: Optional[str] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._device = "cpu"

    def _ensure_initialized(self):
        """Lazy initialization - loads FinBERT model."""
        if self._initialized:
            return

        if self._initialization_error:
            return

        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline
            )
            import torch

            model_name = "ProsusAI/finbert"
            logger.info("Loading FinBERT model: %s", model_name)

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)

            # Use GPU if available, otherwise CPU
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            self._model.eval()  # Set to evaluation mode

            # Create pipeline for easy inference
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self._model,
                tokenizer=self._tokenizer,
                device=0 if self._device == "cuda" else -1,
                max_length=512,
                truncation=True,
                top_k=None  # Get all class probabilities
            )

            self._initialized = True
            logger.info("FinBERT service initialized on %s", self._device)

        except Exception as e:
            self._initialization_error = str(e)
            logger.error("Failed to initialize FinBERT service: %s", e)

    def shutdown(self) -> None:
        """Shut down executor and release models from memory."""
        self._executor.shutdown(wait=False)
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._initialized = False
        self._initialization_error = None

    def _analyze_sync(self, text: str) -> FinancialSentiment:
        """
        Synchronous sentiment analysis (runs in thread pool).

        FinBERT outputs: positive, negative, neutral with probabilities.
        """
        if not self._initialized or not text:
            return FinancialSentiment(
                text=text or "",
                sentiment="neutral",
                score=0.5,
                positive_prob=0.33,
                negative_prob=0.33,
                neutral_prob=0.34
            )

        try:
            # Run inference (tokenizer handles truncation via max_length=512 tokens)
            results = self._pipeline(text)

            # results is a list with one element containing scores for all classes
            # [{'label': 'positive', 'score': 0.9}, {'label': 'negative', ...}, ...]
            scores = {r['label'].lower(): r['score'] for r in results[0]}

            positive_prob = scores.get('positive', 0.0)
            negative_prob = scores.get('negative', 0.0)
            neutral_prob = scores.get('neutral', 0.0)

            # Determine sentiment label based on highest probability
            if positive_prob >= negative_prob and positive_prob >= neutral_prob:
                sentiment = "positive"
                score = positive_prob
            elif negative_prob >= positive_prob and negative_prob >= neutral_prob:
                sentiment = "negative"
                score = negative_prob
            else:
                sentiment = "neutral"
                score = neutral_prob

            return FinancialSentiment(
                text=text,
                sentiment=sentiment,
                score=score,
                positive_prob=positive_prob,
                negative_prob=negative_prob,
                neutral_prob=neutral_prob
            )

        except Exception as e:
            logger.error("FinBERT analysis error: %s", e)
            return FinancialSentiment(
                text=text,
                sentiment="neutral",
                score=0.5,
                positive_prob=0.33,
                negative_prob=0.33,
                neutral_prob=0.34
            )

    def _analyze_batch_sync(self, texts: list[str]) -> list[FinancialSentiment]:
        """
        Batch sentiment analysis (more efficient for multiple texts).
        """
        if not self._initialized or not texts:
            return [
                FinancialSentiment(
                    text=t or "",
                    sentiment="neutral",
                    score=0.5,
                    positive_prob=0.33,
                    negative_prob=0.33,
                    neutral_prob=0.34
                )
                for t in texts
            ]

        try:
            # Filter empty texts; tokenizer handles truncation via max_length=512 tokens
            texts = [t if t else "" for t in texts]

            # Batch inference
            batch_results = self._pipeline(texts)

            sentiments = []
            for i, results in enumerate(batch_results):
                scores = {r['label'].lower(): r['score'] for r in results}

                positive_prob = scores.get('positive', 0.0)
                negative_prob = scores.get('negative', 0.0)
                neutral_prob = scores.get('neutral', 0.0)

                if positive_prob >= negative_prob and positive_prob >= neutral_prob:
                    sentiment = "positive"
                    score = positive_prob
                elif negative_prob >= positive_prob and negative_prob >= neutral_prob:
                    sentiment = "negative"
                    score = negative_prob
                else:
                    sentiment = "neutral"
                    score = neutral_prob

                sentiments.append(FinancialSentiment(
                    text=texts[i],
                    sentiment=sentiment,
                    score=score,
                    positive_prob=positive_prob,
                    negative_prob=negative_prob,
                    neutral_prob=neutral_prob
                ))

            return sentiments

        except Exception as e:
            logger.error("FinBERT batch analysis error: %s", e)
            return [
                FinancialSentiment(
                    text=t,
                    sentiment="neutral",
                    score=0.5,
                    positive_prob=0.33,
                    negative_prob=0.33,
                    neutral_prob=0.34
                )
                for t in texts
            ]

    async def analyze(self, text: str) -> FinancialSentiment:
        """
        Analyze financial sentiment of text.

        Args:
            text: The text to analyze (typically a context sentence)

        Returns:
            FinancialSentiment with probabilities and label
        """
        self._ensure_initialized()

        if self._initialization_error:
            return FinancialSentiment(
                text=text or "",
                sentiment="neutral",
                score=0.5,
                positive_prob=0.33,
                negative_prob=0.33,
                neutral_prob=0.34
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._analyze_sync,
                text
            )
            return result

        except Exception as e:
            logger.error("FinBERT analysis failed: %s", e)
            return FinancialSentiment(
                text=text or "",
                sentiment="neutral",
                score=0.5,
                positive_prob=0.33,
                negative_prob=0.33,
                neutral_prob=0.34
            )

    async def analyze_batch(self, texts: list[str]) -> list[FinancialSentiment]:
        """
        Analyze financial sentiment for multiple texts.

        More efficient than calling analyze() multiple times.

        Args:
            texts: List of texts to analyze

        Returns:
            List of FinancialSentiment results
        """
        self._ensure_initialized()

        if self._initialization_error or not texts:
            return [
                FinancialSentiment(
                    text=t or "",
                    sentiment="neutral",
                    score=0.5,
                    positive_prob=0.33,
                    negative_prob=0.33,
                    neutral_prob=0.34
                )
                for t in texts
            ]

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self._executor,
                self._analyze_batch_sync,
                texts
            )
            return results

        except Exception as e:
            logger.error("FinBERT batch analysis failed: %s", e)
            return [
                FinancialSentiment(
                    text=t or "",
                    sentiment="neutral",
                    score=0.5,
                    positive_prob=0.33,
                    negative_prob=0.33,
                    neutral_prob=0.34
                )
                for t in texts
            ]

    def to_normalized_score(self, sentiment: FinancialSentiment) -> float:
        """
        Convert FinBERT sentiment to normalized score (-1 to 1).

        Maps:
        - positive -> positive value (0 to 1)
        - negative -> negative value (-1 to 0)
        - neutral -> close to 0

        Args:
            sentiment: FinancialSentiment result

        Returns:
            Float between -1 and 1
        """
        # Weight by probabilities
        # positive contributes positively, negative contributes negatively
        score = (
            sentiment.positive_prob * 1.0 +
            sentiment.neutral_prob * 0.0 +
            sentiment.negative_prob * -1.0
        )
        return score

    def is_available(self) -> bool:
        """Check if FinBERT service is available and ready."""
        if self._initialization_error:
            return False
        if not self._initialized:
            try:
                self._ensure_initialized()
            except Exception:
                return False
        return self._initialized


# Global FinBERT service instance
_finbert_service: Optional[FinBERTService] = None


def get_finbert_service() -> FinBERTService:
    """Get or create the global FinBERT service instance."""
    global _finbert_service
    if _finbert_service is None:
        _finbert_service = FinBERTService()
    return _finbert_service
