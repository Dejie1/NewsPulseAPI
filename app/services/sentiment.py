"""
Sentiment analysis service using NLTK VADER.
VADER is specifically designed for social media/news text.
"""

from typing import Optional
from pydantic import BaseModel


class SentimentScores(BaseModel):
    """Sentiment analysis scores from VADER."""
    negative: float  # Proportion of negative sentiment (0-1)
    neutral: float   # Proportion of neutral sentiment (0-1)
    positive: float  # Proportion of positive sentiment (0-1)
    compound: float  # Normalized compound score (-1 to 1)
    label: str       # "positive", "negative", or "neutral"


class SentimentAnalyzer:
    """
    Sentiment analyzer using NLTK's VADER.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically
    attuned to sentiments expressed in social media and news articles.

    It handles:
    - Emoticons and emojis
    - Slang and abbreviations
    - Punctuation emphasis (e.g., "good!!!")
    - Capitalization (e.g., "AMAZING")
    - Degree modifiers (e.g., "very good", "kind of bad")
    - Contrasting conjunctions (e.g., "good, but not great")
    """

    def __init__(self):
        self._analyzer = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization - loads VADER lexicon from local file."""
        if self._initialized:
            return

        import os
        import nltk

        # Add custom nltk_data paths
        nltk_paths = [
            '/home/www/nltk_data',
            '/usr/share/nltk_data',
            '/usr/local/share/nltk_data',
            os.path.expanduser('~/nltk_data'),
        ]
        for path in nltk_paths:
            if path not in nltk.data.path:
                nltk.data.path.insert(0, path)

        # Try to load VADER, download if not found (with SSL workaround)
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self._analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            # Try downloading with SSL verification disabled
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._default_https_context = _create_unverified_https_context

            nltk.download('vader_lexicon', quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            self._analyzer = SentimentIntensityAnalyzer()

        self._initialized = True

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
                label="neutral"
            )

        # Get VADER scores
        scores = self._analyzer.polarity_scores(text)

        # Determine label based on compound score
        # Standard VADER thresholds
        compound = scores['compound']
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return SentimentScores(
            negative=scores['neg'],
            neutral=scores['neu'],
            positive=scores['pos'],
            compound=compound,
            label=label
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentScores]:
        """
        Analyze sentiment for multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentScores
        """
        return [self.analyze(text) for text in texts]

    def get_average_sentiment(self, texts: list[str]) -> SentimentScores:
        """
        Get average sentiment across multiple texts.
        Useful for analyzing overall sentiment of a news source.

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
                label="neutral"
            )

        scores = self.analyze_batch(texts)
        n = len(scores)

        avg_neg = sum(s.negative for s in scores) / n
        avg_neu = sum(s.neutral for s in scores) / n
        avg_pos = sum(s.positive for s in scores) / n
        avg_compound = sum(s.compound for s in scores) / n

        # Determine label
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
            label=label
        )


# Global analyzer instance
_sentiment_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create the global sentiment analyzer instance."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer
