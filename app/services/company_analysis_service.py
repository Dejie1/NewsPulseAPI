"""
Company analysis service orchestrating NER + FinBERT.
Detects Magnificent 7 mentions and analyzes financial sentiment for each.
"""

import asyncio
import logging
from typing import Optional
from pydantic import BaseModel

from app.services.ner_service import get_ner_service, CompanyEntity
from app.services.finbert_service import get_finbert_service

logger = logging.getLogger(__name__)


class CompanyMention(BaseModel):
    """Complete analysis result for a company mention."""
    company_name: str
    ticker: str
    matched_text: str
    sentiment_score: float      # Normalized score (-1 to 1)
    confidence_score: float     # NER confidence
    context_sentence: str
    sentiment_label: str        # "positive", "negative", "neutral"
    positive_prob: float
    negative_prob: float
    neutral_prob: float


class CompanyAnalysisService:
    """
    Orchestrates NER + FinBERT pipeline for company mention analysis.

    Process:
    1. Extract company mentions using GLiNER-spaCy NER
    2. For each mention, extract the context sentence
    3. Analyze financial sentiment of each context using FinBERT
    4. Combine results into comprehensive mention records
    """

    def __init__(self):
        self._ner_service = None
        self._finbert_service = None
        self._semaphore = asyncio.Semaphore(3)  # Limit concurrent ML inference

    @property
    def ner_service(self):
        """Lazy load NER service."""
        if self._ner_service is None:
            self._ner_service = get_ner_service()
        return self._ner_service

    @property
    def finbert_service(self):
        """Lazy load FinBERT service."""
        if self._finbert_service is None:
            self._finbert_service = get_finbert_service()
        return self._finbert_service

    async def analyze_text(self, text: str) -> list[CompanyMention]:
        """
        Analyze text for company mentions with financial sentiment.

        Args:
            text: The article text to analyze

        Returns:
            List of CompanyMention with NER + sentiment analysis
        """
        if not text or len(text.strip()) < 10:
            return []

        async with self._semaphore:
            # Step 1: Extract company entities
            entities = await self.ner_service.extract_companies(text)

            if not entities:
                return []

            # Step 2: Deduplicate by (company, context)
            # Multiple mentions of same company with same context should be merged
            unique_mentions: dict[tuple[str, str], CompanyEntity] = {}
            for entity in entities:
                key = (entity.ticker, entity.context_sentence[:100])
                if key not in unique_mentions:
                    unique_mentions[key] = entity
                else:
                    # Keep the one with higher confidence
                    if entity.confidence > unique_mentions[key].confidence:
                        unique_mentions[key] = entity

            entities = list(unique_mentions.values())

            # Step 3: Analyze sentiment for each context sentence
            context_sentences = [e.context_sentence for e in entities]

            if context_sentences:
                sentiments = await self.finbert_service.analyze_batch(context_sentences)
            else:
                sentiments = []

            # Step 4: Combine NER + sentiment results
            mentions = []
            for i, entity in enumerate(entities):
                if i < len(sentiments):
                    sentiment = sentiments[i]
                    normalized_score = self.finbert_service.to_normalized_score(sentiment)
                    sentiment_label = sentiment.sentiment
                    positive_prob = sentiment.positive_prob
                    negative_prob = sentiment.negative_prob
                    neutral_prob = sentiment.neutral_prob
                else:
                    normalized_score = 0.0
                    sentiment_label = "neutral"
                    positive_prob = 0.33
                    negative_prob = 0.33
                    neutral_prob = 0.34

                mentions.append(CompanyMention(
                    company_name=entity.company_name,
                    ticker=entity.ticker,
                    matched_text=entity.matched_text,
                    sentiment_score=normalized_score,
                    confidence_score=entity.confidence,
                    context_sentence=entity.context_sentence,
                    sentiment_label=sentiment_label,
                    positive_prob=positive_prob,
                    negative_prob=negative_prob,
                    neutral_prob=neutral_prob
                ))

            return mentions

    async def analyze_article(
        self,
        article_id: int,
        text: str,
        title: str = ""
    ) -> list[dict]:
        """
        Analyze article for company mentions, ready for database insertion.

        Args:
            article_id: The database article ID
            text: Article text content
            title: Article title (optional, for fallback analysis)

        Returns:
            List of dicts ready for Supabase upsert
        """
        # Combine title and content for better analysis
        full_text = f"{title}\n\n{text}" if title else text

        mentions = await self.analyze_text(full_text)

        # Convert to database format
        records = []
        for mention in mentions:
            records.append({
                "article_id": article_id,
                "company_name": mention.company_name,
                "ticker": mention.ticker,
                "sentiment_score": mention.sentiment_score,
                "confidence_score": mention.confidence_score,
                "context_sentence": mention.context_sentence[:500],
                "sentiment_label": mention.sentiment_label
            })

        return records

    async def analyze_articles_batch(
        self,
        articles: list[dict],
        max_concurrent: int = 3
    ) -> list[dict]:
        """
        Analyze multiple articles concurrently.

        Args:
            articles: List of dicts with 'id', 'full_content', 'title' keys
            max_concurrent: Maximum concurrent analyses

        Returns:
            List of mention records ready for database
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        all_records = []

        async def analyze_one(article: dict) -> list[dict]:
            async with semaphore:
                article_id = article.get("id")
                content = article.get("full_content") or article.get("content") or ""
                title = article.get("title", "")

                if not content:
                    return []

                return await self.analyze_article(article_id, content, title)

        tasks = [analyze_one(article) for article in articles]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error("Analysis error: %s", result)
                continue
            all_records.extend(result)

        return all_records

    def get_summary_by_company(
        self,
        mentions: list[CompanyMention]
    ) -> dict[str, dict]:
        """
        Aggregate mentions by company.

        Returns:
            Dict mapping ticker to aggregated stats
        """
        summary: dict[str, dict] = {}

        for mention in mentions:
            ticker = mention.ticker
            if ticker not in summary:
                summary[ticker] = {
                    "company_name": mention.company_name,
                    "ticker": ticker,
                    "mention_count": 0,
                    "total_sentiment": 0.0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0,
                    "contexts": []
                }

            summary[ticker]["mention_count"] += 1
            summary[ticker]["total_sentiment"] += mention.sentiment_score

            if mention.sentiment_label == "positive":
                summary[ticker]["positive_count"] += 1
            elif mention.sentiment_label == "negative":
                summary[ticker]["negative_count"] += 1
            else:
                summary[ticker]["neutral_count"] += 1

            # Keep first 5 contexts
            if len(summary[ticker]["contexts"]) < 5:
                summary[ticker]["contexts"].append({
                    "sentence": mention.context_sentence[:200],
                    "sentiment": mention.sentiment_label,
                    "score": mention.sentiment_score
                })

        # Calculate averages
        for ticker, data in summary.items():
            if data["mention_count"] > 0:
                data["avg_sentiment"] = data["total_sentiment"] / data["mention_count"]
            else:
                data["avg_sentiment"] = 0.0

        return summary


# Global company analysis service instance
_company_analyzer: Optional[CompanyAnalysisService] = None


def get_company_analyzer() -> CompanyAnalysisService:
    """Get or create the global company analysis service instance."""
    global _company_analyzer
    if _company_analyzer is None:
        _company_analyzer = CompanyAnalysisService()
    return _company_analyzer
