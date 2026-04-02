"""
Core aggregator service.
Orchestrates fetching from multiple sources, deduplication, and caching.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

from app.models import Article, FeedSource, AggregationResult, AggregationStatus
from app.config import settings, get_feed_sources
from app.services.rss_parser import RSSParserService
from app.services.cache import get_cache, CacheService
from app.services.deduplicator import Deduplicator
from app.services.content_extractor import get_content_extractor


class AggregatorService:
    """
    Main aggregator service that coordinates all components.
    This is the primary interface for triggering aggregation.
    """

    def __init__(self):
        self.parser = RSSParserService()
        self.cache = get_cache(settings.cache_ttl_seconds)
        self.deduplicator = Deduplicator()
        self.content_extractor = get_content_extractor()

    async def aggregate(self, force_refresh: bool = False) -> AggregationResult:
        """
        Main aggregation method.
        Fetches from all sources, deduplicates, caches, and returns results.

        Args:
            force_refresh: If True, bypasses cache and fetches fresh data.

        Returns:
            AggregationResult with articles and metadata.
        """
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached_articles = self.cache.get_articles()
            if cached_articles is not None:
                return AggregationResult(
                    success=True,
                    articles=cached_articles,
                    total_count=len(cached_articles),
                    sources_fetched=len(get_feed_sources()),
                    sources_failed=0,
                    cached=True,
                    timestamp=datetime.now(timezone.utc)
                )

        # Prevent concurrent updates using async lock
        if self.cache.is_updating:
            # Return stale cache if available during update
            cached = self.cache.get_articles()
            if cached:
                return AggregationResult(
                    success=True,
                    articles=cached,
                    total_count=len(cached),
                    cached=True,
                    timestamp=datetime.now(timezone.utc),
                    errors=["Update in progress, returning cached data"]
                )

        async with self.cache._update_lock:
            return await self._perform_aggregation()

    async def _perform_aggregation(self) -> AggregationResult:
        """
        Perform the actual aggregation from all sources.
        """
        sources = get_feed_sources()
        errors: list[str] = []
        all_articles: list[Article] = []
        sources_failed = 0

        # Fetch all feeds concurrently
        tasks = [self.parser.fetch_feed(source) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for source, result in zip(sources, results):
            if isinstance(result, Exception):
                errors.append(f"{source.name}: {str(result)}")
                sources_failed += 1
                continue

            articles, error = result
            if error:
                errors.append(error)
                sources_failed += 1
            else:
                all_articles.extend(articles)

        # Deduplicate
        unique_articles = self.deduplicator.deduplicate(all_articles)

        # Normalize to UTC so mixed feed date formats never break sorting
        for article in unique_articles:
            published_at = article.published_at
            if published_at.tzinfo is None or published_at.tzinfo.utcoffset(published_at) is None:
                article.published_at = published_at.replace(tzinfo=timezone.utc)
            else:
                article.published_at = published_at.astimezone(timezone.utc)

        # Sort by published date (newest first)
        unique_articles.sort(key=lambda a: a.published_at, reverse=True)

        # Limit total articles
        unique_articles = unique_articles[:settings.max_total_articles]

        # Update cache
        self.cache.set_articles(unique_articles)

        return AggregationResult(
            success=sources_failed < len(sources),  # Success if at least one source worked
            articles=unique_articles,
            total_count=len(unique_articles),
            sources_fetched=len(sources),
            sources_failed=sources_failed,
            cached=False,
            timestamp=datetime.now(timezone.utc),
            errors=errors
        )

    def get_status(self) -> AggregationStatus:
        """
        Get current status of the aggregator.
        """
        last_update = self.cache.last_update_time
        return AggregationStatus(
            is_running=self.cache.is_updating,
            last_run=datetime.fromtimestamp(last_update) if last_update else None,
            last_success=datetime.fromtimestamp(last_update) if last_update and self.cache.is_valid() else None,
            articles_in_cache=self.cache.get_count(),
            cache_age_seconds=self.cache.get_age_seconds()
        )

    def get_cached_articles(
        self,
        source: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> list[Article]:
        """
        Get articles from cache with optional filtering.

        Args:
            source: Filter by source name (optional)
            limit: Maximum number of articles to return
            offset: Number of articles to skip

        Returns:
            List of articles matching criteria
        """
        articles = self.cache.get_articles() or []

        # Filter by source if specified
        if source:
            articles = [a for a in articles if a.source.lower() == source.lower()]

        # Apply pagination
        return articles[offset:offset + limit]

    def clear_cache(self) -> None:
        """Clear the article cache."""
        self.cache.clear()

    async def extract_content_for_article(self, article_url: str) -> Optional[Article]:
        """
        Extract content for a specific article by URL.
        First checks cache, then extracts if needed.

        Args:
            article_url: URL of the article

        Returns:
            Article with content, or None if not found
        """
        # Find article in cache
        articles = self.cache.get_articles() or []
        article = next((a for a in articles if a.link == article_url), None)

        if not article:
            return None

        # If content already extracted, return as-is
        if article.content_extracted and article.content:
            return article

        # Extract content
        updated_article = await self.content_extractor.extract_for_article(article)

        # Update in cache
        self._update_article_in_cache(updated_article)

        return updated_article

    async def extract_content_for_all(self, limit: int = 10) -> int:
        """
        Extract content for articles that don't have it yet.
        Runs in background, processes `limit` articles at a time.

        Args:
            limit: Maximum number of articles to process

        Returns:
            Number of articles processed
        """
        articles = self.cache.get_articles() or []

        # Find articles without content
        pending = [a for a in articles if not a.content_extracted][:limit]

        if not pending:
            return 0

        # Extract content
        updated = await self.content_extractor.extract_for_articles(pending)

        # Update cache
        for article in updated:
            self._update_article_in_cache(article)

        return len(updated)

    def _update_article_in_cache(self, updated_article: Article) -> None:
        """Update a single article in the cache without resetting TTL."""
        self.cache.update_article(updated_article)

    def get_article_by_url(self, url: str) -> Optional[Article]:
        """Get a specific article by URL from cache."""
        articles = self.cache.get_articles() or []
        return next((a for a in articles if a.link == url), None)


# Global aggregator instance
_aggregator: Optional[AggregatorService] = None


def get_aggregator() -> AggregatorService:
    """Get or create the global aggregator instance."""
    global _aggregator
    if _aggregator is None:
        _aggregator = AggregatorService()
    return _aggregator
