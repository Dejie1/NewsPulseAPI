"""
Simple in-memory cache for aggregated articles.
For POC purposes - can be replaced with Redis/database later.
"""

import asyncio
import time
from typing import Optional
from app.models import Article


class CacheService:
    """
    In-memory cache for storing aggregated articles.
    Thread-safe for single-process use.
    """

    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._articles: list[Article] = []
        self._last_update: Optional[float] = None
        self._update_lock = asyncio.Lock()

    def get_articles(self) -> Optional[list[Article]]:
        """
        Get cached articles if they exist and aren't expired.
        Returns None if cache is empty or expired.
        """
        if not self._articles or self._last_update is None:
            return None

        age = time.time() - self._last_update
        if age > self.ttl_seconds:
            return None

        return self._articles.copy()

    def set_articles(self, articles: list[Article]) -> None:
        """Store articles in cache with current timestamp."""
        self._articles = articles.copy()
        self._last_update = time.time()

    def clear(self) -> None:
        """Clear the cache."""
        self._articles = []
        self._last_update = None

    def is_valid(self) -> bool:
        """Check if cache has valid (non-expired) data."""
        if not self._articles or self._last_update is None:
            return False
        return (time.time() - self._last_update) <= self.ttl_seconds

    def get_age_seconds(self) -> Optional[float]:
        """Get the age of the cache in seconds."""
        if self._last_update is None:
            return None
        return time.time() - self._last_update

    def get_count(self) -> int:
        """Get number of articles in cache."""
        return len(self._articles)

    @property
    def is_updating(self) -> bool:
        """Check if cache is currently being updated."""
        return self._update_lock.locked()

    @property
    def last_update_time(self) -> Optional[float]:
        """Get the timestamp of last update."""
        return self._last_update

    def update_article(self, updated_article: Article) -> None:
        """Update a single article in-place without resetting TTL."""
        for i, a in enumerate(self._articles):
            if a.link == updated_article.link:
                self._articles[i] = updated_article
                break


# Global cache instance
_cache: Optional[CacheService] = None


def get_cache(ttl_seconds: int = 300) -> CacheService:
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        _cache = CacheService(ttl_seconds)
    return _cache
