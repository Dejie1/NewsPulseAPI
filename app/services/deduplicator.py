"""
Deduplication logic for articles.
Prevents duplicate articles from appearing in the feed.
"""

import hashlib
from typing import Optional
from app.models import Article


class Deduplicator:
    """
    Handles deduplication of articles based on URL or title.
    Uses multiple strategies to catch duplicates.
    """

    @staticmethod
    def normalize_url(url: str) -> str:
        """
        Normalize URL for comparison.
        Removes trailing slashes, query params, and fragments.
        """
        url = url.strip().lower()
        # Remove trailing slash
        url = url.rstrip("/")
        # Remove common tracking params (simplified)
        if "?" in url:
            url = url.split("?")[0]
        if "#" in url:
            url = url.split("#")[0]
        return url

    @staticmethod
    def normalize_title(title: str) -> str:
        """
        Normalize title for comparison.
        Lowercases and removes extra whitespace.
        """
        return " ".join(title.lower().split())

    @staticmethod
    def generate_fingerprint(article: Article) -> str:
        """
        Generate a unique fingerprint for an article.
        Uses URL as primary identifier, falls back to title hash.
        """
        normalized_url = Deduplicator.normalize_url(article.link)
        return hashlib.md5(normalized_url.encode()).hexdigest()

    @staticmethod
    def deduplicate(articles: list[Article]) -> list[Article]:
        """
        Remove duplicate articles from a list.
        Keeps the first occurrence of each unique article.
        Uses both URL and title-based deduplication.
        """
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        unique_articles: list[Article] = []

        for article in articles:
            normalized_url = Deduplicator.normalize_url(article.link)
            normalized_title = Deduplicator.normalize_title(article.title)

            # Skip if we've seen this URL or very similar title
            if normalized_url in seen_urls:
                continue
            if normalized_title in seen_titles:
                continue

            seen_urls.add(normalized_url)
            seen_titles.add(normalized_title)
            unique_articles.append(article)

        return unique_articles

    @staticmethod
    def merge_and_deduplicate(
        existing: list[Article],
        new_articles: list[Article]
    ) -> list[Article]:
        """
        Merge new articles with existing ones, removing duplicates.
        Preserves order with existing articles first.
        """
        # Combine lists and deduplicate
        combined = existing + new_articles
        return Deduplicator.deduplicate(combined)
