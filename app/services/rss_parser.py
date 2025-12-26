"""
RSS Parser service using feedparser.
Handles fetching and parsing individual RSS feeds.
"""

import asyncio
from datetime import datetime
from typing import Optional
import feedparser
import httpx
from dateutil import parser as date_parser

from app.models import Article, FeedSource
from app.config import settings
from app.utils.rate_limiter import get_rate_limiter


class RSSParserService:
    """
    Service for fetching and parsing RSS feeds.
    Handles HTTP requests, parsing, and data normalization.
    """

    def __init__(self):
        self.rate_limiter = get_rate_limiter(settings.requests_per_second)
        self.timeout = settings.request_timeout_seconds

    async def fetch_feed(self, source: FeedSource) -> tuple[list[Article], Optional[str]]:
        """
        Fetch and parse a single RSS feed.
        Returns a tuple of (articles, error_message).
        Error message is None if successful.
        """
        try:
            # Rate limit before making request
            await self.rate_limiter.acquire()

            # Fetch the feed content
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    source.url,
                    headers={
                        "User-Agent": "NewsAggregator/1.0 (POC)",
                        "Accept": "application/rss+xml, application/xml, text/xml"
                    },
                    follow_redirects=True
                )
                response.raise_for_status()
                content = response.text

            # Parse the RSS content
            feed = feedparser.parse(content)

            if feed.bozo and not feed.entries:
                return [], f"Failed to parse feed: {feed.bozo_exception}"

            # Convert entries to Article objects
            articles = []
            for entry in feed.entries[:settings.max_articles_per_feed]:
                article = self._parse_entry(entry, source)
                if article:
                    articles.append(article)

            return articles, None

        except httpx.TimeoutException:
            return [], f"Timeout fetching {source.url}"
        except httpx.HTTPStatusError as e:
            return [], f"HTTP {e.response.status_code} for {source.url}"
        except Exception as e:
            return [], f"Error fetching {source.name}: {str(e)}"

    def _parse_entry(self, entry: dict, source: FeedSource) -> Optional[Article]:
        """
        Parse a single feed entry into an Article.
        Handles various RSS formats and field names.
        """
        try:
            # Get title (required)
            title = entry.get("title", "").strip()
            if not title:
                return None

            # Get link (required)
            link = entry.get("link", "").strip()
            if not link:
                return None

            # Get published date
            published_at = self._parse_date(entry)

            # Get description
            description = self._get_description(entry)

            # Get image
            image = self._get_image(entry)

            return Article(
                title=title,
                link=link,
                source=source.name,
                published_at=published_at,
                description=description,
                image=image,
                category=source.category
            )

        except Exception:
            # Skip malformed entries
            return None

    def _parse_date(self, entry: dict) -> datetime:
        """
        Parse publication date from entry.
        Handles multiple date field names and formats.
        """
        date_fields = ["published", "pubDate", "updated", "created"]

        for field in date_fields:
            date_str = entry.get(field) or entry.get(f"{field}_parsed")
            if date_str:
                try:
                    if isinstance(date_str, str):
                        return date_parser.parse(date_str)
                    # feedparser sometimes returns time.struct_time
                    if hasattr(date_str, "tm_year"):
                        return datetime(*date_str[:6])
                except Exception:
                    continue

        # Fallback to current time if no date found
        return datetime.utcnow()

    def _get_description(self, entry: dict) -> Optional[str]:
        """
        Extract description/summary from entry.
        Prefers summary over full content.
        """
        # Try summary first
        summary = entry.get("summary", "").strip()
        if summary:
            return self._clean_html(summary)[:500]  # Truncate for POC

        # Try description
        description = entry.get("description", "").strip()
        if description:
            return self._clean_html(description)[:500]

        # Try content
        content = entry.get("content", [])
        if content and isinstance(content, list):
            first_content = content[0].get("value", "")
            if first_content:
                return self._clean_html(first_content)[:500]

        return None

    def _get_image(self, entry: dict) -> Optional[str]:
        """
        Extract image URL from entry.
        Handles various RSS formats for images.
        """
        # Try media:thumbnail
        media_thumbnail = entry.get("media_thumbnail", [])
        if media_thumbnail and isinstance(media_thumbnail, list):
            return media_thumbnail[0].get("url")

        # Try media:content
        media_content = entry.get("media_content", [])
        if media_content and isinstance(media_content, list):
            for media in media_content:
                if media.get("medium") == "image" or media.get("type", "").startswith("image"):
                    return media.get("url")

        # Try enclosures
        enclosures = entry.get("enclosures", [])
        if enclosures:
            for enc in enclosures:
                if enc.get("type", "").startswith("image"):
                    return enc.get("href") or enc.get("url")

        # Try image field
        image = entry.get("image")
        if image:
            if isinstance(image, dict):
                return image.get("href") or image.get("url")
            if isinstance(image, str):
                return image

        return None

    def _clean_html(self, text: str) -> str:
        """
        Simple HTML tag removal.
        For POC - consider using proper HTML parser for production.
        """
        import re
        # Remove HTML tags
        clean = re.sub(r"<[^>]+>", "", text)
        # Normalize whitespace
        clean = " ".join(clean.split())
        return clean.strip()
