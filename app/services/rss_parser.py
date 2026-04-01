"""
RSS Parser service using feedparser.
Handles fetching and parsing individual RSS feeds.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional
import feedparser
import httpx
from dateutil import parser as date_parser

from app.models import Article, FeedSource
from app.config import settings
from app.utils.rate_limiter import get_rate_limiter


# Mapping of common RSS tag terms to normalized categories.
# Keys are lowercase. Add entries as new feeds introduce new terms.
CATEGORY_MAP: dict[str, str] = {
    # Technology
    "tech": "technology", "technology": "technology", "gadgets": "technology",
    "software": "technology", "hardware": "technology", "apps": "technology",
    "ai": "technology", "artificial intelligence": "technology",
    "cybersecurity": "technology", "security": "technology",
    "mobile": "technology", "smartphones": "technology", "computing": "technology",
    "internet": "technology", "it": "technology", "programming": "technology",
    "developer": "technology", "cloud": "technology", "robotics": "technology",
    "tech policy": "technology",
    # Science
    "science": "science", "space": "science", "environment": "science",
    "health": "science", "medicine": "science", "biology": "science",
    "physics": "science", "climate": "science", "research": "science",
    # Culture & Entertainment
    "culture": "culture", "entertainment": "culture", "gaming": "culture",
    "games": "culture", "movies": "culture", "music": "culture",
    "lifestyle": "culture", "food": "culture", "arts": "culture",
    "tv": "culture", "streaming": "culture", "social media": "culture",
    "features": "culture",
    # Business & Finance
    "business": "business", "finance": "business", "economy": "business",
    "markets": "business", "startups": "business", "venture capital": "business",
    "cryptocurrency": "business", "crypto": "business",
    # News & Politics
    "news": "news", "politics": "news", "policy": "news",
    "world": "news", "law": "news", "government": "news",
    # Cars & Automotive
    "cars": "cars", "automotive": "cars", "ev": "cars",
    "electric vehicles": "cars", "transportation": "cars",
}


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
        Fetch and parse a single RSS feed or news sitemap.
        Returns a tuple of (articles, error_message).
        Error message is None if successful.
        """
        if source.is_sitemap:
            return await self._fetch_sitemap(source)

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

    async def _fetch_sitemap(self, source: FeedSource) -> tuple[list[Article], Optional[str]]:
        """
        Fetch and parse a news sitemap (e.g. Reuters).
        News sitemaps use the sitemap + news:news XML namespace.
        Uses curl_cffi for fetching to bypass potential blocks.
        """
        try:
            await self.rate_limiter.acquire()

            from lxml import etree
            from curl_cffi import requests as crequests

            response = crequests.get(
                source.url,
                impersonate="chrome120",
                timeout=30,
            )
            if response.status_code != 200:
                return [], f"HTTP {response.status_code} fetching sitemap for {source.name}"
            content = response.content  # bytes for lxml

            root = etree.fromstring(content)
            ns = {
                "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
                "news": "http://www.google.com/schemas/sitemap-news/0.9",
                "image": "http://www.google.com/schemas/sitemap-image/1.1",
            }

            articles = []
            for url_elem in root.findall("sm:url", ns)[:settings.max_articles_per_feed]:
                article = self._parse_sitemap_entry(url_elem, ns, source)
                if article:
                    articles.append(article)

            return articles, None

        except Exception as e:
            return [], f"Error fetching sitemap {source.name}: {str(e)}"

    def _parse_sitemap_entry(self, url_elem, ns: dict, source: FeedSource) -> Optional[Article]:
        """Parse a single <url> entry from a news sitemap."""
        try:
            loc = url_elem.findtext("sm:loc", namespaces=ns)
            if not loc:
                return None

            # Skip non-English articles (e.g. /es/, /pt/, /ja/ paths)
            from urllib.parse import urlparse
            path_parts = urlparse(loc).path.strip("/").split("/")
            if path_parts and len(path_parts[0]) == 2 and path_parts[0].isalpha():
                return None  # Likely a language prefix like /es/, /pt/

            news_elem = url_elem.find("news:news", ns)
            title = ""
            published_at = None
            if news_elem is not None:
                title = news_elem.findtext("news:title", default="", namespaces=ns).strip()
                pub_date = news_elem.findtext("news:publication_date", namespaces=ns)
                if pub_date:
                    try:
                        published_at = date_parser.parse(pub_date)
                        if published_at.tzinfo is None:
                            published_at = published_at.replace(tzinfo=timezone.utc)
                        else:
                            published_at = published_at.astimezone(timezone.utc)
                    except Exception:
                        published_at = datetime.now(timezone.utc)

            if not title:
                return None

            # Get image
            image = None
            image_elem = url_elem.find("image:image", ns)
            if image_elem is not None:
                image = image_elem.findtext("image:loc", namespaces=ns)

            return Article(
                title=title,
                link=loc,
                source=source.name,
                published_at=published_at or datetime.now(timezone.utc),
                description=None,
                image=image,
                category=source.category,
            )
        except Exception:
            return None

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

            # Extract full content from RSS if feed provides it
            content = None
            content_extracted = False
            if source.full_content_in_feed:
                raw_content = self._get_full_content(entry)
                if raw_content:
                    content = raw_content
                    content_extracted = True

            category = self._infer_category(entry, source)

            return Article(
                title=title,
                link=link,
                source=source.name,
                published_at=published_at,
                description=description,
                image=image,
                category=category,
                content=content,
                content_extracted=content_extracted
            )

        except Exception:
            # Skip malformed entries
            return None

    def _infer_category(self, entry: dict, source: FeedSource) -> str:
        """
        Infer article category from RSS entry tags/categories.
        Falls back to the feed source's default category.
        """
        tags = entry.get("tags", [])
        for tag in tags:
            term = (tag.get("term") or "").strip().lower()
            if term in CATEGORY_MAP:
                return CATEGORY_MAP[term]
        # No recognized tag found — use the source default
        return source.category or "general"

    def _parse_date(self, entry: dict) -> datetime:
        """
        Parse publication date from entry.
        Handles multiple date field names and formats.
        """
        def to_utc(dt: datetime) -> datetime:
            if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        date_fields = ["published", "pubDate", "updated", "created"]

        for field in date_fields:
            date_str = entry.get(field) or entry.get(f"{field}_parsed")
            if date_str:
                try:
                    if isinstance(date_str, str):
                        return to_utc(date_parser.parse(date_str))
                    # feedparser sometimes returns time.struct_time
                    if hasattr(date_str, "tm_year"):
                        return to_utc(datetime(*date_str[:6]))
                except Exception:
                    continue

        # Fallback to current time if no date found
        return datetime.now(timezone.utc)

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

        # Fallback: some feeds embed thumbnails only in HTML snippets.
        html_candidates: list[str] = []
        content = entry.get("content", [])
        if content and isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    value = item.get("value")
                    if isinstance(value, str) and value.strip():
                        html_candidates.append(value)

        summary = entry.get("summary")
        description = entry.get("description")
        if isinstance(summary, str) and summary.strip():
            html_candidates.append(summary)
        if isinstance(description, str) and description.strip():
            html_candidates.append(description)

        for html_text in html_candidates:
            image_url = self._extract_image_from_html(html_text)
            if image_url:
                return image_url

        return None

    def _extract_image_from_html(self, html_text: str) -> Optional[str]:
        """Extract the first image URL from an HTML snippet."""
        import re

        if not html_text:
            return None

        # Prefer src / data-src, then fall back to the first srcset candidate.
        patterns = [
            r'<img[^>]+(?:src|data-src)=["\']([^"\']+)["\']',
            r'<img[^>]+srcset=["\']([^"\']+)["\']',
        ]

        for pattern in patterns:
            match = re.search(pattern, html_text, flags=re.IGNORECASE)
            if not match:
                continue

            url = match.group(1).strip()
            if not url:
                continue

            if " " in url:
                # srcset format: "url 320w, url2 640w"
                url = url.split(",", 1)[0].split(" ", 1)[0].strip()

            if url.startswith("//"):
                return f"https:{url}"
            return url

        return None

    def _get_full_content(self, entry: dict) -> Optional[str]:
        """
        Extract full article content from RSS entry (no truncation).
        Used for feeds that embed the complete article in the RSS description.
        """
        # Try summary first (feedparser often puts description content here)
        summary = entry.get("summary", "").strip()
        if summary and len(summary) > 500:
            return self._clean_html(summary)

        # Try description
        description = entry.get("description", "").strip()
        if description and len(description) > 500:
            return self._clean_html(description)

        # Try content
        content = entry.get("content", [])
        if content and isinstance(content, list):
            first_content = content[0].get("value", "")
            if first_content and len(first_content) > 500:
                return self._clean_html(first_content)

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
