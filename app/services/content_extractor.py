"""
Content extractor service using trafilatura.
Crawls article URLs and extracts the main content.
"""

import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import trafilatura
import requests
from urllib.parse import urlparse
from curl_cffi import requests as crequests  # Import the impersonating requests

from app.models import Article
from app.utils.rate_limiter import RateLimiter


# Browser-like headers to avoid bot detection
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}


class ContentExtractorService:
    """
    Service for extracting full article content from URLs.
    Uses trafilatura for robust content extraction.
    """

    def __init__(self, requests_per_second: float = 1.0):
        # Slower rate limit for crawling (be nice to servers)
        self.rate_limiter = RateLimiter(requests_per_second)
        # Thread pool for blocking trafilatura calls
        self._executor = ThreadPoolExecutor(max_workers=5)

    async def extract_content(self, url: str) -> Optional[str]:
        """
        Extract main content from a URL.

        Args:
            url: The article URL to crawl

        Returns:
            Extracted text content, or None if extraction failed
        """
        try:
            await self.rate_limiter.acquire()

            # Add jitter: random delay to break predictable patterns
            jitter = random.uniform(2.0, 6.0)
            await asyncio.sleep(jitter)

            # trafilatura is blocking, so run in thread pool
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                self._executor,
                self._extract_sync,
                url
            )
            return content

        except Exception as e:
            print(f"Content extraction failed for {url}: {e}")
            return None

    # Browser fingerprints to rotate through (curl_cffi supported versions)
    BROWSER_FINGERPRINTS = [
        "chrome110",
        "chrome120",
        "chrome124",
        "safari15_5",
        "safari17_0",
    ]

    def _extract_sync(self, url: str) -> Optional[str]:
        """
        Synchronous content extraction (runs in thread pool).
        Uses custom headers to mimic browser requests.
        Falls back to trafilatura's built-in fetcher if curl_cffi fails.
        """
        try:
            # Build headers with proper Referer for the domain
            parsed = urlparse(url)

            headers = {
                "Referer": f"{parsed.scheme}://{parsed.netloc}/",
                "Accept-Language": "en-US,en;q=0.9",
            }

            # Rotate browser fingerprint to avoid detection
            browser = random.choice(self.BROWSER_FINGERPRINTS)

            downloaded = None

            # Try curl_cffi first (browser impersonation)
            try:
                response = crequests.get(
                    url,
                    headers=headers,
                    timeout=30,  # Increased timeout for servers
                    allow_redirects=True,
                    impersonate=browser
                )

                if response.status_code == 200:
                    downloaded = response.text
                else:
                    print(f"[curl_cffi] HTTP {response.status_code} for {url}")
                    if response.status_code == 403:
                        print("[curl_cffi] Blocked by anti-bot protection, trying fallback...")

            except Exception as curl_error:
                print(f"[curl_cffi] Failed for {url}: {type(curl_error).__name__}: {curl_error}")

            # Fallback: Use trafilatura's built-in fetcher
            if not downloaded:
                print(f"[fallback] Trying trafilatura.fetch_url for {url}")
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    print(f"[fallback] Successfully fetched {url}")
                else:
                    print(f"[fallback] trafilatura.fetch_url also failed for {url}")
                    return None

            # Extract main content
            content = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
                favor_precision=True,
            )

            if content:
                print(f"[extract] Successfully extracted content from {url} ({len(content)} chars)")
            else:
                print(f"[extract] trafilatura.extract returned None for {url}")

            return content

        except requests.exceptions.Timeout:
            print(f"[error] Timeout fetching {url}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"[error] Request error for {url}: {e}")
            return None
        except Exception as e:
            print(f"[error] Extraction error for {url}: {type(e).__name__}: {e}")
            return None

    async def extract_for_article(self, article: Article) -> Article:
        """
        Extract content for an article and return updated article.

        Args:
            article: Article to extract content for

        Returns:
            Updated Article with content field populated
        """
        if article.content_extracted:
            return article

        content = await self.extract_content(article.link)

        # Create new article with content
        return Article(
            title=article.title,
            link=article.link,
            source=article.source,
            published_at=article.published_at,
            description=article.description,
            image=article.image,
            category=article.category,
            content=content,
            content_extracted=True
        )

    async def extract_for_articles(
        self,
        articles: list[Article],
        max_concurrent: int = 3
    ) -> list[Article]:
        """
        Extract content for multiple articles with concurrency limit.

        Args:
            articles: List of articles to process
            max_concurrent: Maximum concurrent extractions

        Returns:
            List of articles with content extracted
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_with_semaphore(article: Article) -> Article:
            async with semaphore:
                return await self.extract_for_article(article)

        tasks = [extract_with_semaphore(a) for a in articles]
        return await asyncio.gather(*tasks)


# Global extractor instance
_extractor: Optional[ContentExtractorService] = None


def get_content_extractor() -> ContentExtractorService:
    """Get or create the global content extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = ContentExtractorService()
    return _extractor
