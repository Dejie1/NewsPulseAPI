"""
Content extractor service using trafilatura.
Crawls article URLs and extracts the main content.
Falls back to Scrapling's StealthyFetcher (headful browser) for anti-bot protected sites.
"""

import asyncio
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union
import trafilatura
import requests
from urllib.parse import urlparse
from curl_cffi import requests as crequests  # Import the impersonating requests

logger = logging.getLogger(__name__)

from app.models import Article
from app.utils.rate_limiter import RateLimiter


def _get_headless_mode() -> Union[bool, str]:
    """Get browser headless mode from env. 'virtual' for servers, False for local testing."""
    mode = os.environ.get("AGGREGATOR_BROWSER_HEADLESS", "false").lower()
    if mode == "virtual":
        return "virtual"
    return mode == "true"


# Domains that require browser-based fetching (anti-bot protected)
BROWSER_REQUIRED_DOMAINS = {
    "reuters.com",
    "arstechnica.com",
}

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

    @staticmethod
    def _is_browser_required(url: str) -> bool:
        """Check if a URL's domain requires browser-based fetching."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower().removeprefix("www.")
        return any(domain == d or domain.endswith("." + d) for d in BROWSER_REQUIRED_DOMAINS)

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
            logger.error("Content extraction failed for %s: %s", url, e)
            return None

    # Browser fingerprints to rotate through (curl_cffi supported versions)
    BROWSER_FINGERPRINTS = [
        "chrome110",
        "chrome120",
        "chrome124",
        "safari15_5",
        "safari17_0",
    ]

    def _fetch_with_browser(self, url: str) -> Optional[str]:
        """
        Fetch page HTML using Scrapling's StealthyFetcher in headful browser mode.
        This bypasses anti-bot protections that block curl_cffi and trafilatura.
        """
        try:
            from scrapling.fetchers import StealthyFetcher

            headless = _get_headless_mode()
            mode_label = "virtual" if headless == "virtual" else ("headless" if headless else "headful")
            logger.info("[browser] Fetching %s with StealthyFetcher (%s mode)...", url, mode_label)
            page = StealthyFetcher.fetch(
                url,
                headless=headless,
                network_idle=True,
                block_webrtc=True,
                humanize=True,
                disable_ads=True,
            )

            html = page.html_content
            if html:
                logger.info("[browser] Successfully fetched %s (%d chars of HTML)", url, len(html))
                return html
            else:
                logger.warning("[browser] StealthyFetcher returned empty content for %s", url)
                return None

        except Exception as e:
            logger.error("[browser] Failed for %s: %s: %s", url, type(e).__name__, e)
            return None

    def _extract_sync(self, url: str) -> Optional[str]:
        """
        Synchronous content extraction (runs in thread pool).
        For browser-required domains (e.g. reuters.com), goes straight to StealthyFetcher.
        Otherwise tries curl_cffi → trafilatura → StealthyFetcher as fallback chain.
        """
        try:
            parsed = urlparse(url)
            use_browser = self._is_browser_required(url)

            downloaded = None

            if use_browser:
                # Skip HTTP-based fetchers for known anti-bot domains
                logger.info("[router] %s requires browser — skipping curl_cffi/trafilatura", parsed.netloc)
                downloaded = self._fetch_with_browser(url)
                if not downloaded:
                    return None
            else:
                # Build headers with proper Referer for the domain
                headers = {
                    "Referer": f"{parsed.scheme}://{parsed.netloc}/",
                    "Accept-Language": "en-US,en;q=0.9",
                }

                # Rotate browser fingerprint to avoid detection
                browser = random.choice(self.BROWSER_FINGERPRINTS)

                # Try curl_cffi first (browser impersonation)
                try:
                    response = crequests.get(
                        url,
                        headers=headers,
                        timeout=30,
                        allow_redirects=True,
                        impersonate=browser
                    )

                    if response.status_code == 200:
                        downloaded = response.text
                    else:
                        logger.warning("[curl_cffi] HTTP %s for %s", response.status_code, url)
                        if response.status_code == 403:
                            logger.warning("[curl_cffi] Blocked by anti-bot protection, trying fallback...")

                except Exception as curl_error:
                    logger.error("[curl_cffi] Failed for %s: %s: %s", url, type(curl_error).__name__, curl_error)

                # Fallback 1: trafilatura's built-in fetcher
                if not downloaded:
                    logger.warning("[fallback] Trying trafilatura.fetch_url for %s", url)
                    downloaded = trafilatura.fetch_url(url)
                    if downloaded:
                        logger.info("[fallback] Successfully fetched %s", url)

                # Fallback 2: StealthyFetcher headful browser
                if not downloaded:
                    logger.warning("[fallback] HTTP fetchers failed, trying browser for %s", url)
                    downloaded = self._fetch_with_browser(url)
                    if not downloaded:
                        logger.error("[fallback] All fetchers failed for %s", url)
                        return None

            # Extract main content from HTML
            content = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
                favor_precision=True,
                deduplicate=True,
            )

            # Strip leading cookie/consent banner text that trafilatura sometimes picks up
            if content:
                lines = content.split("\n")
                boilerplate_keywords = {
                    "cookie", "consent", "tracking", "privacy", "gdpr",
                    "advertising", "analytics", "opt out", "opt-out",
                    "social media", "essential", "third party", "third-party",
                    "manage preferences", "your choices", "do not sell",
                    "these cookies", "this website uses", "audience measurement",
                    "measurement cookies", "traffic sources", "performance statistics",
                    "functioning of the site", "browsing experience",
                }
                # Find last boilerplate line, then start after it
                last_boilerplate = -1
                for i, line in enumerate(lines):
                    lower = line.strip().lower()
                    if any(kw in lower for kw in boilerplate_keywords):
                        last_boilerplate = i
                    # Stop scanning once we're well past the banner (articles don't have
                    # consent text 50+ lines in)
                    if i - last_boilerplate > 5 and last_boilerplate >= 0:
                        break
                if last_boilerplate >= 0:
                    # Skip to after the last boilerplate line, also dropping short
                    # leftover lines (like "On", "Off", section headers)
                    start = last_boilerplate + 1
                    while start < len(lines) and len(lines[start].strip()) < 20:
                        start += 1
                    if start < len(lines):
                        content = "\n".join(lines[start:]).strip()

            if content:
                logger.info("[extract] Successfully extracted content from %s (%d chars)", url, len(content))
            else:
                logger.warning("[extract] trafilatura.extract returned None for %s", url)

            return content

        except requests.exceptions.Timeout:
            logger.error("[error] Timeout fetching %s", url)
            return None
        except requests.exceptions.RequestException as e:
            logger.error("[error] Request error for %s: %s", url, e)
            return None
        except Exception as e:
            logger.error("[error] Extraction error for %s: %s: %s", url, type(e).__name__, e)
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

    def shutdown(self) -> None:
        """Shut down the thread pool executor."""
        self._executor.shutdown(wait=False)

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
