"""
Supabase service for syncing articles to the database.
Matches the existing schema with sources, articles, and company_mentions tables.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from urllib.parse import urlparse
from supabase import create_client, Client

from app.models import Article, SentimentResult

logger = logging.getLogger(__name__)

# Load .env file manually
def _load_env_file():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

_load_env_file()


def get_supabase_credentials() -> tuple[str, str]:
    """Get Supabase credentials from environment, supporting multiple prefixes."""
    # Prefer clean names, fall back to NEXT_PUBLIC_* for compatibility
    url = (
        os.getenv("SUPABASE_URL") or
        os.getenv("NEXT_PUBLIC_SUPABASE_URL") or
        ""
    )
    # Prefer service role key for server-side (bypasses RLS)
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY") or
        os.getenv("SUPABASE_ANON_KEY") or
        os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY") or
        ""
    )
    return url, key


class SupabaseService:
    """Service for interacting with Supabase database."""

    def __init__(self):
        self._client: Optional[Client] = None
        self._sources_cache: dict[str, int] = {}  # domain -> source_id
        self._companies_cache: dict[str, int] = {}  # ticker -> company_id

    @property
    def client(self) -> Client:
        """Lazy initialization of Supabase client."""
        if self._client is None:
            url, key = get_supabase_credentials()
            if not url or not key:
                raise ValueError(
                    "Supabase credentials not configured. "
                    "Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY in .env"
                )
            self._client = create_client(url, key)
        return self._client

    def is_configured(self) -> bool:
        """Check if Supabase is properly configured."""
        url, key = get_supabase_credentials()
        return bool(url and key)

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc or parsed.path.split('/')[0]
        except Exception:
            return "unknown"

    async def _get_or_create_source(self, source_name: str, article_url: str) -> Optional[int]:
        """
        Get existing source_id or create new source.
        Uses domain as unique identifier.
        """
        domain = self._extract_domain(article_url)

        # Check cache first
        if domain in self._sources_cache:
            return self._sources_cache[domain]

        try:
            # Try to find existing source by domain
            response = self.client.table("sources")\
                .select("id")\
                .eq("domain", domain)\
                .single()\
                .execute()

            if response.data:
                source_id = response.data["id"]
                self._sources_cache[domain] = source_id
                return source_id

        except Exception:
            pass  # Source doesn't exist, create it

        try:
            # Create new source
            response = self.client.table("sources").insert({
                "name": source_name,
                "domain": domain,
                "reliability_score": 0.5
            }).execute()

            if response.data:
                source_id = response.data[0]["id"]
                self._sources_cache[domain] = source_id
                return source_id

        except Exception as e:
            logger.error("Error creating source: %s", e)

        return None

    async def upsert_articles(self, articles: list[Article]) -> dict:
        """
        Upsert articles to Supabase.
        Maps to existing schema:
        - link -> url
        - description -> summary
        - image -> image_url
        - source -> looked up in sources table

        Preserves existing full_content if new content is None.
        """
        if not articles:
            return {"inserted": 0, "errors": [], "message": "No articles to sync"}

        results = {"inserted": 0, "errors": [], "total": len(articles)}

        # Fetch existing content for all URLs to preserve during upsert
        # This prevents overwriting existing full_content with NULL
        urls = [article.link for article in articles]
        existing_content: dict[str, str] = {}

        try:
            # Batch fetch existing content (only fetch URLs that exist)
            response = self.client.table("articles")\
                .select("url, full_content")\
                .in_("url", urls)\
                .not_.is_("full_content", "null")\
                .execute()

            existing_content = {
                row["url"]: row["full_content"]
                for row in (response.data or [])
            }
            logger.info("[upsert] Found %s articles with existing content to preserve", len(existing_content))
        except Exception as e:
            logger.error("[upsert] Warning: Could not fetch existing content: %s", e)

        # Process articles one by one to handle source lookups
        rows = []
        for article in articles:
            # Get or create source
            source_id = await self._get_or_create_source(article.source, article.link)

            row = {
                "title": article.title,
                "url": article.link,  # link -> url
                "summary": article.description,  # description -> summary
                "image_url": article.image,  # image -> image_url
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "category": article.category,  # Category from feed source
            }

            # Use new content if available, otherwise preserve existing content
            if article.content is not None:
                row["full_content"] = article.content
            elif article.link in existing_content:
                row["full_content"] = existing_content[article.link]
            # If neither, don't include full_content (will be NULL for new articles)

            if source_id:
                row["source_id"] = source_id

            rows.append(row)

        try:
            # Upsert with conflict resolution on 'url'
            response = self.client.table("articles").upsert(
                rows,
                on_conflict="url"
            ).execute()

            results["inserted"] = len(response.data) if response.data else 0
            results["message"] = f"Successfully synced {results['inserted']} articles"

        except Exception as e:
            results["errors"].append(str(e))
            results["message"] = f"Error syncing articles: {str(e)}"

        return results

    async def update_article_sentiment(self, article_url: str, compound_score: float) -> bool:
        """
        Update the overall_sentiment field on an article.
        """
        try:
            self.client.table("articles")\
                .update({"overall_sentiment": compound_score})\
                .eq("url", article_url)\
                .execute()
            return True
        except Exception as e:
            logger.error("Error updating sentiment: %s", e)
            return False

    async def upsert_sentiments(self, sentiments: list[SentimentResult]) -> dict:
        """
        Update overall_sentiment on articles.
        Your schema stores sentiment directly on the articles table.
        """
        if not sentiments:
            return {"updated": 0, "errors": [], "message": "No sentiments to sync"}

        results = {"updated": 0, "errors": [], "total": len(sentiments)}

        for sentiment in sentiments:
            try:
                self.client.table("articles")\
                    .update({"overall_sentiment": sentiment.compound})\
                    .eq("url", sentiment.article_url)\
                    .execute()
                results["updated"] += 1
            except Exception as e:
                results["errors"].append(f"{sentiment.article_url}: {str(e)}")

        results["message"] = f"Updated sentiment for {results['updated']} articles"
        return results

    async def get_articles(
        self,
        limit: int = 50,
        source: Optional[str] = None,
        order_by: str = "published_at",
        ascending: bool = False
    ) -> list[dict]:
        """
        Fetch articles from Supabase with source info.
        """
        try:
            query = self.client.table("articles")\
                .select("*, sources(name, domain, logo_url)")

            if source:
                # Filter by source name via join
                query = query.eq("sources.name", source)

            query = query.order(order_by, desc=not ascending).limit(limit)
            response = query.execute()

            return response.data if response.data else []

        except Exception as e:
            raise Exception(f"Error fetching articles from Supabase: {str(e)}")

    async def get_article_by_url(self, url: str) -> Optional[dict]:
        """Get a single article by URL."""
        try:
            response = self.client.table("articles")\
                .select("*, sources(name, domain, logo_url)")\
                .eq("url", url)\
                .single()\
                .execute()

            return response.data if response.data else None

        except Exception:
            return None

    async def get_urls_with_content(self) -> set[str]:
        """
        Get set of article URLs that already have full_content extracted.
        Used to skip re-extraction during sync.
        """
        try:
            response = self.client.table("articles")\
                .select("url")\
                .not_.is_("full_content", "null")\
                .execute()

            return {row["url"] for row in (response.data or [])}

        except Exception as e:
            logger.error("Error fetching URLs with content: %s", e)
            return set()

    async def delete_old_articles(self, days: int = 30) -> dict:
        """Delete articles older than specified days."""
        from datetime import timedelta

        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        try:
            response = self.client.table("articles")\
                .delete()\
                .lt("published_at", cutoff_date)\
                .execute()

            deleted_count = len(response.data) if response.data else 0
            return {
                "deleted": deleted_count,
                "message": f"Deleted {deleted_count} articles older than {days} days"
            }

        except Exception as e:
            return {"deleted": 0, "error": str(e)}

    async def get_sources(self) -> list[dict]:
        """Get all sources from the database."""
        try:
            response = self.client.table("sources")\
                .select("*")\
                .order("name")\
                .execute()

            return response.data if response.data else []

        except Exception as e:
            raise Exception(f"Error fetching sources: {str(e)}")


    # =========================================================================
    # COMPANY-RELATED METHODS
    # =========================================================================

    async def get_all_companies(self) -> list[dict]:
        """Get all companies from the database."""
        try:
            response = self.client.table("companies")\
                .select("*")\
                .order("name")\
                .execute()

            return response.data if response.data else []

        except Exception as e:
            logger.error("Error fetching companies: %s", e)
            return []

    async def get_company_by_ticker(self, ticker: str) -> Optional[int]:
        """
        Get company_id by ticker symbol.

        Args:
            ticker: Stock ticker (e.g., "AAPL")

        Returns:
            Company ID or None if not found
        """
        # Check cache first
        if ticker in self._companies_cache:
            return self._companies_cache[ticker]

        try:
            response = self.client.table("companies")\
                .select("id")\
                .eq("ticker", ticker)\
                .single()\
                .execute()

            if response.data:
                company_id = response.data["id"]
                self._companies_cache[ticker] = company_id
                return company_id

        except Exception as e:
            logger.error("Error fetching company %s: %s", ticker, e)

        return None

    async def get_article_id_by_url(self, url: str) -> Optional[int]:
        """Get article ID by URL."""
        try:
            response = self.client.table("articles")\
                .select("id")\
                .eq("url", url)\
                .single()\
                .execute()

            return response.data["id"] if response.data else None

        except Exception:
            return None

    async def get_articles_for_analysis(
        self,
        limit: int = 50,
        only_unanalyzed: bool = True
    ) -> list[dict]:
        """
        Get articles that need company analysis.

        Args:
            limit: Maximum articles to return
            only_unanalyzed: If True, exclude articles already in company_mentions

        Returns:
            List of article dicts with id, url, title, full_content
        """
        try:
            query = self.client.table("articles")\
                .select("id, url, title, full_content")\
                .not_.is_("full_content", "null")\
                .order("published_at", desc=True)\
                .limit(limit)

            response = query.execute()
            articles = response.data if response.data else []

            if only_unanalyzed and articles:
                # Filter out articles that already have mentions
                # Get article IDs that have mentions
                article_ids = [a["id"] for a in articles]
                mentions_response = self.client.table("company_mentions")\
                    .select("article_id")\
                    .in_("article_id", article_ids)\
                    .execute()

                analyzed_ids = set(
                    m["article_id"] for m in (mentions_response.data or [])
                )

                articles = [a for a in articles if a["id"] not in analyzed_ids]

            return articles

        except Exception as e:
            logger.error("Error fetching articles for analysis: %s", e)
            return []

    async def upsert_company_mentions(self, mentions: list[dict]) -> dict:
        """
        Upsert company mentions to Supabase.

        Each mention dict should have:
        - article_id: int
        - company_id: int
        - sentiment_score: float
        - confidence_score: float
        - context_sentence: str

        Returns:
            Result dict with counts and errors
        """
        if not mentions:
            return {"inserted": 0, "message": "No mentions to sync"}

        results = {"inserted": 0, "errors": [], "total": len(mentions)}

        try:
            # Deduplicate by (article_id, company_id) - keep highest confidence
            unique_mentions: dict[tuple[int, int], dict] = {}
            for mention in mentions:
                key = (mention["article_id"], mention["company_id"])
                if key not in unique_mentions:
                    unique_mentions[key] = mention
                else:
                    # Keep the one with higher confidence, or average sentiments
                    existing = unique_mentions[key]
                    if mention["confidence_score"] > existing["confidence_score"]:
                        # Average the sentiment scores
                        avg_sentiment = (existing["sentiment_score"] + mention["sentiment_score"]) / 2
                        mention["sentiment_score"] = avg_sentiment
                        unique_mentions[key] = mention

            deduped_mentions = list(unique_mentions.values())

            # Upsert all mentions
            response = self.client.table("company_mentions").upsert(
                deduped_mentions,
                on_conflict="article_id,company_id"
            ).execute()

            results["inserted"] = len(response.data) if response.data else 0
            results["message"] = f"Successfully synced {results['inserted']} company mentions"

        except Exception as e:
            results["errors"].append(str(e))
            results["message"] = f"Error syncing mentions: {str(e)}"

        return results

    async def get_company_mentions(
        self,
        ticker: Optional[str] = None,
        limit: int = 50,
        days: int = 7
    ) -> list[dict]:
        """
        Get company mentions with article info.

        Args:
            ticker: Filter by company ticker (optional)
            limit: Maximum mentions to return
            days: Only mentions from last N days

        Returns:
            List of mention dicts with article and company info
        """
        from datetime import timedelta

        try:
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

            query = self.client.table("company_mentions")\
                .select("*, articles(id, title, url, published_at), companies(name, ticker)")\
                .gte("created_at", cutoff_date)\
                .order("created_at", desc=True)\
                .limit(limit)

            if ticker:
                # Need to filter by company ticker via join
                # First get company_id
                company_id = await self.get_company_by_ticker(ticker)
                if company_id:
                    query = query.eq("company_id", company_id)
                else:
                    return []

            response = query.execute()
            return response.data if response.data else []

        except Exception as e:
            logger.error("Error fetching company mentions: %s", e)
            return []

    async def seed_companies(self) -> dict:
        """
        Seed the companies table with Magnificent 7.

        Returns:
            Result dict with count
        """
        companies = [
            {
                "name": "Apple",
                "ticker": "AAPL",
                "icon_url": "https://logo.clearbit.com/apple.com"
            },
            {
                "name": "Microsoft",
                "ticker": "MSFT",
                "icon_url": "https://logo.clearbit.com/microsoft.com"
            },
            {
                "name": "Google",
                "ticker": "GOOGL",
                "icon_url": "https://logo.clearbit.com/google.com"
            },
            {
                "name": "Amazon",
                "ticker": "AMZN",
                "icon_url": "https://logo.clearbit.com/amazon.com"
            },
            {
                "name": "Meta",
                "ticker": "META",
                "icon_url": "https://logo.clearbit.com/meta.com"
            },
            {
                "name": "Tesla",
                "ticker": "TSLA",
                "icon_url": "https://logo.clearbit.com/tesla.com"
            },
            {
                "name": "Nvidia",
                "ticker": "NVDA",
                "icon_url": "https://logo.clearbit.com/nvidia.com"
            },
        ]

        try:
            response = self.client.table("companies").upsert(
                companies,
                on_conflict="ticker"
            ).execute()

            return {
                "seeded": len(response.data) if response.data else 0,
                "message": "Magnificent 7 companies seeded successfully"
            }

        except Exception as e:
            return {"seeded": 0, "error": str(e)}

    async def update_daily_metrics(
        self,
        date: str,
        company_id: int
    ) -> bool:
        """
        Update company_daily_metrics aggregation for a specific day.

        Args:
            date: Date string (YYYY-MM-DD)
            company_id: Company ID

        Returns:
            True if successful
        """
        try:
            # Get all mentions for this company on this date
            response = self.client.table("company_mentions")\
                .select("sentiment_score")\
                .eq("company_id", company_id)\
                .gte("created_at", f"{date}T00:00:00")\
                .lt("created_at", f"{date}T23:59:59")\
                .execute()

            mentions = response.data or []

            if not mentions:
                return True

            avg_sentiment = sum(m["sentiment_score"] for m in mentions) / len(mentions)

            self.client.table("company_daily_metrics").upsert({
                "date": date,
                "company_id": company_id,
                "avg_sentiment": avg_sentiment,
                "article_volume": len(mentions)
            }, on_conflict="date,company_id").execute()

            return True

        except Exception as e:
            logger.error("Error updating daily metrics: %s", e)
            return False

    async def get_company_daily_metrics(
        self,
        ticker: str,
        days: int = 7
    ) -> list[dict]:
        """
        Get daily metrics for a company.

        Args:
            ticker: Company ticker
            days: Number of days to fetch

        Returns:
            List of daily metric records
        """
        from datetime import timedelta

        try:
            company_id = await self.get_company_by_ticker(ticker)
            if not company_id:
                return []

            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

            response = self.client.table("company_daily_metrics")\
                .select("*")\
                .eq("company_id", company_id)\
                .gte("date", cutoff_date)\
                .order("date", desc=True)\
                .execute()

            return response.data if response.data else []

        except Exception as e:
            logger.error("Error fetching daily metrics: %s", e)
            return []


# Global service instance
supabase_service = SupabaseService()
