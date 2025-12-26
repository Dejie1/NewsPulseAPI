"""
Supabase service for syncing articles to the database.
Matches the existing schema with sources, articles, and company_mentions tables.
"""

import os
from pathlib import Path
from typing import Optional
from datetime import datetime
from urllib.parse import urlparse
from supabase import create_client, Client

from app.models import Article, SentimentResult

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
            print(f"Error creating source: {e}")

        return None

    async def upsert_articles(self, articles: list[Article]) -> dict:
        """
        Upsert articles to Supabase.
        Maps to existing schema:
        - link -> url
        - description -> summary
        - image -> image_url
        - source -> looked up in sources table
        """
        if not articles:
            return {"inserted": 0, "errors": [], "message": "No articles to sync"}

        results = {"inserted": 0, "errors": [], "total": len(articles)}

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
                "full_content": article.content,  # Full scraped content
            }

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
            print(f"Error updating sentiment: {e}")
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

    async def delete_old_articles(self, days: int = 30) -> dict:
        """Delete articles older than specified days."""
        from datetime import timedelta

        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

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


# Global service instance
supabase_service = SupabaseService()
