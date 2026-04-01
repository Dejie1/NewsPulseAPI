"""
Configuration for the news aggregator service.
Feed sources and settings are defined here for easy modification.
"""

from pydantic_settings import BaseSettings
from app.models import FeedSource


class Settings(BaseSettings):
    """Application settings."""

    # API Settings
    app_name: str = "News Aggregator API"
    debug: bool = False

    # Cache settings
    cache_ttl_seconds: int = 300  # 5 minutes

    # Rate limiting
    requests_per_second: float = 2.0  # Max requests per second to external feeds
    request_timeout_seconds: int = 10

    # Aggregation settings
    max_articles_per_feed: int = 50
    max_total_articles: int = 200

    class Config:
        env_file = ".env"
        env_prefix = "AGGREGATOR_"
        extra = "ignore"  # Ignore NEXT_PUBLIC_* and other env vars


# Global settings instance
settings = Settings()


# Feed source configuration
FEED_SOURCES: list[FeedSource] = [
    FeedSource(
        name="Ars Technica",
        url="https://feeds.arstechnica.com/arstechnica/index",
        category="technology"
    ),
    FeedSource(
        name="Wired Culture",
        url="https://www.wired.com/feed/category/culture/latest/rss",
        category="culture"
    ),
    FeedSource(
        name="Fast Company",
        url="https://www.fastcompany.com/latest/rss",
        category="technology",
        full_content_in_feed=True
    ),
    FeedSource(
        name="The Edge Malaysia",
        url="https://news.google.com/rss/search?q=https%3A%2F%2Ftheedgemalaysia.com%2F&hl=en-MY&gl=MY&ceid=MY%3Aen",
        category="news"
    ),
    FeedSource(
        name="Free Malaysia Today",
        url="https://cms.freemalaysiatoday.com/feed",
        category="news",
        full_content_in_feed=True
    ),
    # FeedSource(
    #     name="Reuters",
    #     url="https://www.reuters.com/arc/outboundfeeds/news-sitemap/?outputType=xml",
    #     category="news",
    #     is_sitemap=True
    # ),
    # FeedSource(
    #     name="Bloomberg",
    #     url="https://news.google.com/rss/search?q=site%3Abloomberg.com&hl=en-US&gl=US&ceid=US%3Aen",
    #     category="business"
    # ),
    FeedSource(
        name="The Verge",
        url="https://www.theverge.com/rss/index.xml",
        category="technology"
    ),
    FeedSource(
        name="SoyaCincau",
        url="https://soyacincau.com/feed/",
        category="technology",
        full_content_in_feed=True
    ),
]


def get_feed_sources() -> list[FeedSource]:
    """
    Get all configured feed sources.
    This function can be extended to load from database or external config.
    """
    return FEED_SOURCES


def add_feed_source(source: FeedSource) -> None:
    """Add a new feed source dynamically."""
    FEED_SOURCES.append(source)
