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


# Global settings instance
settings = Settings()


# Feed source configuration
FEED_SOURCES: list[FeedSource] = [
    FeedSource(
        name="NYTimes Technology",
        url="https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
        category="technology"
    ),
    FeedSource(
        name="Wired Business",
        url="https://www.wired.com/feed/category/business/latest/rss",
        category="business"
    ),
    FeedSource(
        name="Wired Culture",
        url="https://www.wired.com/feed/category/culture/latest/rss",
        category="culture"
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
