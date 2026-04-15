from datetime import datetime, timezone

import pytest

from app.models import Article


@pytest.fixture
def make_article():
    """Build an Article with sensible defaults; override per-test as needed."""

    def _make(
        title: str = "Headline",
        link: str = "https://example.com/story",
        source: str = "Example",
        published_at: datetime | None = None,
    ) -> Article:
        return Article(
            title=title,
            link=link,
            source=source,
            published_at=published_at or datetime.now(timezone.utc),
        )

    return _make
