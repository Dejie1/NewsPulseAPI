import time

from app.services.cache import CacheService


class TestCacheService:
    def test_empty_cache_returns_none(self):
        cache = CacheService(ttl_seconds=60)
        assert cache.get_articles() is None
        assert cache.is_valid() is False
        assert cache.get_count() == 0

    def test_get_returns_copy_not_reference(self, make_article):
        # Callers must not be able to mutate cache state by editing the
        # returned list — a real bug we'd hit if `.copy()` were dropped.
        cache = CacheService(ttl_seconds=60)
        cache.set_articles([make_article(title="One")])

        first = cache.get_articles()
        assert first is not None
        first.clear()

        second = cache.get_articles()
        assert second is not None
        assert len(second) == 1

    def test_expired_entries_return_none(self, make_article):
        cache = CacheService(ttl_seconds=0)
        cache.set_articles([make_article()])
        # ttl=0 → already expired on next read
        time.sleep(0.01)
        assert cache.get_articles() is None
        assert cache.is_valid() is False

    def test_clear_resets_state(self, make_article):
        cache = CacheService(ttl_seconds=60)
        cache.set_articles([make_article()])
        cache.clear()
        assert cache.get_articles() is None
        assert cache.last_update_time is None

    def test_update_article_replaces_in_place(self, make_article):
        cache = CacheService(ttl_seconds=60)
        original = make_article(title="Before", link="https://example.com/x")
        cache.set_articles([original, make_article(title="Other", link="https://example.com/y")])
        update_time = cache.last_update_time

        updated = make_article(title="After", link="https://example.com/x")
        cache.update_article(updated)

        articles = cache.get_articles()
        assert articles is not None
        titles = {a.title for a in articles}
        assert titles == {"After", "Other"}
        # In-place update must not reset the TTL clock.
        assert cache.last_update_time == update_time
