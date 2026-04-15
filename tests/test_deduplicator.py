from app.services.deduplicator import Deduplicator


class TestNormalizeUrl:
    def test_strips_trailing_slash_and_lowercases(self):
        assert (
            Deduplicator.normalize_url("HTTPS://Example.com/Story/")
            == "https://example.com/story"
        )

    def test_drops_query_string_and_fragment(self):
        url = "https://example.com/story?utm_source=rss&ref=feed#top"
        assert Deduplicator.normalize_url(url) == "https://example.com/story"


class TestDeduplicate:
    def test_drops_exact_url_duplicates(self, make_article):
        a = make_article(title="A", link="https://example.com/x")
        b = make_article(title="B (different title same url)", link="https://example.com/x/")

        result = Deduplicator.deduplicate([a, b])

        assert len(result) == 1
        # First occurrence wins — order matters for "freshest source" semantics.
        assert result[0].title == "A"

    def test_drops_same_title_from_same_source(self, make_article):
        a = make_article(title="Stocks rally", link="https://nyt.com/1", source="NYTimes")
        b = make_article(title="Stocks rally", link="https://nyt.com/2", source="NYTimes")

        result = Deduplicator.deduplicate([a, b])

        assert len(result) == 1

    def test_keeps_same_title_across_different_sources(self, make_article):
        # This is the regression the deduplicator's docstring calls out:
        # title-only dedup was too aggressive — different outlets covering the
        # same event should both survive.
        a = make_article(title="Stocks rally", link="https://nyt.com/1", source="NYTimes")
        b = make_article(title="Stocks rally", link="https://wsj.com/1", source="WSJ")

        result = Deduplicator.deduplicate([a, b])

        assert len(result) == 2

    def test_empty_list(self):
        assert Deduplicator.deduplicate([]) == []


class TestMergeAndDeduplicate:
    def test_existing_articles_appear_first(self, make_article):
        existing = [make_article(title="Old", link="https://example.com/old")]
        incoming = [
            make_article(title="Old", link="https://example.com/old"),  # duplicate
            make_article(title="New", link="https://example.com/new"),
        ]

        merged = Deduplicator.merge_and_deduplicate(existing, incoming)

        assert [a.title for a in merged] == ["Old", "New"]
