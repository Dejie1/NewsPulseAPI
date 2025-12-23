"""
Content-based recommendation service using TF-IDF and cosine similarity.
Recommends similar articles based on text content.
"""

from typing import Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models import Article


class RecommendationResult:
    """A single recommendation result."""

    def __init__(self, article: Article, similarity_score: float):
        self.article = article
        self.similarity_score = similarity_score

    def to_dict(self) -> dict:
        return {
            "title": self.article.title,
            "link": self.article.link,
            "source": self.article.source,
            "similarity_score": round(self.similarity_score, 4),
            "published_at": self.article.published_at.isoformat()
        }


class RecommenderService:
    """
    Content-based recommender using TF-IDF vectorization and cosine similarity.

    How it works:
    1. TF-IDF (Term Frequency-Inverse Document Frequency):
       - Converts text into numerical vectors
       - Words that appear frequently in one doc but rarely in others get higher scores
       - Common words like "the", "a", "is" get low scores (stop words removed)

    2. Cosine Similarity:
       - Measures the angle between two vectors
       - 1.0 = identical, 0.0 = completely different
       - Used to find articles with similar content

    Example:
        Article A: "Apple releases new iPhone with AI features"
        Article B: "Google launches Pixel with machine learning"
        Article C: "Stock market crashes amid economic fears"

        A and B are similar (tech, phones, AI/ML) → high cosine similarity
        A and C are different (tech vs finance) → low cosine similarity
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer(
            stop_words='english',      # Remove common English words
            max_features=5000,         # Limit vocabulary size
            ngram_range=(1, 2),        # Use single words and bigrams
            min_df=1,                  # Minimum document frequency
            max_df=0.95                # Ignore words in >95% of docs
        )
        self._tfidf_matrix = None
        self._articles: list[Article] = []
        self._article_index: dict[str, int] = {}  # URL -> index mapping
        self._is_fitted = False

    def fit(self, articles: list[Article]) -> None:
        """
        Fit the recommender with articles.
        This builds the TF-IDF matrix and similarity scores.

        Args:
            articles: List of articles to use for recommendations
        """
        if not articles:
            self._is_fitted = False
            return

        self._articles = articles

        # Build corpus from content/description
        corpus = []
        for i, article in enumerate(articles):
            # Combine title and content/description for better matching
            text = article.title
            if article.content:
                text += " " + article.content
            elif article.description:
                text += " " + article.description
            corpus.append(text)

            # Build index
            self._article_index[article.link] = i

        # Fit and transform
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)
        self._is_fitted = True

    def get_recommendations(
        self,
        article_url: str,
        n_recommendations: int = 5
    ) -> list[RecommendationResult]:
        """
        Get article recommendations based on a given article.

        Args:
            article_url: URL of the article to find similar articles for
            n_recommendations: Number of recommendations to return

        Returns:
            List of RecommendationResult with similar articles
        """
        if not self._is_fitted or article_url not in self._article_index:
            return []

        # Get the index of the query article
        idx = self._article_index[article_url]

        # Calculate cosine similarity with all other articles
        article_vector = self._tfidf_matrix[idx]
        similarities = cosine_similarity(article_vector, self._tfidf_matrix).flatten()

        # Get indices sorted by similarity (descending)
        # Exclude the article itself (index 0 after sorting will be the article itself)
        similar_indices = similarities.argsort()[::-1]

        # Get top N recommendations (excluding the query article)
        recommendations = []
        for sim_idx in similar_indices:
            if sim_idx == idx:
                continue  # Skip the query article itself
            if len(recommendations) >= n_recommendations:
                break

            recommendations.append(RecommendationResult(
                article=self._articles[sim_idx],
                similarity_score=float(similarities[sim_idx])
            ))

        return recommendations

    def get_recommendations_for_text(
        self,
        text: str,
        n_recommendations: int = 5
    ) -> list[RecommendationResult]:
        """
        Get recommendations based on arbitrary text.
        Useful for finding articles related to a search query.

        Args:
            text: Text to find similar articles for
            n_recommendations: Number of recommendations to return

        Returns:
            List of RecommendationResult
        """
        if not self._is_fitted:
            return []

        # Transform the query text using the fitted vectorizer
        query_vector = self._vectorizer.transform([text])

        # Calculate similarity
        similarities = cosine_similarity(query_vector, self._tfidf_matrix).flatten()

        # Get top N
        top_indices = similarities.argsort()[::-1][:n_recommendations]

        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include if there's some similarity
                recommendations.append(RecommendationResult(
                    article=self._articles[idx],
                    similarity_score=float(similarities[idx])
                ))

        return recommendations

    def get_similar_from_source(
        self,
        article_url: str,
        source: str,
        n_recommendations: int = 3
    ) -> list[RecommendationResult]:
        """
        Get recommendations from a specific source.
        Useful for "More from NYTimes" type features.

        Args:
            article_url: URL of the reference article
            source: Source name to filter by
            n_recommendations: Number of recommendations

        Returns:
            List of RecommendationResult from the specified source
        """
        all_recs = self.get_recommendations(article_url, n_recommendations * 3)
        filtered = [r for r in all_recs if r.article.source == source]
        return filtered[:n_recommendations]

    @property
    def is_ready(self) -> bool:
        """Check if the recommender has been fitted with data."""
        return self._is_fitted

    @property
    def article_count(self) -> int:
        """Number of articles in the recommender."""
        return len(self._articles)


# Global recommender instance
_recommender: Optional[RecommenderService] = None


def get_recommender() -> RecommenderService:
    """Get or create the global recommender instance."""
    global _recommender
    if _recommender is None:
        _recommender = RecommenderService()
    return _recommender
