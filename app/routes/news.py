"""
FastAPI routes for the news aggregator.
Provides endpoints for triggering aggregation and fetching articles.
"""

from typing import Optional
from fastapi import APIRouter, Query, BackgroundTasks, HTTPException

from app.models import (
    Article,
    AggregationResult,
    AggregationStatus,
    TriggerResponse,
    FeedSource,
    SummarizationResponse,
    SentimentResult,
    SentimentAnalysisResponse,
    RecommendationItem,
    RecommendationResponse
)
from app.services.aggregator import get_aggregator
from app.services.summarizer import get_summarization_service
from app.services.sentiment import get_sentiment_analyzer
from app.services.recommender import get_recommender
from app.services.supabase_service import supabase_service
from app.config import get_feed_sources, add_feed_source

router = APIRouter(prefix="/api/news", tags=["news"])


@router.get("/", response_model=AggregationResult)
async def get_news(
    force_refresh: bool = Query(
        False,
        description="Force refresh from sources, bypassing cache"
    ),
    source: Optional[str] = Query(
        None,
        description="Filter by source name"
    ),
    limit: int = Query(
        50,
        ge=1,
        le=200,
        description="Maximum number of articles to return"
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Number of articles to skip"
    )
) -> AggregationResult:
    """
    Get aggregated news articles.

    - Returns cached data if available (fast response)
    - Use `force_refresh=true` to fetch fresh data from all sources
    - Filter by `source` to get articles from a specific feed
    - Use `limit` and `offset` for pagination
    """
    aggregator = get_aggregator()

    # If filtering or paginating, use cached articles
    if source or offset > 0 or limit < 50:
        articles = aggregator.get_cached_articles(
            source=source,
            limit=limit,
            offset=offset
        )
        status = aggregator.get_status()
        return AggregationResult(
            success=True,
            articles=articles,
            total_count=len(articles),
            cached=True,
            sources_fetched=len(get_feed_sources())
        )

    # Otherwise, perform full aggregation
    return await aggregator.aggregate(force_refresh=force_refresh)


@router.post("/aggregate", response_model=AggregationResult)
async def trigger_aggregation(
    force_refresh: bool = Query(
        True,
        description="Force refresh from sources"
    )
) -> AggregationResult:
    """
    Trigger a news aggregation job.

    - This endpoint actively fetches from all RSS sources
    - Use this for cron jobs or manual refresh triggers
    - Returns the aggregation result with all articles
    """
    aggregator = get_aggregator()
    return await aggregator.aggregate(force_refresh=force_refresh)


@router.post("/aggregate/background", response_model=TriggerResponse)
async def trigger_aggregation_background(
    background_tasks: BackgroundTasks
) -> TriggerResponse:
    """
    Trigger aggregation in the background.

    - Returns immediately with a confirmation
    - Aggregation runs in the background
    - Use GET /api/news to fetch results once complete
    """
    aggregator = get_aggregator()

    # Check if already running
    status = aggregator.get_status()
    if status.is_running:
        return TriggerResponse(
            message="Aggregation already in progress",
            job_started=False,
            estimated_sources=len(get_feed_sources())
        )

    # Schedule background task
    async def run_aggregation():
        await aggregator.aggregate(force_refresh=True)

    background_tasks.add_task(run_aggregation)

    return TriggerResponse(
        message="Aggregation job started in background",
        job_started=True,
        estimated_sources=len(get_feed_sources())
    )


@router.get("/status", response_model=AggregationStatus)
async def get_status() -> AggregationStatus:
    """
    Get the current status of the aggregator.

    - Shows if aggregation is currently running
    - Shows last run time and cache status
    - Useful for monitoring and debugging
    """
    aggregator = get_aggregator()
    return aggregator.get_status()


@router.delete("/cache")
async def clear_cache() -> dict:
    """
    Clear the article cache.

    - Forces next request to fetch fresh data
    - Useful for debugging or resetting state
    """
    aggregator = get_aggregator()
    aggregator.clear_cache()
    return {"message": "Cache cleared successfully"}


@router.get("/sources", response_model=list[FeedSource])
async def get_sources() -> list[FeedSource]:
    """
    Get the list of configured feed sources.
    """
    return get_feed_sources()


@router.post("/sources", response_model=FeedSource)
async def add_source(source: FeedSource) -> FeedSource:
    """
    Add a new feed source dynamically.

    Note: For POC purposes only. In production, sources should be
    managed through proper configuration.
    """
    add_feed_source(source)
    return source


# =============================================================================
# CONTENT EXTRACTION ENDPOINTS
# =============================================================================

@router.get("/article", response_model=Article)
async def get_article_content(
    url: str = Query(..., description="Article URL to fetch content for")
) -> Article:
    """
    Get a specific article with full content extracted.

    - Extracts the full article content from the URL
    - Caches the result for future requests
    - Returns the article with `content` field populated
    """
    aggregator = get_aggregator()

    # Try to get and extract content
    article = await aggregator.extract_content_for_article(url)

    if not article:
        raise HTTPException(
            status_code=404,
            detail="Article not found in cache. Fetch news first with GET /api/news/"
        )

    return article


@router.post("/extract-content", response_model=dict)
async def extract_content_batch(
    background_tasks: BackgroundTasks,
    limit: int = Query(
        10,
        ge=1,
        le=50,
        description="Maximum number of articles to extract content for"
    )
) -> dict:
    """
    Extract content for multiple articles in the background.

    - Processes articles that don't have content yet
    - Runs in background, returns immediately
    - Use GET /api/news/status to monitor progress
    """
    aggregator = get_aggregator()

    async def run_extraction():
        await aggregator.extract_content_for_all(limit=limit)

    background_tasks.add_task(run_extraction)

    return {
        "message": f"Content extraction started for up to {limit} articles",
        "status": "processing"
    }


# =============================================================================
# SUMMARIZATION ENDPOINTS
# =============================================================================

@router.get("/summarize", response_model=SummarizationResponse)
async def summarize_article(
    url: str = Query(..., description="Article URL to summarize"),
    max_length: int = Query(
        200,
        ge=50,
        le=1000,
        description="Target summary length in words"
    )
) -> SummarizationResponse:
    """
    Summarize an article's content.

    - First extracts content if not already done
    - Then summarizes using configured summarizer
    - Default: extractive summarization (no AI)
    - Can be upgraded to AI summarization by configuring provider

    To enable AI summarization, configure in app startup:
    ```python
    from app.services.summarizer import configure_ai_summarizer
    configure_ai_summarizer("openai", "your-api-key")
    ```
    """
    aggregator = get_aggregator()
    summarizer = get_summarization_service()

    # Get article (and extract content if needed)
    article = await aggregator.extract_content_for_article(url)

    if not article:
        raise HTTPException(
            status_code=404,
            detail="Article not found. Fetch news first with GET /api/news/"
        )

    # Summarize
    return await summarizer.summarize_article(article, max_length=max_length)


# =============================================================================
# SENTIMENT ANALYSIS ENDPOINTS
# =============================================================================

@router.get("/sentiment", response_model=SentimentResult)
async def analyze_article_sentiment(
    url: str = Query(..., description="Article URL to analyze")
) -> SentimentResult:
    """
    Analyze sentiment of a single article.

    Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) which is
    specifically designed for social media and news text.

    Returns:
    - negative/neutral/positive: Proportion scores (0-1)
    - compound: Normalized score (-1 to 1)
    - label: Overall sentiment classification
    """
    aggregator = get_aggregator()
    analyzer = get_sentiment_analyzer()

    # Get article
    article = aggregator.get_article_by_url(url)
    if not article:
        raise HTTPException(
            status_code=404,
            detail="Article not found. Fetch news first with GET /api/news/"
        )

    # Use content if available, otherwise description
    text = article.content or article.description or article.title
    scores = analyzer.analyze(text)

    return SentimentResult(
        article_title=article.title,
        article_url=article.link,
        negative=scores.negative,
        neutral=scores.neutral,
        positive=scores.positive,
        compound=scores.compound,
        label=scores.label
    )


@router.get("/sentiment/batch", response_model=SentimentAnalysisResponse)
async def analyze_batch_sentiment(
    source: Optional[str] = Query(None, description="Filter by source"),
    limit: int = Query(10, ge=1, le=50, description="Number of articles to analyze")
) -> SentimentAnalysisResponse:
    """
    Analyze sentiment for multiple articles.

    Optionally filter by source to analyze sentiment of a specific news outlet.
    Returns individual scores plus overall average.
    """
    aggregator = get_aggregator()
    analyzer = get_sentiment_analyzer()

    # Get articles
    articles = aggregator.get_cached_articles(source=source, limit=limit)

    if not articles:
        return SentimentAnalysisResponse(
            success=False,
            results=[],
            average_compound=0.0,
            overall_label="neutral",
            articles_analyzed=0
        )

    # Analyze each article
    results = []
    total_compound = 0.0

    for article in articles:
        text = article.content or article.description or article.title
        scores = analyzer.analyze(text)
        total_compound += scores.compound

        results.append(SentimentResult(
            article_title=article.title,
            article_url=article.link,
            negative=scores.negative,
            neutral=scores.neutral,
            positive=scores.positive,
            compound=scores.compound,
            label=scores.label
        ))

    # Calculate average
    avg_compound = total_compound / len(results)
    if avg_compound >= 0.05:
        overall_label = "positive"
    elif avg_compound <= -0.05:
        overall_label = "negative"
    else:
        overall_label = "neutral"

    return SentimentAnalysisResponse(
        success=True,
        results=results,
        average_compound=round(avg_compound, 4),
        overall_label=overall_label,
        articles_analyzed=len(results)
    )


@router.post("/sentiment/text")
async def analyze_text_sentiment(
    text: str = Query(..., description="Text to analyze")
) -> dict:
    """
    Analyze sentiment of arbitrary text.

    Useful for analyzing user comments, custom content, etc.
    """
    analyzer = get_sentiment_analyzer()
    scores = analyzer.analyze(text)

    return {
        "text_length": len(text),
        "negative": scores.negative,
        "neutral": scores.neutral,
        "positive": scores.positive,
        "compound": scores.compound,
        "label": scores.label
    }


# =============================================================================
# RECOMMENDATION ENDPOINTS
# =============================================================================

@router.post("/recommendations/build")
async def build_recommendation_index() -> dict:
    """
    Build/rebuild the recommendation index from cached articles.

    Call this after fetching news to enable recommendations.
    The index is built using TF-IDF vectorization.
    """
    aggregator = get_aggregator()
    recommender = get_recommender()

    # Get all cached articles
    articles = aggregator.get_cached_articles(limit=200)

    if not articles:
        return {
            "success": False,
            "message": "No articles in cache. Fetch news first with GET /api/news/",
            "articles_indexed": 0
        }

    # Build the recommendation index
    recommender.fit(articles)

    return {
        "success": True,
        "message": "Recommendation index built successfully",
        "articles_indexed": recommender.article_count
    }


@router.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    url: str = Query(..., description="Article URL to get recommendations for"),
    limit: int = Query(5, ge=1, le=20, description="Number of recommendations")
) -> RecommendationResponse:
    """
    Get article recommendations based on a given article.

    Uses TF-IDF and cosine similarity to find articles with similar content.
    Make sure to call POST /recommendations/build first to build the index.
    """
    recommender = get_recommender()

    if not recommender.is_ready:
        raise HTTPException(
            status_code=400,
            detail="Recommendation index not built. Call POST /api/news/recommendations/build first."
        )

    recommendations = recommender.get_recommendations(url, n_recommendations=limit)

    if not recommendations:
        # Article might not be in index
        return RecommendationResponse(
            success=False,
            query_article=url,
            recommendations=[],
            algorithm="tfidf_cosine"
        )

    return RecommendationResponse(
        success=True,
        query_article=url,
        recommendations=[
            RecommendationItem(
                title=r.article.title,
                link=r.article.link,
                source=r.article.source,
                similarity_score=r.similarity_score,
                published_at=r.article.published_at
            )
            for r in recommendations
        ],
        algorithm="tfidf_cosine"
    )


@router.get("/recommendations/search", response_model=RecommendationResponse)
async def search_recommendations(
    query: str = Query(..., description="Search query text"),
    limit: int = Query(5, ge=1, le=20, description="Number of results")
) -> RecommendationResponse:
    """
    Find articles similar to a search query.

    Useful for implementing search functionality.
    Uses TF-IDF to match query against article content.
    """
    recommender = get_recommender()

    if not recommender.is_ready:
        raise HTTPException(
            status_code=400,
            detail="Recommendation index not built. Call POST /api/news/recommendations/build first."
        )

    recommendations = recommender.get_recommendations_for_text(query, n_recommendations=limit)

    return RecommendationResponse(
        success=True,
        query_article=f"search: {query}",
        recommendations=[
            RecommendationItem(
                title=r.article.title,
                link=r.article.link,
                source=r.article.source,
                similarity_score=r.similarity_score,
                published_at=r.article.published_at
            )
            for r in recommendations
        ],
        algorithm="tfidf_cosine"
    )


# =============================================================================
# SUPABASE SYNC ENDPOINTS
# =============================================================================

@router.get("/supabase/status")
async def supabase_status() -> dict:
    """
    Check if Supabase is configured and ready.
    """
    return {
        "configured": supabase_service.is_configured(),
        "tables": {
            "articles": "articles (url, title, summary, image_url, overall_sentiment, source_id)",
            "sources": "sources (name, domain, logo_url, reliability_score)"
        }
    }


@router.post("/supabase/sync/articles")
async def sync_articles_to_supabase(
    force_refresh: bool = Query(
        False,
        description="Force fetch fresh articles before syncing"
    )
) -> dict:
    """
    Sync cached articles to Supabase database.

    - Fetches articles if cache is empty or force_refresh=true
    - Upserts articles to Supabase (insert or update)
    - Uses 'link' as unique identifier for deduplication

    Call this endpoint from pg_cron to keep your database updated:
    ```sql
    select cron.schedule('sync-news', '*/15 * * * *',
      $$select net.http_post('https://your-api/api/news/supabase/sync/articles')$$
    );
    ```
    """
    if not supabase_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Supabase not configured. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY"
        )

    aggregator = get_aggregator()

    # Get articles (fetch fresh if needed)
    if force_refresh:
        result = await aggregator.aggregate(force_refresh=True)
        articles = result.articles
    else:
        articles = aggregator.get_cached_articles(limit=200)

        # If no cached articles, fetch fresh
        if not articles:
            result = await aggregator.aggregate(force_refresh=True)
            articles = result.articles

    if not articles:
        return {
            "success": False,
            "message": "No articles to sync",
            "synced": 0
        }

    # Sync to Supabase
    sync_result = await supabase_service.upsert_articles(articles)

    return {
        "success": len(sync_result.get("errors", [])) == 0,
        "synced": sync_result.get("inserted", 0),
        "total_articles": len(articles),
        "message": sync_result.get("message", ""),
        "errors": sync_result.get("errors", [])
    }


@router.post("/supabase/sync/sentiments")
async def sync_sentiments_to_supabase(
    source: Optional[str] = Query(None, description="Filter by source"),
    limit: int = Query(50, ge=1, le=200, description="Number of articles to analyze")
) -> dict:
    """
    Analyze sentiment for cached articles and update overall_sentiment in Supabase.

    - Analyzes sentiment for articles in cache
    - Updates the overall_sentiment column on the articles table
    - Uses compound score (-1 to 1) as the sentiment value
    """
    if not supabase_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Supabase not configured. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY"
        )

    aggregator = get_aggregator()
    analyzer = get_sentiment_analyzer()

    # Get articles
    articles = aggregator.get_cached_articles(source=source, limit=limit)

    if not articles:
        return {
            "success": False,
            "message": "No articles in cache. Sync articles first.",
            "updated": 0
        }

    # Analyze sentiments
    sentiments = []
    for article in articles:
        text = article.content or article.description or article.title
        scores = analyzer.analyze(text)
        sentiments.append(SentimentResult(
            article_title=article.title,
            article_url=article.link,
            negative=scores.negative,
            neutral=scores.neutral,
            positive=scores.positive,
            compound=scores.compound,
            label=scores.label
        ))

    # Update sentiments on articles table
    sync_result = await supabase_service.upsert_sentiments(sentiments)

    return {
        "success": len(sync_result.get("errors", [])) == 0,
        "updated": sync_result.get("updated", 0),
        "total_analyzed": len(sentiments),
        "message": sync_result.get("message", ""),
        "errors": sync_result.get("errors", [])
    }


@router.post("/supabase/sync/all")
async def sync_all_to_supabase(
    background_tasks: BackgroundTasks,
    force_refresh: bool = Query(False, description="Force fetch fresh articles")
) -> dict:
    """
    Sync articles AND sentiments to Supabase in the background.

    This is the recommended endpoint for pg_cron jobs:
    ```sql
    select cron.schedule('full-news-sync', '0 */6 * * *',
      $$select net.http_post('https://your-api/api/news/supabase/sync/all')$$
    );
    ```
    """
    if not supabase_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Supabase not configured. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY"
        )

    async def full_sync():
        aggregator = get_aggregator()
        analyzer = get_sentiment_analyzer()

        # 1. Fetch fresh articles
        result = await aggregator.aggregate(force_refresh=force_refresh)
        articles = result.articles

        if not articles:
            return

        # 2. Extract full content for articles (limit to avoid timeout)
        await aggregator.extract_content_for_all(limit=50)
        # Refresh articles list with extracted content
        articles = aggregator.get_cached_articles(limit=200)

        # 3. Sync articles to Supabase (creates sources as needed)
        await supabase_service.upsert_articles(articles)

        # 4. Analyze sentiments and update articles
        sentiments = []
        for article in articles:
            text = article.content or article.description or article.title
            scores = analyzer.analyze(text)
            sentiments.append(SentimentResult(
                article_title=article.title,
                article_url=article.link,
                negative=scores.negative,
                neutral=scores.neutral,
                positive=scores.positive,
                compound=scores.compound,
                label=scores.label
            ))

        # 5. Update overall_sentiment on articles
        await supabase_service.upsert_sentiments(sentiments)

    background_tasks.add_task(full_sync)

    return {
        "message": "Full sync started in background",
        "status": "processing"
    }


@router.get("/supabase/articles")
async def get_articles_from_supabase(
    limit: int = Query(50, ge=1, le=200, description="Number of articles to fetch"),
    source: Optional[str] = Query(None, description="Filter by source")
) -> dict:
    """
    Fetch articles directly from Supabase.

    Useful for verifying sync worked correctly.
    """
    if not supabase_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Supabase not configured"
        )

    articles = await supabase_service.get_articles(limit=limit, source=source)

    return {
        "success": True,
        "count": len(articles),
        "articles": articles
    }


@router.delete("/supabase/articles/old")
async def cleanup_old_articles(
    days: int = Query(30, ge=1, le=365, description="Delete articles older than N days")
) -> dict:
    """
    Delete old articles from Supabase to manage storage.
    """
    if not supabase_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Supabase not configured"
        )

    result = await supabase_service.delete_old_articles(days=days)

    return result
