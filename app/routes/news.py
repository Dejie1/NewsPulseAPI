"""
FastAPI routes for the news aggregator.
Provides endpoints for triggering aggregation and fetching articles.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Query, BackgroundTasks, HTTPException

logger = logging.getLogger(__name__)

from app.models import (
    Article,
    AggregationResult,
    AggregationStatus,
    TriggerResponse,
    FeedSource,
    SummarizationResponse,
    SentimentResult,
    SentimentAnalysisResponse,
    CompanyMentionResult,
    ArticleCompanyAnalysis,
    CompanyAnalysisResponse,
)
from app.services.aggregator import get_aggregator
from app.services.summarizer import get_summarization_service
from app.services.sentiment import get_sentiment_analyzer
from app.services.supabase_service import supabase_service
from app.config import get_feed_sources, get_source_names, add_feed_source

from enum import Enum

# Build enum dynamically from configured feed sources for Swagger dropdowns
SourceName = Enum("SourceName", {name: name for name in get_source_names()}, type=str)

router = APIRouter(prefix="/api/news")


@router.get("/", response_model=AggregationResult, tags=["News Feed"])
async def get_news(
    force_refresh: bool = Query(
        False,
        description="Force refresh from sources, bypassing cache"
    ),
    source: Optional[SourceName] = Query(
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

    > **Start here.** This is the entry point — call this first to populate the cache.
    > Other endpoints (sentiment, summarize, company analysis) require article URLs from this response.

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


@router.post("/aggregate", response_model=AggregationResult, tags=["News Feed"])
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


@router.post("/aggregate/background", response_model=TriggerResponse, tags=["News Feed"])
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
        try:
            await aggregator.aggregate(force_refresh=True)
        except Exception as e:
            logger.error("Background aggregation failed: %s", e, exc_info=True)

    background_tasks.add_task(run_aggregation)

    return TriggerResponse(
        message="Aggregation job started in background",
        job_started=True,
        estimated_sources=len(get_feed_sources())
    )


@router.get("/status", response_model=AggregationStatus, tags=["News Feed"])
async def get_status() -> AggregationStatus:
    """
    Get the current status of the aggregator.

    - Shows if aggregation is currently running
    - Shows last run time and cache status
    - Useful for monitoring and debugging
    """
    aggregator = get_aggregator()
    return aggregator.get_status()


@router.delete("/cache", tags=["News Feed"])
async def clear_cache() -> dict:
    """
    Clear the article cache.

    - Forces next request to fetch fresh data
    - Useful for debugging or resetting state
    """
    aggregator = get_aggregator()
    aggregator.clear_cache()
    return {"message": "Cache cleared successfully"}


@router.get("/sources", response_model=list[FeedSource], tags=["News Feed"])
async def get_sources() -> list[FeedSource]:
    """
    Get the list of configured feed sources.
    """
    return get_feed_sources()


@router.post("/sources", response_model=FeedSource, tags=["News Feed"])
async def add_source(source: FeedSource) -> FeedSource:
    """
    Add a new feed source dynamically.

    Note: For POC purposes only. In production, sources should be
    managed through proper configuration.
    """
    add_feed_source(source)
    return source



@router.get("/article", response_model=Article, tags=["Content & Summarization"])
async def get_article_content(
    url: str = Query(..., description="Article URL from GET /api/news/ response")
) -> Article:
    """
    Get a specific article with full content extracted.

    > **Prerequisite:** Call `GET /api/news/` first to populate the cache, then use a `link` value from the response.

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


@router.post("/extract-content", response_model=dict, tags=["Content & Summarization"])
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
        try:
            await aggregator.extract_content_for_all(limit=limit)
        except Exception as e:
            logger.error("Background content extraction failed: %s", e, exc_info=True)

    background_tasks.add_task(run_extraction)

    return {
        "message": f"Content extraction started for up to {limit} articles",
        "status": "processing"
    }



@router.get("/summarize", response_model=SummarizationResponse, tags=["Content & Summarization"])
async def summarize_article(
    url: str = Query(..., description="Article URL from GET /api/news/ response"),
    max_length: int = Query(
        200,
        ge=50,
        le=1000,
        description="Target summary length in words"
    )
) -> SummarizationResponse:
    """
    Summarize an article's content.

    > **Prerequisite:** Call `GET /api/news/` first to populate the cache, then use a `link` value from the response.

    - First extracts content if not already done
    - Then summarizes using the LSA extractive algorithm
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



@router.get("/sentiment", response_model=SentimentResult, tags=["Sentiment Analysis"])
async def analyze_article_sentiment(
    url: str = Query(..., description="Article URL from GET /api/news/ response")
) -> SentimentResult:
    """
    Analyze sentiment of a single article.

    > **Prerequisite:** Call `GET /api/news/` first to populate the cache, then use a `link` value from the response.

    Uses RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest) for sentiment analysis.

    Returns:
    - **negative/neutral/positive**: Probability scores (0-1)
    - **compound**: Synthetic score (-1 to 1) calculated as `positive - negative`
    - **label**: Overall sentiment classification
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


@router.get("/sentiment/batch", response_model=SentimentAnalysisResponse, tags=["Sentiment Analysis"])
async def analyze_batch_sentiment(
    source: Optional[SourceName] = Query(None, description="Filter by source"),
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


@router.post("/sentiment/text", tags=["Sentiment Analysis"])
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



@router.get("/supabase/status", tags=["Supabase Sync"])
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


@router.post("/supabase/sync/articles", tags=["Supabase Sync"])
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


@router.post("/supabase/sync/sentiments", tags=["Supabase Sync"])
async def sync_sentiments_to_supabase(
    source: Optional[SourceName] = Query(None, description="Filter by source"),
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


@router.get("/supabase/articles", tags=["Supabase Sync"])
async def get_articles_from_supabase(
    limit: int = Query(50, ge=1, le=200, description="Number of articles to fetch"),
    source: Optional[SourceName] = Query(None, description="Filter by source")
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


@router.delete("/supabase/articles/old", tags=["Supabase Sync"])
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



@router.get("/companies", tags=["Company Analysis"])
async def get_tracked_companies() -> dict:
    """
    Get list of companies being tracked (Magnificent 7).

    Returns company info from the database if Supabase is configured,
    otherwise returns the default list.
    """
    if supabase_service.is_configured():
        companies = await supabase_service.get_all_companies()
        if companies:
            return {
                "success": True,
                "companies": companies,
                "source": "database"
            }

    # Fallback to NER service tracked companies
    from app.services.ner_service import get_ner_service
    ner_service = get_ner_service()

    return {
        "success": True,
        "companies": ner_service.get_tracked_companies(),
        "source": "config"
    }


@router.get("/companies/mentions/{ticker}", tags=["Company Analysis"])
async def get_company_mentions(
    ticker: str,
    limit: int = Query(20, ge=1, le=100, description="Number of mentions to return"),
    days: int = Query(7, ge=1, le=30, description="Look back period in days")
) -> dict:
    """
    Get recent mentions for a specific company by ticker.

    Includes sentiment scores and context sentences.
    """
    if not supabase_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Supabase not configured"
        )

    mentions = await supabase_service.get_company_mentions(
        ticker=ticker.upper(),
        limit=limit,
        days=days
    )

    return {
        "success": True,
        "ticker": ticker.upper(),
        "mentions": mentions,
        "count": len(mentions)
    }


@router.post("/analyze/companies", response_model=CompanyAnalysisResponse, tags=["Company Analysis"])
async def analyze_article_companies(
    url: str = Query(..., description="Article URL from GET /api/news/ response")
) -> CompanyAnalysisResponse:
    """
    Analyze company mentions in a single article.

    > **Prerequisite:** Call `GET /api/news/` first to populate the cache, then use a `link` value from the response.

    - Extracts content if needed
    - Runs **GLiNER-spaCy** NER to detect Magnificent 7 company mentions
    - Analyzes financial sentiment with **FinBERT** for each mention
    - Returns structured results with context and sentiment

    **Note:** First call may be slow (~30-60s) as ML models are loaded lazily.
    """
    aggregator = get_aggregator()

    # Get article with content
    article = await aggregator.extract_content_for_article(url)
    if not article:
        raise HTTPException(
            status_code=404,
            detail="Article not found. Fetch news first with GET /api/news/"
        )

    if not article.content:
        raise HTTPException(
            status_code=400,
            detail="Article has no content. Content extraction may have failed."
        )

    # Analyze with NER + FinBERT
    from app.services.company_analysis_service import get_company_analyzer
    analyzer = get_company_analyzer()

    mentions = await analyzer.analyze_text(article.content)

    return CompanyAnalysisResponse(
        success=True,
        articles_analyzed=1,
        total_mentions=len(mentions),
        results=[ArticleCompanyAnalysis(
            article_url=article.link,
            article_title=article.title,
            mentions=[
                CompanyMentionResult(
                    company_name=m.company_name,
                    ticker=m.ticker,
                    sentiment_score=m.sentiment_score,
                    confidence_score=m.confidence_score,
                    context_sentence=m.context_sentence,
                    sentiment_label=m.sentiment_label
                )
                for m in mentions
            ],
            companies_found=len(set(m.ticker for m in mentions))
        )]
    )


@router.post("/analyze/companies/batch", tags=["Company Analysis"])
async def analyze_companies_batch(
    background_tasks: BackgroundTasks,
    limit: int = Query(20, ge=1, le=50, description="Number of articles to analyze")
) -> dict:
    """
    Analyze company mentions for multiple articles in background.

    Processes articles that have content extracted.
    Use GET /api/news/status to monitor progress.
    """
    aggregator = get_aggregator()

    # Check if there are articles with content
    articles = aggregator.get_cached_articles(limit=limit)
    articles_with_content = [a for a in articles if a.content]

    if not articles_with_content:
        return {
            "message": "No articles with content found. Run content extraction first.",
            "status": "skipped",
            "articles_found": 0
        }

    async def run_analysis():
        try:
            from app.services.company_analysis_service import get_company_analyzer
            analyzer = get_company_analyzer()

            for article in articles_with_content:
                try:
                    await analyzer.analyze_text(article.content)
                except Exception as e:
                    logger.error("Error analyzing %s: %s", article.link, e)
        except Exception as e:
            logger.error("Background company analysis failed: %s", e, exc_info=True)

    background_tasks.add_task(run_analysis)

    return {
        "message": f"Company analysis started for {len(articles_with_content)} articles",
        "status": "processing",
        "articles_found": len(articles_with_content)
    }


@router.post("/supabase/seed/companies", tags=["Company Analysis"])
async def seed_companies() -> dict:
    """
    Seed the companies table with Magnificent 7 companies.

    This should be run once to populate the companies table.
    """
    if not supabase_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Supabase not configured"
        )

    result = await supabase_service.seed_companies()
    return result


@router.post("/supabase/sync/companies", tags=["Supabase Sync"])
async def sync_company_mentions(
    background_tasks: BackgroundTasks,
    limit: int = Query(50, ge=1, le=200, description="Number of articles to analyze"),
    force_reanalyze: bool = Query(False, description="Re-analyze articles already processed")
) -> dict:
    """
    Analyze articles for company mentions and sync to Supabase.

    - Fetches articles with content from Supabase
    - Runs NER + FinBERT analysis on each
    - Upserts results to company_mentions table

    This is the main endpoint for syncing company analysis.
    """
    if not supabase_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Supabase not configured"
        )

    async def full_company_sync():
        try:
            from app.services.company_analysis_service import get_company_analyzer
            analyzer = get_company_analyzer()

            # Get articles to analyze
            articles = await supabase_service.get_articles_for_analysis(
                limit=limit,
                only_unanalyzed=not force_reanalyze
            )

            if not articles:
                logger.info("No articles to analyze for company mentions")
                return

            logger.info("Analyzing %d articles for company mentions", len(articles))

            all_mentions = []
            for article in articles:
                content = article.get("full_content")
                if not content:
                    continue

                try:
                    mentions = await analyzer.analyze_text(content)

                    for mention in mentions:
                        company_id = await supabase_service.get_company_by_ticker(
                            mention.ticker
                        )
                        if company_id:
                            all_mentions.append({
                                "article_id": article["id"],
                                "company_id": company_id,
                                "sentiment_score": mention.sentiment_score,
                                "confidence_score": mention.confidence_score,
                                "context_sentence": mention.context_sentence[:500]
                            })

                except Exception as e:
                    logger.error("Error analyzing article %s: %s", article.get("id"), e)

            # Upsert to Supabase
            if all_mentions:
                result = await supabase_service.upsert_company_mentions(all_mentions)
                logger.info("Synced %d company mentions", result.get("inserted", 0))

                # Update daily metrics for companies with mentions today
                from datetime import datetime, timezone
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

                company_ids_with_mentions = set(m["company_id"] for m in all_mentions)
                for company_id in company_ids_with_mentions:
                    try:
                        await supabase_service.update_daily_metrics(today, company_id)
                    except Exception as e:
                        logger.error("Error updating daily metrics for company %s: %s", company_id, e)

                logger.info("Updated daily metrics for %d companies", len(company_ids_with_mentions))
            else:
                logger.info("No company mentions found to sync")
        except Exception as e:
            logger.error("Background company sync failed: %s", e, exc_info=True)

    background_tasks.add_task(full_company_sync)

    return {
        "message": f"Company analysis sync started for up to {limit} articles",
        "status": "processing",
        "force_reanalyze": force_reanalyze
    }


@router.get("/companies/metrics/{ticker}", tags=["Company Analysis"])
async def get_company_metrics(
    ticker: str,
    days: int = Query(7, ge=1, le=90, description="Number of days to fetch")
) -> dict:
    """
    Get daily sentiment metrics for a company.

    Returns time series data useful for charting sentiment trends.
    """
    if not supabase_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Supabase not configured"
        )

    metrics = await supabase_service.get_company_daily_metrics(
        ticker=ticker.upper(),
        days=days
    )

    return {
        "success": True,
        "ticker": ticker.upper(),
        "days": days,
        "metrics": metrics,
        "count": len(metrics)
    }


@router.post("/supabase/sync/all", tags=["Supabase Sync"])
async def sync_all_to_supabase(
    background_tasks: BackgroundTasks,
    force_refresh: bool = Query(False, description="Force fetch fresh articles"),
    analyze_companies: bool = Query(True, description="Include company NER analysis")
) -> dict:
    """
    Sync articles, sentiments, AND company mentions to Supabase in the background.

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
        try:
            aggregator = get_aggregator()
            analyzer = get_sentiment_analyzer()

            # 1. Fetch fresh articles from RSS
            result = await aggregator.aggregate(force_refresh=force_refresh)
            articles = result.articles

            if not articles:
                logger.info("[sync] No articles fetched from RSS")
                return

            logger.info("[sync] Fetched %d articles from RSS", len(articles))

            # 2. Get URLs that already have content in Supabase (skip re-extraction)
            urls_with_content = await supabase_service.get_urls_with_content()
            logger.info("[sync] Found %d articles with existing content in Supabase", len(urls_with_content))

            # 3. Extract content ONLY for articles missing content
            from app.services.content_extractor import get_content_extractor
            content_extractor = get_content_extractor()

            articles_needing_content = [
                a for a in articles
                if a.link not in urls_with_content and not a.content_extracted
            ][:60]  # Limit to avoid timeout

            logger.info("[sync] Extracting content for %d new articles", len(articles_needing_content))

            if articles_needing_content:
                updated_articles = await content_extractor.extract_for_articles(
                    articles_needing_content,
                    max_concurrent=3
                )
                # Update cache with extracted content
                for updated in updated_articles:
                    aggregator._update_article_in_cache(updated)

            # Refresh articles list with extracted content
            articles = aggregator.get_cached_articles(limit=200)

            # 4. Sync articles to Supabase (creates sources as needed)
            await supabase_service.upsert_articles(articles)
            logger.info("[sync] Synced %d articles to Supabase", len(articles))

            # 5. Analyze sentiments and update articles
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

            # 6. Update overall_sentiment on articles
            await supabase_service.upsert_sentiments(sentiments)

            # 7. Analyze companies and sync mentions
            if analyze_companies:
                from app.services.company_analysis_service import get_company_analyzer
                company_analyzer = get_company_analyzer()

                all_mentions = []
                for article in articles:
                    if not article.content:
                        continue

                    article_id = await supabase_service.get_article_id_by_url(article.link)
                    if not article_id:
                        continue

                    try:
                        mentions = await company_analyzer.analyze_text(article.content)

                        for mention in mentions:
                            company_id = await supabase_service.get_company_by_ticker(
                                mention.ticker
                            )
                            if company_id:
                                all_mentions.append({
                                    "article_id": article_id,
                                    "company_id": company_id,
                                    "sentiment_score": mention.sentiment_score,
                                    "confidence_score": mention.confidence_score,
                                    "context_sentence": mention.context_sentence[:500]
                                })
                    except Exception as e:
                        logger.error("Error analyzing companies for %s: %s", article.link, e)

                if all_mentions:
                    await supabase_service.upsert_company_mentions(all_mentions)

                    # 8. Update daily metrics for companies with mentions today
                    from datetime import datetime, timezone
                    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

                    # Get unique company_ids from mentions
                    company_ids_with_mentions = set(m["company_id"] for m in all_mentions)

                    for company_id in company_ids_with_mentions:
                        try:
                            await supabase_service.update_daily_metrics(today, company_id)
                        except Exception as e:
                            logger.error("Error updating daily metrics for company %s: %s", company_id, e)

                    logger.info("Updated daily metrics for %d companies", len(company_ids_with_mentions))
        except Exception as e:
            logger.error("Background full sync failed: %s", e, exc_info=True)

    background_tasks.add_task(full_sync)

    return {
        "message": "Full sync started in background",
        "status": "processing",
        "analyze_companies": analyze_companies
    }
