"""
FastAPI application entry point.
News Aggregator API for React Native app.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes.news import router as news_router

# Configure structured logging for all app modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    Handles startup and graceful shutdown of services.
    """
    logger.info("Starting %s...", settings.app_name)
    yield
    # Shutdown: clean up thread pools and ML models
    logger.info("Shutting down services...")

    from app.services.content_extractor import get_content_extractor
    from app.services.summarizer import get_summarization_service
    from app.services.sentiment import get_sentiment_analyzer
    from app.services.finbert_service import get_finbert_service
    from app.services.ner_service import get_ner_service

    get_content_extractor().shutdown()
    get_sentiment_analyzer().unload()

    finbert = get_finbert_service()
    finbert.shutdown()

    ner = get_ner_service()
    ner.shutdown()

    summarizer = get_summarization_service().summarizer
    if hasattr(summarizer, "shutdown"):
        summarizer.shutdown()

    logger.info("Shutdown complete.")


app = FastAPI(
    title=settings.app_name,
    description="""
    News Aggregator API - POC for React Native app.

    ## Features
    - Aggregate news from multiple RSS sources
    - In-memory caching with configurable TTL
    - Rate limiting to avoid overwhelming sources
    - Deduplication based on URL and title
    - Background aggregation support

    ## Usage
    - `GET /api/news/` - Get aggregated articles (uses cache)
    - `POST /api/news/aggregate` - Trigger fresh aggregation
    - `GET /api/news/status` - Check aggregator status

    ## For Cron Jobs
    Call `POST /api/news/aggregate` periodically to keep cache fresh.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for React Native app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(news_router)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "get_news": "GET /api/news/",
            "trigger_aggregation": "POST /api/news/aggregate",
            "status": "GET /api/news/status",
            "sources": "GET /api/news/sources"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
