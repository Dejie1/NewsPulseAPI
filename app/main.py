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


openapi_tags = [
    {
        "name": "News Feed",
        "description": "Core news aggregation pipeline. Fetches articles from multiple RSS sources with caching, deduplication, and pagination.",
    },
    {
        "name": "Content & Summarization",
        "description": "Full article content extraction via **trafilatura** and extractive summarization using the **LSA** algorithm.",
    },
    {
        "name": "Sentiment Analysis",
        "description": "General sentiment analysis powered by **RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`). Returns negative/neutral/positive probabilities and a compound score.",
    },
    {
        "name": "Company Analysis",
        "description": "Magnificent 7 stock tracking. Uses **GLiNER-spaCy** for Named Entity Recognition and **FinBERT** for financial sentiment analysis on detected company mentions.",
    },
    {
        "name": "Supabase Sync",
        "description": "Database synchronization layer. Syncs articles, sentiment scores, and company mentions to **Supabase** for persistence and mobile app consumption.",
    },
]

app = FastAPI(
    title="NewsPulse Aggregator API",
    description=(
        "Backend API powering the **NewsPulse** mobile app — a real-time news aggregation "
        "and analysis platform built as a Final Year Project.\n\n"
        "## Architecture\n\n"
        "```\n"
        "RSS Feeds → Aggregator → Content Extraction → NLP Analysis → Supabase → Mobile App\n"
        "```\n\n"
        "## Key Capabilities\n\n"
        "| Feature | Technology |\n"
        "|---|---|\n"
        "| News Aggregation | feedparser + rate-limited RSS fetching |\n"
        "| Content Extraction | trafilatura + curl-cffi |\n"
        "| Sentiment Analysis | RoBERTa (cardiffnlp) |\n"
        "| Financial Sentiment | FinBERT (ProsusAI) |\n"
        "| Entity Recognition | GLiNER-spaCy NER |\n"
        "| Recommendations | Personalized scoring via Supabase RPC |\n"
        "| Database | Supabase (PostgreSQL) |\n"
        "| Mobile App | React Native + Expo |\n"
    ),
    version="1.0.0",
    openapi_tags=openapi_tags,
    lifespan=lifespan,
)

# API is not exposed to the public and is only called by the dev machine and cron job. Restrict`allow_origins` to known callers and drop `allow_credentials` accordingly in prod.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(news_router)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["News Feed"])
async def health_check():
    """Health check endpoint for monitoring and uptime verification."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
