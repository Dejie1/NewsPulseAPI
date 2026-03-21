"""
Pydantic models for the news aggregator.
These define the data structures used throughout the service.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


class Article(BaseModel):
    """Unified article structure returned by the aggregator."""
    title: str
    link: str
    source: str  # e.g., "NYTimes", "Wired Business"
    published_at: datetime
    description: Optional[str] = None
    image: Optional[str] = None
    category: Optional[str] = None  # Category from feed source
    content: Optional[str] = None  # Full article content (crawled)
    content_extracted: bool = False  # Whether content has been extracted

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FeedSource(BaseModel):
    """Configuration for an RSS feed source."""
    name: str
    url: str
    category: Optional[str] = None
    full_content_in_feed: bool = False  # If True, RSS description contains full article content


class AggregationResult(BaseModel):
    """Response model for aggregation endpoints."""
    success: bool
    articles: list[Article] = Field(default_factory=list)
    total_count: int = 0
    sources_fetched: int = 0
    sources_failed: int = 0
    cached: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    errors: list[str] = Field(default_factory=list)


class AggregationStatus(BaseModel):
    """Status response for the aggregation job."""
    is_running: bool = False
    last_run: Optional[datetime] = None
    last_success: Optional[datetime] = None
    articles_in_cache: int = 0
    cache_age_seconds: Optional[float] = None


class TriggerResponse(BaseModel):
    """Response when triggering an aggregation job."""
    message: str
    job_started: bool
    estimated_sources: int = 0


class SummarizationRequest(BaseModel):
    """Request model for article summarization."""
    article_url: str
    max_length: int = Field(default=200, ge=50, le=1000)


class SummarizationResponse(BaseModel):
    """Response model for article summarization."""
    original_title: str
    summary: str
    source: str
    summarized_by: str = "none"  # "none", "ai", "extractive"
    original_length: int = 0
    summary_length: int = 0


# =============================================================================
# SENTIMENT ANALYSIS MODELS
# =============================================================================

class SentimentResult(BaseModel):
    """Sentiment analysis result for a single article."""
    article_title: str
    article_url: str
    negative: float = Field(ge=0, le=1, description="Proportion of negative sentiment")
    neutral: float = Field(ge=0, le=1, description="Proportion of neutral sentiment")
    positive: float = Field(ge=0, le=1, description="Proportion of positive sentiment")
    compound: float = Field(ge=-1, le=1, description="Normalized compound score")
    label: str = Field(description="Overall sentiment: positive, negative, or neutral")


class SentimentAnalysisResponse(BaseModel):
    """Response for sentiment analysis endpoints."""
    success: bool
    results: list[SentimentResult] = Field(default_factory=list)
    average_compound: float = 0.0
    overall_label: str = "neutral"
    articles_analyzed: int = 0


# =============================================================================
# RECOMMENDATION MODELS
# =============================================================================

class RecommendationItem(BaseModel):
    """A single recommendation."""
    title: str
    link: str
    source: str
    similarity_score: float = Field(ge=0, le=1, description="How similar to the query article")
    published_at: datetime


class RecommendationResponse(BaseModel):
    """Response for recommendation endpoints."""
    success: bool
    query_article: Optional[str] = None
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    algorithm: str = "tfidf_cosine"


# =============================================================================
# COMPANY ANALYSIS MODELS (NER + FinBERT)
# =============================================================================

class CompanyEntityResult(BaseModel):
    """A detected company entity from NER."""
    company_name: str = Field(description="Normalized company name (e.g., 'Google')")
    ticker: str = Field(description="Stock ticker symbol (e.g., 'GOOGL')")
    matched_text: str = Field(description="Original text that matched")
    confidence: float = Field(ge=0, le=1, description="NER confidence score")
    context_sentence: str = Field(description="Sentence containing the mention")


class FinancialSentimentResult(BaseModel):
    """Financial sentiment analysis result from FinBERT."""
    sentiment: str = Field(description="positive, negative, or neutral")
    score: float = Field(ge=0, le=1, description="Confidence score")
    positive_prob: float = Field(ge=0, le=1)
    negative_prob: float = Field(ge=0, le=1)
    neutral_prob: float = Field(ge=0, le=1)


class CompanyMentionResult(BaseModel):
    """Complete company mention with NER + FinBERT analysis."""
    company_name: str
    ticker: str
    sentiment_score: float = Field(ge=-1, le=1, description="Normalized sentiment (-1 to 1)")
    confidence_score: float = Field(ge=0, le=1, description="NER confidence")
    context_sentence: str
    sentiment_label: str = Field(description="positive, negative, or neutral")


class ArticleCompanyAnalysis(BaseModel):
    """Analysis results for a single article."""
    article_url: str
    article_title: str
    mentions: list[CompanyMentionResult] = Field(default_factory=list)
    companies_found: int = 0


class CompanyAnalysisResponse(BaseModel):
    """Response for company analysis endpoints."""
    success: bool
    articles_analyzed: int = 0
    total_mentions: int = 0
    results: list[ArticleCompanyAnalysis] = Field(default_factory=list)


class CompanyInfo(BaseModel):
    """Company information from the database."""
    id: int
    name: str
    ticker: str
    icon_url: Optional[str] = None


class CompanyMentionRecord(BaseModel):
    """A company mention record for database operations."""
    article_id: int
    company_id: int
    sentiment_score: float
    confidence_score: float
    context_sentence: str


class CompanyDailyMetrics(BaseModel):
    """Daily aggregated metrics for a company."""
    date: str
    company_id: int
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    avg_sentiment: float
    article_volume: int
