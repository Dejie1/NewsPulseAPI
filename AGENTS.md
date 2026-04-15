# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Cursor, etc.) working in this repository.

## Project Overview

A FastAPI-based RSS news aggregator service designed for React Native apps. Aggregates news from multiple RSS sources, provides sentiment analysis (RoBERTa + FinBERT), article summarization, recommendations, company entity recognition (Magnificent 7 tracking), and syncs to Supabase.

## Development Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run development server with auto-reload
uv run uvicorn app.main:app --reload --port 8000

# Production server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

```

API docs available at `http://localhost:8000/docs` after starting the server.

## Architecture

### Core Data Flows

```
1. AGGREGATION:
   RSS Feeds → RSSParser (rate limited 2 req/sec) → Deduplicator → Cache → API

2. CONTENT EXTRACTION:
   Article URL → trafilatura (curl-cffi) → Clean Text → Cache

3. COMPANY ANALYSIS:
   Article Text → GLiNER NER → Company Mentions → FinBERT Sentiment → Supabase
```

### Service Layer (`app/services/`)

**Core Services:**
- **aggregator.py**: Central orchestrator. Coordinates RSS parsing, deduplication, caching, and content extraction. Access via `get_aggregator()` singleton.
- **cache.py**: In-memory cache with 5-minute TTL. Prevents concurrent updates with `is_updating` flag.
- **rss_parser.py**: Fetches and parses RSS feeds using feedparser.
- **deduplicator.py**: URL and title-based deduplication.
- **content_extractor.py**: Content extraction using trafilatura + curl-cffi. Extracts clean article text from URLs.

**Analysis Services:**
- **sentiment.py**: RoBERTa sentiment analysis (cardiffnlp/twitter-roberta-base-sentiment-latest). Lazy-loads model on first use.
- **summarizer.py**: Extractive summarization using sumy (LSA algorithm). Has hooks for AI providers (OpenAI, Claude, Ollama).
- **recommender.py**: TF-IDF + cosine similarity for article recommendations using scikit-learn.

**Company Analysis (Magnificent 7 tracking):**
- **company_analysis_service.py**: Orchestrates NER + FinBERT pipeline for company mentions.
- **ner_service.py**: GLiNER-spaCy for entity recognition. Maps detected entities to AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA. Falls back to regex if GLiNER unavailable.
- **finbert_service.py**: ProsusAI/finbert for financial sentiment. GPU support with CPU fallback.

**Database:**
- **supabase_service.py**: Syncs articles/company mentions to Supabase. Uses `supabase_service` singleton.

### Key Patterns

1. **Singleton Services**: Module-level singletons via getter functions (`get_aggregator()`, `get_cache()`, etc.)
2. **Lazy Model Loading**: Heavy ML models (RoBERTa, GLiNER, FinBERT) load only on first use. First requests are slow (~30-60s).
3. **Background Tasks**: FastAPI BackgroundTasks for async operations
4. **Cache-First**: GET `/api/news/` returns cached data; POST `/api/news/aggregate` forces refresh

### Route Structure (`app/routes/news.py`)

All routes prefixed with `/api/news`:
- **Core**: `/`, `/aggregate`, `/aggregate/background`, `/status`, `/sources`, `/cache`
- **Content**: `/article`, `/extract-content`, `/summarize`
- **Sentiment**: `/sentiment`, `/sentiment/batch`, `/sentiment/text`
- **Recommendations**: `/recommendations/build`, `/recommendations`, `/recommendations/search`
- **Company Analysis**: `/analyze/companies`, `/analyze/companies/batch`, `/companies`, `/companies/mentions/{ticker}`, `/companies/metrics/{ticker}`
- **Supabase Sync**: `/supabase/sync/articles`, `/supabase/sync/companies`, `/supabase/sync/all`, `/supabase/seed/companies`

## Configuration

### Environment Variables (`.env`)

```env
AGGREGATOR_DEBUG=false
AGGREGATOR_CACHE_TTL_SECONDS=300
AGGREGATOR_REQUESTS_PER_SECOND=2.0
AGGREGATOR_REQUEST_TIMEOUT_SECONDS=10
AGGREGATOR_MAX_ARTICLES_PER_FEED=50
AGGREGATOR_MAX_TOTAL_ARTICLES=200

# Supabase (supports multiple prefixes)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-key
# Or: NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY
```

### Feed Sources

Defined in `app/config.py` as `FEED_SOURCES` list. Add new sources by appending `FeedSource(name, url, category)`.

## Supabase Schema

**Core Tables:**
- `sources` (name, domain, reliability_score)
- `articles` (title, url, summary, full_content, image_url, overall_sentiment, source_id) - uses `url` as unique constraint for upserts

**Company Analysis Tables:**
- `companies` (name, ticker, icon_url) - seed with Magnificent 7
- `company_mentions` (article_id, company_id, sentiment_score, confidence_score, context_sentence)
- `company_daily_metrics` (date, company_id, avg_sentiment, article_volume)

## Performance Notes

- **First request with ML models is slow** (~30-60s) as models load lazily
- **Memory**: Expect ~2-3GB additional RAM when GLiNER + FinBERT loaded
- **Company analysis**: ~1-3 seconds per article for NER + FinBERT
- **Content extraction**: trafilatura ~1-3s per article
