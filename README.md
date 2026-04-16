# NewsPulse Aggregator API

FastAPI service that ingests RSS feeds, runs NLP analysis (sentiment, NER, financial sentiment), and **syncs the results to Supabase**.

## Quick Start

```bash
uv sync

# Dev with auto-reload
uv run uvicorn app.main:app --reload --port 8000

# Prod
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```


- **API docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health


### Aggregation

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/news/` | Get aggregated articles (cached) |
| POST | `/api/news/aggregate` | Trigger fresh aggregation |
| POST | `/api/news/aggregate/background` | Run aggregation in the background |
| GET | `/api/news/status` | Aggregator status |
| GET | `/api/news/sources` | List configured feed sources |
| DELETE | `/api/news/cache` | Clear the cache |

### Content & Summarization

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/news/article?url=...` | Get article with full content extracted |
| POST | `/api/news/extract-content` | Extract content for multiple articles (background) |
| GET | `/api/news/summarize?url=...` | Summarize an article's content |

### Sentiment Analysis

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/news/sentiment?url=...` | Analyze sentiment of an article |
| GET | `/api/news/sentiment/batch` | Analyze multiple articles |
| POST | `/api/news/sentiment/text` | Analyze arbitrary text |

### Company Analysis (FinBERT + GLiNER)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/news/companies` | List tracked companies (Magnificent 7) |
| POST | `/api/news/analyze/companies` | Run NER + FinBERT on a single article |
| POST | `/api/news/analyze/companies/batch` | Batch company analysis |
| GET | `/api/news/companies/mentions/{ticker}` | Mentions for a given ticker |
| GET | `/api/news/companies/metrics/{ticker}` | Aggregated metrics for a ticker |

### Supabase Sync

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/news/supabase/sync/articles` | Sync articles |
| POST | `/api/news/supabase/sync/sentiments` | Sync sentiment scores |
| POST | `/api/news/supabase/sync/companies` | Sync company mentions |
| POST | `/api/news/supabase/sync/all` | Run full sync pipeline |
| GET | `/api/news/supabase/status` | Check Supabase connectivity |

## How It Works

### System Architecture

The mobile app and backend are decoupled — they only meet at Supabase. The FastAPI service is a cron-driven ingestion worker: it pulls from RSS, runs the NLP pipeline, and writes results to Supabase. The mobile app reads directly from Supabase.

```
  ┌──────────────────┐                                    ┌──────────────────┐
  │  React Native    │                                    │  Cron (host)     │
  │   Mobile App     │                                    │   */5 * * * *    │
  └────────┬─────────┘                                    └────────┬─────────┘
           │ read                                                  │ POST /aggregate
           ▼                                                       ▼
  ┌──────────────────┐       sync         ┌─────────────────────────────────────┐
  │     Supabase     │ ◄───────────────── │          FastAPI Backend            │
  │   (PostgreSQL)   │   articles +       │                                     │
  └──────────────────┘   sentiments +     │  RSS Parser → Aggregator → Cache    │
                         company mentions │     (feedparser)  + Dedup + Rate    │
                                          │                                     │
                                          │  Content & NLP Pipeline:            │
                                          │  trafilatura → RoBERTa →            │
                                          │  GLiNER → FinBERT                   │
                                          └──────────────────┬──────────────────┘
                                                             │ fetch
                                                             ▼
                                                  ┌─────────────────────┐
                                                  │  External RSS Feeds │
                                                  │   (see config.py)   │
                                                  └─────────────────────┘
```

### Caching

- **TTL:** 5 min (`AGGREGATOR_CACHE_TTL_SECONDS`)
- **First request:** fetches from RSS, stores result
- **Subsequent requests:** serve cached data (<50ms vs 3–10s for a fresh fetch)
- **Force refresh:** `POST /api/news/aggregate`

In-memory for the POC — swap in Redis if the service needs to scale horizontally.

### Summarization

Exposed as an on-demand endpoint (`GET /api/news/summarize`), not part of the sync pipeline. Uses **sumy** with **LSA (Latent Semantic Analysis)** — picks the most semantically important sentences instead of the first N. Alternatives available via config: TextRank, LexRank, Luhn.

## Usage Examples

```bash
# Get news (cached, fast)
curl http://localhost:8000/api/news/

# Force fresh aggregation
curl -X POST http://localhost:8000/api/news/aggregate

# Get full article content
curl "http://localhost:8000/api/news/article?url=https://www.nytimes.com/..."

# Summary
curl "http://localhost:8000/api/news/summarize?url=https://www.nytimes.com/...&max_length=150"

# Check status
curl http://localhost:8000/api/news/status
```

## Configuration

### Environment Variables

Create a `.env` file (see `.env.example`):

```env
AGGREGATOR_DEBUG=false
AGGREGATOR_CACHE_TTL_SECONDS=300
AGGREGATOR_REQUESTS_PER_SECOND=2.0
AGGREGATOR_REQUEST_TIMEOUT_SECONDS=10
AGGREGATOR_MAX_ARTICLES_PER_FEED=50
AGGREGATOR_MAX_TOTAL_ARTICLES=200
# AGGREGATOR_ALLOWED_ORIGINS=["https://app.example.com"]
```

### Adding New Feed Sources

Edit `app/config.py`:

```python
FEED_SOURCES: list[FeedSource] = [
    FeedSource(
        name="Your Source Name",
        url="https://example.com/rss/feed.xml",
        category="technology",
    ),
]
```

## Project Structure

```
app/
├── main.py                         # FastAPI app entry point
├── config.py                       # Settings & feed sources
├── models.py                       # Pydantic data models
├── services/
│   ├── rss_parser.py               # RSS fetching & parsing (feedparser)
│   ├── aggregator.py               # Core orchestration
│   ├── cache.py                    # In-memory cache (5 min TTL)
│   ├── deduplicator.py             # URL/title deduplication
│   ├── content_extractor.py        # Article extraction (trafilatura + fallbacks)
│   ├── summarizer.py               # Extractive summarization (sumy + LSA)
│   ├── sentiment.py                # General sentiment (RoBERTa)
│   ├── finbert_service.py          # Financial sentiment (FinBERT)
│   ├── ner_service.py              # Entity recognition (GLiNER-spaCy)
│   ├── company_analysis_service.py # NER + FinBERT orchestration
│   └── supabase_service.py         # Supabase sync layer
├── utils/
│   └── rate_limiter.py             # Token bucket rate limiting
└── routes/
    └── news.py                     # API endpoints
```

## Cron

Aggregation is driven by an external cron entry on the host (the API does not self-schedule):

```cron
*/5 * * * * curl -fsS -X POST http://localhost:8000/api/news/aggregate >> /var/log/newspulse-agg.log 2>&1
```

The endpoint runs synchronously and returns the aggregation result. `>> log 2>&1` keeps a record of failures so a dead source surfaces in the host logs rather than silently breaking the feed.

## Tech Stack

| Component | Library | Purpose |
|---|---|---|
| Web Framework | FastAPI | REST API with auto-generated OpenAPI docs |
| RSS Parsing | feedparser | Parse RSS/Atom feeds |
| HTTP Client | httpx + curl-cffi | Async HTTP; curl-cffi impersonation for anti-bot sites |
| Content Extraction | trafilatura + Scrapling | Article text from HTML; headless fallback |
| Summarization | sumy + NLTK | Extractive summarization (LSA) |
| General Sentiment | RoBERTa (cardiffnlp) | News sentiment classification |
| Financial Sentiment | FinBERT (ProsusAI) | Sentiment on detected company mentions |
| Entity Recognition | GLiNER-spaCy | Named entity extraction |
| Data Validation | Pydantic | Type-safe models |
| Persistence | Supabase | PostgreSQL-backed sync target |
