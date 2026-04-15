# NewsPulse Aggregator API

FastAPI service that ingests RSS feeds, runs NLP analysis (sentiment, NER, financial sentiment), and **syncs the results to Supabase**.

## Quick Start

### 1. Install Dependencies (using uv)

```bash
uv sync
uv run python -m nltk.downloader vader_lexicon
# Development mode with auto-reload
uv run uvicorn app.main:app --reload --port 8000

# Production mode
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```


- **API docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

## API Endpoints

### Core Aggregation

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/news/` | Get aggregated articles (cached) |
| POST | `/api/news/aggregate` | Trigger fresh aggregation |
| POST | `/api/news/aggregate/background` | Trigger aggregation in background |
| GET | `/api/news/status` | Get aggregator status |
| GET | `/api/news/sources` | List configured feed sources |
| DELETE | `/api/news/cache` | Clear the cache |

### Content Extraction & Summarization

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/news/article?url=...` | Get article with full content extracted |
| POST | `/api/news/extract-content` | Extract content for multiple articles (background) |
| GET | `/api/news/summarize?url=...` | Summarize an article's content |

### Sentiment Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/news/sentiment?url=...` | Analyze sentiment of an article |
| GET | `/api/news/sentiment/batch` | Analyze multiple articles |
| POST | `/api/news/sentiment/text?text=...` | Analyze arbitrary text |

### Recommendations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/news/recommendations/build` | Build TF-IDF index (required first) |
| GET | `/api/news/recommendations?url=...` | Get similar articles |
| GET | `/api/news/recommendations/search?query=...` | Search by text |

> **Note:** For detailed documentation on sentiment and recommendations, see [docs/ANALYSIS.md](docs/ANALYSIS.md)

## How It Works

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           React Native App                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌───────────────┐ │
│  │ RSS Parser  │→ │  Aggregator  │→ │   Cache     │→ │ API Response  │ │
│  │ (feedparser)│  │              │  │ (in-memory) │  │               │ │
│  └─────────────┘  └──────────────┘  └─────────────┘  └───────────────┘ │
│         │                │                                              │
│         ▼                ▼                                              │
│  ┌─────────────┐  ┌──────────────┐                                     │
│  │Rate Limiter │  │ Deduplicator │                                     │
│  └─────────────┘  └──────────────┘                                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Content & Summarization                       │   │
│  │  ┌─────────────────┐           ┌─────────────────────────────┐  │   │
│  │  │Content Extractor│    →      │      Summarizer             │  │   │
│  │  │ (trafilatura)   │           │ (extractive / AI-ready)     │  │   │
│  │  └─────────────────┘           └─────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         External RSS Feeds                               │
│     NYTimes Technology  │  Wired Business  │  Wired Culture             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. AGGREGATION FLOW
   ┌──────────┐     ┌───────────┐     ┌──────────┐     ┌─────────┐
   │ RSS Feed │ ──► │ RSS Parser│ ──► │Deduplicat│ ──► │  Cache  │
   │ (XML)    │     │(feedparser│     │   or     │     │(memory) │
   └──────────┘     └───────────┘     └──────────┘     └─────────┘
                          │
                          ▼
                    Rate Limited
                   (2 req/sec max)

2. CONTENT EXTRACTION FLOW
   ┌─────────┐     ┌───────────┐     ┌──────────┐     ┌─────────┐
   │ Article │ ──► │  Fetch    │ ──► │trafilatur│ ──► │  Cache  │
   │  URL    │     │  HTML     │     │a extract │     │ Update  │
   └─────────┘     └───────────┘     └──────────┘     └─────────┘

3. SUMMARIZATION FLOW
   ┌─────────┐     ┌───────────┐     ┌──────────┐
   │ Content │ ──► │Summarizer │ ──► │ Summary  │
   │ (text)  │     │           │     │  Output  │
   └─────────┘     └───────────┘     └──────────┘
```

### The Role of Cache

The cache is **central to the system's efficiency**:

```
WITHOUT CACHE (slow):
  Request → Fetch all RSS feeds → Parse → Deduplicate → Return
  Time: 3-10 seconds (network dependent)

WITH CACHE (fast):
  Request → Return cached data
  Time: <50ms
```

**Cache behavior:**
- **TTL (Time-To-Live)**: 5 minutes by default
- **First request**: Fetches from RSS feeds, stores in cache
- **Subsequent requests**: Returns cached data instantly
- **After TTL expires**: Next request fetches fresh data
- **Force refresh**: Use `POST /aggregate` to bypass cache

**Why in-memory cache for POC?**
- Simple, no external dependencies
- Fast enough for single-server deployment
- Can be upgraded to Redis for production (multi-server)

### How Summarization Works (Using Sumy)

The summarizer uses **sumy** library with **LSA (Latent Semantic Analysis)** by default - a proper NLP algorithm, not just truncation:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SUMY SUMMARIZATION PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│  1. TOKENIZATION                                                 │
│     "The fox jumped. It was fast." → ["The fox jumped", "It..."]│
│                                                                  │
│  2. BUILD MATRIX (TF-IDF)                                       │
│     Words → Frequency scores per sentence                        │
│                                                                  │
│  3. LSA (Singular Value Decomposition)                          │
│     Find semantic relationships between sentences                │
│                                                                  │
│  4. RANK SENTENCES                                               │
│     Score each sentence by importance                            │
│                                                                  │
│  5. SELECT TOP N                                                 │
│     Return most important sentences                              │
└─────────────────────────────────────────────────────────────────┘
```

**Available algorithms:**

| Algorithm | How it works | Best for |
|-----------|--------------|----------|
| **LSA** (default) | Latent Semantic Analysis - finds hidden topics | General purpose |
| **TextRank** | Graph-based, like Google PageRank | News articles |
| **LexRank** | Graph + cosine similarity | Multi-document |
| **Luhn** | Classic word frequency | Simple texts |

**Why this is better than simple truncation:**
- Picks the **most important** sentences, not just the first ones
- Understands semantic relationships between words
- No external API calls required

### Content Extraction with Trafilatura

**What is trafilatura?**
- Python library for extracting main content from web pages
- Removes ads, navigation, sidebars, footers
- Returns clean, readable text

**Example:**
```
Input URL: https://nytimes.com/article/...
   └── Full webpage HTML (ads, menus, scripts, article, comments...)

trafilatura.extract()
   └── "President Trump said Nvidia can export some chips. But years of
        U.S. restrictions have propelled China to make everything it
        needs for advanced A.I. The semiconductor industry has been..."
```

## Usage Examples

### From React Native App

```typescript
const API_URL = 'http://your-server:8000';

// 1. Fetch news (uses cache, fast)
const news = await fetch(`${API_URL}/api/news/`);
const data = await news.json();
// data.articles = [{title, link, source, published_at, description, image}, ...]

// 2. Get full article content
const article = await fetch(`${API_URL}/api/news/article?url=${encodeURIComponent(articleUrl)}`);
const fullArticle = await article.json();
// fullArticle.content = "Full article text..."

// 3. Get article summary
const summary = await fetch(`${API_URL}/api/news/summarize?url=${encodeURIComponent(articleUrl)}&max_length=100`);
const summaryData = await summary.json();
// summaryData.summary = "First few sentences..."
```

### Using cURL

```bash
# Get news (cached, fast)
curl http://localhost:8000/api/news/

# Force fresh aggregation
curl -X POST http://localhost:8000/api/news/aggregate

# Get full article content
curl "http://localhost:8000/api/news/article?url=https://www.nytimes.com/..."

# Get article summary
curl "http://localhost:8000/api/news/summarize?url=https://www.nytimes.com/...&max_length=150"

# Extract content for 10 articles in background
curl -X POST "http://localhost:8000/api/news/extract-content?limit=10"

# Check status
curl http://localhost:8000/api/news/status
```

## Configuration

### Environment Variables

Create a `.env` file to customize settings:

```env
AGGREGATOR_DEBUG=false
AGGREGATOR_CACHE_TTL_SECONDS=300
AGGREGATOR_REQUESTS_PER_SECOND=2.0
AGGREGATOR_REQUEST_TIMEOUT_SECONDS=10
AGGREGATOR_MAX_ARTICLES_PER_FEED=50
AGGREGATOR_MAX_TOTAL_ARTICLES=200
```

### Adding New Feed Sources

Edit `app/config.py`:

```python
FEED_SOURCES: list[FeedSource] = [
    FeedSource(
        name="Your Source Name",
        url="https://example.com/rss/feed.xml",
        category="technology"
    ),
]
```

## Project Structure

```
app/
├── main.py                    # FastAPI app entry point
├── config.py                  # Settings & feed sources
├── models.py                  # Pydantic data models
├── services/
│   ├── rss_parser.py          # RSS fetching & parsing (feedparser)
│   ├── aggregator.py          # Core orchestration logic
│   ├── cache.py               # In-memory cache (5 min TTL)
│   ├── deduplicator.py        # URL/title deduplication
│   ├── content_extractor.py   # Full article extraction (trafilatura)
│   └── summarizer.py          # Summarization (extractive + AI hooks)
├── utils/
│   └── rate_limiter.py        # Token bucket rate limiting
└── routes/
    └── news.py                # API endpoints
```

## Cron job

Aggregation is driven by an external cron entry on the host (the API does not self-schedule):

```cron
*/5 * * * * curl -fsS -X POST http://localhost:8000/api/news/aggregate >> /var/log/newspulse-agg.log 2>&1
```

The endpoint runs synchronously and returns the aggregation result; `>> log 2>&1` keeps a record of failures so a dead source surfaces in the host logs rather than silently breaking the feed.

## Tech Stack Summary

| Component | Library | Purpose |
|-----------|---------|---------|
| Web Framework | FastAPI | REST API with auto-docs |
| RSS Parsing | feedparser | Parse RSS/Atom feeds |
| HTTP Client | httpx | Async HTTP requests |
| Content Extraction | trafilatura | Extract article text from HTML |
| Summarization | sumy + NLTK | NLP-based extractive summarization |
| Sentiment Analysis | NLTK VADER | Dictionary-based sentiment scoring |
| Recommendations | scikit-learn | TF-IDF + Cosine Similarity |
| Data Validation | Pydantic | Type-safe models |
| Rate Limiting | Custom | Prevent overwhelming sources |

## Additional Documentation

- **[Analysis Features (Sentiment & Recommendations)](docs/ANALYSIS.md)** - Detailed documentation on sentiment analysis and recommendation system

## Future Improvements

- [x] ~~Article summarization~~ (sumy LSA, AI-ready)
- [x] ~~Full content extraction~~
- [x] ~~Sentiment analysis~~ (VADER)
- [x] ~~Article recommendations~~ (TF-IDF cosine similarity)
- [ ] Redis caching for distributed deployments
- [ ] SQLite/PostgreSQL for article persistence
- [ ] AI summarization (OpenAI/Claude/Ollama)
- [ ] Automatic tagging/categorization
- [ ] User preferences for personalized feeds
- [ ] Full-text search
