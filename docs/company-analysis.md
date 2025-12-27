# Company Analysis (NER + FinBERT)

This feature detects mentions of the "Magnificent 7" tech companies in news articles and analyzes the financial sentiment around each mention.

## Overview

```
Article Content → GLiNER-spaCy (NER) → Company Mentions + Context Sentences
                                            ↓
                              FinBERT (Financial Sentiment per mention)
                                            ↓
                              Supabase (company_mentions table)
```

## Tracked Companies (Magnificent 7)

| Company | Ticker | Aliases Detected |
|---------|--------|------------------|
| Apple | AAPL | apple, apple inc, aapl |
| Microsoft | MSFT | microsoft, microsoft corp, msft |
| Google | GOOGL | google, alphabet, googl, goog |
| Amazon | AMZN | amazon, amazon.com, amzn, aws |
| Meta | META | meta, meta platforms, facebook |
| Tesla | TSLA | tesla, tesla inc, tsla |
| Nvidia | NVDA | nvidia, nvidia corp, nvda |

## How It Works

### 1. Named Entity Recognition (NER)
- Uses **GLiNER-spaCy** to detect organization entities in article text
- Maps detected entities to Magnificent 7 companies using alias matching
- Falls back to regex pattern matching if GLiNER fails to load

### 2. Context Extraction
- Extracts the sentence containing each company mention
- This provides context for sentiment analysis

### 3. Financial Sentiment Analysis
- Uses **FinBERT** (ProsusAI/finbert) for financial-domain sentiment
- Analyzes each context sentence separately
- Returns: positive/negative/neutral label + confidence scores

### 4. Database Sync
- Results stored in `company_mentions` table
- Links to `articles` and `companies` tables via foreign keys
- `company_daily_metrics` table is auto-updated with daily aggregations

## Database Schema

The feature uses these existing tables:

```sql
-- Companies table (seed with Magnificent 7)
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    ticker TEXT UNIQUE NOT NULL,
    icon_url TEXT
);

-- Company mentions (links articles to companies)
CREATE TABLE company_mentions (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT REFERENCES articles(id),
    company_id INTEGER REFERENCES companies(id),
    sentiment_score FLOAT NOT NULL,      -- -1 to 1 (FinBERT normalized)
    confidence_score FLOAT,               -- 0 to 1 (NER confidence)
    context_sentence TEXT,                -- Sentence containing mention
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(article_id, company_id)        -- One entry per company per article
);

-- Daily aggregated metrics
CREATE TABLE company_daily_metrics (
    date DATE NOT NULL,
    company_id INTEGER REFERENCES companies(id),
    avg_sentiment FLOAT,
    article_volume INTEGER,
    PRIMARY KEY (date, company_id)
);
```

## API Endpoints

### Company Information

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/news/companies` | GET | List tracked companies |
| `/api/news/companies/mentions/{ticker}` | GET | Get mentions for a company |
| `/api/news/companies/metrics/{ticker}` | GET | Get daily sentiment metrics |

### Analysis

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/news/analyze/companies?url=...` | POST | Analyze single article |
| `/api/news/analyze/companies/batch` | POST | Batch analyze (background) |

### Supabase Sync

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/news/supabase/seed/companies` | POST | Seed Magnificent 7 to DB |
| `/api/news/supabase/sync/companies` | POST | Sync mentions to Supabase |
| `/api/news/supabase/sync/all` | POST | Full sync (includes companies) |

## Setup Instructions

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install spacy>=3.7.0 gliner-spacy>=0.0.10 transformers>=4.36.0 torch>=2.1.0
```

**Note**: PyTorch is ~2GB. For CPU-only (recommended for servers):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Seed Companies Table

Run once to populate the companies table:

```bash
curl -X POST http://localhost:3000/api/news/supabase/seed/companies
```

### 3. Test Analysis

Test with a single article:

```bash
# First, fetch some news
curl -X POST http://localhost:3000/api/news/aggregate

# Extract content for an article
curl "http://localhost:3000/api/news/article?url=<article_url>"

# Analyze for company mentions
curl -X POST "http://localhost:3000/api/news/analyze/companies?url=<article_url>"
```

### 4. Sync to Supabase

```bash
# Sync company mentions only
curl -X POST "http://localhost:3000/api/news/supabase/sync/companies?limit=50"

# Or use full sync (includes company analysis by default)
curl -X POST "http://localhost:3000/api/news/supabase/sync/all?analyze_companies=true"
```

## Integration with Existing Timer

The existing systemd timer already calls `/api/news/supabase/sync/all`, which now includes company analysis by default.

**No changes needed to the timer** - just restart the server after updating.

The sync flow now includes:
1. Fetch articles from RSS feeds
2. Extract full content
3. Sync articles to Supabase
4. Analyze overall sentiment (VADER)
5. Analyze company mentions (GLiNER + FinBERT)
6. Sync company_mentions to Supabase
7. **Update company_daily_metrics** (auto-aggregation)

To disable company analysis in the sync:
```bash
curl -X POST "http://localhost:3000/api/news/supabase/sync/all?analyze_companies=false"
```

## Performance Considerations

### Model Loading
- **First request is slow** (~30-60 seconds) as models load
- GLiNER: ~400MB
- FinBERT: ~400MB
- Subsequent requests use cached model instances

### Memory Usage
- Expect ~2-3GB additional RAM when models are loaded
- Models run on CPU by default (GPU optional)

### Processing Time
- ~1-3 seconds per article for NER + sentiment analysis
- Background tasks recommended for batch processing

## Example Response

```json
{
  "success": true,
  "articles_analyzed": 1,
  "total_mentions": 3,
  "results": [
    {
      "article_url": "https://example.com/article",
      "article_title": "Tech Giants Report Strong Earnings",
      "companies_found": 2,
      "mentions": [
        {
          "company_name": "Apple",
          "ticker": "AAPL",
          "sentiment_score": 0.75,
          "confidence_score": 0.92,
          "context_sentence": "Apple reported record-breaking iPhone sales this quarter.",
          "sentiment_label": "positive"
        },
        {
          "company_name": "Microsoft",
          "ticker": "MSFT",
          "sentiment_score": 0.45,
          "confidence_score": 0.88,
          "context_sentence": "Microsoft's cloud division showed steady growth.",
          "sentiment_label": "positive"
        }
      ]
    }
  ]
}
```

## Troubleshooting

### Models fail to load

Check if you have enough memory:
```bash
free -h
```

Try loading models manually:
```python
python -c "from transformers import pipeline; p = pipeline('sentiment-analysis', model='ProsusAI/finbert'); print('FinBERT loaded')"
```

### No companies detected

The fallback regex pattern matching will activate if GLiNER fails. Check logs:
```bash
sudo journalctl -u news-aggregator -n 100 | grep -i "ner\|gliner"
```

### Slow first request

This is expected - models load lazily on first use. Consider preloading:
```bash
# Warm up the models after server start
curl -X POST "http://localhost:3000/api/news/analyze/companies?url=<any_article_url>"
```
