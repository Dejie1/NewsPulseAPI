# News Aggregator Deployment Guide

## Server Information

- **Server IP**: 69.62.67.210
- **Port**: 3000
- **Hosting Provider**: Hostinger
- **Server Panel**: 宝塔 (BT Panel)

## Architecture

```
Your Server (Hostinger)                    Supabase
┌───────────────────────────────────┐      ┌──────────────────────────┐
│  FastAPI Server (localhost:3000)  │      │  articles table          │
│                                   │      │  sources table           │
│  + NER (GLiNER-spaCy)             │      │  companies table         │
│  + FinBERT (financial sentiment)  │      │  company_mentions table  │
│                                   │      │  company_daily_metrics   │
│  systemd timer (every 6hrs)       │      │                          │
│       ↓                           │      │                          │
│  curl POST /sync/all              │─────→│  Data synced             │
└───────────────────────────────────┘      └──────────────────────────┘
```

**Note**: Supabase pg_net cannot reach Hostinger servers directly (TCP timeout issue).
The solution is to use a server-side cron/timer that initiates the sync.

## Running the Server

```bash
# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 3000

# Or with auto-reload for development
uvicorn app.main:app --host 0.0.0.0 --port 3000 --reload
```

## Systemd Timer Setup

The timer automatically syncs articles to Supabase every 6 hours.

### Service File: `/etc/systemd/system/news-sync.service`

```ini
[Unit]
Description=News Sync Service

[Service]
Type=oneshot
ExecStart=/usr/bin/curl -X POST http://localhost:3000/api/news/supabase/sync/all
```

### Timer File: `/etc/systemd/system/news-sync.timer`

```ini
[Unit]
Description=Run news sync every 6 hours

[Timer]
OnCalendar=*-*-* 00,06,12,18:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

### Setup Commands

```bash
# Reload systemd after editing files
sudo systemctl daemon-reload

# Enable timer to start on boot
sudo systemctl enable news-sync.timer

# Start the timer
sudo systemctl start news-sync.timer
```

## Useful Commands

### Timer Management

```bash
# Check timer status
sudo systemctl status news-sync.timer

# See when it will run next
sudo systemctl list-timers | grep news

# Run sync manually anytime
sudo systemctl start news-sync.service

# View sync logs
sudo journalctl -u news-sync.service

# Stop the timer
sudo systemctl stop news-sync.timer

# Disable timer (won't start on boot)
sudo systemctl disable news-sync.timer
```

### Server Management

```bash
# Check what's listening on port 3000
netstat -tlnp | grep 3000

# Check server firewall (宝塔)
# Go to 宝塔 Panel → Security → Firewall
# Ensure port 3000 is open (放行) for all IPs (所有IP)
```

## Schedule Reference

Current schedule: `00,06,12,18:00:00` = runs at 00:00, 06:00, 12:00, 18:00 UTC

### Other Schedule Options

```ini
# Every hour
OnCalendar=hourly

# Every 2 hours
OnCalendar=*-*-* */2:00:00

# Every 30 minutes
OnCalendar=*-*-* *:00,30:00

# Every day at midnight
OnCalendar=daily

# Every day at 9am UTC
OnCalendar=*-*-* 09:00:00
```

To change schedule:
```bash
sudo nano /etc/systemd/system/news-sync.timer
sudo systemctl daemon-reload
sudo systemctl restart news-sync.timer
```

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/news/` | GET | Get cached articles |
| `/api/news/aggregate` | POST | Fetch fresh articles from RSS |
| `/api/news/status` | GET | Check aggregator status |

### Supabase Sync Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/news/supabase/status` | GET | Check Supabase connection |
| `/api/news/supabase/sync/articles` | POST | Sync articles to Supabase |
| `/api/news/supabase/sync/sentiments` | POST | Analyze and sync sentiments |
| `/api/news/supabase/sync/companies` | POST | Analyze and sync company mentions |
| `/api/news/supabase/sync/all` | POST | Full sync (includes company analysis) |
| `/api/news/supabase/seed/companies` | POST | Seed Magnificent 7 companies |

### Company Analysis Endpoints (NEW)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/news/companies` | GET | List tracked companies |
| `/api/news/companies/mentions/{ticker}` | GET | Get mentions for a company |
| `/api/news/companies/metrics/{ticker}` | GET | Get daily sentiment metrics |
| `/api/news/analyze/companies?url=...` | POST | Analyze single article for companies |
| `/api/news/analyze/companies/batch` | POST | Batch analyze (background) |

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/news/sentiment?url=...` | GET | Analyze single article sentiment (VADER) |
| `/api/news/summarize?url=...` | GET | Summarize article content |
| `/api/news/sources` | GET | List configured RSS sources |

## Supabase Database Schema

Run these in Supabase SQL Editor to set up the required tables:

```sql
-- Sources table
CREATE TABLE IF NOT EXISTS sources (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    domain TEXT UNIQUE,
    logo_url TEXT,
    reliability_score FLOAT DEFAULT 0.5,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Articles table
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    summary TEXT,
    full_content TEXT,
    image_url TEXT,
    published_at TIMESTAMP WITH TIME ZONE,
    category TEXT,
    overall_sentiment FLOAT,
    source_id INTEGER REFERENCES sources(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_source_id ON articles(source_id);
```

### Add full_content column (if table already exists)

```sql
ALTER TABLE articles ADD COLUMN IF NOT EXISTS full_content TEXT;
```

### Company Analysis Tables (NEW)

```sql
-- Companies table (Magnificent 7)
CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    ticker TEXT UNIQUE NOT NULL,
    icon_url TEXT
);

-- Company mentions (links articles to companies with sentiment)
CREATE TABLE IF NOT EXISTS company_mentions (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT REFERENCES articles(id),
    company_id INTEGER REFERENCES companies(id),
    sentiment_score FLOAT NOT NULL,
    confidence_score FLOAT,
    context_sentence TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(article_id, company_id)
);

-- Daily aggregated metrics
CREATE TABLE IF NOT EXISTS company_daily_metrics (
    date DATE NOT NULL,
    company_id INTEGER REFERENCES companies(id),
    avg_sentiment FLOAT,
    article_volume INTEGER,
    PRIMARY KEY (date, company_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_company_mentions_article ON company_mentions(article_id);
CREATE INDEX IF NOT EXISTS idx_company_mentions_company ON company_mentions(company_id);
CREATE INDEX IF NOT EXISTS idx_company_mentions_created ON company_mentions(created_at DESC);
```

## Environment Variables

Create a `.env` file in the project root:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Or use these alternative names
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
```

## NLTK Setup (Required for Sentiment Analysis)

NLTK needs the VADER lexicon for sentiment analysis. Download it manually:

```bash
# Create directory
mkdir -p /home/www/nltk_data/sentiment
mkdir -p /usr/share/nltk_data/sentiment

# Download VADER lexicon (as actual zip file)
cd /home/www/nltk_data/sentiment
wget --no-check-certificate -O vader_lexicon.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip
chown www:www vader_lexicon.zip

cd /usr/share/nltk_data/sentiment
wget --no-check-certificate -O vader_lexicon.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/sentiment/vader_lexicon.zip
chmod 644 vader_lexicon.zip
```

**Important**: The file must be an actual `.zip` file, not a directory named `.zip`.

Verify:
```bash
file /home/www/nltk_data/sentiment/vader_lexicon.zip
# Should output: "Zip archive data"
```

## Company Analysis Setup (NER + FinBERT)

The company analysis feature uses GLiNER-spaCy for NER and FinBERT for financial sentiment.
It detects mentions of the Magnificent 7 (Apple, Microsoft, Google, Amazon, Meta, Tesla, Nvidia).

### 1. Install New Dependencies

```bash
cd /www/wwwroot/aggregator  # or your project path

# Install with uv
uv sync

# Or with pip (CPU-only torch recommended for servers)
pip install spacy>=3.7.0 gliner-spacy>=0.0.10 transformers>=4.36.0
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Memory Note**: Models require ~2-3GB RAM when loaded. Ensure your server has sufficient memory.

### 2. Create Database Tables

Run in Supabase SQL Editor (see "Company Analysis Tables" section above).

### 3. Seed Companies

After server restart, seed the Magnificent 7:
```bash
curl -X POST http://localhost:3000/api/news/supabase/seed/companies
```

### 4. Verify Setup

Test the analysis endpoint:
```bash
# First request will be slow (models loading ~30-60 seconds)
curl -X POST "http://localhost:3000/api/news/analyze/companies?url=<article_url>"
```

### 5. Integration with Timer

**Good news: No timer changes needed!**

The existing `/api/news/supabase/sync/all` endpoint now includes company analysis by default.
Just restart the server and the next timer run will include company analysis.

To explicitly control company analysis:
```bash
# With company analysis (default)
curl -X POST "http://localhost:3000/api/news/supabase/sync/all?analyze_companies=true"

# Without company analysis
curl -X POST "http://localhost:3000/api/news/supabase/sync/all?analyze_companies=false"
```

### 6. Monitor Company Analysis

```bash
# Check companies in database
curl http://localhost:3000/api/news/companies

# Check mentions for a specific company
curl "http://localhost:3000/api/news/companies/mentions/AAPL?days=7"

# Check daily metrics
curl "http://localhost:3000/api/news/companies/metrics/TSLA?days=7"
```

### Performance Tips

1. **First sync is slow**: Models load on first use (~30-60 seconds)
2. **Warm up after restart**: Run a test analysis after server restart
3. **Memory monitoring**: Watch RAM usage during sync
   ```bash
   watch -n 5 free -h
   ```

## Troubleshooting

### Server not accessible

1. Check if server is running: `netstat -tlnp | grep 3000`
2. Should show `0.0.0.0:3000` (not `127.0.0.1:3000`)
3. Check 宝塔 firewall: port 3000 should be 放行

### Timer not running

```bash
# Check timer status
sudo systemctl status news-sync.timer

# Check for errors
sudo journalctl -u news-sync.service -n 50
```

### Supabase not syncing

```bash
# Test manually
curl -X POST http://localhost:3000/api/news/supabase/sync/articles

# Check Supabase connection
curl http://localhost:3000/api/news/supabase/status
```

### View recent sync logs

```bash
sudo journalctl -u news-sync.service --since "1 hour ago"
```

### Company analysis not working

1. Check if models are loading:
   ```bash
   sudo journalctl -u news-aggregator --since "1 hour ago" | grep -i "ner\|finbert\|gliner"
   ```

2. Check memory (models need ~2-3GB):
   ```bash
   free -h
   ```

3. Test models manually:
   ```bash
   python -c "from transformers import pipeline; p = pipeline('sentiment-analysis', model='ProsusAI/finbert'); print('OK')"
   ```

4. If GLiNER fails, the system falls back to regex pattern matching (still works, just less accurate)

### No company mentions found

1. Verify companies are seeded:
   ```bash
   curl http://localhost:3000/api/news/companies
   ```

2. Check if articles have content:
   ```bash
   curl "http://localhost:3000/api/news/supabase/articles?limit=5"
   # Look for "full_content" field
   ```

3. Run content extraction first:
   ```bash
   curl -X POST http://localhost:3000/api/news/extract-content?limit=20
   ```
