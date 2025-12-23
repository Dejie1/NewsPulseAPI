# News Analysis Features

This document covers the sentiment analysis and recommendation system features.

## Overview

| Feature | Library | Algorithm |
|---------|---------|-----------|
| Sentiment Analysis | NLTK VADER | Dictionary-based sentiment scoring |
| Recommendations | scikit-learn | TF-IDF + Cosine Similarity |

---

## Sentiment Analysis

### What is VADER?

**VADER** (Valence Aware Dictionary and sEntiment Reasoner) is a sentiment analysis tool specifically designed for social media and news text.

Unlike general-purpose sentiment analyzers, VADER understands:
- Emoticons and emojis 😊 👎
- Slang ("sux", "kinda")
- Punctuation emphasis ("Good!!!" vs "Good")
- Capitalization ("AMAZING" vs "amazing")
- Degree modifiers ("very good", "kind of bad")
- Contrasting conjunctions ("good, but not great")

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     VADER SENTIMENT PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT: "The movie was really GREAT! But the ending sucked."    │
│                                                                  │
│  1. TOKENIZE                                                     │
│     → ["movie", "was", "really", "GREAT", "But", "ending",      │
│        "sucked"]                                                 │
│                                                                  │
│  2. LOOKUP SENTIMENT SCORES (from lexicon)                      │
│     → "GREAT" = +3.1 (positive)                                 │
│     → "sucked" = -2.1 (negative)                                │
│                                                                  │
│  3. APPLY MODIFIERS                                              │
│     → "really" boosts next word by 0.293                        │
│     → CAPS boosts by 0.733                                      │
│     → "But" signals contrast                                     │
│                                                                  │
│  4. CALCULATE SCORES                                             │
│     → negative: 0.215                                           │
│     → neutral:  0.450                                           │
│     → positive: 0.335                                           │
│     → compound: 0.4019 (normalized -1 to 1)                     │
│                                                                  │
│  OUTPUT: { compound: 0.4019, label: "positive" }                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Score Interpretation

| Score | Range | Meaning |
|-------|-------|---------|
| `negative` | 0.0 - 1.0 | Proportion of negative sentiment |
| `neutral` | 0.0 - 1.0 | Proportion of neutral sentiment |
| `positive` | 0.0 - 1.0 | Proportion of positive sentiment |
| `compound` | -1.0 to 1.0 | Overall normalized score |

**Label thresholds:**
- `compound >= 0.05` → **positive**
- `compound <= -0.05` → **negative**
- Otherwise → **neutral**

### API Endpoints

#### Analyze Single Article

```bash
GET /api/news/sentiment?url=<article_url>
```

**Response:**
```json
{
  "article_title": "Tech Giants Report Record Profits",
  "article_url": "https://...",
  "negative": 0.045,
  "neutral": 0.712,
  "positive": 0.243,
  "compound": 0.6369,
  "label": "positive"
}
```

#### Analyze Multiple Articles

```bash
GET /api/news/sentiment/batch?limit=10&source=NYTimes%20Technology
```

**Response:**
```json
{
  "success": true,
  "results": [...],
  "average_compound": 0.1523,
  "overall_label": "positive",
  "articles_analyzed": 10
}
```

#### Analyze Custom Text

```bash
POST /api/news/sentiment/text?text=This%20product%20is%20amazing!
```

**Response:**
```json
{
  "text_length": 25,
  "negative": 0.0,
  "neutral": 0.406,
  "positive": 0.594,
  "compound": 0.6239,
  "label": "positive"
}
```

### Use Cases

1. **News Mood Tracking**: Analyze sentiment trends across news sources
2. **Source Comparison**: Compare sentiment between NYTimes vs Wired
3. **Topic Sentiment**: Filter articles by topic and analyze sentiment
4. **Alert System**: Flag articles with extremely negative sentiment

---

## Recommendation System

### What is TF-IDF?

**TF-IDF** (Term Frequency-Inverse Document Frequency) converts text into numerical vectors that capture word importance.

```
TF-IDF = TF × IDF

TF (Term Frequency):     How often a word appears in THIS document
IDF (Inverse Doc Freq):  How rare a word is across ALL documents
```

**Example:**
| Word | TF (in article) | IDF (across all) | TF-IDF |
|------|-----------------|------------------|--------|
| "the" | 0.05 | 0.01 (common) | 0.0005 (low) |
| "quantum" | 0.02 | 0.95 (rare) | 0.019 (high) |
| "computing" | 0.03 | 0.80 (rare-ish) | 0.024 (high) |

Words that appear frequently in one document but rarely in others get **high scores**.

### What is Cosine Similarity?

After converting articles to vectors, we compare them using **cosine similarity** - the angle between two vectors.

```
                    Article B (tech news)
                   /
                  /  θ = small angle
                 /   = high similarity (0.85)
                /
    ──────────────────────────── Article A (tech news)


                    Article C (sports news)
                   /
                  /
                 /  θ = large angle
                /   = low similarity (0.12)
               /
    ──────────────────────────── Article A (tech news)
```

| Similarity | Meaning |
|------------|---------|
| 1.0 | Identical content |
| 0.7 - 0.9 | Very similar |
| 0.4 - 0.7 | Somewhat related |
| 0.0 - 0.4 | Different topics |

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                 RECOMMENDATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. BUILD INDEX (one-time, after fetching news)                 │
│     ┌──────────────────────────────────────────────┐            │
│     │ Article 1: "Apple releases new iPhone..."    │            │
│     │ Article 2: "Google launches Pixel phone..."  │            │
│     │ Article 3: "Stock market crashes..."         │            │
│     └──────────────────────────────────────────────┘            │
│                          ↓                                       │
│     ┌──────────────────────────────────────────────┐            │
│     │         TF-IDF VECTORIZER                    │            │
│     │  - Remove stop words (the, a, is)            │            │
│     │  - Calculate word frequencies                │            │
│     │  - Apply IDF weighting                       │            │
│     └──────────────────────────────────────────────┘            │
│                          ↓                                       │
│     ┌──────────────────────────────────────────────┐            │
│     │         TF-IDF MATRIX                        │            │
│     │  Article 1: [0.2, 0.0, 0.5, 0.1, ...]       │            │
│     │  Article 2: [0.3, 0.0, 0.4, 0.2, ...]       │            │
│     │  Article 3: [0.0, 0.6, 0.0, 0.3, ...]       │            │
│     └──────────────────────────────────────────────┘            │
│                                                                  │
│  2. GET RECOMMENDATIONS (for Article 1)                         │
│     ┌──────────────────────────────────────────────┐            │
│     │  Compare Article 1 vector to all others      │            │
│     │  using COSINE SIMILARITY                     │            │
│     │                                              │            │
│     │  Article 1 vs Article 2: 0.87 (similar!)    │            │
│     │  Article 1 vs Article 3: 0.12 (different)   │            │
│     └──────────────────────────────────────────────┘            │
│                          ↓                                       │
│     Return: [Article 2 (0.87), ...]                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### API Endpoints

#### Build Index (Required First!)

```bash
POST /api/news/recommendations/build
```

**Response:**
```json
{
  "success": true,
  "message": "Recommendation index built successfully",
  "articles_indexed": 45
}
```

#### Get Similar Articles

```bash
GET /api/news/recommendations?url=<article_url>&limit=5
```

**Response:**
```json
{
  "success": true,
  "query_article": "https://nytimes.com/...",
  "recommendations": [
    {
      "title": "Similar Article About Tech",
      "link": "https://...",
      "source": "Wired Business",
      "similarity_score": 0.7234,
      "published_at": "2025-12-09T10:00:00Z"
    },
    ...
  ],
  "algorithm": "tfidf_cosine"
}
```

#### Search by Text Query

```bash
GET /api/news/recommendations/search?query=artificial%20intelligence&limit=5
```

**Response:**
```json
{
  "success": true,
  "query_article": "search: artificial intelligence",
  "recommendations": [...],
  "algorithm": "tfidf_cosine"
}
```

### Use Cases

1. **"Related Articles"**: Show similar articles at the bottom of an article page
2. **"More from this topic"**: Find articles on the same subject
3. **Search**: Find articles matching a user's query
4. **Content Discovery**: Help users find new articles they might like

---

## Complete Usage Flow

```bash
# 1. Fetch news articles
curl http://localhost:8000/api/news/

# 2. Build recommendation index
curl -X POST http://localhost:8000/api/news/recommendations/build

# 3. Pick an article URL from the response
ARTICLE_URL="https://www.nytimes.com/2025/12/09/technology/..."

# 4. Analyze its sentiment
curl "http://localhost:8000/api/news/sentiment?url=$ARTICLE_URL"

# 5. Get similar articles
curl "http://localhost:8000/api/news/recommendations?url=$ARTICLE_URL"

# 6. Analyze sentiment of a news source
curl "http://localhost:8000/api/news/sentiment/batch?source=NYTimes%20Technology&limit=10"

# 7. Search for articles about AI
curl "http://localhost:8000/api/news/recommendations/search?query=artificial%20intelligence"
```

---

## Technical Details

### Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| nltk | >=3.8.0 | VADER sentiment analysis |
| scikit-learn | >=1.4.0 | TF-IDF vectorization, cosine similarity |
| numpy | >=1.26.0 | Numerical computations |

### Performance Notes

- **Sentiment Analysis**: ~1ms per article (very fast)
- **Building Index**: ~100ms for 50 articles (one-time)
- **Getting Recommendations**: ~5ms per query (fast)

### Limitations

**Sentiment (VADER):**
- English only
- May misinterpret sarcasm
- Domain-specific terms might not be in lexicon

**Recommendations (TF-IDF):**
- Requires rebuilding index when new articles are added
- Cold start problem (needs articles in cache first)
- Only considers text similarity, not user preferences

---

## For Your FYP Report

### Key Points to Mention

1. **VADER** is chosen because it's specifically designed for social media/news text (Hutto & Gilbert, 2014)

2. **TF-IDF** is a classic information retrieval technique that remains effective for content-based recommendations

3. **Cosine Similarity** is used because it's magnitude-independent (article length doesn't affect similarity)

4. Both approaches are **unsupervised** - no training data needed

5. The system is **modular** - can easily swap in more advanced models (BERT, transformers) later

### References

- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. ICWSM.
- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management.
