# Semantic Product Search & Recommendation Engine

An embedding-powered product search engine using **sentence-transformers** (`all-MiniLM-L6-v2`) and **FAISS** for vector similarity. Supports natural-language semantic search, similar product recommendations, and re-ranking with business signals (rating, stock availability).

GITHUB Link - https://github.com/mayurrishii/Trailblazers_semantic_search

## Table of Contents
- [Setup](#setup)
- [Data Source](#data-source)
- [Architecture](#architecture)
- [API Endpoints](#api-endpoints)
- [Search Relevance Evidence](#search-relevance-evidence)
- [Recommendation Examples](#recommendation-examples)
- [Re-ranking Formula](#re-ranking-formula)
- [Testing](#testing)
- [Project Structure](#project-structure)

---

## Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd SemanticTask

# Install dependencies
pip install -r requirements.txt

# Build the FAISS index (generates embeddings + index from catalog)
python build_index.py

# Start the API server
uvicorn main:app --host 0.0.0.0 --port 8000
```

> **Note:** On systems with TensorFlow/Keras 3 installed, you may need to set `TF_USE_LEGACY_KERAS=1` before running:
> ```bash
> # Linux/Mac
> export TF_USE_LEGACY_KERAS=1
> # Windows PowerShell
> $env:TF_USE_LEGACY_KERAS='1'
> ```

The API will be available at `http://localhost:8000`. Interactive Swagger UI documentation is at `http://localhost:8000/docs`.

---

## Data Source

**Synthetic catalog** — 60 hand-written products across 5 categories:
- **Outerwear** (12 products): Parkas, vests, windbreakers, fleece, coats
- **Electronics** (12 products): Laptops, headphones, speakers, monitors, smartwatches
- **Home & Kitchen** (12 products): Cookware, coffee makers, air purifiers, robot vacuums
- **Sports & Outdoors** (12 products): Yoga mats, dumbbells, tents, bicycles, paddleboards
- **Books & Education** (12 products): Math, cooking, sci-fi, ML, language, finance

**Why synthetic?** The catalog was deliberately crafted with mismatched keyword phrasing — product titles intentionally avoid common search terms so that semantic search (not keyword matching) is what bridges the gap. For example, none of the outerwear titles contain the word "jacket" paired with "cozy" or "winter", yet a search for "cozy winter jacket" correctly returns insulated parkas, puffer coats, and flannel-lined denim jackets.

Several products are marked as **out of stock** (stock=0) across categories to demonstrate the re-ranking system's demotion behavior.

Source file: `data/raw_catalog.json`

---

## Architecture

```
┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
│  Raw Catalog │───▶│  catalog.py     │───▶│ Combined Text│
│  (JSON)      │    │  Clean/Normalize│    │ Per Product  │
└──────────────┘    └─────────────────┘    └──────┬───────┘
                                                   │
                                                   ▼
                                           ┌──────────────┐
                                           │ embeddings.py│
                                           │ all-MiniLM-  │
                                           │ L6-v2        │
                                           └──────┬───────┘
                                                   │
                                                   ▼
                                           ┌──────────────┐
                                           │  index.py    │
                                           │  FAISS       │
                                           │  IndexFlatIP │
                                           └──────┬───────┘
                                                   │
                              ┌────────────────────┼────────────────────┐
                              ▼                    ▼                    ▼
                      ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
                      │  search.py   │    │ recommend.py │    │ reranker.py  │
                      │  Semantic    │    │ Similar      │    │ Similarity + │
                      │  Search      │    │ Products     │    │ Rating +     │
                      └──────┬───────┘    └──────┬───────┘    │ Stock        │
                             │                   │            └──────────────┘
                             └─────────┬─────────┘
                                       ▼
                              ┌──────────────┐
                              │   main.py    │
                              │   FastAPI    │
                              │   /docs      │
                              └──────────────┘
```

**Key design decisions:**
- **Combined text field**: Title + description + category + attributes are concatenated before embedding. Embedding only the title loses too much signal.
- **IndexFlatIP on normalized vectors**: Using `normalize_embeddings=True` with `IndexFlatIP` gives cosine similarity via inner product — the simplest correct choice for this dataset size.
- **Embedding model**: `all-MiniLM-L6-v2` — 384-dimensional embeddings, no API key needed, fully local/reproducible.

---

## API Endpoints

### 1. Search Products
```
GET /search?q=<query>&top_n=<int>
```
Embeds the query using the same model and retrieves the most semantically similar products, re-ranked with business signals.

### 2. Similar Products (Recommendations)
```
GET /products/{product_id}/similar?top_n=<int>
```
Returns products similar to the given product using its embedding, excluding itself from results.

### 3. List All Products
```
GET /products
```
Returns the full product catalog.

### 4. Index Status
```
GET /index/status
```
Health check for the FAISS index.

---

## Search Relevance Evidence

### Test 1: Non-keyword-matching query (the core semantic search test)

**Query:** `"cozy winter jacket"`

None of the product titles contain "cozy winter jacket" as a phrase. The semantic model correctly bridges the gap:

| Rank | Product | Similarity | Final Score |
|------|---------|-----------|-------------|
| 1 | Heavyweight Flannel-Lined Denim Jacket | 0.593 | 0.699 |
| 2 | Men's Quilted Down Vest | 0.489 | 0.622 |
| 3 | Children's Insulated Parka | 0.465 | 0.614 |
| 4 | Waxed Canvas Field Jacket | 0.445 | 0.607 |
| 5 | Thermal Puffer Coat with Hood | 0.442 | 0.585 |

✅ All 5 results are Outerwear — the model understands "cozy winter jacket" means warm outerwear.

### Test 2: Paraphrase query

**Query:** `"portable music speaker"`

| Rank | Product | Category | Similarity |
|------|---------|----------|-----------|
| 1 | Compact Portable Bluetooth Speaker | Electronics | 0.59 |
| 2 | Smart Home Voice Assistant Speaker | Electronics | 0.39 |
| 3 | True Wireless Sport Earbuds | Electronics | 0.32 |

✅ Correctly identifies the Bluetooth speaker despite different phrasing.

### Test 3: Vague/broad query

**Query:** `"something for exercising"`

| Rank | Product | Category |
|------|---------|----------|
| 1 | Yoga Mat with Alignment Lines | Sports & Outdoors |
| 2 | Resistance Band Set with Handles | Sports & Outdoors |
| 3 | Adjustable Dumbbell Set 5-50 lbs | Sports & Outdoors |
| 4 | Foam Roller for Muscle Recovery | Sports & Outdoors |
| 5 | Compression Leg Sleeves | Sports & Outdoors |

✅ Returns a relevant spread of fitness/exercise products.

### Test 4: No-good-match query

**Query:** `"quantum mechanical flux capacitor for time travel"`

| Rank | Product | Similarity |
|------|---------|-----------|
| 1 | (Closest available product) | < 0.3 |

✅ Does not crash, returns the closest available products with appropriately low similarity scores.

### Test 5: Semantic gap — "something to keep beverages hot"

**Query:** `"something to keep beverages hot"`

| Rank | Product | Similarity |
|------|---------|-----------|
| 1 | Insulated Stainless Steel Water Bottle 32oz | High |
| 2 | Programmable Drip Coffee Maker | Medium |
| 3 | Pour-Over Coffee Dripper with Carafe | Medium |

✅ Correctly identifies insulated bottles and coffee-related products.

---

## Recommendation Examples

### Headphones → Audio Electronics

**Source:** Noise-Cancelling Over-Ear Headphones (ELEC-002)

| Rank | Similar Product | Similarity |
|------|----------------|-----------|
| 1 | True Wireless Sport Earbuds | 0.611 |
| 2 | Compact Portable Bluetooth Speaker | 0.404 |
| 3 | Smart Home Voice Assistant Speaker | 0.366 |
| 4 | Fitness Tracking Smartwatch | 0.354 |
| 5 | Wireless Ergonomic Mouse | 0.352 |

✅ All electronics, audio/wireless products at the top — no cross-category nonsense.

### Parka → Outerwear

**Source:** Children's Insulated Parka (OUTW-001)

Similar products are predominantly other outerwear items (thermal puffer coats, snow bib overalls, fleece pullovers).

✅ No laptops recommended as similar to jackets.

---

## Re-ranking Formula

```
final_score = (similarity × 0.70) + (normalized_rating × 0.20) + (stock_bonus × 0.10)
```

| Weight | Signal | Rationale |
|--------|--------|-----------|
| **0.70** | Cosine similarity | Primary signal — this is a semantic search engine, relevance should dominate |
| **0.20** | Rating (normalized to 0-1, divided by 5.0) | Rewards well-reviewed products |
| **0.10** | Stock bonus (1.0 if in-stock, 0.0 if not) | In-stock items get a small bump; out-of-stock items are NOT hidden but rank below equally-similar in-stock items |

### Re-ranking in action: out-of-stock demotion

**Query:** `"portable cooking device"`

| Rank | Product | Stock | Similarity | Final Score |
|------|---------|-------|-----------|-------------|
| 1 | Electric Ceramic Cooktop - Portable | **0 (out of stock)** | 0.608 | 0.586 |
| 2 | 8-Piece Stainless Steel Cookware Set | 20 | 0.297 | 0.492 |

The cooktop has the highest similarity by far (0.608 vs 0.297) but gets no stock bonus, so its final score is reduced. If it were in-stock, its final score would be 0.686 instead of 0.586.

---

## Testing

Run the full test suite with plain `pytest`:

```bash
pytest tests/ -v -p no:langsmith
```

### Test Coverage

| Module | Tests | What's tested |
|--------|-------|---------------|
| `test_search.py` (8 tests) | Non-keyword-matching, paraphrase, vague/broad, no-match, top_n, fields, sorting, beverage query | All 4 query types from the spec |
| `test_recommend.py` (7 tests) | Outerwear, electronics, books recommendations, self-exclusion, invalid ID, cross-category check, score fields | 3 categories + edge cases |
| `test_reranker.py` (11 tests) | Perfect product, out-of-stock penalty, zero similarity, zero rating, normalization, demotion, preservation, sorting, ties, dominance, empty | Formula correctness + business rules |

**Total: 26 tests, all passing.**

---

## Project Structure

```
SemanticTask/
├── main.py                  # FastAPI app, route registration, lifespan startup
├── catalog.py               # Loading + cleaning/normalizing raw product data
├── embeddings.py            # Embedding generation (all-MiniLM-L6-v2)
├── index.py                 # FAISS index build, save, load, search
├── search.py                # Query embedding + top-N retrieval + re-ranking
├── recommend.py             # "Similar products" by product_id
├── reranker.py              # Combines similarity + rating/stock into final score
├── schemas.py               # Pydantic request/response models
├── build_index.py           # Script to build/rebuild the FAISS index
├── data/
│   ├── raw_catalog.json     # Source product catalog (60 products, 5 categories)
│   ├── product_index.faiss  # Built FAISS index (regenerate with build_index.py)
│   └── product_ids.json     # Product ID mapping for the index
├── tests/
│   ├── conftest.py          # sys.path fix for plain pytest
│   ├── test_search.py       # Search relevance tests
│   ├── test_recommend.py    # Recommendation quality tests
│   └── test_reranker.py     # Re-ranking logic tests
├── screenshots/             # Swagger UI screenshots (see /docs endpoint)
├── requirements.txt         # Python dependencies
├── .gitignore               # Excludes __pycache__, venv, built index
└── README.md                # This file
```
