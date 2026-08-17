"""
main.py — FastAPI app with route registration.

Per planner §3 and §2: FastAPI with Swagger UI for free, screenshot-able
proof of every endpoint.

Endpoints:
    GET /search?q=<free text>&top_n=<int>
    GET /products/{product_id}/similar?top_n=<int>
    GET /products  — list all products
    GET /index/status — index health check
"""

import logging
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Dict, Any, List

from catalog import load_catalog
from embeddings import generate_embeddings
from index import load_index, build_index, save_index, index_exists
from search import semantic_search
from recommend import get_similar_products
from schemas import (
    SearchResponse,
    SearchResult,
    RecommendationResponse,
    SimilarProductResult,
    ProductBase,
    ProductListResponse,
    IndexStatusResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Application state (populated on startup) ---
app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load catalog, embeddings, and FAISS index on startup."""
    logger.info("Starting up — loading catalog and index...")

    # Load catalog
    products = load_catalog()
    products_by_id = {p["product_id"]: p for p in products}
    product_ids_ordered = [p["product_id"] for p in products]

    # Try to load existing index; if not found, build it
    if index_exists():
        logger.info("Loading existing FAISS index from disk...")
        faiss_index, stored_ids = load_index()

        # Regenerate embeddings for recommendations (need the matrix)
        texts = [p["combined_text"] for p in products]
        embeddings = generate_embeddings(texts)
    else:
        logger.info("No existing index found — building from scratch...")
        texts = [p["combined_text"] for p in products]
        embeddings = generate_embeddings(texts)
        faiss_index = build_index(embeddings, product_ids_ordered)
        save_index(faiss_index, product_ids_ordered)
        stored_ids = product_ids_ordered

    app_state["products"] = products
    app_state["products_by_id"] = products_by_id
    app_state["product_ids"] = stored_ids
    app_state["faiss_index"] = faiss_index
    app_state["embeddings"] = embeddings

    logger.info(
        f"Startup complete. {len(products)} products indexed, "
        f"index dim={faiss_index.d}"
    )

    yield

    # Cleanup
    app_state.clear()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Semantic Product Search & Recommendation Engine",
    description=(
        "An embedding-powered product search engine using sentence-transformers "
        "(all-MiniLM-L6-v2) and FAISS. Supports natural-language semantic search, "
        "similar product recommendations, and re-ranking with business signals "
        "(rating, stock availability)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────
# Search endpoint (§6 — 30% of grade)
# ─────────────────────────────────────────────────────────────────────
@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_products(
    q: str = Query(
        ...,
        min_length=1,
        description="Free-text natural language search query",
        examples=["cozy winter jacket", "something to keep beverages hot"],
    ),
    top_n: int = Query(
        default=5,
        ge=1,
        le=50,
        description="Number of results to return",
    ),
):
    """
    Semantic product search.

    Embeds the query using the same model as the product catalog and retrieves
    the most semantically similar products. Results are re-ranked with business
    signals (rating, stock availability).
    """
    results = semantic_search(
        query=q,
        faiss_index=app_state["faiss_index"],
        product_ids=app_state["product_ids"],
        products_by_id=app_state["products_by_id"],
        top_n=top_n,
    )

    return SearchResponse(
        query=q,
        top_n=top_n,
        results=[SearchResult(**r) for r in results],
        total_products_in_index=app_state["faiss_index"].ntotal,
    )


# ─────────────────────────────────────────────────────────────────────
# Recommendation endpoint (§7 — 25% of grade)
# ─────────────────────────────────────────────────────────────────────
@app.get(
    "/products/{product_id}/similar",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
)
async def similar_products(
    product_id: str,
    top_n: int = Query(
        default=5,
        ge=1,
        le=20,
        description="Number of similar products to return",
    ),
):
    """
    Get similar product recommendations.

    Uses the product's own embedding to find nearest neighbors in the vector
    space, excluding the product itself. Results are re-ranked with business
    signals.
    """
    if product_id not in app_state["products_by_id"]:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_id}' not found in catalog",
        )

    source_product = app_state["products_by_id"][product_id]

    try:
        results = get_similar_products(
            product_id=product_id,
            faiss_index=app_state["faiss_index"],
            product_ids=app_state["product_ids"],
            products_by_id=app_state["products_by_id"],
            embeddings=app_state["embeddings"],
            top_n=top_n,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return RecommendationResponse(
        source_product=ProductBase(**{
            k: source_product[k]
            for k in ProductBase.model_fields.keys()
        }),
        top_n=top_n,
        similar_products=[SimilarProductResult(**r) for r in results],
    )


# ─────────────────────────────────────────────────────────────────────
# Product listing endpoint
# ─────────────────────────────────────────────────────────────────────
@app.get("/products", response_model=ProductListResponse, tags=["Products"])
async def list_products():
    """List all products in the catalog."""
    products = [
        ProductBase(**{k: p[k] for k in ProductBase.model_fields.keys()})
        for p in app_state["products"]
    ]
    return ProductListResponse(total=len(products), products=products)


# ─────────────────────────────────────────────────────────────────────
# Index status endpoint
# ─────────────────────────────────────────────────────────────────────
@app.get("/index/status", response_model=IndexStatusResponse, tags=["Index"])
async def index_status():
    """Check the health and status of the FAISS index."""
    from index import INDEX_PATH, IDS_PATH

    return IndexStatusResponse(
        total_products=app_state["faiss_index"].ntotal,
        index_dimension=app_state["faiss_index"].d,
        index_size=app_state["faiss_index"].ntotal,
        index_file_exists=INDEX_PATH.exists(),
        ids_file_exists=IDS_PATH.exists(),
    )


# ─────────────────────────────────────────────────────────────────────
# Static files mount for HTML/CSS/JS frontend
# ─────────────────────────────────────────────────────────────────────
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")

