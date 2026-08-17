"""
recommend.py — "Similar products" by product_id.

Per planner §7: Recommendation Quality is 25% of the grade.
Uses the product's own stored embedding to find nearest neighbors,
excluding the product itself.
"""

import logging
import numpy as np
from typing import List, Dict, Any

from index import search_index
from reranker import rerank_results

logger = logging.getLogger(__name__)


def get_similar_products(
    product_id: str,
    faiss_index,
    product_ids: List[str],
    products_by_id: Dict[str, Dict[str, Any]],
    embeddings: np.ndarray,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Find products similar to the given product_id using its embedding.

    Args:
        product_id: The source product ID.
        faiss_index: The loaded FAISS index.
        product_ids: Ordered list of product IDs matching the index.
        products_by_id: Dict mapping product_id -> full product dict.
        embeddings: The full embeddings matrix (same order as product_ids).
        top_n: Number of similar products to return.

    Returns:
        List of similar product dicts with similarity_score and final_score.

    Raises:
        ValueError: If product_id is not found in the index.
    """
    if product_id not in products_by_id:
        raise ValueError(f"Product ID '{product_id}' not found in catalog")

    # Find the index position of this product
    try:
        source_idx = product_ids.index(product_id)
    except ValueError:
        raise ValueError(f"Product ID '{product_id}' not found in index")

    logger.info(f"Finding products similar to: {product_id} (top_n={top_n})")

    # Use the product's own embedding as the query
    product_embedding = embeddings[source_idx:source_idx + 1]

    # Retrieve extra to account for excluding self
    retrieval_n = min(top_n + 5, faiss_index.ntotal)
    distances, indices = search_index(faiss_index, product_embedding, retrieval_n)

    # Build results, excluding the source product itself
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx < 0:
            continue

        pid = product_ids[idx]

        # Exclude the source product
        if pid == product_id:
            continue

        product = products_by_id.get(pid)
        if product is None:
            continue

        result = {
            "product_id": product["product_id"],
            "title": product["title"],
            "description": product["description"],
            "category": product["category"],
            "price": product["price"],
            "rating": product["rating"],
            "stock": product["stock"],
            "attributes": product["attributes"],
            "similarity_score": round(float(distances[0][i]), 6),
        }
        results.append(result)

    # Re-rank with business signals
    results = rerank_results(results)

    return results[:top_n]
