"""
reranker.py — Combines similarity + rating/stock into final score.

Per planner §8:
  final_score = (similarity * w_sim) + (normalized_rating * w_rating) + (stock_bonus if in_stock else 0)

Weights are named constants, not magic numbers.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# --- Re-ranking weights (named constants, per planner §8) ---
# Similarity is the primary signal (70% weight) because this is a semantic
# search engine — relevance should dominate the ranking.
W_SIMILARITY = 0.70

# Rating rewards well-reviewed products (20% weight). Normalized to 0-1
# by dividing by 5.0 (max rating).
W_RATING = 0.20

# Stock bonus provides a small bump (10% weight) to in-stock items.
# Out-of-stock products are NOT hidden — they just rank below equally-similar
# in-stock items, per planner §8.
W_STOCK_BONUS = 0.10

# Maximum possible rating (used for normalization)
MAX_RATING = 5.0


def compute_final_score(
    similarity: float,
    rating: float,
    stock: int,
) -> float:
    """
    Compute the re-ranked final score for a product.

    Formula:
        final_score = (similarity * W_SIMILARITY)
                    + (normalized_rating * W_RATING)
                    + (W_STOCK_BONUS if in_stock else 0)

    Args:
        similarity: Cosine similarity score (0 to 1 for normalized vectors).
        rating: Product rating (0 to 5).
        stock: Number of items in stock (0 = out of stock).

    Returns:
        The re-ranked final score.
    """
    normalized_rating = min(rating, MAX_RATING) / MAX_RATING
    stock_bonus = W_STOCK_BONUS if stock > 0 else 0.0

    final_score = (
        (similarity * W_SIMILARITY)
        + (normalized_rating * W_RATING)
        + stock_bonus
    )

    return round(final_score, 6)


def rerank_results(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Apply re-ranking to a list of search/recommendation results.

    Each result dict must have 'similarity_score', 'rating', and 'stock' keys.
    Adds a 'final_score' key and re-sorts by it (descending).

    Args:
        results: List of result dicts with similarity_score, rating, stock.

    Returns:
        The same list, sorted by final_score descending, with final_score added.
    """
    for result in results:
        result["final_score"] = compute_final_score(
            similarity=result["similarity_score"],
            rating=result["rating"],
            stock=result["stock"],
        )

    # Sort by final_score descending
    results.sort(key=lambda x: x["final_score"], reverse=True)

    logger.debug(
        f"Re-ranked {len(results)} results. "
        f"Top score: {results[0]['final_score'] if results else 'N/A'}"
    )

    return results
