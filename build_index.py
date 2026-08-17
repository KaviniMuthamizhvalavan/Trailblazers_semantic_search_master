"""
build_index.py — Script to build (or rebuild) the FAISS index from the product catalog.

Per planner §5: Provide a script/flag to rebuild the index so a grader can
regenerate from scratch to confirm it's not hardcoded.

Usage:
    python build_index.py
"""

import logging
import sys

from catalog import load_catalog
from embeddings import generate_embeddings
from index import build_index, save_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Build the FAISS index from the raw product catalog."""
    logger.info("=" * 60)
    logger.info("Building product search index")
    logger.info("=" * 60)

    # 1. Load and clean catalog
    products = load_catalog()
    if not products:
        logger.error("No products loaded. Cannot build index.")
        sys.exit(1)

    # 2. Extract texts and IDs in order
    texts = [p["combined_text"] for p in products]
    product_ids = [p["product_id"] for p in products]

    # 3. Generate embeddings
    embeddings = generate_embeddings(texts)

    # 4. Build FAISS index
    index = build_index(embeddings, product_ids)

    # 5. Save to disk
    save_index(index, product_ids)

    # 6. Self-similarity sanity check (per planner §5)
    logger.info("=" * 60)
    logger.info("Running self-similarity sanity check...")
    logger.info("=" * 60)

    import numpy as np
    passed = True
    check_indices = [0, len(products) // 2, len(products) - 1]

    for ci in check_indices:
        query_vec = embeddings[ci:ci + 1]
        distances, indices = index.search(query_vec, 1)
        top_idx = indices[0][0]
        top_dist = distances[0][0]

        expected_id = product_ids[ci]
        actual_id = product_ids[top_idx]

        if actual_id != expected_id or top_dist < 0.99:
            logger.error(
                f"FAIL: Product {expected_id} — nearest neighbor is "
                f"{actual_id} (similarity={top_dist:.4f})"
            )
            passed = False
        else:
            logger.info(
                f"PASS: Product {expected_id} — self-similarity={top_dist:.4f}"
            )

    if passed:
        logger.info("✓ All self-similarity checks passed!")
    else:
        logger.error("✗ Some self-similarity checks failed. Index may be broken.")
        sys.exit(1)

    logger.info(f"\nIndex built successfully with {index.ntotal} products.")
    logger.info("Files saved:")
    logger.info(f"  - data/product_index.faiss")
    logger.info(f"  - data/product_ids.json")


if __name__ == "__main__":
    main()
