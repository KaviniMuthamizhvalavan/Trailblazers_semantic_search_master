"""
index.py — FAISS index build, save, and load.

Per planner §5: IndexFlatIP on normalized vectors = cosine similarity.
Persists the index + a parallel product-ID mapping to disk.
"""

import json
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
INDEX_PATH = DATA_DIR / "product_index.faiss"
IDS_PATH = DATA_DIR / "product_ids.json"


def build_index(embeddings: np.ndarray, product_ids: List[str]) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP from normalized embeddings.

    IndexFlatIP on L2-normalized vectors computes cosine similarity.

    Args:
        embeddings: (N, D) array of normalized embeddings.
        product_ids: List of product IDs in the same order as embeddings.

    Returns:
        The built FAISS index.
    """
    if len(embeddings) != len(product_ids):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings vs {len(product_ids)} product IDs"
        )

    dimension = embeddings.shape[1]
    logger.info(f"Building FAISS IndexFlatIP: {len(embeddings)} vectors, dim={dimension}")

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    logger.info(f"Index built. Total vectors: {index.ntotal}")
    return index


def save_index(index: faiss.IndexFlatIP, product_ids: List[str]) -> None:
    """Save the FAISS index and product ID mapping to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))
    logger.info(f"FAISS index saved to {INDEX_PATH}")

    with open(IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(product_ids, f, indent=2)
    logger.info(f"Product IDs saved to {IDS_PATH}")


def load_index() -> Tuple[faiss.IndexFlatIP, List[str]]:
    """
    Load a previously saved FAISS index and product ID mapping.

    Returns:
        Tuple of (FAISS index, list of product IDs).

    Raises:
        FileNotFoundError: If index or IDs file doesn't exist.
    """
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
    if not IDS_PATH.exists():
        raise FileNotFoundError(f"Product IDs file not found: {IDS_PATH}")

    index = faiss.read_index(str(INDEX_PATH))
    logger.info(f"FAISS index loaded: {index.ntotal} vectors, dim={index.d}")

    with open(IDS_PATH, "r", encoding="utf-8") as f:
        product_ids = json.load(f)
    logger.info(f"Product IDs loaded: {len(product_ids)} IDs")

    if index.ntotal != len(product_ids):
        raise ValueError(
            f"Index/ID mismatch: {index.ntotal} vectors vs {len(product_ids)} IDs"
        )

    return index, product_ids


def search_index(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    top_n: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index for nearest neighbors.

    Args:
        index: The FAISS index.
        query_embedding: (1, D) query vector.
        top_n: Number of results to return.

    Returns:
        Tuple of (distances array, indices array), each shape (1, top_n).
    """
    # Clamp top_n to index size
    top_n = min(top_n, index.ntotal)
    distances, indices = index.search(query_embedding, top_n)
    return distances, indices


def index_exists() -> bool:
    """Check if a saved index exists on disk."""
    return INDEX_PATH.exists() and IDS_PATH.exists()
