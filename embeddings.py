"""
embeddings.py — Embedding generation for the product catalog.

Uses sentence-transformers (all-MiniLM-L6-v2) per planner §2 and §5.
Generates normalized embeddings so IndexFlatIP = cosine similarity.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

from pathlib import Path

logger = logging.getLogger(__name__)

# Model configuration — saved locally to avoid repeated HuggingFace downloads
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_DIR = Path(__file__).parent / "data" / "models" / "all-MiniLM-L6-v2"

# Module-level cache for the model (loaded once)
_model: SentenceTransformer = None


def get_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model locally on disk."""
    global _model
    if _model is None:
        if MODEL_DIR.exists():
            logger.info(f"Loading cached sentence-transformer model from local path: {MODEL_DIR}")
            _model = SentenceTransformer(str(MODEL_DIR))
        else:
            logger.info(f"Downloading sentence-transformer model '{MODEL_NAME}' (first run)...")
            _model = SentenceTransformer(MODEL_NAME)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            _model.save(str(MODEL_DIR))
            logger.info(f"Model saved locally to {MODEL_DIR} for offline loading.")
            
        logger.info(f"Model loaded. Embedding dimension: {_model.get_embedding_dimension()}")
    return _model


def generate_embeddings(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Generate normalized embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.
        batch_size: Batch size for encoding.

    Returns:
        numpy array of shape (len(texts), embedding_dim) with L2-normalized vectors.
    """
    model = get_model()

    logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})")

    # normalize_embeddings=True so we can use IndexFlatIP for cosine similarity
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = np.array(embeddings, dtype=np.float32)
    logger.info(f"Embeddings generated: shape={embeddings.shape}")

    return embeddings


def generate_query_embedding(query: str) -> np.ndarray:
    """
    Generate a normalized embedding for a single search query.

    Args:
        query: The search query string.

    Returns:
        numpy array of shape (1, embedding_dim).
    """
    model = get_model()
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.array(embedding, dtype=np.float32)
