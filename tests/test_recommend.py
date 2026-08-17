"""
test_recommend.py — Tests for similar product recommendations.

Per planner §7 and §9: tested across 2-3 different categories,
results manually judged sensible.
"""

import pytest
import numpy as np

from catalog import load_catalog
from embeddings import generate_embeddings
from index import build_index
from recommend import get_similar_products


@pytest.fixture(scope="module")
def recommend_env():
    """Set up the full recommendation environment."""
    products = load_catalog()
    texts = [p["combined_text"] for p in products]
    product_ids = [p["product_id"] for p in products]
    embeddings = generate_embeddings(texts)
    faiss_index = build_index(embeddings, product_ids)
    products_by_id = {p["product_id"]: p for p in products}

    return {
        "faiss_index": faiss_index,
        "product_ids": product_ids,
        "products_by_id": products_by_id,
        "embeddings": embeddings,
    }


class TestRecommendations:
    """Tests for the similar products recommendation system."""

    def test_outerwear_recommendations(self, recommend_env):
        """
        §7: Similar products for an outerwear item should be mostly outerwear.
        Using OUTW-001 (Children's Insulated Parka).
        """
        results = get_similar_products(
            product_id="OUTW-001",
            faiss_index=recommend_env["faiss_index"],
            product_ids=recommend_env["product_ids"],
            products_by_id=recommend_env["products_by_id"],
            embeddings=recommend_env["embeddings"],
            top_n=5,
        )

        assert len(results) > 0
        # Source product should NOT be in results
        result_ids = [r["product_id"] for r in results]
        assert "OUTW-001" not in result_ids, "Source product should be excluded"

        # Most results should be outerwear
        outerwear_count = sum(
            1 for r in results if r["category"] == "Outerwear"
        )
        assert outerwear_count >= 3, (
            f"Expected at least 3 outerwear results for a parka, got {outerwear_count}. "
            f"Results: {[(r['title'], r['category']) for r in results]}"
        )

    def test_electronics_recommendations(self, recommend_env):
        """
        §7: Similar products for electronics should be mostly electronics.
        Using ELEC-002 (Noise-Cancelling Headphones).
        """
        results = get_similar_products(
            product_id="ELEC-002",
            faiss_index=recommend_env["faiss_index"],
            product_ids=recommend_env["product_ids"],
            products_by_id=recommend_env["products_by_id"],
            embeddings=recommend_env["embeddings"],
            top_n=5,
        )

        assert len(results) > 0
        result_ids = [r["product_id"] for r in results]
        assert "ELEC-002" not in result_ids

        # Audio-related electronics should appear
        electronics_count = sum(
            1 for r in results if r["category"] == "Electronics"
        )
        assert electronics_count >= 3, (
            f"Expected at least 3 electronics for headphones, got {electronics_count}. "
            f"Results: {[(r['title'], r['category']) for r in results]}"
        )

    def test_book_recommendations(self, recommend_env):
        """
        §7: Similar products for a book should be mostly books.
        Using BOOK-005 (Introduction to Machine Learning with Python).
        """
        results = get_similar_products(
            product_id="BOOK-005",
            faiss_index=recommend_env["faiss_index"],
            product_ids=recommend_env["product_ids"],
            products_by_id=recommend_env["products_by_id"],
            embeddings=recommend_env["embeddings"],
            top_n=5,
        )

        assert len(results) > 0
        result_ids = [r["product_id"] for r in results]
        assert "BOOK-005" not in result_ids

        # Should include other books
        book_count = sum(
            1 for r in results if r["category"] == "Books & Education"
        )
        assert book_count >= 2, (
            f"Expected at least 2 books for an ML book, got {book_count}. "
            f"Results: {[(r['title'], r['category']) for r in results]}"
        )

    def test_excludes_self(self, recommend_env):
        """The source product must never appear in its own recommendations."""
        for pid in ["OUTW-001", "ELEC-001", "HOME-001", "SPRT-001", "BOOK-001"]:
            results = get_similar_products(
                product_id=pid,
                faiss_index=recommend_env["faiss_index"],
                product_ids=recommend_env["product_ids"],
                products_by_id=recommend_env["products_by_id"],
                embeddings=recommend_env["embeddings"],
                top_n=5,
            )
            result_ids = [r["product_id"] for r in results]
            assert pid not in result_ids, f"Product {pid} found in its own recommendations"

    def test_invalid_product_id(self, recommend_env):
        """Should raise ValueError for a non-existent product ID."""
        with pytest.raises(ValueError, match="not found"):
            get_similar_products(
                product_id="NONEXISTENT-999",
                faiss_index=recommend_env["faiss_index"],
                product_ids=recommend_env["product_ids"],
                products_by_id=recommend_env["products_by_id"],
                embeddings=recommend_env["embeddings"],
                top_n=5,
            )

    def test_no_cross_category_nonsense(self, recommend_env):
        """
        §7: A laptop should NOT recommend a jacket. Verify no absurd
        cross-category recommendations at the top.
        """
        results = get_similar_products(
            product_id="ELEC-001",  # Notebook Computer
            faiss_index=recommend_env["faiss_index"],
            product_ids=recommend_env["product_ids"],
            products_by_id=recommend_env["products_by_id"],
            embeddings=recommend_env["embeddings"],
            top_n=3,
        )

        for r in results:
            assert r["category"] != "Outerwear", (
                f"Laptop should not recommend outerwear: {r['title']}"
            )

    def test_results_have_scores(self, recommend_env):
        """Verify results include both similarity and final scores."""
        results = get_similar_products(
            product_id="SPRT-001",
            faiss_index=recommend_env["faiss_index"],
            product_ids=recommend_env["product_ids"],
            products_by_id=recommend_env["products_by_id"],
            embeddings=recommend_env["embeddings"],
            top_n=3,
        )

        for r in results:
            assert "similarity_score" in r
            assert "final_score" in r
            assert 0 <= r["similarity_score"] <= 1
            assert r["final_score"] > 0
