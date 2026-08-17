"""
test_search.py — Tests for semantic search functionality.

Per planner §6 and §9: tests all 4 query types.
"""

import pytest
import numpy as np

from catalog import load_catalog
from embeddings import generate_embeddings
from index import build_index
from search import semantic_search


@pytest.fixture(scope="module")
def search_env():
    """Set up the full search environment once for all tests in this module."""
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
        "products": products,
    }


class TestSemanticSearch:
    """Tests for the semantic search pipeline."""

    def test_non_keyword_matching_query(self, search_env):
        """
        §6 core test: "cozy winter jacket" should return insulated/warm outerwear
        even though none of those exact words appear in product titles.
        This is THE semantic search test.
        """
        results = semantic_search(
            query="cozy winter jacket",
            faiss_index=search_env["faiss_index"],
            product_ids=search_env["product_ids"],
            products_by_id=search_env["products_by_id"],
            top_n=5,
        )

        assert len(results) > 0, "Should return results"

        # At least one result should be from the Outerwear category
        categories = [r["category"] for r in results]
        assert "Outerwear" in categories, (
            f"'cozy winter jacket' should return Outerwear products. "
            f"Got categories: {categories}"
        )

        # The top result should be outerwear-related
        top_result = results[0]
        assert top_result["category"] == "Outerwear", (
            f"Top result should be Outerwear, got: {top_result['title']} "
            f"({top_result['category']})"
        )

    def test_paraphrase_query(self, search_env):
        """
        §6: Near-duplicate/paraphrase query.
        "portable music speaker" should return the Bluetooth speaker.
        """
        results = semantic_search(
            query="portable music speaker",
            faiss_index=search_env["faiss_index"],
            product_ids=search_env["product_ids"],
            products_by_id=search_env["products_by_id"],
            top_n=5,
        )

        assert len(results) > 0
        # Should find the Bluetooth speaker
        top_ids = [r["product_id"] for r in results[:3]]
        assert "ELEC-003" in top_ids or any(
            "speaker" in r["title"] for r in results[:3]
        ), f"Should find the portable speaker. Top results: {[r['title'] for r in results[:3]]}"

    def test_vague_broad_query(self, search_env):
        """
        §6: Vague/broad query should return a spread across a category.
        "something for exercising" should return Sports products.
        """
        results = semantic_search(
            query="something for exercising",
            faiss_index=search_env["faiss_index"],
            product_ids=search_env["product_ids"],
            products_by_id=search_env["products_by_id"],
            top_n=5,
        )

        assert len(results) > 0
        # Should include some sports/fitness products
        categories = [r["category"] for r in results]
        assert "Sports & Outdoors" in categories, (
            f"'something for exercising' should include Sports products. "
            f"Got: {categories}"
        )

    def test_no_good_match_query(self, search_env):
        """
        §6: Query with no good match should still return closest items
        and not crash.
        """
        results = semantic_search(
            query="quantum mechanical flux capacitor for time travel",
            faiss_index=search_env["faiss_index"],
            product_ids=search_env["product_ids"],
            products_by_id=search_env["products_by_id"],
            top_n=5,
        )

        # Should not crash and should return results (closest available)
        assert len(results) > 0, "Should return closest matches, not crash"
        # Similarity scores should be lower than for good matches
        assert all(
            r["similarity_score"] < 0.9 for r in results
        ), "Scores for unrelated query should be low"

    def test_top_n_respected(self, search_env):
        """Verify that the top_n parameter is respected."""
        for n in [1, 3, 10]:
            results = semantic_search(
                query="laptop computer",
                faiss_index=search_env["faiss_index"],
                product_ids=search_env["product_ids"],
                products_by_id=search_env["products_by_id"],
                top_n=n,
            )
            assert len(results) == n, f"Requested {n} results, got {len(results)}"

    def test_results_have_required_fields(self, search_env):
        """Verify that each result has all required fields."""
        results = semantic_search(
            query="kitchen appliance",
            faiss_index=search_env["faiss_index"],
            product_ids=search_env["product_ids"],
            products_by_id=search_env["products_by_id"],
            top_n=3,
        )

        required_fields = [
            "product_id", "title", "description", "category",
            "price", "rating", "stock", "attributes",
            "similarity_score", "final_score",
        ]
        for result in results:
            for field in required_fields:
                assert field in result, f"Missing field: {field}"

    def test_results_sorted_by_final_score(self, search_env):
        """Verify results are sorted by final_score descending."""
        results = semantic_search(
            query="outdoor camping gear",
            faiss_index=search_env["faiss_index"],
            product_ids=search_env["product_ids"],
            products_by_id=search_env["products_by_id"],
            top_n=10,
        )

        for i in range(len(results) - 1):
            assert results[i]["final_score"] >= results[i + 1]["final_score"], (
                f"Results not sorted: {results[i]['final_score']} < "
                f"{results[i + 1]['final_score']}"
            )

    def test_keeping_beverages_hot(self, search_env):
        """
        Another semantic gap test: 'something to keep beverages hot'
        should find the insulated water bottle or thermal carafe.
        """
        results = semantic_search(
            query="something to keep beverages hot",
            faiss_index=search_env["faiss_index"],
            product_ids=search_env["product_ids"],
            products_by_id=search_env["products_by_id"],
            top_n=5,
        )

        assert len(results) > 0
        # Should include insulated bottle or coffee maker
        top_titles = [r["title"] for r in results[:3]]
        found_relevant = any(
            any(word in title for word in ["insulated", "coffee", "thermal", "carafe"])
            for title in top_titles
        )
        assert found_relevant, (
            f"Should find insulated/thermal products. Got: {top_titles}"
        )
