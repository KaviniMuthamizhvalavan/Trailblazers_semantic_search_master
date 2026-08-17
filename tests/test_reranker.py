"""
test_reranker.py — Tests for the re-ranking logic.

Per planner §8 and §9: out-of-stock demotion test is critical.
"""

import pytest

from reranker import compute_final_score, rerank_results, W_SIMILARITY, W_RATING, W_STOCK_BONUS


class TestComputeFinalScore:
    """Tests for the individual score computation."""

    def test_perfect_product(self):
        """High similarity, perfect rating, in stock = highest score."""
        score = compute_final_score(similarity=0.95, rating=5.0, stock=100)
        assert score > 0.9, f"Perfect product should score > 0.9, got {score}"

    def test_out_of_stock_penalty(self):
        """Out-of-stock product should score lower than identical in-stock product."""
        in_stock_score = compute_final_score(similarity=0.8, rating=4.5, stock=50)
        out_of_stock_score = compute_final_score(similarity=0.8, rating=4.5, stock=0)

        assert in_stock_score > out_of_stock_score, (
            f"In-stock ({in_stock_score}) should beat out-of-stock ({out_of_stock_score})"
        )
        assert in_stock_score - out_of_stock_score == pytest.approx(
            W_STOCK_BONUS, abs=1e-6
        ), "Difference should equal the stock bonus weight"

    def test_zero_similarity(self):
        """Zero similarity should still get rating and stock components."""
        score = compute_final_score(similarity=0.0, rating=5.0, stock=10)
        expected = (0.0 * W_SIMILARITY) + (1.0 * W_RATING) + W_STOCK_BONUS
        assert score == pytest.approx(expected, abs=1e-6)

    def test_zero_rating(self):
        """Zero rating should still get similarity and stock components."""
        score = compute_final_score(similarity=0.9, rating=0.0, stock=10)
        expected = (0.9 * W_SIMILARITY) + (0.0 * W_RATING) + W_STOCK_BONUS
        assert score == pytest.approx(expected, abs=1e-6)

    def test_rating_normalization(self):
        """Rating should be normalized to 0-1 range (divided by 5.0)."""
        score_half = compute_final_score(similarity=0.5, rating=2.5, stock=10)
        score_full = compute_final_score(similarity=0.5, rating=5.0, stock=10)

        # Rating 2.5 normalized = 0.5, rating 5.0 normalized = 1.0
        expected_diff = (1.0 - 0.5) * W_RATING
        actual_diff = score_full - score_half
        assert actual_diff == pytest.approx(expected_diff, abs=1e-6)


class TestRerankResults:
    """Tests for the full re-ranking pipeline."""

    def test_out_of_stock_demotion(self):
        """
        §8 critical test: A high-similarity out-of-stock product should rank
        below an equally-similar in-stock product.
        """
        results = [
            {
                "product_id": "A",
                "title": "Out of Stock Product",
                "similarity_score": 0.85,
                "rating": 4.5,
                "stock": 0,  # OUT OF STOCK
            },
            {
                "product_id": "B",
                "title": "In Stock Product",
                "similarity_score": 0.85,
                "rating": 4.5,
                "stock": 25,  # IN STOCK
            },
        ]

        reranked = rerank_results(results)

        assert reranked[0]["product_id"] == "B", (
            "In-stock product should rank above out-of-stock with same similarity"
        )
        assert reranked[1]["product_id"] == "A"

    def test_reranking_preserves_all_results(self):
        """Re-ranking should not drop any results."""
        results = [
            {"product_id": f"P{i}", "title": f"Product {i}",
             "similarity_score": 0.5 + i * 0.1, "rating": 3.0 + i * 0.5,
             "stock": i * 10}
            for i in range(5)
        ]

        reranked = rerank_results(results)
        assert len(reranked) == 5

    def test_sorted_by_final_score_descending(self):
        """Results must be sorted by final_score in descending order."""
        results = [
            {"product_id": "LOW", "title": "Low",
             "similarity_score": 0.3, "rating": 2.0, "stock": 5},
            {"product_id": "HIGH", "title": "High",
             "similarity_score": 0.95, "rating": 4.8, "stock": 50},
            {"product_id": "MED", "title": "Medium",
             "similarity_score": 0.7, "rating": 3.5, "stock": 20},
        ]

        reranked = rerank_results(results)
        for i in range(len(reranked) - 1):
            assert reranked[i]["final_score"] >= reranked[i + 1]["final_score"]

    def test_higher_rating_wins_on_tie(self):
        """When similarity and stock are equal, higher rating should win."""
        results = [
            {"product_id": "A", "title": "A",
             "similarity_score": 0.8, "rating": 3.0, "stock": 10},
            {"product_id": "B", "title": "B",
             "similarity_score": 0.8, "rating": 4.5, "stock": 10},
        ]

        reranked = rerank_results(results)
        assert reranked[0]["product_id"] == "B"

    def test_similarity_dominates(self):
        """
        A much higher similarity should beat a slightly better rating,
        since W_SIMILARITY (0.70) >> W_RATING (0.20).
        """
        results = [
            {"product_id": "HIGH_SIM", "title": "A",
             "similarity_score": 0.95, "rating": 3.0, "stock": 10},
            {"product_id": "HIGH_RAT", "title": "B",
             "similarity_score": 0.40, "rating": 5.0, "stock": 10},
        ]

        reranked = rerank_results(results)
        assert reranked[0]["product_id"] == "HIGH_SIM", (
            "Similarity should dominate over rating"
        )

    def test_empty_results(self):
        """Re-ranking an empty list should not crash."""
        results = rerank_results([])
        assert results == []
