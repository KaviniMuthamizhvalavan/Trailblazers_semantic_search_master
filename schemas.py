"""
schemas.py — Pydantic request/response models for the Semantic Product Search API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ProductBase(BaseModel):
    """Core product fields returned in all responses."""
    product_id: str
    title: str
    description: str
    category: str
    price: float
    rating: float
    stock: int
    attributes: str


class SearchResult(ProductBase):
    """A single search result with similarity and re-ranked scores."""
    similarity_score: float = Field(
        ..., description="Raw cosine similarity score from FAISS"
    )
    final_score: float = Field(
        ..., description="Re-ranked score combining similarity, rating, and stock"
    )


class SearchResponse(BaseModel):
    """Response envelope for search queries."""
    query: str
    top_n: int
    results: List[SearchResult]
    total_products_in_index: int


class SimilarProductResult(ProductBase):
    """A single similar-product recommendation with scores."""
    similarity_score: float
    final_score: float


class RecommendationResponse(BaseModel):
    """Response envelope for similar-product recommendations."""
    source_product: ProductBase
    top_n: int
    similar_products: List[SimilarProductResult]


class ProductListResponse(BaseModel):
    """Response for listing all products."""
    total: int
    products: List[ProductBase]


class IndexStatusResponse(BaseModel):
    """Response for index health/status checks."""
    total_products: int
    index_dimension: int
    index_size: int
    index_file_exists: bool
    ids_file_exists: bool
