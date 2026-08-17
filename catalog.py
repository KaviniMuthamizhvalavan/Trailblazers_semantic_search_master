"""
catalog.py — Loading + cleaning/normalizing raw product data.

Feeds: Embedding & Indexing Setup (20% of grade).

Combines title + description + attributes into one text field per product
before embedding — per planner §4.
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Path to the raw catalog data
DATA_DIR = Path(__file__).parent / "data"
RAW_CATALOG_PATH = DATA_DIR / "raw_catalog.json"


def _clean_text(text: str) -> str:
    """Lowercase, strip HTML/special chars, collapse whitespace."""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^\w\s.,;:!?'-]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _combine_text_fields(product: Dict[str, Any]) -> str:
    """
    Combine title + description + attributes into a single text field.

    Per planner §4: embedding only the title loses too much signal;
    embedding fields separately adds needless complexity.
    """
    title = product.get("title", "")
    description = product.get("description", "")
    attributes = product.get("attributes", "")
    category = product.get("category", "")

    combined = f"{title}. {description} Category: {category}. {attributes}"
    return _clean_text(combined)


def load_catalog(catalog_path: str = None) -> List[Dict[str, Any]]:
    """
    Load, clean, deduplicate, and enrich the product catalog.

    Returns a list of product dicts, each with an added 'combined_text' field
    ready for embedding.

    Args:
        catalog_path: Optional override for the catalog file path.

    Returns:
        List of cleaned product dictionaries.
    """
    path = Path(catalog_path) if catalog_path else RAW_CATALOG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Catalog file not found: {path}")

    logger.info(f"Loading catalog from {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_products = json.load(f)

    logger.info(f"Raw products loaded: {len(raw_products)}")

    # Deduplicate on product_id
    seen_ids = set()
    products = []
    duplicates = 0

    for product in raw_products:
        pid = product.get("product_id")
        if not pid:
            logger.warning("Skipping product with no product_id")
            continue
        if pid in seen_ids:
            duplicates += 1
            logger.warning(f"Duplicate product_id skipped: {pid}")
            continue
        seen_ids.add(pid)

        # Normalize fields
        product["title"] = _clean_text(product.get("title", ""))
        product["description"] = _clean_text(product.get("description", ""))
        product["attributes"] = _clean_text(product.get("attributes", ""))
        product["category"] = product.get("category", "Unknown").strip()
        product["price"] = float(product.get("price", 0.0))
        product["rating"] = float(product.get("rating", 0.0))
        product["stock"] = int(product.get("stock", 0))

        # Build combined text for embedding
        product["combined_text"] = _combine_text_fields(product)

        if not product["combined_text"]:
            logger.warning(f"Product {pid} has empty combined text, skipping")
            continue

        products.append(product)

    dropped = len(raw_products) - len(products)
    logger.info(
        f"Catalog loaded: {len(products)} products "
        f"({duplicates} duplicates removed, {dropped} total dropped)"
    )

    return products


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    catalog = load_catalog()
    print(f"\nLoaded {len(catalog)} products")
    for p in catalog[:3]:
        print(f"  [{p['product_id']}] {p['title']}")
        print(f"    combined_text: {p['combined_text'][:100]}...")
