import json
from pathlib import Path

from app.schemas import Product


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "products.json"


def get_products() -> list[Product]:
    with DATA_PATH.open(encoding="utf-8") as products_file:
        return [Product.model_validate(item) for item in json.load(products_file)]


def find_products(query: str = "") -> list[Product]:
    products = get_products()
    terms = query.lower().split()
    if not terms:
        return products
    return [
        product
        for product in products
        if all(
            term in " ".join(
                [product.name, product.brand, product.category, product.sku, product.description]
                + list(product.specifications.values())
            ).lower()
            for term in terms
        )
    ]


def product_document(product: Product) -> str:
    specifications = ", ".join(f"{key}: {value}" for key, value in product.specifications.items())
    return (
        f"Product: {product.name} (ID: {product.id}, SKU: {product.sku}). "
        f"Brand: {product.brand}. Category: {product.category}. Price: ${product.price:.2f} {product.currency}. "
        f"Rating: {product.rating}/5. Inventory: {product.inventory}. "
        f"Description: {product.description} Specifications: {specifications}"
    )