from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from app.config import get_settings
from app.schemas import (
    ChatRequest,
    ChatResponse,
    CompareRequest,
    IngestRequest,
    Product,
    RecommendationRequest,
    ReviewSummaryResponse,
    Source,
)
from app.services.assistant import answer_question, summarize_reviews
from app.services.catalog import find_products, get_products
from app.services.knowledge_base import knowledge_base
from app.services.orders import track_order


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    if settings.vector_store not in {"memory", "chroma", "pgvector"}:
        raise RuntimeError("VECTOR_STORE must be one of: memory, chroma, pgvector.")
    yield


app = FastAPI(
    title="ShopAssist AI",
    description="Grounded e-commerce assistance for products, policies, recommendations, and orders.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str | bool]:
    settings = get_settings()
    return {
        "status": "ok",
        "llm_configured": bool(settings.openai_api_key),
        "vector_store": settings.vector_store,
        "mongodb_configured": bool(settings.mongodb_uri),
    }


@app.get("/api/products", response_model=list[Product])
def list_products(q: str = Query(default="", max_length=200)) -> list[Product]:
    return find_products(q)


@app.get("/api/products/{product_id}/review-summary", response_model=ReviewSummaryResponse)
def product_review_summary(product_id: str) -> ReviewSummaryResponse:
    product = next((item for item in get_products() if item.id == product_id), None)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found.")

    review_documents = [
        document
        for document in knowledge_base.search(f"{product.name} {product.id} customer review", limit=10)
        if document.metadata.get("type") == "customer_review"
        and document.metadata.get("product_id") == product.id
    ]
    if not review_documents:
        raise HTTPException(status_code=404, detail="No customer reviews found for this product.")

    try:
        summary, ai_generated = summarize_reviews(product.name, review_documents)
    except Exception as error:
        raise HTTPException(status_code=503, detail="Review summarization is temporarily unavailable.") from error

    return ReviewSummaryResponse(
        product_id=product.id,
        summary=summary,
        ai_generated=ai_generated,
        sources=[Source(content=document.page_content, metadata=document.metadata) for document in review_documents],
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = answer_question(request.message)
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    sources = [
        Source(content=document.page_content, metadata=document.metadata)
        for document in result.get("documents", [])
    ]
    return ChatResponse(answer=result["answer"], intent=result["intent"], sources=sources)


@app.post("/api/products/compare")
def compare_products(request: CompareRequest) -> dict[str, list[Product]]:
    products_by_id = {product.id: product for product in get_products()}
    missing = [product_id for product_id in request.product_ids if product_id not in products_by_id]
    if missing:
        raise HTTPException(status_code=404, detail=f"Unknown product IDs: {', '.join(missing)}")
    return {"products": [products_by_id[product_id] for product_id in request.product_ids]}


@app.post("/api/recommendations", response_model=list[Product])
def recommend(request: RecommendationRequest) -> list[Product]:
    matches = find_products(request.query)
    if not matches:
        matches = get_products()
    if request.max_price is not None:
        matches = [product for product in matches if product.price <= request.max_price]
    return sorted(matches, key=lambda product: (product.inventory == 0, -product.rating))[: request.limit]


@app.get("/api/orders/{order_id}")
def get_order(order_id: str) -> dict[str, str]:
    try:
        order = track_order(order_id)
    except Exception as error:
        raise HTTPException(status_code=503, detail="Order service is unavailable.") from error
    if not order:
        raise HTTPException(status_code=404, detail="Order not found.")
    return order


@app.post("/api/knowledge/ingest")
def ingest(request: IngestRequest) -> dict[str, int | str]:
    try:
        document_count = knowledge_base.ingest(request.content, request.source, request.metadata)
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": "ingested", "document_count": document_count}