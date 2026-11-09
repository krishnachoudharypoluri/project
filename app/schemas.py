from pydantic import BaseModel, Field


class Product(BaseModel):
    id: str
    name: str
    brand: str
    category: str
    price: float
    currency: str = "USD"
    rating: float
    inventory: int
    sku: str
    description: str
    specifications: dict[str, str]


class ChatRequest(BaseModel):
    message: str = Field(min_length=2, max_length=4000)


class Source(BaseModel):
    content: str
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    intent: str
    sources: list[Source]


class ReviewSummaryResponse(BaseModel):
    product_id: str
    summary: str
    ai_generated: bool
    sources: list[Source]


class CompareRequest(BaseModel):
    product_ids: list[str] = Field(min_length=2, max_length=4)


class RecommendationRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    max_price: float | None = Field(default=None, gt=0)
    limit: int = Field(default=3, ge=1, le=10)


class IngestRequest(BaseModel):
    content: str = Field(min_length=20, max_length=20000)
    source: str = Field(min_length=1, max_length=200)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)