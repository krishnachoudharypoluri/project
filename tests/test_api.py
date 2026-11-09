from fastapi.testclient import TestClient
from types import SimpleNamespace

from app.main import app
from app.services import assistant


client = TestClient(app)


def test_health_and_product_search() -> None:
    assert client.get("/health").json()["status"] == "ok"
    products = client.get("/api/products", params={"q": "NS-AERO-PRO-01"}).json()
    assert len(products) == 1
    assert products[0]["id"] == "aerosound-pro"


def test_chat_retrieves_product_manual() -> None:
    response = client.post("/api/chat", json={"message": "How do I pair AeroSound Pro?"})
    assert response.status_code == 200
    assert "flashes blue" in response.json()["answer"]
    assert any(source["metadata"].get("type") == "product_manual" for source in response.json()["sources"])


def test_comparison_and_order_tracking() -> None:
    comparison = client.post(
        "/api/products/compare", json={"product_ids": ["aerosound-pro", "aerosound-lite"]}
    )
    assert comparison.status_code == 200
    assert len(comparison.json()["products"]) == 2
    order = client.get("/api/orders/SA-1042")
    assert order.status_code == 200
    assert order.json()["status"] == "In transit"


def test_missing_order_is_not_found() -> None:
    assert client.get("/api/orders/SA-0000").status_code == 404


def test_review_summary_returns_grounded_fallback(monkeypatch) -> None:
    monkeypatch.setattr(assistant, "get_settings", lambda: SimpleNamespace(openai_api_key=None))

    response = client.get("/api/products/aerosound-pro/review-summary")

    assert response.status_code == 200
    assert response.json()["ai_generated"] is False
    assert "noise cancellation" in response.json()["summary"].lower()
    assert all(
        source["metadata"].get("product_id") == "aerosound-pro"
        for source in response.json()["sources"]
    )


def test_review_summary_uses_configured_llm(monkeypatch) -> None:
    class FakeModel:
        def __init__(self, **_: object) -> None:
            pass

        def invoke(self, prompt: str) -> SimpleNamespace:
            assert "AeroSound Pro customer review" in prompt
            return SimpleNamespace(content="Customers like the noise cancellation and comfort.")

    monkeypatch.setattr(
        assistant,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key="test-key", openai_model="gpt-4o"),
    )
    monkeypatch.setattr(assistant, "ChatOpenAI", FakeModel)

    response = client.get("/api/products/aerosound-pro/review-summary")

    assert response.status_code == 200
    assert response.json()["ai_generated"] is True
    assert "Customers like" in response.json()["summary"]


def test_review_summary_returns_not_found_for_unknown_product() -> None:
    assert client.get("/api/products/unknown/review-summary").status_code == 404


def test_ingested_child_match_returns_parent_context() -> None:
    opening = "This is the original introduction to a long product support document. "
    searchable_detail = "Distinctive calibration sequence ZXQ918 requires a full power cycle before setup."
    content = opening + ("Additional reference information for this product. " * 20) + searchable_detail
    ingested = client.post(
        "/api/knowledge/ingest",
        json={"content": content, "source": "long-support-guide", "metadata": {"type": "support"}},
    )
    assert ingested.status_code == 200

    response = client.post("/api/chat", json={"message": "What is ZXQ918?"})
    assert response.status_code == 200
    assert opening in response.json()["answer"]
    assert searchable_detail in response.json()["answer"]