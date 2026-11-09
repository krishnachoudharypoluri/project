from app.config import get_settings


DEMO_ORDERS = {
    "SA-1042": {"order_id": "SA-1042", "status": "In transit", "estimated_delivery": "2026-11-12"},
    "SA-2088": {"order_id": "SA-2088", "status": "Processing", "estimated_delivery": "2026-11-14"},
}


def track_order(order_id: str) -> dict[str, str] | None:
    settings = get_settings()
    if not settings.mongodb_uri:
        return DEMO_ORDERS.get(order_id.upper())

    from pymongo import MongoClient

    with MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000) as client:
        order = client[settings.mongodb_database][settings.mongodb_orders_collection].find_one(
            {"order_id": order_id.upper()}, {"_id": 0}
        )
    return order