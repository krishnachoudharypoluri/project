# ShopAssist AI

ShopAssist is a starter e-commerce assistant for product discovery, recommendations, comparisons, order tracking, and store-policy questions. It uses FastAPI for its API, LangGraph for the request workflow, and LangChain/OpenAI when an API key is configured. Without external credentials it runs with a small local catalog and keyword-based retrieval.

## Run locally

Requires Python 3.11 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API. The default configuration uses the in-memory knowledge base and demo order records. Add `OPENAI_API_KEY` to `.env` to enable GPT-4o responses.

## Optional integrations

Install only the providers you intend to use:

```powershell
python -m pip install -e ".[chroma]"
python -m pip install -e ".[postgres]"
python -m pip install -e ".[mongodb]"
```

Set `VECTOR_STORE=chroma` and optionally `CHROMA_PATH` to persist vectors locally. For PostgreSQL with the pgvector extension, install the `postgres` extra and set `VECTOR_STORE=pgvector` plus `PGVECTOR_CONNECTION` (a SQLAlchemy connection string, for example `postgresql+psycopg://user:password@localhost:5432/shopassist`). Set `MONGODB_URI` to use a MongoDB `orders` collection for order lookups; without it, demo records are used.

The vector integrations use OpenAI embeddings, so configure `OPENAI_API_KEY` when using Chroma or PGVector. The in-memory backend is intended for local demos and does keyword-based ranking rather than embeddings.

## API

- `GET /health` reports service and provider configuration.
- `GET /api/products?q=headphones` searches the catalog.
- `POST /api/chat` runs the LangGraph retrieval and response workflow.
- `POST /api/products/compare` compares two or more product IDs.
- `POST /api/recommendations` returns catalog matches for a request.
- `GET /api/orders/{order_id}` looks up an order in MongoDB or the demo records.
- `POST /api/knowledge/ingest` adds a text document to the selected knowledge base.

Example chat request:

```json
{
  "message": "Does the AeroSound Pro have noise cancellation?"
}
```

## Project layout

```text
app/
  data/          Seed products and store policies
  services/      Catalog, retrieval, order lookup, and LangGraph workflow
  config.py      Environment-backed settings
  main.py        FastAPI application and endpoints
  schemas.py     Request and response models
tests/           Focused API smoke tests
```

Before exposing this starter to customers, add authentication and authorization for ingestion, connect real commerce/order services, and configure production logging, monitoring, and data-retention policies.