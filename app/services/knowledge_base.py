import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.services.catalog import get_products, product_document


POLICY_PATH = Path(__file__).resolve().parents[1] / "data" / "store_policies.md"
DATA_PATH = Path(__file__).resolve().parents[1] / "data"
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)


class KnowledgeBase:
    def __init__(self) -> None:
        seed_documents = self._load_seed_documents()
        self._parents: list[Document] = []
        self._children: list[Document] = []
        self._parent_by_id: dict[str, Document] = {}
        for parent in seed_documents:
            self._add_parent(parent)
        self._store: Any = None

    def _load_seed_documents(self) -> list[Document]:
        documents = [
            Document(page_content=product_document(product), metadata={"type": "product", "product_id": product.id})
            for product in get_products()
        ]
        documents.append(
            Document(
                page_content=POLICY_PATH.read_text(encoding="utf-8"),
                metadata={"type": "store_policy", "source": "store_policies.md"},
            )
        )
        for filename, kind in (("reviews.json", "customer_review"), ("manuals.json", "product_manual")):
            entries = json.loads((DATA_PATH / filename).read_text(encoding="utf-8"))
            documents.extend(
                Document(page_content=entry["content"], metadata={"type": kind, **entry.get("metadata", {})})
                for entry in entries
            )
        return documents

    def _add_parent(self, parent: Document) -> list[Document]:
        parent_id = str(uuid4())
        self._parents.append(parent)
        self._parent_by_id[parent_id] = parent
        children = [
            Document(page_content=chunk, metadata={**parent.metadata, "parent_id": parent_id})
            for chunk in SPLITTER.split_text(parent.page_content)
        ]
        self._children.extend(children)
        return children

    def _get_store(self) -> Any:
        if self._store is not None:
            return self._store

        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for vector-store embeddings.")

        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
        if settings.vector_store == "chroma":
            from langchain_chroma import Chroma

            self._store = Chroma(
                collection_name=settings.chroma_collection,
                embedding_function=embeddings,
                persist_directory=settings.chroma_path,
            )
        elif settings.vector_store == "pgvector":
            if not settings.pgvector_connection:
                raise RuntimeError("PGVECTOR_CONNECTION must be set when VECTOR_STORE=pgvector.")
            from langchain_postgres import PGVector

            self._store = PGVector(
                embeddings=embeddings,
                collection_name="shopassist",
                connection=settings.pgvector_connection,
                use_jsonb=True,
            )
        else:
            raise RuntimeError(f"Unsupported vector store: {settings.vector_store}")

        self._store.add_documents(self._children)
        return self._store

    def search(self, query: str, limit: int = 4) -> list[Document]:
        terms = {term.lower().strip(".,?!:;()[]") for term in query.split() if len(term) > 2}
        scores: dict[str, float] = {}
        candidates: dict[str, Document] = {}

        def add_ranked(documents: list[Document], weight: float) -> None:
            for rank, document in enumerate(documents, start=1):
                key = document.metadata.get("parent_id", document.page_content)
                candidates[key] = document
                scores[key] = scores.get(key, 0.0) + weight / (60 + rank)

        keyword_ranked = sorted(
            self._children,
            key=lambda document: sum(
                term in (document.page_content + " " + " ".join(str(value) for value in document.metadata.values())).lower()
                for term in terms
            ),
            reverse=True,
        )
        add_ranked(
            [document for document in keyword_ranked if any(term in document.page_content.lower() for term in terms)],
            1.0,
        )

        settings = get_settings()
        if settings.vector_store in {"chroma", "pgvector"}:
            add_ranked(self._get_store().similarity_search(query, k=limit * 3), 1.0)

        ordered_keys = sorted(scores, key=scores.get, reverse=True)
        parents: list[Document] = []
        seen: set[tuple[Any, ...]] = set()
        for key in ordered_keys:
            child = candidates[key]
            parent_id = child.metadata.get("parent_id")
            parent = self._parent_by_id.get(parent_id, child) if parent_id else child
            identity = (
                parent.metadata.get("source"),
                parent.metadata.get("product_id"),
                parent.metadata.get("type"),
            )
            if identity not in seen:
                seen.add(identity)
                parents.append(parent)
            if len(parents) == limit:
                break
        return parents

    def ingest(self, content: str, source: str, metadata: dict[str, Any]) -> int:
        store = self._get_store() if get_settings().vector_store in {"chroma", "pgvector"} else None
        document = Document(page_content=content, metadata={**metadata, "source": source})
        children = self._add_parent(document)
        if store is not None:
            store.add_documents(children)
        return len(self._parents)


knowledge_base = KnowledgeBase()