from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"
    vector_store: str = "memory"
    chroma_path: str = "./.chroma"
    chroma_collection: str = "shopassist"
    pgvector_connection: str | None = None
    mongodb_uri: str | None = None
    mongodb_database: str = "shopassist"
    mongodb_orders_collection: str = "orders"


@lru_cache
def get_settings() -> Settings:
    return Settings()