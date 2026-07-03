"""Zentrale Konfiguration via pydantic-settings.

Alle Laufzeitparameter werden aus der Umgebung bzw. der ``.env``-Datei gelesen.
Kein Wert wird im Code hartkodiert – Aenderungen erfolgen ausschliesslich ueber
die Umgebung.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Anwendungsweite Einstellungen."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- PostgreSQL ---
    postgres_user: str = "qsearch"
    postgres_password: str = "qsearch"
    postgres_db: str = "qsearch"
    postgres_host: str = "localhost"
    postgres_port: int = 5433

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    # --- Embedding-Modell ---
    embedding_model: str = "intfloat/multilingual-e5-large"
    vector_size: int = 1024

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- Suche ---
    search_oversampling: float = 2.0

    # --- Flask-API ---
    api_host: str = "0.0.0.0"
    api_port: int = 5001

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """SQLAlchemy-Verbindungs-URL fuer PostgreSQL (psycopg2-Treiber)."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Liefert eine gecachte Settings-Instanz (Singleton)."""
    return Settings()
