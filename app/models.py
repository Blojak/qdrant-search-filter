"""ORM-Modelle: ``Document`` und ``Chunk``.

PostgreSQL ist die Single Source of Truth fuer alle Metadaten. Ein Dokument
besitzt N Chunks; jeder Chunk entspricht genau einem Embedding bzw. einem
Vektor in Qdrant. Die Metadaten sind bewusst in Kategorien gegliedert:
technisch, deskriptiv, administrativ und flexibel (JSONB ``extra``).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base
from app.enums import Classification, DocType, Language


class Document(Base):
    """Ein Dokument als Traeger aller Metadaten (Wahrheit in Postgres)."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # --- technisch ---
    content_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True,
        doc="sha256-Hash des Rohinhalts fuer Deduplizierung",
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )

    # --- deskriptiv ---
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    language: Mapped[Language] = mapped_column(
        SAEnum(Language, name="language_enum", native_enum=True),
        nullable=False, default=Language.UNKNOWN, index=True,
    )
    doc_type: Mapped[DocType] = mapped_column(
        SAEnum(DocType, name="doc_type_enum", native_enum=True),
        nullable=False, default=DocType.OTHER, index=True,
    )
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
        doc="Fachliches Dokumentdatum (nicht Ingestion-Zeitpunkt)",
    )

    # --- administrativ ---
    classification: Mapped[Classification] = mapped_column(
        SAEnum(Classification, name="classification_enum", native_enum=True),
        nullable=False, default=Classification.INTERNAL, index=True,
    )
    source: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # --- flexibel / typspezifisch ---
    extra: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="Chunk.chunk_index",
    )

    def __repr__(self) -> str:  # pragma: no cover - Debug-Hilfe
        return f"<Document id={self.id} filename={self.filename!r}>"


class Chunk(Base):
    """Ein Textabschnitt eines Dokuments. Ein Chunk = ein Vektor in Qdrant."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_doc_index"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="0-basierte Position innerhalb des Dokuments",
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)

    document: Mapped["Document"] = relationship(back_populates="chunks")

    def __repr__(self) -> str:  # pragma: no cover - Debug-Hilfe
        return f"<Chunk id={self.id} doc={self.document_id} idx={self.chunk_index}>"
