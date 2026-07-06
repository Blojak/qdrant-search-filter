"""ORM models: ``Document`` and ``Chunk``.

PostgreSQL is the single source of truth for all metadata. A document owns
N chunks; each chunk corresponds to exactly one embedding / one vector in
Qdrant. The metadata is deliberately grouped into categories: technical,
descriptive, administrative and flexible (JSONB ``extra``).
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


def _enum_column(enum_cls: type, type_name: str) -> SAEnum:
    """Create a native PG enum column that persists the enum *values* (not the
    names), so the Postgres representation is identical to the denormalized
    Qdrant payload (e.g. ``de`` instead of ``DE``)."""
    return SAEnum(
        enum_cls,
        name=type_name,
        native_enum=True,
        values_callable=lambda cls: [member.value for member in cls],
    )


class Document(Base):
    """A document carrying all metadata (source of truth in Postgres)."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # --- technical ---
    content_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True,
        doc="sha256 hash of the raw content for deduplication",
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )

    # --- descriptive ---
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    language: Mapped[Language] = mapped_column(
        _enum_column(Language, "language_enum"),
        nullable=False, default=Language.UNKNOWN, index=True,
    )
    doc_type: Mapped[DocType] = mapped_column(
        _enum_column(DocType, "doc_type_enum"),
        nullable=False, default=DocType.OTHER, index=True,
    )
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
        doc="Functional document date (not the ingestion timestamp)",
    )

    # --- administrative ---
    classification: Mapped[Classification] = mapped_column(
        _enum_column(Classification, "classification_enum"),
        nullable=False, default=Classification.INTERNAL, index=True,
    )
    source: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # --- flexible / type-specific ---
    extra: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Full source text, kept so hits can be highlighted inside the document.
    # Nullable: documents ingested before this column existed have no body.
    body: Mapped[str | None] = mapped_column(Text, nullable=True)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="Chunk.chunk_index",
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Document id={self.id} filename={self.filename!r}>"


class Chunk(Base):
    """A text segment of a document. One chunk = one vector in Qdrant."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_doc_index"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer, nullable=False, doc="0-based position within the document",
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Character offsets into Document.body (text[start:end] == chunk text).
    # Nullable for chunks created before these columns existed.
    start_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_char: Mapped[int | None] = mapped_column(Integer, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="chunks")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Chunk id={self.id} doc={self.document_id} idx={self.chunk_index}>"
