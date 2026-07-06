"""Ingestion pipeline.

Flow: read text -> compute content hash (dedup) -> write document metadata to
Postgres -> split into chunks -> embed -> write chunks to Postgres and the
vectors plus a denormalized filter payload to Qdrant.

Postgres remains the source of truth. Qdrant only holds the vector and the
few filter-relevant fields. The Postgres-generated chunk id is reused as the
Qdrant point id (one chunk = one vector).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from qdrant_client import models as qm
from sqlalchemy import select

from app.chunking import chunk_text
from app.config import get_settings
from app.db import session_scope
from app.embedding import embed_passages
from app.enums import Classification, DocType, Language
from app.models import Chunk, Document
from app.vectorstore import (
    PAYLOAD_CHUNK_INDEX,
    PAYLOAD_CLASSIFICATION,
    PAYLOAD_CREATED_AT,
    PAYLOAD_DOC_ID,
    PAYLOAD_DOC_TYPE,
    PAYLOAD_LANGUAGE,
    get_client,
)

# Upsert points to Qdrant in batches so a single large document does not
# produce one oversized HTTP request (which would time out).
_UPSERT_BATCH = 256


@dataclass
class DocumentMeta:
    """Caller-supplied metadata for a document to ingest."""

    filename: str
    mime_type: str = "text/plain"
    title: str | None = None
    language: Language = Language.UNKNOWN
    doc_type: DocType = DocType.OTHER
    classification: Classification = Classification.INTERNAL
    created_at: datetime | None = None
    source: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class IngestResult:
    """Outcome of an ingestion call."""

    document_id: int
    filename: str
    num_chunks: int
    deduplicated: bool  # True if the document already existed (by hash)


def compute_hash(content: str) -> str:
    """sha256 over the UTF-8 bytes of the raw content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def read_document_file(path: str | Path) -> tuple[str, str]:
    """Read a supported file and return ``(text, mime_type)``.

    Supports ``.txt`` and (optionally) ``.pdf`` via pypdf. Everything else is
    treated as plain UTF-8 text.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text, "application/pdf"
    return path.read_text(encoding="utf-8"), "text/plain"


def _build_payload(doc: Document, chunk_index: int) -> dict:
    """Denormalized filter payload for a single chunk (copy from Postgres)."""
    payload: dict = {
        PAYLOAD_DOC_ID: doc.id,
        PAYLOAD_DOC_TYPE: doc.doc_type.value,
        PAYLOAD_LANGUAGE: doc.language.value,
        PAYLOAD_CLASSIFICATION: doc.classification.value,
        PAYLOAD_CHUNK_INDEX: chunk_index,
    }
    if doc.created_at is not None:
        payload[PAYLOAD_CREATED_AT] = int(doc.created_at.timestamp())
    return payload


def ingest_text(content: str, meta: DocumentMeta) -> IngestResult:
    """Ingest raw text under the given metadata.

    Deduplicates by content hash: if a document with the same hash already
    exists, nothing is written and the existing document is returned.
    """
    settings = get_settings()
    content_hash = compute_hash(content)
    client = get_client()

    with session_scope() as session:
        existing = session.execute(
            select(Document).where(Document.content_hash == content_hash)
        ).scalar_one_or_none()
        if existing is not None:
            return IngestResult(
                document_id=existing.id,
                filename=existing.filename,
                num_chunks=len(existing.chunks),
                deduplicated=True,
            )

        chunks = chunk_text(content)
        if not chunks:
            raise ValueError("no content to ingest (empty after chunking)")

        doc = Document(
            content_hash=content_hash,
            filename=meta.filename,
            mime_type=meta.mime_type,
            size_bytes=len(content.encode("utf-8")),
            title=meta.title,
            language=meta.language,
            doc_type=meta.doc_type,
            created_at=meta.created_at,
            classification=meta.classification,
            source=meta.source,
            extra=meta.extra,
            body=content,  # full text for highlighting; offsets index into this
        )
        doc.chunks = [
            Chunk(
                chunk_index=c.index,
                text=c.text,
                char_count=c.char_count,
                start_char=c.start,
                end_char=c.end,
            )
            for c in chunks
        ]
        session.add(doc)
        session.flush()  # assign doc.id and chunk ids

        vectors = embed_passages([c.text for c in doc.chunks])
        points = [
            qm.PointStruct(
                id=chunk.id,
                vector=vector,
                payload=_build_payload(doc, chunk.chunk_index),
            )
            for chunk, vector in zip(doc.chunks, vectors)
        ]
        for start in range(0, len(points), _UPSERT_BATCH):
            client.upsert(
                collection_name=settings.qdrant_collection,
                points=points[start : start + _UPSERT_BATCH],
                wait=True,
            )

        return IngestResult(
            document_id=doc.id,
            filename=doc.filename,
            num_chunks=len(doc.chunks),
            deduplicated=False,
        )


def ingest_file(path: str | Path, meta: DocumentMeta | None = None) -> IngestResult:
    """Read a file from disk and ingest it. If ``meta`` is omitted, sensible
    defaults are derived from the filename."""
    path = Path(path)
    text, mime_type = read_document_file(path)
    if meta is None:
        meta = DocumentMeta(filename=path.name, mime_type=mime_type)
    else:
        meta.mime_type = mime_type
    return ingest_text(text, meta)


def delete_document(doc_id: int) -> bool:
    """Delete a document from both stores.

    Returns ``True`` if the document existed and was removed, ``False`` if no
    document with that id exists. Postgres is the source of truth, so it is
    removed (and committed) first; the now-orphaned vectors are then dropped
    from Qdrant.
    """
    settings = get_settings()
    client = get_client()

    with session_scope() as session:
        doc = session.get(Document, doc_id)
        if doc is None:
            return False
        session.delete(doc)  # cascade removes the document's chunks

    # Postgres delete is committed at this point; drop the vectors in Qdrant.
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key=PAYLOAD_DOC_ID, match=qm.MatchValue(value=doc_id)
                    )
                ]
            )
        ),
        wait=True,
    )
    return True
