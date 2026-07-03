"""Semantic search over the indexed chunks.

Flow: embed the query -> search Qdrant with the vector plus optional metadata
filters (with quantization rescoring/oversampling) -> collect the matching
chunk/document ids -> load the full metadata from Postgres for display.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from qdrant_client import models as qm
from sqlalchemy import select

from app.config import get_settings
from app.db import session_scope
from app.embedding import embed_query
from app.enums import Classification, DocType, Language
from app.models import Chunk, Document
from app.vectorstore import (
    PAYLOAD_CLASSIFICATION,
    PAYLOAD_CREATED_AT,
    PAYLOAD_DOC_TYPE,
    PAYLOAD_LANGUAGE,
    get_client,
)


@dataclass
class SearchFilters:
    """Optional metadata filters applied during the vector search."""

    doc_type: DocType | None = None
    language: Language | None = None
    classification: Classification | None = None
    created_from: datetime | None = None
    created_to: datetime | None = None


@dataclass
class SearchHit:
    """A single chunk hit enriched with its full document metadata."""

    score: float
    chunk_id: int
    chunk_index: int
    chunk_text: str
    document: dict


def _build_filter(filters: SearchFilters | None) -> qm.Filter | None:
    """Translate ``SearchFilters`` into a Qdrant filter (all conditions AND)."""
    if filters is None:
        return None

    must: list[qm.FieldCondition] = []
    if filters.doc_type is not None:
        must.append(
            qm.FieldCondition(
                key=PAYLOAD_DOC_TYPE,
                match=qm.MatchValue(value=filters.doc_type.value),
            )
        )
    if filters.language is not None:
        must.append(
            qm.FieldCondition(
                key=PAYLOAD_LANGUAGE,
                match=qm.MatchValue(value=filters.language.value),
            )
        )
    if filters.classification is not None:
        must.append(
            qm.FieldCondition(
                key=PAYLOAD_CLASSIFICATION,
                match=qm.MatchValue(value=filters.classification.value),
            )
        )
    if filters.created_from is not None or filters.created_to is not None:
        must.append(
            qm.FieldCondition(
                key=PAYLOAD_CREATED_AT,
                range=qm.Range(
                    gte=(
                        int(filters.created_from.timestamp())
                        if filters.created_from
                        else None
                    ),
                    lte=(
                        int(filters.created_to.timestamp())
                        if filters.created_to
                        else None
                    ),
                ),
            )
        )
    return qm.Filter(must=must) if must else None


def _document_to_dict(doc: Document) -> dict:
    """Serialize a document's metadata to a JSON-friendly dict."""
    return {
        "id": doc.id,
        "filename": doc.filename,
        "title": doc.title,
        "mime_type": doc.mime_type,
        "size_bytes": doc.size_bytes,
        "language": doc.language.value,
        "doc_type": doc.doc_type.value,
        "classification": doc.classification.value,
        "source": doc.source,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "ingested_at": doc.ingested_at.isoformat() if doc.ingested_at else None,
        "extra": doc.extra,
    }


def search(
    query: str,
    filters: SearchFilters | None = None,
    limit: int = 10,
) -> list[SearchHit]:
    """Run a filtered semantic search and return enriched chunk hits."""
    settings = get_settings()
    vector = embed_query(query)
    client = get_client()

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=vector,
        query_filter=_build_filter(filters),
        limit=limit,
        with_payload=True,
        search_params=qm.SearchParams(
            quantization=qm.QuantizationSearchParams(
                rescore=True,
                oversampling=settings.search_oversampling,
            ),
        ),
    )
    points = response.points
    if not points:
        return []

    chunk_ids = [int(p.id) for p in points]
    with session_scope() as session:
        chunks = (
            session.execute(select(Chunk).where(Chunk.id.in_(chunk_ids)))
            .scalars()
            .all()
        )
        chunk_by_id = {c.id: c for c in chunks}
        doc_ids = {c.document_id for c in chunks}
        docs = (
            session.execute(select(Document).where(Document.id.in_(doc_ids)))
            .scalars()
            .all()
        )
        doc_by_id = {_document_to_dict(d)["id"]: _document_to_dict(d) for d in docs}

        hits: list[SearchHit] = []
        for point in points:  # preserve Qdrant ranking
            chunk = chunk_by_id.get(int(point.id))
            if chunk is None:  # vector without matching Postgres row (should not happen)
                continue
            hits.append(
                SearchHit(
                    score=point.score,
                    chunk_id=chunk.id,
                    chunk_index=chunk.chunk_index,
                    chunk_text=chunk.text,
                    document=doc_by_id[chunk.document_id],
                )
            )
    return hits
