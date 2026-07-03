"""Qdrant integration: client, collection setup and payload keys.

The collection stores one vector per chunk. Configuration per spec: cosine
distance, original vectors ``on_disk``, int8 scalar quantization (kept in
RAM), plus payload indexes for the denormalized filter fields. The setup is
idempotent.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from app.config import get_settings

# Payload keys of the denormalized filter fields (copy from Postgres).
PAYLOAD_DOC_ID = "doc_id"
PAYLOAD_DOC_TYPE = "doc_type"
PAYLOAD_LANGUAGE = "language"
PAYLOAD_CLASSIFICATION = "classification"
PAYLOAD_CREATED_AT = "created_at"  # Unix timestamp (int) for range filters
PAYLOAD_CHUNK_INDEX = "chunk_index"

# Payload fields with their index schema.
_PAYLOAD_INDEXES: dict[str, qm.PayloadSchemaType] = {
    PAYLOAD_DOC_ID: qm.PayloadSchemaType.INTEGER,
    PAYLOAD_DOC_TYPE: qm.PayloadSchemaType.KEYWORD,
    PAYLOAD_LANGUAGE: qm.PayloadSchemaType.KEYWORD,
    PAYLOAD_CLASSIFICATION: qm.PayloadSchemaType.KEYWORD,
    PAYLOAD_CREATED_AT: qm.PayloadSchemaType.INTEGER,
}


def get_client() -> QdrantClient:
    """Create a Qdrant client from the configuration."""
    settings = get_settings()
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collection(client: QdrantClient | None = None) -> None:
    """Create the collection incl. quantization and payload indexes.

    Idempotent: if the collection already exists, only missing payload
    indexes are added.
    """
    settings = get_settings()
    client = client or get_client()
    name = settings.qdrant_collection

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(
                size=settings.vector_size,
                distance=qm.Distance.COSINE,
                on_disk=True,  # original vectors on disk
            ),
            quantization_config=qm.ScalarQuantization(
                scalar=qm.ScalarQuantizationConfig(
                    type=qm.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,  # quantized vectors kept in RAM
                ),
            ),
        )

    _ensure_payload_indexes(client, name)


def _ensure_payload_indexes(client: QdrantClient, name: str) -> None:
    """Create missing payload indexes (idempotent)."""
    existing = client.get_collection(name).payload_schema or {}
    for field_name, schema in _PAYLOAD_INDEXES.items():
        if field_name in existing:
            continue
        client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=schema,
        )
