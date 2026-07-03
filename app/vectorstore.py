"""Qdrant-Anbindung: Client, Collection-Setup und Payload-Schluessel.

Die Collection speichert pro Chunk einen Vektor. Konfiguration laut Vorgabe:
Cosine-Distanz, Original-Vektoren ``on_disk``, int8 Scalar Quantization
(im RAM), sowie Payload-Indizes fuer die denormalisierten Filterfelder.
Das Setup ist idempotent.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from app.config import get_settings

# Payload-Schluessel der denormalisierten Filterfelder (Kopie aus Postgres).
PAYLOAD_DOC_ID = "doc_id"
PAYLOAD_DOC_TYPE = "doc_type"
PAYLOAD_LANGUAGE = "language"
PAYLOAD_CLASSIFICATION = "classification"
PAYLOAD_CREATED_AT = "created_at"  # Unix-Timestamp (int) fuer Range-Filter
PAYLOAD_CHUNK_INDEX = "chunk_index"

# Payload-Felder mit ihrem Index-Schema.
_PAYLOAD_INDEXES: dict[str, qm.PayloadSchemaType] = {
    PAYLOAD_DOC_ID: qm.PayloadSchemaType.INTEGER,
    PAYLOAD_DOC_TYPE: qm.PayloadSchemaType.KEYWORD,
    PAYLOAD_LANGUAGE: qm.PayloadSchemaType.KEYWORD,
    PAYLOAD_CLASSIFICATION: qm.PayloadSchemaType.KEYWORD,
    PAYLOAD_CREATED_AT: qm.PayloadSchemaType.INTEGER,
}


def get_client() -> QdrantClient:
    """Erzeugt einen Qdrant-Client aus der Konfiguration."""
    settings = get_settings()
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collection(client: QdrantClient | None = None) -> None:
    """Legt die Collection samt Quantisierung und Payload-Indizes an.

    Idempotent: existiert die Collection bereits, werden nur fehlende
    Payload-Indizes ergaenzt.
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
                on_disk=True,  # Original-Vektoren auf Platte
            ),
            quantization_config=qm.ScalarQuantization(
                scalar=qm.ScalarQuantizationConfig(
                    type=qm.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,  # quantisierte Vektoren im RAM
                ),
            ),
        )

    _ensure_payload_indexes(client, name)


def _ensure_payload_indexes(client: QdrantClient, name: str) -> None:
    """Legt fehlende Payload-Indizes an (idempotent)."""
    existing = client.get_collection(name).payload_schema or {}
    for field_name, schema in _PAYLOAD_INDEXES.items():
        if field_name in existing:
            continue
        client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=schema,
        )
