# qdrant-search — semantic document search (PoC)

Proof of concept for semantic search over a document collection. Documents are
split into chunks, embedded locally with a multilingual model, and made
searchable in Qdrant. Metadata is modeled carefully in PostgreSQL as the single
source of truth.

## Architecture

Two stores with a denormalized filter copy:

- **PostgreSQL** — full metadata (the truth), relational, via SQLAlchemy.
- **Qdrant** — one vector per chunk. The payload holds only the filter-relevant
  fields as a denormalized copy (`doc_id`, `doc_type`, `language`,
  `classification`, `created_at`) so filtering happens during the search.

Search flow: embed query → Qdrant search with vector + filters → hits carry
`doc_id`/`chunk_id` → load full metadata from Postgres for display.

Ingestion flow: read text → content-hash dedup → write document metadata to
Postgres → chunk → embed → write chunks to Postgres and vectors + payload to
Qdrant (the Postgres chunk id is reused as the Qdrant point id).

## Tech stack

Python 3.11+, Qdrant, PostgreSQL, SQLAlchemy 2.0, sentence-transformers
(`intfloat/multilingual-e5-large`), qdrant-client, Flask, pydantic-settings.
All open source. Infrastructure via Docker Compose.

## Metadata model

`documents` (source of truth) groups fields by category:

- **technical**: `content_hash` (sha256, dedup), `filename`, `mime_type`,
  `size_bytes`, `ingested_at`
- **descriptive**: `title`, `language`, `doc_type`, `created_at` (document date)
- **administrative**: `classification`, `source`
- **flexible**: `extra` (JSONB) for the long tail

`language`, `doc_type` and `classification` are controlled vocabularies backed
by Python enums and native PostgreSQL enum types (no free text). `chunks` has a
foreign key to `documents`; one chunk = one embedding = one vector.

## Setup

```bash
# 1. Start infrastructure (Qdrant + PostgreSQL)
docker compose up -d

# 2. Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configuration
cp .env.example .env        # adjust if needed (ports, model, chunk size, ...)
```

> Note: host port `5432` is often taken by a local PostgreSQL, so the container
> is mapped to `5433` by default (`POSTGRES_PORT` in `.env`). The first search or
> ingest downloads the embedding model (~2.2 GB) once.

### pgAdmin (optional database UI)

Compose also starts **pgAdmin** at http://localhost:5050 (`PGADMIN_PORT`). Log in
with `PGADMIN_DEFAULT_EMAIL` / `PGADMIN_DEFAULT_PASSWORD` (default
`admin@example.com` / `admin`). The `qsearch` server is pre-registered (via
`pgadmin/servers.json`); enter the Postgres password (`qsearch`) on first
connect. Inside the compose network pgAdmin reaches Postgres as host `postgres`
on port `5432` (not the host-mapped `5433`).

## Run the API

```bash
python -m app.api          # serves on http://localhost:5001 (API_PORT)
```

Schema and Qdrant collection are created automatically on startup (idempotent).

## Example requests

### Ingest a document

```bash
curl -s -X POST http://localhost:5001/documents \
  -H 'Content-Type: application/json' \
  -d '{
    "filename": "report_en_2024.txt",
    "title": "Q1 2024 Security Report",
    "language": "en",
    "doc_type": "report",
    "classification": "internal",
    "created_at": "2024-03-31T00:00:00+00:00",
    "source": "security-team",
    "extra": {"quarter": "Q1"},
    "content": "The quarterly security report for Q1 2024 ..."
  }'
```

Alternatively pass `"path": "sample_docs/report_en_2024.txt"` instead of
`"content"` to ingest a file readable by the server (supports `.txt` and `.pdf`).

Response (`201` created, `200` if deduplicated by content hash):

```json
{"document_id": 2, "filename": "report_en_2024.txt", "num_chunks": 1, "deduplicated": false}
```

### Search (with optional filters)

```bash
curl -s -X POST http://localhost:5001/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "How do we protect against phishing?",
    "limit": 5,
    "filters": {
      "language": "en",
      "doc_type": "report",
      "created_from": "2024-01-01T00:00:00+00:00",
      "created_to": "2024-12-31T23:59:59+00:00"
    }
  }'
```

Response:

```json
{
  "query": "How do we protect against phishing?",
  "count": 1,
  "results": [
    {
      "score": 0.87,
      "chunk_id": 42,
      "chunk_index": 0,
      "chunk_text": "The quarterly security report ...",
      "document": {"id": 2, "title": "Q1 2024 Security Report", "language": "en", "doc_type": "report", "...": "..."}
    }
  ]
}
```

### Delete a document

Removes the document and its chunks from Postgres and its vectors from Qdrant.

```bash
curl -s -X DELETE http://localhost:5001/documents/4
```

Response `200` (`{"deleted": true, "document_id": 4}`) or `404` if the id does
not exist.

## Evaluating recall

To measure retrieval quality, build a small labeled test set of
`(query, relevant_document_ids)` pairs. Run each query through `/search`, take
the top-k results, and compute **recall@k** = (relevant docs retrieved in top-k)
/ (total relevant docs), averaged over all queries. Vary `k`, `CHUNK_SIZE`,
`CHUNK_OVERLAP` and `SEARCH_OVERSAMPLING` to see their effect. Complementary
metrics: MRR and precision@k.

## Scope

Deliberately excluded (PoC): authentication, frontend, reranking, full
extraction pipeline (only `.txt` and simple `.pdf` text), neural-search plugins,
embedding models inside the cluster, and database migrations (SQLAlchemy
`create_all` on startup instead of Alembic).
