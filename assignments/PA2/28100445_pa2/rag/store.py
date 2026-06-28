"""
rag/store.py
Pipeline integration: loaders, splitters, record manager, vector store.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_classic.indexes import SQLRecordManager, index
from databricks_langchain.embeddings import DatabricksEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader

from utils.client import PGVECTOR_CONNECTION_STRING

load_dotenv()


# ---------------------------------------------------------------------------
# Given code -- look at course repo
# ---------------------------------------------------------------------------


def make_record_manager(namespace: str, db_url: str | None = None) -> SQLRecordManager:
    """Creates SQLRecordManager + schema. Provided -- do not modify."""
    # Look at course repo
    ...


def make_vectorstore(
    collection_name: str,
    embeddings: DatabricksEmbeddings | None = None,
) -> PGVector:
    """Creates PGVector collection. Provided -- do not modify."""
    # Look at course repo
    ...


def get_embeddings() -> DatabricksEmbeddings:
    """Read EMBEDDINGS env var, fall back to 'databricks-gte-large-en'."""
    # Look at course repo
    ...


def _ensure_ivfflat_index(vectorstore: PGVector) -> None:
    """Create IVFFlat index if it does not already exist."""
    import psycopg

    conn = psycopg.connect(PGVECTOR_CONNECTION_STRING.replace("+psycopg", ""))
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT indexname FROM pg_indexes
                WHERE tablename = 'langchain_pg_embedding' AND indexname = %s
                """,
                (f"{vectorstore.collection_name}_ivfflat",),
            )
            if cur.fetchone() is None:
                cur.execute(
                    """
                    ALTER TABLE langchain_pg_embedding
                    ALTER COLUMN embedding TYPE vector(1024);
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {vectorstore.collection_name}_ivfflat
                    ON langchain_pg_embedding
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                    """
                )
                conn.commit()
    finally:
        conn.close()

def file_hash(path: str) -> str:
    """16-char hex SHA-256 prefix of file bytes (rename-resilient source ID)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

# ---------------------------------------------------------------------------
# Task 2.3 functions -- implement these
# ---------------------------------------------------------------------------


def make_chunk_id(source: str, page: int, chunk_index: int) -> str:
    """16-char hex SHA-256 prefix of (source, page, chunk_index)."""
    # Concatenate source, page, and chunk_index with a separator
    # Encode the result to bytes
    # Hash with SHA-256 hashlib.sha256()
    # Return first 16 hex characters
    ...


def load_corpus(corpus_dir: str) -> list[Document]:
    """Load every PDF via PyPDFLoader, stamp 4-field metadata.

    Metadata fields: source, page, doc_type, file_hash.
    """
    # Collect all PDF paths from corpus_dir (sorted)
    # Each PDF: compute file_hash, load via PyPDFLoader
    # Each page: stamp metadata with source (filename), page number, doc_type, file_hash
    # Return flat list of Documents
    ...

def build_store(
    documents: list[Document],
    *,
    embeddings: DatabricksEmbeddings,
    collection_name: str = "baseline",
    chunk_size: int = 250,
    splitter=None,
) -> None:
    """Chunk documents, index via the record manager.

    Uses cleanup="incremental" and source_id_key="file_hash" for
    idempotent ingestion. Ensures an IVFFlat index on the collection.
    """
    # Step 1: Default splitter to RecursiveTextSplitter(chunk_size) if None
    # Step 2: Split all documents into chunks via the splitter
    # Step 3: Stamp each chunk with a deterministic id (from make_chunk_id) and chunk_index
    # Step 4: Create a PGVector collection via make_vectorstore
    # Step 5: Create a record manager via make_record_manager
    # Step 6: Index chunks via langchain_classic.indexes.index (incremental, source_id_key=file_hash)
    # Step 7: Create IVFFlat index on the collection
    ...
