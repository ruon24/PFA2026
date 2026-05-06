"""Build a chromadb HTTP client + collection for the voice agent.

The agent and ingestion script both connect to a Dockerized ChromaDB
(typical run: `docker run -p 8000:8000 chromadb/chroma`) instead of
the PersistentClient used by src/vector_store.py.

Reads CHROMA_HOST / CHROMA_PORT from the environment; load_dotenv
must be called before invoking get_collection().
"""

from __future__ import annotations

import os

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings


def get_collection(collection_name: str = "pdf_rag") -> Collection:
    host = os.environ.get("CHROMA_HOST", "localhost")
    port = int(os.environ.get("CHROMA_PORT", "8000"))
    client = chromadb.HttpClient(
        host=host,
        port=port,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
