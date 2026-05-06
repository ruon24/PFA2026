"""Retrieval-only bridge to the Dockerized ChromaDB collection.

The voice agent uses Gemini Live as the LLM, so we deliberately bypass
src/query_engine.py (Ollama) and only expose vector search + a small
catalog helper. One KnowledgeBase instance is created per worker process;
the SentenceTransformer embedder loads its weights once at startup.

Connects to ChromaDB over HTTP — see agent/chroma_client.py.
"""

from __future__ import annotations

from typing import Any

from src.embedder import EmbeddingGenerator

from agent.chroma_client import get_collection


class KnowledgeBase:
    def __init__(
        self,
        collection_name: str = "pdf_rag",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.embedder = EmbeddingGenerator(model_name=embedding_model)
        self.collection = get_collection(collection_name)

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return up to top_k chunks most relevant to the query.

        Each result is a dict with keys: text, source, grade, distance.
        """
        embedding = self.embedder.generate_embedding(query)
        res = self.collection.query(query_embeddings=[embedding], n_results=top_k)

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out: list[dict[str, Any]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            meta = meta or {}
            out.append(
                {
                    "text": doc,
                    "source": meta.get("source"),
                    "grade": meta.get("grade"),
                    "distance": float(dist) if dist is not None else None,
                }
            )
        return out

    def list_subjects(self) -> dict[str, list[str]]:
        """Group source filenames by school grade.

        Reads metadata for every chunk in the collection (cheap — metadata
        only) and returns {grade: [source_filename, ...], ...}.
        """
        res = self.collection.get(include=["metadatas"])
        by_grade: dict[str, set[str]] = {}
        for meta in res.get("metadatas") or []:
            if not meta:
                continue
            grade = str(meta.get("grade") or "unknown")
            source = meta.get("source")
            if source:
                by_grade.setdefault(grade, set()).add(source)
        return {grade: sorted(files) for grade, files in sorted(by_grade.items())}

    def stats(self) -> dict[str, Any]:
        return {
            "collection": self.collection.name,
            "document_count": self.collection.count(),
            "embedding_dim": self.embedder.get_embedding_dimension(),
        }
