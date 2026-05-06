"""
Smoke tests for the RAG pipeline components.

These exercise the modular `src/*` modules used by `main.py`. They do NOT
require a running Ollama daemon — `QueryEngine` is only constructed, not
queried. End-to-end ingestion + query is covered by running `python main.py`.
"""

from src.pdf_parser import PDFParser
from src.chunker import TextChunker
from src.embedder import EmbeddingGenerator
from src.vector_store import VectorStore
from src.query_engine import QueryEngine


def test_pdf_parser_constructs():
    PDFParser()


def test_chunker_produces_chunks():
    chunker = TextChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_text("This is a test sentence. " * 20)
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)


def test_embedder_generates_correct_shape():
    embedder = EmbeddingGenerator()
    embs = embedder.generate_embeddings(["Hello world", "Test sentence"])
    assert len(embs) == 2
    assert len(embs[0]) == embedder.get_embedding_dimension()


def test_vector_store_roundtrip(tmp_path):
    store = VectorStore(
        collection_name="pytest_smoke",
        persist_directory=str(tmp_path),
    )
    embedder = EmbeddingGenerator()
    embs = embedder.generate_embeddings(["doc one", "doc two"])
    store.add_documents(
        ids=["a", "b"],
        documents=["doc one", "doc two"],
        embeddings=embs,
        metadatas=[{"src": "t"}, {"src": "t"}],
    )
    assert store.get_collection_info()["count"] == 2


def test_query_engine_constructs():
    QueryEngine()
