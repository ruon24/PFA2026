"""One-time ingestion: walk data/MANUELS/ recursively and populate ChromaDB.

Connects to a Dockerized ChromaDB over HTTP. Start the server first:
    docker run -p 8000:8000 chromadb/chroma

Then run from the PFA2026/ directory:
    uv run python -m agent.ingest
    # or with a custom corpus root:
    uv run python -m agent.ingest --pdfs ./data/MANUELS

We attach a `grade` metadata field (parent dir name 4/5/6) for filtering
and for the agent's `list_available_subjects` tool.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.chunker import TextChunker
from src.embedder import EmbeddingGenerator
from src.pdf_parser import PDFParser

# Load env BEFORE importing chroma_client so CHROMA_HOST/PORT are picked up.
load_dotenv(Path(__file__).resolve().parent / ".env.local")
load_dotenv(Path(__file__).resolve().parent / ".env")

from agent.chroma_client import get_collection  # noqa: E402


def ingest(pdf_root: Path, collection_name: str) -> int:
    parser = PDFParser()
    chunker = TextChunker(chunk_size=500, overlap=50)
    embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    collection = get_collection(collection_name)

    pdfs = sorted(pdf_root.rglob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found under {pdf_root}", file=sys.stderr)
        return 1

    total_chunks = 0
    for idx, pdf in enumerate(pdfs, 1):
        grade = pdf.parent.name
        print(f"[{idx}/{len(pdfs)}] {grade}/{pdf.name}")

        try:
            text = parser.extract_text(str(pdf))
        except Exception as exc:
            print(f"  parse failed: {exc}")
            continue
        if not text.strip():
            print("  empty PDF, skipping")
            continue

        chunks = chunker.chunk_text(text)
        if not chunks:
            print("  no chunks, skipping")
            continue

        embeddings = embedder.generate_embeddings(chunks)
        file_id = pdf.stem
        ids = [f"{file_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": pdf.name,
                "grade": grade,
                "file_id": file_id,
                "chunk_id": i,
            }
            for i in range(len(chunks))
        ]
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)
        print(f"  upserted {len(chunks)} chunks")

    print(f"\nDone: {len(pdfs)} files, {total_chunks} chunks")
    print(f"Collection '{collection_name}' now has {collection.count()} documents")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Recursively ingest PDFs into ChromaDB.")
    parser.add_argument("--pdfs", default="./data/MANUELS", help="Root folder for recursive PDF scan")
    parser.add_argument("--collection", default="pdf_rag", help="Collection name")
    args = parser.parse_args()

    return ingest(Path(args.pdfs), args.collection)


if __name__ == "__main__":
    raise SystemExit(main())
