# PFA2026 — RAG Pipeline (PDF → ChromaDB → Ollama)

This file orients an AI coding assistant working in `PFA2026/`. Read it before
making changes.

## What this project is

A retrieval-augmented-generation (RAG) pipeline over a corpus of French-language
school manuals (PDFs in `data/MANUELS/{4,5,6}/`, organized by school year).
Pipeline stages:

1. **Parse** PDFs with `pdfplumber` (`src/pdf_parser.py`).
2. **Chunk** extracted text into overlapping ~500-token windows using the
   `cl100k_base` tokenizer from `tiktoken` (`src/chunker.py`).
3. **Embed** chunks with `sentence-transformers/all-MiniLM-L6-v2`
   (`src/embedder.py`).
4. **Store** embeddings in a persistent ChromaDB collection at `./chroma_db/`
   with cosine distance (`src/vector_store.py`).
5. **Query** the collection by embedding the question, retrieving top-k chunks,
   and asking a local Ollama model (default `llama3.2`) to answer with the
   chunks as context (`src/query_engine.py`).

`src/pipeline.py` (`Pipeline` class) wires the five stages together. `main.py`
is the demo entry point.

## Layout

```
PFA2026/
├── data/
│   └── MANUELS/{4,5,6}/*.pdf      # Source corpus (NOT loose at data/ root)
├── src/
│   ├── __init__.py
│   ├── pdf_parser.py              # PDFParser — pdfplumber wrapper
│   ├── chunker.py                 # TextChunker — token windows w/ overlap
│   ├── embedder.py                # EmbeddingGenerator — sentence-transformers
│   ├── vector_store.py            # VectorStore — ChromaDB persistent client
│   ├── query_engine.py            # QueryEngine — ollama.generate / ollama.chat
│   └── pipeline.py                # Pipeline — orchestrates the five stages
├── main.py                        # Demo: ingest data/ then run two queries
├── rag_pipeline.py                # ⚠ Legacy monolith — see "Known issues"
├── test_pipeline.py               # ⚠ Imports rag_pipeline.py (legacy) — see below
├── requirements.txt
├── README.md                      # ⚠ Setup snippet is Windows-only
└── chroma_db/                     # Created at first run; gitignored
```

## Toolchain & runtime

- **Python**: project requires `tiktoken`, `sentence-transformers`, `chromadb`,
  `pdfplumber`, `pypdf`, `ollama`. See `requirements.txt`.
- **External dependency**: a running Ollama daemon with the configured model
  pulled. Before queries:
  ```bash
  ollama serve            # in a background shell
  ollama pull llama3.2
  ```
  `QueryEngine.check_connection()` is the canary — `Pipeline.query()` returns
  `{"status":"error", ...}` if Ollama is unreachable.
- **Lint / test**: project memory says `lint=ruff check`, `test=pytest`. There
  are no real pytest tests yet; `test_pipeline.py` is a smoke script.
- **Setup (macOS, current host)**:
  ```bash
  python -m venv venv
  source venv/bin/activate          # not the venv\Scripts\activate from README
  pip install -r requirements.txt
  python main.py
  ```

## Configuration knobs

`Pipeline.__init__` takes:
- `collection_name="pdf_rag"`
- `chunk_size=500` (tokens)
- `chunk_overlap=50` (tokens)
- `embedding_model="all-MiniLM-L6-v2"` (384-dim)
- `ollama_model="llama3.2"`

`VectorStore` persists at `./chroma_db/` (path hardcoded — change here if you
want a different location).

`QueryEngine.generate()` uses `temperature=0.3, top_p=0.9, num_ctx=4096`.

## CLI

`src/pipeline.py` has its own `argparse` entry point, separate from `main.py`:

```bash
python -m src.pipeline --pdfs ./data --query "..." --stats
python -m src.pipeline --reset               # drops the collection
```

`main.py` is the simpler scripted demo (ingests `./data` then runs two fixed
queries).

## ⚠ Known issues (review surfaced these — do NOT silently "fix" without asking)

1. **PDF discovery is non-recursive.** `Pipeline.ingest_pdfs` and
   `RAGPipeline.ingest_pdfs` glob `*.pdf` at the top level only. The corpus
   lives under `data/MANUELS/{4,5,6}/`, so the demo currently finds **0 files**.
   Either change the user invocation to pass each subfolder, or change the glob
   to `**/*.pdf` with `recursive=True` (and also match `*.PDF`).
2. **Re-ingestion will throw on duplicate IDs.** ChromaDB's `collection.add`
   raises if an `id` already exists. IDs are deterministic
   (`{file_id}_chunk_{i}`), so the second run on the same corpus crashes.
   Switch `VectorStore.add_documents` to `upsert`, or de-dupe upfront.
3. **`rag_pipeline.py` is a stale monolith.** It duplicates `src/` with subtle
   differences:
   - Different `query()` return type (`str` vs `dict`).
   - No `max(1, …)` guard on the chunker step (raises if
     `chunk_size == overlap`).
   - Hardcoded `./pdfs` path in its `__main__`.
   `test_pipeline.py` imports from `rag_pipeline`, so the "tests" exercise the
   legacy code path, not the one `main.py` runs. Likely action: delete
   `rag_pipeline.py`, port `test_pipeline.py` to import from `src.*`.
4. **`chunk_text_by_sentences` in `src/chunker.py` is unused and buggy** (naive
   `". "` split, swallows trailing periods, ignores `self.overlap`). Either
   remove it or rewrite before calling.
5. **Empty / scanned PDFs are not handled.** If `pdfplumber` returns `None` for
   every page, the pipeline tries to add zero chunks/embeddings to ChromaDB.
   Add an early-skip for empty extractions.
6. **README setup is Windows-only** (`venv\Scripts\activate`); current host is
   macOS.

## Conventions

- All new code goes under `src/`. Treat `rag_pipeline.py` as legacy.
- Keep `Pipeline.query()` returning a **dict** with at least
  `status, answer, sources` — `main.py` and any future API depend on this shape.
- Keep `VectorStore` the only thing that touches ChromaDB. Don't sprinkle
  `chromadb` calls elsewhere.
- Keep `QueryEngine` the only thing that touches `ollama`. Same reason.
- The chunker and embedder must agree on units. Today the chunker measures in
  `cl100k_base` tokens (OpenAI's tokenizer) but the embedder is MiniLM (its own
  WordPiece tokenizer with a 256-token cap). 500 cl100k tokens ≈ usually fits
  MiniLM, but long French manual passages may be truncated by the embedder
  silently. If you change `chunk_size`, sanity-check against MiniLM's limit.

## When you change something, verify

- Re-run `python main.py` from a clean `chroma_db/` (delete the folder, or
  `python -m src.pipeline --reset`).
- Confirm `Pipeline.get_stats()` reports `document_count > 0` and
  `ollama_connected: True`.
- Run `ruff check .` before claiming done.
