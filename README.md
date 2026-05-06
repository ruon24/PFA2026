# PFA26 - RAG Pipeline
# PDF → Parse → Chunk → Embed → ChromaDB → Ollama

## Project Structure
```
PFA26/
├── venv/                  # Virtual environment
├── data/                  # Input PDFs (discovered recursively)
│   └── MANUELS/{4,5,6}/   # Current corpus: school-year manuals
├── chroma_db/             # ChromaDB persistence (gitignored)
├── src/
│   ├── __init__.py
│   ├── pdf_parser.py      # PDF text extraction
│   ├── chunker.py         # Text chunking
│   ├── embedder.py        # Embedding generation
│   ├── vector_store.py    # ChromaDB storage
│   ├── query_engine.py    # Ollama query
│   └── pipeline.py        # Main pipeline
├── main.py                # Entry point
├── test_pipeline.py       # pytest smoke tests
├── requirements.txt       # Dependencies
├── CLAUDE.md              # Notes for AI assistants
└── README.md
```

PDFs anywhere under `data/` are discovered recursively (any depth, `.pdf`
or `.PDF`). Place new manuals wherever — no top-level flatten required.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed locally with the configured model.
  Before running queries:
  ```bash
  ollama serve              # in a background shell, leave running
  ollama pull llama3.2      # one-time
  ```

## Setup

1. Create a virtualenv: `python -m venv venv`
2. Activate it:
   ```bash
   source venv/bin/activate     # macOS / Linux
   venv\Scripts\activate        # Windows
   ```
3. Install deps: `pip install -r requirements.txt`
4. Run: `python main.py`

The first run downloads the `all-MiniLM-L6-v2` sentence-transformers model
(~90 MB) and embeds every chunk. Re-runs are idempotent (ChromaDB upsert)
but will re-embed; delete `chroma_db/` for a fully clean state.

## Tests

```bash
pytest -q
```

The smoke tests don't require a running Ollama daemon (only the `main.py`
end-to-end run does).
