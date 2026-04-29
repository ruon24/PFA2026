# PFA26 - RAG Pipeline
# PDF → Parse → Chunk → Embed → ChromaDB → Ollama

## Project Structure
```
PFA26/
├── venv/                  # Virtual environment
├── data/                  # Input PDFs
├── chroma_db/             # ChromaDB persistence
├── src/
│   ├── __init__.py
│   ├── pdf_parser.py      # PDF text extraction
│   ├── chunker.py         # Text chunking
│   ├── embedder.py        # Embedding generation
│   ├── vector_store.py    # ChromaDB storage
│   ├── query_engine.py    # Ollama query
│   └── pipeline.py        # Main pipeline
├── main.py                # Entry point
├── requirements.txt       # Dependencies
└── README.md
```

## Setup Steps
1. Create venv: `python -m venv venv`
2. Activate: `venv\Scripts\activate`
3. Install deps: `pip install -r requirements.txt`
4. Run: `python main.py`