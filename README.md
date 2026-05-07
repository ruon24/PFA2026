# PFA2026 — Tunisian Derja Voice Tutor

A real-time voice tutor for Tunisian primary school students (grades 4–6)
that listens and speaks in **Tunisian Derja**, retrieves answers from the
official school manuals, and explains them in kid-friendly terms.

## Architecture

```
[Browser: agent-starter-react]  ←WebRTC→  [LiveKit server (Docker)]
                                                     │ joins room
                                                     ▼
                                         [Python worker — agent/agent.py]
                                                     │
                                                     ├─ Gemini Live (native audio)
                                                     │   model: gemini-2.5-flash-native-audio-preview-12-2025
                                                     │   voice: Aoede
                                                     │
                                                     └─ Function tools
                                                         • search_knowledge_base()
                                                         • list_available_subjects()
                                                              │
                                                              ▼
                                                    [ChromaDB (Docker)]
                                                    [SentenceTransformer all-MiniLM-L6-v2]
```

Two parts share this repo:

- **`src/`** — original RAG pipeline (Ollama-based, standalone CLI).
- **`agent/`** — voice tutor on top of the same vector store, served via
  LiveKit Agents and Gemini Live.

## Layout

```
PFA2026/
├── data/MANUELS/{4,5,6}/     # Source PDFs (recursive, French)
├── src/                      # PDF → chunk → embed → ChromaDB → Ollama
├── agent/                    # LiveKit voice tutor (Gemini Live)
│   ├── agent.py              # Worker entry point
│   ├── kb.py                 # Retrieval-only bridge to ChromaDB (HTTP)
│   ├── ingest.py             # Recursive PDF ingestion w/ grade metadata
│   ├── chroma_client.py      # HTTP client builder for Dockerized Chroma
│   ├── prompts.py            # Derja system prompt
│   ├── mint_token.py         # JWT minter for the LiveKit Playground
│   └── .env.example          # Copy to .env or .env.local
├── docker-compose.yml        # Chroma + LiveKit dev stack
├── requirements.txt
└── main.py                   # Standalone Ollama RAG demo (legacy)
```

The voice agent **does not use Ollama** — Gemini synthesizes the Derja
answer directly from retrieved chunks.

## Prerequisites

- **Python 3.10+** and [`uv`](https://github.com/astral-sh/uv).
- **Docker** (for Chroma + LiveKit).
- **Node.js 20+** and **pnpm** (for the React frontend).
- **Google AI Studio API key** for Gemini Live —
  [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).
- Corpus PDFs under `data/MANUELS/{4,5,6}/` (already in repo for this project).

## Setup

### 1. Install Python deps

```bash
cd PFA2026/
uv venv
uv pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp agent/.env.example agent/.env
# edit agent/.env and paste your GOOGLE_API_KEY
```

The defaults `LIVEKIT_URL=ws://localhost:7880`, `LIVEKIT_API_KEY=devkey`,
`LIVEKIT_API_SECRET=secret` match the `--dev` mode in the compose file.

### 3. Bring up the dev stack

```bash
docker compose up -d
```

Starts:
- `chroma` on `:8000` (named volume `chroma_data` for persistence).
- `livekit-dev` on `:7880` / `:7881` / `:7882`/udp (image
  `livekit/livekit-server:master` — the released `latest` is too old for
  the React SDK's protocol version).

Verify with `docker compose ps`. Tail logs with
`docker compose logs -f livekit`.

### 4. Ingest the manuals (one-time, ~5–15 min)

```bash
uv run python -m agent.ingest
```

Walks `data/MANUELS/` recursively, chunks each PDF (500 tokens, 50
overlap), embeds with `all-MiniLM-L6-v2`, and upserts into
`pdf_rag`. Idempotent — safe to re-run after editing PDFs. Custom
corpus root: `--pdfs ./path/to/manuals`.

After it finishes you should see roughly:
```
Done: 24 files, 7840 chunks
Collection 'pdf_rag' now has 7840 documents
```

### 5. Start the voice worker

```bash
uv run python -m agent.agent dev
```

Look for `registered worker {"agent_name":"derja-tutor",...}`. The
worker connects to LiveKit and idles waiting for room dispatch.

### 6. Start the frontend

The frontend lives in a sibling repo (`agent-starter-react` from the
LiveKit examples). One-time setup:

```bash
git clone https://github.com/livekit-examples/agent-starter-react.git
cd agent-starter-react
pnpm install
cp .env.example .env.local
```

Edit `agent-starter-react/.env.local`:
```
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
AGENT_NAME=derja-tutor
```

`AGENT_NAME` is required — without it the token won't carry a
`RoomAgentDispatch` for our worker and no agent joins.

Then:
```bash
pnpm dev
```

Open **http://localhost:3000** → click **Start call** → grant mic →
the agent should greet in Derja within a couple seconds.

## Daily commands

```bash
# Stack
docker compose up -d                 # start chroma + livekit
docker compose down                  # stop, keep data
docker compose down -v               # stop, wipe chroma volume (re-ingest required)
docker compose logs -f livekit       # tail one service

# Worker (from PFA2026/)
uv run python -m agent.agent dev     # dev mode
uv run python -m agent.agent start   # production mode

# Frontend (from agent-starter-react/)
pnpm dev                             # http://localhost:3000

# One-off
uv run python -m agent.ingest        # re-ingest after corpus changes
uv run python -m agent.mint_token    # JWT for the hosted Playground (optional)
```

## Known limits

- **Gemini Live native-audio sessions cap at 15 min** — the call ends
  there. Refresh the page to start a new one.
- **Tunisian Derja support is not officially documented** in Gemini.
  Quality depends entirely on the system prompt in `agent/prompts.py`.
  Iterate there, not on the model.
- **Manuals are in French**, the agent must reply in Derja. The system
  prompt instructs the model to translate retrieved chunks before
  speaking. If you see French leaking into responses, tighten section 1
  of `prompts.py`.

## Standalone Ollama RAG (legacy)

The original CLI pipeline is still here, independent of the voice
stack. Requires a running Ollama:

```bash
ollama serve
ollama pull llama3.2
uv run python main.py
```

See `CLAUDE.md` for the full set of architectural notes and known
issues in `src/`.

## Tests

```bash
uv run pytest -q
```

The smoke tests don't require Ollama, ChromaDB, or LiveKit — only the
end-to-end demos do.
