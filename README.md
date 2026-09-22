# Retriva — Local Financial AI Assistant

Retriva is a fully local, agentic RAG assistant for financial documents
(SEC 10-K / 10-Q filings, their tables) plus any PDFs users upload.

Nothing leaves the machine: the LLM, embeddings and reranker all run from the
local HuggingFace cache — no API keys, no cloud calls.

## What it does

| Requirement | Where it lives |
|---|---|
| Per-chat long-term memory | `RetrivaEngine.memory_add` / `memory_search` (Chroma collection `retriva_chat_memory`, filtered by `chat_id`) |
| Chats are isolated from each other | every memory read is scoped to the current `chat_id` |
| User profile / bio (shared across chats) | `User.bio` + `GET/PUT /api/profile`, injected into every prompt |
| Finance data available to every user | shared Chroma collection `retriva_financial_docs` (11k+ chunks) |
| User PDF uploads are ingested & queryable | `RetrivaEngine.ingest_pdf` + `POST /api/ingest` |
| Token streaming | `RetrivaEngine.stream_query` (`TextIteratorStreamer`) + SSE in `api_server.py` |
| Normal chatbot behaviour | `answer_directly` agent tool |
| RAG as a tool + hybrid retrieval | `retrieve_documents` tool = dense vector + BM25 fused with RRF, then a cross-encoder reranker |
| Agentic RAG | `_plan` → tool selection → targeted table lookup → confidence check |
| Clarification feedback | `ask_clarification` tool (low confidence) |
| 100% local | `HF_LOCAL_ONLY` forces offline mode |

## Architecture

```
User ──▶ React UI (frontend/retriva)
          │  POST /api/stream (SSE)
          ▼
      FastAPI (api_server.py)
          │  Auth + chats + messages (SQLite: retriva.db)
          ▼
      RetrivaEngine (app/rag_engine.py)
          │
          ├── Agent planner (_plan)
          │     ├─ retrieve_documents ─▶ hybrid retrieve ─▶ rerank ─▶ targeted table lookup
          │     ├─ answer_directly    ─▶ Qwen2.5-1.5B-Instruct (local)
          │     └─ ask_clarification
          │
          ├── ChromaDB: retriva_financial_docs (shared knowledge base)
          └── ChromaDB: retriva_chat_memory   (memory scoped to each chat_id)
```

## Setup

```powershell
# 1. Python deps (a venv such as .\retriva is already provided)
.\retriva\Scripts\python.exe -m pip install -r requirements.txt

# 2. (Re)build the knowledge base from data/finance
$env:CLEAR_DB="1"; .\retriva\Scripts\python.exe ingest.py   # full rebuild
.\retriva\Scripts\python.exe ingest.py                      # incremental

# 3. Run the API
.\retriva\Scripts\python.exe api_server.py                  # http://localhost:8000

# 4. Run the UI
cd frontend\retriva; npm install; npm run dev               # http://localhost:5173

# Optional: CLI mode
.\retriva\Scripts\python.exe main.py
```

Models are expected in the local HF cache:

* `Qwen/Qwen2.5-1.5B-Instruct`
* `sentence-transformers/all-MiniLM-L6-v2`
* `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Configuration

All settings live in `app/config.py` and can be overridden with environment
variables (e.g. `MAX_NEW_TOKENS`, `TOP_N_RERANKED`, `CLARIFICATION_THRESHOLD`,
`RAG_CONFIDENCE_THRESHOLD`, `SHARE_KNOWLEDGE_ACROSS_USERS`).

Conversation memory is always scoped per chat (`chat_id`) and cannot be
changed into cross-chat sharing. The user's bio is the only cross-chat context.

> Note: on a CPU-only machine generation is slow. Streaming makes the UI usable;
> lower `MAX_NEW_TOKENS` for faster replies.
