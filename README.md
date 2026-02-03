# Asuman Memory System

Production-ready conversational memory for Asuman — an AI assistant on OpenClaw that speaks Turkish+English via WhatsApp.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  FastAPI  (:8787)                     │
│  /v1/recall  /v1/capture  /v1/store  /v1/health      │
└──────────┬──────────┬──────────┬─────────────────────┘
           │          │          │
     ┌─────▼─────┐ ┌──▼───┐ ┌───▼───┐
     │  Hybrid   │ │Ingest│ │Entity │
     │  Search   │ │      │ │Extract│
     │(RRF Fuse) │ │JSONL │ │ KG    │
     └──┬──┬──┬──┘ └──┬───┘ └───┬───┘
        │  │  │        │         │
   ┌────▼┐ │ ┌▼────┐ ┌▼─────────▼─┐
   │Vec  │ │ │FTS5 │ │   SQLite   │
   │Srch │ │ │BM25 │ │ (storage)  │
   └──┬──┘ │ └──┬──┘ └─────┬──────┘
      │    │    │           │
      └────┴────┴───────────┘
         sqlite-vec + FTS5
         single .sqlite file

   ┌────────────┐   ┌──────────┐
   │ OpenRouter │   │  Turkish │
   │ Embeddings │   │   NLP    │
   │ qwen3-8b   │   │(zeyrek + │
   │ 4096d      │   │dateparser)│
   └────────────┘   └──────────┘
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set API key
export OPENROUTER_API_KEY="sk-or-..."

# Start API
python -m asuman_memory

# Health check
curl http://localhost:8787/v1/health
```

## Modules

| Module | Description |
|--------|-------------|
| `config.py` | Environment-based configuration |
| `embeddings.py` | OpenRouter embedding client (qwen3-embedding-8b) |
| `storage.py` | SQLite + sqlite-vec + FTS5 storage |
| `search.py` | Hybrid search with RRF fusion |
| `turkish.py` | Turkish NLP (zeyrek lemmatization, dateparser, ASCII folding) |
| `triggers.py` | Trigger patterns + importance scoring |
| `entities.py` | Knowledge graph (entity extraction) |
| `ingest.py` | Session JSONL ingestion |
| `api.py` | FastAPI HTTP API |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/recall` | Search memories (hybrid search) |
| POST | `/v1/capture` | Ingest messages |
| POST | `/v1/store` | Store a memory |
| DELETE | `/v1/forget` | Delete memory |
| GET | `/v1/search` | Interactive search |
| GET | `/v1/stats` | Statistics |
| GET | `/v1/health` | Health check |

## Initial Data Load

```bash
python scripts/initial_load.py
```

## Tests

```bash
pytest tests/ -v
```

## Configuration

Environment variables:
- `OPENROUTER_API_KEY` — required
- `ASUMAN_MEMORY_DB` — SQLite path (default: `~/.asuman/memory.sqlite`)
- `ASUMAN_MEMORY_MODEL` — embedding model (default: `qwen/qwen3-embedding-8b`)
- `ASUMAN_MEMORY_PORT` — API port (default: `8787`)
- `ASUMAN_MEMORY_DIMENSIONS` — vector dimensions (default: `4096`)

## Dependencies

< 25MB total. No torch, no transformers, no heavy ML.
