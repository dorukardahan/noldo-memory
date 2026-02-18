# Agent Memory

Persistent, local-first memory for AI agents. One SQLite file, no Docker, no cloud DB.

Your agent remembers conversations, decisions, and facts across sessions with hybrid recall.

## Features (v0.3.0)

- **Hybrid search** — semantic + BM25 + recency + spaced-repetition strength, RRF fusion
- **Write-time dedup** — near-duplicates merge instead of piling up
- **Per-agent routing** — isolated memory DBs per agent (`?agent=bureau`)
- **API key auth + rate limiting** — secure by default
- **Export/Import** — JSONL-compatible backup and restore
- **Instruction auto-capture** — detects and prioritizes rules/instructions
- **Knowledge graph** — entity extraction, typed relations, conflict detection
- **Category-aware consolidation** — semantic dedup with adaptive thresholds
- **Ebbinghaus decay** — strength-based memory aging
- **Multilingual NLP** — Turkish + English

## Quick Start

```bash
git clone https://github.com/dorukardahan/asuman-memory.git
cd whatsapp-memory
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Embeddings — pick one:
# Option A: OpenRouter (cloud, no GPU needed)
export OPENROUTER_API_KEY="sk-or-..."

# Option B: Local llama-server (no API key needed)
export AGENT_MEMORY_EMBEDDING_URL="http://localhost:8090/v1/embeddings"
export AGENT_MEMORY_MODEL="local-model"

# API key (required for all endpoints except /v1/health)
export AGENT_MEMORY_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"

python -m agent_memory
# → API running at http://localhost:8787
```

Verify:
```bash
curl http://localhost:8787/v1/health
curl -H "X-API-Key: $AGENT_MEMORY_API_KEY" http://localhost:8787/v1/stats
```

### Per-Agent Routing

All endpoints accept an `agent` query parameter:

```bash
# Store to a specific agent's memory
curl -X POST -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"text": "Deploy completed", "agent": "devops"}' \
  http://localhost:8787/v1/store

# Search across all agents
curl -X POST -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query": "deployment", "agent": "all"}' \
  http://localhost:8787/v1/recall
```

## API

Base URL: `http://localhost:8787` (`/docs` for Swagger)

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/recall` | POST | Hybrid recall (semantic + BM25 + recency) |
| `/v1/store` | POST | Store one memory (auto-merge) |
| `/v1/capture` | POST | Batch ingest from session messages |
| `/v1/rule` | POST | Explicit instruction store |
| `/v1/forget` | DELETE | Delete by ID or query |
| `/v1/search` | GET | Quick search |
| `/v1/decay` | POST | Apply Ebbinghaus strength decay |
| `/v1/consolidate` | POST | Dedup + archive weak memories |
| `/v1/export` | GET | Export memories as JSON |
| `/v1/import` | POST | Import memories from JSON |
| `/v1/agents` | GET | List agent DBs |
| `/v1/stats` | GET | DB statistics |
| `/v1/metrics` | GET | Operational metrics |
| `/v1/health` | GET | Health check (public) |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `AGENT_MEMORY_API_KEY` | — | **Required.** API key for authentication |
| `OPENROUTER_API_KEY` | — | OpenRouter key (for cloud embeddings) |
| `AGENT_MEMORY_EMBEDDING_URL` | — | Local embedding server URL |
| `AGENT_MEMORY_DB` | `~/.asuman/memory.sqlite` | SQLite database path |
| `AGENT_MEMORY_MODEL` | `qwen/qwen3-embedding-8b` | Embedding model name |
| `AGENT_MEMORY_DIMENSIONS` | `4096` | Embedding dimensions |
| `AGENT_MEMORY_HOST` | `127.0.0.1` | API bind address |
| `AGENT_MEMORY_PORT` | `8787` | API port |

## Export & Import

```bash
# Export all memories for an agent
curl -s -H "X-API-Key: $KEY" \
  "http://localhost:8787/v1/export?agent=main" > memories.json

# Export including soft-deleted
curl -s -H "X-API-Key: $KEY" \
  "http://localhost:8787/v1/export?agent=main&include_deleted=true" > full-backup.json

# Import memories (embeds automatically, skips duplicates by ID)
curl -X POST -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"memories": [...], "agent": "main", "skip_duplicates": true}' \
  http://localhost:8787/v1/import
```

## Security

- **Authentication**: API key via `X-API-Key` header (all endpoints except `/v1/health`)
- **Rate limiting**: 120 requests/minute per IP (in-memory sliding window)
- **CORS**: Restricted to localhost origins
- **Audit logging**: Structured request log to `/var/log/asuman-memory-audit.log`
- **File permissions**: SQLite files `chmod 600` (owner-only)
- **Centralized error handling**: Structured JSON error responses, no stack traces exposed

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Lint
pip install ruff
ruff check agent_memory/ tests/
```

CI runs on every push/PR to `main` (GitHub Actions: lint + 151 tests).

## License

MIT
