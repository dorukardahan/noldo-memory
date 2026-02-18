# Agent Memory

Persistent, local-first memory for AI agents. One SQLite file, no Docker, no cloud DB.

Your agent remembers conversations, decisions, and facts across sessions with hybrid recall.

## Features (v0.4.0+)

- Hybrid search (semantic + BM25 + recency + spaced-repetition strength) with RRF fusion
- Write-time dedup (near-duplicates merge instead of piling up)
- Instruction auto-capture (v0.3.0)
- Knowledge-graph conflict detection (v0.3.0)
- Result + embedding caching with write invalidation (v0.3.0)
- Typed relation patterns (v0.4.0)
- Multi-agent session sync (v0.4.0)
- Aggressive Ebbinghaus decay (v0.4.0)
- Multilingual NLP (Turkish + English)

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

python -m agent_memory
# → API running at http://localhost:8787
```

Verify:
```bash
curl http://localhost:8787/v1/health
```

## API

Base URL: `http://localhost:8787` (`/docs` for Swagger)

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/recall` | POST | Hybrid recall |
| `/v1/store` | POST | Store one memory (auto-merge) |
| `/v1/capture` | POST | Batch ingest |
| `/v1/rule` | POST | Explicit instruction store |
| `/v1/forget` | DELETE | Delete by ID/query |
| `/v1/search` | GET | Quick search |
| `/v1/decay` | POST | Apply decay |
| `/v1/consolidate` | POST | Dedup + archive weak memories |
| `/v1/agents` | GET | List agent DBs |
| `/v1/stats` | GET | DB stats |
| `/v1/health` | GET | Health check |
| `/v1/metrics` | GET | Operational metrics |

## Config

| Variable | Default |
|---|---|
| `AGENT_MEMORY_DB` | `~/.agent-memory/memory.sqlite` |
| `AGENT_MEMORY_MODEL` | `qwen/qwen3-embedding-8b` |
| `AGENT_MEMORY_DIMENSIONS` | `4096` |
| `AGENT_MEMORY_HOST` | `127.0.0.1` |
| `AGENT_MEMORY_PORT` | `8787` |
| `AGENT_MEMORY_SESSIONS_DIR` | `~/.openclaw/agents/main/sessions` |
| `OPENROUTER_API_KEY` | — |
| `AGENT_MEMORY_EMBEDDING_URL` | — |
| `AGENT_MEMORY_CONFIG` | — |


## Authentication (v0.3.0+)

All endpoints except `/v1/health` require an API key:

```bash
# Set in environment
export AGENT_MEMORY_API_KEY="your-secret-key"

# Include in requests
curl -H "X-API-Key: $AGENT_MEMORY_API_KEY" http://localhost:8787/v1/stats
```

Generate a key: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`

Health endpoint is always public (for monitoring).


### Export & Import

```bash
# Export all memories for an agent
curl -s -H "X-API-Key: $KEY" \
  "http://localhost:8787/v1/export?agent=main" > memories.json

# Import memories
curl -X POST -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"memories": [...], "agent": "main"}' \
  http://localhost:8787/v1/import
```

## Security

- API key authentication on all write/read endpoints
- Rate limiting: 120 requests/minute per IP
- CORS restricted to localhost
- Audit logging to `/var/log/asuman-memory-audit.log`
- SQLite files: `chmod 600` (owner-only)
## License

MIT
