# Agent Memory

Persistent, local-first memory for AI agents. One SQLite file, no Docker, no cloud DB.

Your agent remembers conversations, decisions, and facts across sessions — with intelligent recall that surfaces the right memories at the right time.

## What It Does

- **Hybrid search** — semantic vectors + BM25 keywords + recency + spaced repetition strength, fused via Reciprocal Rank Fusion
- **Write-time dedup** — new memories merge with near-duplicates (cosine > 0.85) instead of piling up
- **Spaced repetition** — frequently recalled memories strengthen; unused ones decay naturally
- **Knowledge graph** — auto-extracted entities, relationships, and temporal facts
- **Multi-agent** — each agent gets its own DB; cross-agent search with `agent=all`
- **Multilingual NLP** — Turkish + English lemmatization, temporal parsing, stopword filtering

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

## API Reference

Base URL: `http://localhost:8787` — Swagger UI at `/docs`

| Endpoint | Method | What it does |
|---|---|---|
| `/v1/recall` | POST | Hybrid search across all 4 layers |
| `/v1/store` | POST | Store one memory (with auto-merge) |
| `/v1/capture` | POST | Batch ingest messages |
| `/v1/forget` | DELETE | Delete by ID or query |
| `/v1/search` | GET | Quick search (CLI/debug) |
| `/v1/stats` | GET | DB counts and health |
| `/v1/health` | GET | Service health check |
| `/v1/decay` | POST | Apply Ebbinghaus decay to stale memories |
| `/v1/consolidate` | POST | Bulk dedup + archive weak memories |
| `/v1/agents` | GET | List all agent DBs with stats |

### Core: Store and Recall

```bash
# Store a fact
curl -X POST http://localhost:8787/v1/store \
  -H 'Content-Type: application/json' \
  -d '{"text": "Deploy backups run daily at 04:00", "category": "fact", "importance": 0.9}'

# Recall it later
curl -X POST http://localhost:8787/v1/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "when do backups run?", "limit": 5}'
```

### Batch Ingest

```bash
curl -X POST http://localhost:8787/v1/capture \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"text": "We should rotate logs weekly.", "role": "user", "session": "2026-02-08"},
      {"text": "Agreed — compress and keep 4 weeks.", "role": "assistant", "session": "2026-02-08"}
    ]
  }'
# → {"stored": 1, "merged": 1, "total": 2}
```

### Multi-Agent

Pass `agent` parameter to any endpoint to route to a specific agent's DB:

```bash
curl -X POST http://localhost:8787/v1/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "deployment status", "agent": "devops", "limit": 5}'

# Search across ALL agent DBs
curl -X POST http://localhost:8787/v1/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "deployment status", "agent": "all", "limit": 10}'
```

## How Search Works

Four scoring layers, fused with Reciprocal Rank Fusion (RRF):

| Layer | Weight | What it measures |
|---|---|---|
| Semantic | 0.40 | Embedding cosine similarity (meaning) |
| Keyword | 0.25 | BM25 full-text search (exact terms) |
| Recency | 0.15 | How recently the memory was created |
| Strength | 0.20 | Spaced repetition score (use frequency) |

Weights are configurable via environment or JSON config overlay.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `AGENT_MEMORY_DB` | `~/.agent-memory/memory.sqlite` | SQLite database path |
| `AGENT_MEMORY_MODEL` | `qwen/qwen3-embedding-8b` | Embedding model |
| `AGENT_MEMORY_DIMENSIONS` | `4096` | Vector dimensions |
| `AGENT_MEMORY_HOST` | `127.0.0.1` | Bind address |
| `AGENT_MEMORY_PORT` | `8787` | API port |
| `AGENT_MEMORY_SESSIONS_DIR` | `~/.openclaw/agents/main/sessions` | Session JSONL dir |
| `OPENROUTER_API_KEY` | — | For cloud embeddings (OpenRouter) |
| `AGENT_MEMORY_EMBEDDING_URL` | — | For local embeddings (llama-server etc.) |
| `AGENT_MEMORY_CONFIG` | — | JSON config overlay path |

### Config Overlay

Override search weights and batch size without changing code:

```json
{
  "weight_semantic": 0.40,
  "weight_keyword": 0.25,
  "weight_recency": 0.15,
  "weight_strength": 0.20,
  "batch_size": 50
}
```

## OpenClaw Integration

### Session Sync (cron)

Auto-ingest OpenClaw session transcripts:

```cron
*/30 * * * * /path/to/whatsapp-memory/scripts/cron_sync.sh
```

### Systemd Service

```ini
[Unit]
Description=Agent Memory API
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/whatsapp-memory
EnvironmentFile=/path/to/.env
ExecStart=/path/to/.venv/bin/python -m agent_memory
Restart=always

[Install]
WantedBy=multi-user.target
```

### Agent TOOLS.md Snippet

Add this to your agent's `TOOLS.md` so it knows how to recall:

```markdown
### Memory Recall
Search past conversations and decisions:
\`\`\`bash
curl -s -X POST http://127.0.0.1:8787/v1/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "<search term>", "limit": 5}'
\`\`\`
```

## Maintenance

```bash
# Weekly: decay unused memories
curl -X POST http://localhost:8787/v1/decay

# Weekly: dedup + archive stale
curl -X POST http://localhost:8787/v1/consolidate

# Backup (WAL-safe)
bash scripts/backup_db.sh

# Status check
./scripts/manage.sh status
```

## Architecture

```
FastAPI (:8787)
├── /v1/recall    → 4-layer hybrid search → RRF fusion → ranked results
├── /v1/store     → embed → find nearest → merge or insert
├── /v1/capture   → batch ingest → importance scoring → merge-or-insert
├── /v1/forget    → delete by ID or query match
├── /v1/decay     → Ebbinghaus strength decay pass
└── /v1/consolidate → cosine dedup + stale archival

Storage: SQLite + sqlite-vec (vectors) + FTS5 (keywords)
Embeddings: OpenRouter API or local llama-server
NLP: zeyrek (Turkish lemmatizer) + dateparser + stopwords
```

## License

MIT
