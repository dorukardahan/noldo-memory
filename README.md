# Agent Memory

Persistent, local-first memory for AI agents. One SQLite file, no Docker, no cloud DB.

Your agent remembers conversations, decisions, and facts across sessions — with intelligent recall that surfaces the right memories at the right time.

## What It Does

- **Hybrid search** — semantic vectors + BM25 keywords + recency + spaced repetition strength, fused via Reciprocal Rank Fusion
- **Write-time dedup** — new memories merge with near-duplicates (cosine > 0.85) instead of piling up
- **Spaced repetition** — frequently recalled memories strengthen; unused ones decay naturally
- **Knowledge graph** — auto-extracted entities, relationships, and temporal facts
- **Multi-agent** — each agent gets its own DB; cross-agent search with `agent=all`
- **Instruction capture** — auto-detects rules/preferences in conversation ("always use dark mode", "bundan sonra Türkçe yaz") and stores them as high-importance memories. Safewords: 📌 💾 `/rule` `/save` `/remember`
- **Conflict detection** — when a new fact contradicts an existing one (same entity + relation), flags it and auto-resolves (newest wins) or marks for review
- **Result caching** — SQLite-backed search cache + embedding cache with eager invalidation on writes
- **Multilingual NLP** — Turkish + English lemmatization, temporal parsing, stopword filtering

## Why Not Just Use Files?

Most agent frameworks (OpenClaw, Claude Code, etc.) ship with a built-in memory layer — typically keyword search over markdown files like `MEMORY.md`. That works for curated notes you write by hand, but it has limits:

| | File-based memory | Agent Memory (this) |
|---|---|---|
| Search | Keyword / text match | Semantic + keyword + recency + strength |
| Data | Hand-written notes | Auto-ingested conversations |
| Dedup | Manual | Automatic (cosine merge) |
| Recall quality | Exact match or miss | Finds related context even with different wording |
| Maintenance | You prune files | Spaced repetition + decay handle it |
| Multi-agent | Shared files or nothing | Per-agent DBs with cross-search |

**The two layers complement each other.** File-based memory holds curated, high-signal notes. Agent Memory holds everything — the full conversation history with intelligent retrieval.

### Dual Memory Architecture (OpenClaw example)

```
Your Agent
├── Built-in memory (MEMORY.md, memory/*.md)
│   └── Keyword search over hand-written notes
│   └── "What did I explicitly decide?"
│
├── Agent Memory API (this repo, port 8787)
│   └── 4-layer hybrid search over all conversations
│   └── "What was discussed 3 weeks ago about deployments?"
│
└── Hooks (glue between them)
    ├── bootstrap-context  → pull recent memories at session start
    ├── pre-session-save   → snapshot context before session ends
    ├── post-compaction    → restore context after token compaction
    └── session-bridge     → track topics across sessions
```

You don't need to pick one. Run both — Agent Memory catches what you'd never think to write down.

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
| `/v1/rule` | POST | Explicitly store a rule/instruction |
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

### Instruction Capture

Rules and preferences are detected automatically during `/v1/store` and `/v1/capture`. You can also store them explicitly:

```bash
# Explicit rule
curl -X POST http://localhost:8787/v1/rule \
  -H 'Content-Type: application/json' \
  -d '{"text": "Always reply in Turkish when I write in Turkish"}'

# Auto-detected during normal store (no extra call needed)
curl -X POST http://localhost:8787/v1/store \
  -H 'Content-Type: application/json' \
  -d '{"text": "From now on use dark mode in all code snippets"}'
# → detected as instruction, stored with importance=1.0 and category="instruction"
```

Trigger patterns: "always ...", "never ...", "from now on ...", "bundan sonra ...", "her zaman ...", "asla ..."
Safewords (force capture): 📌 💾 `/rule` `/save` `/remember`

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

## Hooks (OpenClaw Integration)

Hooks let the memory API plug into your agent's lifecycle automatically. No manual recall needed — the agent gets context injected at the right moments.

### bootstrap-context

Runs at session start. Pulls recent memories and injects them as context so the agent "remembers" without being asked.

```js
// hooks/bootstrap-context/handler.js
// Derives agent ID from workspace path, fetches /v1/recall
// Injects top memories into session bootstrap
```

### pre-session-save

Runs before a session ends or is compacted. Stores a snapshot of the current context so nothing is lost.

### post-compaction-restore

After token compaction (when context gets too long and is summarized), this hook restores critical context that the compaction summary might have dropped.

### session-memory-bridge

Tracks active topics across sessions. If you discussed "deployment pipeline" in session A, session B will know it's a hot topic even without explicit recall.

### Hook Setup (OpenClaw)

Add hook directories to your config:

```json
{
  "hooks": {
    "internal": {
      "load": {
        "extraDirs": [
          "/path/to/hooks/bootstrap-context",
          "/path/to/hooks/pre-session-save",
          "/path/to/hooks/post-compaction-restore",
          "/path/to/hooks/session-memory-bridge"
        ]
      }
    }
  }
}
```

Example hook implementations are in the `hooks/` directory (coming soon — currently deployed but not yet in repo).

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

## Embedding Models

You need an embedding model for semantic search. Two options:

### Cloud (OpenRouter)

Zero setup, works on any machine. ~$0.01/1M tokens.

```bash
export OPENROUTER_API_KEY="sk-or-..."
export AGENT_MEMORY_MODEL="qwen/qwen3-embedding-8b"
export AGENT_MEMORY_DIMENSIONS=4096
```

### Local (llama-server)

No API costs, no network dependency. Needs ~4GB RAM for a quantized 4B model.

```bash
# Download a quantized embedding model (example: Qwen3-Embedding-4B Q8)
llama-server -m qwen3-embedding-4b-q8.gguf \
  --port 8090 --embedding --n-gpu-layers 0

# Point Agent Memory at it
export AGENT_MEMORY_EMBEDDING_URL="http://localhost:8090/v1/embeddings"
export AGENT_MEMORY_MODEL="local"
export AGENT_MEMORY_DIMENSIONS=2560  # depends on your model
```

**Choosing a model:**

| Model | RAM | Dimensions | Speed | Quality |
|---|---|---|---|---|
| Qwen3-Embedding-0.6B (Q8) | ~1 GB | 1024 | Fast | Good for small DBs |
| Qwen3-Embedding-4B (Q8) | ~4 GB | 2560 | Moderate | Good balance |
| Qwen3-Embedding-8B (cloud) | — | 4096 | Fast (API) | Best quality |

If you switch models, re-embed your existing memories:
```bash
python scripts/reindex_embeddings.py
```

## Deployment

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

### Session Sync (cron)

Auto-ingest OpenClaw session transcripts:

```cron
*/30 * * * * /path/to/whatsapp-memory/scripts/cron_sync.sh
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
├── /v1/recall    → cache check → 4-layer hybrid search → RRF fusion → cache store → results
├── /v1/store     → rule detection → embed → conflict check → merge or insert → invalidate cache
├── /v1/capture   → batch ingest → rule detection → importance scoring → merge-or-insert
├── /v1/rule      → explicit instruction store (importance=1.0)
├── /v1/forget    → delete by ID or query match → invalidate cache
├── /v1/decay     → Ebbinghaus strength decay pass
└── /v1/consolidate → cosine dedup + stale archival

Storage: SQLite + sqlite-vec (vectors) + FTS5 (keywords) + search/embedding cache tables
Embeddings: OpenRouter API or local llama-server (with warm-up on startup)
NLP: zeyrek (Turkish lemmatizer) + dateparser + stopwords
KG: Entity extraction + relation-scoped conflict detection
```

## License

MIT
