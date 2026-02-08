# OpenClaw Memory — Persistent Memory for OpenClaw Agents

A local-first conversational memory system that gives OpenClaw agents long-term recall across sessions.

- **Single SQLite file** (portable, backup-friendly)
- **Hybrid search** for recall (semantic + keyword + recency + strength)
- **Zero cloud dependency** for storage and retrieval (embeddings are the only external call)

## Key Features

- **4-layer hybrid search**: semantic (sqlite-vec) + keyword (FTS5 BM25) + recency + Ebbinghaus strength
- **Reciprocal Rank Fusion (RRF)** for merging results from multiple layers
- **Write-time semantic merge** (cosine similarity > 0.85 ⇒ merge instead of duplicate)
- **Ebbinghaus spaced repetition** (frequently retrieved memories strengthen, unused ones decay)
- **Consolidation endpoint** for bulk dedup + stale cleanup
- **Multilingual NLP (Turkish + English built-in)**: zeyrek lemmatization, dateparser, stopwords
- **Knowledge graph**: entities, relationships, temporal facts
- **OpenClaw session JSONL ingestion** with auto-sync
- **FastAPI with 9 endpoints** (Swagger UI at `/docs`)
- **Single SQLite file** — no Docker, no external DB

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                       FastAPI  (:8787)                             │
│  /v1/recall  /v1/capture  /v1/store  /v1/forget  /v1/search        │
│  /v1/stats   /v1/health   /v1/decay  /v1/consolidate               │
└──────────┬───────────────┬───────────────┬────────────────────────┘
           │               │               │
     ┌─────▼─────┐   ┌────▼─────┐   ┌────▼────────┐
     │  Hybrid   │   │  Ingest  │   │  Knowledge   │
     │  Search   │   │  JSONL   │   │   Graph      │
     │ (RRF Fuse)│   │ (auto)   │   │ (entities)   │
     └──┬──┬──┬──┘   └────┬─────┘   └────┬─────────┘
        │  │  │           │              │
   ┌────▼┐ │ ┌▼────┐  ┌──▼──────────────▼──┐
   │Vec  │ │ │FTS5 │  │       SQLite         │
   │Srch │ │ │BM25 │  │  memories + vectors  │
   └──┬──┘ │ └──┬──┘  │  fts + entities/KG   │
      │    │    │     └─────────┬────────────┘
      │    │    │               │
      │    │    │         ┌─────▼─────┐
      │    │    └────────►│ Recency   │
      │    │              │  scoring  │
      │    │              └───────────┘
      │    │
      │    │              ┌───────────┐
      │    └─────────────►│ Strength  │
      │                   │ (decay +  │
      └──────────────────►│ boost)    │
                          └───────────┘

         sqlite-vec + FTS5 + strength/decay
         single .sqlite database file

   ┌──────────────┐
   │ OpenRouter    │
   │ Embeddings    │
   │ (optional)    │
   └──────────────┘
```

## API (9 Endpoints)

Swagger UI: **http://localhost:8787/docs**

### 1) `POST /v1/recall` — hybrid search

Hybrid search across 4 layers (semantic + BM25 + recency + strength), fused via RRF.

**Request**
```json
{
  "query": "What did we decide about backups?",
  "limit": 5,
  "min_score": 0.0
}
```

**Response (example)**
```json
{
  "query": "What did we decide about backups?",
  "count": 1,
  "triggered": true,
  "results": [
    {
      "id": "a1b2c3d4e5f6",
      "text": "We agreed to run a daily SQLite backup at 04:00...",
      "category": "assistant",
      "importance": 0.75,
      "created_at": 1706900000.0,
      "score": 0.0162,
      "semantic_score": 0.8421,
      "keyword_score": 0.0,
      "recency_score": 0.9812,
      "strength_score": 0.9034,
      "confidence_tier": "MEDIUM"
    }
  ]
}
```

### 2) `POST /v1/capture` — batch ingest (write-time merge)

Ingest a batch of messages. Each message is importance-scored, (optionally) embedded, then **merge-or-insert** is applied to avoid duplicates.

**Request**
```json
{
  "messages": [
    {"text": "We should rotate logs weekly.", "role": "user", "session": "2026-02-08"},
    {"text": "Agreed — compress and keep 4 weeks.", "role": "assistant", "session": "2026-02-08"}
  ]
}
```

**Response**
```json
{
  "stored": 1,
  "merged": 1,
  "total": 2
}
```

### 3) `POST /v1/store` — store one memory (write-time merge)

Store a single memory (fact/decision/note). Uses write-time merge when a near-duplicate already exists.

**Request**
```json
{
  "text": "Backups: run daily at 04:00; keep last 7 copies.",
  "category": "fact",
  "importance": 0.9
}
```

**Response**
```json
{
  "id": "f7e8d9c0b1a2",
  "stored": true,
  "merged": false,
  "similarity": null
}
```

### 4) `DELETE /v1/forget` — delete

Delete a memory by ID, or by query (deletes first match).

**Request (by id)**
```json
{ "id": "f7e8d9c0b1a2" }
```

**Request (by query)**
```json
{ "query": "daily at 04:00" }
```

### 5) `GET /v1/search` — interactive search

For CLI/debug use.

Example:
```bash
curl "http://localhost:8787/v1/search?query=backup&limit=5"
```

### 6) `GET /v1/stats` — stats

Returns DB-level stats (counts by category, entities, relationships, temporal facts).

### 7) `GET /v1/health` — health

Includes uptime + whether storage/embedder are available.

### 8) `POST /v1/decay` — run Ebbinghaus decay

Cron-friendly endpoint to apply strength decay to stale, unused memories.

```bash
curl -X POST http://localhost:8787/v1/decay
```

### 9) `POST /v1/consolidate` — dedup + archive stale

Bulk maintenance endpoint:
- merges duplicates (high cosine similarity)
- archives stale memories below a minimum strength threshold

```bash
curl -X POST http://localhost:8787/v1/consolidate
```

## Configuration

Environment variables keep the historical `ASUMAN_MEMORY_*` prefix for compatibility.

| Variable | Default | Description |
|---|---:|---|
| `OPENROUTER_API_KEY` | *(required for semantic search)* | OpenRouter API key used for embeddings |
| `ASUMAN_MEMORY_DB` | `~/.asuman/memory.sqlite` | SQLite database path |
| `ASUMAN_MEMORY_MODEL` | `qwen/qwen3-embedding-8b` | Embedding model name |
| `ASUMAN_MEMORY_DIMENSIONS` | `4096` | Vector dimensions |
| `ASUMAN_MEMORY_HOST` | `127.0.0.1` | API bind address |
| `ASUMAN_MEMORY_PORT` | `8787` | API server port |
| `ASUMAN_SESSIONS_DIR` | `~/.openclaw/agents/main/sessions` | OpenClaw session JSONL directory |
| `ASUMAN_MEMORY_CONFIG` | *(none)* | Optional JSON config overlay file |

### JSON config overlay

Set `ASUMAN_MEMORY_CONFIG=/path/to/config.json` to override any `Config` field.

```json
{
  "weight_semantic": 0.40,
  "weight_keyword": 0.25,
  "weight_recency": 0.15,
  "weight_strength": 0.20,
  "batch_size": 50
}
```

## Quick Start

```bash
git clone https://github.com/dorukardahan/asuman-memory.git
cd whatsapp-memory
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY="sk-or-..."
python -m asuman_memory
```

Health check:
```bash
curl http://localhost:8787/v1/health
```

## OpenClaw Integration (auto-sync)

This repo includes a session ingester that reads OpenClaw JSONL transcripts and stores them into the SQLite memory DB.

### 1) Cron sync (`scripts/cron_sync.sh`)

- Run incremental sync every 30 minutes
- Write logs to a file

Example crontab entry:
```cron
*/30 * * * * /path/to/whatsapp-memory/scripts/cron_sync.sh
```

### 2) Systemd service (example)

A minimal service file that runs the API:

```ini
[Unit]
Description=OpenClaw Memory API
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/whatsapp-memory
Environment=OPENROUTER_API_KEY=sk-or-...
ExecStart=/path/to/whatsapp-memory/.venv/bin/python -m asuman_memory
Restart=always

[Install]
WantedBy=multi-user.target
```

### 3) Calling `/v1/recall` from agent prompts (TOOLS.md-style snippet)

In your agent’s `TOOLS.md` (or a skill), describe the memory call as a tool-like HTTP action:

```markdown
### Memory recall (HTTP)

When the user asks to remember past decisions, preferences, or prior context, call:

```bash
curl -s -X POST http://127.0.0.1:8787/v1/recall \
  -H 'Content-Type: application/json' \
  -d '{"query":"<user question>","limit":5,"min_score":0.0}'
```

Summarize the returned `results[].text` into 3–7 bullets and cite the memory IDs.
```

## Search Weights (4 layers)

Default layer weights used in RRF fusion:

- **Semantic:** 0.40
- **Keyword:** 0.25
- **Recency:** 0.15
- **Strength:** 0.20

## Ebbinghaus Strength (spaced repetition)

Each memory has a **strength** value that approximates retention:

- **Default strength:** `1.0`
- **Retrieval boost:** `+0.3` per access (cap `5.0`)
- **Weekly decay:** `-0.05` for unused memories (floor `0.3`)

Strength is combined with the other layers via RRF. The `/v1/decay` endpoint applies the decay step; retrieval boosting happens automatically on top hits.

## Write-time Semantic Merge

To prevent “a thousand near-identical memories”:

- On `store` / `capture`, find nearest neighbor in the same **category**
- If cosine similarity **> 0.85**, **merge** instead of insert
- Text is appended with `\n• new_text`
- Embeddings are averaged
- Strength gets a small boost (`+0.2`)

## Maintenance

### Weekly decay

```bash
curl -X POST http://localhost:8787/v1/decay
```

### Consolidation (dedup + stale cleanup)

```bash
curl -X POST http://localhost:8787/v1/consolidate
```

### Backups

A backup helper script is included:

```bash
bash scripts/backup_db.sh
```

It creates a safe SQLite `.backup` copy (WAL-safe) and prunes old backups.

### Management script

```bash
./scripts/manage.sh status
./scripts/manage.sh sync
./scripts/manage.sh health
```

## Multilingual NLP (Turkish + English)

This project includes optional language tooling for better recall on morphologically rich languages:

- **Lemmatization** via `zeyrek`
- **Temporal parsing** via `dateparser`
- **Stopword filtering** (Turkish + English)

You can run the system without these features, but they improve keyword recall and entity extraction on Turkish text.

## License

MIT
