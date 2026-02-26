# Agent Memory

Persistent, local-first memory for AI agents. One SQLite file, no Docker, no cloud DB.

Your OpenClaw agent remembers conversations, decisions, and facts across sessions with hybrid recall.

## Features

- **5-layer hybrid search** — semantic (sqlite-vec) + BM25 keyword + recency + strength + importance, fused with Reciprocal Rank Fusion (k=60)
- **Two-pass cross-encoder reranking** — fast primary reranker + background quality reranker with async cache refresh
- **Knowledge graph** — entity extraction, typed relations, temporal facts with conflict detection
- **Ebbinghaus decay** — spaced-repetition strength model with importance-adjusted curves
- **Turkish + English NLP** — morphological analysis (zeyrek), temporal parsing, ASCII folding, stopwords
- **Per-agent isolation** — each agent gets its own SQLite database, cross-agent search supported
- **Write-time semantic merge** — deduplicates similar memories at ingest (cosine ≥ 0.85)
- **Consolidation & GC** — periodic dedup, weak memory archival, permanent purge of old soft-deletes
- **Resilient embedding** — batch fallback to individual, 3-retry with exponential backoff, text truncation
- **Vectorless backfill** — cron script to embed memories that failed initial embedding
- **Security** — API key auth, rate limiting (120 req/min), audit logging, localhost-only CORS
- **Export/Import** — JSON export/import for backup and migration

## Quick Start

```bash
# Clone
git clone https://github.com/dorukardahan/asuman-memory.git
cd asuman-memory

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and set at least: OPENROUTER_API_KEY, AGENT_MEMORY_API_KEY

# Load env vars for this shell
set -a
source .env
set +a

# Run API
python -m agent_memory
```

The API starts on `http://127.0.0.1:8787`.

## Embedding Server

Agent Memory needs an embedding API compatible with the OpenAI `/v1/embeddings` format. Run locally (recommended) or use a cloud API.

### Hardware auto-detection

The included script detects your system resources and recommends the best model:

```bash
./scripts/detect-hardware.sh          # show recommendation
./scripts/detect-hardware.sh --json   # machine-readable output
./scripts/detect-hardware.sh --apply  # write settings to .env
```

### Model profiles

| Profile | Model | Size | Dimensions | Min RAM | Min Cores | Use Case |
|---------|-------|------|------------|---------|-----------|----------|
| **minimal** | EmbeddingGemma 300M | 314MB | 768 | 1-2GB | 1-2 | Raspberry Pi, $5 VPS |
| **light** | Qwen3-Embedding-0.6B | 610MB | 1024 | 2-4GB | 2-4 | Small VPS, shared hosting |
| **standard** | Qwen3-Embedding-4B | 4.0GB | 2560 | 4-8GB | 4-8 | Mid-range server |
| **heavy** | Qwen3-Embedding-8B | 8.1GB | 4096 | 12GB+ | 8+ | Dedicated server |
| **gpu** | Qwen3-Embedding-8B + GPU | 8.1GB | 4096 | 8GB+ | 4+ | NVIDIA GPU ≥6GB VRAM |

**Quality comparison** (MTEB Multilingual leaderboard, higher = better):

| Model | MTEB Score | Multilingual | Speed (relative) |
|-------|-----------|--------------|-------------------|
| EmbeddingGemma 300M | ~58 | 100+ langs | ⚡⚡⚡⚡ fastest |
| Qwen3-Embedding-0.6B | ~63 | 100+ langs | ⚡⚡⚡ fast |
| Qwen3-Embedding-4B | ~68 | 100+ langs | ⚡⚡ balanced |
| Qwen3-Embedding-8B | ~70.58 (#1) | 100+ langs | ⚡ thorough |

> **Tip:** Qwen3-Embedding-4B offers the best quality/resource ratio — close to the 8B quality at half the RAM. Start with 4B unless you have strong reasons to go bigger or smaller.

### Local setup

```bash
# 1. Install llama.cpp (llama-server binary)
# Option A: Build from source
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build --target llama-server -j
sudo cp build/bin/llama-server /usr/local/bin/

# Option B: Download prebuilt release
# https://github.com/ggml-org/llama.cpp/releases

# 2. Download embedding model (pick one based on your profile)
mkdir -p /opt/models && cd /opt/models
# Standard: wget https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF/resolve/main/Qwen3-Embedding-4B-Q8_0.gguf
# Light:    wget https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf
# Heavy:    wget https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF/resolve/main/Qwen3-Embedding-8B-Q8_0.gguf

# 3. Run (standard profile example)
llama-server --model Qwen3-Embedding-4B-Q8_0.gguf \
  --embedding --pooling last --host 0.0.0.0 --port 8090 \
  --ctx-size 8192 --batch-size 2048 --threads 12 --parallel 2
```

Set `OPENROUTER_BASE_URL=http://127.0.0.1:8090/v1` and `AGENT_MEMORY_DIMENSIONS=2560` in `.env`.

A systemd unit is included: `embedding-server.service.example`.

### Cloud API

If you prefer not to run a local server, use OpenRouter or any OpenAI-compatible API:

```bash
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=your-key
AGENT_MEMORY_MODEL=openai/text-embedding-3-large
AGENT_MEMORY_DIMENSIONS=3072
```

### Dimension compatibility

⚠️ **Changing embedding dimensions requires re-embedding all memories.** If you switch profiles, run:

```bash
# Reads config from environment / .env — no hardcoded values
.venv/bin/python scripts/reindex_embeddings.py

# Dry run first to see what would happen:
.venv/bin/python scripts/reindex_embeddings.py --dry-run

# For a specific agent DB:
AGENT_MEMORY_DB=~/.agent-memory/memory-codex.sqlite .venv/bin/python scripts/reindex_embeddings.py
```

> **Note on paths:** All scripts read DB and workspace paths from environment variables (`AGENT_MEMORY_DB`, `AGENT_MEMORY_DATA_DIR`, `OPENCLAW_WORKSPACE`) with sensible defaults. Set these in your `.env` file if your paths differ from the defaults.

> **Important:** With `--parallel N`, each slot gets `ctx-size / N` tokens. Set `ctx-size` high enough for your longest texts, or use `AGENT_MEMORY_MAX_EMBED_CHARS` to truncate.

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/recall` | POST | Hybrid search (semantic + BM25 + recency + strength + importance). Supports `namespace` and `max_tokens` for context budget. |
| `/v1/capture` | POST | Batch ingest messages (auto memory_type classification) |
| `/v1/store` | POST | Store a single memory (auto memory_type: fact/preference/rule/conversation) |
| `/v1/rule` | POST | Store a rule/instruction (importance=1.0) |
| `/v1/forget` | DELETE | Delete by ID or query |
| `/v1/search` | GET | Interactive search (CLI/debug) |
| `/v1/pin` | POST | Pin a memory (protects from decay/gc/consolidation) |
| `/v1/unpin` | POST | Unpin a memory |
| `/v1/decay` | POST | Run Ebbinghaus strength decay (respects pinned) |
| `/v1/consolidate` | POST | Deduplicate + archive stale memories |
| `/v1/compress` | POST | Summarize old long memories (dry_run supported) |
| `/v1/gc` | POST | Permanently purge old soft-deleted memories (respects pinned) |
| `/v1/amnesia-check` | POST | Check memory coverage by topic list (healthy/warning/amnesia) |
| `/v1/stats` | GET | Database statistics |
| `/v1/agents` | GET | List agent databases |
| `/v1/health` | GET | Basic health check |
| `/v1/health/deep` | GET | DB integrity, embedding probe, vectorless count, disk usage |
| `/v1/metrics` | GET | Operational metrics (JSON) |
| `/v1/metrics/prometheus` | GET | Prometheus text exposition format metrics |
| `/v1/admin/rotate-key` | POST | Rotate API key (with optional expiry) |
| `/v1/export` | GET | Export memories as JSON |
| `/v1/import` | POST | Import memories from JSON |

All endpoints accept `?agent=<id>` for per-agent routing. Use `agent=all` for cross-agent operations.

**Per-agent API keys:** Extra keys can be added to `memory-api-keys.json` with an optional `"agent"` field to restrict access to a single agent's data.

## Hooks

OpenClaw hook examples for Memory API integration are available in [`hooks/`](./hooks/).

They show how to:

- capture messages and session summaries into memory,
- inject recalled context on bootstrap,
- persist context around compaction/session transitions,
- capture important tool outputs and subagent failures.

See [`hooks/README.md`](./hooks/README.md) for full setup and configuration instructions.

## Search Architecture

```text
Query → [Semantic (0.50)] → sqlite-vec cosine
      → [Keyword  (0.25)] → FTS5 BM25 trigram
      → [Recency  (0.10)] → exp(-0.01 × days)
      → [Strength (0.07)] → Ebbinghaus retention
      → [Importance(0.08)] → write-time scoring
      ↓
      RRF fusion (k=60)
      ↓
      Primary reranker (MiniLM, top-10)
      ↓
      Background reranker (BGE-v2-m3, top-3, async cache update)
```

## Configuration

All configuration is environment-driven. `AGENT_MEMORY_*` variables are canonical.

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MEMORY_CONFIG` | _unset_ | Optional JSON config file path loaded before env vars |
| `OPENROUTER_API_KEY` | `""` | Embedding API key (required for semantic embeddings) |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Embedding API base URL |
| `AGENT_MEMORY_MODEL` | `qwen/qwen3-embedding-4b` | Embedding model name |
| `AGENT_MEMORY_DIMENSIONS` | `2560` | Embedding dimensions (must match model) |
| `AGENT_MEMORY_MAX_EMBED_CHARS` | `3500` | Truncate text before embedding |
| `AGENT_MEMORY_DB` | `$HOME/.agent-memory/memory.sqlite`* | SQLite path |
| `AGENT_MEMORY_HOST` | `127.0.0.1` | API bind address |
| `AGENT_MEMORY_PORT` | `8787` | API port |
| `AGENT_MEMORY_API_KEY` | `""` | API authentication key |
| `AGENT_MEMORY_SESSIONS_DIR` | `$HOME/.openclaw/agents/main/sessions` | Session source directory |
| `AGENT_MEMORY_W_SEMANTIC` | `0.50` | Semantic score weight |
| `AGENT_MEMORY_W_KEYWORD` | `0.25` | Keyword/BM25 score weight |
| `AGENT_MEMORY_W_RECENCY` | `0.10` | Recency score weight |
| `AGENT_MEMORY_W_STRENGTH` | `0.07` | Memory-strength score weight |
| `AGENT_MEMORY_W_IMPORTANCE` | `0.08` | Importance score weight |
| `AGENT_MEMORY_RERANKER_ENABLED` | `true` | Enable primary cross-encoder reranker |
| `AGENT_MEMORY_RERANKER_MODEL` | `balanced` | Reranker preset (`fast`, `balanced`, `quality`) or HF model id |
| `AGENT_MEMORY_RERANKER_TOP_K` | `10` | Docs reranked in primary pass |
| `AGENT_MEMORY_RERANKER_WEIGHT` | `0.22` | Primary reranker blend weight |
| `AGENT_MEMORY_RERANKER_THREADS` | `4` | Torch threads for primary reranker |
| `AGENT_MEMORY_RERANKER_MAX_DOC_CHARS` | `600` | Per-doc char limit in primary reranker |
| `AGENT_MEMORY_RERANKER_PREWARM` | `true` | Prewarm primary reranker at startup |
| `AGENT_MEMORY_RERANKER_TWO_PASS_ENABLED` | `true` | Enable background second pass |
| `AGENT_MEMORY_RERANKER_TWO_PASS_MODEL` | `quality` | Background reranker preset/model |
| `AGENT_MEMORY_RERANKER_TWO_PASS_TOP_K` | `3` | Docs reranked in second pass |
| `AGENT_MEMORY_RERANKER_TWO_PASS_WEIGHT` | `0.35` | Two-pass reranker blend weight |
| `AGENT_MEMORY_RERANKER_TWO_PASS_THREADS` | `2` | Torch threads for second pass |
| `AGENT_MEMORY_RERANKER_TWO_PASS_MAX_DOC_CHARS` | `450` | Per-doc char limit in second pass |
| `AGENT_MEMORY_RERANKER_TWO_PASS_PREWARM` | `false` | Prewarm background reranker at startup |
| `AGENT_MEMORY_EMBED_WORKER_ENABLED` | `true` | Enable background embed worker (auto-embeds vectorless memories) |
| `AGENT_MEMORY_EMBED_WORKER_INTERVAL` | `300` | Seconds between embed worker sweeps |

\* `AGENT_MEMORY_DB` fallback behavior in code: if `$HOME/.asuman` exists and `$HOME/.agent-memory` does not, default becomes `$HOME/.asuman/memory.sqlite`.

Legacy fallbacks are still accepted when the new key is unset: `ASUMAN_MEMORY_CONFIG`, `ASUMAN_MEMORY_DB`, `ASUMAN_MEMORY_MODEL`, `ASUMAN_MEMORY_PORT`, `ASUMAN_MEMORY_DIMENSIONS`, `ASUMAN_MEMORY_HOST`, `ASUMAN_SESSIONS_DIR`.

## Cron Jobs

A `crontab.example` file is included with all recommended cron entries. Install with:

```bash
# Review and edit paths first
cat crontab.example
# Then install (merges with existing crontab)
(crontab -l 2>/dev/null; cat crontab.example) | crontab -
```

**Important:** API calls from cron use `scripts/cron_api_call.sh` which reads the API key from a file — never hardcode secrets in crontab.

| Job | Schedule | Purpose |
|-----|----------|---------|
| Session sync | `*/30 * * * *` | Pull OpenClaw sessions → memory |
| Ebbinghaus decay | `0 2 * * *` | Reduce memory strength over time |
| Consolidation | `30 2 * * 0` | Deduplicate similar + archive weak memories |
| GC purge | `0 3 * * 0` | Delete soft-deleted memories > 30 days |
| Workspace export | `0 4 * * 0` | Export summaries to agent workspace dirs |
| Vectorless backfill | `0 */6 * * *` | Re-embed failed memories |
| SQLite backup | `0 7 * * *` | Hot backup all databases |

## Deployment

### 1) Embedding server

The memory API needs an OpenAI-compatible `/v1/embeddings` endpoint. An `embedding-server.service.example` systemd unit is included.

```bash
# Download a model (e.g. Qwen3-Embedding-4B)
mkdir -p /opt/models
# Place your .gguf file in /opt/models/

# Install systemd unit
sudo cp embedding-server.service.example /etc/systemd/system/embedding-server.service
# Edit paths in the unit file, then:
sudo systemctl daemon-reload
sudo systemctl enable --now embedding-server
```

- Set `OPENROUTER_BASE_URL=http://127.0.0.1:8090/v1` in `.env` for local mode.
- Keep `AGENT_MEMORY_DIMENSIONS` aligned with your model output (Qwen3-4B = 2560).
- If you increase `--parallel`, also increase `--ctx-size` or reduce `AGENT_MEMORY_MAX_EMBED_CHARS`.

Alternatively, use OpenRouter cloud: set `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`.

### 2) Memory API service

Use `asuman-memory.service.example` as the base unit file:

```bash
sudo cp asuman-memory.service.example /etc/systemd/system/asuman-memory.service
# Edit paths and EnvironmentFile in the unit
sudo systemctl daemon-reload
sudo systemctl enable --now asuman-memory
sudo systemctl status asuman-memory --no-pager
```

Keep secrets in an env file (e.g. `/etc/asuman-memory.env`) referenced by the service `EnvironmentFile=` entry.

### 3) Multi-agent provisioning

For OpenClaw setups with multiple agents, use the provision script:

```bash
# Show status of all agents
./scripts/provision-agent-memory.sh --status

# Provision memory dirs + verify API access for all agents
./scripts/provision-agent-memory.sh

# Dry run (no changes)
./scripts/provision-agent-memory.sh --dry-run
```

This scans `openclaw.json`, creates `memory/` dirs in each agent workspace, and verifies API connectivity.

### 4) OpenClaw hooks

Seven hooks connect OpenClaw sessions to the memory system. See the `hooks/` directory for documented examples of each:

| Hook | Purpose |
|------|---------|
| `realtime-capture` | Capture messages to memory in real-time |
| `session-end-capture` | Batch capture when a session ends |
| `bootstrap-context` | Inject recalled memories into session context |
| `after-tool-call` | Capture tool results as memories |
| `pre-session-save` | Tag sessions with memory metadata |
| `post-compaction-restore` | Restore memory context after compaction |
| `subagent-complete` | Capture sub-agent results |

To install hooks, copy the `.example` files and configure in your OpenClaw config:

```bash
cp hooks/realtime-capture/handler.js.example ~/.openclaw/hooks/realtime-capture/handler.js
# Repeat for each hook you want, then set MEMORY_API_KEY in each
```

### 5) OpenClaw native memory (important)

This system **replaces** OpenClaw's built-in `memorySearch`. Disable it in your `openclaw.json`:

```json
{
  "memorySearch": {
    "enabled": false
  }
}
```

All memory operations are handled by the Agent Memory API via hooks — OpenClaw's native memory system must be off to avoid conflicts.

### 6) Production checklist

- [ ] Set strong secrets: `OPENROUTER_API_KEY`, `AGENT_MEMORY_API_KEY`
- [ ] Pin an explicit DB path with enough disk (`AGENT_MEMORY_DB`)
- [ ] Keep `AGENT_MEMORY_HOST=127.0.0.1` unless reverse-proxying
- [ ] Install embedding server (local or cloud)
- [ ] Install cron jobs from `crontab.example`
- [ ] Run `scripts/provision-agent-memory.sh` for multi-agent setups
- [ ] Copy and configure hooks from `hooks/` directory
- [ ] Disable OpenClaw native `memorySearch`
- [ ] Back up SQLite DB regularly (`/v1/export` or `scripts/backup_db.sh`)
- [ ] Monitor: `/v1/health`, `/v1/metrics`, `journalctl -u asuman-memory`

## API Usage Examples

### Store a memory

```bash
curl -s -X POST http://127.0.0.1:8787/v1/store \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $AGENT_MEMORY_API_KEY" \
  -d '{"text": "User prefers dark mode", "agent": "main", "source": "preference", "importance": 0.8}'
```

### Recall memories

```bash
curl -s -X POST http://127.0.0.1:8787/v1/recall \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $AGENT_MEMORY_API_KEY" \
  -d '{"query": "UI preferences", "agent": "main", "limit": 5}'
```

### Batch capture messages

```bash
curl -s -X POST http://127.0.0.1:8787/v1/capture \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $AGENT_MEMORY_API_KEY" \
  -d '{"messages": [
    {"role": "user", "content": "Deploy to production"},
    {"role": "assistant", "content": "Deployed v2.1 to prod at 14:30"}
  ], "agent": "main"}'
```

### Health check

```bash
curl -s http://127.0.0.1:8787/v1/health
# {"status":"ok","checks":{"storage":true,"embedding":true}}
```

### Stats

```bash
curl -s -H "X-API-Key: $AGENT_MEMORY_API_KEY" http://127.0.0.1:8787/v1/stats?agent=main
```

## Full Setup Guide (for AI agents)

If you're an AI agent setting up this system, follow these steps in order:

```bash
# 1. Clone and install
git clone https://github.com/dorukardahan/asuman-memory.git
cd asuman-memory
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
./scripts/detect-hardware.sh            # see recommended profile
./scripts/detect-hardware.sh --apply    # updates model, dimensions, and base URL in .env

# Generate a strong API key and save it
AGENT_MEMORY_API_KEY=$(openssl rand -base64 32)
mkdir -p ~/.agent-memory
echo "$AGENT_MEMORY_API_KEY" > ~/.agent-memory/memory-api-key
chmod 600 ~/.agent-memory/memory-api-key
# Set it in .env:
sed -i "s|^AGENT_MEMORY_API_KEY=.*|AGENT_MEMORY_API_KEY=$AGENT_MEMORY_API_KEY|" .env

# For cloud embedding (instead of local): set OPENROUTER_API_KEY in .env
# NOTE: .env must be clean KEY=VALUE format — no inline comments (#)
#       systemd EnvironmentFile does not support inline comments

# 3. Download embedding model (pick based on detect-hardware output)
mkdir -p /opt/models && cd /opt/models
# Standard profile (Qwen3-Embedding-4B, 4.0GB):
wget https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF/resolve/main/Qwen3-Embedding-4B-Q8_0.gguf
# Light profile (Qwen3-Embedding-0.6B, 610MB):
# wget https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf
# Heavy profile (Qwen3-Embedding-8B, 8.1GB):
# wget https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF/resolve/main/Qwen3-Embedding-8B-Q8_0.gguf
cd -

# 4. Create log directory and install systemd services
sudo mkdir -p /var/log
# Edit BOTH service files: fix paths marked with ">>> CHANGE <<<" comments
sudo cp embedding-server.service.example /etc/systemd/system/embedding-server.service
sudo cp asuman-memory.service.example /etc/systemd/system/asuman-memory.service
sudo systemctl daemon-reload
sudo systemctl enable --now embedding-server
sudo systemctl enable --now asuman-memory

# 5. Verify
curl -s http://127.0.0.1:8787/v1/health
# Should return: {"status":"ok","checks":{"storage":true,"embedding":true}}

# 6. Install cron jobs
cat crontab.example                     # review paths first
(crontab -l 2>/dev/null; cat crontab.example) | crontab -

# 7. (Optional) OpenClaw integration
# Copy hooks from hooks/ directory to ~/.openclaw/hooks/
# Disable native memory: set memorySearch.enabled = false in openclaw.json
# Run: ./scripts/provision-agent-memory.sh

# 8. (Optional) Multi-agent provisioning
./scripts/provision-agent-memory.sh --status
./scripts/provision-agent-memory.sh
```

## Migrations

Database schema migrations are in the `migrations/` directory. They are applied automatically when the API starts — no manual migration step is needed. If you need to apply migrations manually:

```bash
sqlite3 /path/to/memory.sqlite < migrations/001_temporal_facts_extensions.sql
```

## Tests

```bash
# Install dev dependencies first
pip install -r requirements-dev.txt

# Run tests
.venv/bin/python -m pytest tests/ -x -q
```

## Docker

Quick start with Docker Compose (runs both the Memory API and embedding server):

```bash
# 1. Clone and configure
git clone https://github.com/dorukardahan/asuman-memory.git && cd asuman-memory
cp .env.example .env
# Edit .env: set OPENROUTER_API_KEY, AGENT_MEMORY_API_KEY, model settings

# 2. Download embedding model
mkdir -p models
wget -P models https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF/resolve/main/Qwen3-Embedding-4B-Q8_0.gguf

# 3. Start services
docker compose up -d

# 4. Verify
curl http://localhost:8787/v1/health
```

**Volumes:**
- `memory-data` — SQLite databases and backups (persistent across restarts)
- `./models` — Embedding model GGUF files (bind mount)

**Customize threads/parallelism** via `.env`:
```bash
EMBEDDING_THREADS=8        # CPU threads for embedding
EMBEDDING_PARALLEL=2       # Concurrent requests
EMBEDDING_MODEL_FILE=Qwen3-Embedding-4B-Q8_0.gguf  # Model filename
```

> **Note:** For production use with OpenClaw hooks and cron jobs, the native systemd setup (see Full Setup Guide above) is recommended. Docker is ideal for testing, development, and standalone deployments.

## License

MIT
