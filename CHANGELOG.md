# CHANGELOG

## [1.0.0](https://github.com/dorukardahan/asuman-memory/compare/v0.4.0...v1.0.0) (2026-02-22)

### Features
- Reranker for improved recall precision
- Resilient embedding with graceful fallback
- Automated backfill for memories missing embeddings
- Generic cleanup and garbage collection

### CI/CD
- python-semantic-release with conventional commits
- Automated changelog generation
- Generic rescore script with CLI args

### Fixes
- Fixed auth bypass in CI test environment
- CI timeout configuration

## [0.4.0](https://github.com/dorukardahan/asuman-memory/compare/v0.3.0...v0.4.0) (2026-02-19)

### Features
- Temporal parsing — natural language time expressions in search queries
- Feature registry — modular feature toggle system
- GC endpoint for memory cleanup
- Importance-adjusted decay — rate varies by memory importance
- Noise filters for low-quality memories

### Fixes
- Temporal parsing review fixes + single-word patterns
- Synced with live VPS production state

## [0.3.0](https://github.com/dorukardahan/asuman-memory/compare/v0.2.1...v0.3.0) (2026-02-18)

### Features
- Security hardening — API key auth, audit logging, input validation
- CI/CD pipeline — GitHub Actions with ruff lint + pytest
- Metrics — memory usage stats and health endpoints
- Export/Import — bulk memory endpoints
- Improved consolidation — better dedup and merge logic
- PR template for contributions

### Fixes
- Resolved all ruff lint errors
- Graceful audit log handler for CI environment
- Repo cleanup — version sync, README rewrite

## [0.2.1](https://github.com/dorukardahan/asuman-memory/compare/v0.2.0...v0.2.1) (2026-02-17)

### Features
- Memory system overhaul — 15 improvements across P0, P1, P2 priorities
- Improved recall accuracy and ranking
- Better deduplication and noise filtering
- Enhanced consolidation pipeline

## [0.2.0](https://github.com/dorukardahan/asuman-memory/compare/v0.1.1...v0.2.0) (2026-02-16)

### Features
- Per-agent routing — each agent gets its own memory database
- Package rename: `asuman_memory` → `agent_memory` (fully generic)
- Instruction capture — detects and stores user preferences
- KG conflict detection — catches contradicting memories
- Caching layer for frequently accessed memories
- Multi-agent session sync — 13 typed relation patterns
- Aggressive Ebbinghaus decay tuning

### Docs
- Complete README rewrite
- Dual memory architecture guide
- Embedding integration documentation

## [0.1.1](https://github.com/dorukardahan/asuman-memory/compare/v0.1.0...v0.1.1) (2026-02-08)

### Features
- B12 Patterns — Ebbinghaus forgetting curve decay, write-time merge, consolidation
- Genericized codebase — preparing for multi-agent use
- Improved architecture diagram

### Fixes
- Pass config search weights to HybridSearch correctly
- Security + performance + resilience improvements
- Removed remaining Asuman-specific references

## [0.1.0](https://github.com/dorukardahan/asuman-memory/releases/tag/v0.1.0) (2026-02-04)

### Features
- Complete `asuman_memory` package — hybrid search (vector + keyword)
- Turkish NLP with Zeyrek lemmatizer
- SQLite + sqlite-vec for vector storage
- FastAPI REST API (port 8787)
- systemd service + OpenClaw sync integration
- Production polish — backup scripts, logrotate, test suite
- Configurable decay and importance scoring

### Security
- Removed hardcoded API keys
