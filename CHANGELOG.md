# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- Keep `memory_type` canonical across API, ingest, storage, search, and docs. Unknown or operational labels are normalized to `other` instead of persisting drift such as `incident`, `deployment`, or `config_change`.

### Changed
- Project rebranded from "asuman-memory" to "NoldoMem" (`noldo-memory`)
- GitHub repo: dorukardahan/noldo-memory

## [1.0.0] - 2026-02-26

Production-ready release with full monitoring, per-agent security, and 173 tests.

### Added
- Per-agent API key restrictions (key-level agent scope enforcement)
- Prometheus metrics endpoint (`/v1/metrics/prometheus`) with request/cache/memory gauges
- Request duration histograms and cache hit/miss tracking in middleware
- Memory compression endpoint (`/v1/compress`) for summarizing old long memories
- Parallel search: semantic + keyword run concurrently via asyncio

### Fixed
- Cache miss tracking on parse failures
- Status type hint in metrics middleware

## [0.9.0] - 2026-02-25

### Added
- Parallel search execution (semantic + keyword via `asyncio.create_task`)
- Memory compression module (`agent_memory/compression.py`)
- `/v1/compress` endpoint with dry_run support

## [0.8.0] - 2026-02-25

### Added
- Graceful degradation: `search_mode` and `degraded` flags in recall response
- Amnesia detection endpoint (`/v1/amnesia-check`) with coverage scoring
- Namespace support for memory isolation (`namespace` param on recall/store)
- Post-compaction restore validation hook
- Pre-session save hook (auto-pin important memories)

## [0.7.0] - 2026-02-25

### Added
- Adaptive reranker gating (skip cross-encoder when score spread > threshold)
- MMR diversity post-processing for search results
- Critical memory pinning (`/v1/pin`, `/v1/unpin`) — pinned memories survive decay/gc
- Context budget (`max_tokens`) on `/v1/recall` with token estimation
- Memory type classification (fact/preference/rule/conversation) on ingest
- Deep health check endpoint (`/v1/health/deep`)
- API key rotation endpoint (`/v1/admin/rotate-key`) with multi-key support

### Fixed
- Weight drift: importance weight 0.25 → 0.08 (aligned with config)
- Smart recall gating: expanded anti-trigger patterns

## [0.6.0] - 2026-02-24

### Added
- Docker support (Dockerfile + docker-compose.yml)
- Hardware auto-detection script (5 profiles: minimal → gpu)

### Fixed
- Path resolution standardized to 3-tier logic (env var → ~/.agent-memory → legacy fallback)
- CI compatibility: `sys.executable` instead of hardcoded venv paths
- Crontab security: API key read from file instead of inline

## [0.5.0] - 2026-02-22

### Added
- 15 search and storage improvements
- Embed worker with circuit breaker (5 fail → 5min cooldown)
- Background asyncio worker for vector backfill

### Fixed
- Vectorless memories: 5 silent failure paths identified and fixed
- `/v1/import` retry logic (3 attempts)
- All ruff lint errors resolved (22 → 0)

## [0.4.0] - 2026-02-19

### Added
- Multi-agent session sync with typed relation patterns
- 13 semantic relation types for knowledge graph
- Aggressive Ebbinghaus decay with importance adjustment

## [0.3.0] - 2026-02-18

### Added
- Instruction capture and conflict detection
- Search result caching
- Knowledge graph integration
- Security hardening (API key auth, audit logging, rate limiting)
- CI/CD pipeline (GitHub Actions: lint + test)
- Operational metrics endpoint (`/v1/metrics`)

## [0.2.0] - 2026-02-16

### Added
- Per-agent database routing (multi-agent support)
- Package rename: `noldo_memory` → `agent_memory`
- Ebbinghaus strength decay with spaced repetition
- Write-time semantic merge (deduplication)
- Memory consolidation endpoint

## [0.1.0] - 2026-02-04

### Added
- Initial release
- Hybrid search: semantic (sqlite-vec) + keyword (FTS5 BM25)
- FastAPI REST API with SQLite storage
- OpenClaw session sync integration
- Entity extraction and knowledge graph
- Memory importance scoring
- Export/import endpoints
