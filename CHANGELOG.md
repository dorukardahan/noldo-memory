# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- Use one validity timestamp per search, keep explicit as-of queries independent from ingestion-date phrases, and screen recalled metadata before OpenClaw prompt formatting.
- Exclude scheduled future revisions from ordinary history recall while retaining them for explicit forgetting; capture long messages within the existing text bound.
- Treat an absent embedder as degraded; atomically import only linear revision families and preserve retained child boundaries.
- Keep recall degradation status local to each request; preserve rule evidence and include historical text in query-based forgetting.
- Declare forgetting in the OpenClaw tool manifest; validate partial-import lineage against retained rows.
- Reinforce only results admitted to recall output, and prevent degraded searches from becoming semantic cache hits.
- Bind OpenClaw tools and lifecycle capture to trusted agent context; reject missing scope and model-supplied cross-agent overrides.
- Preserve distinct statements and provenance instead of merging by vector similarity; deduplicate exact retries without growing text.
- Invalidate in-flight and cached recall after writes, and disable Hermes provider-local caching by default.
- Scope reranker score caches by agent and document content; retain explicit revision families during decay archival.
- Apply historical-query inference consistently to the existing administrative recall endpoint.
- Keep filtered vector fallback on the same L2 metric as the existing sqlite-vec index.

### Added
- Explicit memory revisions with validity intervals, historical recall, revision-family forgetting and export/import evidence preservation.
- Bounded media-derivative evidence metadata, current-turn capture and delivery-aware outgoing OpenClaw text capture.
- Optional model-calibrated automatic recall admission, synthetic regression/replay tools and dated native/platform comparison.


## [1.27.16] - 2026-07-23

### Fixed
- Remove provider-owned background workers from the Hermes adapter, relying on the Hermes v0.19 host executor for turn sync and queued prefetch work while keeping inline memory-write hooks synchronous.
- Bound the Hermes recall cache with configurable TTL/LRU limits and invalidate it across session and compaction boundaries.
- Keep Hermes adapter discovery network-free and make live readiness an explicit, bounded, privacy-safe doctor option.
- Isolate Hermes recall cache entries by the full effective request scope, discard recall/tool results that finish after close, reject late backend work with a bounded provider-local shutdown wait, and match Hermes v0.19 context sanitization semantics.
- Reset lifecycle gates on clean provider reinitialization while rejecting reuse until any timed-out operation from the previous lifecycle has drained.
- Bind Hermes request bodies and network admission to the same lifecycle generation so preempted old-session work cannot run against a reinitialized client.
- Use the active published config for readiness probes so a provider initialized with a non-default profile probes the correct endpoint.
- Register host-lane recall and write callbacks before request snapshots so shutdown cannot miss preempted old-lifecycle work.
- Bind host-lane admission to the current session generation so a concurrent session switch cannot retarget old callback content.

## [1.27.15] - 2026-07-20

### Fixed
- Prevent the session-end TODO scanner from treating ordinary conversational words as actionable follow-ups.
- Remove a duplicated project-description phrase from package metadata.

### Changed
- Align Ruff linting with the declared Python 3.10 compatibility floor.

### Internal
- Remove a stale backup script artifact from the public source tree.

## [1.27.14] - 2026-07-11

### Fixed
- Harden native OpenClaw operational capture: recursively redact secret-keyed structured values (including nested arrays, objects, and JSON-encoded object/array strings), safely close true cycles without dropping repeated DAG references, redact complete quoted/raw assignment values, and normalize whitespace before persistence. Skip only canonical/known-qualified NoldoMem recall, store, and pin tool IDs while preserving unrelated and normal operational capture.
- Restore reliable source and wheel builds with the supported setuptools legacy backend and enforce distribution builds in CI.

## [1.27.13] - 2026-05-28

### Fixed
- Keep the Hermes NoldoMem adapter's cached session id aligned when Hermes v2026.5.28 rotates sessions during compression, reset, resume, or branch flows.
- Preserve optional `/v1/store` `session_id` provenance as `source_session`, so Hermes adapter writes can be traced to the originating session after rotation.

### Changed
- Document Hermes v2026.5.28 memory-provider toolset gating and the Dorry/Dobby scope requirement so NoldoMem tools remain visible while legacy `hermes` scope stays historical.

## [1.27.12] - 2026-05-22

### Changed
- Align recommended OpenClaw compaction hook timeout settings with OpenClaw 2026.5.20's 30 second before/after compaction default, leaving enough room for NoldoMem's bounded capture request.

## [1.27.11] - 2026-05-17

### Fixed
- Raise the Hermes NoldoMem adapter default HTTP timeout to 8 seconds so healthy recall/store calls with hosted reranking or inline embedding do not fail at the old 2 second boundary.

### Changed
- Document OpenClaw 2026.5.3 plugin hook timeout policy for the NoldoMem native plugin, so optional lifecycle capture cannot stall the gateway when the memory API is slow.
- Clarify that custom NoldoMem deployments should remove native `memory_search` / `memory_get` from explicit agent tool allow lists while keeping `noldomem_recall`, `noldomem_store`, and `noldomem_pin`.

## [1.27.10] - 2026-05-09

### Added
- Add a native Hermes `MemoryProvider` adapter under `adapters/hermes/noldomem`, with bounded recall, explicit NoldoMem tools, graceful degradation, and docs for using NoldoMem as the single long-term memory backend.

### Fixed
- Send `/v1/pin` requests with the public `id` field from the OpenClaw plugin instead of the plugin-local `memory_id` parameter name.

## [1.27.9] - 2026-05-03

### Changed
- Default new local embedding setups to Qwen3-Embedding-0.6B / 1024 dimensions and make larger CPU models an explicit `detect-hardware.sh --prefer-quality` choice. This keeps fresh installs aligned with the production-tested low-latency profile and reduces dimension-mismatch risk.

## [1.27.8] - 2026-05-03

### Fixed
- Tokenize full-text search queries before building FTS5 MATCH expressions, so JSON snippets and quoted model names no longer break recall with syntax errors.

## [1.27.7] - 2026-05-03

### Changed
- Declare the native OpenClaw plugin runtime entrypoint and 2026.5.2 package compatibility metadata, so fresh local plugin installs use the current installer path instead of legacy index discovery.

## [1.27.6] - 2026-05-03

### Fixed
- Respect sqlite-vec's 4096 KNN limit during scoped vector search and fall back to exact in-scope cosine scoring when a sparse namespace or memory-type filter still needs more candidates.

## [1.27.5] - 2026-05-03

### Fixed
- Preserve semantic recall for namespace or memory-type scoped vector search by widening sqlite-vec candidates before applying metadata filters. This prevents sparse workspace/session scopes from falling back to keyword-only scoring when closer global memories crowd out scoped vectors.

## [1.27.4] - 2026-05-03

### Fixed
- Declare NoldoMem tool contracts and startup activation in the OpenClaw plugin manifest so `noldomem_recall`, `noldomem_store`, and `noldomem_pin` register reliably on OpenClaw 2026.5.2+ manifest-first startup paths.
- Embed explicit `/v1/store` and `/v1/rule` writes inline when the background embed worker is disabled, preventing fresh `noldomem_store` memories from staying vectorless until the next backfill.

## [1.27.3] - 2026-04-27

### Fixed
- Bound `agent=all` recall latency by reranking once after cross-agent merge instead of reranking once per agent database.
- Disabled slow local cross-encoder runtime fallback for hosted reranker failures by default. Set `AGENT_MEMORY_RERANKER_API_LOCAL_FALLBACK=true` to opt in.
- Allowed hosted reranker timeout values down to 1 second for interactive CPU-only deployments.

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
