# Asuman Memory — Feature Registry

> Son güncelleme: 2026-02-19
> Bu dosya implement edilmiş özelliklerin kaydıdır. Yeni feature eklendiğinde burası da güncellenmelidir.

## Implemented Features

| # | Feature | File(s) | Status | Wired | Notes |
|---|---------|---------|--------|-------|-------|
| 1 | Hybrid Search (RRF) | `search.py` | ✅ Active | `/v1/recall`, `/v1/search` | 4-layer: semantic 55%, keyword 25%, recency 10%, strength 10% |
| 2 | Temporal Query Parsing | `turkish.py` → `api.py` → `search.py` | ✅ Active | `/v1/recall` | `parse_temporal()` + `dateparser` lib, Turkish + English, time_range filter |
| 3 | Search Result Cache | `storage.py`, `search.py` | ✅ Active | Read + invalidate | 3-layer: LRU in-memory + SQLite embedding_cache + search_result_cache (1hr TTL) |
| 4 | Entity Extraction (NER) | `entities.py` | ✅ Active | `/v1/capture` | 7 types: person, place, org, tech, product, date, concept |
| 5 | Knowledge Graph | `entities.py` | ✅ Active | `/v1/capture` | 13 typed relations, co-occurrence links |
| 6 | Contradiction Detection | `conflict_detector.py` | ✅ Active | Via `entities.py` | Temporal fact conflicts, auto-resolution, confidence margin 0.20 |
| 7 | Temporal Facts | `storage.py` | ✅ Active | Via conflict_detector | `temporal_facts` table: valid_from, valid_to, is_active, typed relations |
| 8 | Rule/Instruction Capture | `rules.py` | ✅ Active | `/v1/store`, `/v1/rule` | Turkish + English patterns, safewords: `/rule`, `/save`, `📌` |
| 9 | Importance Scoring | `triggers.py` | ✅ Active | `/v1/capture` | Recalibrated Feb 17 — base 0.20, semantic-first |
| 10 | Write-time Semantic Merge | `storage.py` | ✅ Active | `/v1/capture`, `/v1/store` | `merge_or_store()`: MD5 + cosine similarity dedup |
| 11 | Ebbinghaus Decay | `storage.py` | ✅ Active | `/v1/decay` | Importance-adjusted rates, GC phase for zombies |
| 12 | Consolidation | `api.py` | ✅ Active | `/v1/consolidate` | Unidirectional Jaccard + category-aware thresholds |
| 13 | Per-Agent DB Routing | `pool.py` | ✅ Active | All endpoints | `memory.sqlite` (main) + `memory-{agent_id}.sqlite` per specialist |
| 14 | Cross-Agent Search | `api.py` | ✅ Active | `/v1/recall?agent=all` | Queries all agent DBs, merges by score |
| 15 | GC (Garbage Collection) | `api.py`, `storage.py` | ✅ Active | `/v1/gc` | `gc_purge()`: permanent delete of soft-deleted memories after N days |
| 16 | Turkish NLP | `turkish.py` | ⚠️ Partial | Only temporal parsing wired | `lemmatize()`, `ascii_fold()`, `normalize_text()` available but not used in search |
| 17 | Audit Logging | `middleware.py` | ✅ Active | Global | `/var/log/asuman-memory-audit.log` |
| 18 | Analytics | `api.py` | ✅ Active | `/v1/stats`, `/v1/metrics` | Memory counts, entity stats, operational metrics |

## NOT Implemented (and why)

| Feature | Reason | Worth doing? |
|---------|--------|-------------|
| HYDE (Hypothetical Document Embedding) | RRF 4-layer already performs well | Low — marginal improvement for complexity |
| Cross-encoder Reranking | RRF fusion is sufficient at current scale (4814 memories) | Low — revisit if >50K memories |
| Proactive Memory Push | Complex, needs gateway integration | Medium — would require hook or bootstrap changes |
| Turkish Lemmatization in Search | `zeyrek` is slow, FTS5 trigram tokenizer handles Turkish OK | Low — can enable later if search quality drops |

## Dead Code

| File | Status | Notes |
|------|--------|-------|
| `ingest.py` | Superseded | Batch JSONL ingestion — functionality absorbed into `/v1/capture`. Keep for manual bulk imports via `scripts/initial_load.py`. |

## API Endpoints (11 total)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/recall` | Hybrid search (temporal-aware) |
| POST | `/v1/capture` | Batch ingest with semantic merge |
| POST | `/v1/store` | Store single memory |
| DELETE | `/v1/forget` | Delete memory |
| GET | `/v1/search` | Interactive search |
| POST | `/v1/rule` | Store instruction/rule |
| POST | `/v1/decay` | Run Ebbinghaus decay |
| POST | `/v1/consolidate` | Deduplicate + archive stale |
| POST | `/v1/gc` | Permanent delete of soft-deleted |
| GET | `/v1/stats` | Statistics |
| GET | `/v1/health` | Health check |
| GET | `/v1/agents` | List agent databases |
| GET | `/v1/metrics` | Operational metrics |
