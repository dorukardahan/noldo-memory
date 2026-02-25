"""FastAPI HTTP API for the OpenClaw memory system.

Endpoints:
    POST   /v1/recall        -- Hybrid search (semantic + BM25 + recency + strength + importance)
    POST   /v1/capture       -- Batch ingest (write-time semantic merge)
    POST   /v1/store         -- Store one memory (write-time semantic merge)
    DELETE /v1/forget        -- Delete memory
    GET    /v1/search        -- Interactive search
    GET    /v1/stats         -- Statistics
    GET    /v1/health        -- Health check (fast)
    GET    /v1/health/deep   -- Deep health check (DB/embedding diagnostics)
    GET    /v1/agents        -- List known agent memory databases
    POST   /v1/decay         -- Run Ebbinghaus strength decay
    POST   /v1/consolidate   -- Deduplicate + archive stale memories

All endpoints accept an optional ``agent`` parameter for per-agent DB routing:
    - None / "main"  -> main memory database (default, backward compatible)
    - "{agent_id}"   -> agent-specific database (memory-{agent_id}.sqlite)
    - "all"          -> cross-agent search (recall/search/stats/decay/consolidate only)

Run: ``python -m agent_memory.api``
"""

from __future__ import annotations

import asyncio
import os
import logging
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field

from starlette.responses import JSONResponse
from .middleware import APIKeyMiddleware, RateLimitMiddleware, AuditLogMiddleware

from .config import Config, load_config
from .embed_worker import EmbedWorker
from .embeddings import OpenRouterEmbeddings
from .entities import KnowledgeGraph
from .pool import StoragePool
from .reranker import CrossEncoderReranker
from .search import HybridSearch, SearchWeights
from .storage import MemoryStorage
from .rules import RuleDetector
from .triggers import score_importance, should_trigger
from .turkish import parse_temporal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state (initialised in lifespan)
# ---------------------------------------------------------------------------

_storage_pool: Optional[StoragePool] = None
_embedder: Optional[OpenRouterEmbeddings] = None
_embed_worker: Optional[EmbedWorker] = None
_config: Optional[Config] = None
_start_time: float = 0.0

# Per-agent caches (lazily populated)
_search_cache: Dict[str, HybridSearch] = {}
_kg_cache: Dict[str, KnowledgeGraph] = {}
_search_weights: Optional[SearchWeights] = None
_reranker: Optional[CrossEncoderReranker] = None
_bg_reranker: Optional[CrossEncoderReranker] = None
_rule_detector = RuleDetector()

# Audit logging handler (file-based, graceful fallback for CI/test)
try:
    _default_audit_dir = os.environ.get("AGENT_MEMORY_DATA_DIR") or (
        str(Path.home() / ".asuman") if (Path.home() / ".asuman").exists() and not (Path.home() / ".agent-memory").exists()
        else str(Path.home() / ".agent-memory")
    )
    _audit_log_path = os.environ.get("AGENT_MEMORY_AUDIT_LOG", os.path.join(_default_audit_dir, "agent-memory-audit.log"))
    _audit_handler = logging.FileHandler(_audit_log_path)
    _audit_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger("audit").addHandler(_audit_handler)
except OSError:
    pass  # CI/test or hardened environment - audit log not available
logging.getLogger("audit").setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Agent routing helpers
# ---------------------------------------------------------------------------

def _get_storage(agent: Optional[str] = None) -> MemoryStorage:
    """Get the MemoryStorage for the given agent."""
    if _storage_pool is None:
        raise HTTPException(503, "Storage pool not initialised")
    try:
        return _storage_pool.get(agent)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


def _get_search(agent: Optional[str] = None) -> HybridSearch:
    """Get or create a HybridSearch for the given agent."""
    if _storage_pool is None or _search_weights is None:
        raise HTTPException(503, "Search not initialised")
    key = StoragePool.normalize_key(agent)
    if key not in _search_cache:
        storage = _storage_pool.get(agent)
        _search_cache[key] = HybridSearch(
            storage=storage,
            embedder=_embedder,
            weights=_search_weights,
            reranker=_reranker,
            rerank_weight=(_config.reranker_weight if _config else 0.20),
            bg_reranker=_bg_reranker,
            bg_two_pass_enabled=(_config.reranker_two_pass_enabled if _config else False),
            bg_rerank_weight=(_config.reranker_two_pass_weight if _config else 0.35),
        )
    return _search_cache[key]


def _get_kg(agent: Optional[str] = None) -> KnowledgeGraph:
    """Get or create a KnowledgeGraph for the given agent."""
    if _storage_pool is None:
        raise HTTPException(503, "Storage not initialised")
    key = StoragePool.normalize_key(agent)
    if key not in _kg_cache:
        storage = _storage_pool.get(agent)
        _kg_cache[key] = KnowledgeGraph(storage=storage)
    return _kg_cache[key]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic."""
    global _storage_pool, _embedder, _embed_worker, _config, _start_time, _search_weights, _reranker, _bg_reranker

    _config = load_config()
    errors = _config.validate()
    if errors:
        logger.warning("Config validation warnings: %s", errors)

    # Derive base directory from db_path's parent
    base_dir = str(Path(_config.db_path).parent)

    _storage_pool = StoragePool(
        base_dir=base_dir,
        dimensions=_config.embedding_dimensions,
    )

    # Pre-open main storage to ensure schema exists
    _storage_pool.get("main")

    if _config.openrouter_api_key:
        _embedder = OpenRouterEmbeddings(
            api_key=_config.openrouter_api_key,
            model=_config.embedding_model,
            dimensions=_config.embedding_dimensions,
        )
        _embedder.set_storage(_storage_pool.get("main"))
    else:
        logger.warning("No OPENROUTER_API_KEY -- semantic search disabled")
        _embedder = None

    _search_weights = SearchWeights(
        semantic=_config.weight_semantic,
        keyword=_config.weight_keyword,
        recency=_config.weight_recency,
        strength=_config.weight_strength,
        importance=_config.weight_importance,
    )

    _reranker = CrossEncoderReranker(
        enabled=_config.reranker_enabled,
        model_name=_config.reranker_model,
        top_k=_config.reranker_top_k,
        torch_threads=_config.reranker_threads,
        max_doc_chars=_config.reranker_max_doc_chars,
    )

    if _config.reranker_enabled and _config.reranker_prewarm:
        t0 = time.time()
        ok = _reranker.warmup()
        dt = round(time.time() - t0, 2)
        if ok:
            logger.info("Primary reranker prewarmed in %ss", dt)
        else:
            logger.warning("Primary reranker prewarm failed in %ss (will fallback)", dt)

    _bg_reranker = None
    if _config.reranker_two_pass_enabled:
        primary_model = CrossEncoderReranker._resolve_model_name(_config.reranker_model)
        bg_model = CrossEncoderReranker._resolve_model_name(_config.reranker_two_pass_model)
        if bg_model != primary_model:
            _bg_reranker = CrossEncoderReranker(
                enabled=True,
                model_name=_config.reranker_two_pass_model,
                top_k=_config.reranker_two_pass_top_k,
                torch_threads=_config.reranker_two_pass_threads,
                max_doc_chars=_config.reranker_two_pass_max_doc_chars,
            )
            if _config.reranker_two_pass_prewarm:
                t0 = time.time()
                ok = _bg_reranker.warmup()
                dt = round(time.time() - t0, 2)
                if ok:
                    logger.info("Two-pass background reranker prewarmed in %ss", dt)
                else:
                    logger.warning("Two-pass background prewarm failed in %ss", dt)

    _start_time = time.time()

    _embed_worker = None
    if _config.embed_worker_enabled:
        if _embedder is None:
            logger.info("Embed worker enabled but embedder unavailable; worker not started")
        else:
            try:
                _embed_worker = EmbedWorker(
                    storage_pool=_storage_pool,
                    embedder=_embedder,
                    interval_seconds=_config.embed_worker_interval,
                    batch_size=2,
                    max_sub_batch=2,
                    sleep_between=1.0,
                )
                _embed_worker.start()
            except Exception as exc:
                logger.warning("Failed to start embed worker: %s", exc)
                _embed_worker = None
    else:
        logger.info("Embed worker disabled via AGENT_MEMORY_EMBED_WORKER_ENABLED")

    agents = _storage_pool.get_all_agents()
    logger.info(
        "Memory API ready -- base_dir=%s agents=%s emb_model=%s reranker=%s/%s top_k=%s threads=%s max_chars=%s prewarm=%s two_pass=%s bg_model=%s bg_top_k=%s",
        base_dir,
        agents,
        _config.embedding_model,
        "on" if _config.reranker_enabled else "off",
        _config.reranker_model,
        _config.reranker_top_k,
        _config.reranker_threads,
        _config.reranker_max_doc_chars,
        _config.reranker_prewarm,
        _config.reranker_two_pass_enabled,
        _config.reranker_two_pass_model,
        _config.reranker_two_pass_top_k,
    )

    # Start warmup background task
    asyncio.create_task(warmup_loop())

    yield

    # Shutdown
    if _embed_worker is not None:
        try:
            await _embed_worker.stop()
        except Exception as exc:
            logger.warning("Embed worker shutdown error: %s", exc)
        finally:
            _embed_worker = None

    _storage_pool.close_all()
    _search_cache.clear()
    _kg_cache.clear()


async def warmup_loop():
    """Background task to keep the embedding model warm."""
    while True:
        await asyncio.sleep(600)  # 10 minutes
        if _embedder:
            try:
                # Use a specific string that will likely be in cache after first call
                await _embedder.embed("warmup")
                logger.debug("Embedding warmup successful")
            except Exception as exc:
                logger.warning("Embedding warmup failed: %s", exc)


app = FastAPI(
    title="OpenClaw Memory API",
    version="0.3.0",
    lifespan=lifespan,
)

# --- Security middleware (order matters: last added = first to run) ---
# Audit logging (runs on every request)
app.add_middleware(AuditLogMiddleware)

# Rate limiting: 120 requests/minute per IP
app.add_middleware(RateLimitMiddleware, max_requests=120, window_seconds=60)

# API key authentication
_api_key = os.environ.get("AGENT_MEMORY_API_KEY", "")
_extra_keys_path = os.path.join(
    os.environ.get("AGENT_MEMORY_DATA_DIR", os.path.expanduser("~/.asuman")),
    "memory-api-keys.json",
)
if _api_key:
    app.add_middleware(APIKeyMiddleware, api_key=_api_key, extra_keys_path=_extra_keys_path)
    logger.info("API key authentication enabled (extra keys: %s)", _extra_keys_path)
else:
    logger.warning("No AGENT_MEMORY_API_KEY set -- API is UNAUTHENTICATED")

# CORS: restrict to localhost origins only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# --- Centralized error handling ---

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    logger.warning("HTTP %d: %s (path=%s)", exc.status_code, exc.detail, request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": str(exc.detail), "status_code": exc.status_code},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.warning("Validation error: %s (path=%s)", str(exc)[:200], request.url.path)
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "detail": exc.errors()},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.exception("Unhandled error: %s (path=%s)", exc, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )



# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RecallRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=100, le=100000,
                                       description="Trim results to fit within this token budget")
    namespace: Optional[str] = Field(default=None, description="Filter by namespace (None = all)")
    agent: Optional[str] = None


class CaptureRequest(BaseModel):
    messages: List[Dict[str, Any]]
    agent: Optional[str] = None


class StoreRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    category: str = "other"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    namespace: str = Field(default="default", description="Namespace for topic-based grouping")
    agent: Optional[str] = None


class ForgetRequest(BaseModel):
    id: Optional[str] = None
    query: Optional[str] = None
    agent: Optional[str] = None


class DecayRequest(BaseModel):
    agent: Optional[str] = None


class ConsolidateRequest(BaseModel):
    agent: Optional[str] = None


class CompressRequest(BaseModel):
    agent: Optional[str] = None
    age_days: int = Field(default=30, ge=7, description="Compress memories older than this many days")
    min_chars: int = Field(default=500, ge=100, description="Only compress memories longer than this")
    dry_run: bool = Field(default=False, description="Preview without actually compressing")


class GCRequest(BaseModel):
    agent: Optional[str] = None
    soft_deleted_days: int = Field(default=30, ge=7, description="Purge memories soft-deleted for this many days")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/recall")
async def recall(req: RecallRequest) -> Dict[str, Any]:
    """Search memories using hybrid search (semantic + BM25 + recency).

    Use ``agent="all"`` to search across all agent databases.
    """
    if req.agent == "all":
        return await _recall_all(req)

    search = _get_search(req.agent)

    # Temporal-aware search: parse time expressions from query
    time_range = None
    temporal = parse_temporal(req.query)
    if temporal is not None:
        time_range = (temporal[0].timestamp(), temporal[1].timestamp())

    results = await search.search(
        query=req.query,
        limit=req.limit,
        min_score=req.min_score,
        time_range=time_range,
        namespace=req.namespace,
    )

    agent_key = StoragePool.normalize_key(req.agent)
    result_dicts = [r.to_dict() for r in results]

    # Apply token budget trimming if requested
    trimmed = False
    if req.max_tokens is not None and result_dicts:
        from .token_utils import trim_results_to_budget
        original_count = len(result_dicts)
        result_dicts = trim_results_to_budget(result_dicts, req.max_tokens)
        trimmed = len(result_dicts) < original_count

    response = {
        "query": req.query,
        "agent": agent_key,
        "count": len(result_dicts),
        "triggered": should_trigger(req.query),
        "search_mode": search.last_search_mode,
        "results": result_dicts,
    }
    if search.last_search_degraded:
        response["degraded"] = True
    if trimmed:
        response["trimmed"] = True
        response["max_tokens"] = req.max_tokens
    if time_range is not None:
        response["time_range"] = {
            "start": time_range[0],
            "end": time_range[1],
        }
    return response


async def _recall_all(req: RecallRequest) -> Dict[str, Any]:
    """Cross-agent recall: query all agent DBs, merge by score."""
    if _storage_pool is None:
        raise HTTPException(503, "Storage pool not initialised")

    # Temporal-aware cross-agent search
    time_range = None
    temporal = parse_temporal(req.query)
    if temporal is not None:
        time_range = (temporal[0].timestamp(), temporal[1].timestamp())

    all_results: List[Dict[str, Any]] = []
    for agent_id in _storage_pool.get_all_agents():
        try:
            search = _get_search(agent_id)
            results = await search.search(
                query=req.query,
                limit=req.limit,
                min_score=req.min_score,
                time_range=time_range,
            )
            for r in results:
                d = r.to_dict()
                d["agent"] = agent_id
                all_results.append(d)
        except Exception as exc:
            logger.warning("Cross-agent recall failed for %s: %s", agent_id, exc)

    # Sort by score descending, take top N
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    all_results = all_results[: req.limit]

    response = {
        "query": req.query,
        "agent": "all",
        "count": len(all_results),
        "triggered": should_trigger(req.query),
        "results": all_results,
        "cross_agent": True,
    }
    if time_range is not None:
        response["time_range"] = {
            "start": time_range[0],
            "end": time_range[1],
        }
    return response


@app.post("/v1/capture")
async def capture(req: CaptureRequest) -> Dict[str, Any]:
    """Ingest a batch of messages into memory.

    Each message dict should have at least ``text`` and ``role``.
    """
    if req.agent == "all":
        raise HTTPException(400, "Cannot capture to 'all' -- specify an agent")

    storage = _get_storage(req.agent)

    # Pre-filter / normalize
    cleaned: List[Dict[str, Any]] = []
    for msg in req.messages:
        text = (msg.get("text") or msg.get("content") or "").strip()
        if len(text) < 3:
            continue
        role = msg.get("role", "user")
        cleaned.append({
            "text": text[:4000],  # raised from 2000 [S10, 2026-02-17]
            "role": role,
            "session": msg.get("session", ""),
            "timestamp": msg.get("timestamp", ""),
        })

    if not cleaned:
        return {"stored": 0, "merged": 0, "total": len(req.messages)}

    texts = [m["text"] for m in cleaned]

    vectors: List[Optional[List[float]]] = [None] * len(texts)
    if _embedder is not None:
        for attempt in range(3):
            try:
                vectors = await _embedder.embed_batch(texts)
                break
            except Exception as exc:
                if attempt < 2:
                    logger.warning("Batch embed retry %d/3: %s", attempt + 1, exc)
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    logger.error("Batch embed failed after 3 attempts: %s", exc)
                    vectors = [None] * len(texts)

    items: List[Dict[str, Any]] = []
    for m, vec in zip(cleaned, vectors):
        importance = score_importance(m["text"], {"role": m["role"]})
        items.append({
            "text": m["text"],
            "vector": vec,
            "category": m["role"],
            "importance": importance,
            "source_session": m["session"],
        })

    # Store (write-time semantic merge, per message)
    stored_n = 0
    merged_n = 0
    from .ingest import classify_memory_type

    for it in items:
        res = storage.merge_or_store(
            text=it["text"],
            vector=it.get("vector"),
            category=it.get("category", "other"),
            importance=float(it.get("importance", 0.5)),
            source_session=it.get("source_session"),
            memory_type=classify_memory_type(it["text"]),
        )
        if res.get("action") == "merged":
            merged_n += 1
        else:
            stored_n += 1

    # Knowledge graph (best-effort)
    kg = _get_kg(req.agent)
    for m in cleaned:
        try:
            kg.process_text(m["text"], timestamp=m.get("timestamp", ""))
        except Exception:
            pass

    # Invalidate search cache
    storage.invalidate_search_cache(agent=req.agent or "main")

    agent_key = StoragePool.normalize_key(req.agent)
    return {
        "stored": stored_n,
        "merged": merged_n,
        "total": len(req.messages),
        "agent": agent_key,
    }


@app.post("/v1/store")
async def store(req: StoreRequest) -> Dict[str, Any]:
    """Manually store a single memory."""
    if req.agent == "all":
        raise HTTPException(400, "Cannot store to 'all' -- specify an agent")

    storage = _get_storage(req.agent)

    vector = None
    if _embedder:
        for attempt in range(3):
            try:
                vector = await _embedder.embed(req.text)
                break
            except Exception as exc:
                if attempt < 2:
                    logger.warning("Embed retry %d/3: %s", attempt + 1, exc)
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    logger.error("Embed failed after 3 attempts: %s", exc)

    # Rule detection: override category/importance if instruction detected
    category = req.category
    importance = req.importance
    detected = _rule_detector.detect(req.text)
    if detected or _rule_detector.check_safeword(req.text):
        category = "rule"
        importance = 1.0
        logger.info("Rule detected in /v1/store: %s", req.text[:60])

    from .ingest import classify_memory_type

    res = storage.merge_or_store(
        text=req.text,
        vector=vector,
        category=category,
        importance=importance,
        source_session=None,
        namespace=req.namespace,
        memory_type=classify_memory_type(req.text),
    )

    # Invalidate search cache
    storage.invalidate_search_cache(agent=req.agent or "main")

    agent_key = StoragePool.normalize_key(req.agent)
    return {
        "id": res["id"],
        "stored": res["action"] == "inserted",
        "merged": res["action"] == "merged",
        "similarity": res.get("similarity"),
        "agent": agent_key,
    }


@app.post("/v1/rule")
async def store_rule(req: StoreRequest) -> Dict[str, Any]:
    """Explicitly store a rule/instruction (safeword bypass)."""
    if req.agent == "all":
        raise HTTPException(400, "Cannot store to 'all' -- specify an agent")

    storage = _get_storage(req.agent)

    vector = None
    if _embedder:
        for attempt in range(3):
            try:
                vector = await _embedder.embed(req.text)
                break
            except Exception as exc:
                if attempt < 2:
                    logger.warning("Embed retry %d/3: %s", attempt + 1, exc)
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    logger.error("Embed failed after 3 attempts: %s", exc)

    res = storage.merge_or_store(
        text=req.text,
        vector=vector,
        category="rule",
        importance=1.0,
        source_session=None,
    )

    storage.invalidate_search_cache(agent=req.agent or "main")

    agent_key = StoragePool.normalize_key(req.agent)
    logger.info("Rule stored via /v1/rule: %s", req.text[:60])
    return {
        "id": res["id"],
        "stored": res["action"] == "inserted",
        "merged": res["action"] == "merged",
        "agent": agent_key,
    }


@app.post("/v1/pin")
async def pin_memory(req: ForgetRequest) -> Dict[str, Any]:
    """Pin a memory: protects it from decay, gc, and consolidation."""
    if not req.id:
        raise HTTPException(400, "Provide 'id'")
    storage = _get_storage(req.agent)
    pinned = storage.pin_memory(req.id)
    return {"pinned": pinned, "id": req.id}


@app.post("/v1/unpin")
async def unpin_memory(req: ForgetRequest) -> Dict[str, Any]:
    """Unpin a memory: allows decay/gc/consolidation again."""
    if not req.id:
        raise HTTPException(400, "Provide 'id'")
    storage = _get_storage(req.agent)
    unpinned = storage.unpin_memory(req.id)
    return {"unpinned": unpinned, "id": req.id}


@app.delete("/v1/forget")
async def forget(req: ForgetRequest) -> Dict[str, Any]:
    """Delete a memory by ID or by searching for it."""
    if req.agent == "all":
        raise HTTPException(400, "Cannot forget from 'all' -- specify an agent")

    storage = _get_storage(req.agent)

    if req.id:
        deleted = storage.delete_memory(req.id)
        if deleted:
            storage.invalidate_search_cache(agent=req.agent or "main")
        return {"deleted": deleted, "id": req.id}

    if req.query:
        results = storage.search_text(req.query, limit=1)
        if results:
            mid = results[0]["id"]
            deleted = storage.delete_memory(mid)
            if deleted:
                storage.invalidate_search_cache(agent=req.agent or "main")
            return {"deleted": deleted, "id": mid}
        return {"deleted": False, "reason": "no match found"}

    raise HTTPException(400, "Provide 'id' or 'query'")


@app.get("/v1/search")
async def search_interactive(
    query: str = Query(..., min_length=1),
    limit: int = Query(default=5, ge=1, le=50),
    agent: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Interactive search (for CLI/debug).

    Use ``agent=all`` to search across all agent databases.
    """
    if agent == "all":
        # Reuse recall logic
        req = RecallRequest(query=query, limit=limit, agent="all")
        return await _recall_all(req)

    search = _get_search(agent)
    results = await search.search(query=query, limit=limit)

    agent_key = StoragePool.normalize_key(agent)
    return {
        "query": query,
        "agent": agent_key,
        "count": len(results),
        "results": [r.to_dict() for r in results],
    }


@app.post("/v1/decay")
async def decay(req: DecayRequest = DecayRequest()) -> Dict[str, Any]:
    """Run weekly Ebbinghaus strength decay (cron-friendly).

    Use ``agent="all"`` to decay across all agent databases.
    """
    if _storage_pool is None:
        raise HTTPException(503, "Storage pool not initialised")

    if req.agent == "all":
        total_decayed = 0
        total_memories = 0
        per_agent: Dict[str, int] = {}
        for agent_id in _storage_pool.get_all_agents():
            storage = _storage_pool.get(agent_id)
            decayed = storage.decay_all(days_threshold=7, decay_amount=0.05, min_strength=0.3)
            total = storage.stats().get("total_memories", 0)
            total_decayed += decayed
            total_memories += total
            per_agent[agent_id] = decayed
        return {"decayed": total_decayed, "total": total_memories, "per_agent": per_agent}

    storage = _get_storage(req.agent)
    decayed = storage.decay_all(days_threshold=7, decay_amount=0.05, min_strength=0.3)
    total = storage.stats().get("total_memories", 0)

    agent_key = StoragePool.normalize_key(req.agent)
    return {"decayed": decayed, "total": total, "agent": agent_key}


@app.post("/v1/gc")
async def gc(req: GCRequest = GCRequest()) -> Dict[str, Any]:
    """Permanently DELETE memories soft-deleted for 30+ days and clean orphaned vectors.

    Use ``agent="all"`` to GC across all agent databases.
    """
    if _storage_pool is None:
        raise HTTPException(503, "Storage pool not initialised")

    if req.agent == "all":
        total_purged = 0
        total_vectors = 0
        per_agent: Dict[str, Dict[str, int]] = {}
        for agent_id in _storage_pool.get_all_agents():
            storage = _storage_pool.get(agent_id)
            result = storage.gc_purge(soft_deleted_days=req.soft_deleted_days)
            total_purged += result["purged_memories"]
            total_vectors += result["purged_vectors"]
            per_agent[agent_id] = result
        return {
            "purged_memories": total_purged,
            "purged_vectors": total_vectors,
            "per_agent": per_agent,
        }

    storage = _get_storage(req.agent)
    result = storage.gc_purge(soft_deleted_days=req.soft_deleted_days)

    agent_key = StoragePool.normalize_key(req.agent)
    return {
        "purged_memories": result["purged_memories"],
        "purged_vectors": result["purged_vectors"],
        "agent": agent_key,
    }


@app.post("/v1/consolidate")
async def consolidate(req: ConsolidateRequest = ConsolidateRequest()) -> Dict[str, Any]:
    """Deduplicate and cleanup memories.

    1) Find memory pairs with cosine similarity > threshold
       (same category: 0.82, cross-category: 0.92)
    2) Merge duplicates (keep higher importance; combine text)
    3) Soft-delete memories with strength < 0.3 and not accessed in 30 days

    Use ``agent="all"`` to consolidate all agent databases.
    """
    if _storage_pool is None:
        raise HTTPException(503, "Storage pool not initialised")

    if req.agent == "all":
        total_merged = 0
        total_archived = 0
        per_agent: Dict[str, Dict[str, int]] = {}
        for agent_id in _storage_pool.get_all_agents():
            try:
                result = _consolidate_single(agent_id)
                total_merged += result["merged"]
                total_archived += result["archived"]
                per_agent[agent_id] = result
            except Exception as exc:
                logger.warning("Consolidation failed for %s: %s", agent_id, exc)
                per_agent[agent_id] = {"error": str(exc)}
        return {
            "merged": total_merged,
            "archived": total_archived,
            "per_agent": per_agent,
        }

    agent_key = StoragePool.normalize_key(req.agent)
    result = _consolidate_single(agent_key)
    result["agent"] = agent_key
    return result


@app.post("/v1/compress")
async def compress(req: CompressRequest = CompressRequest()) -> Dict[str, Any]:
    """Compress old, long memories by replacing text with summary.

    Pinned and high-importance (>=0.8) memories are skipped.
    Use dry_run=true to preview without changes.
    """
    from .compression import compress_old_memories

    if _storage_pool is None:
        raise HTTPException(503, "Storage pool not initialised")

    if req.agent == "all":
        total = {"compressed": 0, "skipped": 0, "saved_chars": 0}
        for agent_id in _storage_pool.get_all_agents():
            try:
                storage = _storage_pool.get(agent_id)
                result = compress_old_memories(
                    storage, agent=agent_id,
                    age_days=req.age_days, min_chars=req.min_chars,
                    dry_run=req.dry_run,
                )
                total["compressed"] += result["compressed"]
                total["skipped"] += result["skipped"]
                total["saved_chars"] += result["saved_chars"]
            except Exception as exc:
                logger.warning("Compression failed for %s: %s", agent_id, exc)
        total["dry_run"] = req.dry_run
        return total

    agent_key = StoragePool.normalize_key(req.agent)
    storage = _get_storage(agent_key)
    result = compress_old_memories(
        storage, agent=agent_key,
        age_days=req.age_days, min_chars=req.min_chars,
        dry_run=req.dry_run,
    )
    result["agent"] = agent_key
    return result


def _consolidate_single(agent_id: str) -> Dict[str, Any]:
    """Run consolidation on a single agent's database."""
    storage = _storage_pool.get(agent_id)
    conn = storage._get_conn()
    now = time.time()

    total_before = conn.execute(
        "SELECT COUNT(*) AS c FROM memories WHERE deleted_at IS NULL"
    ).fetchone()["c"]

    # Build candidate list
    rows = conn.execute(
        """
        SELECT id, text, category, importance, vector_rowid, strength, last_accessed_at, created_at
          FROM memories
         WHERE deleted_at IS NULL AND vector_rowid IS NOT NULL
        """
    ).fetchall()

    # Union-find for clusters
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Similarity discovery via sqlite-vec NN search
    pairs = 0
    for r in rows:
        mid = r["id"]
        vec_rowid = r["vector_rowid"]
        emb_row = conn.execute(
            "SELECT embedding FROM memory_vectors WHERE rowid = ?",
            (vec_rowid,),
        ).fetchone()
        if not emb_row or emb_row[0] is None:
            continue
        blob = emb_row[0]

        nn = conn.execute(
            """
            SELECT mv.rowid AS vec_rowid, mv.distance AS distance
              FROM memory_vectors mv
             WHERE mv.embedding MATCH ?
             ORDER BY mv.distance
             LIMIT 6
            """,
            (blob,),
        ).fetchall()

        for n in nn:
            if n["vec_rowid"] == vec_rowid:
                continue
            sim = 1.0 - (float(n["distance"]) / 2.0)
            if sim <= 0.82:
                continue
            other = conn.execute(
                "SELECT id, category FROM memories WHERE vector_rowid = ? AND deleted_at IS NULL",
                (n["vec_rowid"],),
            ).fetchone()
            if not other:
                continue
            oid = other["id"]
            if oid == mid:
                continue
            # Category-aware threshold: same category = 0.82, different = 0.92
            same_cat = (r["category"] or "other") == (other["category"] or "other")
            threshold = 0.82 if same_cat else 0.92
            if sim < threshold:
                continue
            union(mid, oid)
            pairs += 1

    # Group clusters
    clusters: Dict[str, List[sqlite3.Row]] = {}
    by_id: Dict[str, sqlite3.Row] = {r["id"]: r for r in rows}
    for mid in by_id.keys():
        root = find(mid)
        clusters.setdefault(root, []).append(by_id[mid])

    merged = 0
    try:
        conn.execute("BEGIN")

        for root, members in clusters.items():
            if len(members) < 2:
                continue

            keeper = max(members, key=lambda rr: float(rr["importance"] or 0.0))
            keeper_id = keeper["id"]

            texts: List[str] = []
            imps: List[float] = []
            strengths: List[float] = []
            last_accs: List[float] = []
            vecs: List[np.ndarray] = []

            for m in members:
                texts.append((m["text"] or "").strip())
                imps.append(float(m["importance"] or 0.5))
                strengths.append(float(m["strength"] or 1.0))
                last_accs.append(float(m["last_accessed_at"] or m["created_at"] or 0.0))

                emb_row = conn.execute(
                    "SELECT embedding FROM memory_vectors WHERE rowid = ?",
                    (m["vector_rowid"],),
                ).fetchone()
                if emb_row and emb_row[0] is not None:
                    vecs.append(np.frombuffer(emb_row[0], dtype=np.float32))

            merged_text = "\n\u2022 ".join([t for t in texts if t])
            new_importance = max(imps) if imps else float(keeper["importance"] or 0.5)
            new_strength = min(max(strengths) + 0.2, 5.0) if strengths else 1.0
            new_last_acc = max(last_accs) if last_accs else now

            conn.execute(
                """
                UPDATE memories
                   SET text = ?,
                       importance = ?,
                       strength = ?,
                       last_accessed_at = ?,
                       updated_at = ?
                 WHERE id = ?
                """,
                (merged_text, new_importance, new_strength, new_last_acc, now, keeper_id),
            )
            conn.execute("DELETE FROM memory_fts WHERE id = ?", (keeper_id,))
            conn.execute(
                "INSERT OR REPLACE INTO memory_fts(id, text) VALUES (?, ?)",
                (keeper_id, merged_text),
            )

            if vecs:
                try:
                    avg = np.mean(np.vstack(vecs), axis=0).astype(np.float32)
                    conn.execute(
                        "UPDATE memory_vectors SET embedding = ? WHERE rowid = ?",
                        (avg.tobytes(), keeper["vector_rowid"]),
                    )
                except Exception:
                    pass

            for m in members:
                if m["id"] == keeper_id:
                    continue
                conn.execute(
                    "UPDATE memories SET deleted_at = ? WHERE id = ?",
                    (now, m["id"]),
                )
                conn.execute("DELETE FROM memory_fts WHERE id = ?", (m["id"],))
                merged += 1

        # Archive weak, stale memories
        cutoff = now - (30 * 86400.0)
        cur = conn.execute(
            """
            UPDATE memories
               SET deleted_at = ?
             WHERE deleted_at IS NULL
               AND COALESCE(strength, 1.0) < 0.3
               AND COALESCE(last_accessed_at, created_at) < ?
            """,
            (now, cutoff),
        )
        archived = int(cur.rowcount or 0)

        conn.commit()

    except Exception as exc:
        conn.rollback()
        logger.exception("Consolidation failed: %s", exc)
        raise HTTPException(500, "consolidation failed")

    total_after = conn.execute(
        "SELECT COUNT(*) AS c FROM memories WHERE deleted_at IS NULL"
    ).fetchone()["c"]

    logger.info(
        "Consolidation complete [%s]: merged=%d archived=%d before=%d after=%d pairs=%d",
        agent_id,
        merged,
        archived,
        total_before,
        total_after,
        pairs,
    )

    return {
        "merged": merged,
        "archived": archived,
        "total_before": total_before,
        "total_after": total_after,
    }


@app.get("/v1/stats")
async def stats(
    agent: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Return memory statistics.

    Use ``agent=all`` for aggregated stats across all agents.
    """
    if _storage_pool is None:
        raise HTTPException(503, "Storage pool not initialised")

    if agent == "all":
        per_agent: Dict[str, Dict[str, Any]] = {}
        totals = {"total_memories": 0, "entities": 0, "relationships": 0, "temporal_facts": 0}
        for agent_id in _storage_pool.get_all_agents():
            s = _storage_pool.get(agent_id).stats()
            per_agent[agent_id] = s
            totals["total_memories"] += s.get("total_memories", 0)
            totals["entities"] += s.get("entities", 0)
            totals["relationships"] += s.get("relationships", 0)
            totals["temporal_facts"] += s.get("temporal_facts", 0)
        return {**totals, "per_agent": per_agent}

    storage = _get_storage(agent)
    s = storage.stats()
    s["agent"] = StoragePool.normalize_key(agent)
    return s


@app.get("/v1/agents")
async def list_agents() -> Dict[str, Any]:
    """List all known agent memory databases."""
    if _storage_pool is None:
        raise HTTPException(503, "Storage pool not initialised")

    agents = _storage_pool.get_all_agents()
    details: Dict[str, Any] = {}
    for agent_id in agents:
        storage = _storage_pool.get(agent_id)
        s = storage.stats()
        details[agent_id] = {
            "total_memories": s.get("total_memories", 0),
            "entities": s.get("entities", 0),
            "db_path": storage.db_path,
        }

    return {"agents": agents, "count": len(agents), "details": details}


@app.get("/v1/health")
async def health() -> Dict[str, Any]:
    """Health check with real probes. [S12, 2026-02-17]

    Returns 200 with status "ok"|"degraded"|"down".
    Checks: storage (SQLite read), embedding (probe text), vectorless count.
    """
    checks: Dict[str, bool] = {"storage": False, "embedding": False}

    # Storage check — can we read from main DB?
    if _storage_pool:
        try:
            main_stats = _storage_pool.get("main").stats()
            checks["storage"] = True
            main_stats.get("total_memories", 0)  # validate stats accessible
            _storage_pool.get_all_agents()  # validate agent listing works
        except Exception:
            pass

    # Embedding check — can we actually embed a probe text?
    if _embedder:
        try:
            await asyncio.wait_for(_embedder.embed("health_probe"), timeout=5.0)
            checks["embedding"] = True
        except Exception:
            pass

    # Vectorless count — memories missing vector embeddings
    if _storage_pool and checks["storage"]:
        try:
            conn = _storage_pool.get("main")._get_conn()
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM memories WHERE vector_rowid IS NULL AND deleted_at IS NULL"
            ).fetchone()
            row["c"] if row else 0  # validate vectorless query works
        except Exception:
            pass

    all_ok = all(checks.values())
    overall = "ok" if all_ok else ("degraded" if checks["storage"] else "down")

    # Public health: only status + checks (no metadata leakage).
    # Detailed info (total_memories, agents, vectorless_count) is in /v1/stats (auth required).
    return {
        "status": overall,
        "checks": checks,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@app.get("/v1/health/deep")
async def health_deep() -> Dict[str, Any]:
    """Deep health check with diagnostics for operators.

    Includes SQLite quick integrity probe, embedding probe latency,
    vectorless memory count, DB file size and service uptime.
    """
    checks: Dict[str, Dict[str, Any]] = {
        "db_integrity": {"ok": False, "result": None, "latency_ms": None},
        "embedding": {"ok": False, "latency_ms": None},
        "vectorless": {"ok": False, "count": None},
        "disk_usage": {"ok": False, "db_path": None, "db_size_bytes": None},
    }

    storage_available = False

    if _storage_pool:
        try:
            storage = _storage_pool.get("main")
            conn = storage._get_conn()
            storage_available = True

            quick_check_start = time.perf_counter()
            quick_check_row = conn.execute("PRAGMA quick_check").fetchone()
            quick_check_latency = round((time.perf_counter() - quick_check_start) * 1000, 2)
            quick_check_result = str(quick_check_row[0]) if quick_check_row else "unknown"
            checks["db_integrity"] = {
                "ok": quick_check_result.lower() == "ok",
                "result": quick_check_result,
                "latency_ms": quick_check_latency,
            }

            vectorless_row = conn.execute(
                "SELECT COUNT(*) AS c FROM memories WHERE vector_rowid IS NULL"
            ).fetchone()
            checks["vectorless"] = {
                "ok": True,
                "count": int(vectorless_row["c"]) if vectorless_row else 0,
            }

            db_path = Path(storage.db_path)
            checks["disk_usage"] = {
                "ok": db_path.exists(),
                "db_path": str(db_path),
                "db_size_bytes": db_path.stat().st_size if db_path.exists() else None,
            }
        except Exception as exc:
            checks["db_integrity"]["error"] = str(exc)
    else:
        checks["db_integrity"]["error"] = "storage pool not initialised"

    if _embedder:
        embed_start = time.perf_counter()
        try:
            await asyncio.wait_for(_embedder.embed("health_probe_deep"), timeout=5.0)
            checks["embedding"] = {
                "ok": True,
                "latency_ms": round((time.perf_counter() - embed_start) * 1000, 2),
            }
        except Exception as exc:
            checks["embedding"] = {
                "ok": False,
                "latency_ms": round((time.perf_counter() - embed_start) * 1000, 2),
                "error": str(exc),
            }
    else:
        checks["embedding"]["error"] = "embedder not configured"

    core_ok = (
        checks["db_integrity"].get("ok", False)
        and checks["vectorless"].get("ok", False)
        and checks["disk_usage"].get("ok", False)
    )
    embedding_ok = checks["embedding"].get("ok", False)

    if core_ok and embedding_ok:
        status = "ok"
    elif core_ok or storage_available:
        status = "degraded"
    else:
        status = "down"

    return {
        "status": status,
        "uptime_seconds": round(time.time() - _start_time, 1),
        "checks": checks,
    }


@app.get("/v1/metrics")
async def metrics(
    agent: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Return operational metrics for monitoring."""
    if _storage_pool is None:
        raise HTTPException(503, "Storage pool not initialised")

    # Per-agent memory counts
    agent_counts = {}
    total_memories = 0
    if agent and agent != "all":
        s = _get_storage(agent).stats()
        agent_counts[StoragePool.normalize_key(agent)] = s.get("total_memories", 0)
        total_memories = s.get("total_memories", 0)
    else:
        for aid in _storage_pool.get_all_agents():
            s = _storage_pool.get(aid).stats()
            count = s.get("total_memories", 0)
            agent_counts[aid] = count
            total_memories += count

    return {
        "uptime_seconds": round(time.time() - _start_time, 1),
        "total_memories": total_memories,
        "agent_counts": agent_counts,
        "embedding_available": _embedder is not None,
    }



@app.get("/v1/export")
async def export_memories(
    agent: Optional[str] = Query(default=None),
    include_deleted: bool = Query(default=False),
) -> List[Dict[str, Any]]:
    """Export memories as JSON array (JSONL-compatible per line).

    Returns all non-deleted memories for the given agent.
    Use ``include_deleted=true`` to include soft-deleted records.
    """
    if _storage_pool is None:
        raise HTTPException(503, "Storage pool not initialised")

    agent_key = StoragePool.normalize_key(agent) if agent else None
    storage = _get_storage(agent_key)
    conn = storage._get_conn()

    where = "" if include_deleted else "WHERE deleted_at IS NULL"
    rows = conn.execute(
        f"""
        SELECT id, text, category, importance, strength,
               source_session, created_at, updated_at,
               last_accessed_at, deleted_at
          FROM memories {where}
         ORDER BY created_at ASC
        """
    ).fetchall()

    result = []
    for r in rows:
        result.append({
            "id": r["id"],
            "text": r["text"],
            "category": r["category"],
            "importance": r["importance"],
            "strength": r["strength"],
            "source_session": r["source_session"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "last_accessed_at": r["last_accessed_at"],
            "deleted_at": r["deleted_at"],
        })

    logger.info("Exported %d memories for agent=%s", len(result), agent or "main")
    return result


class ImportRequest(BaseModel):
    """Import request: list of memory objects."""
    memories: List[Dict[str, Any]] = Field(..., min_length=1)
    agent: Optional[str] = None
    skip_duplicates: bool = True


@app.post("/v1/import")
async def import_memories(req: ImportRequest) -> Dict[str, Any]:
    """Import memories from JSONL/JSON array.

    Each memory object must have at least ``text``.
    Optional fields: category, importance, strength, tags, created_at.
    If ``skip_duplicates`` is true (default), memories with existing IDs are skipped.
    """
    if _storage_pool is None:
        raise HTTPException(503, "Storage pool not initialised")

    agent_key = StoragePool.normalize_key(req.agent) if req.agent else None
    storage = _get_storage(agent_key)

    imported = 0
    skipped = 0

    for mem in req.memories:
        text = mem.get("text", "").strip()
        if not text:
            skipped += 1
            continue

        mid = mem.get("id")
        if mid and req.skip_duplicates:
            existing = storage._get_conn().execute(
                "SELECT 1 FROM memories WHERE id = ?", (mid,)
            ).fetchone()
            if existing:
                skipped += 1
                continue

        # Embed if embedder available
        vector = None
        if _embedder is not None:
            for attempt in range(3):
                try:
                    vector = await _embedder.embed(text)
                    break
                except Exception as exc:
                    if attempt < 2:
                        logger.warning(
                            "Import embed retry %d/3 for memory %s: %s",
                            attempt + 1,
                            mid or "new",
                            exc,
                        )
                        await asyncio.sleep(0.5 * (attempt + 1))
                    else:
                        logger.error(
                            "Import embed failed after 3 attempts for memory %s: %s",
                            mid or "new",
                            exc,
                        )

        storage.store_memory(
            text=text,
            vector=vector,
            category=mem.get("category", "other"),
            importance=float(mem.get("importance", 0.5)),
            source_session=mem.get("source_session"),
            memory_id=mid,
        )
        imported += 1

    logger.info(
        "Imported %d memories (skipped %d) for agent=%s",
        imported, skipped, req.agent or "main",
    )
    return {"imported": imported, "skipped": skipped, "total": len(req.memories)}


# ---------------------------------------------------------------------------
# Amnesia Detection
# ---------------------------------------------------------------------------

class AmnesiaCheckRequest(BaseModel):
    topics: List[str] = Field(..., min_length=1, max_length=20,
                               description="Topics to check coverage for")
    agent: Optional[str] = None
    min_match_score: float = Field(default=0.01, ge=0.0, le=1.0)


@app.post("/v1/amnesia-check")
async def amnesia_check(req: AmnesiaCheckRequest) -> Dict[str, Any]:
    """Check how well the agent remembers a list of topics.

    Returns a coverage score (0-1) and per-topic match details.
    Useful for post-compaction validation and memory health monitoring.
    """
    search = _get_search(req.agent)
    agent_key = StoragePool.normalize_key(req.agent)

    topic_results = []
    hits = 0

    for topic in req.topics:
        results = await search.search(
            query=topic, limit=3, min_score=req.min_match_score, agent=agent_key,
        )
        matched = len(results) > 0
        if matched:
            hits += 1
        topic_results.append({
            "topic": topic,
            "matched": matched,
            "match_count": len(results),
            "top_score": round(results[0].score, 4) if results else 0.0,
            "top_text": results[0].text[:200] if results else None,
        })

    coverage = hits / len(req.topics) if req.topics else 0.0

    return {
        "agent": agent_key,
        "coverage": round(coverage, 2),
        "total_topics": len(req.topics),
        "matched_topics": hits,
        "topics": topic_results,
        "status": "healthy" if coverage >= 0.7 else "warning" if coverage >= 0.4 else "amnesia",
    }


# ---------------------------------------------------------------------------
# Admin: Key Rotation
# ---------------------------------------------------------------------------

@app.post("/v1/admin/rotate-key")
async def rotate_key(expire_old_hours: int = 24) -> Dict[str, Any]:
    """Generate a new API key and optionally expire the current extra keys.

    - Generates a new 32-byte URL-safe key
    - Appends to memory-api-keys.json
    - Optionally sets expires_at on existing extra keys
    - Returns the new key (only time it's visible!)
    """
    import json as _json

    new_key = secrets.token_urlsafe(32)
    keys_path = _extra_keys_path

    existing_keys: list = []
    try:
        with open(keys_path) as f:
            data = _json.load(f)
            existing_keys = data.get("keys", [])
    except (FileNotFoundError, _json.JSONDecodeError):
        pass

    # Expire old extra keys if requested
    if expire_old_hours > 0:
        expire_at = time.time() + (expire_old_hours * 3600)
        for entry in existing_keys:
            if entry.get("expires_at") is None:
                entry["expires_at"] = expire_at

    # Add new key
    existing_keys.append({
        "key": new_key,
        "label": f"rotated-{int(time.time())}",
        "created_at": time.time(),
        "expires_at": None,
    })

    os.makedirs(os.path.dirname(keys_path), exist_ok=True)
    with open(keys_path, "w") as f:
        _json.dump({"keys": existing_keys}, f, indent=2)

    logger.info("API key rotated. Total keys: %d", len(existing_keys))
    return {
        "new_key": new_key,
        "total_keys": len(existing_keys),
        "old_keys_expire_in_hours": expire_old_hours if expire_old_hours > 0 else None,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the API server via uvicorn."""
    import uvicorn

    cfg = load_config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    logger.info("Starting OpenClaw Memory API on %s:%d", cfg.api_host, cfg.api_port)
    uvicorn.run(
        "agent_memory.api:app",
        host=cfg.api_host,
        port=cfg.api_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
