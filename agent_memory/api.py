"""FastAPI HTTP API for the OpenClaw memory system.

Endpoints:
    POST   /v1/recall        -- Hybrid search (semantic + BM25 + recency + strength)
    POST   /v1/capture       -- Batch ingest (write-time semantic merge)
    POST   /v1/store         -- Store one memory (write-time semantic merge)
    DELETE /v1/forget        -- Delete memory
    GET    /v1/search        -- Interactive search
    GET    /v1/stats         -- Statistics
    GET    /v1/health        -- Health check
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
import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .config import Config, load_config
from .embeddings import OpenRouterEmbeddings
from .entities import KnowledgeGraph
from .pool import StoragePool
from .search import HybridSearch, SearchResult, SearchWeights
from .storage import MemoryStorage
from .rules import RuleDetector
from .triggers import get_confidence_tier, score_importance, should_trigger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state (initialised in lifespan)
# ---------------------------------------------------------------------------

_storage_pool: Optional[StoragePool] = None
_embedder: Optional[OpenRouterEmbeddings] = None
_config: Optional[Config] = None
_start_time: float = 0.0

# Per-agent caches (lazily populated)
_search_cache: Dict[str, HybridSearch] = {}
_kg_cache: Dict[str, KnowledgeGraph] = {}
_search_weights: Optional[SearchWeights] = None
_rule_detector = RuleDetector()


# ---------------------------------------------------------------------------
# Agent routing helpers
# ---------------------------------------------------------------------------

def _get_storage(agent: Optional[str] = None) -> MemoryStorage:
    """Get the MemoryStorage for the given agent."""
    if _storage_pool is None:
        raise HTTPException(503, "Storage pool not initialised")
    return _storage_pool.get(agent)


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
    global _storage_pool, _embedder, _config, _start_time, _search_weights

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
    )

    agents = _storage_pool.get_all_agents()
    logger.info(
        "Memory API ready -- base_dir=%s agents=%s model=%s",
        base_dir, agents, _config.embedding_model,
    )

    # Start warmup background task
    asyncio.create_task(warmup_loop())

    yield

    # Shutdown
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
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RecallRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    agent: Optional[str] = None


class CaptureRequest(BaseModel):
    messages: List[Dict[str, Any]]
    agent: Optional[str] = None


class StoreRequest(BaseModel):
    text: str
    category: str = "other"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    agent: Optional[str] = None


class ForgetRequest(BaseModel):
    id: Optional[str] = None
    query: Optional[str] = None
    agent: Optional[str] = None


class DecayRequest(BaseModel):
    agent: Optional[str] = None


class ConsolidateRequest(BaseModel):
    agent: Optional[str] = None


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
    results = await search.search(
        query=req.query,
        limit=req.limit,
        min_score=req.min_score,
    )

    agent_key = StoragePool.normalize_key(req.agent)
    return {
        "query": req.query,
        "agent": agent_key,
        "count": len(results),
        "triggered": should_trigger(req.query),
        "results": [r.to_dict() for r in results],
    }


async def _recall_all(req: RecallRequest) -> Dict[str, Any]:
    """Cross-agent recall: query all agent DBs, merge by score."""
    if _storage_pool is None:
        raise HTTPException(503, "Storage pool not initialised")

    all_results: List[Dict[str, Any]] = []
    for agent_id in _storage_pool.get_all_agents():
        try:
            search = _get_search(agent_id)
            results = await search.search(
                query=req.query,
                limit=req.limit,
                min_score=req.min_score,
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

    return {
        "query": req.query,
        "agent": "all",
        "count": len(all_results),
        "triggered": should_trigger(req.query),
        "results": all_results,
        "cross_agent": True,
    }


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
            "text": text[:2000],
            "role": role,
            "session": msg.get("session", ""),
            "timestamp": msg.get("timestamp", ""),
        })

    if not cleaned:
        return {"stored": 0, "merged": 0, "total": len(req.messages)}

    texts = [m["text"] for m in cleaned]

    vectors: List[Optional[List[float]]] = [None] * len(texts)
    if _embedder is not None:
        try:
            vectors = await _embedder.embed_batch(texts)
        except Exception:
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
    for it in items:
        res = storage.merge_or_store(
            text=it["text"],
            vector=it.get("vector"),
            category=it.get("category", "other"),
            importance=float(it.get("importance", 0.5)),
            source_session=it.get("source_session"),
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
        try:
            vector = await _embedder.embed(req.text)
        except Exception:
            pass

    # Rule detection: override category/importance if instruction detected
    category = req.category
    importance = req.importance
    detected = _rule_detector.detect(req.text)
    if detected or _rule_detector.check_safeword(req.text):
        category = "rule"
        importance = 1.0
        logger.info("Rule detected in /v1/store: %s", req.text[:60])

    res = storage.merge_or_store(
        text=req.text,
        vector=vector,
        category=category,
        importance=importance,
        source_session=None,
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
        try:
            vector = await _embedder.embed(req.text)
        except Exception:
            pass

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


@app.post("/v1/consolidate")
async def consolidate(req: ConsolidateRequest = ConsolidateRequest()) -> Dict[str, Any]:
    """Deduplicate and cleanup memories.

    1) Find memory pairs with cosine similarity > 0.90
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
            if sim <= 0.90:
                continue
            other = conn.execute(
                "SELECT id FROM memories WHERE vector_rowid = ? AND deleted_at IS NULL",
                (n["vec_rowid"],),
            ).fetchone()
            if not other:
                continue
            oid = other["id"]
            if oid == mid:
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
    """Health check."""
    status: Dict[str, Any] = {
        "status": "ok",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "storage": _storage_pool is not None,
        "embedder": _embedder is not None,
    }

    if _storage_pool:
        try:
            agents = _storage_pool.get_all_agents()
            status["agents"] = agents
            main_stats = _storage_pool.get("main").stats()
            status["total_memories"] = main_stats["total_memories"]
            status["entities"] = main_stats["entities"]
        except Exception:
            pass

    return status


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
