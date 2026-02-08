"""FastAPI HTTP API for the OpenClaw memory system.

Endpoints:
    POST   /v1/recall        — Hybrid search (semantic + BM25 + recency + strength)
    POST   /v1/capture       — Batch ingest (write-time semantic merge)
    POST   /v1/store         — Store one memory (write-time semantic merge)
    DELETE /v1/forget        — Delete memory
    GET    /v1/search        — Interactive search
    GET    /v1/stats         — Statistics
    GET    /v1/health        — Health check
    POST   /v1/decay         — Run Ebbinghaus strength decay
    POST   /v1/consolidate   — Deduplicate + archive stale memories

Run: ``python -m asuman_memory.api``
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .config import Config, load_config
from .embeddings import OpenRouterEmbeddings
from .entities import KnowledgeGraph
from .search import HybridSearch, SearchResult
from .storage import MemoryStorage
from .triggers import get_confidence_tier, score_importance, should_trigger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state (initialised in lifespan)
# ---------------------------------------------------------------------------

_storage: Optional[MemoryStorage] = None
_embedder: Optional[OpenRouterEmbeddings] = None
_search: Optional[HybridSearch] = None
_kg: Optional[KnowledgeGraph] = None
_config: Optional[Config] = None
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic."""
    global _storage, _embedder, _search, _kg, _config, _start_time

    _config = load_config()
    errors = _config.validate()
    if errors:
        logger.warning("Config validation warnings: %s", errors)

    _storage = MemoryStorage(
        db_path=_config.db_path,
        dimensions=_config.embedding_dimensions,
    )

    if _config.openrouter_api_key:
        _embedder = OpenRouterEmbeddings(
            api_key=_config.openrouter_api_key,
            model=_config.embedding_model,
            dimensions=_config.embedding_dimensions,
        )
    else:
        logger.warning("No OPENROUTER_API_KEY — semantic search disabled")
        _embedder = None

    _search = HybridSearch(storage=_storage, embedder=_embedder)
    _kg = KnowledgeGraph(storage=_storage)
    _start_time = time.time()

    logger.info(
        "Memory API ready — db=%s model=%s",
        _config.db_path, _config.embedding_model,
    )

    yield

    # Shutdown
    if _storage:
        _storage.close()


app = FastAPI(
    title="OpenClaw Memory API",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RecallRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class CaptureRequest(BaseModel):
    messages: List[Dict[str, Any]]


class StoreRequest(BaseModel):
    text: str
    category: str = "other"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class ForgetRequest(BaseModel):
    id: Optional[str] = None
    query: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/recall")
async def recall(req: RecallRequest) -> Dict[str, Any]:
    """Search memories using hybrid search (semantic + BM25 + recency)."""
    if _search is None:
        raise HTTPException(503, "Search engine not initialised")

    results = await _search.search(
        query=req.query,
        limit=req.limit,
        min_score=req.min_score,
    )

    return {
        "query": req.query,
        "count": len(results),
        "triggered": should_trigger(req.query),
        "results": [r.to_dict() for r in results],
    }


@app.post("/v1/capture")
async def capture(req: CaptureRequest) -> Dict[str, Any]:
    """Ingest a batch of messages into memory.

    Each message dict should have at least ``text`` and ``role``.

    Notes:
      - Works even when the embedder is not configured (vectors stored as NULL).
      - Uses a single embed_batch + store_memories_batch for efficiency.
    """
    if _storage is None:
        raise HTTPException(503, "Storage not initialised")

    # Pre-filter / normalize
    cleaned: List[Dict[str, Any]] = []
    for msg in req.messages:
        text = (msg.get("text") or "").strip()
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
        res = _storage.merge_or_store(
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
    if _kg:
        for m in cleaned:
            try:
                _kg.process_text(m["text"], timestamp=m.get("timestamp", ""))
            except Exception:
                pass

    return {"stored": stored_n, "merged": merged_n, "total": len(req.messages)}


@app.post("/v1/store")
async def store(req: StoreRequest) -> Dict[str, Any]:
    """Manually store a single memory."""
    if _storage is None:
        raise HTTPException(503, "Storage not initialised")

    vector = None
    if _embedder:
        try:
            vector = await _embedder.embed(req.text)
        except Exception:
            pass

    res = _storage.merge_or_store(
        text=req.text,
        vector=vector,
        category=req.category,
        importance=req.importance,
        source_session=None,
    )

    return {"id": res["id"], "stored": res["action"] == "inserted", "merged": res["action"] == "merged", "similarity": res.get("similarity")}


@app.delete("/v1/forget")
async def forget(req: ForgetRequest) -> Dict[str, Any]:
    """Delete a memory by ID or by searching for it."""
    if _storage is None:
        raise HTTPException(503, "Storage not initialised")

    if req.id:
        deleted = _storage.delete_memory(req.id)
        return {"deleted": deleted, "id": req.id}

    if req.query:
        # Search and delete first match
        results = _storage.search_text(req.query, limit=1)
        if results:
            mid = results[0]["id"]
            deleted = _storage.delete_memory(mid)
            return {"deleted": deleted, "id": mid}
        return {"deleted": False, "reason": "no match found"}

    raise HTTPException(400, "Provide 'id' or 'query'")


@app.get("/v1/search")
async def search_interactive(
    query: str = Query(..., min_length=1),
    limit: int = Query(default=5, ge=1, le=50),
) -> Dict[str, Any]:
    """Interactive search (for CLI/debug)."""
    if _search is None:
        raise HTTPException(503, "Search not initialised")

    results = await _search.search(query=query, limit=limit)
    return {
        "query": query,
        "count": len(results),
        "results": [r.to_dict() for r in results],
    }


@app.post("/v1/decay")
async def decay() -> Dict[str, Any]:
    """Run weekly Ebbinghaus strength decay (cron-friendly)."""
    if _storage is None:
        raise HTTPException(503, "Storage not initialised")

    decayed = _storage.decay_all(days_threshold=7, decay_amount=0.05, min_strength=0.3)
    total = _storage.stats().get("total_memories", 0)
    return {"decayed": decayed, "total": total}


@app.post("/v1/consolidate")
async def consolidate() -> Dict[str, Any]:
    """Deduplicate and cleanup memories.

    1) Find memory pairs with cosine similarity > 0.90
    2) Merge duplicates (keep higher importance; combine text)
    3) Soft-delete memories with strength < 0.3 and not accessed in 30 days
    """
    if _storage is None:
        raise HTTPException(503, "Storage not initialised")

    conn = _storage._get_conn()  # internal connection; safe within single-process service
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

        # Merge duplicates cluster-by-cluster
        for root, members in clusters.items():
            if len(members) < 2:
                continue

            # Pick keeper: highest importance
            keeper = max(members, key=lambda rr: float(rr["importance"] or 0.0))
            keeper_id = keeper["id"]

            # Aggregate fields
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

            merged_text = "\n• ".join([t for t in texts if t])
            new_importance = max(imps) if imps else float(keeper["importance"] or 0.5)
            new_strength = min(max(strengths) + 0.2, 5.0) if strengths else 1.0
            new_last_acc = max(last_accs) if last_accs else now

            # Update keeper row
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

            # Update keeper vector to mean of cluster
            if vecs:
                try:
                    avg = np.mean(np.vstack(vecs), axis=0).astype(np.float32)
                    conn.execute(
                        "UPDATE memory_vectors SET embedding = ? WHERE rowid = ?",
                        (avg.tobytes(), keeper["vector_rowid"]),
                    )
                except Exception:
                    pass

            # Soft-delete the rest
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
        "Consolidation complete: merged=%d archived=%d total_before=%d total_after=%d pairs=%d",
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
async def stats() -> Dict[str, Any]:
    """Return memory statistics."""
    if _storage is None:
        raise HTTPException(503, "Storage not initialised")
    return _storage.stats()


@app.get("/v1/health")
async def health() -> Dict[str, Any]:
    """Health check."""
    status: Dict[str, Any] = {
        "status": "ok",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "storage": _storage is not None,
        "embedder": _embedder is not None,
    }

    if _storage:
        try:
            s = _storage.stats()
            status["total_memories"] = s["total_memories"]
            status["entities"] = s["entities"]
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
        "asuman_memory.api:app",
        host=cfg.api_host,
        port=cfg.api_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
