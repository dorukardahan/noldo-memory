"""FastAPI HTTP API for Asuman Memory System.

Endpoints:
    POST   /v1/recall   — Search memories
    POST   /v1/capture  — Ingest messages
    POST   /v1/store    — Store a memory
    DELETE /v1/forget    — Delete memory
    GET    /v1/search    — Interactive search
    GET    /v1/stats     — Statistics
    GET    /v1/health    — Health check

Run: ``python -m asuman_memory.api``
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

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
    title="Agent Memory API",
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
    """
    if _storage is None or _embedder is None:
        raise HTTPException(503, "Storage/embedder not initialised")

    stored = 0
    for msg in req.messages:
        text = msg.get("text", "")
        role = msg.get("role", "user")
        if not text or len(text.strip()) < 3:
            continue

        importance = score_importance(text, {"role": role})

        try:
            vector = await _embedder.embed(text)
        except Exception:
            vector = None

        _storage.store_memory(
            text=text[:2000],
            vector=vector,
            category=role,
            importance=importance,
            source_session=msg.get("session", ""),
        )

        # Knowledge graph
        if _kg:
            try:
                _kg.process_text(text, timestamp=msg.get("timestamp", ""))
            except Exception:
                pass

        stored += 1

    return {"stored": stored, "total": len(req.messages)}


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

    mid = _storage.store_memory(
        text=req.text,
        vector=vector,
        category=req.category,
        importance=req.importance,
    )

    return {"id": mid, "stored": True}


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

    logger.info("Starting Agent Memory API on %s:%d", cfg.api_host, cfg.api_port)
    uvicorn.run(
        "asuman_memory.api:app",
        host=cfg.api_host,
        port=cfg.api_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
