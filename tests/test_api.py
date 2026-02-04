"""Tests for the FastAPI HTTP API.

Uses httpx AsyncClient against the FastAPI app. We manually initialise
the module-level state that normally comes from the lifespan handler,
pointing it at a temp database so the production DB is untouched.
"""

from __future__ import annotations

import time

import pytest
from httpx import AsyncClient, ASGITransport

import asuman_memory.api as api_module
from asuman_memory.api import app
from asuman_memory.storage import MemoryStorage
from asuman_memory.search import HybridSearch
from asuman_memory.entities import KnowledgeGraph


class _StubEmbedder:
    """Minimal stub that returns a zero-vector — enough for capture/store."""

    async def embed(self, text: str):
        return [0.0, 0.0, 0.0, 0.0]

    async def embed_batch(self, texts):
        return [[0.0, 0.0, 0.0, 0.0]] * len(texts)


@pytest.fixture(autouse=True)
def _init_api_state(tmp_path):
    """Wire the api module globals to a temp DB so every test starts clean."""
    db_path = str(tmp_path / "api_test.sqlite")
    storage = MemoryStorage(db_path=db_path, dimensions=4)
    stub_embedder = _StubEmbedder()

    api_module._storage = storage
    api_module._embedder = stub_embedder
    api_module._search = HybridSearch(storage=storage, embedder=stub_embedder)
    api_module._kg = KnowledgeGraph(storage=storage)
    api_module._start_time = time.time()

    yield

    storage.close()
    api_module._storage = None
    api_module._embedder = None
    api_module._search = None
    api_module._kg = None


@pytest.fixture
async def client():
    """Create a test HTTP client against the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# /v1/health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestHealth:
    async def test_health_returns_ok(self, client):
        resp = await client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data
        assert data["storage"] is True

    async def test_health_has_memory_count(self, client):
        resp = await client.get("/v1/health")
        data = resp.json()
        assert "total_memories" in data


# ---------------------------------------------------------------------------
# /v1/stats
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestStats:
    async def test_stats_returns_counts(self, client):
        resp = await client.get("/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_memories" in data
        assert "entities" in data
        assert "by_category" in data
        assert "relationships" in data

    async def test_stats_after_store(self, client):
        await client.post("/v1/store", json={"text": "test memory for stats"})
        resp = await client.get("/v1/stats")
        assert resp.json()["total_memories"] >= 1


# ---------------------------------------------------------------------------
# /v1/store
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestStore:
    async def test_store_memory(self, client):
        resp = await client.post("/v1/store", json={
            "text": "Test memory from pytest",
            "category": "test",
            "importance": 0.8,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["stored"] is True
        assert "id" in data

    async def test_store_minimal(self, client):
        resp = await client.post("/v1/store", json={
            "text": "Minimal store test",
        })
        assert resp.status_code == 200
        assert resp.json()["stored"] is True

    async def test_store_validation_error(self, client):
        resp = await client.post("/v1/store", json={})
        assert resp.status_code == 422  # missing required 'text'


# ---------------------------------------------------------------------------
# /v1/recall
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRecall:
    async def test_recall_basic(self, client):
        await client.post("/v1/store", json={"text": "User yarın toplantı var"})

        resp = await client.post("/v1/recall", json={
            "query": "toplantı",
            "limit": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "query" in data
        assert "count" in data
        assert "triggered" in data
        assert "results" in data
        assert isinstance(data["results"], list)

    async def test_recall_with_trigger(self, client):
        resp = await client.post("/v1/recall", json={
            "query": "hatırlıyor musun dün ne konuştuk",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["triggered"] is True

    async def test_recall_empty_results(self, client):
        resp = await client.post("/v1/recall", json={
            "query": "xyznonexistent12345",
            "limit": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

    async def test_recall_validation(self, client):
        resp = await client.post("/v1/recall", json={})
        assert resp.status_code == 422

    async def test_recall_result_fields(self, client):
        await client.post("/v1/store", json={"text": "Python ile FastAPI projesi"})
        resp = await client.post("/v1/recall", json={"query": "Python", "limit": 1})
        if resp.json()["count"] > 0:
            result = resp.json()["results"][0]
            assert "id" in result
            assert "text" in result
            assert "score" in result
            assert "confidence_tier" in result


# ---------------------------------------------------------------------------
# /v1/capture
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCapture:
    async def test_capture_messages(self, client):
        resp = await client.post("/v1/capture", json={
            "messages": [
                {"text": "Bu bir test mesajıdır capture için", "role": "user"},
                {"text": "Anladım, test mesajı kaydedildi durumda", "role": "assistant"},
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["stored"] >= 1
        assert data["total"] == 2

    async def test_capture_filters_short(self, client):
        resp = await client.post("/v1/capture", json={
            "messages": [
                {"text": "ok", "role": "user"},
                {"text": "", "role": "user"},
            ]
        })
        assert resp.status_code == 200
        assert resp.json()["stored"] == 0

    async def test_capture_empty(self, client):
        resp = await client.post("/v1/capture", json={"messages": []})
        assert resp.status_code == 200
        assert resp.json()["stored"] == 0


# ---------------------------------------------------------------------------
# /v1/forget
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestForget:
    async def test_forget_by_id(self, client):
        store_resp = await client.post("/v1/store", json={"text": "to be forgotten"})
        mid = store_resp.json()["id"]

        resp = await client.request("DELETE", "/v1/forget", json={"id": mid})
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    async def test_forget_nonexistent(self, client):
        resp = await client.request("DELETE", "/v1/forget", json={"id": "nonexistent123"})
        assert resp.status_code == 200
        assert resp.json()["deleted"] is False

    async def test_forget_by_query(self, client):
        await client.post("/v1/store", json={"text": "unique forgettable text xyz987"})

        resp = await client.request("DELETE", "/v1/forget", json={
            "query": "unique forgettable text xyz987"
        })
        assert resp.status_code == 200

    async def test_forget_no_params(self, client):
        resp = await client.request("DELETE", "/v1/forget", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /v1/search (GET)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSearchInteractive:
    async def test_search_get(self, client):
        await client.post("/v1/store", json={"text": "searchable memory about Python programming"})

        resp = await client.get("/v1/search", params={"query": "Python", "limit": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert "query" in data
        assert "count" in data
        assert "results" in data

    async def test_search_missing_query(self, client):
        resp = await client.get("/v1/search")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# OpenAPI / docs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestDocs:
    async def test_openapi_json(self, client):
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "paths" in data
        assert "/v1/health" in data["paths"]
        assert "/v1/recall" in data["paths"]
        assert "/v1/store" in data["paths"]
        assert "/v1/forget" in data["paths"]
        assert "/v1/search" in data["paths"]
        assert "/v1/stats" in data["paths"]
        assert "/v1/capture" in data["paths"]

    async def test_docs_page(self, client):
        resp = await client.get("/docs")
        assert resp.status_code == 200
        assert "swagger" in resp.text.lower() or "openapi" in resp.text.lower()
