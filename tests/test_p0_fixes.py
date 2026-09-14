"""Regression tests for recent P0-P2 fixes.

Covers:
1) Agent ID validation in StoragePool.normalize_key via /v1/store
2) Import idempotency and memory_id preservation via /v1/import
3) Search cache separation by min_score in /v1/recall
4) Script smoke tests for manage.sh and openclaw_sync.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

import agent_memory.api as api_module
from agent_memory.api import app
from agent_memory.config import Config
from agent_memory.pool import StoragePool
from agent_memory.search import SearchWeights, _build_cache_query_norm


class _StubEmbedder:
    """Minimal stub embedder used by API tests."""

    async def embed(self, text: str):
        return [0.0, 0.0, 0.0, 0.0]

    async def embed_batch(self, texts):
        return [[0.0, 0.0, 0.0, 0.0]] * len(texts)

    def set_storage(self, storage):
        pass


@pytest.fixture(autouse=True)
def _init_api_state(tmp_path):
    """Wire API globals to an isolated temp database per test."""
    pool = StoragePool(base_dir=str(tmp_path), dimensions=4)
    pool.get("main")

    api_module._storage_pool = pool
    api_module._embedder = _StubEmbedder()
    api_module._search_cache = {}
    api_module._kg_cache = {}
    api_module._search_weights = SearchWeights(
        semantic=0.55,
        keyword=0.25,
        recency=0.10,
        strength=0.10,
    )
    api_module._config = Config(api_key="", openrouter_api_key="test-key")
    api_module._start_time = time.time()

    yield

    pool.close_all()
    api_module._storage_pool = None
    api_module._embedder = None
    api_module._search_cache = {}
    api_module._kg_cache = {}
    api_module._search_weights = None
    api_module._config = None


@pytest.fixture
async def client():
    """Async test client bound to FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _store_with_agent(client: AsyncClient, agent: str, text: str):
    """Store helper that sends agent in both query and body.

    /v1/store currently reads agent from body model, but we include query param
    too for compatibility with expected calling style.
    """
    return await client.post(
        "/v1/store",
        params={"agent": agent},
        json={"text": text, "agent": agent},
    )


@pytest.mark.asyncio
class TestAgentValidation:
    async def test_valid_agent_ids(self, client):
        for agent in ("main", "devops", "my-agent-1"):
            resp = await _store_with_agent(client, agent=agent, text=f"memory for {agent}")
            assert resp.status_code == 200
            assert resp.json()["agent"] == agent

    async def test_invalid_agent_path_traversal(self, client):
        resp = await _store_with_agent(client, agent="../etc", text="should fail")
        assert resp.status_code == 400
        assert "invalid agent id" in resp.json()["error"].lower()

    async def test_invalid_agent_slash(self, client):
        resp = await _store_with_agent(client, agent="foo/bar", text="should fail")
        assert resp.status_code == 400
        assert "invalid agent id" in resp.json()["error"].lower()

    async def test_invalid_agent_empty(self, client):
        resp = await _store_with_agent(client, agent="", text="empty agent should map to main")
        assert resp.status_code == 200
        assert resp.json()["agent"] == "main"

    async def test_reserved_agent_all(self, client):
        resp = await _store_with_agent(client, agent="all", text="reserved")
        assert resp.status_code == 400
        assert "cannot store to 'all'" in resp.json()["error"].lower()


@pytest.mark.asyncio
class TestImportIdempotency:
    async def test_import_same_id_twice(self, client):
        payload = {
            "memories": [
                {
                    "id": "test-123",
                    "text": "Imported memory should be idempotent",
                    "category": "test",
                }
            ]
        }

        first = await client.post("/v1/import", json=payload)
        assert first.status_code == 200
        assert first.json()["imported"] == 1
        assert first.json()["skipped"] == 0

        second = await client.post("/v1/import", json=payload)
        assert second.status_code == 200
        assert second.json()["imported"] == 0
        assert second.json()["skipped"] == 1

        stats = await client.get("/v1/stats")
        assert stats.status_code == 200
        assert stats.json()["total_memories"] == 1

        exported = await client.get("/v1/export")
        assert exported.status_code == 200
        ids = [m["id"] for m in exported.json()]
        assert ids.count("test-123") == 1

    async def test_import_preserves_id(self, client):
        memory_id = "test-keep-id"
        resp = await client.post(
            "/v1/import",
            json={
                "memories": [
                    {
                        "id": memory_id,
                        "text": "Memory imported with explicit id",
                        "category": "test",
                    }
                ]
            },
        )
        assert resp.status_code == 200
        assert resp.json()["imported"] == 1

        exported = await client.get("/v1/export")
        assert exported.status_code == 200
        assert any(m["id"] == memory_id for m in exported.json())


@pytest.mark.asyncio
class TestSearchCacheMinScore:
    async def test_different_min_scores_cached_separately(self, client):
        store_resp = await client.post(
            "/v1/store",
            json={"text": "python cache threshold memory", "category": "test", "importance": 0.8},
        )
        assert store_resp.status_code == 200

        low = await client.post(
            "/v1/recall",
            json={"query": "python", "limit": 10, "min_score": 0.0},
        )
        assert low.status_code == 200
        low_count = low.json()["count"]
        assert low_count >= 1

        high = await client.post(
            "/v1/recall",
            json={"query": "python", "limit": 10, "min_score": 0.5},
        )
        assert high.status_code == 200
        high_count = high.json()["count"]

        assert high_count < low_count
        assert high_count == 0

        storage = api_module._storage_pool.get("main")
        rows = storage._get_conn().execute(
            """
            SELECT min_score
              FROM search_result_cache
             WHERE query_norm = ? AND limit_val = ? AND agent = ?
             ORDER BY min_score
            """,
            (_build_cache_query_norm("python", None, None), 10, "main"),
        ).fetchall()

        cached_scores = {float(r["min_score"]) for r in rows}
        assert 0.0 in cached_scores
        assert 0.5 in cached_scores


class TestScriptSmoke:
    def test_manage_sh_usage(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            ["bash", "scripts/manage.sh"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert proc.returncode in (0, 1)
        combined = f"{proc.stdout}\n{proc.stderr}"
        assert "usage:" in combined.lower()

    def test_sync_help(self):
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "scripts/openclaw_sync.py", "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )

        assert proc.returncode == 0
        combined = f"{proc.stdout}\n{proc.stderr}"
        assert "usage:" in combined.lower()
