"""Endpoint budgets, shard caps and uncached scoped diagnostics."""
import asyncio
import time
from pathlib import Path

import pytest

from tests import test_api
from agent_memory import api

client = test_api.client
_init_api_state = test_api._init_api_state


@pytest.mark.asyncio
async def test_recall_caller_budget_bounds_stalled_provider(client, monkeypatch):
    async def stalled(text):
        await asyncio.sleep(20)
        return [0., 0., 0., 0.]
    monkeypatch.setattr(api._embedder, 'embed', stalled)
    started = time.monotonic()
    response = await client.post('/v1/recall', json={'query': 'Synthetic Python fact', 'timeout_ms': 100})
    assert time.monotonic() - started < .4
    assert response.status_code in {200, 504}


@pytest.mark.asyncio
async def test_all_shards_rejects_oversized_inventory_without_search(client, monkeypatch):
    monkeypatch.setattr(api._storage_pool, 'get_all_agents', lambda: ['main'] + [f'agent{i}' for i in range(32)])
    response = await client.post('/v1/recall', json={'query': 'Synthetic Python fact', 'agent': 'all'})
    assert response.status_code == 503
    assert response.json()['error'] == 'too_many_agent_shards'


@pytest.mark.asyncio
async def test_fresh_probe_bypasses_caches_no_mutation(client, monkeypatch):
    from agent_memory.embeddings import OpenRouterEmbeddings
    import httpx
    calls = []
    async def respond(request):
        calls.append(request)
        return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1., 0., 0., 0.]}]})
    storage = api._get_storage()
    embedder = OpenRouterEmbeddings(api_key='synthetic', dimensions=4)
    embedder._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    embedder.set_storage(storage)
    monkeypatch.setattr(api, '_embedder', embedder)
    before = dict(storage.stats())
    conn = storage._get_conn()
    cache_before = conn.execute('SELECT count(*) FROM embedding_cache').fetchone()[0]
    try:
        for _ in range(2):
            response = await client.get('/v1/health/backend')
            assert response.status_code == 200
            assert response.json()['status'] == 'ok'
        assert len(calls) == 2
        assert dict(storage.stats()) == before
        assert conn.execute('SELECT count(*) FROM embedding_cache').fetchone()[0] == cache_before
        assert embedder._cache == {}
        absent = api._storage_pool._db_path('absent')
        assert not Path(absent).exists()
        assert (await client.get('/v1/health/backend', params={'agent': 'absent'})).status_code == 404
        assert not Path(absent).exists()
    finally:
        await embedder.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('value', [True, False, 0, -1, '100', 99, 10001, [], {}])
async def test_timeout_invalid_no_mutation(client, value):
    response = await client.post('/v1/store', json={'text': 'Synthetic durable fact', 'timeout_ms': value})
    assert response.status_code == 422
    assert api._get_storage().stats()['total_memories'] == 0
