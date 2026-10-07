"""Integration metrics must observe executed work, not just render a schema."""
import asyncio
import time

import pytest
import httpx
from tests import test_api
from agent_memory import api, embeddings
from agent_memory.metrics import collector

client = test_api.client
_init_api_state = test_api._init_api_state


@pytest.mark.asyncio
async def test_actual_admission_index_provider_recall_and_status_emit_stage_metrics(client, monkeypatch):
    collector.reset()
    monkeypatch.setattr(embeddings, 'load_config', lambda: api._config)
    provider = embeddings.OpenRouterEmbeddings(api_key='synthetic', dimensions=4)
    async def reply(request):
        import json
        texts = json.loads(request.content)['input']
        return httpx.Response(200, json={'data': [dict(index=i, embedding=[1., 0., 0., 0.]) for i in range(len(texts))]})
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
    api._embedder = provider
    try:
        capture = await client.post('/v1/capture', json={'messages': [
            {'text': 'Synthetic Python SQLite uses a violet observatory.', 'role': 'user', 'session': 'synthetic'}]})
        assert capture.status_code == 200
        await test_api.drain_indexing()
        recall = await client.post('/v1/recall', json={'query': 'Python SQLite observatory', 'min_semantic_score': .5})
        assert recall.status_code == 200 and recall.json()['count'] > 0
        status = await client.get('/v1/operations/' + capture.json()['request_id'])
        assert status.status_code == 200 and status.json()['state'] == 'completed'
        snapshot = collector.snapshot()
        actual = {(operation, stage) for operation, stage, _ in snapshot['stage_count']}
        assert {('capture', 'normalize'), ('capture', 'dedup'), ('capture', 'persist'),
                ('capture', 'queue'), ('capture', 'total'), ('index', 'embedding'),
                ('index', 'graph'), ('recall', 'embedding'), ('recall', 'bm25'),
                ('recall', 'vector'), ('recall', 'access'), ('status', 'total')} <= actual
        assert snapshot['stage_attempts']['embedding'] >= 2
        assert snapshot['job_outcomes'] == {('embed', 'completed'): 1, ('graph', 'completed'): 1}
        assert snapshot['index_queue_depth'] == snapshot['index_inflight'] == 0
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_provider_retry_and_completed_job_emit_runtime_metrics(client, monkeypatch):
    collector.reset()
    monkeypatch.setattr(embeddings, 'load_config', lambda: api._config)
    provider = embeddings.OpenRouterEmbeddings(api_key='synthetic', dimensions=4)
    calls = 0

    async def reply(request):
        nonlocal calls
        import json
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={'error': 'synthetic'})
        texts = json.loads(request.content)['input']
        return httpx.Response(200, json={'data': [
            dict(index=i, embedding=[1., 0., 0., 0.]) for i in range(len(texts))
        ]})

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
    api._embedder = provider
    try:
        stored = await client.post('/v1/store', json={'text': 'Synthetic runtime retry metric.'})
        assert stored.status_code == 200
        await test_api.drain_indexing()
        snapshot = collector.snapshot()
        assert snapshot['stage_attempts']['embedding'] == 2
        assert snapshot['stage_retries']['embedding'] == 1
        assert snapshot['job_outcomes'] == {('embed', 'completed'): 1}
        assert ('index', 'embedding', 'completed') in snapshot['stage_count']
        assert snapshot['index_queue_depth'] == snapshot['index_inflight'] == 0
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_terminal_provider_failure_emits_failed_job_and_zero_gauges(client, monkeypatch):
    collector.reset()
    monkeypatch.setattr(embeddings, 'load_config', lambda: api._config)
    monkeypatch.setattr(api._config, 'index_job_max_attempts', 1)
    provider = embeddings.OpenRouterEmbeddings(api_key='synthetic', dimensions=4)
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(503, json={'error': 'synthetic'})
    ))
    api._embedder = provider
    try:
        stored = await client.post('/v1/store', json={'text': 'Synthetic terminal failure metric.'})
        assert stored.status_code == 200
        await test_api.drain_indexing()
        status = await client.get('/v1/operations/' + stored.json()['request_id'],
                                  params={'operation': 'store'})
        assert status.status_code == 200 and status.json()['state'] == 'failed'
        deadline = time.monotonic() + 1
        while collector.snapshot()['job_outcomes'].get(('embed', 'failed')) != 1:
            assert time.monotonic() < deadline
            await asyncio.sleep(.01)
        snapshot = collector.snapshot()
        assert snapshot['stage_attempts']['embedding'] == 2
        assert snapshot['stage_retries']['embedding'] == 1
        assert snapshot['job_outcomes'] == {('embed', 'failed'): 1}
        assert ('index', 'embedding', 'failed') in snapshot['stage_count']
        assert snapshot['index_queue_depth'] == snapshot['index_inflight'] == 0
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_forget_during_provider_emits_blocked_job_without_stale_commit(client, monkeypatch):
    from agent_memory.index_worker import IndexWorker

    collector.reset()
    monkeypatch.setattr(embeddings, 'load_config', lambda: api._config)
    provider = embeddings.OpenRouterEmbeddings(api_key='synthetic', dimensions=4)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def reply(request):
        import json
        entered.set()
        await release.wait()
        texts = json.loads(request.content)['input']
        return httpx.Response(200, json={'data': [
            dict(index=i, embedding=[1., 0., 0., 0.]) for i in range(len(texts))
        ]})

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
    api._embedder = provider
    stored = await client.post('/v1/store', json={
        'text': 'Synthetic blocked runtime metric.', 'session_id': 'metric-source'
    })
    assert stored.status_code == 200
    worker = IndexWorker(api._storage_pool, api._embedder, api._config)
    await worker.start()
    try:
        await asyncio.wait_for(entered.wait(), 1)
        forgotten = await client.request('DELETE', '/v1/forget', json={'id': stored.json()['id']})
        assert forgotten.status_code == 200 and forgotten.json()['deleted']
        release.set()
        deadline = time.monotonic() + 2
        storage = api._storage_pool.get('main')
        while (collector.snapshot()['job_outcomes'].get(('embed', 'blocked')) != 1
               or ('index', 'embedding', 'blocked') not in collector.snapshot()['stage_count']):
            assert time.monotonic() < deadline
            await asyncio.sleep(.01)
        while storage._get_conn().execute(
                "SELECT count(*) FROM memory_index_jobs WHERE state IN ('pending','running')"
        ).fetchone()[0]:
            assert time.monotonic() < deadline
            await asyncio.sleep(.01)
        snapshot = collector.snapshot()
        assert snapshot['job_outcomes'] == {('embed', 'blocked'): 1}
        assert ('index', 'embedding', 'blocked') in snapshot['stage_count']
        assert snapshot['index_queue_depth'] == snapshot['index_inflight'] == 0
        assert storage._get_conn().execute('SELECT count(*) FROM memory_vectors').fetchone()[0] == 0
    finally:
        release.set()
        await worker.stop()
        await provider.aclose()
