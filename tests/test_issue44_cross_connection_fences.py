"""Cross-connection forget fencing and durable indexing status timing."""
import asyncio
import threading
import time
import uuid

import pytest

from agent_memory import api, operations
from agent_memory.storage import MemoryStorage
from tests import test_api

client = test_api.client
_init_api_state = test_api._init_api_state


@pytest.mark.asyncio
@pytest.mark.parametrize('forget_during_rerank', [False, True])
async def test_owned_recall_drops_snapshot_forgotten_during_rerank(client, monkeypatch, forget_during_rerank):
    storage = api._get_storage()
    secret = "Synthetic copper telescope memory that will be forgotten."
    mid = storage.store_memory(secret, vector=[1., 0., 0., 0.], source_session='synthetic-rerank-forget')
    storage.store_memory("Synthetic copper telescope independent control.", vector=[1., 0., 0., 0.])
    entered, release = threading.Event(), threading.Event()

    class PausedReranker:
        top_k = 10

        def score(self, query, texts, ids):
            assert secret in texts
            entered.set()
            assert release.wait(3)
            return [1.] * len(texts)

    monkeypatch.setattr(api, '_reranker', PausedReranker())
    pending = asyncio.create_task(client.post('/v1/recall', json={
        'query': 'synthetic copper telescope', 'min_score': 0, 'limit': 10,
    }))
    try:
        assert await asyncio.to_thread(entered.wait, 2), 'rerank gate was not reached'
        if forget_during_rerank:
            forgotten = await client.request('DELETE', '/v1/forget', json={'id': mid})
            assert forgotten.status_code == 200, forgotten.text
    finally:
        release.set()
        response = await pending
    assert response.status_code == 200, response.text
    if forget_during_rerank:
        assert secret not in response.text
        assert mid not in [row['id'] for row in response.json()['results']]
    else:
        assert secret in response.text


def test_generation_is_shared_per_shard_without_cache_write_or_open_reset(tmp_path):
    writer = MemoryStorage(str(tmp_path/'shared.sqlite'), dimensions=4)
    observer = MemoryStorage.open_existing(str(tmp_path/'shared.sqlite'), 4)
    other = MemoryStorage(str(tmp_path/'other.sqlite'), dimensions=4)
    reopened = None
    try:
        initial = observer.cache_generation
        writer._get_conn().execute('DELETE FROM embedding_cache')
        writer._get_conn().commit()
        assert observer.cache_generation == initial
        other.invalidate_search_cache()
        assert observer.cache_generation == initial
        writer.invalidate_search_cache()
        assert observer.cache_generation == initial + 1
        reopened = MemoryStorage(str(tmp_path/'shared.sqlite'), dimensions=4)
        assert reopened.cache_generation == observer.cache_generation
    finally:
        if reopened is not None:
            reopened.close()
        other.close()
        observer.close()
        writer.close()


@pytest.mark.parametrize('outcome', ['completed', 'failed'])
def test_public_elapsed_includes_terminal_index_job_timestamp(tmp_path, outcome):
    storage = MemoryStorage(str(tmp_path/'timing.sqlite'), dimensions=4)
    key = str(uuid.uuid4())
    try:
        accepted = operations.accept(storage, namespace='default', operation='store',
            request_id=key, payload={'synthetic': outcome}, total=1,
            rows=[{'text': 'Synthetic indexed timing assertion.', 'namespace': 'default',
                   'category': 'other', 'importance': .5, 'source_session': None}],
            embed_required=True, graph_required=False, queue_cap=100,
            deadline=time.monotonic()+2)
        job = operations.claim(storage)
        assert job is not None
        created = time.time()-12
        with storage.transaction() as conn:
            conn.execute('UPDATE memory_operations SET created_at=?,updated_at=? WHERE request_id=?',
                         (created, created+.1, key))
        assert operations.finish(storage, job, error='synthetic_failure' if outcome=='failed' else None,
                                 max_attempts=1)
        result = operations.public_status(storage, 'default', 'store', key)
        assert result is not None
        assert result['state'] == outcome
        assert 11 <= result['timing']['elapsed_seconds'] <= 14
        assert result['durable'] and accepted['durable']
    finally:
        storage.close()
