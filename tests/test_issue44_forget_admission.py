"""Source fencing preserves independent assertions and never revives old work."""
import time
import uuid

import pytest

from agent_memory import api, operations
from agent_memory.embed_worker import EmbedWorker
from agent_memory.pool import StoragePool
from tests import test_api

client = test_api.client
_init_api_state = test_api._init_api_state


def admit(storage, text, *, namespace='default', session='same-synthetic-source', key=None):
    key = key or str(uuid.uuid4())
    row = dict(text=text, namespace=namespace, source_session=session, category='fact', importance=.5,
               source='capture', trust_level='user', evidence={'kind': 'reported', 'role': 'user'})
    args = dict(namespace=namespace, operation='capture', request_id=key,
                payload=[row], rows=[row], total=1, embed_required=True,
                graph_required=True, queue_cap=1000, deadline=time.monotonic() + 2)
    receipt = operations.accept(storage, **args)
    mid = storage._get_conn().execute(
        'SELECT id FROM memories WHERE text=? AND namespace=? AND source_session IS ? ORDER BY created_at DESC',
        (text, namespace, session)).fetchone()[0]
    return key, mid, args, receipt


@pytest.mark.asyncio
async def test_forget_fences_another_committed_operation_from_same_source(client):
    first_key, second_key = str(uuid.uuid4()), str(uuid.uuid4())
    async def write(key, text):
        response = await client.post('/v1/capture', json={'request_id': key, 'messages': [
            {'role': 'user', 'text': text, 'session': 'same-synthetic-source'}]})
        assert response.status_code == 200
        return response.json()
    await write(first_key, 'Synthetic assertion one uses Python and SQLite.')
    await write(second_key, 'Synthetic assertion two uses Rust and PostgreSQL.')
    storage = api._get_storage()
    rows = storage._get_conn().execute('SELECT id FROM memories ORDER BY created_at').fetchall()
    sibling = storage.get_memory(rows[1]['id'])
    storage.forget_memory(rows[0]['id'])
    for key in (first_key, second_key):
        assert operations.public_status(storage, 'default', 'capture', key)['state'] == 'blocked'
    assert storage.get_memory(rows[1]['id']) == sibling
    storage.relearn_source('same-synthetic-source')
    assert operations.public_status(storage, 'default', 'capture', second_key)['state'] == 'blocked'


@pytest.mark.parametrize('stage_state', ['pending', 'running', 'completed'])
@pytest.mark.parametrize('namespace', ['default', 'other'])
def test_source_fence_is_permanent_without_sibling_deletion(tmp_path, stage_state, namespace):
    pool = StoragePool(str(tmp_path), 4)
    try:
        storage = pool.get('main')
        first_key, first_id, _, _ = admit(storage, 'Python uses SQLite')
        key, sibling_id, replay_args, original = admit(storage, 'Rust uses PostgreSQL', namespace=namespace)
        unrelated_key, unrelated_id, _, _ = admit(storage, 'Redis uses memory', session='independent')
        other_storage = pool.get('other-agent')
        other_key, other_id, _, _ = admit(other_storage, 'Rust uses PostgreSQL')
        conn = storage._get_conn()
        with storage.transaction():
            conn.execute('UPDATE memory_index_jobs SET state=?,attempts=1,lease_until=? WHERE memory_id=?',
                         (stage_state, time.time() + 35, sibling_id))
        sibling = storage.get_memory(sibling_id)
        # Old generation must be rejected even after relearn/reopen.
        stale_job = dict(memory_id=sibling_id, stage='embed', attempts=1)
        storage.forget_memory(first_id)
        assert storage.get_memory(first_id) is None
        assert storage.get_memory(sibling_id) == sibling
        for ns, identity in [('default', first_key), (namespace, key)]:
            status = operations.public_status(storage, ns, 'capture', identity)
            assert status['state'] == 'blocked' and status['durable'] is True
            assert status['counts']['stored'] == 1
        jobs = conn.execute('SELECT state,attempts,lease_until FROM memory_index_jobs WHERE memory_id=?',
                            (sibling_id,)).fetchall()
        assert all(tuple(row) == ('blocked', 2, None) for row in jobs)
        assert operations.public_status(storage, 'default', 'capture', unrelated_key)['state'] == 'accepted'
        assert storage.get_memory(unrelated_id) is not None
        assert operations.public_status(other_storage, 'default', 'capture', other_key)['state'] == 'accepted'
        assert other_storage.get_memory(other_id) is not None
        storage.relearn_source('same-synthetic-source')
        storage.close()
        assert not operations.finish(storage, stale_job, vector=[1., 0., 0., 0.])
        replay_args['deadline'] = time.monotonic() + 2
        replay = operations.accept(storage, **replay_args)
        assert replay['state'] == 'blocked' and replay['stored'] == original['stored']
        assert storage.get_memory(sibling_id) == sibling
        # Exact-provenance new ingestion must create a new generation, not merge
        # into the retained but permanently fenced sibling's jobs.
        fresh_key, fresh_id, _, _ = admit(storage, 'Rust uses PostgreSQL', namespace=namespace)
        assert fresh_id != sibling_id
        assert operations.public_status(storage, namespace, 'capture', fresh_key)['state'] == 'accepted'
        legacy = EmbedWorker(pool, None)
        assert sibling_id not in [row['id'] for row in legacy._get_vectorless_memories(storage)]
    finally:
        pool.close_all()


def test_coadmission_blocks_receipt_but_preserves_independent_family(tmp_path):
    pool = StoragePool(str(tmp_path), 4)
    try:
        storage = pool.get()
        rows = [dict(text=text, namespace='default', source_session='co-source', source='capture',
                     category='fact', importance=.5)
                for text in ['Python uses SQLite', 'Rust uses PostgreSQL']]
        key = str(uuid.uuid4())
        receipt = operations.accept(storage, namespace='default', operation='capture', request_id=key,
            payload=rows, rows=rows, total=2, embed_required=True, graph_required=True,
            queue_cap=1000, deadline=time.monotonic() + 2)
        ids = [row[0] for row in storage._get_conn().execute('SELECT id FROM memories ORDER BY created_at')]
        sibling = storage.get_memory(ids[1])
        storage.forget_memory(ids[0])
        status = operations.public_status(storage, 'default', 'capture', key)
        assert status['state'] == 'blocked' and status['counts']['stored'] == receipt['stored'] == 2
        assert status['indexing_state'] == 'blocked'
        assert storage.get_memory(ids[1]) == sibling
        storage.relearn_source('co-source')
        assert storage._get_conn().execute('SELECT DISTINCT state FROM memory_index_jobs').fetchall()[0][0] == 'blocked'
    finally:
        pool.close_all()


def test_forget_without_identified_source_does_not_infer_text_match(tmp_path):
    pool = StoragePool(str(tmp_path), 4)
    try:
        storage = pool.get()
        _, first, _, _ = admit(storage, 'Identical synthetic assertion', session=None)
        key, sibling, _, _ = admit(storage, 'Identical synthetic assertion', session='explicit-source')
        storage.forget_memory(first)
        assert storage.get_memory(sibling) is not None
        assert operations.public_status(storage, 'default', 'capture', key)['state'] == 'accepted'
    finally:
        pool.close_all()
