"""Disposable durable-worker, transaction, lease and provenance regressions."""
import asyncio
import importlib
import threading
import time
import uuid
from types import SimpleNamespace

import pytest

from agent_memory import operations
from agent_memory.embed_worker import EmbedWorker
from agent_memory.entities import KnowledgeGraph
from agent_memory.pool import StoragePool


@pytest.fixture
def pool(tmp_path):
    result = StoragePool(str(tmp_path), 4)
    yield result
    result.close_all()


def admit(storage, *, graph=False, embed=True, session='synthetic', text='Python uses SQLite'):
    key = str(uuid.uuid4())
    row = dict(text=text, category='fact', importance=.5, source_session=session,
               namespace='default', source='capture', trust_level='user',
               evidence={'kind': 'reported', 'role': 'user'})
    with storage.transaction() as conn:
        mid = storage.store_memory(**row, vector=None)
        now = time.time()
        for stage in (['embed'] if embed else []) + (['graph'] if graph else []):
            conn.execute('INSERT INTO memory_index_jobs VALUES (?,?,?,?,?,?,?,?,?)',
                         (mid, stage, 'pending', 0, None, now, None, now, now))
        import json
        conn.execute('INSERT INTO memory_operations VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            ('default', 'capture', key, operations.fingerprint(row), 'accepted', 1,
             json.dumps({'stored': 1, 'total': 1}), json.dumps([mid]), None, now, now))
    return key, mid, operations.public_status(storage, 'default', 'capture', key)


class Embedder:
    def __init__(self, delay=0):
        self.delay = delay
        self.calls = []
        self.entered = asyncio.Event()

    async def embed(self, text, *, deadline, background, cache):
        self.calls.append((asyncio.get_running_loop(), deadline, background, cache))
        self.entered.set()
        await asyncio.sleep(self.delay)
        return [1., 0., 0., 0.]


def worker(pool, embedder):
    module = importlib.import_module('agent_memory.index_worker')
    return module.IndexWorker(pool, embedder, SimpleNamespace(
        index_job_timeout_seconds=1, index_job_max_attempts=3, embed_worker_enabled=False))


async def eventually(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, 'durable worker did not reach expected state'
        await asyncio.sleep(.01)


def test_storage_migrates_without_explicit_operations_call(pool):
    conn = pool.get()._get_conn()
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='memory_index_jobs'").fetchone()
    assert conn.execute('SELECT count(*) FROM memory_index_jobs').fetchone()[0] == 0


def test_vector_stage_rolls_back_under_outer_transaction(pool):
    storage = pool.get()
    operations.migrate(storage._get_conn())
    storage._get_conn().commit()
    _, mid, _ = admit(storage)
    job = operations.claim(storage)
    with pytest.raises(RuntimeError, match='rollback'):
        with storage.transaction():
            assert operations.finish(storage, job, vector=[1., 0., 0., 0.])
            raise RuntimeError('rollback')
    assert storage.get_memory(mid)['vector_rowid'] is None
    assert storage._get_conn().execute('SELECT state FROM memory_index_jobs').fetchone()[0] == 'running'
    assert storage._get_conn().execute('SELECT count(*) FROM memory_vectors').fetchone()[0] == 0


def test_completed_vector_invalidates_live_cache_atomically(pool):
    storage = pool.get()
    _, mid, _ = admit(storage)
    storage.cache_search_result('synthetic', 1, 0., 'main', '[]')
    job = operations.claim(storage)
    assert operations.finish(storage, job, vector=[1., 0., 0., 0.])
    assert storage.get_cached_search_result('synthetic', 1, 0., 'main') is None


def test_graph_persistence_failure_rolls_back_graph_and_stage(pool):
    storage = pool.get()
    _, mid, _ = admit(storage, graph=True, embed=False)
    job = operations.claim(storage)
    def fail(memory):
        KnowledgeGraph(storage).process_text(memory['text'], source_memory_id=mid)
        raise RuntimeError('synthetic graph failure')
    with pytest.raises(RuntimeError):
        operations.finish(storage, job, graph=fail)
    assert storage._get_conn().execute('SELECT count(*) FROM entities').fetchone()[0] == 0
    assert storage._get_conn().execute('SELECT count(*) FROM graph_sources').fetchone()[0] == 0
    assert storage._get_conn().execute('SELECT state FROM memory_index_jobs').fetchone()[0] == 'running'


@pytest.mark.asyncio
async def test_provider_stall_fails_bounded_attempt_without_vector(pool):
    storage = pool.get()
    _, mid, _ = admit(storage)
    embedder = Embedder(20)
    task = worker(pool, embedder)
    await task.start()
    try:
        await asyncio.wait_for(embedder.entered.wait(), 1)
        started = time.monotonic()
        await eventually(lambda: storage._get_conn().execute('SELECT state FROM memory_index_jobs').fetchone()[0] == 'pending')
        assert time.monotonic() - started < 1.4
        assert storage.get_memory(mid)['vector_rowid'] is None
        assert storage._get_conn().execute('SELECT attempts,error_code FROM memory_index_jobs').fetchone()[:] == (1, 'index_deadline_exceeded')
    finally:
        await task.stop()


def test_revision_and_forget_respect_outer_transaction(pool):
    storage = pool.get()
    operations.migrate(storage._get_conn())
    storage._get_conn().commit()
    _, mid, _ = admit(storage)
    with pytest.raises(RuntimeError, match='rollback'):
        with storage.transaction():
            storage.revise_memory(mid, text='Python now uses PostgreSQL', vector=None,
                                  valid_from=time.time(), source_session='synthetic')
            storage.forget_memory(mid)
            raise RuntimeError('rollback')
    assert storage.get_memory(mid)['valid_to'] is None
    assert storage.stats()['total_memories'] == 1
    assert not storage.source_is_forgotten('synthetic')


def test_legacy_scan_excludes_all_job_owned_rows(pool):
    storage = pool.get()
    operations.migrate(storage._get_conn())
    storage._get_conn().commit()
    _, mid, _ = admit(storage)
    legacy_id = storage.store_memory('An old vectorless fact', source_session='legacy')
    legacy = EmbedWorker(pool, Embedder())
    assert legacy._count_vectorless_memories(storage) == 1
    assert [row['id'] for row in legacy._get_vectorless_memories(storage)] == [legacy_id]
    assert not legacy._update_memory_vector(storage, mid, [1., 0., 0., 0.])


@pytest.mark.asyncio
async def test_worker_uses_owned_sqlite_and_parent_provider_loop(pool):
    storage = pool.get()
    operations.migrate(storage._get_conn())
    storage._get_conn().commit()
    key, mid, _ = admit(storage, graph=True)
    embedder = Embedder()
    task = worker(pool, embedder)
    await task.start()
    try:
        await eventually(lambda: operations.public_status(storage, 'default', 'capture', key)['state'] == 'completed')
        assert storage.get_memory(mid)['vector_rowid'] is not None
        assert storage._get_conn().execute('SELECT count(*) FROM graph_sources WHERE memory_id=?', (mid,)).fetchone()[0] > 0
        assert embedder.calls[0][0] is asyncio.get_running_loop()
        assert embedder.calls[0][2:] == (True, False)
        assert storage.get_memory(mid)['evidence'] == '{"kind": "reported", "role": "user"}'
    finally:
        await task.stop()


@pytest.mark.asyncio
async def test_forget_during_provider_then_relearn_never_resurrects(pool):
    storage = pool.get()
    operations.migrate(storage._get_conn())
    storage._get_conn().commit()
    key, mid, _ = admit(storage)
    embedder = Embedder(.15)
    task = worker(pool, embedder)
    await task.start()
    try:
        await asyncio.wait_for(embedder.entered.wait(), 1)
        storage.forget_memory(mid)
        storage.relearn_source('synthetic')
        await asyncio.sleep(.25)
        assert storage.get_memory(mid) is None
        assert operations.public_status(storage, 'default', 'capture', key)['state'] == 'blocked'
        assert storage._get_conn().execute('SELECT count(*) FROM memory_vectors').fetchone()[0] == 0
        fresh_key, fresh_mid, _ = admit(storage)
        await eventually(lambda: operations.public_status(storage, 'default', 'capture', fresh_key)['state'] == 'completed')
        assert fresh_mid != mid
    finally:
        await task.stop()


def test_expired_attempt_cannot_write_after_restart_claim(pool):
    storage = pool.get()
    operations.migrate(storage._get_conn())
    storage._get_conn().commit()
    _, mid, _ = admit(storage)
    first = operations.claim(storage)
    with storage.transaction() as conn:
        conn.execute('UPDATE memory_index_jobs SET lease_until=?', (time.time() - 1,))
    storage.close()  # Reload epoch leases through a real reopened connection.
    second = operations.claim(storage)
    assert second['attempts'] == first['attempts'] + 1
    assert 34 < second['lease_until'] - time.time() <= 35
    assert not operations.finish(storage, first, vector=[1., 0., 0., 0.])
    assert operations.finish(storage, second, vector=[0., 1., 0., 0.])
    assert storage.get_memory(mid)['vector_rowid'] is not None


@pytest.mark.asyncio
async def test_graph_deadline_retains_slot_until_thread_drains(pool, monkeypatch):
    storage = pool.get()
    operations.migrate(storage._get_conn())
    storage._get_conn().commit()
    key, mid, _ = admit(storage, graph=True, embed=False)
    _, other, _ = admit(storage, graph=True, embed=False, session='other', text='PostgreSQL and Redis')
    entered = threading.Event()
    released = threading.Event()
    calls = []
    original = getattr(KnowledgeGraph, 'prepare_text', None)
    assert original is not None, 'graph preparation must be separable from persistence'
    def stall(self, *args, **kwargs):
        calls.append(threading.get_ident())
        entered.set()
        released.wait(20)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(KnowledgeGraph, 'prepare_text', stall)
    task = worker(pool, Embedder())
    await task.start()
    try:
        await eventually(entered.is_set)
        started = time.monotonic()
        await asyncio.sleep(1.1)
        assert time.monotonic() - started < 1.4
        assert len(calls) == 1
        assert calls[0] != threading.get_ident()
        storage.forget_memory(mid)
        storage.relearn_source('synthetic')
        assert storage._get_conn().execute('SELECT count(*) FROM entities').fetchone()[0] == 0
        released.set()
        await eventually(lambda: storage._get_conn().execute("SELECT state FROM memory_index_jobs WHERE memory_id=?", (other,)).fetchone()[0] == 'completed')
        assert operations.public_status(storage, 'default', 'capture', key)['state'] == 'blocked'
        assert storage._get_conn().execute('SELECT count(*) FROM graph_sources WHERE memory_id=?', (mid,)).fetchone()[0] == 0
    finally:
        released.set()
        await task.stop()
