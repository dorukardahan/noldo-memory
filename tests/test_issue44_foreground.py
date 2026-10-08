"""Owned foreground SQLite deadlines and lifecycle; disposable state only."""
import asyncio
import sqlite3
import threading
import time
import uuid

import pytest

from agent_memory import api
from agent_memory.storage import MemoryStorage
from tests import test_api

client = test_api.client
_init_api_state = test_api._init_api_state


@pytest.mark.asyncio
async def test_cold_open_is_off_loop_and_expiry_fences_commit(client, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    original = MemoryStorage._ensure_schema
    threads = []
    def stall(self):
        threads.append(threading.get_ident())
        entered.set()
        release.wait(20)
        return original(self)
    monkeypatch.setattr(MemoryStorage, '_ensure_schema', stall)
    key = str(uuid.uuid4())
    task = asyncio.create_task(client.post('/v1/store', json={
        'agent': 'cold', 'request_id': key, 'text': 'A synthetic cold assertion', 'timeout_ms': 100}))
    started = time.monotonic()
    try:
        await asyncio.sleep(.02)
        live_start = time.monotonic()
        live = await client.get('/v1/health/live')
        assert live.status_code == 200 and time.monotonic() - live_start < .25
        response = await task
        assert time.monotonic() - started < .4
        assert response.status_code == 504
        assert threads and threads[0] != threading.get_ident()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(.1)
    path = api._storage_pool._db_path('cold')
    with sqlite3.connect(path) as conn:
        tables = {row[0] for row in conn.execute('SELECT name FROM sqlite_master')}
        if 'memories' in tables:
            assert conn.execute('SELECT count(*) FROM memories').fetchone()[0] == 0
        if 'memory_operations' in tables:
            assert conn.execute('SELECT count(*) FROM memory_operations').fetchone()[0] == 0


@pytest.mark.asyncio
async def test_sql_writer_contention_keeps_live_and_status_responsive(client):
    key = str(uuid.uuid4())
    response = await client.post('/v1/store', json={'request_id': key, 'text': 'A synthetic admitted assertion'})
    assert response.status_code == 200
    conn = sqlite3.connect(api._storage_pool._db_path('main'))
    conn.execute('BEGIN IMMEDIATE')
    try:
        started = time.monotonic()
        writes = [asyncio.create_task(client.post('/v1/store', json={
            'text': f'Another synthetic assertion {i}', 'timeout_ms': 100})) for i in range(8)]
        live, status = await asyncio.gather(client.get('/v1/health/live'), client.get(
            f'/v1/operations/{key}', params={'operation': 'store'}))
        assert time.monotonic() - started < .25
        assert live.status_code == status.status_code == 200
        responses = await asyncio.gather(*writes)
        assert time.monotonic() - started < .5
        assert all(item.status_code in {429, 503, 504} for item in responses)
    finally:
        conn.rollback()
        conn.close()
    assert api._get_storage().stats()['total_memories'] == 1


@pytest.mark.asyncio
async def test_recall_sync_search_work_retains_bounded_owner_not_event_loop(client, monkeypatch):
    warm = await client.post('/v1/recall', json={'query': 'Warm synthetic control'})
    assert warm.status_code == 200
    original = MemoryStorage.search_text
    threads = []
    def stall(self, *args, **kwargs):
        threads.append(threading.get_ident())
        time.sleep(.5)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(MemoryStorage, 'search_text', stall)
    started = time.monotonic()
    task = asyncio.create_task(client.post('/v1/recall', json={
        'query': 'Synthetic unique cold query', 'timeout_ms': 100}))
    await asyncio.sleep(.02)
    live = await client.get('/v1/health/live')
    response = await task
    assert live.status_code == 200
    assert time.monotonic() - started < .4
    assert response.status_code in {429, 503, 504}
    assert threads and all(thread != threading.get_ident() for thread in threads)
    await asyncio.sleep(.55)  # Drain the actual bounded work, not only the await.


@pytest.mark.asyncio
async def test_backend_existing_shard_never_calls_mutating_constructor(client, monkeypatch):
    storage = api._storage_pool.get('existing')
    storage.store_memory('Synthetic retained assertion')
    before = storage._get_conn().total_changes
    storage.close()
    api._storage_pool._storages.pop('existing')
    calls = []
    original = MemoryStorage.__init__
    def forbidden(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(MemoryStorage, '__init__', forbidden)
    api._embedder = None
    response = await client.get('/v1/health/backend', params={'agent': 'existing'})
    assert response.status_code == 200
    assert calls == []
    assert before > 0
