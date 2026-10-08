"""Regression-first acceptance repairs, entirely disposable source state."""
import asyncio
import gc
import sqlite3
import time
import warnings
import uuid

import pytest

from agent_memory import api
from agent_memory.config import Config
from agent_memory.index_worker import IndexWorker
from agent_memory.pool import StoragePool
from agent_memory.recall_owner import ProviderBridge
from agent_memory.storage import MemoryStorage
from scripts import benchmark_issue44
from tests import test_api

client = test_api.client
_init_api_state = test_api._init_api_state


def test_bridge_closed_owner_never_allocates_unawaited_coroutine():
    owner = asyncio.new_event_loop()
    owner.close()
    bridge = ProviderBridge(object(), owner, time.monotonic() + 1)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        with pytest.raises((RuntimeError, TimeoutError)):
            asyncio.run(bridge.embed('Synthetic closed-owner query'))
        gc.collect()
    assert not any('never awaited' in str(item.message) for item in observed)


@pytest.mark.asyncio
async def test_bridge_deadline_keeps_owner_until_provider_cancellation_drains():
    started = asyncio.Event()
    release = asyncio.Event()
    drained = asyncio.Event()

    class Provider:
        async def embed(self, text):
            started.set()
            try:
                await asyncio.sleep(20)
            finally:
                await release.wait()
                drained.set()

    bridge = ProviderBridge(Provider(), asyncio.get_running_loop(), time.monotonic() + .1)
    work = asyncio.create_task(asyncio.to_thread(lambda: asyncio.run(bridge.embed('Synthetic stalled query'))))
    try:
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.sleep(.2)
        assert not work.done(), 'foreground capacity released before actual provider drain'
    finally:
        release.set()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(work, 2)
    assert drained.is_set()


def test_index_connection_never_reruns_corpus_migration(tmp_path, monkeypatch):
    pool = StoragePool(str(tmp_path), 4)
    storage = pool.get()
    storage.store_memory('Synthetic retained corpus assertion')
    conn = storage._get_conn()
    conn.execute('UPDATE memories SET last_accessed_at=NULL')
    conn.commit()
    worker = IndexWorker(pool, None, Config())
    calls = []
    original = MemoryStorage.__init__

    def track(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(MemoryStorage, '__init__', track)
    try:
        opened = worker._open('main')
        assert calls == [], 'worker reran schema/backfill under ordinary foreground writes'
        assert opened._get_conn().execute('SELECT last_accessed_at FROM memories').fetchone()[0] is None
    finally:
        worker._close()
        pool.close_all()


@pytest.mark.asyncio
async def test_short_writer_contention_uses_remaining_acceptance_budget(client):
    conn = sqlite3.connect(api._storage_pool._db_path('main'))
    conn.execute('BEGIN IMMEDIATE')
    work = asyncio.create_task(client.post('/v1/store', json={
        'text': 'Synthetic assertion after brief contention', 'timeout_ms': 1000}))
    try:
        await asyncio.sleep(.12)
    finally:
        conn.rollback()
        conn.close()
    response = await work
    assert response.status_code == 200, 'arbitrary 50ms lock wait discarded remaining operation budget'


@pytest.mark.asyncio
async def test_parallel_cold_writes_share_one_schema_bootstrap(client):
    responses = await asyncio.gather(*(client.post('/v1/store', json={
        'agent': 'synthetic-cold', 'request_id': str(uuid.uuid4()),
        'text': f'Synthetic cold assertion number {i}'}) for i in range(3)))
    assert all(response.status_code == 200 for response in responses)


@pytest.mark.asyncio
async def test_revision_replay_uses_ledger_before_mutable_predecessor(client):
    previous = await client.post('/v1/store', json={'text': 'Synthetic old durable assertion'})
    assert previous.status_code == 200
    payload = {'text': 'Synthetic replacement assertion', 'request_id': str(uuid.uuid4()),
               'supersedes': previous.json()['id'], 'session_id': 'synthetic-revision-source'}
    first = await client.post('/v1/store', json=payload)
    assert first.status_code == 200
    replay = await client.post('/v1/store', json=payload)
    assert replay.status_code == 200
    assert replay.json()['id'] == first.json()['id']
    forgotten = await client.request('DELETE', '/v1/forget', json={'id': first.json()['id']})
    assert forgotten.status_code == 200
    replay = await client.post('/v1/store', json=payload)
    assert replay.status_code == 200 and replay.json()['state'] == 'blocked'
    conflict = await client.post('/v1/store', json={**payload, 'text': 'Changed synthetic replacement'})
    assert conflict.status_code == 409
    assert api._get_storage().stats()['total_memories'] == 0


@pytest.mark.parametrize('status', [429, 503, 504])
def test_benchmark_rejects_any_baseline_failure_even_with_accepted_floor(status):
    rows = [{'concurrency': 1, 'http_status': status}]
    report = {'distributions': {'capture': {'successes': 120, 'admitted_over_budget': 0}},
              'backlog': [], 'upper_envelope': {'passed': True}}
    assert benchmark_issue44.acceptance_failed(rows, report)


def test_benchmark_keeps_explicit_upper_overload_separate():
    rows = [{'concurrency': 4, 'http_status': 429}, {'concurrency': 8, 'http_status': 429}]
    report = {'distributions': {'capture': {'successes': 120, 'admitted_over_budget': 0}},
              'backlog': [], 'upper_envelope': {'passed': True}}
    assert not benchmark_issue44.acceptance_failed(rows, report)


def test_benchmark_rejected_write_cannot_mutate():
    rows = [{'concurrency': 4, 'http_status': 429, 'rejected_no_mutation': False}]
    report = {'distributions': {'capture': {'successes': 120, 'admitted_over_budget': 0}},
              'backlog': [], 'upper_envelope': {'passed': True}}
    assert benchmark_issue44.acceptance_failed(rows, report)
