"""Disposable behavior regressions for durable operation admission and fencing."""
import asyncio

import time
import uuid

import pytest

from tests import test_api
import agent_memory.api as api
from agent_memory.storage import MemoryStorage

client = test_api.client
_init_api_state = test_api._init_api_state


def identity():
    return str(uuid.uuid4())


def body(key=None):
    return {'request_id': identity() if key is None else key, 'messages': [
        {'role': 'user', 'text': 'Ada builds a Python service and tests SQLite consistency.', 'session': 'synthetic-session'}]}


async def status(client, key, operation='capture', namespace='default'):
    return await client.get(f'/v1/operations/{key}', params={'operation': operation, 'namespace': namespace})


@pytest.mark.asyncio
async def test_capture_accepts_without_waiting_for_embedding(client, monkeypatch):
    async def stall(texts):
        await asyncio.sleep(20)
        return [[0.0] * 4 for _ in texts]
    monkeypatch.setattr(api._embedder, 'embed_batch', stall)
    request = body()
    started = time.monotonic()
    response = await client.post('/v1/capture', json=request)
    assert time.monotonic() - started < 2
    assert response.status_code == 200
    result = response.json()
    assert result['durable'] is True and result['state'] == 'accepted'
    assert result['stage_states'] == {'embed': 'pending', 'graph': 'pending'}
    storage = api._get_storage()
    assert storage._get_conn().execute('SELECT count(*) FROM memory_operations').fetchone()[0] == 1
    assert storage._get_conn().execute('SELECT count(*) FROM memory_index_jobs').fetchone()[0] == 2
    public = (await status(client, request['request_id'])).json()
    assert public['durable'] is True
    assert not {'id', 'memory_ids', 'text', 'session', 'namespace', 'agent', 'fingerprint'} & public.keys()


@pytest.mark.asyncio
async def test_same_identity_replay_and_payload_conflict(client):
    request = body()
    first = await client.post('/v1/capture', json=request)
    second = await client.post('/v1/capture', json={**request, 'timeout_ms': 1000})
    assert second.status_code == 200
    assert second.json()['stored'] == first.json()['stored']
    changed = {**request, 'messages': [{'role': 'user', 'text': 'A different assertion exists.'}]}
    conflict = await client.post('/v1/capture', json=changed)
    assert conflict.status_code == 409
    assert api._get_storage().stats()['total_memories'] == 1


@pytest.mark.asyncio
async def test_atomic_admission_rolls_back_every_row_and_receipt(client, monkeypatch):
    request = body()
    request['messages'].append({'role': 'user', 'text': 'A second durable assertion in this batch.'})
    original = MemoryStorage.merge_or_store
    calls = 0
    def failing(self, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('synthetic failure')
        return original(self, **kwargs)
    monkeypatch.setattr(MemoryStorage, 'merge_or_store', failing)
    response = await client.post('/v1/capture', json=request)
    assert response.status_code in {500, 503}
    storage = api._get_storage()
    assert storage.stats()['total_memories'] == 0
    assert (await status(client, request['request_id'])).status_code == 404


@pytest.mark.asyncio
async def test_queue_full_leaves_no_new_rows_or_receipt(client):
    api._config.index_queue_max_jobs = 1
    request = body()
    response = await client.post('/v1/capture', json=request)
    assert response.status_code == 429
    assert api._get_storage().stats()['total_memories'] == 0
    assert (await status(client, request['request_id'])).status_code == 404


@pytest.mark.asyncio
async def test_store_without_provider_completes_without_graph(client):
    api._embedder = None
    key = identity()
    response = await client.post('/v1/store', json={'text': 'An explicit durable SQLite fact.', 'request_id': key})
    assert response.status_code == 200
    assert response.json()['state'] == 'completed'
    assert response.json()['stage_states'] == {'embed': 'not_required', 'graph': 'not_required'}
    receipt = (await status(client, key, 'store')).json()
    assert receipt['state'] == 'completed' and receipt['durable'] is True


@pytest.mark.asyncio
async def test_forget_blocks_old_receipt_and_relearn_does_not_resurrect(client):
    request = body()
    response = await client.post('/v1/capture', json=request)
    assert response.status_code == 200
    storage = api._get_storage()
    mid = storage._get_conn().execute('SELECT id FROM memories').fetchone()[0]
    storage.forget_memory(mid)
    assert (await status(client, request['request_id'])).json()['state'] == 'blocked'
    storage.relearn_source('synthetic-session')
    replay = await client.post('/v1/capture', json=request)
    assert replay.json()['state'] == 'blocked'
    assert storage.stats()['total_memories'] == 0
    fresh = await client.post('/v1/capture', json={**request, 'request_id': identity()})
    assert fresh.json()['durable'] is True
    assert storage.stats()['total_memories'] == 1


@pytest.mark.asyncio
async def test_status_rejects_malformed_and_unknown_id(client):
    assert (await status(client, 'not-a-uuid')).status_code == 422
    assert (await status(client, identity())).status_code == 404


@pytest.mark.asyncio
async def test_operation_error_log_normalizes_identity(client, caplog):
    key = identity()
    response = await status(client, key)
    assert response.status_code == 404
    assert key not in caplog.text
    assert '/v1/operations/{request_id}' in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', ['ABC', str(uuid.uuid4()).upper(), 12, True, [], {}])
async def test_write_rejects_invalid_identity_before_mutation(client, bad):
    response = await client.post('/v1/capture', json=body(bad))
    assert response.status_code == 422
    assert api._get_storage().stats()['total_memories'] == 0


@pytest.mark.asyncio
async def test_total_normalized_char_cap_rejects_no_mutation(client):
    request = body()
    request['messages'] = [{'role': 'user', 'text': f'Assertion {i} '+ 'x' * 3900} for i in range(40)]
    response = await client.post('/v1/capture', json=request)
    assert response.status_code == 413
    assert api._get_storage().stats()['total_memories'] == 0


@pytest.mark.asyncio
async def test_mixed_32_row_long_multitool_batch_is_durably_admitted_within_budget(client):
    roles = ['user', 'assistant', 'tool']
    messages = [
        {
            'role': roles[i % len(roles)],
            'text': f'Synthetic eligible row {i:02d} records a bounded multitool assertion. ' + 'x' * 3060,
            'session': f'synthetic-upper-{i:02d}',
        }
        for i in range(32)
    ]
    messages.extend([
        {'role': 'system', 'text': 'Excluded control plane instruction.'},
        {'role': 'tool', 'text': 'HEARTBEAT_OK'},
    ])
    request = {'request_id': identity(), 'messages': messages}
    normalized = api._capture_candidates(api.CaptureRequest.model_validate(request))
    assert len(normalized) == 32
    assert 99_000 <= sum(len(row['text']) for row in normalized) <= 101_000

    started = time.monotonic()
    response = await client.post('/v1/capture', json=request)
    elapsed = time.monotonic() - started

    assert response.status_code == 200
    assert elapsed < 2
    receipt = response.json()
    assert receipt['durable'] is True and receipt['state'] == 'accepted'
    assert receipt['stored'] == 32
    storage = api._get_storage()
    assert storage.stats()['total_memories'] == 32
    assert storage._get_conn().execute('SELECT count(*) FROM memory_index_jobs').fetchone()[0] == 64


@pytest.mark.asyncio
async def test_200_message_control_accepts_and_201_rejects_without_mutation(client):
    control = [
        {'role': 'user', 'text': f'Synthetic bounded control assertion {i:03d}.',
         'session': f'synthetic-control-{i:03d}'}
        for i in range(200)
    ]
    accepted = await client.post('/v1/capture', json={'request_id': identity(), 'messages': control})
    assert accepted.status_code == 200
    assert accepted.json()['stored'] == 200
    assert api._get_storage().stats()['total_memories'] == 200

    before = api._get_storage()._get_conn().total_changes
    rejected = await client.post('/v1/capture', json={
        'request_id': identity(),
        'messages': control + [{'role': 'user', 'text': 'This 201st row must be rejected.'}],
    })
    assert rejected.status_code == 422
    assert api._get_storage().stats()['total_memories'] == 200
    assert api._get_storage()._get_conn().total_changes == before


@pytest.mark.asyncio
async def test_liveness_does_not_call_backend(client, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('must not access backend')
    monkeypatch.setattr(api, '_get_storage', forbidden)
    started = time.monotonic()
    response = await client.get('/v1/health/live')
    assert response.status_code == 200
    assert time.monotonic() - started < .25
    assert response.json() == {'status': 'ok'}


def test_old_schema_migration_adds_tables_without_old_jobs(tmp_path):
    path = str(tmp_path / 'old.sqlite')
    storage = MemoryStorage(path, dimensions=4)
    mid = storage.store_memory('An old assertion without a derived vector.')
    storage.close()
    storage = MemoryStorage(path, dimensions=4)
    try:
        conn = storage._get_conn()
        assert conn.execute('SELECT count(*) FROM memory_index_jobs').fetchone()[0] == 0
        assert storage.get_memory(mid)['text'] == 'An old assertion without a derived vector.'
        assert [row['name'] for row in conn.execute('PRAGMA table_info(memory_operations)')] == [
            'namespace', 'operation', 'request_id', 'fingerprint', 'state', 'durable', 'receipt_json',
            'memory_ids_json', 'error_code', 'created_at', 'updated_at']
    finally:
        storage.close()
