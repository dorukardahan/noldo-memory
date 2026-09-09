"""Real API/storage tests with independent synthetic evidence and temporary DBs."""

import pytest

from tests.test_api import _init_api_state  # noqa: F401
from httpx import AsyncClient, ASGITransport
from agent_memory.api import app


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as value:
        yield value


@pytest.mark.asyncio
async def test_existing_media_extraction_roundtrips_without_fetching(client):
    for modality in ('image', 'audio', 'document', 'link'):
        for role, delivery in [('user', 'received'), ('assistant', 'delivered')]:
            evidence = dict(modality=modality, representation='extracted_text', assertion='derived',
                            delivery=delivery, observed_at=1700000000, event_id=f'{modality}-{role}',
                            reference='https://example.org/observatory?expires=soon')
            response = await client.post('/v1/capture', json={'agent': 'alpha', 'messages': [{
                'role': role, 'session': 'session-a', 'evidence': evidence,
                'content': [{'type': 'text', 'text': f'The observatory {modality} describes a violet dome.'},
                            {'type': 'image_url', 'image_url': {'url': 'https://example.org/missing.png'}}],
            }]})
            assert response.status_code == 200
            assert response.json()['stored'] == 1
    recalled = (await client.post('/v1/recall', json={'agent': 'alpha', 'query': 'violet observatory dome', 'limit': 20})).json()
    assert len(recalled['results']) == 8
    for result in recalled['results']:
        assert result['source_session'] == 'session-a'
        assert result['evidence']['assertion'] == 'derived'
        assert result['evidence']['reference'] == 'https://example.org/observatory'
    other = (await client.post('/v1/recall', json={'agent': 'beta', 'query': 'violet observatory dome'})).json()
    assert other['results'] == []


@pytest.mark.asyncio
async def test_reference_only_is_not_invented_media_content(client):
    result = await client.post('/v1/capture', json={'messages': [
        {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'https://example.org/missing.png'}}]},
        {'role': 'user', 'text': 'https://example.org/missing.png',
         'evidence': {'modality': 'image', 'representation': 'reference_only'}},
    ]})
    assert result.status_code == 200
    assert result.json()['stored'] == 0


@pytest.mark.asyncio
async def test_revision_current_history_and_export_roundtrip(client):
    old = (await client.post('/v1/store', json={
        'agent': 'alpha', 'text': 'I prefer morning observatory visits.', 'session_id': 'session-a',
        'memory_type': 'preference',
    })).json()['id']
    revised = await client.post('/v1/store', json={
        'agent': 'alpha', 'text': 'I now prefer evening observatory visits.', 'session_id': 'session-b',
        'supersedes': old, 'valid_from': 1700000000,
    })
    assert revised.status_code == 200
    new = revised.json()['id']
    query = {'agent': 'alpha', 'query': 'observatory visits'}
    current = (await client.post('/v1/recall', json=query)).json()['results']
    assert [r['id'] for r in current] == [new]
    historical = (await client.post('/v1/recall', json={**query, 'as_of': 1699999999})).json()['results']
    assert [r['id'] for r in historical] == [old]
    both = (await client.post('/v1/recall', json={**query, 'include_history': True})).json()['results']
    assert {r['id'] for r in both} == {old, new}
    exported = (await client.get('/v1/export', params={'agent': 'alpha'})).json()
    assert next(r for r in exported if r['id'] == old)['valid_to'] == 1700000000
    imported = await client.post('/v1/import', json={'agent': 'recovery', 'memories': exported})
    assert imported.status_code == 200
    recovered = (await client.post('/v1/recall', json={**query, 'agent': 'recovery'})).json()['results']
    assert [r['id'] for r in recovered] == [new]


@pytest.mark.asyncio
async def test_revision_rejects_cross_scope_stale_and_inferred_updates(client):
    old = (await client.post('/v1/store', json={'agent': 'alpha', 'text': 'The observatory has a violet dome.'})).json()['id']
    update = {'text': 'The observatory has a silver dome.', 'supersedes': old}
    assert (await client.post('/v1/store', json={**update, 'agent': 'beta'})).status_code == 404
    assert (await client.post('/v1/store', json={**update, 'agent': 'alpha', 'evidence': {'assertion': 'inferred'}})).status_code == 422
    assert (await client.post('/v1/store', json={**update, 'agent': 'alpha'})).status_code == 200
    assert (await client.post('/v1/store', json={**update, 'agent': 'alpha'})).status_code == 409


@pytest.mark.asyncio
async def test_forget_removes_revision_history_and_cached_results(client):
    old = (await client.post('/v1/store', json={'text': 'The observatory has a violet dome.'})).json()['id']
    new = (await client.post('/v1/store', json={'text': 'The observatory has a silver dome.', 'supersedes': old})).json()['id']
    await client.post('/v1/recall', json={'query': 'observatory dome', 'include_history': True})
    assert (await client.request('DELETE', '/v1/forget', json={'id': new})).json()['deleted']
    remaining = (await client.post('/v1/recall', json={'query': 'observatory dome', 'include_history': True})).json()['results']
    assert remaining == []


def test_revision_failure_rolls_back_entire_change(tmp_storage, monkeypatch):
    old = tmp_storage.store_memory('The observatory has a violet dome.')
    original = tmp_storage.store_memory

    def fail_after_insert(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('synthetic failure')

    monkeypatch.setattr(tmp_storage, 'store_memory', fail_after_insert)
    with pytest.raises(RuntimeError):
        tmp_storage.revise_memory(old, text='A silver dome.', vector=None, valid_from=1700000000)
    assert tmp_storage.get_memory(old)['valid_to'] is None
    assert tmp_storage.stats()['total_memories'] == 1


@pytest.mark.asyncio
async def test_import_validates_entire_evidence_batch_before_writing(client):
    response = await client.post('/v1/import', json={'memories': [
        {'text': 'The synthetic observatory opens in the evening.'},
        {'text': 'The observatory closes before dawn.', 'valid_from': 20, 'valid_to': 10},
    ]})
    assert response.status_code == 422
    assert (await client.get('/v1/export')).json() == []
    response = await client.post('/v1/import', json={'memories': [
        {'id': 'a', 'text': 'The observatory opens in the evening.', 'supersedes': 'b', 'valid_from': 10, 'valid_to': 10},
        {'id': 'b', 'text': 'The observatory opens in the morning.', 'supersedes': 'a', 'valid_from': 10, 'valid_to': 10},
    ]})
    assert response.status_code == 422
    assert (await client.get('/v1/export')).json() == []


@pytest.mark.asyncio
async def test_automatic_admission_abstains_during_embedding_outage(client, monkeypatch):
    import agent_memory.api as api
    await client.post('/v1/store', json={'text': 'The observatory has a violet dome.'})

    async def unavailable(text):
        raise ConnectionError('synthetic outage')

    monkeypatch.setattr(api._embedder, 'embed', unavailable)
    response = (await client.post('/v1/recall', json={
        'query': 'violet observatory dome', 'min_semantic_score': .5,
    })).json()
    assert response['degraded'] and response['results'] == []
    explicit = (await client.post('/v1/recall', json={'query': 'violet observatory dome'})).json()
    assert explicit['results'], 'explicit lexical lookup remains available'


@pytest.mark.asyncio
async def test_exact_store_retry_avoids_another_embedding_call(client, monkeypatch):
    import agent_memory.api as api
    api._config.embed_worker_enabled = False
    calls = []
    embed = api._embedder.embed

    async def counted(text):
        calls.append(text)
        return await embed(text)

    monkeypatch.setattr(api._embedder, 'embed', counted)
    payload = {'text': 'I prefer quiet evening observatory visits.', 'session_id': 'session-a'}
    first = (await client.post('/v1/store', json=payload)).json()
    again = (await client.post('/v1/store', json=payload)).json()
    assert first['id'] == again['id']
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_capture_duplicate_events_skip_repeat_batch_embedding(client, monkeypatch):
    import agent_memory.api as api
    calls = []
    original = api._embedder.embed_batch

    async def counted(texts):
        calls.append(len(texts))
        return await original(texts)

    monkeypatch.setattr(api._embedder, 'embed_batch', counted)
    event = {'role': 'user', 'text': 'I prefer quiet observatory visits.', 'session': 'session-a',
             'evidence': {'event_id': 'synthetic-event-a'}}
    first = (await client.post('/v1/capture', json={'messages': [event, event]})).json()
    retry = (await client.post('/v1/capture', json={'messages': [event]})).json()
    assert first['stored'] == 1 and first['merged'] == 1
    assert retry['stored'] == 0 and retry['merged'] == 1
    assert calls == [1]


@pytest.mark.asyncio
async def test_historical_inference_matches_existing_admin_recall_scope(client):
    old = (await client.post('/v1/store', json={
        'agent': 'alpha', 'text': 'The observatory had a violet dome.',
    })).json()['id']
    new = (await client.post('/v1/store', json={
        'agent': 'alpha', 'text': 'The observatory now has a silver dome.', 'supersedes': old,
    })).json()['id']
    for scope in ('alpha', 'all'):
        query = {'agent': scope, 'query': 'Previously the observatory dome'}
        history = (await client.post('/v1/recall', json=query)).json()['results']
        assert {r['id'] for r in history} == {old, new}
        current = (await client.post('/v1/recall', json={**query, 'include_history': False})).json()['results']
        assert [r['id'] for r in current] == [new]


def test_decay_preserves_revision_family_but_still_archives_unversioned_rows(tmp_storage):
    import time
    old = tmp_storage.store_memory('The observatory had a violet dome.', importance=.2)
    new = tmp_storage.revise_memory(old, text='The observatory has a silver dome.',
                                    vector=None, valid_from=1700000000)['id']
    ordinary = tmp_storage.store_memory('An unrelated faded observatory note.', importance=.2)
    conn = tmp_storage._get_conn()
    conn.execute('UPDATE memories SET created_at = ?, last_accessed_at = ?, strength = .3',
                 (time.time() - 120 * 86400, time.time() - 120 * 86400))
    conn.commit()
    tmp_storage.decay_all()
    rows = {r['id']: r for r in conn.execute('SELECT id, deleted_at FROM memories')}
    assert rows[old]['deleted_at'] is None
    assert rows[new]['deleted_at'] is None
    assert rows[ordinary]['deleted_at'] is not None
    assert {r['id'] for r in tmp_storage.search_text('observatory', include_history=True)} == {old, new}
    assert tmp_storage.delete_memory(new)
    assert tmp_storage.search_text('observatory', include_history=True) == []
