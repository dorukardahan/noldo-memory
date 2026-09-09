"""Real API/storage tests with independent synthetic evidence and temporary DBs."""

import pytest

from tests.test_api import _init_api_state  # noqa: F401
from httpx import AsyncClient, ASGITransport
from agent_memory.api import app


@pytest.fixture
async def client(monkeypatch):
    # Each independent test gets fresh middleware state, including the unchanged
    # 120-request limiter, just as it gets fresh API globals and databases.
    monkeypatch.setattr(app, "middleware_stack", app.build_middleware_stack())
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


@pytest.mark.asyncio
@pytest.mark.parametrize('stored_state', ['current', 'other_namespace', 'matching'])
async def test_partial_import_validates_the_parent_that_is_actually_retained(client, stored_state):
    import agent_memory.api as api
    storage = api._storage_pool.get('alpha')
    storage.store_memory('The observatory had a violet dome.', memory_id='parent',
                         namespace='other' if stored_state == 'other_namespace' else 'default',
                         valid_to=None if stored_state == 'current' else 10)
    response = await client.post('/v1/import', json={'agent': 'alpha', 'memories': [
        {'id': 'parent', 'text': 'The observatory had a blue dome.', 'valid_to': 10},
        {'id': 'child', 'text': 'The observatory has a silver dome.', 'valid_from': 10, 'supersedes': 'parent'},
    ]})
    if stored_state == 'matching':
        assert response.status_code == 200
        assert response.json()['imported'] == 1
    else:
        assert response.status_code == 422
        assert storage.get_memory('child') is None
    assert storage.get_memory('parent')['text'] == 'The observatory had a violet dome.'


@pytest.mark.asyncio
async def test_import_rejects_lineage_to_a_parent_skipped_for_empty_content(client):
    response = await client.post('/v1/import', json={'memories': [
        {'id': 'parent', 'text': '', 'valid_to': 10},
        {'id': 'child', 'text': 'The observatory has a silver dome.', 'valid_from': 10, 'supersedes': 'parent'},
    ]})
    assert response.status_code == 422
    assert (await client.get('/v1/export')).json() == []


@pytest.mark.asyncio
@pytest.mark.parametrize('scope', ['alpha', 'all'])
@pytest.mark.parametrize('floor', [0.0, 0.5])
async def test_rejected_automatic_recall_does_not_reinforce_memories(client, monkeypatch, scope, floor):
    import agent_memory.api as api
    stored = (await client.post('/v1/store', json={
        'agent': 'alpha', 'text': 'The observatory has a violet dome.',
    })).json()['id']
    storage = api._storage_pool.get('alpha')
    before = storage.get_memory(stored)

    async def unavailable(text):
        raise ConnectionError('synthetic outage')

    monkeypatch.setattr(api._embedder, 'embed', unavailable)
    query = {'agent': scope, 'query': 'violet observatory dome'}
    for _ in range(2):
        response = (await client.post('/v1/recall', json={**query, 'min_semantic_score': floor})).json()
        assert response['results'] == []
    after = storage.get_memory(stored)
    assert after['strength'] == before['strength']
    assert after['last_accessed_at'] == before['last_accessed_at']
    admitted = (await client.post('/v1/recall', json=query)).json()['results']
    assert [r['id'] for r in admitted] == [stored]
    assert storage.get_memory(stored)['strength'] > before['strength']


@pytest.mark.asyncio
async def test_rule_endpoint_preserves_evidence_and_source_session(client):
    payload = {'agent': 'alpha', 'text': 'Always choose quiet observatory visits.',
               'session_id': 'session-a', 'evidence': {'event_id': 'rule-a', 'assertion': 'reported'}}
    first = (await client.post('/v1/rule', json=payload)).json()
    second = (await client.post('/v1/rule', json={**payload, 'evidence': {'event_id': 'rule-b'}})).json()
    assert first['id'] != second['id']
    exported = (await client.get('/v1/export', params={'agent': 'alpha'})).json()
    assert {m['evidence']['event_id'] for m in exported} == {'rule-a', 'rule-b'}
    assert all(m['source_session'] == 'session-a' for m in exported)
    rejected = await client.post('/v1/rule', json={**payload, 'supersedes': first['id']})
    assert rejected.status_code == 422


@pytest.mark.asyncio
async def test_forgetting_by_old_only_text_deletes_revision_family(client):
    old = (await client.post('/v1/store', json={
        'agent': 'alpha', 'text': 'I prefer violet domes.',
    })).json()['id']
    await client.post('/v1/store', json={
        'agent': 'alpha', 'text': 'I now prefer silver roofs.', 'supersedes': old,
    })
    other = (await client.post('/v1/store', json={
        'agent': 'beta', 'text': 'I prefer violet domes.',
    })).json()['id']
    response = await client.request('DELETE', '/v1/forget', json={'agent': 'alpha', 'query': 'violet'})
    assert response.json()['deleted']
    assert (await client.get('/v1/export', params={'agent': 'alpha'})).json() == []
    assert [m['id'] for m in (await client.get('/v1/export', params={'agent': 'beta'})).json()] == [other]


@pytest.mark.asyncio
async def test_overlapping_recall_does_not_discard_a_healthy_semantic_result(client, monkeypatch):
    import asyncio
    import agent_memory.api as api
    started, release = asyncio.Event(), asyncio.Event()
    storage = api._storage_pool.get('alpha')
    storage.store_memory('The observatory has a violet dome.', vector=[1, 0, 0, 0])

    async def embedding(text):
        if 'healthy' in text:
            started.set()
            await release.wait()
            return [1, 0, 0, 0]
        raise ConnectionError('synthetic outage')

    monkeypatch.setattr(api._embedder, 'embed', embedding)
    query = {'agent': 'alpha', 'min_semantic_score': 0.0}
    healthy = asyncio.create_task(client.post('/v1/recall', json={**query, 'query': 'healthy observatory'}))
    try:
        await asyncio.wait_for(started.wait(), 2)
        failed = (await client.post('/v1/recall', json={**query, 'query': 'outage observatory'})).json()
    finally:
        release.set()
    success = (await asyncio.wait_for(healthy, 2)).json()
    assert failed['degraded'] and failed['results'] == []
    assert not success.get('degraded', False)
    assert success['results']


@pytest.mark.asyncio
async def test_overlapping_healthy_recall_does_not_admit_or_cache_an_outage(client, monkeypatch):
    import asyncio
    import agent_memory.api as api
    started, release = asyncio.Event(), asyncio.Event()
    storage = api._storage_pool.get('alpha')
    for text in ('The observatory has a violet dome.', 'The observatory opens at dusk.'):
        storage.store_memory(text, vector=[1, 0, 0, 0])

    async def embedding(text):
        if 'outage' in text:
            raise ConnectionError('synthetic outage')
        return [1, 0, 0, 0]

    class Reranker:
        top_k = 10
        def score(self, query, docs, ids):
            return [.9] * len(docs)

    async def run_score(func, query, *args):
        if 'outage' in query:
            started.set()
            await release.wait()
        return func(query, *args)

    monkeypatch.setattr(api._embedder, 'embed', embedding)
    monkeypatch.setattr(api, '_reranker', Reranker())
    monkeypatch.setattr(asyncio, 'to_thread', run_score)
    query = {'agent': 'alpha', 'min_semantic_score': 0.0}
    outage = asyncio.create_task(client.post('/v1/recall', json={**query, 'query': 'outage observatory'}))
    try:
        await asyncio.wait_for(started.wait(), 2)
        healthy = (await client.post('/v1/recall', json={**query, 'query': 'healthy observatory'})).json()
    finally:
        release.set()
    failed = (await asyncio.wait_for(outage, 2)).json()
    assert healthy['results'] and not healthy.get('degraded', False)
    assert failed['degraded'] and failed['results'] == []
    assert not any('outage' in r['query_norm'] for r in storage._get_conn().execute('SELECT query_norm FROM search_result_cache'))


@pytest.mark.asyncio
async def test_absent_embedder_is_degraded_and_never_admitted_or_cached(client, monkeypatch):
    import agent_memory.api as api
    storage = api._storage_pool.get('alpha')
    storage.store_memory('The observatory has a violet dome.')
    monkeypatch.setattr(api, '_embedder', None)
    query = {'agent': 'alpha', 'query': 'violet observatory dome'}
    for _ in range(2):
        response = (await client.post('/v1/recall', json={**query, 'min_semantic_score': 0})).json()
        assert response['degraded'] and response['search_mode'] == 'keyword_only'
        assert response['results'] == []
    assert storage._get_conn().execute('SELECT COUNT(*) FROM search_result_cache').fetchone()[0] == 0
    assert (await client.post('/v1/recall', json=query)).json()['results']


@pytest.mark.asyncio
@pytest.mark.parametrize('existing_child', [False, True])
async def test_import_rejects_branching_revision_families(client, existing_child):
    parent = {'id': 'parent', 'text': 'The observatory had a violet dome.', 'valid_to': 10}
    child = {'id': 'child', 'text': 'The observatory has a silver dome.', 'valid_from': 10, 'supersedes': 'parent'}
    fork = {**child, 'id': 'fork', 'text': 'The observatory has a blue dome.'}
    if existing_child:
        assert (await client.post('/v1/import', json={'memories': [parent, child]})).status_code == 200
        records = [fork]
    else:
        records = [parent, child, fork]
    response = await client.post('/v1/import', json={'memories': records})
    assert response.status_code == 422
    after = (await client.get('/v1/export')).json()
    assert {m['id'] for m in after} == ({'parent', 'child'} if existing_child else set())


@pytest.mark.asyncio
async def test_import_rolls_back_all_rows_and_vectors_when_a_later_record_is_invalid(client):
    import agent_memory.api as api
    response = await client.post('/v1/import', json={'memories': [
        {'text': 'The observatory has a violet dome.'},
        {'text': 'The observatory opens at dusk.', 'importance': 'invalid'},
    ]})
    assert response.status_code == 422
    storage = api._storage_pool.get('main')
    assert (await client.get('/v1/export')).json() == []
    assert storage._get_conn().execute('SELECT COUNT(*) FROM memory_vectors').fetchone()[0] == 0
    assert storage.search_text('observatory', include_history=True) == []


@pytest.mark.asyncio
async def test_import_rechecks_retained_lineage_after_awaiting_embeddings(client, monkeypatch):
    import asyncio
    import agent_memory.api as api
    started, release = asyncio.Event(), asyncio.Event()
    parent = {'id': 'parent', 'text': 'The observatory had a violet dome.', 'valid_to': 10}
    assert (await client.post('/v1/import', json={'memories': [parent]})).status_code == 200
    storage = api._storage_pool.get('main')

    async def embed(text):
        started.set()
        await release.wait()
        return [1, 0, 0, 0]

    monkeypatch.setattr(api._embedder, 'embed', embed)
    task = asyncio.create_task(client.post('/v1/import', json={'memories': [
        {'id': 'child', 'text': 'The observatory has a silver dome.', 'valid_from': 10, 'supersedes': 'parent'},
    ]}))
    try:
        await asyncio.wait_for(started.wait(), 2)
        storage.store_memory('The observatory has a blue dome.', memory_id='winner',
                             valid_from=10, supersedes='parent')
    finally:
        release.set()
    assert (await asyncio.wait_for(task, 2)).status_code == 422
    assert storage.get_memory('child') is None
    assert storage.get_memory('winner') is not None


@pytest.mark.asyncio
@pytest.mark.parametrize('replacement', [{'valid_to': 20}, {'namespace': 'other'}, {'text': 'The observatory had a blue dome.'}])
async def test_import_parent_replacement_preserves_retained_child_edges(client, replacement):
    parent = {'id': 'parent', 'text': 'The observatory had a violet dome.', 'valid_to': 10}
    child = {'id': 'child', 'text': 'The observatory has a silver dome.', 'valid_from': 10, 'supersedes': 'parent'}
    assert (await client.post('/v1/import', json={'memories': [parent, child]})).status_code == 200
    response = await client.post('/v1/import', json={'skip_duplicates': False, 'memories': [{**parent, **replacement}]})
    assert response.status_code == (200 if 'text' in replacement else 422)
    rows = {m['id']: m for m in (await client.get('/v1/export')).json()}
    assert rows['parent']['valid_to'] == rows['child']['valid_from'] == 10
    assert rows['parent']['namespace'] == rows['child']['namespace'] == 'default'


@pytest.mark.asyncio
async def test_history_excludes_scheduled_future_but_forget_can_find_it(client):
    import time
    old = (await client.post('/v1/store', json={'text': 'The observatory dome is violet.'})).json()['id']
    future = time.time() + 86400
    new = (await client.post('/v1/store', json={
        'text': 'The observatory roof will become silver.', 'supersedes': old, 'valid_from': future,
    })).json()['id']
    history = (await client.post('/v1/recall', json={'query': 'previously observatory'})).json()['results']
    assert [r['id'] for r in history] == [old]
    scheduled = (await client.post('/v1/recall', json={'query': 'observatory', 'as_of': future + 1})).json()['results']
    assert [r['id'] for r in scheduled] == [new]
    assert (await client.request('DELETE', '/v1/forget', json={'query': 'silver'})).json()['deleted']
    assert (await client.get('/v1/export')).json() == []


@pytest.mark.asyncio
@pytest.mark.parametrize('scope', ['alpha', 'all'])
async def test_as_of_uses_validity_instead_of_ingestion_date_words(client, scope):
    import time
    now = time.time()
    old = (await client.post('/v1/store', json={
        'agent': 'alpha', 'text': 'The observatory had a violet dome.',
    })).json()['id']
    new = (await client.post('/v1/store', json={
        'agent': 'alpha', 'text': 'The observatory has a silver dome.',
        'supersedes': old, 'valid_from': now - 2 * 86400,
    })).json()['id']
    result = (await client.post('/v1/recall', json={
        'agent': scope, 'query': 'yesterday observatory dome', 'as_of': now - 86400,
    })).json()
    assert [r['id'] for r in result['results']] == [new]
    assert 'time_range' not in result


@pytest.mark.asyncio
async def test_all_search_lanes_share_one_validity_time_across_embedding_await(tmp_storage, monkeypatch):
    import asyncio
    import agent_memory.search as search_module
    from types import SimpleNamespace
    from agent_memory.search import HybridSearch
    clock = {'now': 100.0}
    # Replace each module's time reference, not the process/event-loop clock.
    fake_time = SimpleNamespace(time=lambda: clock['now'], perf_counter=__import__('time').perf_counter)
    monkeypatch.setattr(search_module, 'time', fake_time)
    monkeypatch.setattr('agent_memory.storage.time', fake_time)
    old = tmp_storage.store_memory('The observatory had a violet dome.', vector=[1, 0, 0, 0])
    new = tmp_storage.revise_memory(old, text='The observatory has a silver dome.',
                                    vector=[1, 0, 0, 0], valid_from=101)['id']
    keyword_finished = asyncio.Event()
    original = tmp_storage.search_text

    def keyword(*args, **kwargs):
        result = original(*args, **kwargs)
        keyword_finished.set()
        return result

    class Embedder:
        async def embed(self, text):
            await keyword_finished.wait()
            clock['now'] = 102.0
            return [1, 0, 0, 0]

    monkeypatch.setattr(tmp_storage, 'search_text', keyword)
    results = await HybridSearch(tmp_storage, Embedder()).search('observatory dome', rerank=False)
    assert [r.id for r in results] == [old]
    assert new not in {r.id for r in results}
