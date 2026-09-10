"""Source replay, transactions and derived deletion using only synthetic data."""
import asyncio
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

import agent_memory.api as api
from agent_memory.storage import MemoryStorage, ForgottenSourceError
from agent_memory.entities import KnowledgeGraph
from tests.test_api import _init_api_state  # noqa: F401
from httpx import AsyncClient, ASGITransport

TEXT = 'Mira Sol uses Python and SQLite at the violet observatory.'


@pytest.fixture
async def client(monkeypatch):
    monkeypatch.setattr(api.app, "middleware_stack", api.app.build_middleware_stack())
    async with AsyncClient(transport=ASGITransport(app=api.app), base_url="http://test") as value:
        yield value


@pytest.mark.asyncio
async def test_capture_forget_replay_relearn_and_agent_isolation(client):
    payload = {'agent': 'alpha', 'messages': [{'role': 'user', 'session': 'source-a', 'text': TEXT}]}
    assert (await client.post('/v1/capture', json=payload)).json()['stored'] == 1
    rows = (await client.get('/v1/export', params={'agent': 'alpha'})).json()
    storage = api._storage_pool.get('alpha')
    conn = storage._get_conn()
    assert conn.execute('SELECT count(*) FROM graph_sources').fetchone()[0] > 0
    storage.cache_search_result('observatory', 5, 0.0, 'alpha', json.dumps(rows), 300)
    storage.cache_embedding('synthetic-key', b'synthetic-vector')
    receipt = (await client.request('DELETE', '/v1/forget', json={'agent': 'alpha', 'id': rows[0]['id']})).json()
    assert receipt['deleted'] and receipt['unidentified_records'] == 0
    assert receipt['source_keys'] == [storage.source_key('source-a')]
    for table in ('memories', 'memory_fts', 'memory_vectors', 'graph_sources', 'entities', 'relationships', 'temporal_facts', 'search_result_cache', 'embedding_cache'):
        assert conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0] == 0, table
    for _ in range(2):
        result = (await client.post('/v1/capture', json=payload)).json()
        assert result['stored'] == 0 and result['blocked'] == 1
    assert (await client.post('/v1/recall', json={'agent': 'alpha', 'query': 'violet observatory'})).json()['results'] == []
    # Namespace and producer labels cannot disguise the same source.
    for endpoint in ('/v1/store', '/v1/rule'):
        result = await client.post(endpoint, json={'agent': 'alpha', 'text': TEXT, 'session_id': 'source-a', 'namespace': 'other', 'source': 'new-label'})
        assert result.status_code == 409
    assert (await client.post('/v1/import', json={'agent': 'alpha', 'memories': rows})).status_code == 409
    assert (await client.post('/v1/capture', json={**payload, 'agent': 'beta'})).json()['stored'] == 1
    assert (await client.post('/v1/relearn-source', json={'agent': 'beta', 'source_key': receipt['source_keys'][0], 'confirm': True})).json()['cleared'] is False
    assert storage.source_is_forgotten('source-a')
    fresh = {**payload, 'messages': [{**payload['messages'][0], 'session': 'independent-b'}]}
    assert (await client.post('/v1/capture', json=fresh)).json()['stored'] == 1
    assert (await client.post('/v1/relearn-source', json={'agent': 'alpha', 'session_id': 'source-a'})).status_code == 422
    assert (await client.post('/v1/relearn-source', json={'agent': 'all', 'session_id': 'source-a', 'confirm': True})).status_code == 400
    assert (await client.post('/v1/relearn-source', json={'agent': 'alpha', 'source_key': receipt['source_keys'][0], 'confirm': True})).json() == {'cleared': True, 'restored': False}
    assert len((await client.get('/v1/export', params={'agent': 'alpha'})).json()) == 1
    assert (await client.post('/v1/capture', json=payload)).json()['stored'] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('endpoint', ['/v1/capture', '/v1/store', '/v1/import'])
async def test_forget_wins_against_embedding_in_flight(client, endpoint):
    old = (await client.post('/v1/store', json={'text': TEXT, 'session_id': 'source-a'})).json()['id']
    entered, release = asyncio.Event(), asyncio.Event()

    class PausedEmbedder:
        async def embed(self, text):
            entered.set()
            await release.wait()
            return [0., 0., 0., 0.]

        async def embed_batch(self, texts):
            return [await self.embed(t) for t in texts]

    api._embedder = PausedEmbedder()
    api._config.embed_worker_enabled = False
    if endpoint == '/v1/capture':
        body = {'messages': [{'text': TEXT + ' A revised extraction.', 'role': 'user', 'session': 'source-a'}]}
    elif endpoint == '/v1/store':
        body = {'text': TEXT + ' A revised extraction.', 'session_id': 'source-a'}
    else:
        body = {'memories': [{'id': 'new-id', 'text': TEXT, 'source_session': 'source-a'}]}
    pending = asyncio.create_task(client.post(endpoint, json=body))
    await asyncio.wait_for(entered.wait(), 2)
    assert (await client.request('DELETE', '/v1/forget', json={'id': old})).json()['deleted']
    release.set()
    response = await pending
    if endpoint == '/v1/capture':
        assert response.json()['blocked'] == 1 and response.json()['stored'] == 0
    else:
        assert response.status_code == 409
    assert (await client.get('/v1/export')).json() == []
    assert api._storage_pool.get('main')._get_conn().execute('SELECT count(*) FROM graph_sources').fetchone()[0] == 0


def test_persistent_marker_migration_atomic_batch_and_missing_identity(tmp_path):
    path = str(tmp_path / 'synthetic.sqlite')
    storage = MemoryStorage(path, dimensions=4)
    first = storage.store_memory(TEXT, source_session='source-a')
    second = storage.revise_memory(first, text='Mira Sol now uses Rust.', vector=None, valid_from=1700000000, source_session='source-b')['id']
    assert storage.delete_memory(second)
    expected = {storage.source_key('source-a'), storage.source_key('source-b')}
    conn = storage._get_conn()
    assert {r[0] for r in conn.execute('SELECT * FROM forgotten_sources')} == expected
    assert [r[1] for r in conn.execute('PRAGMA table_info(forgotten_sources)')] == ['source_key']
    assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    backup_path = str(tmp_path / 'recovery.sqlite')
    backup = sqlite3.connect(backup_path)
    conn.backup(backup)
    backup.close()
    recovered = MemoryStorage(backup_path, dimensions=4)
    assert recovered.source_is_forgotten('source-a')
    assert recovered.source_is_forgotten('source-b')
    recovered.close()
    storage.close()
    storage = MemoryStorage(path, dimensions=4)
    with pytest.raises(ForgottenSourceError):
        storage.store_memories_batch([{'text': 'Independent row must roll back.', 'source_session': 'fresh'}, {'text': TEXT, 'source_session': 'source-a'}])
    assert storage.stats()['total_memories'] == 0
    # Old rows without source identity receive no invented content fingerprint.
    legacy = storage.store_memory(TEXT, memory_id='legacy-content-derived-id')
    storage.delete_memory(legacy)
    assert {r[0] for r in storage._get_conn().execute('SELECT * FROM forgotten_sources')} == expected
    assert storage.store_memory(TEXT)
    assert storage.relearn_source('source-b')
    assert storage.source_is_forgotten('source-a')
    assert not storage.source_is_forgotten('source-b')
    storage.close()
    # Existing schemas acquire the empty additive table without inventing history.
    legacy_path = str(tmp_path / 'legacy.sqlite')
    db = sqlite3.connect(legacy_path)
    db.execute('CREATE TABLE unrelated(value TEXT)')
    db.execute("INSERT INTO unrelated VALUES ('preserved')")
    db.commit()
    db.close()
    migrated = MemoryStorage(legacy_path, dimensions=4)
    assert migrated._get_conn().execute('SELECT value FROM unrelated').fetchone()[0] == 'preserved'
    assert migrated._get_conn().execute('SELECT count(*) FROM forgotten_sources').fetchone()[0] == 0
    migrated.close()


def test_concurrent_capture_and_forget_serialize(tmp_path):
    path = str(tmp_path / 'race.sqlite')
    storage = MemoryStorage(path, dimensions=4)
    mid = storage.store_memory(TEXT, source_session='source-a', category='user')
    storage.close()
    gate = Barrier(2)

    def capture():
        db = MemoryStorage(path, dimensions=4)
        gate.wait()
        try:
            return db.merge_or_store(TEXT, None, 'user', .5, 'source-a')['action']
        except ForgottenSourceError:
            return 'blocked'
        finally:
            db.close()

    def forget():
        db = MemoryStorage(path, dimensions=4)
        gate.wait()
        try:
            return db.delete_memory(mid)
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        capture_result, forget_result = executor.submit(capture), executor.submit(forget)
        assert capture_result.result() in {'merged', 'blocked'}
        assert forget_result.result()
    db = MemoryStorage(path, dimensions=4)
    assert db.get_memory(mid) is None
    with pytest.raises(ForgottenSourceError):
        db.store_memory(TEXT, source_session='source-a')
    with pytest.raises(ValueError, match='no longer exists'):
        KnowledgeGraph(db).process_text(TEXT, source_memory_id=mid)
    assert db._get_conn().execute('SELECT count(*) FROM graph_sources').fetchone()[0] == 0
    db.close()


def test_shared_graph_support_and_independent_source_survive(tmp_path):
    db = MemoryStorage(str(tmp_path / 'graph.sqlite'), dimensions=4)
    kg = KnowledgeGraph(db)
    a = db.store_memory(TEXT, source_session='source-a')
    b = db.store_memory('Python and SQLite support a new independent project.', source_session='source-b')
    kg.process_text(TEXT, source_memory_id=a)
    kg.process_text('Python and SQLite support a new independent project.', source_memory_id=b)
    assert db.delete_memory(a)
    conn = db._get_conn()
    assert db.get_memory(b)
    assert conn.execute('SELECT count(*) FROM graph_sources WHERE memory_id = ?', (b,)).fetchone()[0] > 0
    assert all('violet' not in row[0] for row in conn.execute('SELECT context FROM relationships'))
    assert db.delete_memory(b)
    assert conn.execute('SELECT count(*) FROM entities').fetchone()[0] == 0
    assert conn.execute('SELECT count(*) FROM graph_sources').fetchone()[0] == 0
    db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('batch', [False, True])
async def test_embedding_cache_fences_pre_forget_requests(tmp_path, batch):
    from threading import Event
    from agent_memory.embeddings import OpenRouterEmbeddings
    embedder = OpenRouterEmbeddings.__new__(OpenRouterEmbeddings)
    embedder._cache_generation = 0
    embedder._cache, embedder._cache_order, embedder._cache_size = {}, [], 10
    embedder._storage = MemoryStorage(str(tmp_path / 'cache.sqlite'), dimensions=4)
    entered, release = Event(), Event()

    def synthetic_embedding(texts):
        entered.set()
        assert release.wait(5)
        return [[0., 0., 0., 0.] for _ in texts]

    embedder._call_api = synthetic_embedding
    pending = asyncio.create_task(embedder.embed_batch([TEXT]) if batch else embedder.embed(TEXT))
    assert await asyncio.to_thread(entered.wait, 2)
    embedder.clear_cache()
    release.set()
    await pending
    assert not embedder._cache and not embedder._cache_order
    assert embedder._storage._get_conn().execute('SELECT count(*) FROM embedding_cache').fetchone()[0] == 0
    embedder._storage.close()


def test_late_embedding_worker_does_not_restore_deleted_vector(tmp_path):
    from agent_memory.embed_worker import EmbedWorker
    db = MemoryStorage(str(tmp_path / 'worker.sqlite'), dimensions=4)
    mid = db.store_memory(TEXT, source_session='source-a')
    db.delete_memory(mid)
    worker = EmbedWorker.__new__(EmbedWorker)
    assert worker._update_memory_vector(db, mid, [0., 0., 0., 0.]) is False
    assert db._get_conn().execute('SELECT count(*) FROM memory_vectors').fetchone()[0] == 0
    db.close()


def test_manual_graph_support_is_not_deleted_as_derived(tmp_path):
    db = MemoryStorage(str(tmp_path / 'manual.sqlite'), dimensions=4)
    mid = db.store_memory(TEXT, source_session='source-a')
    KnowledgeGraph(db).process_text(TEXT, source_memory_id=mid)
    entity = db.search_entities('Python')[0]
    db.store_entity(entity['name'], entity['type'])
    db.delete_memory(mid)
    assert db.get_entity(entity['id']) is not None
    db.close()


def test_replacement_does_not_leave_vector_or_fts_copy_after_forget(tmp_path):
    db = MemoryStorage(str(tmp_path / 'replace.sqlite'), dimensions=4)
    db.store_memory('The synthetic telescope is violet.', vector=[1., 0., 0., 0.], memory_id='replace', source_session='source-a')
    db.store_memory('The synthetic telescope is amber.', vector=[0., 1., 0., 0.], memory_id='replace', source_session='source-a')
    assert db._get_conn().execute('SELECT count(*) FROM memory_vectors').fetchone()[0] == 1
    assert db.search_text('violet') == []
    assert db.delete_memory('replace')
    for table in ('memories', 'memory_vectors', 'memory_fts'):
        assert db._get_conn().execute(f'SELECT count(*) FROM {table}').fetchone()[0] == 0
    db.close()
