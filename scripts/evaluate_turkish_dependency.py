"""Compare real capture/index/recall with installed or import-blocked NLP packages.

Uses recorded real embeddings of independently synthetic text, temporary databases
and the ASGI API. Does not measure generated answers or call a model/service.
"""
import argparse
import asyncio
import gzip
import importlib.abc
import json
import sys
import tempfile
import time
from pathlib import Path


async def evaluate(repo, fixtures, block):
    attempted = []
    class BlockNLP(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'zeyrek', 'nltk'}:
                attempted.append(fullname)
                raise ModuleNotFoundError(fullname)
    if block:
        sys.meta_path.insert(0, BlockNLP())
    sys.path.insert(0, str(repo))
    from httpx import ASGITransport, AsyncClient
    import agent_memory.api as api
    from agent_memory.config import Config
    from agent_memory.pool import StoragePool
    from agent_memory.search import SearchWeights, normalize_query

    corpus = json.loads((fixtures / 'turkish_alignment_cases.json').read_text())
    snapshot = json.loads(gzip.decompress((fixtures / 'turkish_alignment_vectors.json.gz').read_bytes()))
    vectors = {normalize_query(t): v for t, v in zip(snapshot['texts'], snapshot['vectors'])}
    class RecordedEmbedder:
        calls = 0
        async def embed(self, text):
            self.calls += 1
            return vectors[normalize_query(text)]
        async def embed_batch(self, texts):
            return [await self.embed(text) for text in texts]
        def set_storage(self, storage):
            pass
    with tempfile.TemporaryDirectory(prefix='noldomem-tr-') as scratch:
        api._storage_pool = StoragePool(scratch, dimensions=len(snapshot['vectors'][0]))
        api._config = Config(api_key='', embed_worker_enabled=False)
        api._embedder = RecordedEmbedder()
        api._search_cache, api._kg_cache = {}, {}
        api._search_weights = SearchWeights()
        api._reranker = api._bg_reranker = None
        api._start_time = time.time()
        try:
            async with AsyncClient(transport=ASGITransport(app=api.app), base_url='http://synthetic') as client:
                for agent, episodes in [('alpha', corpus['episodes']), ('beta', [{'id': 'other', 'text': corpus['other_agent']}])]:
                    response = await client.post('/v1/capture', json={'agent': agent, 'messages': [
                        {'role': 'user', 'text': row['text'], 'session': 'session-a', 'evidence': {'event_id': row['id']}}
                        for row in episodes]})
                    response.raise_for_status()
                exported = (await client.get('/v1/export', params={'agent': 'alpha'})).json()
                identities = {m['id']: m['evidence']['event_id'] for m in exported}
                rows = []
                for episode in corpus['episodes']:
                    for query in episode['queries']:
                        response = await client.post('/v1/recall', json={'agent': 'alpha', 'query': query, 'limit': 5})
                        response.raise_for_status()
                        results = response.json()['results']
                        rows.append({'expected': episode['id'], 'query': query,
                                     'ranked': [identities[r['id']] for r in results],
                                     'scores': [round(r['semantic_score'], 6) for r in results],
                                     'source_sessions': sorted({r['source_session'] for r in results})})
                negatives = []
                for query in corpus['unrelated']:
                    response = await client.post('/v1/recall', json={'agent': 'alpha', 'query': query,
                                                                  'min_semantic_score': .5})
                    response.raise_for_status()
                    negatives.append({'query': query, 'count': len(response.json()['results'])})
                return {'blocked_nlp': block, 'nlp_import_attempts': attempted,
                        'loaded_nlp': sorted({n.split('.')[0] for n in sys.modules if n.split('.')[0] in {'zeyrek', 'nltk'}}),
                        'captured': len(exported), 'cases': rows, 'unrelated_at_floor_0_5': negatives,
                        'embedding_calls': api._embedder.calls, 'cross_agent_leaks': 0,
                        'generated_answers': 'not measured'}
        finally:
            api._storage_pool.close_all()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--block-nlp', action='store_true')
    args = parser.parse_args()
    fixtures = Path(__file__).resolve().parents[1] / 'tests/fixtures'
    print(json.dumps(asyncio.run(evaluate(args.repo.resolve(), fixtures, args.block_nlp)), ensure_ascii=False, indent=2))
