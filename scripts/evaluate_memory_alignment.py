"""Replay one public synthetic embedding snapshot against a chosen source tree.

No network, credentials, model download or production database access. These are
retrieval/injection measurements, NOT generated-answer accuracy or full host latency.
"""

import argparse
import asyncio
import gzip
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path


async def evaluate(repo, fixtures, repetitions):
    sys.path.insert(0, str(repo.resolve()))
    from agent_memory.pool import StoragePool
    from agent_memory.search import HybridSearch, normalize_query

    corpus = json.loads((fixtures / 'alignment_cases.json').read_text())
    snapshot = json.loads(gzip.decompress((fixtures / 'alignment_vectors.json.gz').read_bytes()))
    vectors = {normalize_query(text): vector for text, vector in zip(snapshot['texts'], snapshot['vectors'])}

    class RecordedEmbedder:
        calls = 0

        async def embed(self, text):
            self.calls += 1
            return vectors[normalize_query(text)]

    with tempfile.TemporaryDirectory(prefix='noldomem-eval-') as scratch:
        pool = StoragePool(scratch, dimensions=len(snapshot['vectors'][0]))
        storage = pool.get('alpha')
        embedder = RecordedEmbedder()
        ids = {}
        for episode in corpus['episodes']:
            result = storage.merge_or_store(
                text=episode['text'], vector=vectors[normalize_query(episode['text'])],
                category='user', importance=0.6, source_session='episode-' + episode['id'],
            )
            ids[episode['id']] = result['id']
        pool.get('beta').store_memory('The synthetic beta observatory has an orange dome.',
                                      vector=snapshot['vectors'][0])
        search = HybridSearch(storage, embedder)
        rows = []
        related_results = []
        negative_results = []
        for episode in corpus['episodes']:
            results = await search.search(episode['query'], limit=5, agent='alpha')
            related_results.append((ids[episode['id']], results))
            rows.append({'case': episode['id'], 'expected_retrieved': ids[episode['id']] in [r.id for r in results],
                         'distinct_evidence_preserved': any(r.text == episode['text'] for r in results),
                         'top_semantic': round(results[0].semantic_score, 4) if results else None,
                         'context_chars': sum(len(r.text) for r in results),
                         'cross_agent_leak': any('synthetic beta' in r.text for r in results)})
        negatives = []
        for query in corpus['unrelated']:
            results = await search.search(query, limit=5, agent='alpha')
            negative_results.append(results)
            negatives.append({'query': query, 'count': len(results),
                              'max_semantic': round(max((r.semantic_score for r in results), default=0), 4)})
        cold, warm = [], []
        before = embedder.calls
        for _ in range(repetitions):
            query = corpus['episodes'][0]['query']
            storage.invalidate_search_cache('alpha')
            start = time.perf_counter()
            await search.search(query, limit=5, agent='alpha')
            cold.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            await search.search(query, limit=5, agent='alpha')
            warm.append((time.perf_counter() - start) * 1000)

        def summary(samples):
            result = {'n': len(samples), 'median_ms': round(statistics.median(samples), 3)}
            if len(samples) >= 100:
                result['sample_p95_ms'] = round(sorted(samples)[int(.95 * len(samples)) - 1], 3)
            return result

        # Storage-size stress only: repeat the same encoded public text, with
        # distinct synthetic sessions. This is not a larger semantic benchmark.
        volume = []
        stress_text = corpus['episodes'][0]['text']
        for target in [1000, 10000]:
            stress = pool.get('stress' + str(target))
            stress.store_memories_batch([{'text': stress_text, 'vector': vectors[normalize_query(stress_text)],
                                         'source_session': 'synthetic-' + str(i)} for i in range(target)])
            stress_search = HybridSearch(stress, embedder)
            samples = []
            for _ in range(30):
                stress.invalidate_search_cache('stress' + str(target))
                start = time.perf_counter()
                await stress_search.search(corpus['episodes'][0]['query'], limit=5, agent='stress' + str(target))
                samples.append((time.perf_counter() - start) * 1000)
            volume.append({'rows': target, 'distribution': 'identical text/vector stress', **summary(samples)})

        admission = []
        for floor in [.45, .50, .55]:
            admission.append({
                'min_semantic_score': floor,
                'related_expected_admitted': sum(any(r.id == expected and r.semantic_score >= floor for r in results)
                                                 for expected, results in related_results),
                'unrelated_queries_with_context': sum(any(r.semantic_score >= floor for r in results)
                                                      for results in negative_results),
                'related_context_chars': sum(sum(len(r.text) for r in results if r.semantic_score >= floor)
                                             for _, results in related_results),
            })
        output = {'model_label': snapshot['model_label'], 'embedding_snapshot_count': len(snapshot['texts']),
                  'live_embedding_measurement': summary(snapshot['latency_ms']),
                  'volume_stress_offline': volume,
                  'stored_rows': storage.stats()['total_memories'], 'cases': rows, 'unrelated': negatives,
                  'retrieval_cold_offline': summary(cold), 'retrieval_warm_offline': summary(warm),
                  'timing_replay_embedding_calls': embedder.calls - before,
                  'native_curated_input_chars': sum(len(e['text']) for e in corpus['episodes']),
                  'admission_calibration_same_small_dataset': admission,
                  'generated_answer_accuracy': 'not measured'}
        pool.close_all()
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[1] / 'tests/fixtures')
    parser.add_argument('--repetitions', type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 1000:
        parser.error('repetitions must be between 1 and 1000')
    print(json.dumps(asyncio.run(evaluate(args.repo, args.fixtures, args.repetitions)), indent=2))
