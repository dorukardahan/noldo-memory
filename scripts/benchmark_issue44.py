"""Disposable issue44 ASGI/SQLite performance fixture (NOT production profiling).

Run through isolated_run.py; produces payload-free sample CSV and summary JSON.
Provider vectors are explicitly synthetic gate controls, not relevance evidence.
Default 60 samples/path/run across two accepted baselines plus concurrency 4/8
upper-envelope runs gives >=100 successful samples/path without counting overload
responses as accepted latency.
Each path/run has a fresh limiter observation window, below its 120-write cap.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from httpx import AsyncClient, ASGITransport
from agent_memory import api
from agent_memory.config import Config
from agent_memory.index_worker import IndexWorker
from agent_memory.pool import StoragePool
from agent_memory.search import SearchWeights


def utc():
    return datetime.now(timezone.utc).isoformat()


def fixed_error_code(status):
    return {
        413: 'payload_too_large',
        422: 'validation_error',
        429: 'overload',
        503: 'storage_busy',
        504: 'deadline_exceeded',
    }.get(status, 'request_failed')


def stats(rows):
    values = sorted(r['elapsed_seconds'] for r in rows)
    successes = [r for r in rows if r['http_status'] == 200]
    accepted_values = sorted(r['elapsed_seconds'] for r in successes)

    def percentile(series, p):
        return series[max(0, math.ceil(len(series) * p) - 1)] if series else None

    return dict(
        samples=len(rows),
        successes=len(successes),
        errors=len(rows) - len(successes),
        p50=percentile(values, .50),
        p95=percentile(values, .95),
        p99=percentile(values, .99),
        max=max(values, default=None),
        accepted_p50=percentile(accepted_values, .50),
        accepted_p95=percentile(accepted_values, .95),
        accepted_p99=percentile(accepted_values, .99),
        accepted_max=max(accepted_values, default=None),
        timeouts=sum(r['http_status'] == 504 for r in rows),
        overload=sum(r['http_status'] == 429 for r in rows),
        admitted_over_budget=sum(
            r['http_status'] == 200 and r['elapsed_seconds'] > r['budget_seconds'] for r in rows
        ),
        queue_depth_max=max((r['queue_depth'] for r in rows), default=0),
        index_inflight_max=max((r['index_inflight'] for r in rows), default=0),
        foreground_inflight_max=max((r['foreground_inflight'] for r in rows), default=0),
        provider_inflight_max=max((r['provider_inflight'] for r in rows), default=0),
    )


class GateEmbedder:
    """No external service, deterministic fake vector; real worker/search code."""
    def __init__(self):
        self.inflight = self.maximum = 0
    async def embed(self, text, **kwargs):
        self.inflight += 1
        self.maximum = max(self.maximum, self.inflight)
        try:
            await asyncio.sleep(0)
            return [1., 0., 0., 0.]
        finally:
            self.inflight -= 1
    async def probe(self, **kwargs):
        return await self.embed('Synthetic diagnostic', **kwargs)


async def measure_upper_envelope():
    """Exercise real admission caps without recording payloads or identifiers."""
    with tempfile.TemporaryDirectory(prefix='issue44-upper-') as directory:
        pool = StoragePool(directory, 4)
        embedder = GateEmbedder()
        api._storage_pool, api._embedder = pool, embedder
        api._config = Config(api_key='', openrouter_api_key='', embed_worker_enabled=False)
        api._search_cache, api._kg_cache = {}, {}
        api._search_weights = SearchWeights()
        api._reranker = api._bg_reranker = None
        api._start_time = time.time()
        api.app.middleware_stack = api.app.build_middleware_stack()
        roles = ('user', 'assistant', 'tool')
        long_messages = [
            {
                'role': roles[i % len(roles)],
                'text': f'Synthetic eligible row {i:02d} records a bounded multitool assertion. ' + 'x' * 3060,
                'session': f'synthetic-upper-{i:02d}',
            }
            for i in range(32)
        ]
        long_messages.extend([
            {'role': 'system', 'text': 'Excluded control plane instruction.'},
            {'role': 'tool', 'text': 'HEARTBEAT_OK'},
        ])
        control = [
            {
                'role': 'user',
                'text': f'Synthetic bounded control assertion {i:03d}.',
                'session': f'synthetic-control-{i:03d}',
            }
            for i in range(200)
        ]
        cases = {}

        async def post(name, messages):
            before = pool.get().stats()['total_memories']
            started = time.monotonic()
            response = await client.post('/v1/capture', json={
                'request_id': str(uuid.uuid4()),
                'messages': messages,
            })
            elapsed = time.monotonic() - started
            after = pool.get().stats()['total_memories']
            body = response.json()
            cases[name] = {
                'http_status': response.status_code,
                'elapsed_seconds': elapsed,
                'memory_count_before': before,
                'memory_count_after': after,
                'stored': body.get('stored'),
                'error_code': fixed_error_code(response.status_code) if response.status_code != 200 else '',
            }
            return response

        try:
            async with AsyncClient(transport=ASGITransport(app=api.app), base_url='http://synthetic') as client:
                long_result = await post('mixed_32_rows_about_100k_chars', long_messages)
                control_result = await post('control_200_messages', control)
                rejected = await post('reject_201_messages', control + [
                    {'role': 'user', 'text': 'This 201st row must be rejected.'}
                ])
                oversized = await post('reject_normalized_char_cap', [
                    {'role': 'user', 'text': f'Oversized assertion {i:02d}. ' + 'x' * 3900}
                    for i in range(40)
                ])
            cases['mixed_32_rows_about_100k_chars']['eligible_rows'] = 32
            cases['mixed_32_rows_about_100k_chars']['normalized_chars_approx'] = 100_000
            passed = (
                long_result.status_code == 200
                and cases['mixed_32_rows_about_100k_chars']['stored'] == 32
                and cases['mixed_32_rows_about_100k_chars']['elapsed_seconds'] <= 2
                and control_result.status_code == 200
                and cases['control_200_messages']['stored'] == 200
                and cases['control_200_messages']['elapsed_seconds'] <= 2
                and rejected.status_code == 422
                and cases['reject_201_messages']['memory_count_before']
                    == cases['reject_201_messages']['memory_count_after']
                and oversized.status_code == 413
                and cases['reject_normalized_char_cap']['memory_count_before']
                    == cases['reject_normalized_char_cap']['memory_count_after']
            )
            return {
                'measurement': 'Actual in-process ASGI request + SQLite admission runtime; no provider wait.',
                'payload_recorded': False,
                'cases': cases,
                'passed': passed,
            }
        finally:
            if pool._foreground:
                await pool._foreground.stop()
            pool.close_all()


async def measure(samples):
    output, drains = [], []
    started = utc()
    run_matrix = (('baseline-a', 1), ('baseline-b', 1), ('upper-c4', 4), ('upper-c8', 8))
    for run_name, concurrency in run_matrix:
        for operation, budget in [('capture', 2), ('store', 2), ('recall', 6), ('status', 1)]:
            with tempfile.TemporaryDirectory(prefix='issue44-bench-') as directory:
                pool = StoragePool(directory, 4)
                embedder = GateEmbedder()
                api._storage_pool, api._embedder = pool, embedder
                api._config = Config(api_key='', openrouter_api_key='', embed_worker_enabled=False)
                api._search_cache, api._kg_cache = {}, {}
                api._search_weights = SearchWeights()
                api._reranker = api._bg_reranker = None
                api._start_time = time.time()
                api.app.middleware_stack = api.app.build_middleware_stack()
                ids = []
                worker = IndexWorker(pool, embedder, api._config)
                async with AsyncClient(transport=ASGITransport(app=api.app), base_url='http://synthetic') as client:
                    if operation in {'status', 'recall'}:
                        seed = await client.post('/v1/store', json={'text': 'Synthetic Python SQLite benchmark assertion', 'request_id': str(uuid.uuid4())})
                        seed.raise_for_status()
                        ids.append(seed.json()['request_id'])
                        await worker.start()
                        until = time.monotonic() + 5
                        while pool.get()._get_conn().execute("SELECT count(*) FROM memory_index_jobs WHERE state IN ('pending','running')").fetchone()[0]:
                            if time.monotonic() > until:
                                raise TimeoutError('seed did not index')
                            await asyncio.sleep(.005)
                    else:
                        await worker.start()
                    semaphore = asyncio.Semaphore(concurrency)
                    peak_active = 0
                    async def sample(i):
                        nonlocal peak_active
                        async with semaphore:
                            identity = str(uuid.uuid4())
                            start = time.monotonic()
                            if operation == 'capture':
                                response = await client.post('/v1/capture', json={'request_id': identity, 'messages': [
                                    {'role': 'user', 'text': f'Synthetic Python SQLite capture assertion number {i}.', 'session': f'synthetic-{i}'}]})
                            elif operation == 'store':
                                response = await client.post('/v1/store', json={'request_id': identity,
                                    'text': f'Synthetic Python SQLite store assertion number {i}.', 'session_id': f'synthetic-{i}'})
                            elif operation == 'recall':
                                response = await client.post('/v1/recall', json={'query': 'Python SQLite', 'min_semantic_score': .5})
                            else:
                                response = await client.get('/v1/operations/' + ids[0], params={'operation': 'store'})
                            elapsed = time.monotonic() - start
                            conn = pool.get()._get_conn()
                            queue = conn.execute("SELECT count(*) FROM memory_index_jobs WHERE state='pending'").fetchone()[0]
                            running = conn.execute("SELECT count(*) FROM memory_index_jobs WHERE state='running'").fetchone()[0]
                            active = len(pool._foreground.active) if pool._foreground else 0
                            peak_active = max(peak_active, active)
                            response.json()
                            # No identity, text, session, source, hash, URL or path in samples.
                            output.append(dict(run=run_name, operation=operation, concurrency=concurrency, sample=i,
                                cold_warm='cold' if i < concurrency else 'warm', observed_utc=utc(),
                                elapsed_seconds=elapsed, budget_seconds=budget, http_status=response.status_code,
                                error_code=(fixed_error_code(response.status_code)
                                            if response.status_code != 200 else ''),
                                queue_depth=queue, index_inflight=running, foreground_inflight=active,
                                provider_inflight=embedder.inflight))
                    try:
                        await asyncio.gather(*(sample(i) for i in range(samples)))
                        drain_start = time.monotonic()
                        # Observe the default persisted 35s lease/reclamation;
                        # this is a measured drain window, NOT a raised HTTP cap.
                        until = drain_start + 45
                        drain_expired = False
                        while pool.get()._get_conn().execute("SELECT count(*) FROM memory_index_jobs WHERE state IN ('pending','running')").fetchone()[0]:
                            if time.monotonic() > until:
                                drain_expired = True
                                break
                            await asyncio.sleep(.005)
                        failed = pool.get()._get_conn().execute("SELECT count(*) FROM memory_index_jobs WHERE state='failed'").fetchone()[0]
                        drains.append(dict(run=run_name, operation=operation, concurrency=concurrency,
                            drain_seconds=time.monotonic() - drain_start, failed_jobs=failed,
                            observation_expired=drain_expired,
                            provider_inflight_max=embedder.maximum, foreground_sampled_max=peak_active))
                    finally:
                        await worker.stop()
                        if pool._foreground:
                            await pool._foreground.stop()
                        pool.close_all()
    upper_envelope = await measure_upper_envelope()
    summary = dict(window_start_utc=started, window_end_utc=utc(),
        source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_dirty=bool(subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=no'], text=True
        ).strip()),
        python=platform.python_version(), packages={n: importlib.metadata.version(n) for n in ('httpx', 'fastapi', 'sqlite-vec')},
        measurement='Actual in-process ASGI request + SQLite end-to-end; no TCP adapter/production measurement.',
        provider='Synthetic fixed four-dimensional gate control, zero injected delay; NOT semantic quality evidence.',
        workload=dict(samples_per_path_per_run=samples,
                      runs=[{'name': name, 'concurrency': concurrency} for name, concurrency in run_matrix],
                      payload='one short synthetic assertion/query',
                      limiter='Fresh real middleware per isolated path/run, no limit bypass.',
                      cold_warm_definition=(
                          'Cold is the first concurrency-sized request group against a fresh temporary DB, '
                          'fresh storage pool and fresh synthetic provider; warm is every later request in '
                          'that same isolated path/run.'
                      )),
        distributions={op: stats([r for r in output if r['operation'] == op]) for op in ('capture', 'store', 'recall', 'status')},
        per_run={f'{op}:{name}': stats([r for r in output if r['operation'] == op and r['run'] == name])
                 for name, _ in run_matrix for op in ('capture', 'store', 'recall', 'status')},
        cold_warm={kind: stats([r for r in output if r['cold_warm'] == kind]) for kind in ('cold', 'warm')},
        cold_warm_by_operation={
            f'{op}:{kind}': stats([
                r for r in output if r['operation'] == op and r['cold_warm'] == kind
            ])
            for op in ('capture', 'store', 'recall', 'status')
            for kind in ('cold', 'warm')
        },
        upper_envelope=upper_envelope,
        backlog=drains, production='Unavailable/not accessed; no live attribution or SLO claim.')
    return output, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=60)
    args = parser.parse_args()
    if not 34 <= args.samples <= 100:
        parser.error('samples must be 34..100 to meet total sample floor and limiter window')
    rows, report = asyncio.run(measure(args.samples))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'samples.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['distributions'], indent=2))
    raise SystemExit(int(any(d['successes'] < 100 or d['admitted_over_budget']
                             for d in report['distributions'].values())
                         or any(d['observation_expired'] or d['failed_jobs'] for d in report['backlog'])
                         or not report['upper_envelope']['passed']))
