"""One bounded durable index consumer; SQLite never crosses thread affinity.

The executor has one slot and only one submission is outstanding. Timed-out
blocking graph extraction retains that slot until the actual thread drains.
Provider coroutines run on the owning API loop, bypassing all embedding caches.
"""
from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import operations
from .entities import KnowledgeGraph
from .metrics import record_attempt_metric, record_job_metric, record_stage_metric, set_index_metric_gauges
from .storage import MemoryStorage

logger = logging.getLogger(__name__)


class IndexWorker:
    """CAS/lease-safe durable stages, independent of legacy scan enablement."""

    _active = None

    def __init__(self, pool, embedder, config):
        self.pool = pool
        self.embedder = embedder
        budget = getattr(config, 'index_job_timeout_seconds', 30)
        attempts = getattr(config, 'index_job_max_attempts', 3)
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or not 1 <= budget <= 30:
            raise ValueError('invalid_index_job_timeout_seconds')
        if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 3:
            raise ValueError('invalid_index_job_max_attempts')
        self.budget = float(budget)
        self.max_attempts = attempts
        self._task = None
        self._executor = None
        self._stopping = False
        self._storage = None
        self._agent = None
        self._cursor = 0

    async def start(self):
        if self._task is not None and not self._task.done():
            return
        if IndexWorker._active is not None:
            raise RuntimeError('index_worker_already_running')
        IndexWorker._active = self
        self._stopping = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='memory-index')
        self._task = asyncio.create_task(self._run(), name='memory-index-worker')

    async def stop(self):
        """Bound shutdown; an uncancellable thread keeps ownership until drained."""
        self._stopping = True
        task = self._task
        if task is None or task.done():
            return
        task.cancel()
        # Do not cancel a second time: that could abandon the real thread.
        await asyncio.wait({task}, timeout=1)

    async def _offload(self, fn, *args):
        future = asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Cancellation is not proof that SQLite/graph work terminated.
            # Keep the only admission slot and connection alive until it does.
            try:
                await asyncio.shield(future)
            except Exception:
                pass
            raise

    def _open(self, agent):
        """Called ONLY by our one executor thread, never pool.get()."""
        if self._agent != agent:
            self._close()
            self._storage = MemoryStorage.open_existing(self.pool._db_path(agent), self.pool.dimensions)
            self._storage._get_conn().execute('PRAGMA busy_timeout=50')
            if not self._storage._get_conn().execute(
                    "SELECT 1 FROM sqlite_master WHERE name='memory_index_jobs'").fetchone():
                # A cold foreground admission may still be creating this DB.
                # Never bootstrap/backfill corpus tables from the index lane.
                self._close()
                return None
            self._agent = agent
        return self._storage

    def _close(self):
        if self._storage is not None:
            self._storage.close()
            self._storage = None
            self._agent = None

    def _claim_next(self):
        # Discovery accesses paths only; do not borrow any pool connection.
        agents = self.pool.get_all_agents()
        if not agents:
            return None
        for offset in range(len(agents)):
            index = (self._cursor + offset) % len(agents)
            agent = self.pool.normalize_key(agents[index])
            if not Path(self.pool._db_path(agent)).is_file():
                continue
            storage = self._open(agent)
            if storage is None:
                continue
            job = operations.claim(storage, budget=self.budget, max_attempts=self.max_attempts)
            if job is not None:
                self._cursor = (index + 1) % len(agents)
                self._refresh_gauges()
                return job
        self._refresh_gauges()
        return None

    def _refresh_gauges(self):
        """Read process-visible durable queue state without creating a shard."""
        queue_depth = inflight = 0
        for agent in self.pool.get_all_agents():
            path = Path(self.pool._db_path(self.pool.normalize_key(agent)))
            if not path.is_file():
                continue
            conn = None
            try:
                conn = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=.05)
                rows = conn.execute(
                    "SELECT state,count(*) FROM memory_index_jobs "
                    "WHERE state IN ('pending','running') GROUP BY state"
                ).fetchall()
                counts = {state: int(count) for state, count in rows}
                queue_depth += counts.get('pending', 0)
                inflight += counts.get('running', 0)
            except sqlite3.Error:
                continue
            finally:
                if conn is not None:
                    conn.close()
        set_index_metric_gauges(queue_depth=queue_depth, inflight=inflight)

    def _job_state(self, job):
        row = self._storage._get_conn().execute(
            'SELECT state FROM memory_index_jobs WHERE memory_id=? AND stage=?',
            (job['memory_id'], job['stage']),
        ).fetchone()
        return row['state'] if row is not None else None

    def _terminal_context(self, job, source_session):
        state = self._job_state(job)
        return state, state == 'blocked' or self._storage.source_is_forgotten(source_session)

    @staticmethod
    def _record_terminal(job, state):
        # Forget owns blocked transitions after its transaction commits; the
        # worker records only transitions it commits itself.
        if state in {'completed', 'failed'}:
            record_job_metric(job['stage'], state)

    def _memory(self, job):
        memory = operations.fenced_memory(self._storage, job)
        if memory is None:
            with self._storage.transaction() as conn:
                # Missing/deleted/forgotten source must never become success.
                row = conn.execute("SELECT 1 FROM memory_index_jobs WHERE memory_id=? AND stage=? AND attempts=? AND state='running'",
                    (job['memory_id'], job['stage'], job['attempts'])).fetchone()
                if row and (self._storage.get_memory(job['memory_id']) is None or
                            self._storage.get_memory(job['memory_id']).get('deleted_at') is not None or
                            self._storage.source_is_forgotten(self._storage.get_memory(job['memory_id']).get('source_session'))):
                    operations.block_linked_operations(conn, job['memory_id'])
        return memory

    def _prepare(self, memory):
        return KnowledgeGraph(self._storage).prepare_text(memory['text'], source=memory.get('source') or '')

    def _complete(self, job, deadline, metric_stage, metric_started, vector=None, prepared=None):
        graph = None
        if prepared is not None:
            def graph(memory):
                return KnowledgeGraph(self._storage).persist_prepared(prepared, source_memory_id=memory['id'])
        # The outer transaction includes the budget's postcheck; expiry cannot
        # commit half a graph/vector or falsely mark its stage completed.
        with self._storage.transaction(), operations.sql_budget(self._storage, deadline):
            changed = operations.finish(self._storage, job, vector=vector, graph=graph,
                max_attempts=self.max_attempts, deadline=deadline)
            state = self._job_state(job)
        self._record_terminal(job, state)
        outcome = 'completed' if changed and state == 'completed' else 'cancelled'
        record_stage_metric(operation='index', stage=metric_stage,
                            duration_seconds=time.monotonic() - metric_started, outcome=outcome)
        return changed, state

    def _fail(self, job, code, metric_stage, metric_started, metric_outcome):
        changed = operations.finish(self._storage, job, error=code, max_attempts=self.max_attempts)
        state = self._job_state(job)
        self._record_terminal(job, state)
        outcome = metric_outcome if changed else 'cancelled'
        record_stage_metric(operation='index', stage=metric_stage,
                            duration_seconds=time.monotonic() - metric_started, outcome=outcome)
        return changed, state

    async def _execute(self, job):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.budget
        started = time.monotonic()
        stage = 'embedding' if job['stage'] == 'embed' else 'graph'
        committing = False
        try:
            memory = await self._offload(self._memory, job)
            if memory is None:
                state = await self._offload(self._job_state, job)
                outcome = 'blocked' if state == 'blocked' else 'cancelled'
                record_stage_metric(operation='index', stage=stage,
                                    duration_seconds=time.monotonic() - started, outcome=outcome)
                return
            if self._stopping:
                record_stage_metric(operation='index', stage=stage,
                                    duration_seconds=time.monotonic() - started, outcome='cancelled')
                return
            if deadline <= loop.time():
                raise TimeoutError
            if job['stage'] == 'embed':
                if self.embedder is None:
                    await self._offload(self._fail, job, 'embedding_unavailable',
                                        stage, started, 'failed')
                    return
                vector = await asyncio.wait_for(self.embedder.embed(
                    memory['text'], deadline=deadline, background=True, cache=False),
                    timeout=max(.001, deadline - loop.time()))
                # Reject malformed vectors before any storage assignment.
                if not isinstance(vector, (list, tuple)) or len(vector) != self.pool.dimensions or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector):
                    await self._offload(self._fail, job, 'embedding_invalid_vector',
                                        stage, started, 'failed')
                    return
                if not await self._offload(self._memory, job):
                    _, blocked = await self._offload(
                        self._terminal_context, job, memory.get('source_session')
                    )
                    outcome = 'blocked' if blocked else 'cancelled'
                    record_stage_metric(operation='index', stage=stage,
                                        duration_seconds=time.monotonic() - started, outcome=outcome)
                    return
                if deadline <= loop.time():
                    raise TimeoutError
                committing = True
                await self._offload(self._complete, job, deadline, stage, started, vector)
            else:
                record_attempt_metric('graph')
                # wait_for cancels _offload on expiry, but _offload deliberately
                # drains the actual thread before releasing this single slot.
                prepared = await asyncio.wait_for(self._offload(self._prepare, memory),
                    timeout=max(.001, deadline - loop.time()))
                if not await self._offload(self._memory, job):
                    _, blocked = await self._offload(
                        self._terminal_context, job, memory.get('source_session')
                    )
                    outcome = 'blocked' if blocked else 'cancelled'
                    record_stage_metric(operation='index', stage=stage,
                                        duration_seconds=time.monotonic() - started, outcome=outcome)
                    return
                if deadline <= loop.time():
                    raise TimeoutError
                committing = True
                await self._offload(self._complete, job, deadline, stage, started, None, prepared)
        except asyncio.CancelledError:
            if not committing:
                record_stage_metric(operation='index', stage=stage,
                                    duration_seconds=time.monotonic() - started, outcome='cancelled')
            raise
        except (TimeoutError, asyncio.TimeoutError, operations.AdmissionError):
            await self._offload(self._fail, job, 'index_deadline_exceeded',
                                stage, started, 'timeout')
        except Exception:
            # New surfaces contain fixed codes only, never provider/source data.
            await self._offload(self._fail, job, 'index_stage_failed',
                                stage, started, 'failed')

    async def _run(self):
        try:
            while not self._stopping:
                try:
                    job = await self._offload(self._claim_next)
                    if job is None:
                        await asyncio.sleep(.1)
                    else:
                        await self._execute(job)
                        await self._offload(self._refresh_gauges)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning('index_worker_iteration_failed')
                    await asyncio.sleep(.1)
        except asyncio.CancelledError:
            pass
        finally:
            await self._offload(self._refresh_gauges)
            await self._offload(self._close)
            self._executor.shutdown(wait=False, cancel_futures=True)
            if IndexWorker._active is self:
                IndexWorker._active = None
