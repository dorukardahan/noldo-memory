"""Bounded off-loop foreground SQLite ownership, with a reserved read slot.

There is no executor backlog: admitted calls retain capacity until the real
thread drains, even after HTTP cancellation. Each connection is opened, used
and closed on its owner thread. Cancellation/deadline fences prevent late
admission; a commit that already won is resolvable by its stable identity.
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import operations
from .storage import MemoryStorage


class Foreground:
    def __init__(self, pool):
        self.pool = pool
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='memory-foreground')
        self.active = {}
        self.writes = 0
        self.closed = False

    def close(self):
        self.closed = True
        for event, _ in self.active.values():
            event.set()
        self.executor.shutdown(wait=False, cancel_futures=True)

    async def stop(self):
        self.close()
        futures = set(self.active)
        if futures:
            await asyncio.wait(futures, timeout=1)

    async def run(self, agent, deadline, fn, *, write=False, heavy=False,
                  valid=lambda: True, metric_operation=None):
        if self.closed or not valid():
            raise operations.AdmissionError('foreground_closed', 503)
        if time.monotonic() >= deadline:
            raise operations.AdmissionError('deadline_exceeded', 504)
        heavy = heavy or write
        if len(self.active) >= 4 or (heavy and self.writes >= 3):
            raise operations.AdmissionError('foreground_full', 429)
        event = threading.Event()
        path = self.pool._db_path(agent)
        dimensions = self.pool.dimensions
        queued_at = time.monotonic()
        if heavy:
            self.writes += 1

        def execute():
            storage = None
            try:
                if metric_operation is not None:
                    from .metrics import record_stage_metric
                    record_stage_metric(operation=metric_operation, stage='queue',
                                        duration_seconds=time.monotonic() - queued_at)
                if event.is_set() or not valid() or time.monotonic() >= deadline:
                    raise operations.AdmissionError('deadline_exceeded', 504)
                if Path(path).is_file():
                    storage = MemoryStorage.open_existing(path, dimensions, readonly=not write,
                                                          deadline=deadline, cancelled=event)
                    if write and not storage._get_conn().execute(
                        "SELECT 1 FROM sqlite_master WHERE name='memory_operations'").fetchone():
                        storage.close()
                        storage = None
                if storage is None:
                    if not write:
                        raise operations.AdmissionError('operation_not_found', 404)
                    storage = MemoryStorage(path, dimensions, deadline=deadline, cancelled=event)
                storage._request_valid = lambda: not event.is_set() and not self.closed and valid()
                with operations.sql_budget(storage, deadline):
                    if not storage._request_valid():
                        raise operations.AdmissionError('deadline_exceeded', 504)
                    return fn(storage)
            finally:
                if storage is not None:
                    storage.close()

        future = asyncio.get_running_loop().run_in_executor(self.executor, execute)
        self.active[future] = (event, heavy)

        def drained(done):
            _, was_write = self.active.pop(done, (None, False))
            if was_write:
                self.writes -= 1
            # Consume late exceptions; never put their content in logs.
            if not done.cancelled():
                done.exception()
        future.add_done_callback(drained)
        try:
            return await asyncio.wait_for(asyncio.shield(future), max(.001, deadline - time.monotonic()))
        except TimeoutError:
            event.set()
            raise operations.AdmissionError('deadline_exceeded', 504) from None
        except asyncio.CancelledError:
            event.set()
            raise
