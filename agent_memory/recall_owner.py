"""Request-local recall context; never borrows event-loop SQLite connections."""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future
from contextvars import ContextVar

from . import operations
from .pool import StoragePool
from .storage import MemoryStorage

current = ContextVar('memory_owned_recall', default=None)


class ProviderBridge:
    """Provider I/O remains on its owning API loop with the SAME deadline."""
    def __init__(self, embedder, loop, deadline):
        self.embedder, self.loop, self.deadline = embedder, loop, deadline

    async def embed(self, text):
        if self.deadline <= time.monotonic():
            raise TimeoutError
        if self.loop.is_closed():
            raise RuntimeError('provider_owner_closed')
        completion = Future()
        cancelled = threading.Event()
        task = None

        async def call():
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            return await self.embedder.embed(text)

        def finished(done):
            # This acknowledgement means the real provider task has drained,
            # not merely that its cross-thread proxy was cancelled.
            try:
                result = done.result()
            except BaseException as exc:
                completion.set_exception(exc)
            else:
                completion.set_result(result)

        def start():
            nonlocal task
            if cancelled.is_set() or self.deadline <= time.monotonic():
                completion.set_exception(TimeoutError())
                return
            # Allocate the coroutine on its owning loop, never before a queued
            # callback which may be abandoned during loop shutdown.
            task = self.loop.create_task(call())
            task.add_done_callback(finished)

        def cancel():
            if task is not None and not task.done():
                task.cancel()

        self.loop.call_soon_threadsafe(start)
        future = asyncio.wrap_future(completion)
        try:
            return await asyncio.wait_for(asyncio.shield(future),
                max(.001, self.deadline - time.monotonic()))
        except BaseException:
            cancelled.set()
            self.loop.call_soon_threadsafe(cancel)
            # Preserve foreground admission until cancellation cleanup really
            # finishes. Timeout of the HTTP caller cannot recycle this slot.
            try:
                await asyncio.shield(future)
            except BaseException:
                pass
            raise


class RecallOwner:
    def __init__(self, pool, seed, key, deadline, embedder, loop, config):
        self.pool, self.deadline, self.config = pool, deadline, config
        self.storages = {key: seed}
        self.seed = seed
        self.searches = {}
        self.embedder = ProviderBridge(embedder, loop, deadline) if embedder is not None else None

    def storage(self, agent):
        key = StoragePool.normalize_key(agent)
        if time.monotonic() >= self.deadline or not self.seed._request_valid():
            raise operations.AdmissionError('deadline_exceeded', 504)
        if key not in self.storages:
            storage = MemoryStorage.open_existing(self.pool._db_path(key), self.pool.dimensions,
                                                 deadline=self.deadline)
            storage._request_valid = self.seed._request_valid
            self.storages[key] = storage
        return self.storages[key]

    def close(self):
        for storage in self.storages.values():
            if storage is not self.seed:
                storage.close()

    def run(self, coroutine):
        token = current.set(self)
        try:
            # asyncio.run also drains actual default-executor reranking work;
            # the foreground slot cannot be recycled merely on await timeout.
            async def bounded():
                return await asyncio.wait_for(coroutine, max(.001, self.deadline - time.monotonic()))
            return asyncio.run(bounded())
        finally:
            current.reset(token)
            self.close()
