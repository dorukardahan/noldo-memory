"""Request-local recall context; never borrows event-loop SQLite connections."""
from __future__ import annotations

import asyncio
import time
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
        async def call():
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            return await asyncio.wait_for(self.embedder.embed(text), remaining)
        future = asyncio.run_coroutine_threadsafe(call(), self.loop)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future),
                max(.001, self.deadline - time.monotonic()))
        except BaseException:
            future.cancel()
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
