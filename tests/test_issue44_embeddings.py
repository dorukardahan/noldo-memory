"""Disposable HTTP behavior tests for bounded, cancellable embeddings."""

import asyncio
import json
import math
import sqlite3
import struct
import time
from contextlib import asynccontextmanager

import pytest

from agent_memory.embeddings import EmbeddingError, OpenRouterEmbeddings


class Provider:
    """An actual local HTTP peer; no upstream credentials or endpoints."""

    def __init__(self, mode="normal", delay=0):
        self.mode = mode
        self.delay = delay
        self.calls = []
        self.tasks = set()
        self.started = asyncio.Event()
        self.disconnected = asyncio.Event()
        self.active = 0
        self.maximum = 0
        self.vector_data = None

    async def handle(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n")
                          if line.lower().startswith(b"content-length:"))
            payload = json.loads(await reader.readexactly(length))
            texts = payload["input"]
            self.calls.append(texts)
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            self.started.set()
            if self.mode == "stall":
                await reader.read()
                self.disconnected.set()
                return
            if self.mode == "reset":
                writer.transport.abort()
                return
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.mode == "trickle":
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 10000\r\nConnection: close\r\n\r\n")
                await writer.drain()
                while True:
                    writer.write(b" ")
                    await writer.drain()
                    await asyncio.sleep(0.02)
            status = int(self.mode) if self.mode.isdigit() else 200
            data = self.vector_data
            if data is None:
                data = [{"index": i, "embedding": [float(len(text)), 2., 3., 4.]}
                        for i, text in enumerate(texts)]
            body = (b"not-json-sensitive-provider-payload" if self.mode == "badjson"
                    else json.dumps({"data": data}).encode())
            writer.write(f"HTTP/1.1 {status} Test\r\nContent-Length: {len(body)}\r\n"
                         "Retry-After: 99999999\r\nConnection: close\r\n\r\n".encode() + body)
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.CancelledError):
            self.disconnected.set()
        finally:
            self.active = max(0, self.active - 1)
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            self.tasks.discard(task)


@asynccontextmanager
async def local_provider(mode="normal", delay=0):
    peer = Provider(mode, delay)
    server = await asyncio.start_server(peer.handle, "127.0.0.1", 0)
    peer.url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1"
    try:
        yield peer
    finally:
        server.close()
        await server.wait_closed()
        for task in list(peer.tasks):
            task.cancel()
        await asyncio.gather(*list(peer.tasks), return_exceptions=True)


def client(peer, **kwargs):
    return OpenRouterEmbeddings(api_key="synthetic", dimensions=4,
                                base_url=peer.url, **kwargs)


async def close(embedder):
    # Allows the RED run against the old class, which has no async lifecycle.
    if hasattr(embedder, "aclose"):
        await embedder.aclose()


@pytest.mark.asyncio
async def test_batch_is_split_by_item_and_character_caps():
    async with local_provider() as peer:
        embedder = client(peer)
        try:
            texts = [str(i) * 3500 for i in range(10)]
            vectors = await embedder.embed_batch(texts)
            assert len(vectors) == 10
            assert len(peer.calls) >= 3
            assert all(len(batch) <= 8 and sum(map(len, batch)) <= 16000 for batch in peer.calls)
        finally:
            await close(embedder)


@pytest.mark.asyncio
async def test_cancellation_closes_actual_provider_io():
    async with local_provider("stall") as peer:
        embedder = client(peer, max_retries=1, timeout_seconds=0.1)
        task = asyncio.create_task(embedder.embed("cancel-sensitive"))
        try:
            await asyncio.wait_for(peer.started.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.02)
            assert peer.disconnected.is_set(), "cancellation left provider socket/work running"
        finally:
            await close(embedder)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["stall", "trickle", "reset", "429", "500", "503", "badjson"])
async def test_provider_faults_have_one_total_budget(mode, caplog):
    async with local_provider(mode) as peer:
        embedder = client(peer, timeout_seconds=0.12)
        start = time.monotonic()
        try:
            with pytest.raises(EmbeddingError) as error:
                await asyncio.wait_for(embedder.embed("sensitive-input"), 0.5)
            assert time.monotonic() - start < 0.35
            assert len(peer.calls) <= 2
            assert "sensitive" not in str(error.value) + caplog.text
            assert peer.url not in str(error.value) + caplog.text
        finally:
            await close(embedder)


@pytest.mark.asyncio
@pytest.mark.parametrize("data", [
    [],
    [{"index": 1, "embedding": [1, 2, 3, 4]}],
    [{"index": True, "embedding": [1, 2, 3, 4]}],
    [{"index": 0, "embedding": [1, 2, 3]}],
    [{"index": 0, "embedding": [1, 2, 3, math.nan]}],
    [{"index": 0, "embedding": [1, 2, 3, math.inf]}],
    [{"index": 0, "embedding": [1, 2, 3, True]}],
    [{"index": 0, "embedding": [1, 2, 3, "4"]}],
    [{"index": 0, "embedding": [1, 2, 3, 10 ** 400]}],
    [{"index": 0, "embedding": [1, 2, 3, 1e100]}],
    [{"index": 0, "embedding": [1, 2, 3, 4]}, {"index": 0, "embedding": [1, 2, 3, 4]}],
])
async def test_malformed_vectors_fail_without_cache_mutation(data):
    async with local_provider() as peer:
        peer.vector_data = data
        embedder = client(peer)
        try:
            with pytest.raises(EmbeddingError):
                await embedder.embed_batch(["malformed"])
            assert embedder._cache == {}
        finally:
            await close(embedder)


class PersistentCache:
    def __init__(self):
        # A real thread-affine SQLite connection proves cache use stays on its owner.
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE TABLE cache (key TEXT PRIMARY KEY, value BLOB)")
        self.reads = 0
        self.writes = 0

    def get_cached_embedding(self, key):
        self.reads += 1
        row = self.conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def cache_embedding(self, key, value):
        self.writes += 1
        self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, value))

    def clear_embedding_cache(self):
        self.conn.execute("DELETE FROM cache")


@pytest.mark.asyncio
async def test_uncached_and_probe_bypass_both_cache_layers():
    async with local_provider() as peer:
        embedder = client(peer)
        storage = PersistentCache()
        embedder.set_storage(storage)
        try:
            await embedder.embed("warm")
            before = (dict(embedder._cache), list(embedder._cache_order), storage.reads, storage.writes)
            await embedder.embed("warm", cache=False)
            await embedder.probe()
            await embedder.probe()
            assert len(peer.calls) == 4
            assert peer.calls[-1] == peer.calls[-2]
            assert (embedder._cache, embedder._cache_order, storage.reads, storage.writes) == before
            await embedder.embed("warm")
            assert len(peer.calls) == 4
        finally:
            await close(embedder)
            storage.conn.close()


@pytest.mark.asyncio
async def test_generation_fences_inflight_cache_writes():
    async with local_provider(delay=0.06) as peer:
        embedder = client(peer)
        storage = PersistentCache()
        embedder.set_storage(storage)
        try:
            task = asyncio.create_task(embedder.embed_batch(["a", "b"]))
            await peer.started.wait()
            embedder.clear_cache()
            assert len(await task) == 2
            assert embedder._cache == {}
            assert storage.writes == 0
        finally:
            await close(embedder)
            storage.conn.close()


@pytest.mark.asyncio
async def test_subbatches_share_callers_remaining_deadline():
    async with local_provider(delay=0.08) as peer:
        embedder = client(peer, max_batch_items=1)
        try:
            # This case measures a shared subbatch budget, not cold TLS setup.
            await embedder.embed('warm connection')
            peer.calls.clear()
            start = time.monotonic()
            with pytest.raises(EmbeddingError, match="deadline"):
                await embedder.embed_batch(["a", "b", "c"], deadline=start + 0.13)
            assert time.monotonic() - start < 0.3
            assert len(peer.calls) == 2
        finally:
            await close(embedder)


@pytest.mark.asyncio
async def test_background_reserves_capacity_and_overload_is_bounded():
    async with local_provider("stall") as peer:
        embedder = client(peer, timeout_seconds=0.4)
        tasks = []
        try:
            tasks.append(asyncio.create_task(embedder.embed("bg", background=True)))
            await peer.started.wait()
            tasks.extend(asyncio.create_task(embedder.embed(str(i), background=True)) for i in range(20))
            await asyncio.sleep(0.02)
            peer.mode = "normal"
            assert await embedder.embed("foreground") == [10., 2., 3., 4.]
            assert peer.maximum <= 2
            tasks[0].cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert sum(isinstance(result, EmbeddingError) and "queue_full" in str(result)
                       for result in results) >= 15
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await close(embedder)


@pytest.mark.asyncio
async def test_concurrency_one_prioritizes_foreground_then_background():
    async with local_provider(delay=0.03) as peer:
        embedder = client(peer, concurrency=1)
        try:
            first = asyncio.create_task(embedder.embed("first", background=True))
            await peer.started.wait()
            background = asyncio.create_task(embedder.embed("background", background=True))
            foreground = asyncio.create_task(embedder.embed("foreground"))
            await asyncio.gather(first, background, foreground)
            assert peer.calls == [["first"], ["foreground"], ["background"]]
            assert peer.maximum == 1
        finally:
            await close(embedder)


@pytest.mark.asyncio
async def test_resilient_has_no_nested_retry_multiplication():
    async with local_provider("503") as peer:
        embedder = client(peer, timeout_seconds=0.2)
        try:
            result = await embedder.embed_batch_resilient(["a", "b", "c"])
            assert result == [None, None, None]
            assert len(peer.calls) <= 2
        finally:
            await close(embedder)


@pytest.mark.asyncio
async def test_close_cancels_inflight_and_rejects_reuse():
    async with local_provider("stall") as peer:
        embedder = client(peer)
        task = asyncio.create_task(embedder.embed("closing"))
        await peer.started.wait()
        await embedder.aclose()
        assert task.done()
        assert embedder._cache == {}
        with pytest.raises(EmbeddingError, match="closed"):
            await embedder.embed("after-close")


@pytest.mark.parametrize("name,value", [
    ("timeout_seconds", math.nan), ("timeout_seconds", math.inf), ("timeout_seconds", True),
    ("timeout_seconds", 10 ** 400),
    ("timeout_seconds", 0), ("timeout_seconds", "2"), ("max_retries", 0),
    ("max_retries", True), ("max_batch_items", 9), ("max_batch_chars", 16001),
    ("concurrency", 0), ("concurrency", 3), ("cache_size", -1),
])
def test_invalid_constructor_controls_are_fixed_errors(name, value):
    with pytest.raises(ValueError, match="invalid_embedding_control"):
        OpenRouterEmbeddings(api_key="synthetic", **{name: value})


@pytest.mark.asyncio
@pytest.mark.parametrize("deadline", [math.nan, math.inf, 10 ** 400, True, "2", [], {}])
async def test_invalid_deadline_never_contacts_provider(deadline):
    async with local_provider() as peer:
        embedder = client(peer)
        try:
            with pytest.raises(EmbeddingError, match="invalid_embedding_deadline"):
                await embedder.embed("invalid", deadline=deadline)
            assert peer.calls == []
        finally:
            await close(embedder)


@pytest.mark.asyncio
async def test_persistent_cache_roundtrip():
    async with local_provider() as peer:
        embedder = client(peer)
        storage = PersistentCache()
        embedder.set_storage(storage)
        try:
            await embedder.embed("persist")
            embedder._cache.clear()
            embedder._cache_order.clear()
            assert await embedder.embed("persist") == list(struct.unpack("4f", struct.pack("4f", 7, 2, 3, 4)))
            assert len(peer.calls) == 1
        finally:
            await close(embedder)
            storage.conn.close()
