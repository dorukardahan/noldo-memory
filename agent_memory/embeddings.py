"""Bounded, cancellable OpenAI-compatible embedding client.

Public async requests use httpx, never a blocking executor. One monotonic budget
covers bounded admission, every sub-batch, provider attempts and backoff. The
legacy blocking ``_call_api`` remains only for direct synchronous compatibility;
it is deliberately not used by any public embedding request.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import struct
import threading
import time
from typing import TYPE_CHECKING, List, Optional

import httpx
import numpy as np
import requests

from .config import load_config
from .metrics import record_attempt_metric

if TYPE_CHECKING:
    from .storage import MemoryStorage

logger = logging.getLogger(__name__)
_RETRYABLE = (429, 500, 502, 503, 504)
_PROBE_TEXT = "Embedding backend diagnostic."


class EmbeddingError(Exception):
    """A fixed, payload-free embedding failure code."""


def _finite(value):
    try:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        return False


def _integer(value, minimum, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, int)
            or value < minimum or (maximum is not None and value > maximum)):
        raise ValueError("invalid_embedding_control")
    return value


class OpenRouterEmbeddings:
    """Loop-owned async provider client with a thread-owned optional cache.

    Old positive retry/timeout arguments remain accepted, but cannot extend the
    hard limits of two attempts and two seconds. New batch/concurrency controls
    are strict. There are at most twice ``concurrency`` admitted queue waiters;
    one queue place and, with concurrency two, one I/O slot are foreground-only.
    With concurrency one, foreground priority yields to background after three
    foreground grants, so continuous read traffic cannot starve background work.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        base_url: Optional[str] = None,
        max_retries: int = 2,
        cache_size: int = 1024,
        timeout_seconds: float = 2,
        *,
        max_batch_items: int = 8,
        max_batch_chars: int = 16000,
        concurrency: int = 2,
    ) -> None:
        self.max_retries = min(_integer(max_retries, 1), 2)
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not _finite(timeout_seconds) or timeout_seconds < 0.1):
            raise ValueError("invalid_embedding_control")
        self.timeout_seconds = min(float(timeout_seconds), 2.0)
        self.max_batch_items = _integer(max_batch_items, 1, 8)
        self.max_batch_chars = _integer(max_batch_chars, 1000, 16000)
        self.concurrency = _integer(concurrency, 1, 2)
        self._cache_size = _integer(cache_size, 0)
        cfg = load_config()
        self.api_key: str = api_key or cfg.openrouter_api_key
        self.model: str = model or cfg.embedding_model
        self.dimensions = _integer(cfg.embedding_dimensions if dimensions is None else dimensions, 1, 65536)
        self.base_url: str = (base_url or cfg.openrouter_base_url).rstrip("/")
        self._url = f"{self.base_url}/embeddings"
        self._headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        token = os.environ.get("EMBEDDING_SERVER_TOKEN", "")
        if token:
            self._headers["X-Embedding-Token"] = f"Bearer {token}"

        self._cache_generation = 0
        self._cache: dict[str, List[float]] = {}
        self._cache_order: list[str] = []
        self._storage: Optional[MemoryStorage] = None
        self._storage_thread = None
        self._client: Optional[httpx.AsyncClient] = None
        self._loop = None
        self._closed = False
        self._inflight = 0
        self._background_inflight = 0
        self._foreground_streak = 0
        self._waiters: list[tuple[asyncio.Future, bool]] = []
        self._operations: set[asyncio.Task] = set()

    def _check_storage_owner(self):
        if self._storage is not None and self._storage_thread != threading.get_ident():
            raise EmbeddingError("embedding_cache_owner")

    def clear_cache(self):
        """Clear both cache layers and fence requests already in flight."""
        self._check_storage_owner()
        self._cache_generation += 1
        self._cache.clear()
        self._cache_order.clear()
        if self._storage is not None:
            self._storage.clear_embedding_cache()

    def set_storage(self, storage: MemoryStorage) -> None:
        """Attach a persistent cache owned by the calling thread."""
        self._storage = storage
        self._storage_thread = threading.get_ident()

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _cache_get(self, text: str) -> Optional[List[float]]:
        return self._cache.get(self._cache_key(text))

    def _cache_put(self, text: str, vector: List[float]) -> None:
        if not self._cache_size:
            return
        key = self._cache_key(text)
        if key in self._cache:
            return
        if len(self._cache_order) >= self._cache_size:
            self._cache.pop(self._cache_order.pop(0), None)
        self._cache[key] = vector
        self._cache_order.append(key)

    def _get_cached(self, text):
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        self._check_storage_owner()
        if self._storage is not None:
            blob = self._storage.get_cached_embedding(self._cache_key(text))
            if blob:
                try:
                    vector = list(struct.unpack(f"{len(blob) // 4}f", blob))
                    vector = self._validate_vector(vector)
                except (EmbeddingError, struct.error, TypeError):
                    logger.warning("embedding_cache_invalid")
                else:
                    self._cache_put(text, vector)
                    return vector
        return None

    def _put_cached(self, text, vector, generation):
        if generation != self._cache_generation or self._closed:
            return
        self._check_storage_owner()
        self._cache_put(text, vector)
        if self._storage is not None:
            self._storage.cache_embedding(self._cache_key(text), struct.pack(f"{len(vector)}f", *vector))

    def _deadline(self, deadline):
        if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
                                     or not _finite(deadline)):
            raise EmbeddingError("invalid_embedding_deadline")
        cap = time.monotonic() + self.timeout_seconds
        return cap if deadline is None else min(cap, deadline)

    @staticmethod
    def _remaining(deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EmbeddingError("embedding_deadline")
        return remaining

    def _check_loop(self):
        if self._closed:
            raise EmbeddingError("embedding_closed")
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise EmbeddingError("embedding_loop_owner")

    def _dispatch(self):
        while self._inflight < self.concurrency:
            foreground = next((i for i, (future, bg) in enumerate(self._waiters)
                               if not bg and not future.done()), None)
            background = next((i for i, (future, bg) in enumerate(self._waiters)
                               if bg and not future.done()), None)
            if self.concurrency == 2 and self._background_inflight:
                background = None
            if background is not None and (foreground is None or
                                           (self.concurrency == 1 and self._foreground_streak >= 3)):
                index = background
            else:
                index = foreground
            if index is None:
                break
            future, bg = self._waiters.pop(index)
            self._inflight += 1
            self._background_inflight += int(bg)
            self._foreground_streak = 0 if bg else self._foreground_streak + 1
            future.set_result(None)

    async def _acquire(self, background, deadline):
        self._remaining(deadline)
        future = asyncio.get_running_loop().create_future()
        entry = (future, background)
        self._waiters.append(entry)
        self._dispatch()
        if not future.done():
            limit = self.concurrency * 2
            bg_count = sum(bg for _, bg in self._waiters)
            if len(self._waiters) > limit or (background and bg_count > limit - 1):
                self._waiters.remove(entry)
                future.cancel()
                raise EmbeddingError("embedding_queue_full")
        try:
            # Shield preserves the grant marker across a cancellation/grant race.
            await asyncio.wait_for(asyncio.shield(future), self._remaining(deadline))
        except BaseException as exc:
            if entry in self._waiters:
                self._waiters.remove(entry)
                future.cancel()
            elif future.done() and not future.cancelled():
                self._release(background)
            if isinstance(exc, asyncio.TimeoutError):
                raise EmbeddingError("embedding_deadline") from None
            raise

    def _release(self, background):
        self._inflight -= 1
        self._background_inflight -= int(background)
        if not self._closed:
            self._dispatch()

    def _validate_vector(self, vector):
        if not isinstance(vector, list) or len(vector) != self.dimensions:
            raise EmbeddingError("embedding_invalid_response")
        result = []
        for number in vector:
            if (isinstance(number, bool) or not isinstance(number, (int, float))
                    or not _finite(number) or abs(number) > 3.4028234663852886e38):
                raise EmbeddingError("embedding_invalid_response")
            result.append(float(number))
        return result

    def _validate_response(self, data, count):
        if not isinstance(data, dict) or not isinstance(data.get("data"), list) or len(data["data"]) != count:
            raise EmbeddingError("embedding_invalid_response")
        vectors = [None] * count
        for item in data["data"]:
            if not isinstance(item, dict):
                raise EmbeddingError("embedding_invalid_response")
            index = item.get("index")
            if (isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= count
                    or vectors[index] is not None):
                raise EmbeddingError("embedding_invalid_response")
            vectors[index] = self._validate_vector(item.get("embedding"))
        return vectors

    async def _post(self, texts, deadline):
        if self._client is None:
            self._client = httpx.AsyncClient(
                trust_env=False, follow_redirects=False,
                limits=httpx.Limits(max_connections=self.concurrency, max_keepalive_connections=self.concurrency),
            )
        async with self._client.stream("POST", self._url, headers=self._headers,
                                       json={"model": self.model, "input": texts},
                                       timeout=self._remaining(deadline)) as response:
            if response.status_code != 200:
                return response.status_code, None
            # Bounds response allocation even if a provider streams arbitrary data.
            limit = min(16 * 1024 * 1024, self.dimensions * len(texts) * 32 + 65536)
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > limit:
                    raise EmbeddingError("embedding_invalid_response")
                body.extend(chunk)
            import json
            try:
                data = json.loads(body)
            except (ValueError, UnicodeError, RecursionError):
                raise EmbeddingError("embedding_invalid_response") from None
            return 200, self._validate_response(data, len(texts))

    async def _call_api_async(self, texts, deadline):
        for attempt in range(self.max_retries):
            record_attempt_metric('embedding', retry=attempt > 0)
            try:
                status, vectors = await asyncio.wait_for(self._post(texts, deadline), self._remaining(deadline))
            except (asyncio.TimeoutError, httpx.TimeoutException):
                if deadline <= time.monotonic():
                    raise EmbeddingError("embedding_deadline") from None
                code = "embedding_transport"
            except httpx.HTTPError:
                code = "embedding_transport"
            else:
                if status == 200:
                    self._remaining(deadline)
                    return vectors
                if status not in _RETRYABLE:
                    raise EmbeddingError(f"embedding_http_{status}")
                code = f"embedding_http_{status}"
            if attempt + 1 == self.max_retries:
                raise EmbeddingError(code) from None
            logger.warning("embedding_retry attempt=%d", attempt + 1)
            # Ignore provider Retry-After: it cannot enlarge the shared budget.
            remaining = self._remaining(deadline)
            await asyncio.sleep(min(0.05 * (2 ** attempt), remaining))
            self._remaining(deadline)
        raise EmbeddingError("embedding_unavailable")

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Legacy direct synchronous helper, never called by the async API.

        Kept for historical direct callers/tests that patch ``requests.post``.
        Socket timeouts cannot bound a trickling synchronous response; callers
        requiring total deadlines or cancellation must use ``embed_batch``.
        """
        deadline = self._deadline(None)
        for attempt in range(self.max_retries):
            record_attempt_metric('embedding', retry=attempt > 0)
            try:
                response = requests.post(self._url, headers=self._headers,
                                         json={"model": self.model, "input": texts},
                                         timeout=self._remaining(deadline))
                if response.status_code == 200:
                    try:
                        return self._validate_response(response.json(), len(texts))
                    except (ValueError, UnicodeError, RecursionError):
                        raise EmbeddingError("embedding_invalid_response") from None
                code = f"embedding_http_{response.status_code}"
                if response.status_code not in _RETRYABLE:
                    raise EmbeddingError(code)
            except requests.RequestException:
                code = "embedding_transport"
            if attempt + 1 == self.max_retries:
                raise EmbeddingError(code) from None
            time.sleep(min(0.05 * (2 ** attempt), self._remaining(deadline)))
        raise EmbeddingError("embedding_unavailable")

    def _truncate(self, texts):
        limit = load_config().max_embed_chars
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            limit = 3500
        limit = min(limit, self.max_batch_chars)
        if any(not isinstance(text, str) for text in texts):
            raise EmbeddingError("embedding_invalid_input")
        return [text[:limit] for text in texts]

    def _batches(self, texts, indices, max_items=None):
        batch = []
        chars = 0
        for index in indices:
            size = len(texts[index])
            if batch and (len(batch) >= (max_items or self.max_batch_items)
                          or chars + size > self.max_batch_chars):
                yield batch
                batch, chars = [], 0
            batch.append(index)
            chars += size
        if batch:
            yield batch

    async def embed(self, text: str, *, deadline=None, background=False, cache=True) -> List[float]:
        """Embed one text, retaining configured per-text truncation and caching."""
        return (await self.embed_batch([text], deadline=deadline, background=background, cache=cache))[0]

    async def embed_batch(self, texts: List[str], *, deadline=None, background=False, cache=True) -> List[List[float]]:
        """Embed bounded sub-batches under one total queue/I/O/retry deadline."""
        deadline = self._deadline(deadline)
        self._check_loop()
        self._remaining(deadline)
        if not isinstance(background, bool) or not isinstance(cache, bool):
            raise EmbeddingError("embedding_invalid_input")
        generation = self._cache_generation
        texts = self._truncate(texts)
        results = [None] * len(texts)
        missing = []
        for index, text in enumerate(texts):
            vector = self._get_cached(text) if cache else None
            if vector is None:
                missing.append(index)
            else:
                results[index] = vector
        self._remaining(deadline)
        if not missing:
            return results
        task = asyncio.current_task()
        self._operations.add(task)
        acquired = False
        try:
            await self._acquire(background, deadline)
            acquired = True
            for indices in self._batches(texts, missing):
                self._remaining(deadline)
                vectors = await self._call_api_async([texts[i] for i in indices], deadline)
                self._remaining(deadline)
                for index, vector in zip(indices, vectors):
                    results[index] = vector
                    if cache:
                        self._put_cached(texts[index], vector, generation)
            return results
        finally:
            if acquired:
                self._release(background)
            self._operations.discard(task)

    async def embed_numpy(self, text: str) -> np.ndarray:
        """Convenience: return embedding as a numpy array."""
        return np.array(await self.embed(text), dtype=np.float32)

    async def embed_batch_resilient(self, texts: List[str], max_sub_batch: int = 8,
                                    *, deadline=None, background=False, cache=True) -> List[Optional[List[float]]]:
        """Return partial results without multiplying provider retries.

        Historical individual fallback has been removed: each sub-batch has
        exactly one retry owner and shares the same overall monotonic deadline.
        """
        max_items = min(_integer(max_sub_batch, 1), self.max_batch_items)
        deadline = self._deadline(deadline)
        self._check_loop()
        generation = self._cache_generation
        results = [None] * len(texts)
        normalized = self._truncate(texts)
        for indices in self._batches(normalized, range(len(texts)), max_items):
            if generation != self._cache_generation or deadline <= time.monotonic():
                break
            try:
                vectors = await self.embed_batch([normalized[i] for i in indices], deadline=deadline,
                                                 background=background, cache=cache)
            except EmbeddingError:
                logger.warning("embedding_subbatch_failed")
                continue
            for index, vector in zip(indices, vectors):
                results[index] = vector
        return results

    async def probe(self, *, deadline=None) -> List[float]:
        """Fresh fixed synthetic request with no cache reads, writes or clear."""
        return await self.embed(_PROBE_TEXT, deadline=deadline, cache=False)

    async def aclose(self) -> None:
        """Cancel owned I/O/waiters and release HTTP connections permanently."""
        if self._closed:
            return
        self._check_loop()
        self._closed = True
        self._cache_generation += 1
        current = asyncio.current_task()
        tasks = [task for task in self._operations if task is not current]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._client is not None:
            await self._client.aclose()
