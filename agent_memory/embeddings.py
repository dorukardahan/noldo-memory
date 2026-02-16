"""OpenRouter Embedding Client.

Uses raw ``requests`` (NOT the OpenAI SDK) to call the OpenRouter
``/v1/embeddings`` endpoint.  Features:

* Async-friendly (uses ``asyncio.to_thread`` around blocking requests)
* Batch support — send multiple texts in one call
* Retry with exponential back-off (3 attempts)
* Simple in-memory LRU cache for repeated texts
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import struct
import time
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import requests

from .config import Config, load_config

if TYPE_CHECKING:
    from .storage import MemoryStorage

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when the embedding API returns an error."""


class OpenRouterEmbeddings:
    """Lightweight async wrapper around the OpenRouter embeddings endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        cache_size: int = 1024,
    ) -> None:
        cfg = load_config()
        self.api_key: str = api_key or cfg.openrouter_api_key
        self.model: str = model or cfg.embedding_model
        self.dimensions: int = dimensions or cfg.embedding_dimensions
        self.base_url: str = (base_url or cfg.openrouter_base_url).rstrip("/")
        self.max_retries: int = max_retries

        self._url = f"{self.base_url}/embeddings"
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build an LRU cache keyed on text hash
        self._cache_size = cache_size
        self._cache: dict[str, List[float]] = {}
        self._cache_order: list[str] = []
        self._storage: Optional[MemoryStorage] = None

    def set_storage(self, storage: MemoryStorage) -> None:
        """Set the storage reference for persistent caching."""
        self._storage = storage

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _cache_get(self, text: str) -> Optional[List[float]]:
        key = self._cache_key(text)
        return self._cache.get(key)

    def _cache_put(self, text: str, vector: List[float]) -> None:
        key = self._cache_key(text)
        if key in self._cache:
            return
        if len(self._cache_order) >= self._cache_size:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        self._cache[key] = vector
        self._cache_order.append(key)

    # ------------------------------------------------------------------
    # Low-level HTTP call with retries
    # ------------------------------------------------------------------

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Blocking HTTP POST to OpenRouter with exponential back-off."""
        payload = {
            "model": self.model,
            "input": texts,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self._url,
                    headers=self._headers,
                    json=payload,
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # OpenAI-compatible response: data.data[i].embedding
                    embeddings = sorted(data["data"], key=lambda d: d["index"])
                    return [item["embedding"] for item in embeddings]

                # Retryable server errors
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = 2 ** attempt
                    logger.warning(
                        "OpenRouter %s (attempt %d/%d) — retrying in %ds",
                        resp.status_code, attempt + 1, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    last_exc = EmbeddingError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                    continue

                # Non-retryable
                raise EmbeddingError(
                    f"HTTP {resp.status_code}: {resp.text[:500]}"
                )

            except requests.RequestException as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Request error (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, self.max_retries, exc, wait,
                )
                time.sleep(wait)
                last_exc = exc

        raise EmbeddingError(
            f"Failed after {self.max_retries} retries: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> List[float]:
        """Embed a single text string. Returns a vector (list of floats)."""
        # 1. Check in-memory LRU cache
        cached = self._cache_get(text)
        if cached is not None:
            return cached

        # 2. Check persistent SQLite cache
        key = self._cache_key(text)
        if self._storage:
            blob = self._storage.get_cached_embedding(key)
            if blob:
                vec = list(struct.unpack(f"{len(blob) // 4}f", blob))
                self._cache_put(text, vec)  # backfill LRU
                return vec

        # 3. Call API
        vectors = await asyncio.to_thread(self._call_api, [text])
        vec = vectors[0]

        # 4. Store in both caches
        self._cache_put(text, vec)
        if self._storage:
            blob = struct.pack(f"{len(vec)}f", *vec)
            self._storage.cache_embedding(key, blob)

        return vec

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in one API call. Returns list of vectors."""
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(texts):
            # 1. Check LRU
            cached = self._cache_get(text)
            if cached is not None:
                results[i] = cached
                continue

            # 2. Check Persistent Cache
            key = self._cache_key(text)
            if self._storage:
                blob = self._storage.get_cached_embedding(key)
                if blob:
                    vec = list(struct.unpack(f"{len(blob) // 4}f", blob))
                    results[i] = vec
                    self._cache_put(text, vec)
                    continue

            # 3. API needed
            uncached_indices.append(i)
            uncached_texts.append(text)

        if uncached_texts:
            vectors = await asyncio.to_thread(self._call_api, uncached_texts)
            for idx, vec in zip(uncached_indices, vectors):
                results[idx] = vec
                self._cache_put(texts[idx], vec)
                if self._storage:
                    key = self._cache_key(texts[idx])
                    blob = struct.pack(f"{len(vec)}f", *vec)
                    self._storage.cache_embedding(key, blob)

        return results  # type: ignore[return-value]

    async def embed_numpy(self, text: str) -> np.ndarray:
        """Convenience: return embedding as a numpy array."""
        vec = await self.embed(text)
        return np.array(vec, dtype=np.float32)
