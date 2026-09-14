"""Background worker for embedding vectorless memories."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from .embeddings import OpenRouterEmbeddings
    from .pool import StoragePool
    from .storage import MemoryStorage

logger = logging.getLogger("embed_worker")


class EmbedWorker:
    """Async background worker that retries embeddings for vectorless memories."""

    CIRCUIT_BREAKER_THRESHOLD = 5
    CIRCUIT_BREAKER_BACKOFF_SECONDS = 300.0

    def __init__(
        self,
        storage_pool: "StoragePool",
        embedder: "OpenRouterEmbeddings",
        interval_seconds: int = 300,
        batch_size: int = 2,
        max_sub_batch: int = 2,
        sleep_between: float = 1.0,
    ) -> None:
        self.storage_pool = storage_pool
        self.embedder = embedder
        self.interval_seconds = max(1, int(interval_seconds))
        self.batch_size = max(1, int(batch_size))
        self.max_sub_batch = max(1, int(max_sub_batch))
        self.sleep_between = max(0.0, float(sleep_between))

        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._consecutive_embedding_failures = 0

    def start(self) -> None:
        """Start the background worker task."""
        if self._task and not self._task.done():
            logger.debug("Embed worker already running")
            return

        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="embed-worker")
        logger.info(
            "Embed worker started (interval=%ss batch_size=%d max_sub_batch=%d sleep_between=%.1fs)",
            self.interval_seconds,
            self.batch_size,
            self.max_sub_batch,
            self.sleep_between,
        )

    async def stop(self) -> None:
        """Stop the background worker task gracefully."""
        task = self._task
        if task is None:
            return

        self._stop_event.set()
        try:
            await asyncio.wait_for(task, timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("Embed worker stop timed out; cancelling task")
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Embed worker exited with error during stop: %s", exc)
        finally:
            self._task = None

        logger.info("Embed worker stopped")

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._process_all_agents_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Embed worker main loop error: %s", exc)

            if await self._sleep_or_stop(self.interval_seconds):
                break

    async def _process_all_agents_once(self) -> None:
        agents = self.storage_pool.get_all_agents()
        for agent_id in agents:
            if self._stop_event.is_set():
                return
            try:
                await self._process_agent(agent_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Embed worker failed for agent '%s': %s", agent_id, exc)

    async def _process_agent(self, agent_id: str) -> None:
        storage = self.storage_pool.get(agent_id)

        vectorless_count = self._count_vectorless_memories(storage)
        if vectorless_count <= 0:
            return

        logger.info(
            "Embed worker: agent=%s has %d vectorless memories",
            agent_id,
            vectorless_count,
        )

        vectorless = self._get_vectorless_memories(storage)
        if not vectorless:
            return

        for start in range(0, len(vectorless), self.batch_size):
            if self._stop_event.is_set():
                return

            # Backlog rows may have been forgotten or replaced during an await.
            batch = [storage.get_memory(item["id"]) for item in vectorless[start:start + self.batch_size]]
            batch = [item for item in batch if item and item["deleted_at"] is None and item["vector_rowid"] is None]
            if not batch:
                continue
            texts = [item["text"] for item in batch]

            try:
                vectors = await self.embedder.embed_batch_resilient(
                    texts,
                    max_sub_batch=self.max_sub_batch,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Embed worker: batch embed failed for agent=%s: %s", agent_id, exc)
                should_stop = await self._record_embedding_failure()
                if should_stop:
                    return
                if await self._sleep_or_stop(self.sleep_between):
                    return
                continue

            if len(vectors) != len(batch):
                logger.warning(
                    "Embed worker: vector count mismatch for agent=%s (got=%d expected=%d)",
                    agent_id,
                    len(vectors),
                    len(batch),
                )

            updated = 0
            for idx, memory in enumerate(batch):
                vector = vectors[idx] if idx < len(vectors) else None

                if vector is None:
                    logger.warning(
                        "Embed worker: no vector generated for memory %s (agent=%s)",
                        memory["id"],
                        agent_id,
                    )
                    should_stop = await self._record_embedding_failure()
                    if should_stop:
                        return
                    continue

                self._record_embedding_success()
                if self._update_memory_vector(storage, memory["id"], vector):
                    updated += 1

            if updated:
                try:
                    storage.invalidate_search_cache(agent=agent_id)
                except Exception:
                    pass
                logger.info("Embed worker: updated %d vectors for agent=%s", updated, agent_id)

            has_more = (start + self.batch_size) < len(vectorless)
            if has_more and await self._sleep_or_stop(self.sleep_between):
                return

    async def _record_embedding_failure(self) -> bool:
        self._consecutive_embedding_failures += 1
        if self._consecutive_embedding_failures < self.CIRCUIT_BREAKER_THRESHOLD:
            return False

        logger.error(
            "Embed worker circuit breaker opened after %d consecutive embedding failures; backing off for %ss",
            self._consecutive_embedding_failures,
            int(self.CIRCUIT_BREAKER_BACKOFF_SECONDS),
        )
        self._consecutive_embedding_failures = 0
        return await self._sleep_or_stop(self.CIRCUIT_BREAKER_BACKOFF_SECONDS)

    def _record_embedding_success(self) -> None:
        self._consecutive_embedding_failures = 0

    async def _sleep_or_stop(self, seconds: float) -> bool:
        if seconds <= 0:
            return self._stop_event.is_set()

        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
            return True
        except asyncio.TimeoutError:
            return self._stop_event.is_set()

    def _count_vectorless_memories(self, storage: "MemoryStorage") -> int:
        conn = storage._get_conn()
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
              FROM memories
             WHERE vector_rowid IS NULL
               AND deleted_at IS NULL
            """
        ).fetchone()
        return int(row["count"]) if row else 0

    def _get_vectorless_memories(self, storage: "MemoryStorage") -> List[Dict[str, Any]]:
        conn = storage._get_conn()
        rows = conn.execute(
            """
            SELECT id, text
              FROM memories
             WHERE vector_rowid IS NULL
               AND deleted_at IS NULL
             ORDER BY created_at ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def _update_memory_vector(self, storage: "MemoryStorage", memory_id: str, vector: List[float]) -> bool:
        conn = storage._get_conn()

        try:
            conn.execute("BEGIN")
            blob = np.array(vector, dtype=np.float32).tobytes()
            cur = conn.execute(
                "INSERT INTO memory_vectors(embedding) VALUES (?)",
                (blob,),
            )
            vector_rowid = cur.lastrowid

            updated = conn.execute(
                """
                UPDATE memories
                   SET vector_rowid = ?, updated_at = ?
                 WHERE id = ?
                   AND vector_rowid IS NULL
                   AND deleted_at IS NULL
                """,
                (vector_rowid, time.time(), memory_id),
            )
            if int(updated.rowcount or 0) < 1:
                conn.rollback()
                return False

            conn.commit()
            return True
        except Exception as exc:
            conn.rollback()
            logger.warning("Embed worker: failed to update vector for memory %s: %s", memory_id, exc)
            return False
