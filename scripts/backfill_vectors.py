#!/usr/bin/env python3
"""Backfill vector embeddings for vectorless memories.

Finds all memories without vectors and generates embeddings for them.
Uses bounded SQLite write retries so API contention fails visibly instead of
spinning forever. For large repairs, prefer running while the API is quiesced.

Suggested cron:
    0 */6 * * * cd /path/to/agent-memory && . .env && \
        venv/bin/python scripts/backfill_vectors.py --agent all

Usage:
    python scripts/backfill_vectors.py              # backfill all agents
    python scripts/backfill_vectors.py --agent main # backfill specific agent
    python scripts/backfill_vectors.py --dry-run    # show what would be done
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_memory.config import load_config
from agent_memory.embeddings import OpenRouterEmbeddings
from agent_memory.pool import StoragePool

logger = logging.getLogger("backfill_vectors")


def get_vectorless_memories(
    storage,
    batch_size: int = 100,
    exclude_ids: Sequence[str] = (),
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Get memories without vectors (vector_rowid IS NULL AND deleted_at IS NULL)."""
    conn = storage._get_conn()
    params: List[Any] = []
    exclude_clause = ""
    if exclude_ids:
        placeholders = ",".join("?" for _ in exclude_ids)
        exclude_clause = f" AND id NOT IN ({placeholders})"
        params.extend(exclude_ids)
    params.extend([batch_size, offset])
    cursor = conn.execute(
        f"""
        SELECT id, text, category, importance, source_session
        FROM memories
        WHERE vector_rowid IS NULL
          AND deleted_at IS NULL
          {exclude_clause}
        ORDER BY created_at DESC
        LIMIT ?
        OFFSET ?
        """,
        params,
    )
    return [dict(row) for row in cursor.fetchall()]


def count_vectorless_memories(storage) -> int:
    """Count total vectorless memories."""
    conn = storage._get_conn()
    row = conn.execute(
        """
        SELECT COUNT(*) as count
        FROM memories
        WHERE vector_rowid IS NULL
          AND deleted_at IS NULL
        """
    ).fetchone()
    return row["count"] if row else 0


def _is_sqlite_lock_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "database is locked" in text or "database is busy" in text


def _rollback_quietly(conn) -> None:
    try:
        conn.rollback()
    except Exception:
        pass


def update_memory_vector(
    storage,
    memory_id: str,
    vector: List[float],
    max_attempts: int = 3,
    retry_base_sleep: float = 0.5,
) -> str:
    """Update a memory with its vector.

    Returns one of:
    - "succeeded": vector was written and memory row was updated.
    - "skipped": memory was already updated/deleted by another writer.
    - "locked": SQLite stayed locked after bounded retries.
    - "failed": non-lock failure.
    """
    import numpy as np

    conn = storage._get_conn()
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        try:
            conn.execute("BEGIN IMMEDIATE")
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
                _rollback_quietly(conn)
                logger.info("Skipped vector update for memory %s (already updated/deleted)", memory_id)
                return "skipped"

            conn.commit()
            return "succeeded"
        except sqlite3.OperationalError as exc:
            _rollback_quietly(conn)
            if _is_sqlite_lock_error(exc):
                if attempt < attempts:
                    sleep_for = max(0.0, retry_base_sleep) * (2 ** (attempt - 1))
                    logger.warning(
                        "SQLite locked while updating memory %s (attempt %d/%d); retrying in %.2fs",
                        memory_id,
                        attempt,
                        attempts,
                        sleep_for,
                    )
                    time.sleep(sleep_for)
                    continue
                logger.error(
                    "SQLite stayed locked while updating memory %s after %d attempts",
                    memory_id,
                    attempts,
                )
                return "locked"
            logger.error("SQLite update failed for memory %s: %s", memory_id, exc)
            return "failed"
        except Exception as exc:
            _rollback_quietly(conn)
            logger.error("Failed to update vector for memory %s: %s", memory_id, exc)
            return "failed"

    return "failed"


async def backfill_agent(
    agent_id: str,
    storage,
    embedder: OpenRouterEmbeddings,
    batch_size: int = 2,
    max_sub_batch: int = 2,
    sleep_between_batches: float = 1.0,
    max_write_attempts: int = 3,
    write_retry_base_sleep: float = 0.5,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Backfill vectors for a single agent."""
    stats = {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0, "locked": 0}

    # Check total count first
    total_vectorless = count_vectorless_memories(storage)
    if total_vectorless == 0:
        logger.info("Agent [%s]: No vectorless memories found", agent_id)
        return stats

    logger.info(
        "Agent [%s]: Found %d vectorless memories, processing in batches of %d",
        agent_id, total_vectorless, batch_size
    )

    retry_excluded_ids: set[str] = set()
    max_batches = max(1, (total_vectorless + max(1, batch_size) - 1) // max(1, batch_size) + 1)
    abort_agent = False

    if dry_run:
        offset = 0
        while True:
            memories = get_vectorless_memories(storage, batch_size=batch_size, offset=offset)
            if not memories:
                break
            for mem in memories:
                logger.info(
                    "[DRY-RUN] Would backfill: %s (%.50s...)",
                    mem["id"], mem["text"]
                )
            stats["processed"] += len(memories)
            stats["skipped"] += len(memories)
            offset += len(memories)
            if len(memories) < batch_size:
                break
        return stats

    for _batch_index in range(max_batches):
        # Get batch of vectorless memories
        memories = get_vectorless_memories(
            storage,
            batch_size=batch_size,
            exclude_ids=tuple(retry_excluded_ids),
        )
        if not memories:
            break

        # Prepare texts for embedding
        texts = [mem["text"] for mem in memories]

        try:
            vectors = await embedder.embed_batch_resilient(texts, max_sub_batch=max_sub_batch)
        except Exception as exc:
            logger.error("Embedding batch failed for %s: %s", agent_id, exc)
            # Fall back to individual
            vectors = []
            for text in texts:
                try:
                    vec = await embedder.embed(text)
                    vectors.append(vec)
                except Exception as single_exc:
                    logger.warning("Individual embed failed: %s", single_exc)
                    vectors.append(None)

        # Update each memory with its vector
        batch_succeeded = 0
        batch_failed = 0
        for mem, vector in zip(memories, vectors):
            stats["processed"] += 1

            if vector is None:
                retry_excluded_ids.add(mem["id"])
                stats["failed"] += 1
                batch_failed += 1
                logger.warning(
                    "No vector generated for memory %s (%.50s...)",
                    mem["id"], mem["text"]
                )
                continue

            status = update_memory_vector(
                storage,
                mem["id"],
                vector,
                max_attempts=max_write_attempts,
                retry_base_sleep=write_retry_base_sleep,
            )
            if status == "succeeded":
                stats["succeeded"] += 1
                batch_succeeded += 1
                logger.debug("Updated vector for memory %s", mem["id"])
            elif status == "skipped":
                stats["skipped"] += 1
            elif status == "locked":
                retry_excluded_ids.add(mem["id"])
                stats["failed"] += 1
                stats["locked"] += 1
                batch_failed += 1
                abort_agent = True
                logger.error(
                    "Agent [%s]: aborting backfill after persistent SQLite lock; "
                    "remaining vectorless memories will be retried by a later run",
                    agent_id,
                )
                break
            else:
                retry_excluded_ids.add(mem["id"])
                stats["failed"] += 1
                batch_failed += 1

        # Log progress every batch
        logger.info(
            "Agent [%s]: Progress %d/%d (%.1f%%)",
            agent_id, stats["processed"], total_vectorless,
            (stats["processed"] / total_vectorless * 100) if total_vectorless else 0
        )

        if abort_agent:
            break

        if batch_succeeded == 0 and batch_failed > 0:
            logger.error(
                "Agent [%s]: aborting backfill after a batch made no write progress; "
                "remaining vectorless memories will be retried by a later run",
                agent_id,
            )
            break

        if sleep_between_batches > 0:
            await asyncio.sleep(sleep_between_batches)

        # Safety: if we got fewer than batch_size, we're done
        if len(memories) < batch_size:
            break
    else:
        logger.error(
            "Agent [%s]: stopped after %d batches without reaching a clean end; "
            "this prevents runaway retries",
            agent_id,
            max_batches,
        )

    return stats


async def main():
    parser = argparse.ArgumentParser(
        description="Backfill vector embeddings for vectorless memories"
    )
    parser.add_argument(
        "--agent",
        default="all",
        help='Agent ID to backfill, or "all" for all agents (default: all)',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of memories to process per batch (default: 2)",
    )
    parser.add_argument(
        "--max-sub-batch",
        type=int,
        default=2,
        help="Max texts per embedding API call (default: 2)",
    )
    parser.add_argument(
        "--sleep-between-batches",
        type=float,
        default=1.0,
        help="Sleep seconds between batches (default: 1.0)",
    )
    parser.add_argument(
        "--max-write-attempts",
        type=int,
        default=3,
        help="Max SQLite write attempts per memory before failing fast (default: 3)",
    )
    parser.add_argument(
        "--write-retry-base-sleep",
        type=float,
        default=0.5,
        help="Base sleep seconds for SQLite write retry backoff (default: 0.5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = load_config()

    # Check for API key
    if not cfg.openrouter_api_key:
        logger.error("No OPENROUTER_API_KEY configured - cannot generate embeddings")
        sys.exit(1)

    # Derive base directory from db_path's parent for pool
    base_dir = str(Path(cfg.db_path).parent)
    pool = StoragePool(base_dir=base_dir, dimensions=cfg.embedding_dimensions)

    embedder = OpenRouterEmbeddings(
        api_key=cfg.openrouter_api_key,
        model=cfg.embedding_model,
        dimensions=cfg.embedding_dimensions,
    )

    # Determine which agents to process
    if args.agent == "all":
        agent_ids = pool.get_all_agents()
    else:
        agent_ids = [args.agent]

    if not agent_ids:
        logger.info("No agents found to backfill")
        return

    logger.info(
        "Starting backfill for agents: %s (dry-run=%s)",
        ", ".join(agent_ids), args.dry_run
    )

    # Process each agent
    total_stats = {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0, "locked": 0}
    for agent_id in agent_ids:
        try:
            storage = pool.get(agent_id)
            stats = await backfill_agent(
                agent_id=agent_id,
                storage=storage,
                embedder=embedder,
                batch_size=args.batch_size,
                max_sub_batch=args.max_sub_batch,
                sleep_between_batches=args.sleep_between_batches,
                max_write_attempts=args.max_write_attempts,
                write_retry_base_sleep=args.write_retry_base_sleep,
                dry_run=args.dry_run,
            )
            total_stats["processed"] += stats["processed"]
            total_stats["succeeded"] += stats["succeeded"]
            total_stats["failed"] += stats["failed"]
            total_stats["skipped"] += stats["skipped"]
            total_stats["locked"] += stats["locked"]
        except Exception as exc:
            logger.error("Failed to backfill agent %s: %s", agent_id, exc)

    pool.close_all()

    # Summary
    logger.info("=" * 50)
    logger.info("Backfill complete")
    logger.info("  Total processed: %d", total_stats["processed"])
    logger.info("  Succeeded: %d", total_stats["succeeded"])
    logger.info("  Failed: %d", total_stats["failed"])
    logger.info("  Skipped: %d", total_stats["skipped"])
    logger.info("  Locked: %d", total_stats["locked"])
    logger.info("=" * 50)

    if total_stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
