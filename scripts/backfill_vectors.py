#!/usr/bin/env python3
"""Backfill vector embeddings for vectorless memories.

Finds all memories without vectors and generates embeddings for them.
Safe to run concurrently with the API (uses WAL mode, busy_timeout).

Suggested cron: 0 */6 * * * cd /path/to/agent-memory && . .env && venv/bin/python scripts/backfill_vectors.py --agent all

Usage:
    python scripts/backfill_vectors.py              # backfill all agents
    python scripts/backfill_vectors.py --agent main # backfill specific agent
    python scripts/backfill_vectors.py --dry-run    # show what would be done
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_memory.config import load_config
from agent_memory.embeddings import OpenRouterEmbeddings
from agent_memory.pool import StoragePool

logger = logging.getLogger("backfill_vectors")


def get_vectorless_memories(storage, batch_size: int = 100) -> List[Dict[str, Any]]:
    """Get memories without vectors (vector_rowid IS NULL AND deleted_at IS NULL)."""
    conn = storage._get_conn()
    cursor = conn.execute(
        """
        SELECT id, text, category, importance, source_session
        FROM memories
        WHERE vector_rowid IS NULL
          AND deleted_at IS NULL
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (batch_size,),
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


def update_memory_vector(storage, memory_id: str, vector: List[float]) -> bool:
    """Update a memory with its vector. Returns True if successful."""
    import numpy as np
    import time

    conn = storage._get_conn()
    try:
        # Insert vector into memory_vectors table
        blob = np.array(vector, dtype=np.float32).tobytes()
        cur = conn.execute(
            "INSERT INTO memory_vectors(embedding) VALUES (?)",
            (blob,),
        )
        vector_rowid = cur.lastrowid

        # Update memory with vector_rowid
        conn.execute(
            """
            UPDATE memories
            SET vector_rowid = ?, updated_at = ?
            WHERE id = ?
            """,
            (vector_rowid, time.time(), memory_id),
        )
        conn.commit()
        return True
    except Exception as exc:
        conn.rollback()
        logger.error("Failed to update vector for memory %s: %s", memory_id, exc)
        return False


async def backfill_agent(
    agent_id: str,
    storage,
    embedder: OpenRouterEmbeddings,
    batch_size: int = 2,
    max_sub_batch: int = 2,
    sleep_between_batches: float = 1.0,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Backfill vectors for a single agent."""
    stats = {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0}

    # Check total count first
    total_vectorless = count_vectorless_memories(storage)
    if total_vectorless == 0:
        logger.info("Agent [%s]: No vectorless memories found", agent_id)
        return stats

    logger.info(
        "Agent [%s]: Found %d vectorless memories, processing in batches of %d",
        agent_id, total_vectorless, batch_size
    )

    processed_total = 0
    while True:
        # Get batch of vectorless memories
        memories = get_vectorless_memories(storage, batch_size=batch_size)
        if not memories:
            break

        if dry_run:
            for mem in memories:
                logger.info(
                    "[DRY-RUN] Would backfill: %s (%.50s...)",
                    mem["id"], mem["text"]
                )
            stats["processed"] += len(memories)
            processed_total += len(memories)
            continue

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
        for mem, vector in zip(memories, vectors):
            stats["processed"] += 1
            processed_total += 1

            if vector is None:
                stats["failed"] += 1
                logger.warning(
                    "No vector generated for memory %s (%.50s...)",
                    mem["id"], mem["text"]
                )
                continue

            if update_memory_vector(storage, mem["id"], vector):
                stats["succeeded"] += 1
                logger.debug("Updated vector for memory %s", mem["id"])
            else:
                stats["failed"] += 1

        # Log progress every batch
        logger.info(
            "Agent [%s]: Progress %d/%d (%.1f%%)",
            agent_id, processed_total, total_vectorless,
            (processed_total / total_vectorless * 100) if total_vectorless else 0
        )

        if sleep_between_batches > 0:
            await asyncio.sleep(sleep_between_batches)

        # Safety: if we got fewer than batch_size, we're done
        if len(memories) < batch_size:
            break

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
        # Discover all agents from pool directory
        agents_dir = Path(base_dir) / "agents"
        if agents_dir.exists():
            agent_ids = [
                d.name for d in agents_dir.iterdir()
                if d.is_dir() and (d / "memory.db").exists()
            ]
        else:
            agent_ids = ["main"]
        # Also check main database
        main_db = Path(base_dir) / "memory.db"
        if main_db.exists():
            agent_ids.append("main")
        # Deduplicate
        agent_ids = list(set(agent_ids))
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
    total_stats = {"processed": 0, "succeeded": 0, "failed": 0}
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
                dry_run=args.dry_run,
            )
            total_stats["processed"] += stats["processed"]
            total_stats["succeeded"] += stats["succeeded"]
            total_stats["failed"] += stats["failed"]
        except Exception as exc:
            logger.error("Failed to backfill agent %s: %s", agent_id, exc)

    pool.close_all()

    # Summary
    logger.info("=" * 50)
    logger.info("Backfill complete")
    logger.info("  Total processed: %d", total_stats["processed"])
    logger.info("  Succeeded: %d", total_stats["succeeded"])
    logger.info("  Failed: %d", total_stats["failed"])
    logger.info("=" * 50)

    if total_stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
