#!/usr/bin/env python3
"""Bulk-index all existing JSONL sessions into the Asuman memory database.

Usage:
    python scripts/initial_load.py [--sessions-dir PATH] [--db PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from asuman_memory.config import load_config
from asuman_memory.embeddings import OpenRouterEmbeddings
from asuman_memory.entities import KnowledgeGraph
from asuman_memory.ingest import ingest_sessions
from asuman_memory.storage import MemoryStorage


def _progress(done: int, total: int) -> None:
    pct = (done / total * 100) if total else 0
    print(f"\r  [{done}/{total}] {pct:.0f}%", end="", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-index sessions")
    parser.add_argument("--sessions-dir", help="Override sessions directory")
    parser.add_argument("--db", help="Override database path")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--no-kg", action="store_true", help="Skip knowledge graph extraction")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = load_config()

    storage = MemoryStorage(
        db_path=args.db or cfg.db_path,
        dimensions=cfg.embedding_dimensions,
    )
    embedder = OpenRouterEmbeddings(
        api_key=cfg.openrouter_api_key,
        model=cfg.embedding_model,
        dimensions=cfg.embedding_dimensions,
    )

    kg = None if args.no_kg else KnowledgeGraph(storage=storage)

    print(f"DB:       {storage.db_path}")
    print(f"Model:    {cfg.embedding_model}")
    print(f"Sessions: {args.sessions_dir or cfg.sessions_dir}")
    print()

    t0 = time.time()
    stats = await ingest_sessions(
        storage=storage,
        embedder=embedder,
        sessions_dir=args.sessions_dir,
        batch_size=args.batch_size,
        progress_cb=_progress,
        knowledge_graph=kg,
    )
    elapsed = time.time() - t0

    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"  Sessions: {stats['sessions']}")
    print(f"  Chunks:   {stats['chunks']}")
    print(f"  Stored:   {stats['stored']}")
    print(f"  Skipped:  {stats['skipped_dup']} (duplicates)")

    db_stats = storage.stats()
    print(f"\nDatabase stats:")
    print(f"  Total memories: {db_stats['total_memories']}")
    print(f"  Entities:       {db_stats['entities']}")
    print(f"  Relationships:  {db_stats['relationships']}")

    storage.close()


if __name__ == "__main__":
    asyncio.run(main())
