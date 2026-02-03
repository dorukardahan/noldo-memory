#!/usr/bin/env python3
"""Bulk-index all existing JSONL sessions into the Asuman memory database.

Usage:
    python scripts/initial_load.py [--sessions-dir PATH] [--db PATH]
    python scripts/initial_load.py --dry-run          # parse only, no embedding
    python scripts/initial_load.py --limit 3           # process only 3 session files
    python scripts/initial_load.py --skip-embeddings   # store without vectors
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from asuman_memory.config import load_config
from asuman_memory.embeddings import OpenRouterEmbeddings
from asuman_memory.entities import KnowledgeGraph
from asuman_memory.ingest import discover_sessions, parse_session_file
from asuman_memory.storage import MemoryStorage
from asuman_memory.triggers import score_importance


def _progress(done: int, total: int) -> None:
    pct = (done / total * 100) if total else 0
    bar_len = 30
    filled = int(bar_len * done / total) if total else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  {bar} {done}/{total} ({pct:.0f}%)", end="", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-index sessions into Asuman memory")
    parser.add_argument("--sessions-dir", help="Override sessions directory")
    parser.add_argument("--db", help="Override database path")
    parser.add_argument("--batch-size", type=int, default=50, help="Embedding batch size")
    parser.add_argument("--limit", type=int, default=0, help="Max session files to process (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, don't store anything")
    parser.add_argument("--skip-embeddings", action="store_true", help="Store without embeddings")
    parser.add_argument("--no-kg", action="store_true", help="Skip knowledge graph extraction")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("initial_load")

    cfg = load_config()

    # Check API key
    has_api_key = bool(cfg.openrouter_api_key)
    if not has_api_key and not args.skip_embeddings and not args.dry_run:
        logger.warning(
            "No OPENROUTER_API_KEY found. Running with --skip-embeddings. "
            "Set the env var or use --skip-embeddings explicitly."
        )
        args.skip_embeddings = True

    # Discover sessions
    sessions_dir = args.sessions_dir or cfg.sessions_dir
    sessions = discover_sessions(sessions_dir)
    if not sessions:
        print(f"No session files found in {sessions_dir}")
        return

    if args.limit > 0:
        sessions = sessions[:args.limit]

    print("=" * 60)
    print("  Asuman Memory — Initial Data Load")
    print("=" * 60)
    print(f"  Sessions dir: {sessions_dir}")
    print(f"  Session files: {len(sessions)}")
    print(f"  Database:      {args.db or cfg.db_path}")
    print(f"  Embed model:   {cfg.embedding_model}")
    print(f"  API key:       {'✓ set' if has_api_key else '✗ not set'}")
    print(f"  Mode:          {'DRY RUN' if args.dry_run else 'skip-embed' if args.skip_embeddings else 'FULL (embed + store)'}")
    print(f"  Knowledge graph: {'off' if args.no_kg else 'on'}")
    print("=" * 60)
    print()

    # Phase 1: Parse all sessions
    print("[1/3] Parsing session files...")
    all_chunks = []
    parse_errors = 0
    for i, session_path in enumerate(sessions):
        try:
            chunks = parse_session_file(session_path, gap_hours=cfg.chunk_gap_hours)
            all_chunks.extend(chunks)
            _progress(i + 1, len(sessions))
        except Exception as exc:
            parse_errors += 1
            logger.warning("Failed to parse %s: %s", session_path.name, exc)
    print()
    print(f"  Parsed: {len(all_chunks)} chunks from {len(sessions)} sessions ({parse_errors} errors)")
    print()

    if args.dry_run:
        # Show stats and exit
        roles = {}
        for c in all_chunks:
            roles[c.role] = roles.get(c.role, 0) + 1
        print("  Chunk breakdown by role:")
        for role, count in sorted(roles.items()):
            print(f"    {role}: {count}")

        # Show sample chunks (no message content for security)
        print(f"\n  Sample chunk lengths (first 5):")
        for c in all_chunks[:5]:
            print(f"    session={c.session_id[:8]}... role={c.role} len={len(c.text)} ts={c.timestamp[:19] if c.timestamp else 'N/A'}")

        # Estimate tokens
        total_chars = sum(len(c.text) for c in all_chunks)
        est_tokens = total_chars // 4  # rough estimate
        est_cost = est_tokens / 1_000_000 * 0.01  # $0.01/M tokens for qwen3
        print(f"\n  Total text: {total_chars:,} chars (~{est_tokens:,} tokens)")
        print(f"  Estimated embedding cost: ${est_cost:.4f}")
        print("\n  DRY RUN complete — no data was stored.")
        return

    # Phase 2: Deduplicate and store
    print("[2/3] Deduplicating...")
    seen_md5 = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk.md5 not in seen_md5:
            seen_md5.add(chunk.md5)
            unique_chunks.append(chunk)
    dup_count = len(all_chunks) - len(unique_chunks)
    print(f"  Unique: {len(unique_chunks)} (removed {dup_count} duplicates)")
    print()

    # Initialize storage
    storage = MemoryStorage(
        db_path=args.db or cfg.db_path,
        dimensions=cfg.embedding_dimensions,
    )

    # Check for already-stored chunks
    new_chunks = []
    for chunk in unique_chunks:
        existing = storage.get_memory(chunk.md5)
        if existing is None:
            new_chunks.append(chunk)
    already_stored = len(unique_chunks) - len(new_chunks)
    if already_stored > 0:
        print(f"  Skipping {already_stored} already-stored chunks")
    unique_chunks = new_chunks
    print(f"  To process: {len(unique_chunks)} chunks")
    print()

    if not unique_chunks:
        print("  Nothing new to store!")
        db_stats = storage.stats()
        print(f"\n  Database: {db_stats['total_memories']} memories, {db_stats['entities']} entities")
        storage.close()
        return

    # Initialize embedder (if needed)
    embedder = None
    if not args.skip_embeddings and has_api_key:
        embedder = OpenRouterEmbeddings(
            api_key=cfg.openrouter_api_key,
            model=cfg.embedding_model,
            dimensions=cfg.embedding_dimensions,
        )

    kg = None if args.no_kg else KnowledgeGraph(storage=storage)

    # Phase 3: Embed and store
    action = "Embedding + storing" if embedder else "Storing (no embeddings)"
    print(f"[3/3] {action}...")
    t0 = time.time()
    stored = 0
    embed_errors = 0
    total = len(unique_chunks)
    batch_size = args.batch_size

    for batch_start in range(0, total, batch_size):
        batch = unique_chunks[batch_start : batch_start + batch_size]
        texts = [c.text for c in batch]

        # Batch embed
        vectors = [None] * len(batch)
        if embedder:
            try:
                vectors = await embedder.embed_batch(texts)
            except Exception as exc:
                embed_errors += 1
                logger.warning("Embedding batch %d failed: %s", batch_start // batch_size, exc)

        # Prepare items for batch storage
        items = []
        for chunk, vector in zip(batch, vectors):
            importance = score_importance(chunk.text, {"role": chunk.role})
            items.append({
                "id": chunk.md5,
                "text": chunk.text,
                "vector": vector,
                "category": chunk.role,
                "importance": importance,
                "source_session": chunk.session_id,
            })

        try:
            ids = storage.store_memories_batch(items)
            stored += len(ids)
        except Exception as exc:
            logger.error("Storage batch failed: %s", exc)

        # Knowledge graph
        if kg:
            for chunk in batch:
                try:
                    kg.process_text(
                        chunk.text,
                        source=chunk.session_id,
                        timestamp=chunk.timestamp,
                    )
                except Exception as exc:
                    logger.debug("KG extraction failed: %s", exc)

        _progress(min(batch_start + batch_size, total), total)

    elapsed = time.time() - t0
    print()
    print()

    # Final stats
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  Time:          {elapsed:.1f}s")
    print(f"  Sessions:      {len(sessions)}")
    print(f"  Chunks parsed: {len(all_chunks)}")
    print(f"  Duplicates:    {dup_count}")
    print(f"  Already stored:{already_stored}")
    print(f"  Newly stored:  {stored}")
    if embed_errors:
        print(f"  Embed errors:  {embed_errors}")
    if parse_errors:
        print(f"  Parse errors:  {parse_errors}")

    db_stats = storage.stats()
    print(f"\n  Database totals:")
    print(f"    Memories:      {db_stats['total_memories']}")
    print(f"    Entities:      {db_stats['entities']}")
    print(f"    Relationships: {db_stats['relationships']}")
    if db_stats.get('by_category'):
        print(f"    By category:")
        for cat, count in db_stats['by_category'].items():
            print(f"      {cat}: {count}")
    print("=" * 60)

    storage.close()


if __name__ == "__main__":
    asyncio.run(main())
