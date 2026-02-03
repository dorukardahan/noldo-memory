#!/usr/bin/env python3
"""Incremental sync of OpenClaw sessions into Asuman memory.

Designed to run periodically (via cron or OpenClaw scheduler).
Tracks sync state in a JSON file and only processes new/modified sessions.

Usage:
    python scripts/openclaw_sync.py                  # incremental sync
    python scripts/openclaw_sync.py --full            # re-scan everything
    python scripts/openclaw_sync.py --skip-embeddings # store without vectors
    python scripts/openclaw_sync.py --status          # show sync state
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asuman_memory.config import load_config
from asuman_memory.embeddings import OpenRouterEmbeddings
from asuman_memory.entities import KnowledgeGraph
from asuman_memory.ingest import discover_sessions, parse_session_file
from asuman_memory.storage import MemoryStorage
from asuman_memory.triggers import score_importance

logger = logging.getLogger("openclaw_sync")

STATE_FILE = Path.home() / ".asuman" / "sync_state.json"


def _load_state() -> Dict[str, Any]:
    """Load sync state from JSON file."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        "last_sync": None,
        "sessions_synced": {},  # session_id -> {"mtime": float, "chunks": int}
        "total_synced": 0,
        "sync_count": 0,
    }


def _save_state(state: Dict[str, Any]) -> None:
    """Save sync state to JSON file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _get_modified_sessions(
    sessions: List[Path],
    state: Dict[str, Any],
    full: bool = False,
) -> List[Path]:
    """Return sessions that are new or modified since last sync."""
    if full:
        return sessions

    synced = state.get("sessions_synced", {})
    modified = []
    for session in sessions:
        sid = session.stem
        mtime = session.stat().st_mtime
        prev = synced.get(sid)
        if prev is None or prev.get("mtime", 0) < mtime:
            modified.append(session)
    return modified


async def sync(args: argparse.Namespace) -> Dict[str, Any]:
    """Run incremental sync. Returns stats dict."""
    cfg = load_config()
    state = _load_state()

    has_api_key = bool(cfg.openrouter_api_key)
    skip_embeddings = args.skip_embeddings or not has_api_key

    if not has_api_key and not args.skip_embeddings:
        logger.info("No OPENROUTER_API_KEY — running without embeddings")

    # Discover sessions
    sessions = discover_sessions(args.sessions_dir)
    if not sessions:
        logger.info("No session files found")
        return {"status": "no_sessions", "new": 0, "stored": 0}

    # Find new/modified sessions
    modified = _get_modified_sessions(sessions, state, full=args.full)
    if not modified:
        logger.info("No new/modified sessions since last sync")
        return {"status": "up_to_date", "new": 0, "stored": 0}

    logger.info("Found %d new/modified session(s) to sync", len(modified))

    # Initialize storage
    storage = MemoryStorage(
        db_path=args.db or cfg.db_path,
        dimensions=cfg.embedding_dimensions,
    )

    embedder = None
    if not skip_embeddings:
        embedder = OpenRouterEmbeddings(
            api_key=cfg.openrouter_api_key,
            model=cfg.embedding_model,
            dimensions=cfg.embedding_dimensions,
        )

    kg = KnowledgeGraph(storage=storage)

    # Process each modified session
    total_chunks = 0
    total_stored = 0
    total_skipped = 0
    errors = 0

    for session_path in modified:
        sid = session_path.stem
        try:
            chunks = parse_session_file(session_path, gap_hours=cfg.chunk_gap_hours)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", sid[:8], exc)
            errors += 1
            continue

        # Filter out already-stored chunks
        new_chunks = []
        for chunk in chunks:
            if storage.get_memory(chunk.md5) is None:
                new_chunks.append(chunk)
        skipped = len(chunks) - len(new_chunks)
        total_skipped += skipped

        if not new_chunks:
            # Update mtime in state even if no new chunks
            state.setdefault("sessions_synced", {})[sid] = {
                "mtime": session_path.stat().st_mtime,
                "chunks": len(chunks),
            }
            continue

        # Embed and store
        texts = [c.text for c in new_chunks]
        vectors = [None] * len(new_chunks)

        if embedder and texts:
            try:
                vectors = await embedder.embed_batch(texts)
            except Exception as exc:
                logger.warning("Embedding failed for session %s: %s", sid[:8], exc)

        items = []
        for chunk, vector in zip(new_chunks, vectors):
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
            total_stored += len(ids)
        except Exception as exc:
            logger.error("Storage failed for session %s: %s", sid[:8], exc)
            errors += 1
            continue

        # Knowledge graph
        for chunk in new_chunks:
            try:
                kg.process_text(chunk.text, source=sid, timestamp=chunk.timestamp)
            except Exception:
                pass

        total_chunks += len(chunks)

        # Update state
        state.setdefault("sessions_synced", {})[sid] = {
            "mtime": session_path.stat().st_mtime,
            "chunks": len(chunks),
        }
        logger.info("  Session %s: %d new chunks stored", sid[:8], len(new_chunks))

    # Update state metadata
    state["last_sync"] = datetime.now().isoformat()
    state["total_synced"] = state.get("total_synced", 0) + total_stored
    state["sync_count"] = state.get("sync_count", 0) + 1
    _save_state(state)

    stats = {
        "status": "ok",
        "sessions_processed": len(modified),
        "chunks_parsed": total_chunks,
        "new_stored": total_stored,
        "skipped_existing": total_skipped,
        "errors": errors,
        "db_total": storage.stats()["total_memories"],
    }

    storage.close()
    return stats


def show_status() -> None:
    """Show current sync state."""
    state = _load_state()
    print("=" * 50)
    print("  Asuman Memory — Sync Status")
    print("=" * 50)
    print(f"  Last sync:      {state.get('last_sync', 'never')}")
    print(f"  Sync count:     {state.get('sync_count', 0)}")
    print(f"  Total synced:   {state.get('total_synced', 0)} chunks")
    synced = state.get("sessions_synced", {})
    print(f"  Sessions known: {len(synced)}")

    # Check for new files
    cfg = load_config()
    sessions = discover_sessions()
    modified = _get_modified_sessions(sessions, state)
    print(f"  Pending sync:   {len(modified)} session(s)")
    print("=" * 50)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental OpenClaw session sync")
    parser.add_argument("--sessions-dir", help="Override sessions directory")
    parser.add_argument("--db", help="Override database path")
    parser.add_argument("--full", action="store_true", help="Full re-scan (ignore state)")
    parser.add_argument("--skip-embeddings", action="store_true", help="Store without vectors")
    parser.add_argument("--status", action="store_true", help="Show sync status")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.status:
        show_status()
        return

    stats = await sync(args)
    if stats["status"] == "up_to_date":
        print("✓ Already up to date")
    elif stats["status"] == "no_sessions":
        print("✗ No session files found")
    else:
        print(f"✓ Synced: {stats['new_stored']} new chunks from {stats['sessions_processed']} session(s)")
        print(f"  Skipped: {stats['skipped_existing']} existing")
        if stats['errors']:
            print(f"  Errors: {stats['errors']}")
        print(f"  DB total: {stats['db_total']} memories")


if __name__ == "__main__":
    asyncio.run(main())
