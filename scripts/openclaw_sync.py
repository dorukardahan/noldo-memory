#!/usr/bin/env python3
"""Incremental sync of OpenClaw sessions into memory.

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
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_memory.config import load_config
from agent_memory.embeddings import OpenRouterEmbeddings
from agent_memory.entities import KnowledgeGraph
from agent_memory.ingest import parse_session_file
from agent_memory.pool import StoragePool
from agent_memory.triggers import score_importance

logger = logging.getLogger("openclaw_sync")

STATE_FILE = Path.home() / ".agent-memory" / "sync_state.json"


def discover_all_agent_sessions(base_dir: Optional[str] = None) -> Dict[str, List[Path]]:
    """Discover sessions for all agents.

    Structure: {base_dir}/agents/{agent_id}/sessions/*.jsonl
    """
    root = Path(base_dir or (Path.home() / ".openclaw"))
    agents_dir = root / "agents"

    results = {}

    if agents_dir.is_dir():
        for agent_path in agents_dir.iterdir():
            if agent_path.is_dir():
                sdir = agent_path / "sessions"
                if sdir.is_dir():
                    sessions = sorted(sdir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
                    if sessions:
                        results[agent_path.name] = sessions

    return results


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
    agent_id: str = "main",
) -> List[Path]:
    """Return sessions that are new or modified since last sync."""
    if full:
        return sessions

    synced = state.get("sessions_synced", {})
    modified = []
    for session in sessions:
        sid = session.stem
        # Use the same key format as when saving state (agent_id:sid for non-main)
        state_key = f"{agent_id}:{sid}" if agent_id != "main" else sid
        mtime = session.stat().st_mtime
        prev = synced.get(state_key)
        if prev is None or prev.get("mtime", 0) < mtime:
            modified.append(session)
    return modified


async def sync(args: argparse.Namespace) -> Dict[str, Any]:
    """Run incremental sync. Returns stats dict."""
    cfg = load_config()
    state = _load_state()

    # Apply CLI overrides
    if getattr(args, "db", None):
        cfg.db_path = args.db
    sessions_dir_override = getattr(args, "sessions_dir", None)

    has_api_key = bool(cfg.openrouter_api_key)
    skip_embeddings = args.skip_embeddings or not has_api_key

    if not has_api_key and not args.skip_embeddings:
        logger.info("No OPENROUTER_API_KEY — running without embeddings")

    # Discover sessions for all agents
    if sessions_dir_override:
        sd = Path(sessions_dir_override)
        agent_sessions = {"main": list(sd.glob("*.jsonl"))} if sd.is_dir() else {}
    else:
        agent_sessions = discover_all_agent_sessions()
    if not agent_sessions:
        logger.info("No session files found for any agent")
        return {"status": "no_sessions", "new": 0, "stored": 0}

    # Derive base directory from db_path's parent for pool
    base_dir = str(Path(cfg.db_path).parent)
    pool = StoragePool(base_dir=base_dir, dimensions=cfg.embedding_dimensions)

    embedder = None
    if not skip_embeddings:
        embedder = OpenRouterEmbeddings(
            api_key=cfg.openrouter_api_key,
            model=cfg.embedding_model,
            dimensions=cfg.embedding_dimensions,
        )

    # Global stats
    total_chunks = 0
    total_stored = 0
    total_skipped = 0
    errors = 0
    processed_agents = 0

    for agent_id, sessions in agent_sessions.items():
        # Find new/modified sessions for this agent
        modified = _get_modified_sessions(sessions, state, full=args.full, agent_id=agent_id)
        if not modified:
            continue

        logger.info("Agent [%s]: Found %d new/modified session(s)", agent_id, len(modified))
        processed_agents += 1
        storage = pool.get(agent_id)
        kg = KnowledgeGraph(storage=storage)

        for session_path in modified:
            sid = session_path.stem
            # Prefix sid with agent_id in state to avoid collision across agents if needed
            # but stem is usually unique UUID. For safety, we track by absolute path/stem.
            state_key = f"{agent_id}:{sid}" if agent_id != "main" else sid

            try:
                chunks = parse_session_file(session_path, gap_hours=cfg.chunk_gap_hours)
            except Exception as exc:
                logger.warning("Failed to parse %s/%s: %s", agent_id, sid[:8], exc)
                errors += 1
                continue

            # Filter out already-stored chunks (batch SELECT)
            chunk_ids = [c.md5 for c in chunks]
            existing_ids: set[str] = set()
            if chunk_ids:
                conn = storage._get_conn()
                placeholders = ",".join(["?"] * len(chunk_ids))
                rows = conn.execute(
                    f"SELECT id FROM memories WHERE id IN ({placeholders})",
                    chunk_ids,
                ).fetchall()
                existing_ids = {r["id"] for r in rows}

            new_chunks = [c for c in chunks if c.md5 not in existing_ids]
            skipped = len(chunks) - len(new_chunks)
            total_skipped += skipped

            if not new_chunks:
                state.setdefault("sessions_synced", {})[state_key] = {
                    "mtime": session_path.stat().st_mtime,
                    "chunks": len(chunks),
                    "agent": agent_id
                }
                continue

            # Embed and store
            texts = [c.text for c in new_chunks]
            vectors = [None] * len(new_chunks)

            if embedder and texts:
                try:
                    vectors = await embedder.embed_batch_resilient(texts, max_sub_batch=8)
                except Exception as exc:
                    logger.warning("Embedding failed for %s/%s: %s", agent_id, sid[:8], exc)

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
                logger.error("Storage failed for %s/%s: %s", agent_id, sid[:8], exc)
                errors += 1
                continue

            # Knowledge graph
            for chunk in new_chunks:
                try:
                    kg.process_text(chunk.text, source=sid, timestamp=chunk.timestamp)
                except Exception:
                    pass

            total_chunks += len(chunks)

            # Update state (multi-agent aware)
            state.setdefault("sessions_synced", {})[state_key] = {
                "mtime": session_path.stat().st_mtime,
                "chunks": len(chunks),
                "agent": agent_id
            }
            logger.info("  %s/%s: %d new chunks stored", agent_id, sid[:8], len(new_chunks))

    if total_stored == 0 and errors == 0:
        return {"status": "up_to_date", "new": 0, "stored": 0}

    # Update state metadata
    state["last_sync"] = datetime.now().isoformat()
    state["total_synced"] = state.get("total_synced", 0) + total_stored
    state["sync_count"] = state.get("sync_count", 0) + 1
    _save_state(state)

    stats = {
        "status": "ok",
        "agents_processed": processed_agents,
        "chunks_parsed": total_chunks,
        "new_stored": total_stored,
        "skipped_existing": total_skipped,
        "errors": errors,
    }

    pool.close_all()
    return stats


def show_status() -> None:
    """Show current sync state."""
    state = _load_state()
    print("=" * 50)
    print("  OpenClaw Memory — Multi-Agent Sync Status")
    print("=" * 50)
    print(f"  Last sync:      {state.get('last_sync', 'never')}")
    print(f"  Sync count:     {state.get('sync_count', 0)}")
    print(f"  Total synced:   {state.get('total_synced', 0)} chunks")
    synced = state.get("sessions_synced", {})
    print(f"  Sessions known: {len(synced)}")

    # Check for new files across all agents
    agent_sessions = discover_all_agent_sessions()
    pending_count = 0
    for agent_id, sessions in agent_sessions.items():
        modified = _get_modified_sessions(sessions, state, agent_id=agent_id)
        pending_count += len(modified)

    print(f"  Pending sync:   {pending_count} session(s) across all agents")
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
        print(f"✓ Synced: {stats['new_stored']} new chunks from {stats['agents_processed']} agent(s)")
        print(f"  Skipped: {stats['skipped_existing']} existing")
        if stats['errors']:
            print(f"  Errors: {stats['errors']}")


if __name__ == "__main__":
    asyncio.run(main())
