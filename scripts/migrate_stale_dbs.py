#!/usr/bin/env python3
"""Migrate memories from stale agent DBs to active agent DBs.

Plan:
  bureau (82)          -> agent-asuman
  researcher (14)      -> agent-asuman
  kimi-orchestrator (12) -> main
  projects (5)         -> main
  x-accounts (9)       -> 500

Preserves: text, category, memory_type, importance, strength,
           created_at, updated_at, last_accessed_at, pinned,
           source, trust_level, lesson_status, lesson_scope,
           resolved_at, namespace, original_text.

vector_rowid is intentionally set to NULL — vectors must be
re-embedded via backfill_vectors.py after migration because
vector rowids are DB-local and cannot be copied across DBs.

Adds source_session prefix '[migrated-from:{agent}]' for traceability.
Does NOT delete source DBs. Idempotent (skips already-migrated rows).
Atomic per source DB (rollback on any error).
"""

import sqlite3
import sys
import time
import uuid
from pathlib import Path

BASE_DIR = Path("/root/.asuman")

MIGRATIONS = [
    ("bureau", "agent-asuman"),
    ("researcher", "agent-asuman"),
    ("kimi-orchestrator", "main"),
    ("projects", "main"),
    ("x-accounts", "500"),
]

# Columns that may not exist in older stale DBs.
# Maps column name -> default value if missing.
OPTIONAL_COLUMNS = {
    "strength": 1.0,
    "last_accessed_at": None,  # will fall back to created_at
    "pinned": 0,
    "source": "api",
    "trust_level": "user",
    "lesson_status": None,
    "lesson_scope": None,
    "resolved_at": None,
    "namespace": "default",
    "original_text": None,
    "memory_type": "other",
}


def db_path(agent: str) -> Path:
    if agent == "main":
        return BASE_DIR / "memory.sqlite"
    return BASE_DIR / f"memory-{agent}.sqlite"


def _safe_get(row, col, row_keys, default=None):
    """Get column value from sqlite3.Row, returning default if column missing."""
    if col in row_keys:
        val = row[col]
        return val if val is not None else default
    return default


def migrate(src_agent: str, dst_agent: str, dry_run: bool = False) -> int:
    src_path = db_path(src_agent)
    dst_path = db_path(dst_agent)

    if not src_path.exists():
        print(f"  SKIP: {src_path} does not exist")
        return 0

    src = sqlite3.connect(str(src_path))
    src.row_factory = sqlite3.Row

    rows = src.execute(
        "SELECT * FROM memories WHERE deleted_at IS NULL"
    ).fetchall()

    if not rows:
        print(f"  SKIP: {src_agent} has no active memories")
        src.close()
        return 0

    if dry_run:
        print(f"  DRY-RUN: would migrate {len(rows)} memories from {src_agent} -> {dst_agent}")
        src.close()
        return len(rows)

    dst = sqlite3.connect(str(dst_path))
    dst.execute("PRAGMA journal_mode=WAL")
    dst.execute("PRAGMA busy_timeout = 5000")

    # Idempotency: check which source IDs are already migrated (per-row, not per-DB)
    migrated_tag = f"[migrated-from:{src_agent}]"
    existing_count = dst.execute(
        "SELECT COUNT(*) FROM memories WHERE source_session LIKE ?",
        (f"{migrated_tag}%",),
    ).fetchone()[0]

    # Build set of source IDs already in destination for per-row skip
    if existing_count > 0:
        existing_sessions = set()
        for r in dst.execute(
            "SELECT source_session FROM memories WHERE source_session LIKE ?",
            (f"{migrated_tag}%",),
        ).fetchall():
            existing_sessions.add(r[0])
    else:
        existing_sessions = set()

    # Detect source schema once
    sample_keys = rows[0].keys() if rows else []

    migrated = 0
    skipped = 0

    # Atomic: use a single transaction, rollback everything on unexpected error
    try:
        for row in rows:
            row_keys = row.keys() if not sample_keys else sample_keys
            orig_ss = row["source_session"] if "source_session" in row_keys else ""
            orig_ss = orig_ss or ""
            source_session = f"{migrated_tag} {orig_ss}".rstrip()

            # Per-row idempotency: skip if this exact source_session already exists
            if source_session in existing_sessions:
                skipped += 1
                continue

            new_id = uuid.uuid4().hex[:16]
            created_at = row["created_at"]
            last_accessed = _safe_get(row, "last_accessed_at", row_keys, default=created_at)

            dst.execute(
                """INSERT INTO memories
                   (id, text, category, memory_type, importance, strength,
                    created_at, updated_at, last_accessed_at, pinned,
                    source, trust_level, lesson_status, lesson_scope,
                    resolved_at, namespace, original_text, source_session,
                    vector_rowid, deleted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)""",
                (
                    new_id,
                    row["text"],
                    row["category"],
                    _safe_get(row, "memory_type", row_keys, "other"),
                    row["importance"],
                    _safe_get(row, "strength", row_keys, 1.0),
                    created_at,
                    row["updated_at"],
                    last_accessed,
                    _safe_get(row, "pinned", row_keys, 0),
                    _safe_get(row, "source", row_keys, "api"),
                    _safe_get(row, "trust_level", row_keys, "user"),
                    _safe_get(row, "lesson_status", row_keys, None),
                    _safe_get(row, "lesson_scope", row_keys, None),
                    _safe_get(row, "resolved_at", row_keys, None),
                    _safe_get(row, "namespace", row_keys, "default"),
                    _safe_get(row, "original_text", row_keys, None),
                    source_session,
                ),
            )
            # FTS sync
            dst.execute(
                "INSERT OR REPLACE INTO memory_fts(id, text) VALUES (?, ?)",
                (new_id, row["text"]),
            )
            migrated += 1

        dst.commit()
    except Exception as e:
        dst.rollback()
        print(f"  FATAL: rolling back all changes — {e}")
        dst.close()
        src.close()
        return 0

    dst.close()
    src.close()
    if skipped > 0:
        print(f"  Skipped {skipped} already-migrated rows")
    return migrated


def main():
    dry_run = "--dry-run" in sys.argv
    mode = "DRY-RUN" if dry_run else "LIVE"
    print(f"=== Stale DB Migration ({mode}) ===")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    total = 0
    for src, dst in MIGRATIONS:
        print(f"Migrating: {src} -> {dst}")
        count = migrate(src, dst, dry_run=dry_run)
        print(f"  Result: {count} memories migrated")
        total += count
        print()

    print(f"=== Total: {total} memories migrated ===")


if __name__ == "__main__":
    main()
