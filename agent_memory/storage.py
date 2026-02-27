"""SQLite + sqlite-vec storage layer.

Single-file database with:
* ``sqlite-vec`` extension for vector similarity search
* FTS5 virtual table for full-text search
* Knowledge graph tables (entities, relationships, temporal_facts)
* Auto-create schema on first use
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# sqlite-vec extension loading
# ---------------------------------------------------------------------------

def _load_vec_extension(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension into *conn*.

    Hardening: only enable extension loading for the duration of the load call.
    """
    conn.enable_load_extension(True)
    try:
        import sqlite_vec  # noqa: F401
        sqlite_vec.load(conn)
    except Exception as exc:
        logger.error("Failed to load sqlite-vec: %s", exc)
        raise
    finally:
        try:
            conn.enable_load_extension(False)
        except Exception:
            # Some sqlite builds may not support toggling; ignore.
            pass


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- Memory metadata
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    category TEXT DEFAULT 'other',
    importance REAL DEFAULT 0.5,
    source_session TEXT,
    created_at REAL NOT NULL,
    updated_at REAL,
    vector_rowid INTEGER,
    -- Ebbinghaus strength + last access timestamp
    strength REAL DEFAULT 1.0,
    last_accessed_at REAL,
    -- Soft-delete / archive support
    deleted_at REAL,
    -- Critical memory pinning: pinned memories survive decay/gc/consolidation
    pinned INTEGER DEFAULT 0,
    -- Namespace for topic-based memory grouping
    namespace TEXT DEFAULT 'default'
);

CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);

-- Full-text search (trigram tokenizer for Turkish/multilingual)
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    id,
    text,
    tokenize='trigram'
);

-- Knowledge graph: entities
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT,
    aliases TEXT DEFAULT '[]',
    first_seen REAL,
    last_seen REAL,
    mention_count INTEGER DEFAULT 1,
    metadata TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);

-- Knowledge graph: relationships
CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES entities(id),
    target_id TEXT REFERENCES entities(id),
    relation_type TEXT,
    confidence REAL DEFAULT 0.5,
    context TEXT DEFAULT '',
    created_at REAL
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);

-- Temporal facts
CREATE TABLE IF NOT EXISTS temporal_facts (
    id TEXT PRIMARY KEY,
    entity_id TEXT REFERENCES entities(id),
    fact TEXT NOT NULL,
    valid_from REAL,
    valid_to REAL,
    source_memory_id TEXT REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_tf_entity ON temporal_facts(entity_id);

-- Cache tables
CREATE TABLE IF NOT EXISTS search_result_cache (
    query_norm TEXT NOT NULL,
    limit_val INTEGER NOT NULL,
    min_score REAL NOT NULL DEFAULT 0.0,
    agent TEXT NOT NULL DEFAULT 'main',
    results_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    PRIMARY KEY (query_norm, limit_val, min_score, agent)
);

CREATE TABLE IF NOT EXISTS embedding_cache (
    text_hash TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    created_at REAL NOT NULL
);
"""


class MemoryStorage:
    """SQLite-backed storage with vector search and FTS5."""

    def __init__(self, db_path: Optional[str] = None, dimensions: int = 4096) -> None:
        from .config import load_config

        cfg = load_config()
        self.db_path = db_path or cfg.db_path
        self.dimensions = dimensions or cfg.embedding_dimensions

        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()
        # Schema migration for typed relations
        self.run_migrations()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout = 5000")  # 5s wait on write contention
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA cache_size = -64000")  # 64MB page cache
            self._conn.execute("PRAGMA mmap_size = 300000000")  # 300MB mmap
            self._conn.execute("PRAGMA temp_store = MEMORY")
            _load_vec_extension(self._conn)
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        cur = conn.cursor()
        # Regular tables
        for stmt in _SCHEMA_SQL.split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    cur.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # table/index already exists variant

        # sqlite-vec virtual table (separate because CREATE VIRTUAL TABLE
        # doesn't support IF NOT EXISTS in some sqlite-vec versions)
        try:
            cur.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
                    embedding float[{self.dimensions}]
                )
            """)
        except sqlite3.OperationalError:
            pass  # already exists

        # Safe migrations for existing DBs ---------------------------------
        def _col_exists(table: str, col: str) -> bool:
            try:
                rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
                return any(r[1] == col for r in rows)
            except Exception:
                return False

        def _add_col(table: str, coldef: str, colname: str) -> None:
            if _col_exists(table, colname):
                return
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {coldef}")
                logger.info("Schema migration: added %s.%s", table, colname)
            except sqlite3.OperationalError as exc:
                # Column may already exist due to concurrent migration.
                logger.debug("Schema migration skipped for %s.%s: %s", table, colname, exc)

        # B12 patterns: strength + last_accessed_at
        _add_col("memories", "strength REAL DEFAULT 1.0", "strength")
        _add_col("memories", "last_accessed_at REAL", "last_accessed_at")
        # Consolidation: soft-delete
        _add_col("memories", "deleted_at REAL", "deleted_at")
        # Critical memory pinning: pinned memories survive decay/gc/consolidation
        _add_col("memories", "pinned INTEGER DEFAULT 0", "pinned")
        _add_col("memories", "namespace TEXT DEFAULT 'default'", "namespace")
        _add_col("memories", "memory_type TEXT DEFAULT 'other'", "memory_type")
        _add_col("memories", "lesson_status TEXT DEFAULT 'active'", "lesson_status")
        _add_col("memories", "lesson_scope TEXT", "lesson_scope")
        _add_col("memories", "resolved_at REAL", "resolved_at")

        # Backfill last_accessed_at for existing rows (keep idempotent)
        try:
            conn.execute(
                "UPDATE memories SET last_accessed_at = created_at WHERE last_accessed_at IS NULL"
            )
        except Exception:
            pass

        # Migrate search_result_cache PK to include min_score (cache is ephemeral)
        try:
            cols = conn.execute("PRAGMA table_info(search_result_cache)").fetchall()
            if cols:
                pk_cols = [c[1] for c in cols if c[5] > 0]
                if "min_score" not in pk_cols and len(pk_cols) == 3:
                    conn.execute("DROP TABLE search_result_cache")
                    conn.execute("""CREATE TABLE IF NOT EXISTS search_result_cache (
                        query_norm TEXT NOT NULL,
                        limit_val INTEGER NOT NULL,
                        min_score REAL NOT NULL DEFAULT 0.0,
                        agent TEXT NOT NULL DEFAULT 'main',
                        results_json TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL,
                        PRIMARY KEY (query_norm, limit_val, min_score, agent)
                    )""")
                    logger.info("Schema migration: rebuilt search_result_cache with min_score in PK")
        except Exception as exc:
            logger.debug("search_result_cache migration skipped: %s", exc)

        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_lesson_status ON memories(lesson_status)")
        except Exception:
            pass

        conn.commit()

    # ------------------------------------------------------------------
    # Memory CRUD
    # ------------------------------------------------------------------

    def store_memory(
        self,
        text: str,
        vector: Optional[List[float]] = None,
        category: str = "other",
        importance: float = 0.5,
        source_session: Optional[str] = None,
        namespace: str = "default",
        memory_id: Optional[str] = None,
        memory_type: str = "other",
    ) -> str:
        """Insert a memory. Returns the memory ID."""
        conn = self._get_conn()
        mid = memory_id or uuid.uuid4().hex[:16]
        now = time.time()

        vector_rowid: Optional[int] = None
        if vector is not None:
            blob = np.array(vector, dtype=np.float32).tobytes()
            cur = conn.execute(
                "INSERT INTO memory_vectors(embedding) VALUES (?)", (blob,)
            )
            vector_rowid = cur.lastrowid

        conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, text, category, memory_type, importance, source_session, namespace,
                created_at, updated_at, vector_rowid,
                strength, last_accessed_at, deleted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
            (
                mid,
                text,
                category,
                memory_type,
                importance,
                source_session,
                namespace,
                now,
                now,
                vector_rowid,
                1.0,
                now,
            ),
        )

        # FTS5 sync
        conn.execute(
            "INSERT OR REPLACE INTO memory_fts(id, text) VALUES (?, ?)",
            (mid, text),
        )
        conn.commit()
        return mid

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Return a single memory dict or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return dict(row) if row else None

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory and its vector/FTS entries. Returns True if found."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT vector_rowid FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return False

        vector_rowid = row["vector_rowid"]
        if vector_rowid is not None:
            conn.execute(
                "DELETE FROM memory_vectors WHERE rowid = ?", (vector_rowid,)
            )
        conn.execute("DELETE FROM memory_fts WHERE id = ?", (memory_id,))
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        return True

    def update_memory(
        self,
        memory_id: str,
        text: Optional[str] = None,
        category: Optional[str] = None,
        importance: Optional[float] = None,
        vector: Optional[List[float]] = None,
    ) -> bool:
        """Update fields of an existing memory. Returns True if found."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return False

        updates: list[str] = []
        params: list[Any] = []
        if text is not None:
            updates.append("text = ?")
            params.append(text)
            # update FTS
            conn.execute("DELETE FROM memory_fts WHERE id = ?", (memory_id,))
            conn.execute(
                "INSERT INTO memory_fts(id, text) VALUES (?, ?)", (memory_id, text)
            )
        if category is not None:
            updates.append("category = ?")
            params.append(category)
        if importance is not None:
            updates.append("importance = ?")
            params.append(importance)
        if vector is not None:
            old_rowid = row["vector_rowid"]
            blob = np.array(vector, dtype=np.float32).tobytes()
            if old_rowid:
                conn.execute(
                    "UPDATE memory_vectors SET embedding = ? WHERE rowid = ?",
                    (blob, old_rowid),
                )
            else:
                cur = conn.execute(
                    "INSERT INTO memory_vectors(embedding) VALUES (?)", (blob,)
                )
                updates.append("vector_rowid = ?")
                params.append(cur.lastrowid)

        updates.append("updated_at = ?")
        params.append(time.time())
        params.append(memory_id)

        conn.execute(
            f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params
        )
        conn.commit()
        return True

    # ------------------------------------------------------------------
    # Ebbinghaus strength / spaced repetition
    # ------------------------------------------------------------------

    def pin_memory(self, memory_id: str) -> bool:
        """Pin a memory: protects it from decay, gc, and consolidation."""
        conn = self._get_conn()
        try:
            cur = conn.execute(
                "UPDATE memories SET pinned = 1 WHERE id = ? AND deleted_at IS NULL",
                (memory_id,),
            )
            conn.commit()
            return cur.rowcount > 0
        except Exception as exc:
            logger.debug("pin_memory failed for %s: %s", memory_id, exc)
            return False

    def unpin_memory(self, memory_id: str) -> bool:
        """Unpin a memory: allows decay/gc/consolidation again."""
        conn = self._get_conn()
        try:
            cur = conn.execute(
                "UPDATE memories SET pinned = 0 WHERE id = ? AND deleted_at IS NULL",
                (memory_id,),
            )
            conn.commit()
            return cur.rowcount > 0
        except Exception as exc:
            logger.debug("unpin_memory failed for %s: %s", memory_id, exc)
            return False

    def boost_strength(self, memory_id: str) -> bool:
        """Boost strength on retrieval (spaced repetition).

        strength = min(strength + 0.3, 5.0)
        last_accessed_at = now
        """
        conn = self._get_conn()
        now = time.time()
        try:
            cur = conn.execute(
                """
                UPDATE memories
                   SET strength = MIN(COALESCE(strength, 1.0) + 0.3, 5.0),
                       last_accessed_at = ?
                 WHERE id = ? AND deleted_at IS NULL
                """,
                (now, memory_id),
            )
            conn.commit()
            return cur.rowcount > 0
        except Exception as exc:
            logger.debug("boost_strength failed for %s: %s", memory_id, exc)
            return False

    def decay_all(
        self,
        days_threshold: int = 7,
        decay_amount: float = 0.05,
        min_strength: float = 0.3,
    ) -> int:
        """Aggressive Ebbinghaus decay + boost with importance-adjusted rate.

        - 30+ days unaccessed -> drop to floor (importance-adjusted)
        - < 7 days old -> protect (min 0.8)
        - Very recent (last 24h) -> boost (+0.1)
        - 7-30 days -> importance-adjusted decay (low importance decays faster)
        - GC phase: soft-delete zombies at floor with low importance, unaccessed 14+ days
        """
        conn = self._get_conn()
        now = time.time()
        try:
            # Phase 1: Importance-adjusted decay
            # decay_multiplier = 1.0 + (1.0 - importance)
            # importance=0.2 -> 1.8x decay, importance=0.8 -> 1.2x decay
            cur = conn.execute(
                """
                UPDATE memories
                   SET strength = CASE
                       -- 1. Very stale (30+ days): Drop to floor
                       WHEN (COALESCE(last_accessed_at, created_at) < ? - 2592000)
                            THEN ?

                       -- 2. Very recent (last 24h): Small boost
                       WHEN (COALESCE(last_accessed_at, created_at) >= ? - 86400)
                            THEN MIN(COALESCE(strength, 1.0) + 0.1, 5.0)

                       -- 3. Recent (7-day window): Maintain high (min 0.8)
                       WHEN (COALESCE(last_accessed_at, created_at) >= ? - 604800)
                            THEN MAX(COALESCE(strength, 1.0), 0.8)

                       -- 4. Mid-stale (7-30 days): Importance-adjusted decay
                       ELSE MAX(
                           COALESCE(strength, 1.0) - (0.15 * (1.0 + (1.0 - COALESCE(importance, 0.5)))),
                           ?
                       )
                   END
                 WHERE deleted_at IS NULL
                   AND COALESCE(pinned, 0) = 0
                   AND COALESCE(memory_type, 'other') != 'lesson'
                   AND (
                       (COALESCE(last_accessed_at, created_at) < ? - 604800) -- stale
                       OR
                       (COALESCE(last_accessed_at, created_at) >= ? - 86400) -- very recent (for boost)
                   )
                """,
                (now, min_strength, now, now, min_strength, now, now),
            )
            conn.commit()
            decayed = int(cur.rowcount or 0)
            logger.info(
                "Decay phase 1: updated=%d (min_strength=%.2f, importance-adjusted)",
                decayed,
                min_strength,
            )

            # Phase 1b: Lesson decay (3x slower — lessons survive much longer)
            # 90+ days stale -> floor, 21-day protect window, 21-90 days -> gentle decay
            lesson_cur = conn.execute(
                """
                UPDATE memories
                   SET strength = CASE
                       WHEN (COALESCE(last_accessed_at, created_at) < ? - 7776000)
                            THEN ?
                       WHEN (COALESCE(last_accessed_at, created_at) >= ? - 86400)
                            THEN MIN(COALESCE(strength, 1.0) + 0.1, 5.0)
                       WHEN (COALESCE(last_accessed_at, created_at) >= ? - 1814400)
                            THEN MAX(COALESCE(strength, 1.0), 0.8)
                       ELSE MAX(
                           COALESCE(strength, 1.0) - 0.05,
                           ?
                       )
                   END
                 WHERE deleted_at IS NULL
                   AND COALESCE(pinned, 0) = 0
                   AND COALESCE(memory_type, 'other') = 'lesson'
                   AND (
                       (COALESCE(last_accessed_at, created_at) < ? - 1814400)
                       OR
                       (COALESCE(last_accessed_at, created_at) >= ? - 86400)
                   )
                """,
                (now, min_strength, now, now, min_strength, now, now),
            )
            conn.commit()
            lesson_decayed = int(lesson_cur.rowcount or 0)
            if lesson_decayed:
                logger.info("Decay phase 1b: lesson decay=%d (3x slower rate)", lesson_decayed)

            # Phase 2: GC — soft-delete zombies at strength floor with low importance,
            # unaccessed for 14+ days
            gc_cur = conn.execute(
                """
                UPDATE memories
                   SET deleted_at = CAST(strftime('%s', 'now') AS INTEGER)
                 WHERE deleted_at IS NULL
                   AND COALESCE(pinned, 0) = 0
                   AND COALESCE(memory_type, 'other') != 'lesson'
                   AND strength <= ?
                   AND importance <= 0.3
                   AND COALESCE(last_accessed_at, created_at) < ? - 1209600
                """,
                (min_strength, now),
            )
            conn.commit()
            gc_count = int(gc_cur.rowcount or 0)
            if gc_count > 0:
                logger.info("Decay GC phase: soft-deleted %d zombie memories", gc_count)

            return decayed + gc_count
        except Exception as exc:
            logger.exception("decay_all failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # Garbage collection — permanent delete
    # ------------------------------------------------------------------

    def gc_purge(self, soft_deleted_days: int = 30) -> Dict[str, int]:
        """Permanently DELETE memories soft-deleted for *soft_deleted_days*+.

        Also cleans orphaned vectors whose memories no longer exist.
        Returns dict with counts: purged_memories, purged_vectors.
        """
        conn = self._get_conn()
        now = time.time()
        cutoff = now - (soft_deleted_days * 86400)
        try:
            # Find memories to permanently delete
            rows = conn.execute(
                "SELECT id, vector_rowid FROM memories WHERE deleted_at IS NOT NULL AND deleted_at < ?",
                (cutoff,),
            ).fetchall()

            mem_ids = [r["id"] for r in rows]
            vec_rowids = [r["vector_rowid"] for r in rows if r["vector_rowid"] is not None]

            if not mem_ids:
                return {"purged_memories": 0, "purged_vectors": 0}

            # Delete from FTS
            for mid in mem_ids:
                try:
                    conn.execute("DELETE FROM memory_fts WHERE id = ?", (mid,))
                except Exception:
                    pass

            # Delete from memories table
            placeholders = ",".join("?" * len(mem_ids))
            conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", mem_ids)

            # Delete orphaned vectors
            purged_vecs = 0
            for vr in vec_rowids:
                try:
                    conn.execute("DELETE FROM memory_vectors WHERE rowid = ?", (vr,))
                    purged_vecs += 1
                except Exception:
                    pass

            conn.commit()
            logger.info(
                "GC purge: permanently deleted %d memories, %d vectors",
                len(mem_ids), purged_vecs,
            )
            return {"purged_memories": len(mem_ids), "purged_vectors": purged_vecs}
        except Exception as exc:
            conn.rollback()
            logger.exception("gc_purge failed: %s", exc)
            return {"purged_memories": 0, "purged_vectors": 0}

    # ------------------------------------------------------------------
    # Write-time semantic merge
    # ------------------------------------------------------------------

    def merge_or_store(
        self,
        text: str,
        vector: Optional[List[float]],
        category: str,
        importance: float,
        source_session: Optional[str],
        similarity_threshold: float = 0.85,
        memory_type: str = "other",
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Store a memory, merging into nearest neighbor when highly similar.

        If *vector* is provided, we search the nearest neighbor via sqlite-vec.
        If similarity > threshold AND category matches, we merge content and
        average embeddings.
        """
        conn = self._get_conn()
        now = time.time()

        if memory_type == "lesson":
            mid = self.store_memory(
                text=text,
                vector=vector,
                category=category,
                importance=importance,
                source_session=source_session,
                memory_type=memory_type,
                namespace=namespace,
            )
            return {"action": "inserted", "id": mid, "similarity": None}

        if vector is None:
            mid = self.store_memory(
                text=text,
                vector=None,
                category=category,
                importance=importance,
                source_session=source_session,
                memory_type=memory_type,
                namespace=namespace,
            )
            return {"action": "inserted", "id": mid, "similarity": None}

        try:
            nn = self.search_vectors(vector, limit=1, min_score=0.0, namespace=namespace)
        except Exception as exc:
            logger.warning("merge_or_store: vector search failed; inserting: %s", exc)
            mid = self.store_memory(
                text=text,
                vector=vector,
                category=category,
                importance=importance,
                source_session=source_session,
                memory_type=memory_type,
                namespace=namespace,
            )
            return {"action": "inserted", "id": mid, "similarity": None}

        if not nn:
            mid = self.store_memory(
                text=text,
                vector=vector,
                category=category,
                importance=importance,
                source_session=source_session,
                memory_type=memory_type,
                namespace=namespace,
            )
            return {"action": "inserted", "id": mid, "similarity": None}

        best = nn[0]
        best_id = best.get("id")
        similarity = float(best.get("score") or 0.0)

        if (
            best_id
            and similarity >= similarity_threshold
            and (best.get("category") or "other") == category
            and best.get("deleted_at") is None
        ):
            old_text = (best.get("text") or "").rstrip()
            new_text = text.strip()
            merged_text = f"{old_text}\n• {new_text}" if old_text else new_text

            # Average embeddings (old + new)
            vec_rowid = best.get("vector_rowid")
            if vec_rowid is not None:
                row = conn.execute(
                    "SELECT embedding FROM memory_vectors WHERE rowid = ?", (vec_rowid,)
                ).fetchone()
                if row and row[0] is not None:
                    try:
                        old_vec = np.frombuffer(row[0], dtype=np.float32)
                        new_vec = np.asarray(vector, dtype=np.float32)
                        if old_vec.shape == new_vec.shape:
                            avg_vec = ((old_vec + new_vec) / 2.0).astype(np.float32)
                            conn.execute(
                                "UPDATE memory_vectors SET embedding = ? WHERE rowid = ?",
                                (avg_vec.tobytes(), vec_rowid),
                            )
                    except Exception:
                        pass

            # Merge text + metadata
            conn.execute(
                """
                UPDATE memories
                   SET text = ?,
                       importance = MAX(COALESCE(importance, 0.5), ?),
                       updated_at = ?,
                       strength = MIN(COALESCE(strength, 1.0) + 0.2, 5.0),
                       last_accessed_at = ?
                 WHERE id = ?
                """,
                (merged_text, importance, now, now, best_id),
            )
            conn.execute("DELETE FROM memory_fts WHERE id = ?", (best_id,))
            conn.execute(
                "INSERT OR REPLACE INTO memory_fts(id, text) VALUES (?, ?)",
                (best_id, merged_text),
            )
            conn.commit()

            logger.info(
                "merge_or_store: merged into %s (sim=%.4f cat=%s)",
                best_id,
                similarity,
                category,
            )
            return {"action": "merged", "id": best_id, "similarity": similarity}

        mid = self.store_memory(
            text=text,
            vector=vector,
            category=category,
            importance=importance,
            source_session=source_session,
            memory_type=memory_type,
            namespace=namespace,
        )
        return {"action": "inserted", "id": mid, "similarity": similarity}

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def store_memories_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[str]:
        """Store many memories in a single transaction.

        Each item dict should have at least ``text``; optional keys:
        ``vector``, ``category``, ``memory_type``, ``importance``, ``source_session``, ``id``.
        """
        conn = self._get_conn()
        ids: List[str] = []
        now = time.time()

        try:
            for item in items:
                mid = item.get("id") or uuid.uuid4().hex[:16]
                text = item["text"]
                vector = item.get("vector")
                category = item.get("category", "other")
                memory_type = item.get("memory_type", "other")
                importance = item.get("importance", 0.5)
                source = item.get("source_session")
                namespace = item.get("namespace", "default")

                vector_rowid: Optional[int] = None
                if vector is not None:
                    blob = np.array(vector, dtype=np.float32).tobytes()
                    cur = conn.execute(
                        "INSERT INTO memory_vectors(embedding) VALUES (?)", (blob,)
                    )
                    vector_rowid = cur.lastrowid

                conn.execute(
                    """INSERT OR REPLACE INTO memories
                       (id, text, category, memory_type, importance, source_session, namespace,
                        created_at, updated_at, vector_rowid,
                        strength, last_accessed_at, deleted_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
                    (
                        mid,
                        text,
                        category,
                        memory_type,
                        importance,
                        source,
                        namespace,
                        now,
                        now,
                        vector_rowid,
                        1.0,
                        now,
                    ),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO memory_fts(id, text) VALUES (?, ?)",
                    (mid, text),
                )
                ids.append(mid)

            conn.commit()
        except Exception:
            conn.rollback()
            raise

        return ids

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        min_score: float = 0.0,
        namespace: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Nearest-neighbour search via sqlite-vec (cosine distance).

        Returns dicts with keys: ``id``, ``text``, ``category``, ``importance``,
        ``created_at``, ``score`` (cosine similarity ∈ [0, 1]).
        """
        conn = self._get_conn()
        blob = np.array(query_vector, dtype=np.float32).tobytes()

        rows = conn.execute(
            """
            SELECT
                mv.rowid   AS vec_rowid,
                mv.distance AS distance
            FROM memory_vectors mv
            WHERE mv.embedding MATCH ?
            ORDER BY mv.distance
            LIMIT ?
            """,
            (blob, limit),
        ).fetchall()

        results: List[Dict[str, Any]] = []
        for r in rows:
            # sqlite-vec returns cosine *distance* (0 = identical, 2 = opposite)
            similarity = 1.0 - (r["distance"] / 2.0)
            if similarity < min_score:
                continue
            where = ["vector_rowid = ?", "deleted_at IS NULL", "importance >= 0.3"]
            params: List[Any] = [r["vec_rowid"]]
            if namespace is not None:
                where.append("namespace = ?")
                params.append(namespace)
            if memory_type is not None:
                where.append("COALESCE(memory_type, 'other') = ?")
                params.append(memory_type)
                if memory_type == "lesson":
                    where.append("COALESCE(lesson_status, 'active') = 'active'")
            mem = conn.execute(
                f"SELECT * FROM memories WHERE {' AND '.join(where)}",
                tuple(params),
            ).fetchone()
            if mem:
                d = dict(mem)
                d["score"] = round(similarity, 4)
                results.append(d)

        return results

    # ------------------------------------------------------------------
    # Full-text search
    # ------------------------------------------------------------------

    def search_text(
        self, query: str, limit: int = 10, namespace: Optional[str] = None, memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """FTS5 search. Returns memories sorted by BM25 relevance."""
        conn = self._get_conn()

        # Escape the query for FTS5 — wrap each token in double quotes
        safe_query = " OR ".join(
            f'"{tok}"' for tok in query.split() if tok.strip()
        )
        if not safe_query:
            return []

        rows = conn.execute(
            """
            SELECT fts.id, bm25(memory_fts) AS rank
            FROM memory_fts fts
            WHERE memory_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (safe_query, limit),
        ).fetchall()

        results: List[Dict[str, Any]] = []
        for r in rows:
            where = ["id = ?", "deleted_at IS NULL", "importance >= 0.3"]
            params: List[Any] = [r["id"]]
            if namespace is not None:
                where.append("namespace = ?")
                params.append(namespace)
            if memory_type is not None:
                where.append("COALESCE(memory_type, 'other') = ?")
                params.append(memory_type)
                if memory_type == "lesson":
                    where.append("COALESCE(lesson_status, 'active') = 'active'")
            mem = conn.execute(
                f"SELECT * FROM memories WHERE {' AND '.join(where)}",
                tuple(params),
            ).fetchone()
            if mem:
                d = dict(mem)
                d["bm25_rank"] = r["rank"]
                results.append(d)

        return results

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def store_entity(
        self,
        name: str,
        entity_type: str = "unknown",
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entity_id: Optional[str] = None,
    ) -> str:
        """Insert or update an entity. Returns entity ID."""
        conn = self._get_conn()
        now = time.time()
        normalized = name.lower().strip()

        # Try to find existing by normalized name + type
        existing = conn.execute(
            "SELECT id, mention_count FROM entities WHERE lower(name) = ? AND type = ?",
            (normalized, entity_type),
        ).fetchone()

        if existing:
            eid = existing["id"]
            conn.execute(
                """UPDATE entities
                   SET last_seen = ?, mention_count = mention_count + 1
                   WHERE id = ?""",
                (now, eid),
            )
            conn.commit()
            return eid

        eid = entity_id or uuid.uuid4().hex[:16]
        conn.execute(
            """INSERT INTO entities (id, name, type, aliases, first_seen, last_seen,
                                     mention_count, metadata)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?)""",
            (
                eid,
                name,
                entity_type,
                json.dumps(aliases or []),
                now,
                now,
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()
        return eid

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Return entity dict or None."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
        if row:
            d = dict(row)
            d["aliases"] = json.loads(d.get("aliases") or "[]")
            d["metadata"] = json.loads(d.get("metadata") or "{}")
            return d
        return None

    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities by name (LIKE)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM entities WHERE lower(name) LIKE ? ORDER BY mention_count DESC LIMIT ?",
            (f"%{query.lower()}%", limit),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["aliases"] = json.loads(d.get("aliases") or "[]")
            d["metadata"] = json.loads(d.get("metadata") or "{}")
            results.append(d)
        return results

    def link_entities(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "mentioned_with",
        confidence: float = 0.5,
        context: str = "",
    ) -> str:
        """Create or strengthen a relationship between two entities."""
        conn = self._get_conn()
        now = time.time()

        existing = conn.execute(
            """SELECT id, confidence FROM relationships
               WHERE source_id = ? AND target_id = ? AND relation_type = ?""",
            (source_id, target_id, relation_type),
        ).fetchone()

        if existing:
            # Strengthen
            new_conf = min(1.0, existing["confidence"] + 0.1)
            conn.execute(
                "UPDATE relationships SET confidence = ?, context = ? WHERE id = ?",
                (new_conf, context, existing["id"]),
            )
            conn.commit()
            return existing["id"]

        rid = uuid.uuid4().hex[:16]
        conn.execute(
            """INSERT INTO relationships (id, source_id, target_id, relation_type,
                                          confidence, context, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (rid, source_id, target_id, relation_type, confidence, context, now),
        )
        conn.commit()
        return rid

    def store_temporal_fact(
        self,
        entity_id: str,
        fact: str,
        valid_from: Optional[float] = None,
        valid_to: Optional[float] = None,
        source_memory_id: Optional[str] = None,
    ) -> str:
        """Store a temporal fact for an entity."""
        conn = self._get_conn()
        fid = uuid.uuid4().hex[:16]
        conn.execute(
            """INSERT INTO temporal_facts (id, entity_id, fact, valid_from, valid_to, source_memory_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (fid, entity_id, fact, valid_from or time.time(), valid_to, source_memory_id),
        )
        conn.commit()
        return fid

    def get_entity_facts(self, entity_id: str, current_only: bool = True) -> List[Dict[str, Any]]:
        """Get temporal facts for an entity."""
        conn = self._get_conn()
        if current_only:
            rows = conn.execute(
                """SELECT * FROM temporal_facts
                   WHERE entity_id = ? AND (valid_to IS NULL OR valid_to > ?)
                   ORDER BY valid_from DESC""",
                (entity_id, time.time()),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM temporal_facts WHERE entity_id = ? ORDER BY valid_from DESC",
                (entity_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def cache_search_result(
        self,
        query_norm: str,
        limit_val: int,
        min_score: float,
        agent: str,
        results_json: str,
        ttl: int = 3600,
    ) -> None:
        """Store a search result in the cache."""
        conn = self._get_conn()
        now = time.time()
        expires_at = now + ttl
        conn.execute(
            """INSERT OR REPLACE INTO search_result_cache
               (query_norm, limit_val, min_score, agent, results_json, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (query_norm, limit_val, min_score, agent, results_json, now, expires_at),
        )
        conn.commit()

    def get_cached_search_result(
        self, query_norm: str, limit_val: int, min_score: float, agent: str
    ) -> Optional[str]:
        """Retrieve a cached search result if not expired."""
        conn = self._get_conn()
        now = time.time()
        row = conn.execute(
            """SELECT results_json FROM search_result_cache
               WHERE query_norm = ? AND limit_val = ? AND min_score = ? AND agent = ?
                 AND expires_at > ?""",
            (query_norm, limit_val, min_score, agent, now),
        ).fetchone()
        return row["results_json"] if row else None

    def invalidate_search_cache(self, agent: Optional[str] = None) -> None:
        """Delete expired cache entries, or all entries for a specific agent."""
        conn = self._get_conn()
        now = time.time()
        if agent:
            conn.execute("DELETE FROM search_result_cache WHERE agent = ?", (agent,))
        else:
            conn.execute("DELETE FROM search_result_cache WHERE expires_at <= ?", (now,))
        conn.commit()

    def cache_embedding(self, text_hash: str, embedding_blob: bytes) -> None:
        """Store an embedding in the persistent cache."""
        conn = self._get_conn()
        now = time.time()
        conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (text_hash, embedding, created_at) VALUES (?, ?, ?)",
            (text_hash, embedding_blob, now),
        )
        conn.commit()

    def get_cached_embedding(self, text_hash: str) -> Optional[bytes]:
        """Retrieve a cached embedding blob."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT embedding FROM embedding_cache WHERE text_hash = ?", (text_hash,)
        ).fetchone()
        return row["embedding"] if row else None

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return database statistics."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"]
        by_cat = conn.execute(
            "SELECT category, COUNT(*) AS c FROM memories GROUP BY category"
        ).fetchall()
        entity_count = conn.execute("SELECT COUNT(*) AS c FROM entities").fetchone()["c"]
        rel_count = conn.execute("SELECT COUNT(*) AS c FROM relationships").fetchone()["c"]
        fact_count = conn.execute("SELECT COUNT(*) AS c FROM temporal_facts").fetchone()["c"]

        return {
            "total_memories": total,
            "by_category": {r["category"]: r["c"] for r in by_cat},
            "entities": entity_count,
            "relationships": rel_count,
            "temporal_facts": fact_count,
        }

    # ------------------------------------------------------------------
    # Temporal Facts & Typed Relations (Extensions)
    # ------------------------------------------------------------------

    def run_migrations(self) -> None:
        """Apply schema extensions for typed relations."""
        conn = self._get_conn()

        # Check if columns exist
        try:
            existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(temporal_facts)").fetchall()}
        except Exception:
            existing_cols = set()

        updates = []
        if "relation_type" not in existing_cols:
            updates.append("ALTER TABLE temporal_facts ADD COLUMN relation_type TEXT")
        if "object_entity_id" not in existing_cols:
            updates.append("ALTER TABLE temporal_facts ADD COLUMN object_entity_id TEXT")
        if "object_value" not in existing_cols:
            updates.append("ALTER TABLE temporal_facts ADD COLUMN object_value TEXT")
        if "confidence" not in existing_cols:
            updates.append("ALTER TABLE temporal_facts ADD COLUMN confidence REAL DEFAULT 0.7")
        if "is_active" not in existing_cols:
            updates.append("ALTER TABLE temporal_facts ADD COLUMN is_active INTEGER DEFAULT 1")

        for stmt in updates:
            try:
                conn.execute(stmt)
                logger.info(f"Migration applied: {stmt}")
            except sqlite3.OperationalError as e:
                logger.warning(f"Migration failed (maybe already exists): {e}")

        # Indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_temporal_entity_rel_active ON temporal_facts(entity_id, relation_type, is_active)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_temporal_valid_from ON temporal_facts(valid_from)")
        conn.commit()

    def store_typed_fact(
        self,
        entity_id: str,
        relation_type: str,
        object_value: str,
        confidence: float = 0.7,
        valid_from: Optional[float] = None,
        source_memory_id: Optional[str] = None,
        object_entity_id: Optional[str] = None
    ) -> str:
        """Store a structured temporal fact."""
        conn = self._get_conn()
        fid = uuid.uuid4().hex[:16]
        # Construct fact string for backward compatibility / display
        fact_text = f"{relation_type}: {object_value}"

        # Ensure migration runs if not already
        # self.run_migrations() # Assuming run in __init__

        conn.execute(
            """INSERT INTO temporal_facts
               (id, entity_id, fact, valid_from, valid_to, source_memory_id,
                relation_type, object_value, object_entity_id, confidence, is_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
            (fid, entity_id, fact_text, valid_from or time.time(), None, source_memory_id,
             relation_type, object_value, object_entity_id, confidence)
        )
        conn.commit()
        return fid

    def get_active_facts(self, entity_id: str, relation_type: str) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM temporal_facts
               WHERE entity_id = ? AND relation_type = ? AND is_active = 1""",
            (entity_id, relation_type)
        ).fetchall()
        return [dict(r) for r in rows]

    def deactivate_fact(self, fact_id: str, reason: str = "superseded") -> None:
        conn = self._get_conn()
        now = time.time()
        conn.execute(
            "UPDATE temporal_facts SET is_active = 0, valid_to = ? WHERE id = ?",
            (now, fact_id)
        )
        conn.commit()

    def get_conflicts(self, entity_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Identify conflicts (multiple active facts for same exclusive relation)."""
        conn = self._get_conn()

        query = """
            SELECT entity_id, relation_type, COUNT(*) as count
            FROM temporal_facts
            WHERE is_active = 1 AND relation_type IS NOT NULL
        """
        params = []
        if entity_id:
            query += " AND entity_id = ?"
            params.append(entity_id)

        query += """
            GROUP BY entity_id, relation_type
            HAVING count > 1
            LIMIT ?
        """
        params.append(limit)

        rows = conn.execute(query, tuple(params)).fetchall()

        results = []
        for r in rows:
            facts = conn.execute(
                "SELECT * FROM temporal_facts WHERE entity_id = ? AND relation_type = ? AND is_active = 1",
                (r["entity_id"], r["relation_type"])
            ).fetchall()
            results.append({
                "entity_id": r["entity_id"],
                "relation_type": r["relation_type"],
                "facts": [dict(f) for f in facts]
            })
        return results
