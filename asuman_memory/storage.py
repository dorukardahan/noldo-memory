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
from typing import Any, Dict, List, Optional, Tuple

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
    vector_rowid INTEGER
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

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
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
        memory_id: Optional[str] = None,
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
               (id, text, category, importance, source_session, created_at, updated_at, vector_rowid)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (mid, text, category, importance, source_session, now, now, vector_rowid),
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
    # Batch operations
    # ------------------------------------------------------------------

    def store_memories_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[str]:
        """Store many memories in a single transaction.

        Each item dict should have at least ``text``; optional keys:
        ``vector``, ``category``, ``importance``, ``source_session``, ``id``.
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
                importance = item.get("importance", 0.5)
                source = item.get("source_session")

                vector_rowid: Optional[int] = None
                if vector is not None:
                    blob = np.array(vector, dtype=np.float32).tobytes()
                    cur = conn.execute(
                        "INSERT INTO memory_vectors(embedding) VALUES (?)", (blob,)
                    )
                    vector_rowid = cur.lastrowid

                conn.execute(
                    """INSERT OR REPLACE INTO memories
                       (id, text, category, importance, source_session,
                        created_at, updated_at, vector_rowid)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (mid, text, category, importance, source, now, now, vector_rowid),
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
            mem = conn.execute(
                "SELECT * FROM memories WHERE vector_rowid = ?", (r["vec_rowid"],)
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
        self, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """FTS5 search. Returns memories sorted by BM25 relevance."""
        conn = self._get_conn()

        # Escape the query for FTS5 — wrap each token in double quotes
        safe_query = " ".join(
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
            mem = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (r["id"],)
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
