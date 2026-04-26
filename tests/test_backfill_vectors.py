import asyncio
import sqlite3
import time

from scripts.backfill_vectors import backfill_agent, get_vectorless_memories, update_memory_vector


class FakeStorage:
    def __init__(self, conn):
        self._conn = conn

    def _get_conn(self):
        return self._conn


class FakeEmbedder:
    async def embed_batch_resilient(self, texts, max_sub_batch=2):
        return [[float(i), 0.0, 0.0] for i, _text in enumerate(texts, start=1)]


def _open_conn(path):
    conn = sqlite3.connect(path, timeout=0.01)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _init_db(path, count=1):
    conn = _open_conn(path)
    conn.execute(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            category TEXT DEFAULT 'other',
            importance REAL DEFAULT 0.5,
            source_session TEXT,
            created_at REAL NOT NULL,
            updated_at REAL,
            vector_rowid INTEGER,
            deleted_at REAL
        )
        """
    )
    conn.execute("CREATE TABLE memory_vectors (embedding BLOB NOT NULL)")
    now = time.time()
    for idx in range(count):
        conn.execute(
            """
            INSERT INTO memories (id, text, category, importance, source_session, created_at)
            VALUES (?, ?, 'other', 0.5, 'test', ?)
            """,
            (f"mem-{idx}", f"text {idx}", now - idx),
        )
    conn.commit()
    return conn


def test_update_memory_vector_writes_once(tmp_path):
    conn = _init_db(tmp_path / "memory.sqlite", count=1)
    storage = FakeStorage(conn)

    status = update_memory_vector(storage, "mem-0", [1.0, 2.0, 3.0])

    assert status == "succeeded"
    row = conn.execute("SELECT vector_rowid FROM memories WHERE id = 'mem-0'").fetchone()
    assert row["vector_rowid"] is not None
    assert conn.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0] == 1


def test_update_memory_vector_skips_already_updated_row(tmp_path):
    conn = _init_db(tmp_path / "memory.sqlite", count=1)
    storage = FakeStorage(conn)

    assert update_memory_vector(storage, "mem-0", [1.0]) == "succeeded"
    assert update_memory_vector(storage, "mem-0", [2.0]) == "skipped"

    assert conn.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0] == 1


def test_update_memory_vector_returns_locked_after_bounded_retries(tmp_path):
    db_path = tmp_path / "memory.sqlite"
    holder = _init_db(db_path, count=1)
    writer = _open_conn(db_path)
    storage = FakeStorage(writer)

    holder.execute("BEGIN IMMEDIATE")
    try:
        status = update_memory_vector(
            storage,
            "mem-0",
            [1.0],
            max_attempts=2,
            retry_base_sleep=0.01,
        )
    finally:
        holder.rollback()

    assert status == "locked"
    row = writer.execute("SELECT vector_rowid FROM memories WHERE id = 'mem-0'").fetchone()
    assert row["vector_rowid"] is None


def test_backfill_agent_stops_on_persistent_lock_without_runaway(tmp_path):
    db_path = tmp_path / "memory.sqlite"
    holder = _init_db(db_path, count=3)
    writer = _open_conn(db_path)
    storage = FakeStorage(writer)

    holder.execute("BEGIN IMMEDIATE")
    try:
        stats = asyncio.run(
            backfill_agent(
                agent_id="main",
                storage=storage,
                embedder=FakeEmbedder(),
                batch_size=2,
                max_sub_batch=2,
                sleep_between_batches=0,
                max_write_attempts=1,
                write_retry_base_sleep=0.01,
            )
        )
    finally:
        holder.rollback()

    assert stats["processed"] == 1
    assert stats["locked"] == 1
    assert stats["failed"] == 1
    assert stats["succeeded"] == 0


def test_backfill_agent_processes_all_successful_batches(tmp_path):
    conn = _init_db(tmp_path / "memory.sqlite", count=5)
    storage = FakeStorage(conn)

    stats = asyncio.run(
        backfill_agent(
            agent_id="main",
            storage=storage,
            embedder=FakeEmbedder(),
            batch_size=2,
            max_sub_batch=2,
            sleep_between_batches=0,
        )
    )

    assert stats["processed"] == 5
    assert stats["succeeded"] == 5
    assert stats["failed"] == 0
    assert len(get_vectorless_memories(storage, batch_size=10)) == 0


def test_backfill_agent_dry_run_does_not_repeat_same_batch(tmp_path):
    conn = _init_db(tmp_path / "memory.sqlite", count=5)
    storage = FakeStorage(conn)

    stats = asyncio.run(
        backfill_agent(
            agent_id="main",
            storage=storage,
            embedder=FakeEmbedder(),
            batch_size=2,
            sleep_between_batches=0,
            dry_run=True,
        )
    )

    assert stats["processed"] == 5
    assert stats["skipped"] == 5
    assert len(get_vectorless_memories(storage, batch_size=10)) == 5
