from __future__ import annotations

import asyncio
import importlib.util
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    module_name = f"_test_{name.replace('-', '_').replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_memory_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            category TEXT DEFAULT 'other',
            memory_type TEXT DEFAULT 'other',
            importance REAL DEFAULT 0.5,
            strength REAL DEFAULT 1.0,
            created_at REAL NOT NULL,
            updated_at REAL,
            last_accessed_at REAL,
            pinned INTEGER DEFAULT 0,
            source TEXT DEFAULT 'api',
            trust_level TEXT DEFAULT 'user',
            lesson_status TEXT,
            lesson_scope TEXT,
            resolved_at REAL,
            namespace TEXT DEFAULT 'default',
            original_text TEXT,
            source_session TEXT,
            vector_rowid INTEGER,
            deleted_at REAL
        )
        """
    )
    conn.execute("CREATE VIRTUAL TABLE memory_fts USING fts5(id, text)")


def insert_memory(conn: sqlite3.Connection, memory_id: str, text: str, source_session: str = "s1") -> None:
    now = time.time()
    conn.execute(
        """
        INSERT INTO memories (
            id, text, category, memory_type, importance, strength,
            created_at, updated_at, last_accessed_at, pinned, source,
            trust_level, namespace, source_session
        )
        VALUES (?, ?, 'user', 'conversation', 0.5, 1.0, ?, ?, ?, 0, 'api', 'user', 'default', ?)
        """,
        (memory_id, text, now, now, now, source_session),
    )


def test_reindex_preserves_live_vectors_when_later_embedding_batch_fails(tmp_path, monkeypatch):
    reindex = load_script("reindex_embeddings.py")
    db_path = tmp_path / "memory.sqlite"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    create_memory_schema(conn)
    conn.execute("CREATE VIRTUAL TABLE memory_vectors USING vec0(embedding float[3])")
    for memory_id, text, vec in (
        ("mem-1", "one", [1.0, 0.0, 0.0]),
        ("mem-2", "two", [0.0, 1.0, 0.0]),
    ):
        cur = conn.execute("INSERT INTO memory_vectors(embedding) VALUES (?)", (reindex.float_list_to_blob(vec),))
        insert_memory(conn, memory_id, text)
        conn.execute("UPDATE memories SET vector_rowid = ? WHERE id = ?", (cur.lastrowid, memory_id))
    before_refs = {
        row["id"]: row["vector_rowid"]
        for row in conn.execute("SELECT id, vector_rowid FROM memories ORDER BY id")
    }
    conn.commit()
    conn.close()

    calls = 0

    def fake_embed_batch(texts):
        nonlocal calls
        calls += 1
        if texts == ["test"]:
            return [[0.0, 0.0, 1.0]]
        if texts == ["one"]:
            return [[0.5, 0.0, 0.0]]
        raise RuntimeError("embedding endpoint failed mid-reindex")

    monkeypatch.setattr(reindex, "DB_PATH", str(db_path))
    monkeypatch.setattr(reindex, "EMBED_DIM", 3)
    monkeypatch.setattr(reindex, "MAX_CHARS", 100)
    monkeypatch.setattr(reindex, "embed_batch", fake_embed_batch)
    monkeypatch.setattr(sys, "argv", ["reindex_embeddings.py", "--batch-size", "1", "--sleep", "0"])

    with pytest.raises(SystemExit):
        reindex.main()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    after_refs = {
        row["id"]: row["vector_rowid"]
        for row in conn.execute("SELECT id, vector_rowid FROM memories ORDER BY id")
    }
    assert after_refs == before_refs
    assert conn.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0] == 2


def test_migrate_stale_dbs_uses_source_row_id_for_idempotency(tmp_path, monkeypatch):
    migrate_stale = load_script("migrate_stale_dbs.py")
    monkeypatch.setattr(migrate_stale, "BASE_DIR", tmp_path)

    src = sqlite3.connect(tmp_path / "memory-bureau.sqlite")
    src.row_factory = sqlite3.Row
    create_memory_schema(src)
    insert_memory(src, "src-1", "first", source_session="same-session")
    src.commit()
    src.close()

    dst = sqlite3.connect(tmp_path / "memory-agent-asuman.sqlite")
    create_memory_schema(dst)
    dst.commit()
    dst.close()

    assert migrate_stale.migrate("bureau", "agent-asuman") == 1

    src = sqlite3.connect(tmp_path / "memory-bureau.sqlite")
    insert_memory(src, "src-2", "second", source_session="same-session")
    src.commit()
    src.close()

    assert migrate_stale.migrate("bureau", "agent-asuman") == 1
    dst = sqlite3.connect(tmp_path / "memory-agent-asuman.sqlite")
    migrated_texts = {
        row[0]
        for row in dst.execute("SELECT text FROM memories WHERE source_session LIKE '[migrated-from:bureau:%' ORDER BY text")
    }
    assert migrated_texts == {"first", "second"}


def test_initial_load_skips_knowledge_graph_when_storage_batch_fails(tmp_path, monkeypatch):
    initial_load = load_script("initial_load.py")
    kg_calls: list[str] = []

    class FakeStorage:
        def __init__(self, *args, **kwargs):
            pass

        def get_memory(self, memory_id):
            return None

        def store_memories_batch(self, items):
            raise RuntimeError("storage unavailable")

        def stats(self):
            return {"total_memories": 0, "entities": 0, "relationships": 0, "by_category": {}}

        def close(self):
            pass

    class FakeKnowledgeGraph:
        def __init__(self, storage):
            pass

        def process_text(self, text, source=None, timestamp=None):
            kg_calls.append(text)

    chunk = SimpleNamespace(
        md5="chunk-1",
        text="Ahmet uses SQLite for memory",
        role="user",
        session_id="session-1",
        timestamp="2026-05-21T00:00:00Z",
    )
    cfg = SimpleNamespace(
        openrouter_api_key="",
        sessions_dir=str(tmp_path),
        db_path=str(tmp_path / "memory.sqlite"),
        embedding_model="test-model",
        embedding_dimensions=4,
        chunk_gap_hours=4,
    )
    session = tmp_path / "session.jsonl"
    session.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(initial_load, "load_config", lambda: cfg)
    monkeypatch.setattr(initial_load, "discover_sessions", lambda _path: [session])
    monkeypatch.setattr(initial_load, "parse_session_file", lambda _path, gap_hours=4: [chunk])
    monkeypatch.setattr(initial_load, "MemoryStorage", FakeStorage)
    monkeypatch.setattr(initial_load, "KnowledgeGraph", FakeKnowledgeGraph)
    monkeypatch.setattr(sys, "argv", ["initial_load.py", "--skip-embeddings", "--batch-size", "1"])

    asyncio.run(initial_load.main())

    assert kg_calls == []


def test_openclaw_sync_persists_state_when_modified_session_has_no_new_chunks(tmp_path, monkeypatch):
    openclaw_sync = load_script("openclaw_sync.py")
    state_file = tmp_path / "sync_state.json"
    session = tmp_path / "abc.jsonl"
    session.write_text("{}", encoding="utf-8")
    old_mtime = session.stat().st_mtime - 10
    state_file.write_text(
        json.dumps({"last_sync": None, "sessions_synced": {"abc": {"mtime": old_mtime, "chunks": 1}}, "total_synced": 0, "sync_count": 0}),
        encoding="utf-8",
    )

    conn = sqlite3.connect(tmp_path / "memory.sqlite")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO memories(id) VALUES ('chunk-1')")
    conn.commit()

    class FakeStorage:
        def _get_conn(self):
            return conn

    class FakePool:
        closed = False

        def __init__(self, *args, **kwargs):
            pass

        def get(self, agent_id):
            return FakeStorage()

        def close_all(self):
            self.closed = True

    chunk = SimpleNamespace(md5="chunk-1", text="same text", role="user", session_id="abc", timestamp="2026")
    cfg = SimpleNamespace(
        db_path=str(tmp_path / "memory.sqlite"),
        embedding_dimensions=4,
        openrouter_api_key="",
        chunk_gap_hours=4,
    )
    monkeypatch.setattr(openclaw_sync, "STATE_FILE", state_file)
    monkeypatch.setattr(openclaw_sync, "load_config", lambda: cfg)
    monkeypatch.setattr(openclaw_sync, "discover_all_agent_sessions", lambda: {"main": [session]})
    monkeypatch.setattr(openclaw_sync, "parse_session_file", lambda _path, gap_hours=4: [chunk])
    monkeypatch.setattr(openclaw_sync, "StoragePool", FakePool)

    stats = asyncio.run(openclaw_sync.sync(SimpleNamespace(db=None, sessions_dir=None, skip_embeddings=True, full=False)))

    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert stats["status"] == "up_to_date"
    assert saved["sessions_synced"]["abc"]["mtime"] == session.stat().st_mtime


def test_export_to_workspace_creates_missing_memory_directory(tmp_path):
    db_path = tmp_path / "memory.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            text TEXT,
            category TEXT,
            importance REAL,
            strength REAL,
            created_at REAL,
            deleted_at REAL
        )
        """
    )
    conn.execute("CREATE TABLE entities (name TEXT, type TEXT, mention_count INTEGER, aliases TEXT)")
    conn.commit()
    conn.close()

    workspace = tmp_path / "workspace"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export-to-workspace.py")],
        env={
            **os_environ_without_secrets(),
            "AGENT_MEMORY_DB": str(db_path),
            "OPENCLAW_WORKSPACE": str(workspace),
        },
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert (workspace / "memory" / "memory-export.md").is_file()


def os_environ_without_secrets() -> dict[str, str]:
    import os

    blocked = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    return {k: v for k, v in os.environ.items() if not any(part in k.upper() for part in blocked)}


@pytest.mark.asyncio
async def test_lifespan_cancels_warmup_task_on_shutdown(tmp_path, monkeypatch):
    from agent_memory import api
    from agent_memory.config import Config

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def fake_warmup_loop():
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    cfg = Config(
        db_path=str(tmp_path / "memory.sqlite"),
        openrouter_api_key="",
        reranker_enabled=False,
        reranker_prewarm=False,
        reranker_two_pass_enabled=False,
        embed_worker_enabled=False,
    )
    monkeypatch.setattr(api, "load_config", lambda: cfg)
    monkeypatch.setattr(api, "warmup_loop", fake_warmup_loop)

    async with api.lifespan(api.app):
        await asyncio.wait_for(started.wait(), timeout=1)

    was_cancelled_before_cleanup = cancelled.is_set()
    dangling = [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]
    for task in dangling:
        if task.get_coro().__name__ == "fake_warmup_loop":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert was_cancelled_before_cleanup


def test_embed_worker_skips_stale_vectorless_snapshot_without_orphan_vector(tmp_path):
    from agent_memory.embed_worker import EmbedWorker

    conn = sqlite3.connect(tmp_path / "memory.sqlite")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL,
            vector_rowid INTEGER,
            deleted_at REAL
        )
        """
    )
    conn.execute("CREATE TABLE memory_vectors (embedding BLOB NOT NULL)")
    conn.execute("CREATE TABLE memory_index_jobs (memory_id TEXT)")
    conn.execute("INSERT INTO memories(id, text, created_at) VALUES ('mem-1', 'text', ?)", (time.time(),))
    conn.commit()

    class FakeStorage:
        def _get_conn(self):
            return conn

    worker = EmbedWorker(storage_pool=object(), embedder=object())

    assert worker._update_memory_vector(FakeStorage(), "mem-1", [1.0, 0.0]) is True
    assert worker._update_memory_vector(FakeStorage(), "mem-1", [0.0, 1.0]) is False
    assert conn.execute("SELECT COUNT(*) FROM memory_vectors").fetchone()[0] == 1
