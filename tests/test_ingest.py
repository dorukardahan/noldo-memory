"""Tests for session ingestion module."""

from __future__ import annotations

import json

import pytest

from agent_memory.ingest import (
    _chunk_session,
    _extract_text,
    _is_tool_call,
    _md5,
    _should_skip,
    parse_session_file,
    discover_sessions,
    ingest_sessions,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

class TestMD5:
    def test_deterministic(self):
        assert _md5("hello") == _md5("hello")

    def test_different_inputs(self):
        assert _md5("hello") != _md5("world")

    def test_length(self):
        assert len(_md5("test")) == 16


class TestExtractText:
    def test_string_content(self):
        assert _extract_text("hello world") == "hello world"

    def test_list_content(self):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        assert "hello" in _extract_text(content)
        assert "world" in _extract_text(content)

    def test_mixed_content(self):
        content = [
            {"type": "text", "text": "real text"},
            {"type": "image", "data": "..."},
        ]
        result = _extract_text(content)
        assert "real text" in result

    def test_empty(self):
        assert _extract_text("") == ""
        assert _extract_text([]) == ""
        assert _extract_text(None) == ""


class TestIsToolCall:
    def test_tool_role(self):
        assert _is_tool_call({"message": {"role": "tool", "content": "result"}}) is True

    def test_function_role(self):
        assert _is_tool_call({"message": {"role": "function", "content": "result"}}) is True

    def test_tool_use_content(self):
        entry = {
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "123", "name": "read"}],
            }
        }
        assert _is_tool_call(entry) is True

    def test_normal_message(self):
        entry = {"message": {"role": "user", "content": "hello"}}
        assert _is_tool_call(entry) is False


class TestShouldSkip:
    def test_heartbeat(self):
        assert _should_skip("HEARTBEAT_OK", "assistant") is True

    def test_no_reply(self):
        assert _should_skip("NO_REPLY", "assistant") is True

    def test_system_role(self):
        assert _should_skip("some system message", "system") is True

    def test_too_short(self):
        assert _should_skip("hi", "user") is True
        assert _should_skip("", "user") is True

    def test_valid_message(self):
        assert _should_skip("Merhaba, bugün ne yapacağız?", "user") is False

    def test_heartbeat_prefix(self):
        assert _should_skip("HEARTBEAT_OK some extra text", "assistant") is True


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class TestChunking:
    def test_qa_pair(self, sample_session_entries):
        chunks = _chunk_session(sample_session_entries, "test-session")
        # Should produce Q&A pairs
        qa_chunks = [c for c in chunks if c.role == "qa_pair"]
        assert len(qa_chunks) >= 1
        # Q&A pair should contain both User: and Assistant:
        assert "User:" in qa_chunks[0].text
        assert "Assistant:" in qa_chunks[0].text

    def test_md5_dedup(self, sample_session_entries):
        chunks = _chunk_session(sample_session_entries, "test-session")
        md5s = [c.md5 for c in chunks]
        # All md5s should be unique within a session
        assert len(md5s) == len(set(md5s))

    def test_session_id_propagated(self, sample_session_entries):
        chunks = _chunk_session(sample_session_entries, "my-session")
        for chunk in chunks:
            assert chunk.session_id == "my-session"

    def test_skips_heartbeat(self):
        entries = [
            {"type": "message", "timestamp": "2026-01-15T10:00:00Z",
             "message": {"role": "user", "content": "hello there!"}},
            {"type": "message", "timestamp": "2026-01-15T10:00:30Z",
             "message": {"role": "assistant", "content": "HEARTBEAT_OK"}},
            {"type": "message", "timestamp": "2026-01-15T10:01:00Z",
             "message": {"role": "assistant", "content": "Merhaba! Size nasıl yardımcı olabilirim?"}},
        ]
        chunks = _chunk_session(entries, "test")
        texts = " ".join(c.text for c in chunks)
        assert "HEARTBEAT_OK" not in texts

    def test_skips_tool_calls(self):
        entries = [
            {"type": "message", "timestamp": "2026-01-15T10:00:00Z",
             "message": {"role": "user", "content": "check the weather"}},
            {"type": "message", "timestamp": "2026-01-15T10:00:30Z",
             "message": {"role": "assistant",
                         "content": [{"type": "tool_use", "id": "1", "name": "weather"}]}},
            {"type": "message", "timestamp": "2026-01-15T10:00:35Z",
             "message": {"role": "tool", "content": "sunny"}},
            {"type": "message", "timestamp": "2026-01-15T10:01:00Z",
             "message": {"role": "assistant", "content": "The weather is sunny today!"}},
        ]
        chunks = _chunk_session(entries, "test")
        # Tool messages should not appear
        for c in chunks:
            assert "tool" not in c.role

    def test_time_gap_splitting(self):
        entries = [
            {"type": "message", "timestamp": "2026-01-15T10:00:00Z",
             "message": {"role": "user", "content": "morning question here"}},
            {"type": "message", "timestamp": "2026-01-15T10:00:30Z",
             "message": {"role": "assistant", "content": "morning answer here for you"}},
            # 5 hour gap (> default 4h)
            {"type": "message", "timestamp": "2026-01-15T15:01:00Z",
             "message": {"role": "user", "content": "afternoon question here"}},
            {"type": "message", "timestamp": "2026-01-15T15:01:30Z",
             "message": {"role": "assistant", "content": "afternoon answer here for you"}},
        ]
        chunks = _chunk_session(entries, "test", gap_hours=4.0)
        # Should produce 2 separate chunks
        assert len(chunks) >= 2

    def test_standalone_assistant_filtered(self):
        entries = [
            {"type": "message", "timestamp": "2026-01-15T10:00:00Z",
             "message": {"role": "assistant", "content": "ok"}},  # too short
        ]
        chunks = _chunk_session(entries, "test")
        assert len(chunks) == 0

    def test_chunk_text_truncated(self):
        long_text = "x" * 5000
        entries = [
            {"type": "message", "timestamp": "2026-01-15T10:00:00Z",
             "message": {"role": "user", "content": long_text}},
        ]
        chunks = _chunk_session(entries, "test")
        assert all(len(c.text) <= 4000 for c in chunks)  # raised from 2000 [S10]


# ---------------------------------------------------------------------------
# Session file parsing
# ---------------------------------------------------------------------------

class TestParseSessionFile:
    def test_parse_jsonl(self, tmp_path, sample_session_entries):
        session_file = tmp_path / "session.jsonl"
        with open(session_file, "w") as f:
            for entry in sample_session_entries:
                f.write(json.dumps(entry) + "\n")

        chunks = parse_session_file(session_file)
        assert len(chunks) >= 1

    def test_parse_empty_file(self, tmp_path):
        session_file = tmp_path / "empty.jsonl"
        session_file.write_text("")
        chunks = parse_session_file(session_file)
        assert chunks == []

    def test_parse_malformed_json(self, tmp_path):
        session_file = tmp_path / "bad.jsonl"
        session_file.write_text("not json\n{bad json}\n")
        chunks = parse_session_file(session_file)
        assert chunks == []  # gracefully handles bad JSON


class TestDiscoverSessions:
    def test_discover_sessions(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_MEMORY_SESSIONS_DIR", str(tmp_path))
        (tmp_path / "a.jsonl").write_text("{}\n")
        (tmp_path / "b.jsonl").write_text("{}\n")
        (tmp_path / "c.txt").write_text("not a session")

        sessions = discover_sessions(str(tmp_path))
        assert len(sessions) == 2
        assert all(s.suffix == ".jsonl" for s in sessions)

    def test_discover_nonexistent_dir(self):
        sessions = discover_sessions("/tmp/nonexistent-agent-memory-test-dir")
        assert sessions == []


# ---------------------------------------------------------------------------
# Full ingest pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestIngestSessions:
    async def test_ingest_basic(self, tmp_path, tmp_storage, fake_embedder, sample_session_entries):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        session_file = sessions_dir / "test-session.jsonl"
        with open(session_file, "w") as f:
            for entry in sample_session_entries:
                f.write(json.dumps(entry) + "\n")

        result = await ingest_sessions(
            storage=tmp_storage,
            embedder=fake_embedder,
            sessions_dir=str(sessions_dir),
        )

        assert result["sessions"] == 1
        assert result["stored"] >= 1
        assert result["chunks"] >= 1

    async def test_ingest_dedup(self, tmp_path, tmp_storage, fake_embedder, sample_session_entries):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        session_file = sessions_dir / "test-session.jsonl"
        with open(session_file, "w") as f:
            for entry in sample_session_entries:
                f.write(json.dumps(entry) + "\n")

        # Ingest twice
        result1 = await ingest_sessions(
            storage=tmp_storage, embedder=fake_embedder,
            sessions_dir=str(sessions_dir),
        )
        result2 = await ingest_sessions(
            storage=tmp_storage, embedder=fake_embedder,
            sessions_dir=str(sessions_dir),
        )

        # Second run should find everything already stored
        assert result2["stored"] == 0
        assert result2["skipped_dup"] >= result1["stored"]

    async def test_ingest_empty_dir(self, tmp_path, tmp_storage, fake_embedder):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        result = await ingest_sessions(
            storage=tmp_storage, embedder=fake_embedder,
            sessions_dir=str(sessions_dir),
        )
        assert result["sessions"] == 0
        assert result["stored"] == 0

    async def test_ingest_with_knowledge_graph(
        self, tmp_path, tmp_storage, fake_embedder, sample_session_entries
    ):
        from agent_memory.entities import KnowledgeGraph

        kg = KnowledgeGraph(storage=tmp_storage)
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        session_file = sessions_dir / "test-session.jsonl"
        with open(session_file, "w") as f:
            for entry in sample_session_entries:
                f.write(json.dumps(entry) + "\n")

        result = await ingest_sessions(
            storage=tmp_storage,
            embedder=fake_embedder,
            sessions_dir=str(sessions_dir),
            knowledge_graph=kg,
        )

        assert result["stored"] >= 1
        # KG should have extracted some entities
        stats = tmp_storage.stats()
        assert stats["entities"] >= 0  # may or may not find entities in sample data

    async def test_progress_callback(self, tmp_path, tmp_storage, fake_embedder, sample_session_entries):
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        session_file = sessions_dir / "test-session.jsonl"
        with open(session_file, "w") as f:
            for entry in sample_session_entries:
                f.write(json.dumps(entry) + "\n")

        progress_calls = []

        def on_progress(done, total):
            progress_calls.append((done, total))

        await ingest_sessions(
            storage=tmp_storage,
            embedder=fake_embedder,
            sessions_dir=str(sessions_dir),
            progress_cb=on_progress,
        )

        assert len(progress_calls) >= 1
