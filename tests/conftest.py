"""Shared fixtures for Asuman Memory tests."""

from __future__ import annotations

from typing import List

import pytest

from asuman_memory.storage import MemoryStorage
from asuman_memory.entities import EntityExtractor, KnowledgeGraph
from asuman_memory.search import HybridSearch


# ---------------------------------------------------------------------------
# Ensure no real API calls leak out
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch):
    """Set a dummy API key so tests never fail on missing key."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-for-pytest")


# ---------------------------------------------------------------------------
# Storage fixture (temporary DB)
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_storage(tmp_path):
    """Create a fresh MemoryStorage backed by a temp SQLite file (4-dim vectors)."""
    db_path = str(tmp_path / "test.sqlite")
    s = MemoryStorage(db_path=db_path, dimensions=4)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Mock embedder
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Deterministic mock embedder that returns predictable vectors.

    The vector is derived from the hash of the text so that identical
    texts always produce the same vector — useful for dedup tests.
    """

    def __init__(self, dimensions: int = 4):
        self.dimensions = dimensions
        self.call_count = 0

    async def embed(self, text: str) -> List[float]:
        self.call_count += 1
        return self._deterministic_vector(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self.call_count += 1
        return [self._deterministic_vector(t) for t in texts]

    def _deterministic_vector(self, text: str) -> List[float]:
        h = hash(text) & 0xFFFFFFFF
        vec = [0.0] * self.dimensions
        for i in range(self.dimensions):
            vec[i] = ((h >> (i * 8)) & 0xFF) / 255.0
        # Normalise to unit length
        mag = max(sum(v * v for v in vec) ** 0.5, 1e-9)
        return [v / mag for v in vec]


@pytest.fixture
def fake_embedder():
    """Return a FakeEmbedder with 4 dimensions."""
    return FakeEmbedder(dimensions=4)


# ---------------------------------------------------------------------------
# Search fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def hybrid_search(tmp_storage, fake_embedder):
    """Return a HybridSearch wired to temp storage + fake embedder."""
    return HybridSearch(storage=tmp_storage, embedder=fake_embedder)


# ---------------------------------------------------------------------------
# Entity extractor / knowledge graph
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor():
    return EntityExtractor()


@pytest.fixture
def knowledge_graph(tmp_storage):
    return KnowledgeGraph(storage=tmp_storage)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_MESSAGES = [
    {"role": "user", "text": "User ile yarın toplantı var, hatırlat lütfen"},
    {"role": "assistant", "text": "Tamam, yarın toplantınız var. Hatırlatırım."},
    {"role": "user", "text": "İstanbul'a ne zaman gidiyoruz?"},
    {"role": "assistant", "text": "Geçen konuşmada 15 Mart demiştiniz."},
    {"role": "user", "text": "Python ile FastAPI projesi yazmak istiyorum"},
]

SAMPLE_SESSION_JSONL = [
    {"type": "message", "timestamp": "2026-01-15T10:00:00Z",
     "message": {"role": "user", "content": "Merhaba, bugün ne yapacağız?"}},
    {"type": "message", "timestamp": "2026-01-15T10:00:30Z",
     "message": {"role": "assistant", "content": "Bugün memory sistemi üzerinde çalışacağız."}},
    {"type": "message", "timestamp": "2026-01-15T10:05:00Z",
     "message": {"role": "user", "content": "sqlite-vec nasıl çalışıyor?"}},
    {"type": "message", "timestamp": "2026-01-15T10:06:00Z",
     "message": {"role": "assistant", "content": "sqlite-vec, SQLite için bir vektör arama uzantısıdır. Cosine distance ile similarity search yapabilirsiniz."}},
]


@pytest.fixture
def sample_messages():
    return SAMPLE_MESSAGES


@pytest.fixture
def sample_session_entries():
    return SAMPLE_SESSION_JSONL
