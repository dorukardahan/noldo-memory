"""Tests for SQLite storage layer."""

import os
import tempfile
import pytest

from asuman_memory.storage import MemoryStorage


@pytest.fixture
def storage(tmp_path):
    """Create a temporary storage instance."""
    db_path = str(tmp_path / "test_memory.sqlite")
    s = MemoryStorage(db_path=db_path, dimensions=4)
    yield s
    s.close()


class TestMemoryCRUD:
    def test_store_and_get(self, storage):
        mid = storage.store_memory(
            text="User yarın toplantı var dedi",
            vector=[1.0, 0.0, 0.0, 0.0],
            category="user",
            importance=0.7,
        )
        mem = storage.get_memory(mid)
        assert mem is not None
        assert mem["text"] == "User yarın toplantı var dedi"
        assert mem["category"] == "user"
        assert abs(mem["importance"] - 0.7) < 0.01

    def test_delete(self, storage):
        mid = storage.store_memory(text="to delete", vector=[1.0, 0.0, 0.0, 0.0])
        assert storage.delete_memory(mid) is True
        assert storage.get_memory(mid) is None
        assert storage.delete_memory("nonexistent") is False

    def test_update(self, storage):
        mid = storage.store_memory(text="original", vector=[1.0, 0.0, 0.0, 0.0])
        storage.update_memory(mid, text="updated", importance=0.9)
        mem = storage.get_memory(mid)
        assert mem["text"] == "updated"
        assert abs(mem["importance"] - 0.9) < 0.01

    def test_store_without_vector(self, storage):
        mid = storage.store_memory(text="no vector here")
        mem = storage.get_memory(mid)
        assert mem is not None
        assert mem["vector_rowid"] is None


class TestVectorSearch:
    def test_basic_search(self, storage):
        storage.store_memory(text="hello world", vector=[1.0, 0.0, 0.0, 0.0])
        storage.store_memory(text="goodbye world", vector=[0.0, 1.0, 0.0, 0.0])

        results = storage.search_vectors([1.0, 0.0, 0.0, 0.0], limit=2)
        assert len(results) >= 1
        # The first result should be the most similar
        assert results[0]["text"] == "hello world"
        assert results[0]["score"] > 0.5

    def test_min_score_filter(self, storage):
        storage.store_memory(text="test", vector=[1.0, 0.0, 0.0, 0.0])
        results = storage.search_vectors([0.0, 1.0, 0.0, 0.0], limit=10, min_score=0.9)
        # Orthogonal vectors should have low similarity
        assert len(results) == 0


class TestFTSSearch:
    def test_text_search(self, storage):
        storage.store_memory(text="User yarın toplantı var dedi")
        storage.store_memory(text="Hava bugün çok güzel")

        results = storage.search_text("toplantı", limit=5)
        assert len(results) >= 1
        assert "toplantı" in results[0]["text"]

    def test_turkish_text(self, storage):
        storage.store_memory(text="hatırlıyor musun dün ne konuştuk")
        results = storage.search_text("konuştuk", limit=5)
        assert len(results) >= 1


class TestEntityCRUD:
    def test_store_and_get_entity(self, storage):
        eid = storage.store_entity(name="User", entity_type="person")
        entity = storage.get_entity(eid)
        assert entity is not None
        assert entity["name"] == "User"
        assert entity["type"] == "person"
        assert entity["mention_count"] == 1

    def test_entity_dedup(self, storage):
        eid1 = storage.store_entity(name="User", entity_type="person")
        eid2 = storage.store_entity(name="user", entity_type="person")
        assert eid1 == eid2
        entity = storage.get_entity(eid1)
        assert entity["mention_count"] == 2

    def test_link_entities(self, storage):
        eid1 = storage.store_entity(name="User", entity_type="person")
        eid2 = storage.store_entity(name="Asuman", entity_type="person")
        rid = storage.link_entities(eid1, eid2, "works_with", context="testing")
        assert rid is not None


class TestBatchOperations:
    def test_batch_store(self, storage):
        items = [
            {"text": f"memory {i}", "vector": [float(i == j) for j in range(4)]}
            for i in range(4)
        ]
        ids = storage.store_memories_batch(items)
        assert len(ids) == 4
        for mid in ids:
            assert storage.get_memory(mid) is not None


class TestStats:
    def test_stats(self, storage):
        storage.store_memory(text="test1", category="user")
        storage.store_memory(text="test2", category="assistant")
        storage.store_entity(name="User", entity_type="person")

        s = storage.stats()
        assert s["total_memories"] == 2
        assert s["entities"] == 1
        assert "user" in s["by_category"]
