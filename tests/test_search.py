"""Tests for hybrid search."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from asuman_memory.search import HybridSearch, SearchWeights, _recency_score, _rrf_fuse
from asuman_memory.storage import MemoryStorage


@pytest.fixture
def storage(tmp_path):
    db_path = str(tmp_path / "search_test.sqlite")
    s = MemoryStorage(db_path=db_path, dimensions=4)
    yield s
    s.close()


class TestRecencyScore:
    def test_recent_is_high(self):
        import time
        score = _recency_score(time.time())
        assert score > 0.99

    def test_old_is_low(self):
        import time
        old = time.time() - 365 * 86400  # 1 year ago
        score = _recency_score(old)
        assert score < 0.1


class TestRRFFusion:
    def test_single_list(self):
        scores = _rrf_fuse([["a", "b", "c"]], [1.0])
        assert scores["a"] > scores["b"] > scores["c"]

    def test_two_lists_boost(self):
        scores = _rrf_fuse(
            [["a", "b", "c"], ["b", "a", "c"]],
            [0.5, 0.5],
        )
        # "b" appears in both lists at good positions
        # Both "a" and "b" should have higher scores than "c"
        assert scores["a"] > scores["c"]
        assert scores["b"] > scores["c"]

    def test_empty_list(self):
        scores = _rrf_fuse([], [])
        assert scores == {}


@pytest.mark.asyncio
class TestHybridSearch:
    async def test_keyword_only(self, storage):
        """Search with only BM25 (no embedder)."""
        storage.store_memory(text="User yarın toplantı var dedi", category="user")
        storage.store_memory(text="Hava bugün çok güzel", category="user")

        search = HybridSearch(storage=storage, embedder=None)
        results = await search.search("toplantı", limit=5)
        assert len(results) >= 1
        assert "toplantı" in results[0].text

    async def test_empty_query(self, storage):
        search = HybridSearch(storage=storage, embedder=None)
        results = await search.search("", limit=5)
        assert results == []

    async def test_semantic_with_mock_embedder(self, storage):
        """Test semantic search with mocked embedder."""
        storage.store_memory(
            text="hatırlıyor musun dün ne konuştuk",
            vector=[1.0, 0.0, 0.0, 0.0],
        )
        storage.store_memory(
            text="hava çok güzel bugün",
            vector=[0.0, 1.0, 0.0, 0.0],
        )

        mock_embedder = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=[1.0, 0.0, 0.0, 0.0])

        search = HybridSearch(storage=storage, embedder=mock_embedder)
        results = await search.search("hatırlıyor", limit=5)
        assert len(results) >= 1
