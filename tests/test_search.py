"""Tests for hybrid search."""

import pytest
from unittest.mock import AsyncMock

from agent_memory.search import HybridSearch, _recency_score, _rrf_fuse
from agent_memory.storage import MemoryStorage


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
        storage.store_memory(text="Ahmet yarın toplantı var dedi", category="user")
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

    async def test_kg_entity_lookup_adds_candidate_without_keyword_hit(self, storage):
        memory_id = storage.store_memory(
            text="Alice and Bob shipped the project last week",
            category="user",
        )
        alice_id = storage.store_entity("Alice", entity_type="person")
        bob_id = storage.store_entity("Bob", entity_type="person")
        storage.link_entities(
            alice_id,
            bob_id,
            relation_type="mentioned_with",
            context="Alice and Bob shipped the project last week",
        )

        search = HybridSearch(storage=storage, embedder=None)
        results = await search.search('What did "Alice" do?', limit=5)

        assert any(r.id == memory_id for r in results)


@pytest.mark.asyncio
async def test_degraded_recall_requires_content_match_not_articles_or_substrings(storage):
    """A natural query must not retrieve flour through 'the' or 'our'."""
    preference = storage.store_memory(text="For Aurora observatory visits, I prefer quiet evenings.")
    unrelated = storage.store_memory(text="The synthetic baking club uses seven cups of rye flour.")
    morphology = storage.store_memory(text="Kitaplıktaki katalogların yerini değiştirdim.")
    # Capture can extract a title-cased entity including an article.
    storage.store_entity("The Aurora", entity_type="person")
    search = HybridSearch(storage=storage, embedder=None)
    result = await search.search(
        "Plan our Aurora observatory visit via the spiral staircase. Which time suits me?", limit=10)
    assert result.degraded
    assert preference in {r.id for r in result}
    assert unrelated not in {r.id for r in result}
    assert unrelated not in {r.id for r in await search.search("Where is our telescope?", limit=10)}
    # Preserve trigram morphology matching; a stem need not be a whole word.
    assert morphology in {r.id for r in await search.search("kitaplık katalog", limit=10)}


@pytest.mark.asyncio
async def test_degraded_recall_keeps_direct_graph_evidence_within_namespace(storage):
    source = storage.store_memory(text="A telescope is reserved for the evening tour.", namespace="observations")
    entity = storage.store_entity("Aurora", entity_type="place")
    storage.store_temporal_fact(entity, "Reserved instrument", source_memory_id=source)
    search = HybridSearch(storage=storage, embedder=None)
    result = await search.search("Aurora", namespace="observations")
    assert source in {r.id for r in result}  # No lexical overlap, but a direct source link.
    assert not await search.search("Aurora", namespace="other")
