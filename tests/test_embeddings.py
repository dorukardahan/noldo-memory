"""Tests for OpenRouter embedding client."""

import os
import pytest
from unittest.mock import patch, MagicMock

from asuman_memory.embeddings import OpenRouterEmbeddings, EmbeddingError


@pytest.fixture
def embedder():
    """Create embedder with a fake API key (no real calls)."""
    return OpenRouterEmbeddings(api_key="test-key", dimensions=4)


class TestCache:
    def test_cache_put_and_get(self, embedder):
        embedder._cache_put("hello", [1.0, 2.0, 3.0, 4.0])
        assert embedder._cache_get("hello") == [1.0, 2.0, 3.0, 4.0]

    def test_cache_miss(self, embedder):
        assert embedder._cache_get("not-cached") is None

    def test_cache_eviction(self):
        embedder = OpenRouterEmbeddings(api_key="test", cache_size=2)
        embedder._cache_put("a", [1.0])
        embedder._cache_put("b", [2.0])
        embedder._cache_put("c", [3.0])  # evicts "a"
        assert embedder._cache_get("a") is None
        assert embedder._cache_get("b") == [2.0]
        assert embedder._cache_get("c") == [3.0]


class TestCallAPI:
    @patch("asuman_memory.embeddings.requests.post")
    def test_successful_call(self, mock_post, embedder):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3, 0.4]},
            ]
        }
        mock_post.return_value = mock_resp

        result = embedder._call_api(["test text"])
        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3, 0.4]

    @patch("asuman_memory.embeddings.requests.post")
    def test_batch_call(self, mock_post, embedder):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]},
                {"index": 1, "embedding": [0.0, 1.0, 0.0, 0.0]},
            ]
        }
        mock_post.return_value = mock_resp

        result = embedder._call_api(["text1", "text2"])
        assert len(result) == 2

    @patch("asuman_memory.embeddings.requests.post")
    def test_non_retryable_error(self, mock_post, embedder):
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_post.return_value = mock_resp

        with pytest.raises(EmbeddingError, match="401"):
            embedder._call_api(["test"])


@pytest.mark.asyncio
class TestAsyncAPI:
    @patch("asuman_memory.embeddings.requests.post")
    async def test_embed_uses_cache(self, mock_post, embedder):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"index": 0, "embedding": [1.0, 2.0, 3.0, 4.0]}]
        }
        mock_post.return_value = mock_resp

        v1 = await embedder.embed("cached text")
        v2 = await embedder.embed("cached text")
        assert v1 == v2
        assert mock_post.call_count == 1  # only one API call

    @patch("asuman_memory.embeddings.requests.post")
    async def test_embed_batch_partial_cache(self, mock_post, embedder):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"index": 0, "embedding": [1.0, 2.0, 3.0, 4.0]}]
        }
        mock_post.return_value = mock_resp

        # Pre-cache one
        embedder._cache_put("cached", [9.0, 8.0, 7.0, 6.0])

        result = await embedder.embed_batch(["cached", "not-cached"])
        assert result[0] == [9.0, 8.0, 7.0, 6.0]
        assert result[1] == [1.0, 2.0, 3.0, 4.0]
