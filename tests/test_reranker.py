"""Tests for reranker adapters."""

from __future__ import annotations

import json
import urllib.request

from agent_memory.reranker import APIReranker


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_api_reranker_reads_key_from_configured_file(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_MEMORY_RERANKER_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_MEMORY_RERANKER_API_KEY_FILE", raising=False)
    key_file = tmp_path / "reranker.key"
    key_file.write_text("test-key\n", encoding="utf-8")

    reranker = APIReranker(api_key_file=str(key_file))

    assert reranker.available is True
    assert reranker.api_key == "test-key"


def test_api_reranker_scores_documents_without_network(monkeypatch):
    requests = []

    def fake_urlopen(req, timeout):
        requests.append((req, timeout))
        body = json.loads(req.data.decode("utf-8"))
        assert body["model"] == "cohere/rerank-4-pro"
        assert body["query"] == "where is the config"
        assert body["documents"] == ["doc one", "doc two"]
        assert req.headers["Authorization"] == "Bearer test-key"
        assert timeout == 7
        return _FakeResponse(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.25},
                    {"index": 0, "relevance_score": 0.75},
                ]
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    reranker = APIReranker(api_key="test-key", timeout_sec=7)

    assert reranker.score("where is the config", ["doc one", "doc two"]) == [0.75, 0.25]
    assert len(requests) == 1


def test_api_reranker_returns_empty_when_unavailable(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_MEMORY_RERANKER_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_MEMORY_RERANKER_API_KEY_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    reranker = APIReranker(api_key="", api_key_file="/missing/key")

    assert reranker.available is False
    assert reranker.score("query", ["doc"]) == []
