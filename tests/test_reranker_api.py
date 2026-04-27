from __future__ import annotations

import pytest
import urllib.error
import urllib.request

import agent_memory.api as api_module
from agent_memory.config import Config, load_config
from agent_memory.reranker import APIReranker


def test_load_config_reads_api_reranker_env(monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_RERANKER_API_ENABLED", "true")
    monkeypatch.setenv("AGENT_MEMORY_RERANKER_API_MODEL", "cohere/rerank-4")
    monkeypatch.setenv("AGENT_MEMORY_RERANKER_API_URL", "https://example.invalid/rerank")
    monkeypatch.setenv("AGENT_MEMORY_RERANKER_API_TIMEOUT", "7")
    monkeypatch.setenv("AGENT_MEMORY_RERANKER_API_LOCAL_FALLBACK", "true")
    monkeypatch.setenv("AGENT_MEMORY_RERANKER_API_KEY_FILE", "~/custom-rerank-key")

    cfg = load_config()

    assert cfg.reranker_api_enabled is True
    assert cfg.reranker_api_model == "cohere/rerank-4"
    assert cfg.reranker_api_url == "https://example.invalid/rerank"
    assert cfg.reranker_api_timeout == 7
    assert cfg.reranker_api_local_fallback is True
    assert cfg.reranker_api_key_file == "~/custom-rerank-key"


def test_api_reranker_reads_key_file(tmp_path):
    key_file = tmp_path / "rerank.key"
    key_file.write_text("test-secret-key\n", encoding="utf-8")

    reranker = APIReranker(
        enabled=True,
        api_key="",
        api_key_file=str(key_file),
    )

    assert reranker.available is True
    assert reranker.api_key == "test-secret-key"


def test_api_reranker_expands_env_vars_in_key_file_path(tmp_path, monkeypatch):
    key_file = tmp_path / "rerank-env.key"
    key_file.write_text("env-secret-key\n", encoding="utf-8")
    monkeypatch.setenv("RERANK_KEY_PATH", str(key_file))

    reranker = APIReranker(
        enabled=True,
        api_key="",
        api_key_file="$RERANK_KEY_PATH",
    )

    assert reranker.available is True
    assert reranker.api_key == "env-secret-key"


def test_api_reranker_requires_non_empty_endpoint():
    reranker = APIReranker(
        enabled=True,
        api_key="test-secret-key",
        api_url="",
    )

    assert reranker.available is False


def test_api_reranker_runtime_failure_uses_local_fallback(monkeypatch):
    class StubFallback:
        top_k = 20

        @property
        def available(self) -> bool:
            return True

        def warmup(self) -> bool:
            return True

        def score(self, query, docs, doc_ids=None):
            return [0.41 for _ in docs]

    def failing_urlopen(*args, **kwargs):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", failing_urlopen)

    reranker = APIReranker(
        enabled=True,
        api_key="test-secret-key",
        api_url="https://example.invalid/rerank",
        fallback_reranker=StubFallback(),
    )

    scores = reranker.score("hello", ["doc a", "doc b"], ["1", "2"])

    assert scores == [0.41, 0.41]


@pytest.mark.asyncio
async def test_lifespan_prefers_api_reranker_when_available(tmp_path, monkeypatch):
    api_instances = []
    cross_instances = []

    class StubApiReranker:
        def __init__(self, **kwargs):
            self.available = True
            self.top_k = kwargs["top_k"]
            self.fallback_reranker = kwargs.get("fallback_reranker")
            api_instances.append(kwargs)

        def warmup(self) -> bool:
            return True

        def score(self, query, docs, doc_ids=None):
            return [0.9 for _ in docs[: self.top_k]]

    class StubCrossReranker:
        def __init__(self, **kwargs):
            cross_instances.append(kwargs)
            self.top_k = kwargs["top_k"]

        @staticmethod
        def _resolve_model_name(value: str) -> str:
            return value

        @property
        def available(self) -> bool:
            return True

        def warmup(self) -> bool:
            return True

        def score(self, query, docs, doc_ids=None):
            return [0.5 for _ in docs[: self.top_k]]

    async def stub_warmup_loop():
        return None

    monkeypatch.setattr(api_module, "APIReranker", StubApiReranker)
    monkeypatch.setattr(api_module, "CrossEncoderReranker", StubCrossReranker)
    monkeypatch.setattr(api_module, "warmup_loop", stub_warmup_loop)
    monkeypatch.setattr(
        api_module,
        "load_config",
        lambda: Config(
            db_path=str(tmp_path / "memory.sqlite"),
            embedding_dimensions=4,
            openrouter_api_key="embed-key",
            reranker_enabled=True,
            reranker_api_enabled=True,
            reranker_api_key="rerank-key",
            reranker_two_pass_enabled=True,
            embed_worker_enabled=False,
        ),
    )

    async with api_module.lifespan(api_module.app):
        assert isinstance(api_module._reranker, StubApiReranker)
        assert api_module._bg_reranker is None
        assert len(api_instances) == 1
        assert len(cross_instances) == 0
        assert api_module._reranker.fallback_reranker is None


@pytest.mark.asyncio
async def test_lifespan_can_enable_api_reranker_local_fallback(tmp_path, monkeypatch):
    api_instances = []
    cross_instances = []

    class StubApiReranker:
        def __init__(self, **kwargs):
            self.available = True
            self.top_k = kwargs["top_k"]
            self.fallback_reranker = kwargs.get("fallback_reranker")
            api_instances.append(kwargs)

        def warmup(self) -> bool:
            return True

        def score(self, query, docs, doc_ids=None):
            return [0.9 for _ in docs[: self.top_k]]

    class StubCrossReranker:
        def __init__(self, **kwargs):
            cross_instances.append(kwargs)
            self.top_k = kwargs["top_k"]

        @staticmethod
        def _resolve_model_name(value: str) -> str:
            return value

        @property
        def available(self) -> bool:
            return True

        def warmup(self) -> bool:
            return True

        def score(self, query, docs, doc_ids=None):
            return [0.5 for _ in docs[: self.top_k]]

    async def stub_warmup_loop():
        return None

    monkeypatch.setattr(api_module, "APIReranker", StubApiReranker)
    monkeypatch.setattr(api_module, "CrossEncoderReranker", StubCrossReranker)
    monkeypatch.setattr(api_module, "warmup_loop", stub_warmup_loop)
    monkeypatch.setattr(
        api_module,
        "load_config",
        lambda: Config(
            db_path=str(tmp_path / "memory.sqlite"),
            embedding_dimensions=4,
            openrouter_api_key="embed-key",
            reranker_enabled=True,
            reranker_api_enabled=True,
            reranker_api_key="rerank-key",
            reranker_api_local_fallback=True,
            reranker_two_pass_enabled=True,
            embed_worker_enabled=False,
        ),
    )

    async with api_module.lifespan(api_module.app):
        assert isinstance(api_module._reranker, StubApiReranker)
        assert api_module._bg_reranker is None
        assert len(api_instances) == 1
        assert len(cross_instances) == 1
        assert isinstance(api_module._reranker.fallback_reranker, StubCrossReranker)


@pytest.mark.asyncio
async def test_lifespan_falls_back_to_local_reranker_when_api_unavailable(tmp_path, monkeypatch):
    cross_instances = []

    class StubApiReranker:
        def __init__(self, **kwargs):
            self.available = False
            self.top_k = kwargs["top_k"]

        def warmup(self) -> bool:
            return False

        def score(self, query, docs, doc_ids=None):
            return []

    class StubCrossReranker:
        def __init__(self, **kwargs):
            cross_instances.append(kwargs)
            self.top_k = kwargs["top_k"]

        @staticmethod
        def _resolve_model_name(value: str) -> str:
            return value

        @property
        def available(self) -> bool:
            return True

        def warmup(self) -> bool:
            return True

        def score(self, query, docs, doc_ids=None):
            return [0.5 for _ in docs[: self.top_k]]

    async def stub_warmup_loop():
        return None

    monkeypatch.setattr(api_module, "APIReranker", StubApiReranker)
    monkeypatch.setattr(api_module, "CrossEncoderReranker", StubCrossReranker)
    monkeypatch.setattr(api_module, "warmup_loop", stub_warmup_loop)
    monkeypatch.setattr(
        api_module,
        "load_config",
        lambda: Config(
            db_path=str(tmp_path / "memory.sqlite"),
            embedding_dimensions=4,
            openrouter_api_key="embed-key",
            reranker_enabled=True,
            reranker_model="balanced",
            reranker_api_enabled=True,
            reranker_api_key="",
            reranker_two_pass_enabled=True,
            reranker_two_pass_model="quality",
            embed_worker_enabled=False,
        ),
    )

    async with api_module.lifespan(api_module.app):
        assert isinstance(api_module._reranker, StubCrossReranker)
        assert isinstance(api_module._bg_reranker, StubCrossReranker)
        assert len(cross_instances) == 2
