from agent_memory.config import load_config


def test_default_embedding_profile_is_small_cpu_friendly(monkeypatch):
    monkeypatch.delenv("AGENT_MEMORY_CONFIG", raising=False)
    monkeypatch.delenv("AGENT_MEMORY_MODEL", raising=False)
    monkeypatch.delenv("AGENT_MEMORY_DIMENSIONS", raising=False)

    cfg = load_config()

    assert cfg.embedding_model == "qwen/qwen3-embedding-0.6b"
    assert cfg.embedding_dimensions == 1024


def test_embedding_profile_env_can_override_defaults(monkeypatch):
    monkeypatch.delenv("AGENT_MEMORY_CONFIG", raising=False)
    monkeypatch.setenv("AGENT_MEMORY_MODEL", "openai/text-embedding-3-large")
    monkeypatch.setenv("AGENT_MEMORY_DIMENSIONS", "3072")

    cfg = load_config()

    assert cfg.embedding_model == "openai/text-embedding-3-large"
    assert cfg.embedding_dimensions == 3072
