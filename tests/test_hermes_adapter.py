import json
import sys
import types
from pathlib import Path


class _MemoryProvider:
    pass


def _sanitize_context(text):
    return text.replace("<memory-context>", "").replace("</memory-context>", "")


agent_module = types.ModuleType("agent")
memory_provider_module = types.ModuleType("agent.memory_provider")
memory_provider_module.MemoryProvider = _MemoryProvider
memory_manager_module = types.ModuleType("agent.memory_manager")
memory_manager_module.sanitize_context = _sanitize_context
sys.modules.setdefault("agent", agent_module)
sys.modules.setdefault("agent.memory_provider", memory_provider_module)
sys.modules.setdefault("agent.memory_manager", memory_manager_module)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "adapters" / "hermes"))

from noldomem import NoldoMemHTTPClient, NoldoMemProvider  # noqa: E402


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def close(self):
        pass


def test_http_client_uses_x_api_key_header(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["api_key"] = req.get_header("X-api-key")
        return _Response({"ok": True})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client = NoldoMemHTTPClient("http://127.0.0.1:8787", "test-api-key", 1.25)
    assert client.pin({"id": "mem_1", "agent": "hermes"}) == {"ok": True}
    assert captured == {
        "url": "http://127.0.0.1:8787/v1/pin",
        "timeout": 1.25,
        "body": {"id": "mem_1", "agent": "hermes"},
        "api_key": "test-api-key",
    }


def test_provider_exposes_stable_tool_names(monkeypatch, tmp_path):
    key_file = tmp_path / "key"
    key_file.write_text("test-key", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("NOLDOMEM_API_KEY_FILE", str(key_file))

    provider = NoldoMemProvider()
    names = [schema["name"] for schema in provider.get_tool_schemas()]

    assert names == ["noldomem_recall", "noldomem_store", "noldomem_pin"]
    assert provider.is_available() is True


def test_provider_pin_tool_uses_id_payload(monkeypatch, tmp_path):
    key_file = tmp_path / "key"
    key_file.write_text("test-key", encoding="utf-8")
    monkeypatch.setenv("NOLDOMEM_API_KEY_FILE", str(key_file))

    captured = {}

    class FakeClient:
        def pin(self, body):
            captured.update(body)
            return {"pinned": True}

    provider = NoldoMemProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path))
    provider._client = FakeClient()

    result = json.loads(provider.handle_tool_call("noldomem_pin", {"memory_id": "mem_1"}))

    assert result["success"] is True
    assert captured == {"id": "mem_1", "agent": "hermes"}


def test_prefetch_formats_recall_without_memory_context_tags(monkeypatch, tmp_path):
    key_file = tmp_path / "key"
    key_file.write_text("test-key", encoding="utf-8")
    monkeypatch.setenv("NOLDOMEM_API_KEY_FILE", str(key_file))

    class FakeClient:
        def recall(self, body):
            return {
                "results": [
                    {
                        "text": "<memory-context>Use BGE-M3 for embeddings.</memory-context>",
                        "memory_type": "rule",
                        "semantic_score": 0.91,
                    }
                ]
            }

    provider = NoldoMemProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path))
    provider._client = FakeClient()

    context = provider.prefetch("embedding model?", session_id="session-1")

    assert "NoldoMem recall:" in context
    assert "Use BGE-M3 for embeddings." in context
    assert "<memory-context>" not in context


def test_store_rejects_invalid_memory_type(monkeypatch, tmp_path):
    key_file = tmp_path / "key"
    key_file.write_text("test-key", encoding="utf-8")
    monkeypatch.setenv("NOLDOMEM_API_KEY_FILE", str(key_file))

    provider = NoldoMemProvider()
    provider.initialize("session-1", hermes_home=str(tmp_path))
    result = json.loads(
        provider.handle_tool_call(
            "noldomem_store",
            {"text": "hello", "memory_type": "deployment"},
        )
    )

    assert result["success"] is False
    assert "invalid memory_type" in result["error"]
