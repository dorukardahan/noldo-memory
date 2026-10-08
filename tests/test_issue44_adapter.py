"""Source-only #44 adapter regressions; every HTTP peer is disposable loopback."""
import asyncio
import http.client
import json
import signal
import socket
import sys
import threading
import time
import urllib.request
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters" / "hermes"))
import noldomem  # noqa: E402


@pytest.fixture(autouse=True)
def disposable_config(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("NOLDOMEM_API_KEY", "synthetic-only")
    for name in ("NOLDOMEM_CONFIG_FILE", "NOLDOMEM_CONFIG", "NOLDOMEM_API_KEY_FILE",
                 "NOLDOMEM_TIMEOUT_SECONDS", "NOLDOMEM_CAPTURE_TIMEOUT_SECONDS",
                 "NOLDOMEM_STORE_TIMEOUT_SECONDS", "NOLDOMEM_RECALL_TIMEOUT_SECONDS",
                 "NOLDOMEM_STATUS_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)


def provider(monkeypatch, tmp_path, **settings):
    cfg = noldomem.NoldoMemConfig(api_key="synthetic-only", sync_turns_enabled=True, **settings)
    instance = noldomem.NoldoMemProvider()
    monkeypatch.setattr(instance, "load_config", lambda *a, **kw: cfg)
    instance.initialize("synthetic-session", hermes_home=str(tmp_path))
    return instance


def capture(instance):
    return instance.sync_turn("synthetic user", "synthetic answer", messages=[
        {"role": "user", "content": "synthetic user"},
        {"role": "assistant", "content": "synthetic answer"},
    ])


def accepted(operation="capture", **extra):
    return {"operation": operation, "state": "accepted", "durable": True,
            "indexing_state": "pending", "stage_states": {"embed": "pending", "graph": "pending"}, **extra}


@contextmanager
def local_http(responder):
    calls = []
    closed = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            calls.append({"method": self.command, "path": self.path, "body": json.loads(body) if body else None,
                          "attempt_id": self.headers.get("X-Request-ID")})
            try:
                result = responder(self, calls)
                if result is not None:
                    raw = json.dumps(result).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                closed.set()

        do_POST = handle_request
        do_GET = handle_request

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls, closed
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)


@pytest.mark.parametrize("legacy,expected", [(8, 8), (30, 10), (60, 10)])
def test_operation_defaults_effective_legacy_cap(monkeypatch, tmp_path, legacy, expected):
    cfg = noldomem.NoldoMemProvider().load_config(str(tmp_path), timeout_seconds=legacy)
    assert cfg.capture_timeout_seconds == expected
    assert cfg.store_timeout_seconds == expected
    assert cfg.recall_timeout_seconds == expected
    assert cfg.status_timeout_seconds == 2


@pytest.mark.parametrize("operation,cap", [("capture", 10), ("store", 10), ("recall", 10), ("status", 2)])
def test_operation_env_precedence_clamps(monkeypatch, tmp_path, operation, cap):
    monkeypatch.setenv("NOLDOMEM_TIMEOUT_SECONDS", "4")
    field = operation + "_timeout_seconds"
    monkeypatch.setenv("NOLDOMEM_" + field.upper(), "60")
    cfg = noldomem.NoldoMemProvider().load_config(str(tmp_path), **{field: 0.3, "timeout_seconds": 8})
    assert getattr(cfg, field) == cap
    monkeypatch.setenv("NOLDOMEM_" + field.upper(), "0")
    assert getattr(noldomem.NoldoMemProvider().load_config(str(tmp_path), **{field: 1}), field) == 0.1


@pytest.mark.parametrize("bad", [True, False, "3", "bad", [], {}, float("nan"), float("inf"), -float("inf")])
def test_invalid_json_timeout_uses_safe_legacy_fallback(monkeypatch, tmp_path, bad):
    cfg = noldomem.NoldoMemProvider().load_config(str(tmp_path), timeout_seconds=3,
        capture_timeout_seconds=bad, store_timeout_seconds=bad,
        recall_timeout_seconds=bad, status_timeout_seconds=bad)
    assert cfg.capture_timeout_seconds == cfg.store_timeout_seconds == cfg.recall_timeout_seconds == 3
    assert cfg.status_timeout_seconds == 2


@pytest.mark.parametrize("bad", [True, False, "bad", [], {}, float("nan"), float("inf")])
def test_invalid_legacy_timeout_never_becomes_small_bool_budget(monkeypatch, tmp_path, bad):
    cfg = noldomem.NoldoMemProvider().load_config(str(tmp_path), timeout_seconds=bad)
    assert cfg.timeout_seconds == 8
    assert cfg.capture_timeout_seconds == 8


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "true", "", "[]", "{}"])
def test_invalid_env_falls_back_without_echo(monkeypatch, tmp_path, bad):
    monkeypatch.setenv("NOLDOMEM_CAPTURE_TIMEOUT_SECONDS", bad)
    cfg = noldomem.NoldoMemProvider().load_config(str(tmp_path), timeout_seconds=4, capture_timeout_seconds=1)
    assert cfg.capture_timeout_seconds == 4


def test_optional_dataclass_fields_inherit_at_runtime(monkeypatch, tmp_path):
    instance = provider(monkeypatch, tmp_path, timeout_seconds=0.4)
    calls = []
    with local_http(lambda h, seen: calls.append(h.command) or accepted("store")) as (url, _, _):
        instance._client = noldomem.NoldoMemHTTPClient(url, "synthetic-only", 0.4)
        result = json.loads(instance.handle_tool_call("noldomem_store", {"text": "synthetic"}))
    assert result["success"] is True
    assert result["data"]["indexing_state"] == "pending"
    assert calls == ["POST"]


@pytest.mark.parametrize("caller", ["capture", "store", "sync_store", "native_store"])
def test_all_write_paths_generate_uuid4_once(monkeypatch, tmp_path, caller):
    instance = provider(monkeypatch, tmp_path)
    bodies = []

    class Client:
        def capture(self, body):
            bodies.append(body.copy())
            return accepted()

        def store(self, body):
            bodies.append(body.copy())
            return {"stored": True, "id": "synthetic-id"}

    instance._client = Client()
    for _ in range(2):
        if caller == "capture":
            capture(instance)
        elif caller == "store":
            instance.handle_tool_call("noldomem_store", {"text": "synthetic"})
        elif caller == "sync_store":
            instance.sync_turn("synthetic user", "synthetic answer")
        else:
            instance.on_memory_write("create", "user", "synthetic")
    ids = [body.get("request_id") for body in bodies]
    assert all(isinstance(value, str) and str(uuid.UUID(value)) == value and uuid.UUID(value).version == 4 for value in ids)
    assert ids[0] != ids[1]  # Identical content is not a durable request identity.


@pytest.mark.parametrize("operation", ["capture", "store"])
def test_response_drop_reconciles_identity_get_once_no_replay(monkeypatch, tmp_path, operation):
    def respond(handler, calls):
        if handler.command == "POST":
            handler.connection.shutdown(socket.SHUT_RDWR)
            handler.connection.close()
            return None
        return accepted(operation)

    with local_http(respond) as (url, calls, _):
        instance = provider(monkeypatch, tmp_path, base_url=url)
        if operation == "capture":
            capture(instance)
        else:
            result = json.loads(instance.handle_tool_call("noldomem_store", {"text": "synthetic"}))
            assert result["success"] is True
            assert result["data"]["state"] == "accepted"
        assert [call["method"] for call in calls] == ["POST", "GET"]
        write, status = calls
        identity = write["body"]["request_id"]
        assert urlsplit(status["path"]).path == "/v1/operations/" + identity
        assert parse_qs(urlsplit(status["path"]).query) == {
            "agent": ["hermes"], "namespace": ["default"], "operation": [operation]}
        assert status["body"] is None
        assert all(uuid.UUID(call["attempt_id"]).version == 4 for call in calls)
        assert write["attempt_id"] != status["attempt_id"]
        assert identity not in {write["attempt_id"], status["attempt_id"]}


@pytest.mark.parametrize("error", [json.JSONDecodeError("bad", "x", 0), UnicodeDecodeError("utf8", b"x", 0, 1, "bad"),
                                   http.client.IncompleteRead(b"x", 5), ConnectionResetError("private error"),
                                   ValueError("private error"), AttributeError("private error")])
def test_legacy_status_ordinary_errors_remain_ambiguous(monkeypatch, tmp_path, error):
    instance = provider(monkeypatch, tmp_path)
    calls = []

    class OldClient:
        def capture(self, body):
            calls.append("write")
            raise RuntimeError("NoldoMem API timed out")

        def capture_status(self, body):
            calls.append("legacy_status")
            raise error

    instance._client = OldClient()
    with pytest.raises(RuntimeError, match="ambiguous") as raised:
        capture(instance)
    assert "private error" not in str(raised.value)
    assert calls == ["write", "legacy_status"]


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(), asyncio.CancelledError()])
def test_status_cancellation_not_swallowed(monkeypatch, tmp_path, error):
    instance = provider(monkeypatch, tmp_path)

    class Client:
        def capture(self, body):
            raise RuntimeError("NoldoMem API timed out")

        def operation_status(self, body):
            raise error

    instance._client = Client()
    with pytest.raises(type(error)):
        capture(instance)
    assert instance._active_operations == {}


@pytest.mark.parametrize("receipt", [{"state": "blocked", "durable": True, "indexing_state": "blocked"},
    {"state": "accepted", "durable": False, "indexing_state": "pending"},
    {"state": "completed", "durable": "true", "indexing_state": "completed"},
    {"stored": 1, "blocked": 1, "merged": 0, "total": 2}])
def test_capture_never_accepts_blocked_or_non_durable_receipt(monkeypatch, tmp_path, receipt):
    instance = provider(monkeypatch, tmp_path)

    class Client:
        def capture(self, body):
            return receipt

    instance._client = Client()
    with pytest.raises(RuntimeError):
        capture(instance)


@pytest.mark.parametrize("state,durable,success", [("accepted", True, True), ("completed", True, True),
    ("failed", True, True), ("blocked", True, False), ("failed", False, False)])
def test_store_reports_durable_acceptance_not_index_completion(monkeypatch, tmp_path, state, durable, success):
    instance = provider(monkeypatch, tmp_path)
    receipt = accepted("store", state=state, durable=durable,
                       indexing_state="failed" if state == "failed" else "pending")

    class Client:
        def store(self, body):
            return receipt

    instance._client = Client()
    result = json.loads(instance.handle_tool_call("noldomem_store", {"text": "synthetic"}))
    assert result["success"] is success
    if success:
        assert result["data"] == receipt


def test_failed_ambiguous_write_invalidates_prefetch_generation(monkeypatch, tmp_path):
    instance = provider(monkeypatch, tmp_path)
    old = instance._recall_snapshot("synthetic", session_id="")

    class Client:
        def capture(self, body):
            raise RuntimeError("NoldoMem API timed out")

        def operation_status(self, body):
            return {"state": "blocked", "durable": True}

    instance._client = Client()
    with pytest.raises(RuntimeError, match="ambiguous"):
        capture(instance)
    assert not instance._recall_snapshot_is_current(old)


def test_no_reconciliation_after_session_fence(monkeypatch, tmp_path):
    instance = provider(monkeypatch, tmp_path)
    calls = []

    class Client:
        def capture(self, body):
            instance.on_session_switch("new-session")
            raise RuntimeError("NoldoMem API timed out")

        def operation_status(self, body):
            calls.append(body)
            return accepted()

    instance._client = Client()
    with pytest.raises(RuntimeError):
        capture(instance)
    assert calls == []


def test_identity_404_is_ambiguous_not_legacy_or_replay(monkeypatch, tmp_path):
    def respond(handler, calls):
        if handler.command == "POST":
            handler.connection.shutdown(socket.SHUT_RDWR)
            handler.connection.close()
        else:
            handler.send_error(404)

    with local_http(respond) as (url, calls, _):
        instance = provider(monkeypatch, tmp_path, base_url=url)
        with pytest.raises(RuntimeError, match="ambiguous"):
            capture(instance)
        assert [call["method"] for call in calls] == ["POST", "GET"]


def test_known_older_server_uses_one_legacy_status_request(monkeypatch, tmp_path):
    def respond(handler, calls):
        if len(calls) == 1:
            return {"stored": 2, "merged": 0, "blocked": 0, "total": 2}
        if len(calls) == 2:
            handler.connection.shutdown(socket.SHUT_RDWR)
            handler.connection.close()
            return None
        return {"state": "complete"}

    with local_http(respond) as (url, calls, _):
        instance = provider(monkeypatch, tmp_path, base_url=url)
        capture(instance)  # Observe an actual legacy HTTP-200 capture receipt first.
        capture(instance)
        assert [urlsplit(call["path"]).path for call in calls] == [
            "/v1/capture", "/v1/capture", "/v1/capture/status"]
        assert calls[2]["body"] == calls[1]["body"]


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="POSIX timers unavailable")
def test_real_trickle_read_closes_response_within_total(monkeypatch, tmp_path):
    def respond(handler, calls):
        handler.send_response(200)
        handler.send_header("Content-Length", "500")
        handler.end_headers()
        for _ in range(500):
            handler.wfile.write(b" ")
            handler.wfile.flush()
            time.sleep(0.01)

    original = urllib.request.urlopen
    responses_closed = []

    def tracked_open(*args, **kwargs):
        response = original(*args, **kwargs)
        original_close = response.close

        def close():
            responses_closed.append(True)
            original_close()

        response.close = close
        return response

    monkeypatch.setattr(urllib.request, "urlopen", tracked_open)
    with local_http(respond) as (url, calls, closed):
        client = noldomem.NoldoMemHTTPClient(url, "synthetic-only", 0.15)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="timed out"):
            client.recall({"query": "synthetic"})
        elapsed = time.monotonic() - started
        assert elapsed < 0.4
        assert responses_closed == [True]
        assert closed.wait(0.5)
        assert len(calls) == 1


def test_http_thread_transport_propagates_ordinary_error_without_retry(monkeypatch):
    calls = []
    errors = []

    def unexpected_open(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("unexpected network")

    monkeypatch.setattr(urllib.request, "urlopen", unexpected_open)
    client = noldomem.NoldoMemHTTPClient("http://127.0.0.1:1", "synthetic-only", 8)

    def run():
        try:
            client.recall({"query": "synthetic"})
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(1)
    assert not thread.is_alive()
    # Host-thread support is required; no POSIX main-thread timer is needed.
    # Real success/stall coverage lives in test_issue44_transport.py.
    assert calls == [True]
    assert errors == ["unexpected network"]


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="POSIX timers unavailable")
def test_write_status_share_total_with_reserved_budget(monkeypatch, tmp_path):
    def respond(handler, calls):
        time.sleep(0.6)
        return accepted()

    with local_http(respond) as (url, calls, _):
        instance = provider(monkeypatch, tmp_path, base_url=url,
                            capture_timeout_seconds=0.2, status_timeout_seconds=0.1)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="ambiguous"):
            capture(instance)
        elapsed = time.monotonic() - started
        assert elapsed < 0.4
        assert [call["method"] for call in calls] == ["POST", "GET"]


@pytest.mark.parametrize("caller", ["tool", "prefetch", "queue"])
def test_recall_uses_operation_specific_total(monkeypatch, tmp_path, caller):
    def respond(handler, calls):
        time.sleep(0.4)
        return {"results": [{"text": "synthetic"}]}

    with local_http(respond) as (url, _, _):
        instance = provider(monkeypatch, tmp_path, base_url=url, recall_timeout_seconds=0.1,
                            recall_cache_ttl_seconds=1)
        started = time.monotonic()
        if caller == "tool":
            assert json.loads(instance.handle_tool_call("noldomem_recall", {"query": "synthetic"}))["success"] is False
        elif caller == "prefetch":
            assert instance.prefetch("synthetic") == ""
        else:
            instance.queue_prefetch("synthetic")
        assert time.monotonic() - started < 0.3
        assert instance._cache == {}
