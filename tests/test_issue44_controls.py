"""Issue 44 controls: isolated configuration, raw ASGI admission and metrics."""
import asyncio
import json
import sys
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from agent_memory.config import Config, load_config
from agent_memory.metrics import MetricsCollector
from agent_memory.middleware import APIKeyMiddleware, AuditLogMiddleware

CONTROLS = {
    "api_write_timeout_seconds": (2, .1, 6),
    "api_recall_timeout_seconds": (6, .1, 6),
    "api_status_timeout_seconds": (1, .1, 1),
    "api_max_body_bytes": (262144, 1024, 1048576),
    "capture_max_total_chars": (128000, 1000, 128000),
    "embed_max_batch_items": (8, 1, 8),
    "embed_max_batch_chars": (16000, 1000, 16000),
    "embed_timeout_seconds": (2, .1, 2),
    "embed_max_retries": (2, 1, 2),
    "index_job_timeout_seconds": (30, 1, 30),
    "index_job_max_attempts": (3, 1, 3),
    "index_queue_max_jobs": (1000, 1, 1000),
    "api_embedding_concurrency": (2, 1, 2),
    "recall_max_agents": (32, 1, 32),
}


@pytest.mark.parametrize("field,bounds", CONTROLS.items())
def test_control_defaults_and_inclusive_bounds(field, bounds):
    default, lower, upper = bounds
    assert getattr(Config(), field) == default
    assert getattr(Config(**{field: lower}), field) == lower
    assert getattr(Config(**{field: upper}), field) == upper


@pytest.mark.parametrize("field,bounds", CONTROLS.items())
@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True,
                                  "PRIVATE_BAD_VALUE", [], {}, None])
def test_invalid_direct_controls_reject_without_value(field, bounds, bad):
    with pytest.raises(ValueError) as error:
        Config(**{field: bad})
    assert str(error.value) == f"invalid_config:{field}"


@pytest.mark.parametrize("field,bounds", CONTROLS.items())
def test_control_rejects_above_cap_and_fractional_integers(field, bounds):
    with pytest.raises(ValueError, match=f"^invalid_config:{field}$"):
        Config(**{field: bounds[2] + 1})
    if not field.endswith("timeout_seconds"):
        with pytest.raises(ValueError, match=f"^invalid_config:{field}$"):
            Config(**{field: bounds[1] + .5})


@pytest.mark.parametrize("field,bounds", CONTROLS.items())
@pytest.mark.parametrize("bad", [True, "PRIVATE_BAD_VALUE", [], {}, float("nan"), 0])
def test_invalid_json_controls_reject_without_value(tmp_path, monkeypatch, field, bounds, bad):
    monkeypatch.delenv("AGENT_MEMORY_" + field.upper(), raising=False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({field: bad}))
    with pytest.raises(ValueError) as error:
        load_config(str(path))
    assert str(error.value) == f"invalid_config:{field}"


@pytest.mark.parametrize("field,bounds", CONTROLS.items())
def test_control_env_overlays_json_and_bad_env_rejects(tmp_path, monkeypatch, field, bounds):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({field: bounds[1]}))
    name = "AGENT_MEMORY_" + field.upper()
    monkeypatch.setenv(name, str(bounds[2]))
    assert getattr(load_config(str(path)), field) == bounds[2]
    monkeypatch.setenv(name, "PRIVATE_BAD_VALUE")
    with pytest.raises(ValueError) as error:
        load_config(str(path))
    assert str(error.value) == f"invalid_config:{field}"


def test_mutated_config_validate_reports_safe_field_code():
    config = Config()
    config.api_max_body_bytes = "PRIVATE_BAD_VALUE"
    assert "invalid_config:api_max_body_bytes" in config.validate()


async def raw_attempt(app, chunks, headers=(), path="/v1/capture"):
    sent = []
    messages = [{"type": "http.request", "body": chunk,
                 "more_body": i < len(chunks) - 1} for i, chunk in enumerate(chunks)]
    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}
    async def send(message):
        sent.append(message)
    await app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
               "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(),
               "query_string": b"", "headers": list(headers), "client": ("127.0.0.1", 1),
               "server": ("localhost", 80)}, receive, send)
    return sent


@pytest.mark.parametrize("headers", [(), ((b"content-length", b"1"),),
                                      ((b"content-length", b"not-a-number"),)])
def test_raw_received_bytes_cap_before_mutation(monkeypatch, headers):
    mutations = []
    async def endpoint(scope, receive, send):
        mutations.append(True)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})
    monkeypatch.setitem(sys.modules, "agent_memory.api",
                        SimpleNamespace(_config=Config(api_max_body_bytes=1024)))
    middleware = AuditLogMiddleware(endpoint)
    result = asyncio.run(raw_attempt(middleware, [b"a" * 600, b"b" * 600], headers))
    assert result[0]["status"] == 413
    assert mutations == []
    assert json.loads(result[1]["body"])["detail"] == "body_too_large"
    assert UUID(dict(result[0]["headers"])[b"x-request-id"].decode()).version == 4


def test_body_exact_limit_preserved_and_live_config_changes(monkeypatch):
    bodies = []
    async def endpoint(scope, receive, send):
        bodies.append((await receive())["body"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})
    module = SimpleNamespace(_config=Config(api_max_body_bytes=1024))
    monkeypatch.setitem(sys.modules, "agent_memory.api", module)
    middleware = AuditLogMiddleware(endpoint)
    result = asyncio.run(raw_attempt(middleware, [b"a" * 512, b"b" * 512]))
    assert result[0]["status"] == 200
    assert bodies == [b"a" * 512 + b"b" * 512]
    module._config = Config(api_max_body_bytes=2048)
    assert asyncio.run(raw_attempt(middleware, [b"x" * 2048]))[0]["status"] == 200


@pytest.mark.parametrize("candidate", [None, "PRIVATE_BAD_ID", str(uuid4()).upper(),
                                        uuid4().hex, "00000000-0000-0000-0000-000000000000"])
def test_attempt_request_id_replaces_noncanonical_values(candidate):
    app = FastAPI()
    app.add_middleware(AuditLogMiddleware)
    @app.get("/v1/health/live")
    async def live():
        return {"ok": True}
    with TestClient(app) as client:
        headers = {} if candidate is None else {"X-Request-ID": candidate}
        first = client.get("/v1/health/live", headers=headers).headers["x-request-id"]
        second = client.get("/v1/health/live", headers=headers).headers["x-request-id"]
    assert str(UUID(first)) == first and UUID(first).version == 4
    assert first != second


def test_canonical_client_attempt_id_is_preserved():
    app = FastAPI()
    app.add_middleware(AuditLogMiddleware)
    @app.get("/v1/health/live")
    async def live():
        return {"ok": True}
    identity = str(uuid4())
    with TestClient(app) as client:
        assert client.get("/v1/health/live", headers={"X-Request-ID": identity}).headers["x-request-id"] == identity


@pytest.mark.parametrize("path", ["/v1/health/backend", "/v1/capture/status", "/v1/operations/" + str(uuid4())])
def test_backend_and_status_not_public_exemptions(path):
    app = FastAPI()
    app.add_middleware(AuditLogMiddleware)
    app.add_middleware(APIKeyMiddleware, api_key="synthetic-test-key")
    with TestClient(app) as client:
        response = client.get(path)
    assert response.status_code == 401
    assert UUID(response.headers["x-request-id"]).version == 4


def test_liveness_and_legacy_doctor_are_public():
    assert {"/v1/health/live", "/v1/health/doctor"} <= APIKeyMiddleware.EXEMPT_PATHS


def test_stage_metrics_finite_labels_and_buckets():
    metrics = MetricsCollector()
    for stage in ("queue", "normalize", "dedup", "persist", "embedding", "graph",
                  "bm25", "vector", "rerank", "access", "total"):
        metrics.record_stage("capture", stage, 1.5, "completed")
    metrics.record_attempt("embedding", retry=False)
    metrics.record_attempt("embedding", retry=True)
    metrics.record_job("embed", "completed")
    metrics.record_job("graph", "failed")
    metrics.set_index_gauges(queue_depth=3, inflight=1)
    rendered = metrics.render_prometheus()
    assert "agent_memory_stage_duration_seconds" in rendered
    for bucket in (1, 2, 6, 8, 10):
        assert f'le="{bucket}"' in rendered
    assert 'agent_memory_index_queue_depth 3' in rendered
    assert 'agent_memory_index_inflight 1' in rendered
    assert 'agent_memory_stage_attempts_total{stage="embedding"} 2' in rendered
    assert 'agent_memory_stage_retries_total{stage="embedding"} 1' in rendered
    for args in [("PRIVATE_ID", "total", 1, "completed"),
                 ("capture", "PRIVATE_CONTENT", 1, "completed"),
                 ("capture", "total", 1, "PRIVATE_ERROR"),
                 ("capture", "total", float("nan"), "completed"),
                 ("capture", "total", float("inf"), "completed")]:
        with pytest.raises(ValueError) as error:
            metrics.record_stage(*args)
        assert "PRIVATE" not in str(error.value)
    assert "PRIVATE" not in metrics.render_prometheus()
    metrics.reset()
    assert 'stage="total"' not in metrics.render_prometheus()


def test_request_metrics_never_label_dynamic_identity():
    metrics = MetricsCollector()
    identity = str(uuid4())
    metrics.record_request("GET", "/v1/operations/" + identity, 200, .01)
    metrics.record_request("GET", "/PRIVATE_CONTENT", 404, .01)
    text = metrics.render_prometheus()
    assert identity not in text
    assert "PRIVATE_CONTENT" not in text
    assert '/v1/operations/{request_id}' in text
