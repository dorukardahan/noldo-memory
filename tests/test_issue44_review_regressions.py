import json
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests import test_api
from agent_memory import api, operations
from agent_memory import pool as pool_module
from agent_memory.storage import MemoryStorage

client = test_api.client
_init_api_state = test_api._init_api_state


@pytest.mark.asyncio
async def test_identical_revision_request_id_replay_is_idempotent(client):
    original = await client.post(
        "/v1/store",
        json={"text": "The synthetic observatory has a violet dome."},
    )
    assert original.status_code == 200, original.text
    request_id = str(uuid.uuid4())
    request = {
        "request_id": request_id,
        "text": "The synthetic observatory has a silver dome.",
        "supersedes": original.json()["id"],
        "valid_from": 1234.0,
    }
    first = await client.post("/v1/store", json=request)
    assert first.status_code == 200, first.text
    replay = await client.post("/v1/store", json=request)
    assert replay.status_code == 200, replay.text
    assert replay.json()["request_id"] == request_id
    assert replay.json()["id"] == first.json()["id"]
    assert api._get_storage().stats()["total_memories"] == 2


@pytest.mark.asyncio
async def test_forgotten_revision_identity_replay_stays_blocked(client):
    original = await client.post(
        "/v1/store",
        json={"text": "The synthetic revision family starts here."},
    )
    assert original.status_code == 200, original.text
    request_id = str(uuid.uuid4())
    request = {
        "request_id": request_id,
        "text": "The synthetic revision family changes here.",
        "supersedes": original.json()["id"],
        "valid_from": 1234.0,
        "session_id": "synthetic-revision-source",
    }
    first = await client.post("/v1/store", json=request)
    assert first.status_code == 200, first.text
    api._get_storage().forget_memory(first.json()["id"])
    status = await client.get(
        f"/v1/operations/{request_id}",
        params={"operation": "store", "namespace": "default"},
    )
    assert status.status_code == 200, status.text
    assert status.json()["state"] == "blocked"
    replay = await client.post("/v1/store", json=request)
    assert replay.status_code == 200, replay.text
    assert replay.json()["state"] == "blocked"


def test_public_status_preserves_durable_false_failed_ledger_state(tmp_path):
    storage = MemoryStorage(str(tmp_path / "failed.sqlite"), dimensions=4)
    request_id = str(uuid.uuid4())
    now = time.time()
    try:
        with storage.transaction() as conn:
            conn.execute(
                "INSERT INTO memory_operations VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "default",
                    "store",
                    request_id,
                    operations.fingerprint({"rows": []}),
                    "failed",
                    0,
                    json.dumps({"stored": 0, "merged": 0, "blocked": 0, "total": 1}),
                    "[]",
                    "admission_failed",
                    now,
                    now,
                ),
            )
        result = operations.public_status(storage, "default", "store", request_id)
        assert result["state"] == "failed"
        assert result["durable"] is False
        assert result["error_code"] == "admission_failed"
    finally:
        storage.close()


def test_storage_pool_concurrent_first_access_is_singleton(tmp_path, monkeypatch):
    created = []
    start = threading.Barrier(8)

    class SlowStorage:
        def __init__(self, *, db_path, dimensions):
            time.sleep(0.05)
            created.append((db_path, dimensions, self))

        def close(self):
            return None

    monkeypatch.setattr(pool_module, "MemoryStorage", SlowStorage)
    pool = pool_module.StoragePool(str(tmp_path), dimensions=4)

    def get_once(_):
        start.wait(timeout=2)
        return pool.get("main")

    with ThreadPoolExecutor(max_workers=8) as executor:
        storages = list(executor.map(get_once, range(8)))

    assert len(created) == 1
    assert all(storage is storages[0] for storage in storages)


@pytest.mark.asyncio
async def test_storage_busy_rolls_back_rows_jobs_and_operation(client, monkeypatch):
    request_id = str(uuid.uuid4())
    original = MemoryStorage.merge_or_store
    calls = 0

    def fail_after_first_insert(self, **kwargs):
        nonlocal calls
        calls += 1
        result = original(self, **kwargs)
        if calls == 1:
            raise sqlite3.OperationalError("synthetic lock")
        return result

    monkeypatch.setattr(MemoryStorage, "merge_or_store", fail_after_first_insert)
    response = await client.post(
        "/v1/capture",
        json={
            "request_id": request_id,
            "messages": [
                {
                    "role": "user",
                    "text": "Synthetic storage-busy rollback assertion.",
                    "session": "synthetic-storage-busy",
                }
            ],
        },
    )
    assert response.status_code == 503
    assert response.json()["error"] == "storage_busy"
    storage = api._get_storage()
    assert storage.stats()["total_memories"] == 0
    assert storage._get_conn().execute("SELECT count(*) FROM memory_index_jobs").fetchone()[0] == 0
    assert storage._get_conn().execute("SELECT count(*) FROM memory_operations").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_python310_foreground_timeout_maps_to_bounded_deadline(tmp_path, monkeypatch):
    from agent_memory import foreground as foreground_module
    from agent_memory.foreground import Foreground
    from agent_memory.pool import StoragePool

    class Python310AsyncioTimeout(Exception):
        pass

    async def raise_compat_timeout(awaitable, timeout):
        awaitable.cancel()
        raise Python310AsyncioTimeout

    pool = StoragePool(str(tmp_path), dimensions=4)
    foreground = Foreground(pool)
    monkeypatch.setattr(foreground_module.asyncio, "TimeoutError", Python310AsyncioTimeout)
    monkeypatch.setattr(foreground_module.asyncio, "wait_for", raise_compat_timeout)
    try:
        with pytest.raises(operations.AdmissionError, match="deadline_exceeded") as exc:
            await foreground.run("main", time.monotonic() + 1, lambda storage: None)
        assert exc.value.status == 504
    finally:
        await foreground.stop()
        pool.close_all()


@pytest.mark.asyncio
async def test_python310_index_timeout_keeps_deadline_error_code(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from agent_memory import index_worker as worker_module

    class Python310AsyncioTimeout(Exception):
        pass

    class TimeoutEmbedder:
        async def embed(self, text, **kwargs):
            raise Python310AsyncioTimeout

    pool = pool_module.StoragePool(str(tmp_path), dimensions=4)
    task = worker_module.IndexWorker(
        pool,
        TimeoutEmbedder(),
        SimpleNamespace(index_job_timeout_seconds=1, index_job_max_attempts=3),
    )
    failures = []

    async def offload(fn, *args):
        if fn.__name__ == "_memory":
            return {"text": "Synthetic timeout assertion", "source_session": "synthetic"}
        if fn.__name__ == "_fail":
            failures.append(args[1])
            return True
        raise AssertionError(fn.__name__)

    task._offload = offload
    monkeypatch.setattr(worker_module.asyncio, "TimeoutError", Python310AsyncioTimeout)
    try:
        await task._execute({"stage": "embed", "memory_id": "synthetic", "attempts": 1})
    finally:
        pool.close_all()
    assert failures == ["index_deadline_exceeded"]


@pytest.mark.asyncio
async def test_python310_recall_timeout_maps_to_504(client, monkeypatch):
    from fastapi import HTTPException
    from starlette.requests import Request

    class Python310AsyncioTimeout(Exception):
        pass

    async def raise_compat_timeout(*args, **kwargs):
        raise Python310AsyncioTimeout

    monkeypatch.setattr(api.asyncio, "TimeoutError", Python310AsyncioTimeout)
    monkeypatch.setattr(api, "_foreground", raise_compat_timeout)
    request = Request({"type": "http", "method": "POST", "path": "/v1/recall", "headers": []})
    request.state.allowed_agent = None
    with pytest.raises(HTTPException) as exc:
        await api.recall(api.RecallRequest(query="Synthetic timeout query"), request)
    assert exc.value.status_code == 504
    assert exc.value.detail == "recall_deadline_exceeded"


@pytest.mark.asyncio
async def test_revision_validity_conflict_remains_http_409(client):
    storage = api._get_storage()
    previous_id = storage.store_memory(
        "The synthetic observatory opens after midnight.",
        valid_from=2_000.0,
    )
    response = await client.post(
        "/v1/store",
        json={
            "text": "The synthetic observatory now opens before midnight.",
            "supersedes": previous_id,
            "valid_from": 1_000.0,
        },
    )
    assert response.status_code == 409, response.text
    assert api._get_storage().stats()["total_memories"] == 1
