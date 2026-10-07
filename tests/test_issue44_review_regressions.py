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
