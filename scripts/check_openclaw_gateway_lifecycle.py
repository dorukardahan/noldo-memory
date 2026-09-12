"""Run the installed Gateway lifecycle probe with an owned temporary API/DB.

No model, auth, channel send, embedding request or production profile is used.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import threading
import time


def check(host, node):
    import uvicorn
    import agent_memory.api as api
    from agent_memory.config import Config
    from agent_memory.pool import StoragePool
    from agent_memory.search import SearchWeights

    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="native-lifecycle-") as scratch:
        root = Path(scratch)
        state = root / "state"
        state.mkdir()
        (root / "tmp").mkdir()
        env = {"HOME": str(root), "OPENCLAW_STATE_DIR": str(state),
               "OPENCLAW_CONFIG_PATH": str(state / "openclaw.json"),
               "CODEX_HOME": str(root / "codex"), "TMPDIR": str(root / "tmp"),
               "PATH": str(node.parent) + ":/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
        api._storage_pool = StoragePool(str(root / "synthetic-db"), dimensions=4)
        api._config = Config(api_key="")
        api._embedder = None
        api._search_cache = {}
        api._search_weights = SearchWeights()
        api._start_time = time.time()
        requests = []

        @api.app.middleware("http")
        async def trace(request, call_next):
            if request.method == "POST":
                requests.append({"path": request.url.path, "body": await request.json()})
            return await call_next(request)

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        endpoint = f"http://127.0.0.1:{sock.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(api.app, lifespan="off", log_level="critical", access_log=False))
        def serve():
            try:
                server.run(sockets=[sock])
            finally:
                api._storage_pool.close_all()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(.01)
        assert server.started
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            gateway_port = reservation.getsockname()[1]
        config = {
            "gateway": {"mode": "local", "bind": "loopback", "port": gateway_port,
                        "auth": {"mode": "none"}, "controlUi": {"enabled": False}},
            "discovery": {"mdns": {"mode": "off"}},
            "plugins": {"enabled": True, "allow": ["noldomem"], "slots": {"memory": "none"},
                        "load": {"paths": [str(repo / "plugin")]},
                        "entries": {"noldomem": {"enabled": True,
                            "hooks": {"allowConversationAccess": True, "allowPromptInjection": True},
                            "config": {"baseUrl": endpoint, "apiKeyFile": "/dev/null",
                                "enableAutoRecall": True, "enableAutoCapture": True,
                                "enableOperationalCapture": False, "enableCompactionCapture": False,
                                "enableSubagentCapture": False}}}},
        }
        proc = None
        try:
            patched = subprocess.run([str(node), str(host / "openclaw.mjs"), "config", "patch", "--stdin"],
                                     input=json.dumps(config), env=env, text=True, capture_output=True, timeout=60)
            assert patched.returncode == 0, patched.stderr
            proc = subprocess.Popen([str(node), str(repo / "scripts/check_openclaw_gateway_lifecycle.mjs"),
                                     str(host), str(repo), endpoint, str(gateway_port)],
                                    env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    start_new_session=True)
            stdout, stderr = proc.communicate(timeout=120)
            assert proc.returncode == 0, stdout + stderr
            lines = [line for line in stdout.splitlines() if line.startswith("LIFECYCLE_RESULT=")]
            assert len(lines) == 1, stdout + stderr
            result = json.loads(lines[0].split("=", 1)[1])
            assert [r["path"] for r in requests] == ["/v1/store", "/v1/recall", "/v1/recall"], requests
            assert requests[0]["body"]["agent"] == "alpha"
            assert requests[1]["body"]["agent"] == "alpha"
            assert requests[2]["body"]["agent"] == "beta"
            result["http_requests"] = requests
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.communicate()
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive()
        result["owned_fixture_removed"] = True
    assert not root.exists()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", type=Path)
    parser.add_argument("node", type=Path)
    args = parser.parse_args()
    check(args.host, args.node)
