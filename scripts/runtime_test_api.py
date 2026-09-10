"""Temporary synthetic API orchestrator; explicit installed Node and candidate.

Run with an isolated child environment. No config loader, credentials or live DB.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time


def run(args):
    sys.path.insert(0, str(args.candidate))
    import uvicorn
    import agent_memory.api as api
    from agent_memory.config import Config
    from agent_memory.pool import StoragePool
    from agent_memory.search import SearchWeights, normalize_query
    snapshot = json.loads(gzip.decompress((args.candidate / 'tests/fixtures/alignment_vectors.json.gz').read_bytes()))
    vectors = {normalize_query(t): v for t, v in zip(snapshot['texts'], snapshot['vectors'])}
    media_snapshot = json.loads(gzip.decompress((args.candidate / 'tests/fixtures/host_media_vectors.json.gz').read_bytes()))
    vectors.update({normalize_query(t): v for t, v in zip(media_snapshot['texts'], media_snapshot['vectors'])})
    class RecordedEmbedder:
        async def embed(self, text):
            return vectors[normalize_query(text)]
        async def embed_batch(self, texts):
            return [await self.embed(text) for text in texts]
        def set_storage(self, storage):
            pass
    api._config = Config(api_key='', embed_worker_enabled=False)
    api._storage_pool = StoragePool(str(Path.home() / 'synthetic-db'), dimensions=1024)
    api._embedder = None if args.lexical_comparison else RecordedEmbedder()
    api._search_cache, api._kg_cache = {}, {}
    api._search_weights = SearchWeights()
    api._start_time = time.time()
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    endpoint = 'http://127.0.0.1:' + str(sock.getsockname()[1])
    server = uvicorn.Server(uvicorn.Config(api.app, lifespan='off', log_level='critical', access_log=False))
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
    try:
        env = {key: os.environ[key] for key in ['HOME', 'TMPDIR', 'PATH', 'OPENCLAW_STATE_DIR']}
        script = ('scripts/compare_openclaw_native.mjs' if args.lexical_comparison else
                  'scripts/model_openclaw_bridge.mjs' if args.model_bridge else 'scripts/check_openclaw_runtime.mjs')
        result = subprocess.run([str(args.node), str(args.candidate / script),
                                 str(args.host), str(args.candidate), endpoint], env=env, timeout=90)
        return result.returncode
    finally:
        server.should_exit = True
        thread.join(5)
        sock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lexical-comparison', action='store_true', help='Compare native and NoldoMem FTS without embeddings or models.')
    parser.add_argument('--model-bridge', action='store_true', help='Run the synthetic stdin bridge, without a model call.')
    for name in ['node', 'host', 'candidate']:
        parser.add_argument('--' + name, type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
