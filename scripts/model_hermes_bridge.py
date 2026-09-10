"""Synthetic stdin bridge through the real Hermes loader and MemoryManager.

Requires an isolated HOME/HERMES_HOME. Does not invoke a model or credentials.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time


def check(host, repo):
    sys.path[:0] = [str(host), str(repo)]
    import uvicorn
    import httpx
    import agent_memory.api as api
    from agent_memory.config import Config
    from agent_memory.pool import StoragePool
    from agent_memory.search import SearchWeights
    from agent.memory_manager import MemoryManager
    from plugins.memory import load_memory_provider

    profile = Path(os.environ['HERMES_HOME'])
    plugin = profile / 'plugins' / 'noldomem'
    shutil.copytree(repo / 'adapters/hermes/noldomem', plugin, dirs_exist_ok=True)
    provider = load_memory_provider('noldomem', register_skills=False)
    assert provider is not None, 'real stable provider loader rejected adapter'
    module = sys.modules[type(provider).__module__]
    api._storage_pool = StoragePool(str(profile / 'synthetic-db'), dimensions=4)
    api._config = Config(api_key='')
    class UnavailableEmbedder:
        async def embed(self, text):
            raise ConnectionError("Synthetic offline embedding outage")

        async def embed_batch(self, texts):
            raise ConnectionError("Synthetic offline embedding outage")
    api._embedder = UnavailableEmbedder()  # Failure injection, no fabricated vectors.
    api._search_cache = {}
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
    manager = MemoryManager()
    command = json.load(sys.stdin)
    assert command['agent'] in {'alpha', 'beta'}
    try:
        cfg = module.NoldoMemConfig(base_url=endpoint, api_key='YOUR_API_KEY', agent=command['agent'], sync_turns_enabled=True)
        provider.load_config = lambda *args, **kwargs: cfg
        manager.add_provider(provider)
        manager.initialize_all(command['session'])
        if command['action'] == 'capture':
            messages = command['messages']
            user = next(m['content'] for m in messages if m['role'] == 'user')
            assistant = next((m['content'] for m in messages if m['role'] == 'assistant'), 'Acknowledged.')
            manager.sync_all(user, assistant, session_id=command['session'], messages=messages)
            assert manager.flush_pending(timeout=5)
            result = {'captured': True}
        elif command['action'] == 'context':
            result = {'context': manager.prefetch_all(command['query'], session_id=command['session']),
                      'system': manager.build_system_prompt(), 'tools': manager.get_all_tool_schemas()}
        elif command['action'] == 'tool':
            result = json.loads(manager.handle_tool_call(command['name'], command['arguments']))
        else:
            raise ValueError('Unknown bridge action')
        rows = httpx.get(endpoint + '/v1/export', params={'agent': command['agent']}).json()
        print('ACCEPTANCE_JSON=' + json.dumps({'result': result, 'rows': rows,
              'host_commit': subprocess.check_output(['git', '-C', str(host), 'rev-parse', 'HEAD'], text=True).strip(),
              'real_provider_loader': True, 'real_memory_manager': True}))
    finally:
        manager.shutdown_all()
        server.should_exit = True
        thread.join(5)
        sock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', type=Path, required=True)
    args = parser.parse_args()
    assert Path(os.environ['HERMES_HOME']) == Path.home(), 'Use an isolated HOME and HERMES_HOME'
    check(args.host.resolve(), Path(__file__).resolve().parents[1])
