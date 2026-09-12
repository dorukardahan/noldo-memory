"""Exercise a supplied Hermes source checkout against a temporary real HTTP API.

Runs the real provider discovery/MemoryManager/MemoryStore. It does not start a
model, gateway, external embedding service, or use a saved credential/profile.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time


def check_memory_echo(manager, endpoint, repo):
    import httpx

    text = 'The Aurora observatory guide is Kestrel.'
    stored = json.loads(manager.handle_tool_call('noldomem_store', {'text': text}))['data']
    recalled = manager.handle_tool_call('noldomem_recall', {'query': 'Aurora observatory guide'})
    assert text in recalled
    rows = [
        {'role': 'user', 'content': 'Check my booking.'},
        {'role': 'assistant', 'content': '', 'tool_calls': [
            {'id': 'recall-call', 'function': {'name': 'noldomem_recall'}},
            {'id': 'document-call', 'function': {'name': 'read_document'}},
        ]},
        {'role': 'tool', 'tool_call_id': 'recall-call', 'content': recalled},
        {'role': 'tool', 'tool_call_id': 'document-call',
         'content': 'The document describes a violet observatory dome.'},
        {'role': 'assistant', 'content': 'The document is available.'},
    ]
    manager.on_session_switch('session-b')
    for _ in range(2):
        manager.sync_all(rows[0]['content'], rows[-1]['content'], session_id='session-b', messages=rows)
        assert manager.flush_pending(timeout=5)
    exported = httpx.get(endpoint + '/v1/export', params={'agent': 'alpha'}).json()
    assert [row['text'] for row in exported if 'Kestrel' in row['text']] == [text]
    assert any('violet' in row['text'] and row['evidence']['assertion'] == 'derived' for row in exported)
    assert json.loads(manager.handle_tool_call('noldomem_forget', {'memory_id': stored['id']}))['data']['deleted']
    manager.sync_all(rows[0]['content'], rows[-1]['content'], session_id='session-b', messages=rows)
    assert manager.flush_pending(timeout=5)
    assert 'Kestrel' not in manager.prefetch_all('Aurora observatory guide', session_id='session-b')
    assert not any('Kestrel' in row['text'] for row in httpx.get(endpoint + '/v1/export', params={'agent': 'alpha'}).json())
    print(json.dumps({'native_loader_manager_http': True, 'recursive_memory_capture_excluded': True,
                      'external_document_result_retained': True, 'forgotten_tool_echo_replay_excluded': True,
                      'adapter_sha256': hashlib.sha256((repo / 'adapters/hermes/noldomem/__init__.py').read_bytes()).hexdigest(),
                      'model_calls': 0}))


def check(host, repo, evidence_only=False, audio_provenance=False, memory_echo_only=False):
    sys.path[:0] = [str(host), str(repo)]
    import uvicorn
    import httpx
    import agent_memory.api as api
    from agent_memory.config import Config
    from agent_memory.pool import StoragePool
    from agent_memory.search import SearchWeights
    from agent.memory_manager import MemoryManager
    from plugins.memory import load_memory_provider
    from tools.memory_tool import MemoryStore

    profile = Path(os.environ['HERMES_HOME'])
    plugin = profile / 'plugins' / 'noldomem'
    shutil.copytree(repo / 'adapters/hermes/noldomem', plugin)
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
    try:
        cfg = module.NoldoMemConfig(base_url=endpoint, api_key='YOUR_API_KEY', agent='alpha', sync_turns_enabled=True)
        provider.load_config = lambda *args, **kwargs: cfg  # Synthetic config, no credential-file read.
        manager.add_provider(provider)
        manager.initialize_all('session-a')
        if memory_echo_only:
            check_memory_echo(manager, endpoint, repo)
            return
        if evidence_only or audio_provenance:
            import asyncio
            from types import SimpleNamespace
            from gateway.run_inbound import GatewayInboundMixin
            from agent.turn_context import _stage_turn_user_message

            transcript = 'The Aurora observatory booking starts at 19:30 on Friday.'
            def synthetic_stt(*args):
                return {'success': True, 'transcript': transcript}
            def no_fallback(*args):
                raise AssertionError('Successful STT must not call fallback')
            original, quoted = asyncio.run(GatewayInboundMixin()._transcribe_one_clip(
                'synthetic-voice.ogg', synthetic_stt, no_fallback))
            assert original == transcript and quoted == f'"{transcript}"'
            # Execute the real host row builder: neither the returned quoted text
            # nor its durable row identifies audio. No private state is inspected.
            row, _ = _stage_turn_user_message(SimpleNamespace(), quoted, None,
                1700000123.456, 'synthetic-platform-1', None, None)
            if audio_provenance:
                from unittest.mock import patch
                from gateway.config import GatewayConfig, Platform
                from gateway.platforms.event import MessageEvent, MessageType
                from gateway.run import GatewayRunner
                from gateway.session import SessionSource
                from gateway.transcription_metadata import user_display_metadata

                event = MessageEvent(text='', message_type=MessageType.VOICE,
                    source=SessionSource(platform=Platform.TELEGRAM, chat_id='synthetic-room', chat_type='dm'),
                    message_id='synthetic-platform-1', media_urls=['synthetic-voice.ogg'], media_types=['audio/ogg'])
                runner = GatewayRunner.__new__(GatewayRunner)
                runner.config = GatewayConfig(stt_enabled=True, stt_echo_transcripts=False)
                runner.adapters = {}
                runner._model = 'synthetic-model'
                runner._base_url = ''
                with patch('tools.transcription_tools.transcribe_audio', synthetic_stt), patch(
                    'tools.transcription_tools.transcribe_audio_local_fallback', no_fallback,
                ):
                    quoted = asyncio.run(runner._prepare_inbound_message_text(event=event, source=event.source, history=[]))
                assert quoted == f'"{transcript}"'
                row, _ = _stage_turn_user_message(SimpleNamespace(), quoted, None,
                    1700000123.456, event.message_id, None, user_display_metadata(event))
            manager.sync_all(quoted, 'Acknowledged.', session_id='session-a', messages=[row])
            assert manager.flush_pending(timeout=5)
            stored = httpx.get(endpoint + '/v1/export', params={'agent': 'alpha'}).json()
            assert len(stored) == 1
            evidence = stored[0]['evidence']
            assert evidence['event_id'] == 'synthetic-platform-1'
            assert evidence['observed_at'] == 1700000123.456
            assert evidence['modality'] == ('audio' if audio_provenance else 'text')
            if audio_provenance:
                assert evidence['assertion'] == 'derived' and evidence['representation'] == 'extracted_text'
                assert evidence['reference'] == 'synthetic-voice.ogg'
            manager.on_session_switch('session-b')
            context = manager.prefetch_all('When does the Aurora observatory booking start?', session_id='session-b')
            assert transcript in context
            if audio_provenance:
                assert 'modality=audio' in context and 'assertion=derived' in context
                assert httpx.get(endpoint + '/v1/export', params={'agent': 'beta'}).json() == []
                forgotten = json.loads(manager.handle_tool_call('noldomem_forget', {'memory_id': stored[0]['id']}))
                assert forgotten['data']['deleted']
                manager.sync_all(quoted, 'Acknowledged.', session_id='session-a', messages=[row])
                assert manager.flush_pending(timeout=5)
                assert httpx.get(endpoint + '/v1/export', params={'agent': 'alpha'}).json() == []
                assert transcript not in manager.prefetch_all('Aurora observatory booking', session_id='session-b')
            print(json.dumps({'host_commit': subprocess.check_output(['git', '-C', str(host), 'rev-parse', 'HEAD'], text=True).strip(),
                'host_dirty': bool(subprocess.check_output(['git', '-C', str(host), 'status', '--porcelain'], text=True).strip()),
                'adapter_sha256': hashlib.sha256((repo / 'adapters/hermes/noldomem/__init__.py').read_bytes()).hexdigest(),
                'native_successful_stt_formatter': True, 'stt_backend': 'synthetic, no media decode',
                'native_turn_row_builder': True, 'native_memory_manager_provider': True,
                'real_temporary_http_capture': True, 'host_event_id_and_time_preserved': True,
                'cross_session_injection': True, 'successful_audio_origin_still_unavailable_in_row': not audio_provenance,
                'candidate_gateway_audio_provenance': audio_provenance,
                'model_calls': 0}))
            return
        manager.sync_all('I prefer quiet evening observatory visits.', 'The plan includes quiet evenings.',
                         session_id='session-a', messages=[
                             {'role': 'user', 'content': [{'type': 'text', 'text': 'I prefer quiet evening observatory visits.'}]},
                             {'role': 'assistant', 'content': 'The plan includes quiet evenings.'},
                         ])
        assert manager.flush_pending(timeout=5)
        manager.on_session_switch('session-b')
        context = manager.prefetch_all('Plan the quiet observatory visit', session_id='session-b')
        assert 'quiet evening' in context and 'id=' in context
        rows = httpx.get(endpoint + '/v1/export', params={'agent': 'alpha'}).json()
        assert {row['evidence']['delivery'] for row in rows} == {'received', 'generated'}
        assert httpx.get(endpoint + '/v1/export', params={'agent': 'beta'}).json() == []
        forgotten_row = next(row for row in rows if row['text'] == 'I prefer quiet evening observatory visits.')
        forgotten = json.loads(manager.handle_tool_call('noldomem_forget', {'memory_id': forgotten_row['id']}))
        assert forgotten['data']['deleted']
        assert all(row['id'] != forgotten_row['id'] for row in httpx.get(endpoint + '/v1/export', params={'agent': 'alpha'}).json())
        manager.sync_all('I prefer quiet evening observatory visits.', 'The plan includes quiet evenings.',
                         session_id='session-a', messages=[
                             {'role': 'user', 'content': 'I prefer quiet evening observatory visits.'},
                         ])
        assert manager.flush_pending(timeout=5)
        assert not any(row['text'] == forgotten_row['text'] for row in httpx.get(endpoint + '/v1/export', params={'agent': 'alpha'}).json())
        relearned = json.loads(manager.handle_tool_call('noldomem_relearn_source', {'source_key': forgotten['data']['source_keys'][0]}))
        assert relearned['data'] == {'cleared': True, 'restored': False}
        manager.sync_all('I prefer quiet evening observatory visits.', 'The plan includes quiet evenings.',
                         session_id='session-a', messages=[
                             {'role': 'user', 'content': 'I prefer quiet evening observatory visits.'},
                         ])
        assert manager.flush_pending(timeout=5)
        assert any(row['text'] == forgotten_row['text'] for row in httpx.get(endpoint + '/v1/export', params={'agent': 'alpha'}).json())
        schemas = manager.get_all_tool_schemas()
        assert any(item['name'] == 'noldomem_recall' for item in schemas)
        # Hermes' real text-only vision envelope survives the provider boundary.
        # The caption is synthetic extractor output, not an actual vision call.
        vision = "[The user sent an image~ Here's what I can see:\nThe Aurora observatory dome is violet.]"
        manager.sync_all(vision, 'The caption is available.', session_id='session-b', messages=[
            {'role': 'user', 'content': vision},
            {'role': 'assistant', 'content': 'The caption is available.'},
        ])
        assert manager.flush_pending(timeout=5)
        media_rows = httpx.get(endpoint + '/v1/export', params={'agent': 'alpha'}).json()
        media = next(row for row in media_rows if row['text'] == vision)
        assert media['evidence']['modality'] == 'image'
        assert media['evidence']['assertion'] == 'derived'
        manager.on_session_switch('session-c')
        media_context = manager.prefetch_all('Choose the Aurora observatory dome color', session_id='session-c')
        assert 'violet' in media_context
        assert 'modality=image' in media_context and 'representation=extracted_text' in media_context
        # Pinned host failure output is not a spoken user fact. No STT call.
        empty_voice = ('[The user sent a voice message but it came through '
                       'empty or inaudible — speech-to-text returned no '
                       'words. Do not guess at the content; ask the user '
                       'to resend or type it out.]')
        before_empty = httpx.get(endpoint + '/v1/export', params={'agent': 'alpha'}).json()
        manager.sync_all(empty_voice, 'Please resend the clip.', session_id='voice-failure', messages=[
            {'role': 'user', 'content': empty_voice},
            {'role': 'assistant', 'content': 'Please resend the clip.'},
        ])
        assert manager.flush_pending(timeout=5)
        after_empty = httpx.get(endpoint + '/v1/export', params={'agent': 'alpha'}).json()
        assert after_empty == before_empty, 'Host empty-audio placeholder became persistent memory'
        typed = 'The synthetic observatory opens on Saturday.'
        manager.sync_all(empty_voice + '\n\n' + typed, 'The date is noted.', session_id='voice-mixed', messages=[
            {'role': 'user', 'content': empty_voice + '\n\n' + typed},
            {'role': 'assistant', 'content': 'The date is noted.'},
        ])
        assert manager.flush_pending(timeout=5)
        mixed_rows = httpx.get(endpoint + '/v1/export', params={'agent': 'alpha'}).json()
        assert any(row['text'] == typed and row['evidence']['assertion'] == 'reported' for row in mixed_rows)
        assert not any('speech-to-text returned no words' in row['text'] for row in mixed_rows)
        # Exact same public input corpus, native bounded startup snapshot.
        corpus = json.loads((repo / 'tests/fixtures/alignment_cases.json').read_text())
        native = MemoryStore(memory_char_limit=3500)
        native.load_from_disk()
        for episode in corpus['episodes']:
            assert native.add('memory', episode['text'])['success']
        reopened = MemoryStore(memory_char_limit=3500)
        reopened.load_from_disk()
        native_context = reopened.format_for_system_prompt('memory')
        assert all(episode['text'] in native_context for episode in corpus['episodes'])
        assert native.replace('memory', corpus['episodes'][0]['text'], 'I prefer morning observatory visits.')['success']
        assert native.remove('memory', 'I prefer morning observatory visits.')['success']
        # Separate authorities cannot propagate native replacement/deletion.
        try:
            provider.on_memory_write('remove', 'memory', 'I prefer quiet evening observatory visits.')
        except RuntimeError:
            mirror_refused = True
        else:
            mirror_refused = False
        assert mirror_refused
        print(json.dumps({'host_commit': subprocess.check_output(['git', '-C', str(host), 'rev-parse', 'HEAD'], text=True).strip(),
                          'real_provider_loader': True, 'real_http_capture_recall_injection': True,
                          'cross_session': True, 'unshared_agent_scope': True,
                          'embedding_mode': 'degraded lexical; no external calls',
                          'assistant_delivery': 'generated, not confirmed delivered',
                          'native_corpus_coverage': len(corpus['episodes']),
                          'native_context_chars': len(native_context),
                          'native_replace_remove': True, 'scoped_forget_over_http': True, 'duplicate_authority_mirror': 'unsupported; refused',
                          'synthetic_vision_derivative_across_sessions': True,
                          'structured_media_origin_in_context': True,
                          'empty_voice_failure_not_captured': True,
                          'typed_text_alongside_failed_voice_preserved': True,
                          'raw_media_extraction': 'not invoked; synthetic extractor output',
                          'generated_answer_accuracy': 'not measured'}, indent=2))
    finally:
        manager.shutdown_all()
        server.should_exit = True
        thread.join(5)
        sock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', type=Path, required=True)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--evidence-only', action='store_true', help='Only the new event metadata and native STT formatting checks.')
    parser.add_argument('--audio-provenance', action='store_true', help='Candidate host structured STT capture, injection and replay checks.')
    parser.add_argument('--memory-echo-only', action='store_true', help='Only native memory tool echo/replay capture checks; no model.')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.child:
        check(args.host.resolve(), repo, args.evidence_only, args.audio_provenance, args.memory_echo_only)
    else:
        with tempfile.TemporaryDirectory(prefix='noldomem-hermes-check-') as scratch:
            env = {'PATH': os.defpath + ':/opt/homebrew/bin:/usr/local/bin', 'HOME': scratch,
                   'HERMES_HOME': scratch, 'TMPDIR': scratch, 'LANG': 'en_US.UTF-8',
                   'AGENT_MEMORY_DATA_DIR': scratch, 'PYTHONDONTWRITEBYTECODE': '1'}
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--host', str(args.host.resolve()), '--child',
                                     *(['--evidence-only'] if args.evidence_only else []),
                                     *(['--audio-provenance'] if args.audio_provenance else []),
                                     *(['--memory-echo-only'] if args.memory_echo_only else [])],
                                    cwd=scratch, env=env)
            raise SystemExit(result.returncode)
