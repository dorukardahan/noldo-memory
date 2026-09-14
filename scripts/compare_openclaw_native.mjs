/** Equal-corpus, no-model lexical comparison using the installed native manager.
 * Requires an isolated HOME/state directory and temporary NoldoMem HTTP API.
 * Neither result is generated-answer or automatic conversation capture evidence.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
import {performance} from 'node:perf_hooks';

const [host, candidate, endpoint] = process.argv.slice(2);
assert(host && candidate && endpoint);
const home = process.env.HOME;
assert(process.env.OPENCLAW_STATE_DIR.startsWith(home + path.sep));
assert(new URL(endpoint).hostname === '127.0.0.1');
const pkg = JSON.parse(await fs.readFile(path.join(host, 'package.json'), 'utf8'));
assert.equal(pkg.version, '2026.9.3');
const corpus = JSON.parse(await fs.readFile(path.join(candidate, 'tests/fixtures/alignment_cases.json'), 'utf8'));
const originalFetch = globalThis.fetch;
let deniedNetwork = 0;
globalThis.fetch = async (url, options) => {
  if (!String(url).startsWith(endpoint + '/')) {
    deniedNetwork += 1;
    throw Error('External network is disabled in the lexical comparison');
  }
  return originalFetch(url, options);
};
async function api(route, body) {
  const response = await fetch(endpoint + route, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  assert(response.ok, `Temporary API failed: ${response.status}`);
  return response.json();
}
const {MemoryIndexManager, closeAllMemoryIndexManagers} = await import(pathToFileURL(path.join(host, 'dist/extensions/memory-core/manager-runtime.js')));
const cfg = {
  plugins: {enabled: true, allow: ['memory-core'], slots: {memory: 'memory-core'}, entries: {'memory-core': {enabled: true}}},
  memory: {search: {enabled: true, provider: 'none', fallback: 'none',
    sources: ['memory'], rememberAcrossConversations: false,
    store: {path: path.join(home, 'indexes', '{agentId}.sqlite'), vector: {enabled: false}},
    query: {maxResults: 5, minScore: 0},
    sync: {watch: false, onSessionStart: false, onSearch: false},
  }},
  agents: {list: ['alpha', 'beta'].map(id => ({id, workspace: path.join(home, id)}))},
};
// The official loader binds Memory Core's real keyed SQLite state service.
async function nativeExport(prefix, name) {
  const dist = path.join(host, 'dist');
  for (const file of (await fs.readdir(dist)).filter(f => f.startsWith(prefix) && f.endsWith('.mjs'))) {
    const text = await fs.readFile(path.join(dist, file), 'utf8');
    const exports = text.match(/export \{([^}]+)\};?\s*$/s)?.[1];
    const alias = exports?.match(new RegExp(`\\b${name} as ([\\w$]+)(?:,|\\s|$)`))?.[1];
    if (alias) return (await import(pathToFileURL(path.join(dist, file))))[alias];
  }
  throw Error(`Installed runtime export unavailable: ${name}`);
}
const load = await nativeExport('loader-runtime-load-', 'loadOpenClawPlugins');
const registry = load({config: cfg, workspaceDir: home,
  env: {HOME: home, OPENCLAW_STATE_DIR: process.env.OPENCLAW_STATE_DIR},
  cache: false, activate: false, allowProcessHomeSessionCatalogs: false,
  onlyPluginIds: ['memory-core'], throwOnLoadError: true,
  logger: {info() {}, warn() {}, error() {}, debug() {}},
});
assert(registry.plugins.some(p => p.id === 'memory-core' && p.status === 'loaded'), 'Native Memory Core failed to load');
try {
  const native = {};
  for (const agent of ['alpha', 'beta']) {
    const directory = path.join(home, agent, 'memory');
    await fs.mkdir(directory, {recursive: true});
    const episodes = agent === 'alpha' ? corpus.episodes : [{text: 'The Aurora observatory has an amber dome. Its separate workshop starts at 09:00.'}];
    for (const [index, item] of episodes.entries()) {
      await fs.writeFile(path.join(directory, `entry-${index}.md`), item.text + '\n');
      const stored = await api('/v1/store', {agent, text: item.text, session_id: 'synthetic-learning'});
      assert(stored.stored);
    }
    native[agent] = await MemoryIndexManager.get({cfg, agentId: agent, purpose: 'cli'});
    assert(native[agent], 'Native manager unavailable');
    await native[agent].sync({reason: 'synthetic-comparison', force: true});
    assert(native[agent].status().fts?.available, 'Native FTS unavailable');
  }
  const samples = [];
  const tasks = [...corpus.episodes.map(e => ({query: e.query, expected: e.id})), ...corpus.unrelated.map(query => ({query, expected: null}))];
  for (const pass of ['first-query', 'repeated-query']) {
    for (const task of tasks) {
      for (const system of ['native', 'noldomem']) {
        const start = performance.now();
        const result = system === 'native'
          ? await native.alpha.search(task.query, {maxResults: 5, minScore: 0})
          : (await api('/v1/recall', {agent: 'alpha', query: task.query, limit: 5, min_score: 0})).results;
        const elapsedMs = performance.now() - start;
        const snippets = result.map(r => r.snippet ?? r.text);
        assert(!snippets.some(t => t.includes('amber dome') || t.includes('09:00')), 'Cross-agent result');
        const ids = snippets.map(t => corpus.episodes.find(e => t.includes(e.text))?.id ?? 'unmapped');
        samples.push({system, pass, ...task, retrieved: ids, expected_rank: task.expected ? ids.indexOf(task.expected) + 1 : null,
          returned_chars: snippets.reduce((n,t) => n+t.length, 0), elapsed_ms: Number(elapsedMs.toFixed(3))});
      }
    }
  }
  const beta = await native.beta.search('amber dome', {maxResults: 5, minScore: 0});
  assert(beta.some(r => r.snippet.includes('amber dome')), 'Other-agent fixture was not indexed');
  assert(deniedNetwork === 0);
  console.log(JSON.stringify({host_version: pkg.version, node: process.version,
    input_episodes: corpus.episodes.length, unrelated_queries: corpus.unrelated.length,
    native_provider: 'none', native_fts: native.alpha.status().fts,
    external_fetch_attempts: deniedNetwork, model_calls: 0, embedding_calls: 0,
    input_mode: 'native Markdown files versus NoldoMem HTTP store; no automatic capture or model answer claim',
    latency_scope: 'native in-process search versus NoldoMem loopback HTTP; descriptive samples only, not a speedup claim',
    samples}, null, 2));
} finally {
  await closeAllMemoryIndexManagers();
  globalThis.fetch = originalFetch;
}
