/** Native installed loader/hook-runner against a candidate and temporary API.
 * Requires an isolated HOME and supplied installed runtime. Never runs a Gateway.
 */
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const [host, candidate, endpoint] = process.argv.slice(2);
assert(host && candidate && endpoint, 'HOST CANDIDATE ENDPOINT required');
assert(process.env.OPENCLAW_STATE_DIR?.startsWith(process.env.HOME + path.sep));
const pkg = JSON.parse(fs.readFileSync(path.join(host, 'package.json'), 'utf8'));
assert.equal(pkg.version, '2026.9.3');

async function nativeExport(prefix, name) {
  const dist = path.join(host, 'dist');
  for (const file of fs.readdirSync(dist).filter(f => f.startsWith(prefix) && f.endsWith('.mjs'))) {
    const text = fs.readFileSync(path.join(dist, file), 'utf8');
    const exports = text.match(/export \{([^}]+)\};?\s*$/s)?.[1];
    const alias = exports?.match(new RegExp(`\\b${name} as ([\\w$]+)(?:,|\\s|$)`))?.[1];
    if (alias) return (await import(pathToFileURL(path.join(dist, file))))[alias];
  }
  throw Error(`Installed runtime export unavailable: ${name}`);
}

const load = await nativeExport('loader-runtime-load-', 'loadOpenClawPlugins');
const createHookRunner = await nativeExport('hook-runner-global-', 'createHookRunner');
const registry = load({
  config: {plugins: {enabled: true, allow: ['noldomem'], slots: {memory: 'none'},
    load: {paths: [path.join(candidate, 'plugin')]},
    entries: {noldomem: {enabled: true, hooks: {allowConversationAccess: true}, config: {
      baseUrl: endpoint, apiKeyFile: '/dev/null', enableAutoRecall: true,
      enableAutoCapture: true, recallMinSemanticScore: .5,
      enableOperationalCapture: false, enableCompactionCapture: false, enableSubagentCapture: false,
    }}},
  }},
  workspaceDir: process.env.HOME, env: {
    HOME: process.env.HOME, OPENCLAW_STATE_DIR: process.env.OPENCLAW_STATE_DIR,
  },
  cache: false, activate: false, allowProcessHomeSessionCatalogs: false,
  onlyPluginIds: ['noldomem'], throwOnLoadError: true,
  logger: {info() {}, warn() {}, error() {}, debug() {}},
});
const plugin = registry.plugins.find(p => p.id === 'noldomem');
assert(plugin && plugin.status !== 'error', JSON.stringify(registry.diagnostics));
assert(plugin.source.startsWith(path.join(candidate, 'plugin') + path.sep), 'Loaded a different plugin copy');
const hooks = createHookRunner(registry, {catchErrors: false});
console.log(JSON.stringify({plugin_status: plugin.status,
  hook_names: registry.typedHooks.map(h => h.hookName),
  diagnostics: registry.diagnostics.map(d => ({level: d.level, message: d.message}))}));
const corpus = JSON.parse(fs.readFileSync(path.join(candidate, 'tests/fixtures/alignment_cases.json'), 'utf8'));
const episode = corpus.episodes[0];
const ctx = {agentId: 'alpha', sessionKey: 'agent:alpha:synthetic-a'};
await hooks.runAgentEnd({success: true, messages: [{role: 'user', content:
  '[Audio]\nTranscript:\n[Voice note could not be transcribed because the audio attachment was too small]'}]}, ctx);
const failedMediaRows = await (await fetch(endpoint + '/v1/export?agent=alpha')).json();
assert.equal(failedMediaRows.length, 0, 'Native empty-audio placeholder was captured');
await hooks.runAgentEnd({success: true, messages: [{role: 'user', content: episode.text}]}, ctx);
const captured = await (await fetch(endpoint + '/v1/export?agent=alpha')).json();
assert(captured.some(r => r.text === episode.text), 'Native agent_end did not capture the current turn');
const recall = await hooks.runBeforePromptBuild({prompt: episode.query, messages: []},
  {agentId: 'alpha', sessionKey: 'agent:alpha:synthetic-b'});
assert(recall?.prependContext?.includes(episode.text), 'Cross-session automatic context missing');
const other = await hooks.runBeforePromptBuild({prompt: episode.query, messages: []},
  {agentId: 'beta', sessionKey: 'agent:beta:synthetic-b'});
assert(!other?.prependContext, 'Cross-agent context leakage');
await hooks.runMessageSent({content: episode.text, success: true, messageId: 'synthetic-delivery'},
  {channelId: 'synthetic', sessionKey: 'agent:alpha:synthetic-a'});
const rows = await (await fetch(endpoint + '/v1/export?agent=alpha')).json();
assert(rows.some(r => r.evidence.delivery === 'delivered'));
assert(rows.some(r => r.evidence.delivery === 'received'));
assert(registry.tools.some(t => t.names?.includes('noldomem_forget')));
const media = JSON.parse(fs.readFileSync(path.join(candidate, 'tests/fixtures/host_media_cases.json'), 'utf8'));
for (const item of media.cases) {
  // The installed formatter is private. This independently authored fixture
  // matches the pinned source contract; actual OCR/ASR is not invoked here.
  const body = item.body;
  await hooks.runAgentEnd({success: true, messages: [{role: 'user', content: body}]}, ctx);
  const recalled = await hooks.runBeforePromptBuild({prompt: item.query, messages: []},
    {agentId: 'alpha', sessionKey: 'agent:alpha:synthetic-c'});
  assert(recalled?.prependContext?.includes(item.derivative), `Missing ${item.id} event context`);
  const stored = await (await fetch(endpoint + '/v1/export?agent=alpha')).json();
  const row = stored.find(r => r.text === item.body);
  assert.equal(row?.evidence.modality, item.modality);
  assert.equal(row?.evidence.assertion, 'derived');
}
// Native tool factories and actual capture/injection, without a model call.
const tools = registry.tools.flatMap(t => t.factory(ctx) || []);
const forgottenId = captured.find(r => r.text === episode.text).id;
const forget = tools.find(t => t.name === 'noldomem_forget');
const receipt = (await forget.execute('synthetic-forget', {memory_id: forgottenId})).details;
assert(receipt.deleted);
// Remove the independently delivered copy too, so no retained source explains recall.
for (const row of await (await fetch(endpoint + '/v1/export?agent=alpha')).json()) {
  if (row.text === episode.text) assert((await forget.execute('synthetic-forget-copy', {memory_id: row.id})).details.deleted);
}
await hooks.runAgentEnd({success: true, messages: [{role: 'user', content: episode.text}]}, ctx);
assert(!(await (await fetch(endpoint + '/v1/export?agent=alpha')).json()).some(r => r.text === episode.text));
const afterForget = await hooks.runBeforePromptBuild({prompt: episode.query, messages: []},
  {agentId: 'alpha', sessionKey: 'agent:alpha:synthetic-fresh'});
assert(!afterForget?.prependContext?.includes(episode.text));
const relearn = tools.find(t => t.name === 'noldomem_relearn_source');
assert.deepEqual((await relearn.execute('synthetic-relearn', {source_key: receipt.source_keys[0]})).details,
  {cleared: true, restored: false});
await hooks.runAgentEnd({success: true, messages: [{role: 'user', content: episode.text}]}, ctx);
assert((await (await fetch(endpoint + '/v1/export?agent=alpha')).json()).some(r => r.text === episode.text));
console.log(JSON.stringify({host_version: pkg.version, node: process.version,
  candidate_native_loader: true, native_hook_runner: true, real_http: true,
  cross_session_implicit_injection: true, received_and_delivered_text: true,
  agent_isolation: true, source_replay_blocked: true, explicit_relearning: true, empty_audio_abstention: true, synthetic_host_format_media_derivatives: media.cases.length,
  raw_media_extraction: 'not invoked; synthetic extractor output', generated_answers: 'not measured'}));
