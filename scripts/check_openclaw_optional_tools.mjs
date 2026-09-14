/** Model-free installed-loader/tool/HTTP regression. Use only a temporary API/DB. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';

const [host, candidate, endpoint] = process.argv.slice(2);
assert(host && candidate && endpoint, 'HOST CANDIDATE TEMPORARY_API required');
assert.equal(new URL(endpoint).hostname, '127.0.0.1');
assert(process.env.OPENCLAW_STATE_DIR, 'Run inside an isolated native state');
const requireHost = createRequire(path.join(host, 'package.json'));
const {projectRuntimeToolInputSchema} = await import(pathToFileURL(
  requireHost.resolve('openclaw/plugin-sdk/agent-harness-runtime')));
const {validateJsonSchemaValue} = await import(pathToFileURL(
  requireHost.resolve('openclaw/plugin-sdk/json-schema-runtime')));
let load;
for (const file of fs.readdirSync(path.join(host, 'dist')).filter(f => f.startsWith('loader-runtime-load-') && f.endsWith('.mjs'))) {
  const source = fs.readFileSync(path.join(host, 'dist', file), 'utf8');
  const exported = source.match(/export \{([^}]+)\};?\s*$/s)?.[1];
  const alias = exported?.match(/\bloadOpenClawPlugins as ([\w$]+)(?:,|\s|$)/)?.[1];
  if (alias) { load = (await import(pathToFileURL(path.join(host, 'dist', file))))[alias]; break; }
}
assert(load, 'Installed native loader unavailable');
const registry = load({config: {plugins: {enabled: true, allow: ['noldomem'],
  slots: {memory: 'none'}, load: {paths: [path.join(candidate, 'plugin')]},
  entries: {noldomem: {enabled: true, config: {baseUrl: endpoint, apiKeyFile: '/dev/null',
    enableOperationalCapture: false, enableCompactionCapture: false, enableSubagentCapture: false}}},
}}, workspaceDir: process.env.HOME, cache: false, activate: false,
allowProcessHomeSessionCatalogs: false, onlyPluginIds: ['noldomem'], throwOnLoadError: true,
logger: {info() {}, warn() {}, error() {}, debug() {}}});
assert.equal(registry.plugins.find(p => p.id === 'noldomem')?.source, path.join(candidate, 'plugin/index.js'));
const agent = 'optional-schema-probe';
assert.deepEqual(await (await fetch(`${endpoint}/v1/export?agent=${agent}`)).json(), [], 'Use a fresh synthetic scope');
const tools = Object.fromEntries(registry.tools.flatMap(t => t.factory({agentId: agent,
  sessionKey: `agent:${agent}:new-fact`}) || []).map(t => [t.name, t]));
async function call(name, args) {
  const tool = tools[name];
  const projected = projectRuntimeToolInputSchema(tool.parameters, `${name}.inputSchema`);
  assert.deepEqual(projected.violations, []);
  const validated = validateJsonSchemaValue({schema: projected.schema,
    cacheKey: `noldomem-optional-regression:${name}:${JSON.stringify(projected.schema)}`, value: args});
  assert(validated.ok, JSON.stringify(validated));
  return tool.execute('synthetic-call', args);
}
const fresh = {content: 'The Aurora observatory preference is quiet visits.',
  source: null, namespace: null, supersedes: null, valid_from: null};
const old = (await call('noldomem_store', fresh)).details;
assert(old.stored && old.id);
const newer = (await call('noldomem_store', {...fresh,
  content: 'The Aurora observatory preference is guided group visits.', supersedes: old.id})).details;
assert(newer.stored && newer.id !== old.id);
const query = {query: 'Aurora observatory preference', limit: null,
  namespace: null, memory_type: null, include_history: null, as_of: null};
const current = (await call('noldomem_recall', query)).details.memories;
assert(current.some(r => r.id === newer.id));
assert(!current.some(r => r.id === old.id));
const history = (await call('noldomem_recall', {...query, include_history: true})).details.memories;
assert(history.some(r => r.id === old.id) && history.some(r => r.id === newer.id));
const invalid = await call('noldomem_store', {...fresh, supersedes: ''});
assert(invalid.details.error, 'Empty revision IDs must remain invalid');
const other = registry.tools.flatMap(t => t.factory({agentId: 'optional-schema-other'}) || []);
const isolated = await other.find(t => t.name === 'noldomem_recall').execute('other', query);
assert.deepEqual(isolated.details.memories, []);
const forbidden = await other.find(t => t.name === 'noldomem_store').execute('other-revision', {...fresh, supersedes: newer.id});
assert(forbidden.details.error, 'Another agent must not revise this family');
const receipt = (await call('noldomem_forget', {memory_id: newer.id})).details;
assert(receipt.deleted && receipt.source_keys.length);
const cleared = (await call('noldomem_relearn_source', {session_id: null, source_key: receipt.source_keys[0]})).details;
assert(cleared.cleared && !cleared.restored);
const invalidRelearn = await call('noldomem_relearn_source', {session_id: null, source_key: null});
assert(invalidRelearn.isError, 'Null must not authorize relearning without an identifier');
console.log(JSON.stringify({native_loader: true, native_schema_projection: true,
  real_tool_http_db: true, null_new_store: true, current_and_previous: true,
  invalid_revision_rejected: true, agent_isolation: true, receipt_relearning: true,
  model_requests: 0}));
