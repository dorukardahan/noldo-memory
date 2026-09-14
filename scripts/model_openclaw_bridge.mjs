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
      enableAutoCapture: true,
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
// A synthetic command is supplied on stdin; no model or credential code here.
const command = JSON.parse(fs.readFileSync(0, 'utf8'));
assert(['alpha', 'beta'].includes(command.agent));
const ctx = {agentId: command.agent, sessionKey: `agent:${command.agent}:${command.session}`};
const tools = registry.tools.flatMap(t => t.factory(ctx) || []);
let result;
if (command.action === 'capture') {
  await hooks.runAgentEnd({success: true, messages: command.messages}, ctx);
  result = {captured: true};
} else if (command.action === 'context') {
  const context = await hooks.runBeforePromptBuild({prompt: command.query, messages: []}, ctx);
  result = {context: context?.prependContext || '', tools: tools.map(({name, description, parameters}) =>
    ({name, description, parameters}))};
} else if (command.action === 'tool') {
  const tool = tools.find(t => t.name === command.name);
  assert(tool, 'Missing registered native tool');
  result = await tool.execute('synthetic-model-choice', command.arguments);
} else { throw Error('Unknown bridge action'); }
const rows = await (await fetch(endpoint + '/v1/export?agent=' + command.agent)).json();
console.log('ACCEPTANCE_JSON=' + JSON.stringify({result, rows, host_version: pkg.version,
  node: process.version, candidate_native_loader: true, native_hook_runner: true}));
