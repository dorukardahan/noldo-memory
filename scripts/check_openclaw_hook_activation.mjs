/** Compare installed standalone tool loading with registry activation.
 * Requires a fresh temporary HOME/state. No auth, model, hook callback or API call.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const [host, candidate] = process.argv.slice(2);
assert(process.env.OPENCLAW_STATE_DIR?.startsWith(process.env.HOME + path.sep));

async function native(prefix, name) {
  const dist = path.join(host, 'dist');
  for (const file of fs.readdirSync(dist).filter(f => f.startsWith(prefix) && f.endsWith('.mjs'))) {
    const source = fs.readFileSync(path.join(dist, file), 'utf8');
    const exports = source.match(/export \{([^}]+)\};?\s*$/s)?.[1];
    const alias = exports?.match(new RegExp(`\\b${name} as ([\\w$]+)(?:,|\\s|$)`))?.[1];
    if (alias) return (await import(pathToFileURL(path.join(dist, file))))[alias];
  }
  throw Error(`Missing installed native export ${name}`);
}

const resolveTools = await native('tools-', 'resolvePluginTools');
const getHooks = await native('hook-runner-global-', 'getGlobalHookRunner');
const load = await native('loader-runtime-load-', 'loadOpenClawPlugins');
const config = {plugins: {
  enabled: true, allow: ['noldomem'], slots: {memory: 'none'},
  load: {paths: [path.join(candidate, 'plugin')]},
  entries: {noldomem: {
    enabled: true,
    hooks: {allowConversationAccess: true, allowPromptInjection: true},
    config: {
      baseUrl: 'http://127.0.0.1:1', apiKeyFile: '/dev/null',
      enableAutoRecall: true, enableAutoCapture: true,
      enableOperationalCapture: false, enableCompactionCapture: false, enableSubagentCapture: false,
    },
  }},
}};
const names = ['noldomem_recall', 'noldomem_store', 'noldomem_pin',
  'noldomem_forget', 'noldomem_relearn_source'];
assert.equal(getHooks(), null);
const tools = resolveTools({
  context: {config, workspaceDir: process.env.HOME, agentId: 'hook-probe',
    sessionKey: 'agent:hook-probe:fixture'},
  toolAllowlist: names,
});
assert.deepEqual(tools.map(t => t.name).sort(), [...names].sort());
const afterStandalone = {
  runner_present: getHooks() !== null,
  recall: !!getHooks()?.hasHooks('before_prompt_build'),
  capture: !!getHooks()?.hasHooks('agent_end'),
};
const registry = load({
  config, workspaceDir: process.env.HOME, activate: true, cache: false,
  onlyPluginIds: ['noldomem'], allowProcessHomeSessionCatalogs: false, throwOnLoadError: true,
  logger: {info() {}, warn() {}, error() {}, debug() {}},
});
assert.equal(registry.plugins.find(p => p.id === 'noldomem').source,
  path.join(candidate, 'plugin/index.js'));
const afterActivation = {
  runner_present: getHooks() !== null,
  recall: !!getHooks()?.hasHooks('before_prompt_build'),
  capture: !!getHooks()?.hasHooks('agent_end'),
};
assert.deepEqual(afterStandalone, {runner_present: false, recall: false, capture: false});
assert.deepEqual(afterActivation, {runner_present: true, recall: true, capture: true});
console.log(JSON.stringify({
  host: JSON.parse(fs.readFileSync(path.join(host, 'package.json'), 'utf8')).version,
  node: process.version, tools: tools.map(t => t.name),
  standalone: afterStandalone, activated: afterActivation,
  model_calls: 0, hook_callbacks_invoked: 0, api_calls: 0,
}));
