/** Model-free Gateway startup + public harness lifecycle test.
 * Synthetic completion events are supplied through the public SDK, not a model.
 * Never invoke this against an existing profile or a production API.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const [host, candidate, endpoint, portText] = process.argv.slice(2);
assert(host && candidate && endpoint && portText);
assert.equal(new URL(endpoint).hostname, '127.0.0.1');
assert(process.env.OPENCLAW_STATE_DIR?.startsWith(process.env.HOME + path.sep));
assert(process.env.OPENCLAW_CONFIG_PATH?.startsWith(process.env.HOME + path.sep));
const requireHost = createRequire(path.join(host, 'package.json'));
const sdk = await import(pathToFileURL(requireHost.resolve('openclaw/plugin-sdk/agent-harness-runtime')));
async function native(prefix, name) {
  const dist = path.join(host, 'dist');
  for (const file of fs.readdirSync(dist).filter(f => f.startsWith(prefix) && f.endsWith('.mjs'))) {
    const source = fs.readFileSync(path.join(dist, file), 'utf8');
    const exports = source.match(/export \{([^}]+)\};?\s*$/s)?.[1];
    const alias = exports?.match(new RegExp(`\\b${name} as ([\\w$]+)(?:,|\\s|$)`))?.[1];
    const symbol = alias || (exports?.split(',').map(s => s.trim()).includes(name) ? name : null);
    if (symbol) return (await import(pathToFileURL(path.join(dist, file))))[symbol];
  }
  throw Error(`Missing native export ${name}`);
}
const start = await native('server-', 'startGatewayServer');
assert.equal(sdk.getAgentHarnessHookRunner(), null);
// The host's supported update rehearsal loads normal startup plugins but does
// not start autonomous scheduled jobs, channel work, or background model runs.
const gateway = await start(Number(portText), {bind: 'loopback', auth: {mode: 'none'},
  controlUiEnabled: false, openAiChatCompletionsEnabled: false,
  openResponsesEnabled: false, updateCanary: true});
try {
  assert(sdk.getAgentHarnessHookRunner()?.hasHooks('agent_end'));
  assert(sdk.getAgentHarnessHookRunner()?.hasHooks('before_prompt_build'));
  const episode = 'For the Aurora observatory visit, I prefer quiet visits without a group.';
  const query = 'Plan the Aurora observatory visit around my preference.';
  const ctx = (agent, session) => ({agentId: agent, sessionKey: `agent:${agent}:${session}`,
    sessionId: session, workspaceDir: process.env.HOME, trigger: 'user'});
  const exported = async agent => (await fetch(`${endpoint}/v1/export?agent=${agent}`)).json();
  assert.deepEqual(await exported('alpha'), []);
  // No plugin callback or hook runner is called directly. These are the same
  // SDK entry points used by the native Codex harness after/before model turns.
  await sdk.awaitAgentHarnessAgentEndHook({ctx: ctx('alpha', 'learning'), event: {
    success: true, messages: [{role: 'user', content: episode},
      {role: 'assistant', content: 'Understood.'}], durationMs: 1,
  }});
  const rows = await exported('alpha');
  const captured = rows.find(r => r.text === episode);
  assert(captured, 'Gateway-activated agent_end did not capture the synthetic event');
  const built = await sdk.resolveAgentHarnessBeforePromptBuildResult({ctx: ctx('alpha', 'later'),
    prompt: query, developerInstructions: 'Synthetic lifecycle fixture.', messages: []});
  assert(built.prompt.includes(episode), 'Automatic cross-session injection absent');
  assert(built.prompt.includes(query));
  const other = await sdk.resolveAgentHarnessBeforePromptBuildResult({ctx: ctx('beta', 'later'),
    prompt: query, developerInstructions: 'Synthetic lifecycle fixture.', messages: []});
  assert(!other.prompt.includes(episode), 'Other agent received the event');
  assert.deepEqual(await exported('beta'), []);
  console.log('LIFECYCLE_RESULT=' + JSON.stringify({
    host: JSON.parse(fs.readFileSync(path.join(host, 'package.json'), 'utf8')).version,
    node: process.version, gateway_startup: true, update_rehearsal: true,
    sdk_completion_dispatch: true, captured_id: captured.id,
    capture_source_session: captured.source_session, injected_prompt: built.prompt,
    other_agent_prompt: other.prompt, capture: true, cross_session_injection: true,
    agent_isolation: true, manual_store_or_recall: false, model_calls: 0,
    full_model_loop: false,
  }));
} finally {
  await gateway.close({reason: 'Synthetic lifecycle test finished'});
}
