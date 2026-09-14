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
const mediaFile = process.argv.includes('--media-file');
const mediaResponse = process.argv.includes('--media-response');
assert(!(mediaFile && mediaResponse));
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
  const episode = mediaResponse ? 'The entrance appears to be the blue door. I cannot determine who the guide is.' :
    mediaFile ? 'The Aurora observatory roof opens at sunrise.' :
    'For the Aurora observatory visit, I prefer quiet visits without a group.';
  const query = mediaResponse ? 'Plan the Aurora observatory entrance and guide.' :
    mediaFile ? 'When does the Aurora observatory roof open?' :
    'Plan the Aurora observatory visit around my preference.';
  let completionText = episode;
  if (mediaFile) {
    const file = path.join(process.env.HOME, 'synthetic-plan.txt');
    fs.writeFileSync(file, episode);
    const apply = await native('', 'applyMediaUnderstanding');
    const inbound = {Body: '<media:document>', media: [{path: file, contentType: 'text/plain'}]};
    const result = await apply({ctx: inbound, workspaceDir: process.env.HOME,
      cfg: {tools: {media: {image: {enabled: false}, audio: {enabled: false}, video: {enabled: false}}}},
      selfServeLocalPaths: true});
    assert(result.appliedFile, 'Native file processing did not run');
    assert(inbound.Body.includes(episode));
    assert(inbound.Body.includes('<file '), 'Native file envelope absent');
    completionText = inbound.Body;
  }
  const ctx = (agent, session) => ({agentId: agent, sessionKey: `agent:${agent}:${session}`,
    sessionId: session, workspaceDir: process.env.HOME, trigger: 'user'});
  const exported = async agent => (await fetch(`${endpoint}/v1/export?agent=${agent}`)).json();
  assert.deepEqual(await exported('alpha'), []);
  // No plugin callback or hook runner is called directly. These are the same
  // SDK entry points used by the native Codex harness after/before model turns.
  if (!mediaFile && !mediaResponse) for (const question of [
    'What time is the Aurora observatory visit, how does my current preference compare with before, and who is guiding it?',
    'For the Aurora visit, which entrance should I use, what should I bring, where should I meet, and who is the guide?',
  ]) await sdk.awaitAgentHarnessAgentEndHook({ctx: ctx('alpha', 'question'), event: {
    success: true, messages: [{role: 'user', content: question},
    {role: 'assistant', content: 'The guide is unknown.'}], durationMs: 1,
  }});
  assert.deepEqual(await exported('alpha'), [], 'Question-only completion became a reported fact');
  const learning = mediaResponse ? {ctx: {...ctx('alpha', 'learning'), runId:'synthetic-media-run',
    channel:'webchat',messageProvider:'webchat'}, event: {
    success:true, messages:[{role:'user',content:'Describe the Aurora observatory visit notes.',
      idempotencyKey:'synthetic-media-run:user',timestamp:123000,__openclaw:{
        upstreamUserText:'Describe the Aurora observatory visit notes.',
        media:[{path:'/synthetic/entrance.png',contentType:'image/png'}],
        mediaImageLayout:{slots:[{kind:'inline',factIndex:0}]},
      }}, {role:'assistant',stopReason:'stop',content:[{type:'text',text:episode}]}],
  }} : {ctx: ctx('alpha', 'learning'), event: {
    success: true, messages: [{role: 'user', content: completionText},
      {role: 'assistant', content: 'Understood.'}], durationMs: 1,
  }};
  await sdk.awaitAgentHarnessAgentEndHook(learning);
  if (mediaResponse) await sdk.awaitAgentHarnessAgentEndHook(learning);
  const rows = await exported('alpha');
  const captured = rows.find(r => mediaResponse ? r.source === 'plugin-media-response' : r.text === episode);
  assert(captured, 'Gateway-activated capture mismatch: ' + JSON.stringify({completionText, rows}));
  if (mediaResponse) {
    assert.equal(rows.length,1,'Repeated completion duplicated the episode');
    assert.equal(captured.evidence.assertion,'inferred');
    assert.equal(captured.evidence.delivery,'generated');
    assert.equal(captured.evidence.representation,'text');
    assert.equal(captured.evidence.event_id,'synthetic-media-run:user');
    assert(captured.text.includes(episode));
  }
  if (mediaFile) {
    assert.equal(captured.evidence.assertion, 'derived');
    assert.equal(captured.evidence.modality, 'document');
    assert.equal(captured.evidence.representation, 'extracted_text');
  }
  // Seed historical evidence outside the configured capture namespace.
  const historical = !mediaFile && !mediaResponse
    ? 'For the Aurora observatory visit, I prefer quiet guided visits on weekdays.' : null;
  if (historical) {
    const seeded = await fetch(`${endpoint}/v1/store`, {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({agent:'alpha',
        text:historical, namespace:'older-session', source:'synthetic-history',
        session_id:'agent:alpha:prior'})});
    assert(seeded.ok, 'Historical fixture seed failed');
  }
  const built = await sdk.resolveAgentHarnessBeforePromptBuildResult({ctx: ctx('alpha', 'later'),
    prompt: query, developerInstructions: 'Synthetic lifecycle fixture.', messages: []});
  assert(built.prompt.includes(episode), 'Automatic cross-session injection absent');
  assert(built.prompt.includes(query));
  if (historical) assert(built.prompt.includes(historical), 'Other own-agent namespace was excluded');
  const other = await sdk.resolveAgentHarnessBeforePromptBuildResult({ctx: ctx('beta', 'later'),
    prompt: query, developerInstructions: 'Synthetic lifecycle fixture.', messages: []});
  assert(!other.prompt.includes(episode), 'Other agent received the event');
  if (historical) assert(!other.prompt.includes(historical), 'Other agent received historical namespace');
  assert.deepEqual(await exported('beta'), []);
  if (mediaResponse) {
    const forgotten = await fetch(`${endpoint}/v1/forget`, {method:'DELETE',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({agent:'alpha',id:captured.id})});
    assert(forgotten.ok);
    await sdk.awaitAgentHarnessAgentEndHook(learning);
    assert.deepEqual(await exported('alpha'),[],'Forgotten source recaptured');
    const afterForget = await sdk.resolveAgentHarnessBeforePromptBuildResult({ctx:ctx('alpha','after-forget'),
      prompt:query, developerInstructions:'Synthetic lifecycle fixture.',messages:[]});
    assert(!afterForget.prompt.includes(episode),'Deleted episode remains in injection');
    const relearn = await fetch(`${endpoint}/v1/relearn-source`, {method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({agent:'alpha',
        session_id:learning.ctx.sessionKey,confirm:true})});
    assert(relearn.ok);
    await sdk.awaitAgentHarnessAgentEndHook(learning);
    assert.equal((await exported('alpha')).length,1,'Explicit relearning did not allow capture');
  }
  console.log('LIFECYCLE_RESULT=' + JSON.stringify({
    host: JSON.parse(fs.readFileSync(path.join(host, 'package.json'), 'utf8')).version,
    node: process.version, gateway_startup: true, update_rehearsal: true,
    sdk_completion_dispatch: true, captured_id: captured.id,
    capture_source_session: captured.source_session, injected_prompt: built.prompt,
    other_agent_prompt: other.prompt, capture: true, cross_session_injection: true,
    agent_isolation: true,
    ...(mediaResponse ? {synthetic_media_response:true, actual_image_decoding:false,
      inferred_not_reported:true, generated_not_delivered:true, repeated_event_deduplicated:true,
      forgotten_source_blocked:true, forgotten_context_absent:true, explicit_api_relearning:true} :
      mediaFile ? {native_local_file_processing: true, derived_document_preserved: true,
      utility_preprocessing_path: true, raw_image_or_audio_decoded: false} :
      {question_only_not_captured: true}),
    manual_store_or_recall: false, model_calls: 0,
    full_model_loop: false,
  }));
} finally {
  await gateway.close({reason: 'Synthetic lifecycle test finished'});
}
