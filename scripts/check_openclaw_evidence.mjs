/** Installed native loader/internal-hook dispatch plus temporary NoldoMem HTTP.
 * Synthetic text derivatives only. No model, decoder, Gateway or transport send.
 */
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
const [host,candidate,endpoint] = process.argv.slice(2);
assert(host && candidate && endpoint);
assert(process.env.OPENCLAW_STATE_DIR?.startsWith(process.env.HOME + path.sep));
assert.equal(JSON.parse(fs.readFileSync(path.join(host,'package.json'))).version,'2026.9.3');
const realFetch = globalThis.fetch;
let externalAttempts = 0;
globalThis.fetch = (input,options) => {
  const url = new URL(typeof input === 'string' || input instanceof URL ? input : input.url);
  if (url.origin !== endpoint) { externalAttempts++; throw Error('External calls forbidden'); }
  return realFetch(input,options);
};
async function nativeExport(name) {
  const dist = path.join(host,'dist');
  for (const file of fs.readdirSync(dist).filter(f => f.endsWith('.mjs'))) {
    const source = fs.readFileSync(path.join(dist,file),'utf8');
    const exports = source.match(/export \{([^}]+)\};?\s*$/s)?.[1];
    const alias = exports?.match(new RegExp(`\\b${name} as ([\\w$]+)(?:,|\\s|$)`))?.[1];
    if (alias) return (await import(pathToFileURL(path.join(dist,file))))[alias];
  }
  throw Error(`Installed export unavailable: ${name}`);
}
if (process.argv.includes('--delivery-projection-only')) {
  const makeEmitter = await nativeExport('createMessageSentEmitter');
  const mirror = await nativeExport('resolveMirroredTranscriptText');
  const seen = [];
  const emitter = makeEmitter({channel:'synthetic',to:'synthetic-chat',logPrefix:'synthetic',
    hookRunner:{hasHooks:()=>true, async runMessageSent(event,ctx){seen.push({event,ctx});}}});
  emitter.emitMessageSent({success:true,content:'The dome is violet.',messageId:'synthetic-delivery',
    mediaUrls:['https://example.test/one/dome.png']});
  emitter.emitMessageSent({success:true,content:'The dome is violet.',messageId:'synthetic-delivery',
    mediaUrls:['https://example.test/two/dome.png']});
  assert.equal(seen.length,2);
  assert.deepEqual(seen[0],seen[1]); // Runtime projection drops the extra media, not just its type.
  assert.equal(mirror({text:'The dome is violet.',mediaUrls:['https://example.test/one/dome.png']}),
    mirror({text:'The dome is violet.',mediaUrls:['https://example.test/two/dome.png']}));
  assert.equal(externalAttempts,0);
  console.log(JSON.stringify({host:'2026.9.3',node:process.version,native_sent_emitter_media_projection:'lossy',
    native_transcript_media_mirror:'basename-only; distinct resources collide',
    transport_send:false,model_calls:0,external_calls:0}));
  process.exit(0);
}
const load = await nativeExport('loadOpenClawPlugins');
const createHookRunner = await nativeExport('createHookRunner');
const setRegistry = await nativeExport('setActivePluginRegistry');
const trigger = await nativeExport('triggerInternalHook');
const createEvent = await nativeExport('createInternalHookEvent');
const config = {plugins:{enabled:true,allow:['noldomem'],slots:{memory:'none'},
  load:{paths:[path.join(candidate,'plugin')]},entries:{noldomem:{enabled:true,
    hooks:{allowConversationAccess:true,allowPromptInjection:true},config:{
      baseUrl:endpoint,apiKeyFile:'/dev/null',enableAutoCapture:true,enableAutoRecall:true,
      autoCaptureSource:'preprocessed',enableOperationalCapture:false,
      enableCompactionCapture:false,enableSubagentCapture:false,
    }}}}};
const registry = load({config,workspaceDir:process.env.HOME,
  env:{HOME:process.env.HOME,OPENCLAW_STATE_DIR:process.env.OPENCLAW_STATE_DIR},
  activate:false,cache:false,allowProcessHomeSessionCatalogs:false,onlyPluginIds:['noldomem'],throwOnLoadError:true,
  logger:{info(){},warn(){},error(){},debug(){}},
});
const plugin = registry.plugins.find(p=>p.id==='noldomem');
assert(plugin?.source.startsWith(path.join(candidate,'plugin')+path.sep));
assert(!registry.typedHooks.some(h=>h.hookName==='agent_end'));
assert(registry.legacyInternalHooks.some(h=>h.event==='message:preprocessed'));
setRegistry(registry,'synthetic-evidence-check'); // This test process only, never the running Gateway.
const hooks = createHookRunner(registry,{catchErrors:false});
const sessionKey='agent:alpha:synthetic-voice';
const transcript='The Aurora observatory booking starts at 19:30 on Friday.';
const context={transcript,bodyForAgent:transcript,messageId:'synthetic-voice-1',timestamp:1700000000123,
  channelId:'synthetic',media:[{kind:'audio',contentType:'audio/ogg',path:'synthetic-voice.ogg'}]};
const emit=()=>trigger(createEvent('message','preprocessed',sessionKey,{...context}));
const rows=async(agent='alpha')=>(await fetch(`${endpoint}/v1/export?agent=${agent}`)).json();
await Promise.all([emit(),emit(),emit()]);
let stored=await rows();
assert.equal(stored.length,1,'duplicate native events created duplicate memories');
assert.equal(stored[0].text,transcript);
assert.equal(stored[0].evidence.modality,'audio');
assert.equal(stored[0].evidence.assertion,'derived');
assert.equal(stored[0].evidence.reference,'synthetic-voice.ogg');
assert.equal(stored[0].evidence.event_id,'synthetic-voice-1');
assert.equal(stored[0].evidence.observed_at,1700000000.123);
const readCtx={agentId:'alpha',sessionKey:'agent:alpha:synthetic-next'};
const recall=()=>hooks.runBeforePromptBuild({prompt:'When does the Aurora observatory booking start?',messages:[]},readCtx);
assert((await recall())?.prependContext?.includes(transcript));
assert.equal((await rows('beta')).length,0);
assert(!(await hooks.runBeforePromptBuild({prompt:'When does the Aurora observatory booking start?',messages:[]},
  {agentId:'beta',sessionKey:'agent:beta:synthetic-next'}))?.prependContext?.includes(transcript));
const tools=registry.tools.flatMap(t=>t.factory(readCtx)||[]);
const forget=tools.find(t=>t.name==='noldomem_forget');
const receipt=(await forget.execute('synthetic-forget',{memory_id:stored[0].id})).details;
assert(receipt.deleted);
await Promise.all([emit(),emit()]);
assert.equal((await rows()).length,0);
assert(!(await recall())?.prependContext?.includes(transcript));
const relearn=tools.find(t=>t.name==='noldomem_relearn_source');
assert((await relearn.execute('synthetic-relearn',{source_key:receipt.source_keys[0]})).details.cleared);
await emit();
assert.equal((await rows()).length,1);
await trigger(createEvent('message','preprocessed','agent:alpha:synthetic-independent',{...context,messageId:'independent-1'}));
assert.equal((await rows()).length,2);
const extractFile = await nativeExport('extractFileContentFromBuffer');
const fileLimits = await nativeExport('resolveInputFileLimits');
const documentText = 'The Aurora observatory roof opens at sunrise.';
const extracted = await extractFile({buffer:Buffer.from(documentText,'utf8'),filename:'plan.txt',mimeType:'text/plain',
  limits:fileLimits({allowedMimes:['text/plain'],maxBytes:4096,maxChars:2000}),config:{}});
assert.equal(extracted.text,documentText);
for (let i=0;i<2;i++) {
  // The installed file-outcome formatter is private. Supply its pinned envelope
  // shape independently; only the byte decoder and hook dispatch are native here.
  const marker=i ? 'fedcba9876543210' : '0123456789abcdef';
  const bodyForAgent=`<file name="plan.txt" mime="text/plain">\n<<<EXTERNAL_UNTRUSTED_CONTENT id="${marker}">>>\nSource: External\n---\n${extracted.text}\n<<<END_EXTERNAL_UNTRUSTED_CONTENT id="${marker}">>>\n</file>`;
  await trigger(createEvent('message','preprocessed','agent:alpha:synthetic-document',{
    bodyForAgent,messageId:'synthetic-file-1',media:[{kind:'file',path:'synthetic-plan.txt'}],channelId:'synthetic',
  }));
}
const documents=(await rows()).filter(row=>row.text===documentText);
assert.equal(documents.length,1,'random host wrapper IDs must not defeat exact event deduplication');
assert.equal(documents[0].evidence.modality,'document');
assert.equal(documents[0].evidence.assertion,'derived');
assert((await hooks.runBeforePromptBuild({prompt:'When does the Aurora observatory roof open?',messages:[]},readCtx))
  ?.prependContext?.includes(documentText));
assert.equal(externalAttempts,0);
console.log(JSON.stringify({host:'2026.9.3',node:process.version,candidate_loader:true,native_internal_hook_dispatch:true,
  real_temporary_api:true,duplicate_events:3,stored_after_duplicates:1,derived_audio_origin:true,
  cross_session_injection:true,agent_isolation:true,forget_replay_and_cache:true,relearning:true,
  independent_source:true,native_utf8_document_decode:true,document_envelope:'synthetic pinned-host format',document_context_and_deduplication:true,
  external_calls:0,model_calls:0,
  limitation:'Synthetic audio hook payload; raw image/audio/PDF decoding, channel delivery and complete model loop not executed'}));
setRegistry(null);
